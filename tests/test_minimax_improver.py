"""Tests for MiniMaxLLMWrapper (default improver, replaces Anthropic path)."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import httpx
import pytest

from autobench.core import (
    ContextManager,
    HarnessConfig,
    HarnessResult,
    RolloutProtocol,
    Verdict,
)
from autobench.evaluator import BenchmarkResult
from autobench.llm.anthropic import LLMImprovementResult
from autobench.llm.minimax import MiniMaxLLMWrapper
from autobench.rsi.loop import ImprovementDelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_harness() -> HarnessConfig:
    return HarnessConfig(
        system_prompt="solve coding problems",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="bash, python",
        verifiers=[],
        budget={"max_tokens": 8192, "max_time_seconds": 30, "max_cost_dollars": 1.0},
    )


def _make_bench_result() -> BenchmarkResult:
    # Mix of verdicts so dominant non-OK is CE (50% of 4 cases).
    cases = [
        HarnessResult(verdict=Verdict.CE, p_score=0.0, latency_ms=10),
        HarnessResult(verdict=Verdict.CE, p_score=0.0, latency_ms=11),
        HarnessResult(verdict=Verdict.OK, p_score=1.0, latency_ms=12),
        HarnessResult(verdict=Verdict.WA, p_score=0.0, latency_ms=13),
    ]
    return BenchmarkResult(
        case_results=cases,
        aggregate_score=0.25,
        total_latency_ms=46.0,
        verdict_counts={"CE": 2, "OK": 1, "WA": 1},
    )


def _mock_chat_response(content: str, *, prompt_tokens: int = 120,
                       completion_tokens: int = 80) -> dict:
    return {
        "id": "test-id",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# (a) wrapper instantiates with env key
# ---------------------------------------------------------------------------

def test_wrapper_instantiates_with_env_key(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-abc")
    w = MiniMaxLLMWrapper()
    assert w.api_key == "test-key-abc"
    assert w.model == "MiniMax-M2.7"
    assert 0.0 < w.temperature <= 1.0
    assert w.base_url == "https://api.minimax.io"
    # Default endpoint mode mirrors worker_agent — Anthropic-compatible
    # messages endpoint, not OpenAI chat-completions.
    assert w.endpoint_mode == "anthropic"


def test_wrapper_rejects_invalid_endpoint_mode(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    with pytest.raises(ValueError, match="endpoint_mode"):
        MiniMaxLLMWrapper(endpoint_mode="bogus")


def test_wrapper_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        MiniMaxLLMWrapper()


def test_wrapper_rejects_zero_temperature(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    with pytest.raises(ValueError, match=r"temperature"):
        MiniMaxLLMWrapper(temperature=0.0)


# ---------------------------------------------------------------------------
# (b) constructs correct payload
# ---------------------------------------------------------------------------

def _install_mock_client_improver(wrapper, response_json):
    """Replace wrapper._http_client with a mock that records calls."""
    captured: dict = {}

    class _StubClient:
        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=response_json)
            return resp

        def close(self):
            captured["closed"] = True

    wrapper._http_client = _StubClient()
    return captured


def test_constructs_correct_payload(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "secret-key")
    wrapper = MiniMaxLLMWrapper(model="MiniMax-M2.7", temperature=0.4,
                                max_tokens=2048, endpoint_mode="openai")

    valid_json = json_dumps_improvement()
    captured = _install_mock_client_improver(
        wrapper, _mock_chat_response(valid_json)
    )

    result = wrapper.suggest_harness_improvements(
        _make_harness(), _make_bench_result()
    )

    assert isinstance(result, LLMImprovementResult)
    assert captured["url"] == "https://api.minimax.io/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["headers"]["Content-Type"] == "application/json"

    payload = captured["json"]
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["temperature"] == 0.4
    assert payload["max_tokens"] == 2048
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    # The user prompt should mention dominant verdict CE since 50% are CE.
    assert "CE" in payload["messages"][1]["content"]


def json_dumps_improvement() -> str:
    return json.dumps({
        "system_prompt_changes": "Prefer short solutions.",
        "rollout_protocol": "iterative",
        "context_manager": "hierarchical",
        "tool_surface_changes": "Add compile-test step",
        "budget_changes": {"max_tokens": 6000},
        "rationale": "Cut CE rate by simplifying generation",
    })


# ---------------------------------------------------------------------------
# (c) parses valid JSON response into ImprovementDelta + HarnessConfig
# ---------------------------------------------------------------------------

def test_parses_valid_json_response(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")

    content = json_dumps_improvement()
    response = _mock_chat_response(content, prompt_tokens=200,
                                   completion_tokens=150)

    with patch.object(
        MiniMaxLLMWrapper, "_call_with_retries", return_value=response
    ):
        result = wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result()
        )

    assert isinstance(result, LLMImprovementResult)
    assert isinstance(result.delta, ImprovementDelta)
    assert result.suggested_harness.rollout_protocol == RolloutProtocol.ITERATIVE
    assert result.suggested_harness.context_manager == ContextManager.HIERARCHICAL
    assert "Prefer short solutions." in result.suggested_harness.system_prompt
    # nervous-bus-ldd1: tool_surface_changes is rejected; the harness has no
    # machinery to materialize improver-proposed tools. tool_surface stays
    # at its base value ("bash, python" from _make_harness()).
    assert result.suggested_harness.tool_surface == "bash, python"
    assert "Add compile-test step" not in result.suggested_harness.tool_surface
    assert result.suggested_harness.budget["max_tokens"] == 6000
    assert result.delta.rollout_protocol_changed is True
    assert result.delta.context_manager_changed is True
    assert result.delta.improvement_summary.startswith("Cut CE rate")
    assert result.tokens_used == 150
    # nervous-bus-dq7l: cost_dollars is always 0.0 — pricing tables removed.
    assert result.cost_dollars == 0.0
    assert result.model_used == "MiniMax-M2.7"


def test_parses_json_inside_markdown_fence(monkeypatch):
    """The system prompt forbids fences, but be resilient if the model adds them."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")

    fenced = "```json\n" + json_dumps_improvement() + "\n```"
    response = _mock_chat_response(fenced)

    with patch.object(
        MiniMaxLLMWrapper, "_call_with_retries", return_value=response
    ):
        result = wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result()
        )

    assert result.suggested_harness.rollout_protocol == RolloutProtocol.ITERATIVE


# ---------------------------------------------------------------------------
# (d) falls back to rule-based improver on HTTP error
# ---------------------------------------------------------------------------

def test_falls_back_on_http_error(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(max_retries=1)

    def _raise(*a, **kw):
        req = httpx.Request("POST", "https://api.minimax.io/v1/chat/completions")
        resp = httpx.Response(500, request=req)
        raise httpx.HTTPStatusError("server boom", request=req, response=resp)

    with patch.object(MiniMaxLLMWrapper, "_call_minimax", side_effect=_raise):
        result = wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result()
        )

    assert isinstance(result, LLMImprovementResult)
    assert "fallback: rule_based" in result.model_used
    assert result.tokens_used == 0
    assert result.cost_dollars == 0.0
    # Rule-based should still produce a HarnessConfig.
    assert isinstance(result.suggested_harness, HarnessConfig)


def test_falls_back_on_garbled_response(monkeypatch):
    """If MiniMax returns content with no JSON object, parsing returns the
    current config unchanged but the wrapper does NOT raise."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")

    response = _mock_chat_response("sorry, I cannot help with that")

    with patch.object(
        MiniMaxLLMWrapper, "_call_with_retries", return_value=response
    ):
        result = wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result()
        )

    # No changes — parser couldn't find JSON, returned current config.
    assert result.suggested_harness.rollout_protocol == RolloutProtocol.SINGLE
    assert result.suggested_harness.context_manager == ContextManager.FULL


# ---------------------------------------------------------------------------
# Sanity: improve() is a drop-in for SelfImprovingHarness.improver_fn
# ---------------------------------------------------------------------------

def test_improve_returns_tuple(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")
    response = _mock_chat_response(json_dumps_improvement())

    with patch.object(
        MiniMaxLLMWrapper, "_call_with_retries", return_value=response
    ):
        new_harness, delta = wrapper.improve(_make_harness(), _make_bench_result())

    assert isinstance(new_harness, HarnessConfig)
    assert isinstance(delta, ImprovementDelta)


# ---------------------------------------------------------------------------
# (e) Anthropic endpoint — payload shape, parse, thinking-block drop
# ---------------------------------------------------------------------------

def _mock_anthropic_response(text: str, *, thinking: str = "",
                              input_tokens: int = 120,
                              output_tokens: int = 80) -> dict:
    """Anthropic-shaped response with a thinking block and a text block.

    MiniMax-M2.7 on /anthropic/v1/messages returns a top-level ``content``
    list. ``_parse_response`` MUST keep only ``"text"`` blocks; thinking
    must never reach the JSON regex extractor.
    """
    blocks = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    blocks.append({"type": "text", "text": text})
    return {
        "id": "msg-test",
        "content": blocks,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def test_anthropic_mode_posts_to_messages_endpoint(monkeypatch):
    """Anthropic mode hits /anthropic/v1/messages with system as top-level field."""
    monkeypatch.setenv("MINIMAX_API_KEY", "secret-anth")
    # thinking_budget=None to keep the shape assertion focused on system/messages.
    wrapper = MiniMaxLLMWrapper(temperature=0.4, max_tokens=2048,
                                 endpoint_mode="anthropic",
                                 thinking_budget=None)

    captured = _install_mock_client_improver(
        wrapper, _mock_anthropic_response(json_dumps_improvement())
    )

    result = wrapper.suggest_harness_improvements(
        _make_harness(), _make_bench_result()
    )

    assert isinstance(result, LLMImprovementResult)
    # Routed to the Anthropic-compatible messages endpoint, NOT chat/completions.
    assert captured["url"] == "https://api.minimax.io/anthropic/v1/messages"
    assert captured["headers"]["Authorization"] == "Bearer secret-anth"

    payload = captured["json"]
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["temperature"] == 0.4
    assert payload["max_tokens"] == 2048
    # System lives at the TOP LEVEL, not as a system-role message.
    assert isinstance(payload["system"], str)
    assert "improving a coding agent harness" in payload["system"]
    # Messages array contains a single user turn; no system role anywhere.
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert all(m["role"] != "system" for m in payload["messages"])

    # Delta was successfully parsed from the text-block JSON.
    assert result.suggested_harness.rollout_protocol == RolloutProtocol.ITERATIVE
    # thinking_budget=None → no thinking field in payload
    assert "thinking" not in payload


def test_anthropic_mode_drops_thinking_before_json_extractor(monkeypatch):
    """Thinking-block content must NOT reach the JSON regex extractor.

    The latent bug we're protecting against: reasoning prose containing a
    stray ``{...}`` fragment could be picked up by ``_parse_llm_response``'s
    permissive regex and parsed as the delta. ``_parse_response`` strips
    thinking at parse time so the extractor only sees clean text-block JSON.
    """
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="anthropic")

    # A thinking block carrying a tempting-looking JSON-shaped fragment.
    poisonous_thinking = (
        "Let me think... maybe {\"rollout_protocol\": \"monte_carlo\", "
        "\"rationale\": \"thinking-only suggestion that must NOT win\"}"
        " — but actually a different approach is better."
    )
    real_decision = json_dumps_improvement()  # rollout_protocol=iterative

    response = _mock_anthropic_response(real_decision, thinking=poisonous_thinking)

    with patch.object(
        MiniMaxLLMWrapper, "_call_with_retries", return_value=response
    ):
        result = wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result()
        )

    # The text-block JSON wins; the thinking JSON is invisible to the parser.
    assert result.suggested_harness.rollout_protocol == RolloutProtocol.ITERATIVE
    assert result.suggested_harness.rollout_protocol != RolloutProtocol.MONTE_CARLO
    # raw_response (what observability captures) reflects ONLY the text block.
    assert "thinking-only suggestion" not in result.raw_response
    assert "monte_carlo" not in result.raw_response
    # Usage came from the Anthropic-shaped fields, not OpenAI fields.
    assert result.tokens_used == 80  # output_tokens
    # nervous-bus-dq7l: cost_dollars is always 0.0 (pricing table removed).
    assert result.cost_dollars == 0.0


def test_anthropic_mode_falls_back_on_malformed_payload(monkeypatch):
    """A response missing the ``content`` array falls back to rule-based."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="anthropic")

    # OpenAI-shaped response sent while we're in anthropic mode — KeyError
    # in _parse_response on "content".
    bad = _mock_chat_response(json_dumps_improvement())

    with patch.object(
        MiniMaxLLMWrapper, "_call_with_retries", return_value=bad
    ):
        result = wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result()
        )

    assert isinstance(result, LLMImprovementResult)
    assert "fallback: rule_based" in result.model_used
    assert result.tokens_used == 0
    assert result.cost_dollars == 0.0


def test_module_level_parse_response_anthropic_concatenates_text_blocks():
    """If a response has multiple text blocks (rare but spec'd), concatenate them."""
    from autobench.llm.minimax import _parse_response
    resp = {
        "content": [
            {"type": "thinking", "thinking": "ignored"},
            {"type": "text", "text": "first half "},
            {"type": "text", "text": "second half"},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 10},
    }
    text, inp, out = _parse_response(resp, "anthropic")
    assert text == "first half second half"
    assert inp == 5
    assert out == 10


def test_module_level_parse_response_openai_legacy_path():
    """OpenAI mode pulls from choices[0].message.content with prompt_/completion_."""
    from autobench.llm.minimax import _parse_response
    resp = _mock_chat_response("HELLO", prompt_tokens=11, completion_tokens=22)
    text, inp, out = _parse_response(resp, "openai")
    assert text == "HELLO"
    assert inp == 11
    assert out == 22


# ---------------------------------------------------------------------------
# (f) thinking_budget — extended-thinking budget cap
# ---------------------------------------------------------------------------

def test_improver_thinking_budget_attached_to_anthropic_payload(monkeypatch):
    """anthropic mode + thinking_budget=512 attaches the correct field."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="anthropic",
                                 thinking_budget=512, max_tokens=4096)

    captured = _install_mock_client_improver(
        wrapper, _mock_anthropic_response(json_dumps_improvement())
    )

    wrapper.suggest_harness_improvements(_make_harness(), _make_bench_result())

    payload = captured["json"]
    assert "thinking" in payload
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 512}


def test_improver_thinking_budget_none_omits_field(monkeypatch):
    """thinking_budget=None must produce a payload WITHOUT the thinking key."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="anthropic", thinking_budget=None)

    captured = _install_mock_client_improver(
        wrapper, _mock_anthropic_response(json_dumps_improvement())
    )

    wrapper.suggest_harness_improvements(_make_harness(), _make_bench_result())

    assert "thinking" not in captured["json"]


def test_improver_thinking_budget_omitted_in_openai_mode(monkeypatch):
    """openai mode silently drops thinking_budget."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai", thinking_budget=512)

    captured = _install_mock_client_improver(
        wrapper, _mock_chat_response(json_dumps_improvement())
    )

    wrapper.suggest_harness_improvements(_make_harness(), _make_bench_result())

    assert "thinking" not in captured["json"]


def test_improver_thinking_budget_validation(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    with pytest.raises(ValueError, match="thinking_budget"):
        MiniMaxLLMWrapper(thinking_budget=-1)
    with pytest.raises(ValueError, match="thinking_budget"):
        MiniMaxLLMWrapper(thinking_budget=0)
    with pytest.raises(ValueError, match="thinking_budget"):
        # exceeds max_tokens default 4096
        MiniMaxLLMWrapper(thinking_budget=8192, max_tokens=4096)


def test_improver_thinking_budget_default_is_2048(monkeypatch):
    """Default budget on the improver should be 2048 — more reasoning than worker."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper()
    assert wrapper.thinking_budget == 2048


# ---------------------------------------------------------------------------
# (g) Persistent httpx.Client — connection reuse
# ---------------------------------------------------------------------------

def test_improver_persistent_http_client_created_on_init(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="anthropic")
    assert wrapper._http_client is not None
    assert isinstance(wrapper._http_client, httpx.Client)
    wrapper.close()
    assert wrapper._http_client is None


def test_improver_close_is_idempotent(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    wrapper = MiniMaxLLMWrapper(endpoint_mode="anthropic")
    wrapper.close()
    wrapper.close()  # must not raise
    assert wrapper._http_client is None
