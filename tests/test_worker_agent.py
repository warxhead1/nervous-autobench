"""Tests for MiniMaxWorker — the autobench code-generation worker agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from autobench.core import ContextManager, HarnessConfig, RolloutProtocol
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_WORKER,
)
from autobench.llm.worker import (
    MiniMaxWorker,
    WorkerResult,
    _calc_backoff_ms,
    _estimate_cost,
    _extract_code,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_harness(**overrides) -> HarnessConfig:
    return HarnessConfig(
        system_prompt=overrides.get("system_prompt", "solve coding problems"),
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface=overrides.get("tool_surface", "read_stdin, write_stdout"),
        verifiers=[],
        budget={"max_tokens": 2048, "max_time_seconds": 10, "max_cost_dollars": 0.50},
    )


def _mock_response(content: str, *, prompt_tokens: int = 100,
                   completion_tokens: int = 50) -> dict:
    return {
        "id": "resp-test",
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
# (1) Construction
# ---------------------------------------------------------------------------

def test_worker_requires_api_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        MiniMaxWorker(endpoint_mode="openai")


def test_worker_reads_env_key(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key-xyz")
    w = MiniMaxWorker(endpoint_mode="openai")
    assert w.api_key == "env-key-xyz"
    assert w.model == "MiniMax-M2.7"
    assert w.fallback_model == "MiniMax-M2.5"


def test_worker_default_timeout_is_240s(monkeypatch):
    """Regression: nervous-bus-g3u4 + nervous-bus-96k. cf-7A A/B
    (tools/ab_minimax_hard_case.json) proved that at 60s read_timeout,
    3-4/5 hard-case MiniMax calls die from MiniMax queue scheduling
    variance — not anything we control. Bumped to 200s to give a 65s
    safety margin over budget=1024 worst case of 121s.
    2026-05-17: further bumped 200→240 after observing read-timeouts
    on M2.7 hard cases (session 01KRQTNMM8...). 65s margin retained.
    Do NOT 'optimize' this back to 60s without re-running the A/B.
    """
    monkeypatch.setenv("MINIMAX_API_KEY", "k")

    # (a) default is 240.0
    w = MiniMaxWorker(endpoint_mode="openai")
    assert w.timeout_seconds == 240.0

    # (b) constructor accepts overrides
    w_override = MiniMaxWorker(endpoint_mode="openai", timeout_seconds=45.0)
    assert w_override.timeout_seconds == 45.0

    # (c) httpx Client is configured with that timeout. httpx normalizes a
    # float into a Timeout(connect=N, read=N, write=N, pool=N) object.
    assert w._http_client is not None
    client_timeout = w._http_client.timeout
    assert client_timeout.read == 240.0
    assert w_override._http_client is not None
    assert w_override._http_client.timeout.read == 45.0


def test_worker_rejects_invalid_temperature(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    with pytest.raises(ValueError, match="temperature"):
        MiniMaxWorker(temperature=0.0, endpoint_mode="openai")
    with pytest.raises(ValueError, match="temperature"):
        MiniMaxWorker(temperature=1.5, endpoint_mode="openai")


# ---------------------------------------------------------------------------
# (2) Payload construction
# ---------------------------------------------------------------------------

def _install_mock_client(w: MiniMaxWorker, mock_resp_or_list):
    """Replace w._http_client with a MagicMock whose .post() returns the given
    response (or cycles through a list of responses).

    Supports the persistent-client pattern: there is no ``with`` context
    manager any more — ``_call_minimax`` calls ``self._http_client.post(...)``
    directly.
    """
    mock_client = MagicMock()
    if isinstance(mock_resp_or_list, list):
        mock_client.post.side_effect = mock_resp_or_list
    else:
        mock_client.post.return_value = mock_resp_or_list
    w._http_client = mock_client
    return mock_client


def test_worker_builds_correct_payload(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(temperature=0.4, endpoint_mode="openai")
    harness = _make_harness(system_prompt="SYS", tool_surface="TOOLS")

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_response("print(1)")
    mock_resp.raise_for_status.return_value = None
    mock_client = _install_mock_client(w, mock_resp)

    result = w.generate("solve me", harness, case_id="case-1")

    # post() was called with the right URL + payload
    post_calls = mock_client.post.call_args_list
    assert len(post_calls) == 1
    args, kwargs = post_calls[0]
    assert args[0] == "https://api.minimax.io/v1/chat/completions"
    payload = kwargs["json"]
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["temperature"] == 0.4
    assert payload["max_tokens"] == 2048  # from harness.budget
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "SYS"
    assert payload["messages"][1]["role"] == "user"
    assert "solve me" in payload["messages"][1]["content"]
    assert "TOOLS" in payload["messages"][1]["content"]

    # Headers: bearer auth
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert headers["Content-Type"] == "application/json"

    # Result populated
    assert result.code == "print(1)"
    assert result.model_used == "MiniMax-M2.7"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.error is None


def test_worker_anthropic_endpoint_payload_shape(monkeypatch):
    """Anthropic mode sends system as top-level field and uses /anthropic/v1/messages."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    # Explicit thinking_budget=None so this test focuses on shape only —
    # thinking-field coverage lives in the dedicated test below.
    w = MiniMaxWorker(temperature=0.4, endpoint_mode="anthropic",
                       thinking_budget=None)
    harness = _make_harness(system_prompt="SYS", tool_surface="TOOLS")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "content": [
            {"type": "thinking", "thinking": "internal reasoning that must not leak"},
            {"type": "text", "text": "print(1)"},
        ],
        "usage": {"input_tokens": 42, "output_tokens": 8},
    }
    mock_resp.raise_for_status.return_value = None
    mock_client = _install_mock_client(w, mock_resp)

    result = w.generate("solve me", harness, case_id="case-anth-1")

    args, kwargs = mock_client.post.call_args_list[0]
    # Anthropic endpoint path + shape
    assert args[0] == "https://api.minimax.io/anthropic/v1/messages"
    payload = kwargs["json"]
    assert payload["system"] == "SYS"
    assert payload["messages"] == [{"role": "user", "content": payload["messages"][0]["content"]}]
    assert "solve me" in payload["messages"][0]["content"]
    # No "system" message in the messages array (that's the OpenAI shape)
    assert all(m["role"] != "system" for m in payload["messages"])

    # Result: thinking block deliberately discarded; only text-block content survives.
    assert result.code == "print(1)"
    assert "internal reasoning" not in result.code
    assert "internal reasoning" not in result.raw_response
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 8
    # thinking_budget=None should result in no "thinking" key in payload.
    payload = kwargs["json"]
    assert "thinking" not in payload


def test_extract_code_strips_think_block():
    """Defensive: <think>...</think> blocks are stripped before fence extraction."""
    from autobench.llm.worker import _extract_code
    raw = "<think>\nLet me reason about this.\nThe answer is 42.\n</think>\n\nimport sys\nprint(42)\n"
    assert _extract_code(raw) == "import sys\nprint(42)"

    # Unclosed <think> trailing into EOF — also handled.
    raw_unclosed = "<think>\nReasoning that never closes...\n"
    assert _extract_code(raw_unclosed) == ""

    # <think> wrapping a fenced block — fence extraction still picks the right body.
    raw_with_fence = "<think>thoughts</think>\n```python\nprint('hi')\n```\n"
    assert _extract_code(raw_with_fence) == "print('hi')"


def test_worker_uses_default_prompt_when_harness_empty(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="openai")
    harness = _make_harness(system_prompt="", tool_surface="")

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_response("x=1")
    mock_resp.raise_for_status.return_value = None
    mock_client = _install_mock_client(w, mock_resp)

    w.generate("p", harness)

    payload = mock_client.post.call_args[1]["json"]
    # Default prompt kicks in
    assert "competitive programming" in payload["messages"][0]["content"].lower()


# ---------------------------------------------------------------------------
# (3) Markdown fence stripping
# ---------------------------------------------------------------------------

def test_extract_code_strips_python_fence():
    raw = "```python\nprint('hi')\n```"
    assert _extract_code(raw) == "print('hi')"


def test_extract_code_strips_bare_fence():
    raw = "```\nimport sys\nsys.exit(0)\n```"
    assert _extract_code(raw) == "import sys\nsys.exit(0)"


def test_extract_code_strips_prose_prefix():
    raw = "Here's the solution:\nprint(42)"
    assert _extract_code(raw) == "print(42)"


def test_extract_code_strips_sure_prefix():
    raw = "Sure! print(42)"
    assert _extract_code(raw) == "print(42)"


def test_extract_code_handles_plain_code():
    raw = "import sys\nprint(sys.stdin.read())"
    assert _extract_code(raw) == "import sys\nprint(sys.stdin.read())"


def test_extract_code_empty():
    assert _extract_code("") == ""
    assert _extract_code("   \n  ") == ""


# ---------------------------------------------------------------------------
# (4) Retry / fallback paths
# ---------------------------------------------------------------------------

def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.minimax.io/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req)
    return httpx.HTTPStatusError(f"{status}", request=req, response=resp)


def test_retry_on_5xx_then_success(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(max_retries=3, endpoint_mode="openai")
    harness = _make_harness()

    # First call raises 500, second call succeeds.
    success_resp = MagicMock()
    success_resp.json.return_value = _mock_response("ok_code")
    success_resp.raise_for_status.return_value = None

    err_resp = MagicMock()
    err_resp.raise_for_status.side_effect = _http_error(500)

    mock_client = _install_mock_client(w, [err_resp, success_resp])

    with patch("autobench.llm.worker.time.sleep"):  # don't actually sleep
        result = w.generate("p", harness)

    assert result.code == "ok_code"
    assert result.error is None
    assert mock_client.post.call_count == 2


def test_total_failure_falls_through_to_fallback_then_empty(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(max_retries=2, endpoint_mode="openai")
    harness = _make_harness()

    # All 5xx — both primary's 2 retries and fallback's 2 retries should fail.
    err_resp = MagicMock()
    err_resp.raise_for_status.side_effect = _http_error(503)

    # 2 primary attempts + 2 fallback attempts = 4 total HTTP posts
    mock_client = _install_mock_client(w, err_resp)

    with patch("autobench.llm.worker.time.sleep"):
        result = w.generate("p", harness)

    # Should NOT raise; should return empty code.
    assert result.code == ""
    assert result.error is not None
    assert "primary=" in result.error and "fallback=" in result.error
    # 4 calls (2 primary retries + 2 fallback retries)
    assert mock_client.post.call_count == 4


def test_non_retriable_4xx_bails_immediately(monkeypatch):
    """A 401 (auth fail) should not retry on the same model."""
    monkeypatch.setenv("MINIMAX_API_KEY", "bad-key")
    w = MiniMaxWorker(max_retries=3, endpoint_mode="openai")
    harness = _make_harness()

    err_resp = MagicMock()
    err_resp.raise_for_status.side_effect = _http_error(401)

    mock_client = _install_mock_client(w, err_resp)

    with patch("autobench.llm.worker.time.sleep"):
        result = w.generate("p", harness)

    # Primary bails after 1 attempt; fallback also bails after 1 attempt.
    assert result.code == ""
    assert result.error is not None
    assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# (5) __call__ interface
# ---------------------------------------------------------------------------

def test_callable_returns_just_code(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="openai")
    harness = _make_harness()

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_response("solution_code")
    mock_resp.raise_for_status.return_value = None
    _install_mock_client(w, mock_resp)

    out = w("the prompt", harness)

    assert isinstance(out, str)
    assert out == "solution_code"
    # Usage tracked on _last_usage
    assert w._last_usage["prompt_tokens"] == 100
    assert w._last_usage["completion_tokens"] == 50
    # nervous-bus-dq7l: cost_usd is always 0.0 — pricing tables removed.
    # MiniMax bills by requests-per-5h, not dollars.
    assert w._last_usage["cost_usd"] == 0.0
    assert w._last_usage["model_used"] == "MiniMax-M2.7"


# ---------------------------------------------------------------------------
# (6) Cost calculation
# ---------------------------------------------------------------------------

def test_estimate_cost_always_zero():
    """nervous-bus-dq7l: _estimate_cost is a stub that always returns 0.0.

    Pre-dq7l, this multiplied tokens by a hardcoded $/MTok rate table.
    The MiniMax coding plan bills by requests-per-5h, NOT by tokens × $,
    so any in-tree rate is a fiction the moment list prices shift. The
    function is preserved only to avoid breaking older import sites.
    """
    assert _estimate_cost(1_000_000, 1_000_000, "MiniMax-M2.7") == 0.0
    assert _estimate_cost(500_000, 250_000, "MiniMax-M2.5") == 0.0
    assert _estimate_cost(1_000_000, 0, "some-other-model") == 0.0
    assert _estimate_cost(0, 0, "MiniMax-M2.7") == 0.0


# ---------------------------------------------------------------------------
# (7) Observability — emits autobench.worker.v1
# ---------------------------------------------------------------------------

def test_worker_emits_observability_event(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True  # force debug-file path

    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_response("print(0)", prompt_tokens=80, completion_tokens=20)
    mock_resp.raise_for_status.return_value = None
    _install_mock_client(w, mock_resp)

    w.generate("p", harness, case_id="case-xyz")

    # Read events from the debug file
    events = [json.loads(line) for line in debug_file.read_text().splitlines() if line.strip()]
    worker_events = [e for e in events if e["type"] == CHANNEL_WORKER]
    assert len(worker_events) == 1
    ev = worker_events[0]
    assert ev["type"] == "autobench.worker.v1"
    data = ev["data"]
    assert data["case_id"] == "case-xyz"
    assert data["model"] == "MiniMax-M2.7"
    assert data["prompt_tokens"] == 80
    assert data["completion_tokens"] == 20
    # nervous-bus-dq7l: cost_usd is always 0.0 (no fabricated $ from tokens).
    assert data["cost_usd"] == 0.0
    assert data["latency_ms"] >= 0
    assert "print(0)" in data["code_preview"]
    assert data["session_id"] == obs.session_id


# ---------------------------------------------------------------------------
# (8) Schema validation
# ---------------------------------------------------------------------------

def test_worker_event_validates_against_schema(monkeypatch, tmp_path):
    """Emitted events must conform to schemas/autobench.worker.v1.json."""
    jsonschema = pytest.importorskip("jsonschema")

    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_response("solution")
    mock_resp.raise_for_status.return_value = None
    _install_mock_client(w, mock_resp)

    w.generate("p", harness, case_id="c1")

    events = [json.loads(line) for line in debug_file.read_text().splitlines()]
    worker_events = [e for e in events if e["type"] == CHANNEL_WORKER]
    assert len(worker_events) == 1

    from tests._paths import SCHEMA_DIR
    schema_path = SCHEMA_DIR / "autobench.worker.v1.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(worker_events[0])


# ---------------------------------------------------------------------------
# (9) Backoff helper
# ---------------------------------------------------------------------------

def test_backoff_exponential():
    # No retry-after header — pure exponential
    e = RuntimeError("x")
    b1 = _calc_backoff_ms(1, e)
    b2 = _calc_backoff_ms(2, e)
    b3 = _calc_backoff_ms(3, e)
    # Base values: 1000, 2000, 4000 — with jitter up to 20% each
    assert 1000 <= b1 < 1200
    assert 2000 <= b2 < 2400
    assert 4000 <= b3 < 4800


def test_backoff_honors_retry_after():
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(429, request=req, headers={"Retry-After": "7"})
    e = httpx.HTTPStatusError("429", request=req, response=resp)
    b = _calc_backoff_ms(1, e)
    assert b == 7000


# ---------------------------------------------------------------------------
# (10) thinking_budget — Anthropic extended-thinking cap
# ---------------------------------------------------------------------------

def test_thinking_budget_attached_to_anthropic_payload(monkeypatch):
    """anthropic mode + thinking_budget=512 attaches the correct field."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="anthropic", thinking_budget=512)
    harness = _make_harness(system_prompt="S", tool_surface="")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "content": [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "print(1)"},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_resp.raise_for_status.return_value = None
    mock_client = _install_mock_client(w, mock_resp)

    w.generate("solve me", harness, case_id="case-thinking")

    payload = mock_client.post.call_args[1]["json"]
    assert "thinking" in payload
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 512}


def test_thinking_budget_none_omits_field(monkeypatch):
    """thinking_budget=None must produce a payload WITHOUT the thinking key."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="anthropic", thinking_budget=None)
    harness = _make_harness(system_prompt="S", tool_surface="")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "print(1)"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_resp.raise_for_status.return_value = None
    mock_client = _install_mock_client(w, mock_resp)

    w.generate("solve me", harness, case_id="case-no-think")

    payload = mock_client.post.call_args[1]["json"]
    assert "thinking" not in payload


def test_thinking_budget_omitted_in_openai_mode(monkeypatch):
    """thinking_budget is silently dropped when endpoint_mode=='openai'."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="openai", thinking_budget=512)
    harness = _make_harness()

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_response("print(1)")
    mock_resp.raise_for_status.return_value = None
    mock_client = _install_mock_client(w, mock_resp)

    w.generate("solve me", harness, case_id="case-openai")

    payload = mock_client.post.call_args[1]["json"]
    assert "thinking" not in payload


def test_thinking_budget_clamped_when_exceeds_max_tokens(monkeypatch):
    """thinking_budget > max_tokens is clamped to max_tokens-1 at call time.

    Constructor accepts any positive int because max_tokens is per-harness
    and not knowable until generate() runs. The clamp keeps the model with
    at least one token of visible output room.
    """
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    # Huge budget vs harness max_tokens=2048 → clamps to 2047.
    w = MiniMaxWorker(endpoint_mode="anthropic", thinking_budget=999999)
    harness = _make_harness()
    assert harness.budget["max_tokens"] == 2048

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "x"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    mock_resp.raise_for_status.return_value = None
    mock_client = _install_mock_client(w, mock_resp)

    w.generate("p", harness)

    payload = mock_client.post.call_args[1]["json"]
    assert payload["thinking"]["budget_tokens"] == 2047


def test_thinking_budget_validation_rejects_negative(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    with pytest.raises(ValueError, match="thinking_budget"):
        MiniMaxWorker(thinking_budget=-1)
    with pytest.raises(ValueError, match="thinking_budget"):
        MiniMaxWorker(thinking_budget=0)


def test_thinking_budget_validation_rejects_non_int(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    with pytest.raises(ValueError, match="thinking_budget"):
        MiniMaxWorker(thinking_budget=1.5)  # type: ignore[arg-type]


def test_thinking_budget_default_is_1024(monkeypatch):
    """Default budget should be 1024 — codified in the constructor signature."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker()
    assert w.thinking_budget == 1024


# ---------------------------------------------------------------------------
# (11) Persistent httpx.Client — connection reuse
# ---------------------------------------------------------------------------

def test_persistent_http_client_created_on_init(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="anthropic")
    assert w._http_client is not None
    assert isinstance(w._http_client, httpx.Client)
    w.close()
    assert w._http_client is None


def test_close_is_idempotent(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="anthropic")
    w.close()
    # second close must not raise
    w.close()
    assert w._http_client is None


def test_post_close_falls_back_to_one_shot_client(monkeypatch):
    """If the persistent client is already closed, _call_minimax must not crash.

    Uses the ``httpx.Client`` patch path that the fallback branch hits.
    """
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxWorker(endpoint_mode="openai")
    w.close()  # force fallback path

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_response("recovered")
    mock_resp.raise_for_status.return_value = None
    mock_client.__enter__.return_value.post.return_value = mock_resp

    with patch("autobench.llm.worker.httpx.Client", return_value=mock_client):
        result = w.generate("p", _make_harness())

    assert result.code == "recovered"
