"""Tests for improver reasoning capture + divergence detection.

Covers:
    * MiniMax-style LLM call: reasoning event emitted with parse_status=ok.
    * MiniMax garbage response: reasoning event emitted with
      parse_status="fell_back_to_rule_based".
    * Rule-based path: reasoning event emitted with model="rule_based" and
      raw_response="matched: <strategy>".
    * Divergence event fires once per iteration.
    * divergent=True when LLM and heuristic disagree.
    * divergent=False when LLM and heuristic match.
    * All emitted events validate against the new JSON schemas.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.core import (
    ContextManager,
    HarnessConfig,
    HarnessResult,
    RolloutProtocol,
    Verdict,
)
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator, BenchmarkResult
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_IMPROVER_DIVERGENCE,
    CHANNEL_IMPROVER_REASONING,
    _dict_diff,
)
from autobench.rsi.loop import ImprovementDelta, SelfImprovingHarness



SCHEMA_REASONING = SCHEMA_DIR / "autobench.improver.reasoning.v1.json"
SCHEMA_DIVERGENCE = SCHEMA_DIR / "autobench.improver.divergence.v1.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Clean debug file with PATH stripped so zellij-pipe always fails."""
    path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


def _make_harness() -> HarnessConfig:
    return HarnessConfig(
        system_prompt="solve coding problems",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="bash, python",
        verifiers=[],
        budget={"max_tokens": 8192, "max_time_seconds": 30, "max_cost_dollars": 1.0},
    )


def _make_bench_result_ce_dominant() -> BenchmarkResult:
    """CE-dominant verdict — rule-based will pick the CE branch."""
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
    """OpenAI-shaped response for wrappers explicitly in endpoint_mode='openai'."""
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


def _mock_anthropic_response(content: str, *, input_tokens: int = 120,
                             output_tokens: int = 80) -> dict:
    """Anthropic-shaped response — required when ``SelfImprovingHarness``
    spins up its own wrapper (which defaults to ``endpoint_mode='anthropic'``)
    or when a test passes ``endpoint_mode='anthropic'`` explicitly.
    """
    return {
        "id": "msg-test",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


# --------------------------------------------------------------------------- #
# 1. MiniMax — successful call emits reasoning with parse_status="ok"
# --------------------------------------------------------------------------- #


def test_minimax_success_emits_reasoning_ok(monkeypatch, debug_file):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    from autobench.llm.minimax import MiniMaxLLMWrapper

    obs = AutobenchObservability(debug_file=debug_file)
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")

    llm_json = json.dumps({
        "system_prompt_changes": "Add worked examples.",
        "rollout_protocol": "iterative",
        "context_manager": "keep",
        "tool_surface_changes": "",
        "budget_changes": {"max_tokens": 6553},
        "rationale": "Reduce over-generation",
    })

    with patch.object(
        wrapper, "_call_with_retries",
        return_value=_mock_chat_response(llm_json),
    ):
        result = wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result_ce_dominant(),
            obs=obs, iteration=3,
        )

    assert result.delta.improvement_summary == "Reduce over-generation"
    events = _events_on(debug_file, CHANNEL_IMPROVER_REASONING)
    assert len(events) == 1
    data = events[0]["data"]
    assert data["model"] == "MiniMax-M2.7"
    assert data["iteration"] == 3
    assert data["parse_status"] == "ok"
    assert "Add worked examples." in data["raw_response"]
    assert data["parsed_delta"]["system_prompt_delta"] == "Add worked examples."
    assert data["parsed_delta"]["budget_delta"] == {"max_tokens": 6553}
    assert data["parsed_delta"]["rollout_protocol_changed"] is True
    assert data["latency_ms"] >= 0.0
    assert "cost_dollars" in data


# --------------------------------------------------------------------------- #
# 2. MiniMax — garbage response falls back to rule-based, status reflects it
# --------------------------------------------------------------------------- #


def test_minimax_garbage_response_emits_fallback(monkeypatch, debug_file):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    from autobench.llm.minimax import MiniMaxLLMWrapper

    obs = AutobenchObservability(debug_file=debug_file)
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")

    # Response with no JSON object — _parse_llm_response will return delta
    # untouched (all default), tripping the "fell_back_to_rule_based" branch.
    garbage = "this is not json at all, just narrative"

    with patch.object(
        wrapper, "_call_with_retries",
        return_value=_mock_chat_response(garbage),
    ):
        wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result_ce_dominant(),
            obs=obs, iteration=1,
        )

    events = _events_on(debug_file, CHANNEL_IMPROVER_REASONING)
    assert len(events) == 1
    data = events[0]["data"]
    assert data["parse_status"] == "fell_back_to_rule_based"
    assert data["raw_response"] == garbage
    assert "fallback_reason" in data


def test_minimax_http_failure_emits_fallback(monkeypatch, debug_file):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    from autobench.llm.minimax import MiniMaxLLMWrapper

    obs = AutobenchObservability(debug_file=debug_file)
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")

    with patch.object(
        wrapper, "_call_with_retries",
        side_effect=RuntimeError("network down"),
    ):
        wrapper.suggest_harness_improvements(
            _make_harness(), _make_bench_result_ce_dominant(),
            obs=obs, iteration=0,
        )

    events = _events_on(debug_file, CHANNEL_IMPROVER_REASONING)
    assert len(events) == 1
    data = events[0]["data"]
    assert data["parse_status"] == "fell_back_to_rule_based"
    assert "network down" in data["fallback_reason"]


# --------------------------------------------------------------------------- #
# 3. Rule-based path emits reasoning with model="rule_based"
# --------------------------------------------------------------------------- #


def test_rule_based_path_emits_reasoning(debug_file):
    obs = AutobenchObservability(debug_file=debug_file)
    evaluator = BenchmarkEvaluator()
    harness = SelfImprovingHarness(
        current_harness=_make_harness(),
        evaluator=evaluator,
        max_iterations=1,
        default_improver="rule_based",
        obs=obs,
    )

    # Inject a pre-canned benchmark result so we don't actually exec anything.
    # We do this by patching evaluator.run() to return our CE-dominant result.
    with patch.object(
        evaluator, "run",
        return_value=_make_bench_result_ce_dominant(),
    ):
        harness.improve([
            BenchmarkCase(id="p1", prompt="print(1)", expected_output="1\n",
                          language="python")
        ])

    events = _events_on(debug_file, CHANNEL_IMPROVER_REASONING)
    assert len(events) >= 1
    rule_events = [e for e in events if e["data"]["model"] == "rule_based"]
    assert len(rule_events) >= 1
    data = rule_events[0]["data"]
    assert data["raw_response"].startswith("matched strategy:")
    # CE-dominant input → ce branch.
    assert "ce_dominant" in data["raw_response"]
    assert data["parse_status"] == "ok"


# --------------------------------------------------------------------------- #
# 4. Divergence event fires once per iteration
# --------------------------------------------------------------------------- #


def test_divergence_fires_once_per_iteration(debug_file):
    obs = AutobenchObservability(debug_file=debug_file)
    evaluator = BenchmarkEvaluator()
    harness = SelfImprovingHarness(
        current_harness=_make_harness(),
        evaluator=evaluator,
        max_iterations=2,
        improvement_threshold=-1.0,  # disable early-exit on score plateau
        default_improver="rule_based",
        obs=obs,
    )

    with patch.object(
        evaluator, "run",
        return_value=_make_bench_result_ce_dominant(),
    ):
        harness.improve([
            BenchmarkCase(id="p1", prompt="print(1)", expected_output="1\n",
                          language="python")
        ])

    events = _events_on(debug_file, CHANNEL_IMPROVER_DIVERGENCE)
    # The rule-based path triggers convergence after 3 plateau iterations,
    # but with max_iterations=2 we should see exactly 2 divergence events.
    assert len(events) == 2, f"expected 2 divergence events, got {len(events)}"
    for ev in events:
        # When LLM path IS the rule-based path, deltas must match → divergent=False.
        assert ev["data"]["divergent"] is False
        assert ev["data"]["divergence_summary"] == "no divergence"


def test_divergence_true_when_llm_disagrees_with_heuristic(monkeypatch, debug_file):
    """Wire a MiniMax wrapper that proposes a non-CE strategy on CE-dominant input."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    from autobench.llm.minimax import MiniMaxLLMWrapper

    obs = AutobenchObservability(debug_file=debug_file)
    wrapper = MiniMaxLLMWrapper(endpoint_mode="openai")

    # LLM proposes a context_manager switch — rule-based on CE would only
    # touch budget.max_tokens → the two deltas will differ.
    llm_json = json.dumps({
        "system_prompt_changes": "",
        "rollout_protocol": "keep",
        "context_manager": "hierarchical",
        "tool_surface_changes": "",
        "budget_changes": {},
        "rationale": "Switch context manager",
    })

    evaluator = BenchmarkEvaluator()
    harness_obj = SelfImprovingHarness(
        current_harness=_make_harness(),
        evaluator=evaluator,
        max_iterations=1,
        default_improver="minimax",
        obs=obs,
    )

    # SelfImprovingHarness builds MiniMaxLLMWrapper with endpoint_mode="anthropic"
    # by default, so the mocked response must be Anthropic-shaped.
    with patch.object(
        evaluator, "run",
        return_value=_make_bench_result_ce_dominant(),
    ), patch(
        "autobench.llm.minimax.MiniMaxLLMWrapper._call_with_retries",
        return_value=_mock_anthropic_response(llm_json),
    ):
        harness_obj.improve([
            BenchmarkCase(id="p1", prompt="print(1)", expected_output="1\n",
                          language="python")
        ])

    events = _events_on(debug_file, CHANNEL_IMPROVER_DIVERGENCE)
    assert len(events) == 1
    data = events[0]["data"]
    assert data["divergent"] is True
    assert data["divergence_summary"] != "no divergence"
    # Heuristic on CE-dominant: budget_delta has max_tokens; context_manager_changed=False
    assert data["heuristic_delta"]["budget_delta"]  # non-empty
    assert data["heuristic_delta"]["context_manager_changed"] is False
    # LLM: context_manager_changed=True; budget_delta empty
    assert data["llm_delta"]["context_manager_changed"] is True


# --------------------------------------------------------------------------- #
# 5. Schema validation
# --------------------------------------------------------------------------- #


def test_new_schemas_load() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    for path in (SCHEMA_REASONING, SCHEMA_DIVERGENCE):
        assert path.exists(), f"missing schema: {path}"
        schema = json.loads(path.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)


def test_all_reasoning_and_divergence_events_validate(monkeypatch, debug_file):
    """Run a small RSI loop with a mocked LLM and verify every emitted event
    on the two new channels validates against its schema."""
    jsonschema = pytest.importorskip("jsonschema")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

    obs = AutobenchObservability(debug_file=debug_file)
    evaluator = BenchmarkEvaluator()
    harness_obj = SelfImprovingHarness(
        current_harness=_make_harness(),
        evaluator=evaluator,
        max_iterations=1,
        default_improver="minimax",
        obs=obs,
    )

    llm_json = json.dumps({
        "system_prompt_changes": "Add output validation.",
        "rollout_protocol": "iterative",
        "context_manager": "keep",
        "tool_surface_changes": "",
        "budget_changes": {"max_tokens": 4096},
        "rationale": "TLE risk",
    })

    # SelfImprovingHarness builds MiniMaxLLMWrapper with endpoint_mode="anthropic"
    # by default, so the mocked response must be Anthropic-shaped.
    with patch.object(
        evaluator, "run",
        return_value=_make_bench_result_ce_dominant(),
    ), patch(
        "autobench.llm.minimax.MiniMaxLLMWrapper._call_with_retries",
        return_value=_mock_anthropic_response(llm_json),
    ):
        harness_obj.improve([
            BenchmarkCase(id="p1", prompt="print(1)", expected_output="1\n",
                          language="python")
        ])

    reasoning_schema = json.loads(SCHEMA_REASONING.read_text())
    divergence_schema = json.loads(SCHEMA_DIVERGENCE.read_text())
    rv = jsonschema.Draft202012Validator(reasoning_schema)
    dv = jsonschema.Draft202012Validator(divergence_schema)

    r_events = _events_on(debug_file, CHANNEL_IMPROVER_REASONING)
    d_events = _events_on(debug_file, CHANNEL_IMPROVER_DIVERGENCE)
    assert r_events, "expected at least one reasoning event"
    assert d_events, "expected at least one divergence event"

    for ev in r_events:
        errors = sorted(rv.iter_errors(ev), key=lambda e: list(e.path))
        assert not errors, f"reasoning schema failed: {[e.message for e in errors]}"
    for ev in d_events:
        errors = sorted(dv.iter_errors(ev), key=lambda e: list(e.path))
        assert not errors, f"divergence schema failed: {[e.message for e in errors]}"


# --------------------------------------------------------------------------- #
# 6. _dict_diff unit tests
# --------------------------------------------------------------------------- #


def test_dict_diff_empty_when_equal():
    a = asdict(ImprovementDelta(system_prompt_delta="x"))
    b = asdict(ImprovementDelta(system_prompt_delta="x"))
    assert _dict_diff(a, b) == ""


def test_dict_diff_ignores_summary_only_changes():
    a = asdict(ImprovementDelta(improvement_summary="alpha"))
    b = asdict(ImprovementDelta(improvement_summary="beta"))
    # Only summary differs → not structural → empty (means: not divergent).
    assert _dict_diff(a, b) == ""


def test_dict_diff_reports_budget_change():
    a = asdict(ImprovementDelta(budget_delta={}))
    b = asdict(ImprovementDelta(budget_delta={"max_tokens": 4096}))
    s = _dict_diff(a, b)
    assert "budget_delta" in s
    assert "max_tokens" in s
