"""nervous-bus-19ur: improver must see the prompt it is editing.

Strategy chosen: (b) full-feed. Append-semantics are preserved (schema-stable
wire format), but ``_build_diagnosis_prompt`` no longer truncates
``system_prompt``/``tool_surface`` — the improver now sees the accumulated
text it is proposing edits against.

These tests pin two invariants across all three improver paths
(MiniMax wrapper, Anthropic wrapper, rsi_loop fallback):

1. No 200-char (or 100-char) truncation marker in the diagnosis prompt.
2. After a 5-iteration accumulation, the diagnosis prompt built at iter 4
   contains content from iter 3's delta (i.e. the most recent append is
   visible to the next improver call).
"""

from __future__ import annotations

import pytest

from autobench.core import (
    ContextManager,
    HarnessConfig,
    HarnessResult,
    RolloutProtocol,
    Verdict,
)
from autobench.evaluator import BenchmarkResult
from autobench.llm.anthropic import AnthropicLLMWrapper
from autobench.llm.minimax import MiniMaxLLMWrapper
from autobench.rsi.loop import ImprovementDelta, _build_diagnosis_prompt


def _bench() -> BenchmarkResult:
    return BenchmarkResult(
        case_results=[
            HarnessResult(verdict=Verdict.CE, p_score=0.0, latency_ms=10),
            HarnessResult(verdict=Verdict.OK, p_score=1.0, latency_ms=11),
        ],
        aggregate_score=0.5,
        total_latency_ms=21.0,
        verdict_counts={"CE": 1, "OK": 1},
    )


def _harness(prompt: str = "solve") -> HarnessConfig:
    return HarnessConfig(
        system_prompt=prompt,
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="bash, python",
        verifiers=[],
        budget={"max_tokens": 8192, "max_time_seconds": 30, "max_cost_dollars": 1.0},
    )


def _simulate_5_iter_accumulation() -> HarnessConfig:
    """Run the production append rule (rsi_loop._apply_delta_to_config style)
    5 times with distinct deltas and return the final harness."""
    h = _harness("BASE_PROMPT")
    for i in range(5):
        delta = ImprovementDelta()
        delta.system_prompt_delta = f"DELTA_MARKER_ITER_{i}"
        h = HarnessConfig(
            system_prompt=h.system_prompt + "\n" + delta.system_prompt_delta,
            rollout_protocol=h.rollout_protocol,
            context_manager=h.context_manager,
            tool_surface=h.tool_surface,
            verifiers=h.verifiers,
            budget=h.budget.copy(),
        )
    return h


def test_rsi_loop_diagnosis_prompt_no_truncation_marker():
    # After 5 iterations the prompt is long; the fallback builder must show
    # the full text (no "..." truncation marker on the system_prompt line).
    h = _simulate_5_iter_accumulation()
    prompt = _build_diagnosis_prompt(h, _bench())
    # Every appended marker is visible.
    for i in range(5):
        assert f"DELTA_MARKER_ITER_{i}" in prompt
    # No legacy first-200 truncation indicator.
    assert "first 200 chars" not in prompt
    assert "first 100 chars" not in prompt


def test_minimax_diagnosis_prompt_full_feed(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    w = MiniMaxLLMWrapper()
    h = _simulate_5_iter_accumulation()
    prompt = w._build_diagnosis_prompt(h, _bench())

    # Iter 3's delta (the N-1 append visible to iter 4's improver) is present.
    assert "DELTA_MARKER_ITER_3" in prompt
    # And iter 4's most-recent append is also visible.
    assert "DELTA_MARKER_ITER_4" in prompt
    # The truncation phrasing is gone.
    assert "first 200 chars" not in prompt
    assert "first 100 chars" not in prompt


def test_anthropic_diagnosis_prompt_full_feed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    w = AnthropicLLMWrapper()
    h = _simulate_5_iter_accumulation()
    prompt = w._build_diagnosis_prompt(h, _bench())

    assert "DELTA_MARKER_ITER_3" in prompt
    assert "DELTA_MARKER_ITER_4" in prompt
    assert "first 200 chars" not in prompt
    assert "first 100 chars" not in prompt
