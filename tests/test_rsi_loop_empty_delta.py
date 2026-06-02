"""Tests for nervous-bus-dxuq: empty delta_summary must NOT halt the run.

When the improver fallback (rule-based or malformed-LLM-parse path) yields
an ``ImprovementDelta`` with no changes, the RSI loop must continue to the
next iteration with the current harness — not break out of the loop via
the threshold-based early-exit (which is intended to detect plateau, not
"improver did nothing").
"""

from __future__ import annotations

from typing import Any

from autobench.core import HarnessConfig
from autobench.evaluator import BenchmarkEvaluator, BenchmarkResult
from autobench.rsi.loop import ImprovementDelta, SelfImprovingHarness


class _StubEvaluator:
    """Returns a fixed-score BenchmarkResult, no real cases executed."""

    def __init__(self, score: float = 0.0) -> None:
        self._score = score
        self.run_count = 0

    def run(self, harness: HarnessConfig, cases: list[Any], obs: Any = None) -> BenchmarkResult:
        self.run_count += 1
        return BenchmarkResult(
            case_results=[],
            aggregate_score=self._score,
            total_latency_ms=0.0,
            verdict_counts={},
            metadata={},
        )


def _empty_delta_improver(
    harness: HarnessConfig,
    result: BenchmarkResult,
    iteration: int = 0,
) -> tuple[HarnessConfig, ImprovementDelta]:
    """Stub improver simulating the fallback path with empty delta."""
    return harness, ImprovementDelta()  # all fields default → empty / no-op


def test_empty_delta_does_not_halt_run() -> None:
    """Empty delta_summary must let the loop proceed to iter 1, not early-exit."""
    evaluator = _StubEvaluator(score=0.0)
    harness = HarnessConfig()
    sih = SelfImprovingHarness(
        current_harness=harness,
        evaluator=evaluator,  # type: ignore[arg-type]
        max_iterations=3,
        improvement_threshold=0.02,
        default_improver=None,
        obs=None,
    )
    _, _, history = sih.improve(benchmark_cases=[], improver_fn=_empty_delta_improver)

    # Pre-fix bug: loop broke after iter 0 because curr_score(0)==prev_score(0)
    # tripped the threshold check, leaving history with a single entry.
    # Post-fix: noop deltas skip the threshold check, so we exhaust max_iterations.
    assert evaluator.run_count == 3, (
        f"expected 3 evaluator runs (max_iterations), got {evaluator.run_count}; "
        f"the loop halted early on empty delta"
    )
    assert len(history) == 3, f"expected 3 history entries, got {len(history)}"


def test_empty_delta_proceeds_past_iter_zero() -> None:
    """Minimal AC: loop reaches iter 1 even when iter 0's improver returned empty."""
    evaluator = _StubEvaluator(score=0.5)
    sih = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=evaluator,  # type: ignore[arg-type]
        max_iterations=2,
        improvement_threshold=0.02,
        default_improver=None,
        obs=None,
    )
    _, _, history = sih.improve(benchmark_cases=[], improver_fn=_empty_delta_improver)
    assert evaluator.run_count >= 2, (
        "loop must proceed to iter 1 after iter 0's empty delta"
    )
    assert len(history) >= 2
