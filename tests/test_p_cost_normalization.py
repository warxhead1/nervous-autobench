"""Tests for real-worker-cost-driven p_cost normalization (nervous-bus-569q).

Pre-fix, `BenchmarkEvaluator._run_case` hardcoded `p_cost = 0.5` for every
HarnessResult. The SICA formula in `score_harness` weights cost at 0.25, so
every iteration's aggregate_score gained a fixed `0.25 * (1 - 0.5) = 0.125`
from the cost term regardless of actual spend — pure noise relative to the
convergence signal the improver verifies AHE predictions against.

This module asserts:

1. Two stub workers with different `_last_usage["cost_usd"]` produce two
   different `p_cost` values, both clamped to [0, 1].
2. A worker with zero/missing usage produces `p_cost == 0.0` (not 0.5).
3. The SICA aggregate_score for a known-cost run matches the formula
   exactly — no residual 0.5 stub.
4. Cost above `max_cost_per_case` clamps to 1.0.
"""

from __future__ import annotations

import pytest

from autobench.core import HarnessConfig, RolloutProtocol, Verdict
from autobench.evaluator import (
    DEFAULT_MAX_COST_PER_CASE_USD,
    DEFAULT_WEIGHTS,
    BenchmarkCase,
    BenchmarkEvaluator,
    _normalize_p_cost,
)


class _StubWorker:
    """Minimal MiniMaxWorker stand-in exposing ._last_usage like the real one."""

    def __init__(self, cost_usd: float, code: str = "print(1)\n") -> None:
        self._code = code
        self._last_usage: dict[str, object] = {
            "cost_usd": cost_usd,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "model": "stub",
        }

    def __call__(self, _prompt: str, _cfg: HarnessConfig) -> str:
        return self._code


def _case(case_id: str) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        prompt="print(1)",
        language="python",
        expected_output="1\n",
    )


# --------------------------------------------------------------------------- #
# Unit tests for the pure helper
# --------------------------------------------------------------------------- #


def test_normalize_p_cost_scales_linearly_below_anchor() -> None:
    # 0.001 / 0.10 = 0.01
    assert _normalize_p_cost({"cost_usd": 0.001}, 0.10) == pytest.approx(0.01)
    # 0.05 / 0.10 = 0.5
    assert _normalize_p_cost({"cost_usd": 0.05}, 0.10) == pytest.approx(0.5)


def test_normalize_p_cost_clamps_above_anchor() -> None:
    # Cost exceeding the per-case anchor saturates at 1.0
    assert _normalize_p_cost({"cost_usd": 0.50}, 0.10) == 1.0
    assert _normalize_p_cost({"cost_usd": 999.0}, 0.10) == 1.0


def test_normalize_p_cost_handles_missing_usage() -> None:
    # No usage dict, no cost field → 0.0 (NOT the legacy 0.5 stub)
    assert _normalize_p_cost(None, 0.10) == 0.0
    assert _normalize_p_cost({}, 0.10) == 0.0
    assert _normalize_p_cost({"cost_usd": 0.0}, 0.10) == 0.0
    # Wrong type → defensive 0.0
    assert _normalize_p_cost("not-a-dict", 0.10) == 0.0  # type: ignore[arg-type]


def test_normalize_p_cost_accepts_alternate_cost_keys() -> None:
    # iteration_summary.normalize_worker_usage tolerates total_cost_usd,
    # cost, total_cost — keep parity here.
    assert _normalize_p_cost({"total_cost_usd": 0.05}, 0.10) == pytest.approx(0.5)
    assert _normalize_p_cost({"cost": 0.05}, 0.10) == pytest.approx(0.5)
    assert _normalize_p_cost({"total_cost": 0.05}, 0.10) == pytest.approx(0.5)


def test_normalize_p_cost_falls_back_to_default_anchor_on_zero() -> None:
    # If caller passes max_cost_per_case <= 0, fall back to the default.
    p = _normalize_p_cost({"cost_usd": DEFAULT_MAX_COST_PER_CASE_USD / 2}, 0.0)
    assert p == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Integration: end-to-end via BenchmarkEvaluator.run
# --------------------------------------------------------------------------- #


def test_evaluator_p_cost_differs_for_different_worker_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two workers with different _last_usage produce different p_cost."""
    # Bypass real sandbox: stub out execute() to return a deterministic OK.
    from autobench.sandbox import ExecutionResult, SandboxedExecutor

    def _fake_execute(self, code: str, language: str, **kwargs):  # noqa: ANN001
        return ExecutionResult(stdout="1\n", stderr="", exit_code=0, latency_ms=10.0)

    monkeypatch.setattr(SandboxedExecutor, "execute", _fake_execute)

    cheap = _StubWorker(cost_usd=0.001)
    pricey = _StubWorker(cost_usd=0.10)

    # Default budget: max_cost_dollars=0.10, one case → anchor = 0.10/1 = 0.10
    harness = HarnessConfig(rollout_protocol=RolloutProtocol.SINGLE)

    cheap_result = BenchmarkEvaluator(generate_fn=cheap).run(harness, [_case("c1")])
    pricey_result = BenchmarkEvaluator(generate_fn=pricey).run(harness, [_case("c2")])

    cheap_p = cheap_result.case_results[0].p_cost
    pricey_p = pricey_result.case_results[0].p_cost

    # Both in [0, 1] and distinct
    assert 0.0 <= cheap_p <= 1.0
    assert 0.0 <= pricey_p <= 1.0
    assert cheap_p != pricey_p
    # Neither equals the legacy stub
    assert cheap_p != 0.5
    # Cheap should be a small fraction; pricey should saturate at anchor (1.0)
    assert cheap_p == pytest.approx(0.001 / 0.10)
    assert pricey_p == pytest.approx(1.0)


def test_evaluator_aggregate_score_matches_sica_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aggregate_score equals U = w_score·p_score + w_cost·(1-p_cost) + w_time·(1-p_time).

    nervous-bus-isgo: cost weight is now 0.0 (dq7l removed all $-rate
    tables). U effectively reduces to 0.5·p_score + 0.5·(1-p_time). This
    test exercises the formula against whatever DEFAULT_WEIGHTS happens
    to hold today rather than pinning a magic number.
    """
    from autobench.sandbox import ExecutionResult, SandboxedExecutor

    # Deterministic OK with 0ms latency → p_time = 1.0
    def _fake_execute(self, code, language, **kwargs):  # noqa: ANN001
        return ExecutionResult(stdout="1\n", stderr="", exit_code=0, latency_ms=0.0)

    monkeypatch.setattr(SandboxedExecutor, "execute", _fake_execute)

    # Worker with cost = 0.05; anchor = 0.10 (single case, default budget)
    worker = _StubWorker(cost_usd=0.05)
    harness = HarnessConfig(rollout_protocol=RolloutProtocol.SINGLE)
    result = BenchmarkEvaluator(generate_fn=worker).run(harness, [_case("c1")])

    case = result.case_results[0]
    # p_score=1.0 (OK), p_cost=0.5, p_time≈1.0
    assert case.p_score == 1.0
    assert case.p_cost == pytest.approx(0.5)
    assert case.p_time == pytest.approx(1.0)

    expected = (
        DEFAULT_WEIGHTS["score"] * case.p_score
        + DEFAULT_WEIGHTS["cost"] * (1.0 - case.p_cost)
        + DEFAULT_WEIGHTS["time"] * (1.0 - case.p_time)
    )
    assert result.aggregate_score == pytest.approx(expected, abs=1e-9)

    # Sanity: a cheaper run should produce the same aggregate as long as
    # cost weight is 0 (post-isgo). When cost weight is non-zero, a
    # cheaper run scores higher. Either invariant must hold.
    worker_cheap = _StubWorker(cost_usd=0.001)
    cheap_result = BenchmarkEvaluator(generate_fn=worker_cheap).run(harness, [_case("c2")])
    cheap_case = cheap_result.case_results[0]
    cheap_expected = (
        DEFAULT_WEIGHTS["score"] * cheap_case.p_score
        + DEFAULT_WEIGHTS["cost"] * (1.0 - cheap_case.p_cost)
        + DEFAULT_WEIGHTS["time"] * (1.0 - cheap_case.p_time)
    )
    assert cheap_result.aggregate_score == pytest.approx(cheap_expected, abs=1e-9)
    if DEFAULT_WEIGHTS["cost"] > 0:
        # Old-style invariant: cheaper run scores strictly higher.
        assert cheap_result.aggregate_score > result.aggregate_score
    else:
        # Post-isgo invariant: cost no longer affects the aggregate.
        assert cheap_result.aggregate_score == pytest.approx(result.aggregate_score, abs=1e-9)
