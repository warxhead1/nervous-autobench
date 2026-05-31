"""Tests for nervous-bus-sf0y: best-iter-keep checkpoint + revert on regression.

Motivating cycle session 01KRSDHKD7M0JQ44AKFY8PR7FN (2026-05-16)::

    iter 0: 0.7031 (OK:18, WA:1, CE:1)  ← BEST
    iter 1: 0.6281 (OK:15, CE:4, WA:1)  ← regressed -0.075 vs best (>2σ=0.027)
    iter 2: 0.7031 (OK:18, WA:1, CE:1)  ← recovered by luck

Without best-iter-keep, iter 2 would have inherited iter 1's harness (carrying
phantom-tool cruft, doubled directives) and any further iterations would have
compounded the wrong-direction edit. With this fix:

  * iter 1's score < best_score - variance_floor → emit
    ``autobench.rsi.checkpoint_revert.v1`` and snap ``harness`` back to
    iter 0's snapshot before the improver runs.
  * The improver's NEXT call sees a REVERT HISTORY block via the
    ``revert_history`` kwarg so it can reconsider its hypothesis instead
    of doubling down.

The variance floor reads from env ``AUTOBENCH_VARIANCE_FLOOR`` (default
0.027). A regression below the floor must NOT trigger a revert — that's noise.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from autobench.core import HarnessConfig, HarnessResult, Verdict
from autobench.evaluator import BenchmarkResult
from autobench.observability import AutobenchObservability
from autobench.rsi_loop import (
    DEFAULT_VARIANCE_FLOOR_2SIGMA,
    ImprovementDelta,
    SelfImprovingHarness,
    _default_variance_floor,
)


def _dummy_case_result() -> HarnessResult:
    """Single OK case so existing total_cases-based math doesn't divide by zero.

    The legacy ``improve_harness`` heuristic baseline gets invoked from the
    obs-emission path and divides by ``len(case_results)``; we keep one case
    so that branch lights up the OK-only path harmlessly.
    """
    return HarnessResult(p_score=1.0, verdict=Verdict.OK)


# --------------------------------------------------------------------------- #
# Test stubs
# --------------------------------------------------------------------------- #


class _ScriptedEvaluator:
    """Returns a scripted score per call; advances index each ``run``."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.run_count = 0
        self.harnesses_seen: list[HarnessConfig] = []

    def run(
        self,
        harness: HarnessConfig,
        cases: list[Any],
        obs: Any = None,
    ) -> BenchmarkResult:
        # Capture by-reference snapshot for cross-iteration identity checks.
        self.harnesses_seen.append(harness)
        idx = min(self.run_count, len(self._scores) - 1)
        self.run_count += 1
        return BenchmarkResult(
            case_results=[_dummy_case_result()],
            aggregate_score=self._scores[idx],
            total_latency_ms=0.0,
            verdict_counts={"OK": 1},
            metadata={},
        )


def _make_marker_improver(marker_per_iter: dict[int, str]):
    """Build an improver that appends a per-iteration marker to system_prompt.

    Lets the test assert that the harness sent into iter N+1 came from a
    specific iteration's snapshot (the markers encode iteration identity).
    """
    def _improver(
        harness: HarnessConfig,
        result: BenchmarkResult,
        iteration: int = 0,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        marker = marker_per_iter.get(iteration, "")
        new = replace(
            harness,
            system_prompt=(harness.system_prompt or "") + f"|edit-from-iter{iteration}:{marker}",
        )
        delta = ImprovementDelta(
            improvement_summary=f"iter{iteration} edit",
            system_prompt_delta=f"edit-from-iter{iteration}:{marker}",
        )
        return new, delta
    return _improver


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_variance_floor_default_matches_constant():
    """No env → default 0.027."""
    import os
    if "AUTOBENCH_VARIANCE_FLOOR" in os.environ:
        os.environ.pop("AUTOBENCH_VARIANCE_FLOOR")
    assert _default_variance_floor() == DEFAULT_VARIANCE_FLOOR_2SIGMA == 0.027


def test_variance_floor_reads_env(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_VARIANCE_FLOOR", "0.05")
    assert _default_variance_floor() == 0.05


def test_variance_floor_env_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_VARIANCE_FLOOR", "not-a-number")
    assert _default_variance_floor() == DEFAULT_VARIANCE_FLOOR_2SIGMA


def test_regression_above_floor_triggers_revert():
    """[0.70, 0.50, 0.55]: iter 1 regresses by 0.20 (>0.027) → revert fires."""
    evaluator = _ScriptedEvaluator(scores=[0.70, 0.50, 0.55])
    obs = AutobenchObservability(session_id="test-sess-sf0y-1")
    base = HarnessConfig(system_prompt="BASE")
    sih = SelfImprovingHarness(
        current_harness=base,
        evaluator=evaluator,  # type: ignore[arg-type]
        max_iterations=3,
        default_improver=None,
        obs=obs,
        variance_floor=0.027,
    )

    # Use markers so we can prove iter 2 starts from iter 0's snapshot.
    improver = _make_marker_improver({0: "A", 1: "B", 2: "C"})
    _, _, history = sih.improve(benchmark_cases=[], improver_fn=improver)

    # Three iterations ran.
    assert evaluator.run_count == 3, f"expected 3 evaluator runs, got {evaluator.run_count}"

    # A revert was recorded for iter 1 → reverted_to_iter=0. (Iter 2's
    # 0.55 is also sub-floor and adds a second entry; the contract only
    # requires the iter-1 entry exist with the right shape.)
    assert len(sih._revert_history) >= 1
    revert = sih._revert_history[0]
    assert revert["iter_regressed"] == 1
    assert revert["best_iter"] == 0
    assert revert["best_score"] == 0.70
    assert revert["iter_score"] == 0.50
    assert abs(revert["regression_delta"] - (-0.20)) < 1e-9  # 0.50 - 0.70
    assert revert["variance_floor_used"] == 0.027

    # Iter 2's evaluator received a harness derived from the best-iter (0)
    # snapshot ("BASE"), NOT from iter 1's harness (which was
    # "BASE|edit-from-iter0:A"). The marker chain proves it:
    #   - iter 0 eval'd "BASE", score 0.70 → best snapshot = "BASE".
    #   - iter 0's improver applied "edit-from-iter0:A" → harness becomes
    #     "BASE|edit-from-iter0:A".
    #   - iter 1 eval'd that, score 0.50 → REGRESSION → revert: harness
    #     snaps back to "BASE", and iter 1's improver appends
    #     "edit-from-iter1:B" → harness becomes "BASE|edit-from-iter1:B".
    #   - iter 2 eval'd "BASE|edit-from-iter1:B".
    # Critically, iter 2's eval prompt does NOT contain "edit-from-iter0:A"
    # — the iter-0 mutation was discarded by the revert (iter 1's improver
    # got a clean "BASE" baseline to edit from). This proves we didn't
    # compound the wrong-direction edits.
    harnesses = evaluator.harnesses_seen
    assert harnesses[0].system_prompt == "BASE"
    assert harnesses[1].system_prompt == "BASE|edit-from-iter0:A"
    # The crux: iter 2 starts from the iter-0 snapshot (BASE), not from
    # iter 1's compounded chain.
    iter2_prompt = harnesses[2].system_prompt
    assert iter2_prompt.startswith("BASE|"), (
        f"iter 2 eval harness should start from BASE snapshot, got {iter2_prompt!r}"
    )
    assert "edit-from-iter0:A" not in iter2_prompt, (
        f"iter 2 eval harness must NOT carry iter-0's mutation after revert "
        f"(it was discarded); got {iter2_prompt!r}"
    )
    assert "edit-from-iter1:B" in iter2_prompt


def test_regression_below_floor_does_not_revert():
    """[0.70, 0.69, 0.71]: regression of 0.01 is BELOW noise floor (0.027) → no revert."""
    evaluator = _ScriptedEvaluator(scores=[0.70, 0.69, 0.71])
    obs = AutobenchObservability(session_id="test-sess-sf0y-2")
    sih = SelfImprovingHarness(
        current_harness=HarnessConfig(system_prompt="BASE"),
        evaluator=evaluator,  # type: ignore[arg-type]
        max_iterations=3,
        default_improver=None,
        obs=obs,
        variance_floor=0.027,
    )
    improver = _make_marker_improver({0: "A", 1: "B", 2: "C"})
    sih.improve(benchmark_cases=[], improver_fn=improver)

    # No revert fired.
    assert sih._revert_history == [], (
        f"sub-floor regression must not trigger a revert; got {sih._revert_history!r}"
    )


def test_revert_event_payload_shape():
    """Verify the observability event carries the contract fields."""
    captured: list[dict[str, Any]] = []

    class _CaptureObs(AutobenchObservability):
        def _publish(self, channel: str, data: dict[str, Any]) -> None:  # type: ignore[override]
            # Mirror the auto-stamp behavior of the real _publish so callers
            # can inspect session_id without having to spin up the pipe.
            data = dict(data)
            data.setdefault("session_id", self.session_id)
            if channel == "autobench.rsi.checkpoint_revert.v1":
                captured.append({"channel": channel, "data": data})

    evaluator = _ScriptedEvaluator(scores=[0.70, 0.50])
    obs = _CaptureObs(session_id="01TESTSESSION")
    sih = SelfImprovingHarness(
        current_harness=HarnessConfig(system_prompt="BASE"),
        evaluator=evaluator,  # type: ignore[arg-type]
        max_iterations=2,
        default_improver=None,
        obs=obs,
        variance_floor=0.027,
    )
    sih.improve(
        benchmark_cases=[],
        improver_fn=_make_marker_improver({0: "A", 1: "B"}),
    )

    assert len(captured) == 1, f"expected 1 checkpoint_revert event, got {len(captured)}"
    evt = captured[0]
    assert evt["channel"] == "autobench.rsi.checkpoint_revert.v1"
    data = evt["data"]
    # _publish auto-stamps session_id; verify it landed.
    assert data["session_id"] == "01TESTSESSION"
    assert data["iter_regressed"] == 1
    assert data["reverted_to_iter"] == 0
    assert data["variance_floor_used"] == 0.027
    # regression_delta = iter_score - best_score = 0.50 - 0.70 = -0.20
    assert abs(data["regression_delta"] - (-0.20)) < 1e-9


def test_improver_sees_revert_context_via_kwarg():
    """The improver's next call receives revert_history with the regression entry."""
    seen_revert_history: list[list[dict[str, Any]]] = []

    def _capture_improver(
        harness: HarnessConfig,
        result: BenchmarkResult,
        iteration: int = 0,
        revert_history: list[dict[str, Any]] | None = None,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        seen_revert_history.append(list(revert_history or []))
        return harness, ImprovementDelta(
            improvement_summary=f"iter{iteration} edit",
            system_prompt_delta=f"x{iteration}",
        )

    evaluator = _ScriptedEvaluator(scores=[0.70, 0.50, 0.55])
    sih = SelfImprovingHarness(
        current_harness=HarnessConfig(system_prompt="BASE"),
        evaluator=evaluator,  # type: ignore[arg-type]
        max_iterations=3,
        default_improver=None,
        obs=None,
        variance_floor=0.027,
    )
    sih.improve(benchmark_cases=[], improver_fn=_capture_improver)

    # Three improver calls, three captured snapshots.
    assert len(seen_revert_history) == 3
    # iter 0: no revert yet
    assert seen_revert_history[0] == []
    # iter 1: still no revert (the revert decision happens AFTER iter 1's
    # eval but BEFORE iter 1's improver call? Re-check: the loop captures
    # result then runs best-iter logic which may set _revert_history,
    # THEN runs the improver. So iter 1's improver call should see the
    # revert entry recorded after iter 1's own eval).
    assert len(seen_revert_history[1]) == 1
    assert seen_revert_history[1][0]["iter_regressed"] == 1
    # iter 2: 0.55 is ALSO below 0.70-floor → second revert lands before
    # iter 2's improver call. So the improver sees both reverts.
    assert len(seen_revert_history[2]) == 2
    assert seen_revert_history[2][0]["iter_regressed"] == 1
    assert seen_revert_history[2][1]["iter_regressed"] == 2


def test_diagnosis_prompt_includes_revert_history_block():
    """When revert_history is non-empty, the MiniMax diagnosis prompt
    surfaces a REVERT HISTORY section."""
    from autobench.minimax_improver import MiniMaxLLMWrapper, _format_revert_history_block

    rendered = _format_revert_history_block([
        {
            "iter_regressed": 1,
            "iter_score": 0.628,
            "best_score": 0.703,
            "best_iter": 0,
            "regression_delta": -0.075,
            "variance_floor_used": 0.027,
        }
    ])
    assert "REVERT HISTORY" in rendered
    assert "iter 1 regressed" in rendered
    assert "-0.075" in rendered
    assert "iter 0" in rendered
    assert "Reconsider hypothesis" in rendered

    # Empty / None → empty string.
    assert _format_revert_history_block(None) == ""
    assert _format_revert_history_block([]) == ""

    # Verify it actually lands in the diagnosis prompt body.
    wrapper = MiniMaxLLMWrapper(api_key="test")
    bench = BenchmarkResult(
        case_results=[], aggregate_score=0.6, total_latency_ms=0.0,
        verdict_counts={"OK": 1}, metadata={},
    )
    prompt = wrapper._build_diagnosis_prompt(
        HarnessConfig(),
        bench,
        revert_history=[{
            "iter_regressed": 1, "iter_score": 0.5, "best_score": 0.7,
            "best_iter": 0, "regression_delta": -0.2, "variance_floor_used": 0.027,
        }],
    )
    assert "REVERT HISTORY" in prompt
    assert "iter 1 regressed" in prompt


def test_default_improvement_threshold_is_variance_floor(monkeypatch):
    """Constructor default for improvement_threshold matches the 2σ floor.

    nervous-bus-sf0y: the legacy 0.01 default sat below the noise floor
    and caused the loop to read pure measurement noise as "convergence".
    Both fields now default to the same env-overridable value so the
    early-exit threshold and the revert floor share calibration.
    """
    monkeypatch.delenv("AUTOBENCH_VARIANCE_FLOOR", raising=False)
    sih = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=_ScriptedEvaluator(scores=[0.0]),  # type: ignore[arg-type]
        max_iterations=1,
        default_improver=None,
    )
    assert sih.improvement_threshold == DEFAULT_VARIANCE_FLOOR_2SIGMA
    assert sih.variance_floor == DEFAULT_VARIANCE_FLOOR_2SIGMA


def test_default_threshold_honours_env_override(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_VARIANCE_FLOOR", "0.05")
    sih = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=_ScriptedEvaluator(scores=[0.0]),  # type: ignore[arg-type]
        max_iterations=1,
        default_improver=None,
    )
    assert sih.improvement_threshold == 0.05
    assert sih.variance_floor == 0.05
