"""Tests for live (partial) AHE prediction refutation — nervous-bus-ykn.

Covers:
    * refute_live unit cases (no refutation, refuted, all-actual, empty changes)
    * End-to-end integration: a 3-iteration mock RSI run where the iter-0
      prediction becomes unachievable mid-iter-1; assert exactly one
      ``autobench.improver.prediction.refuted_live.v1`` event fires with the
      correct payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.audit.ahe import (
    LiveRefutationStatus,
    Prediction,
    refute_live,
)
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_PREDICTION_REFUTED_LIVE,
)


REFUTED_LIVE_SCHEMA = SCHEMA_DIR / "autobench.improver.prediction.refuted_live.v1.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Clean debug file + neutered PATH so zellij pipe never succeeds."""
    path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


def _validate_with_schema(event: dict, schema_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(event)


# --------------------------------------------------------------------------- #
# Unit tests for refute_live
# --------------------------------------------------------------------------- #

def test_refute_live_not_yet_refuted_no_cases_done() -> None:
    """+30 OK predicted, 0 cases done, 40 remaining → still achievable.

    With prior OK count = 1, the variance-aware upper bound still allows
    headroom for +30 OK given sufficient remaining cases and a non-zero
    prior rate estimate.
    """
    pred = Prediction(
        predicted_score_delta=0.1,
        predicted_verdict_class_changes={"OK": 30},
        confidence=0.8,
        rationale="big jump",
    )
    # Prior had 1 OK out of some total; the variance-aware bound accounts
    # for remaining-case distribution variance.
    status = refute_live(
        prediction=pred,
        actuals_so_far={},
        remaining_cases=40,
        prior_iter_counts={"OK": 1},
    )
    assert status.is_refuted is False
    assert status.refutation_reason == ""
    assert status.confidence_at_refute is None


def test_refute_live_refuted_when_max_achievable_falls_short() -> None:
    """+30 OK predicted, 5 OK seen, 2 remaining → refuted.

    Prior OK count = 0. Variance-aware max achievable is computed from
    upper 95% confidence bound on remaining-case verdicts. With 5 so_far
    and 2 remaining, max achievable ≈ +5 (not the old optimistic +7).
    """
    pred = Prediction(
        predicted_score_delta=0.1,
        predicted_verdict_class_changes={"OK": 30},
        confidence=0.9,
        rationale="overconfident",
    )
    status = refute_live(
        prediction=pred,
        actuals_so_far={"OK": 5, "CE": 13},
        remaining_cases=2,
        prior_iter_counts={"OK": 0},
    )
    assert status.is_refuted is True
    assert "OK" in status.refutation_reason
    assert "+30" in status.refutation_reason
    # variance-aware upper bound yields +5 with z=1.645 for 2 remaining cases
    assert "+5" in status.refutation_reason  # max achievable
    assert status.confidence_at_refute == pytest.approx(0.9)


def test_refute_live_negative_delta_refuted_when_floor_too_high() -> None:
    """-5 TLE predicted; but we've already seen 8 TLE, prior was 5.

    Prior TLE = 5. Predicted delta -5 means TLE count should be 0. We've
    seen 8 TLE already, so floor delta = 8 - 5 = +3. Predicted -5 < +3 →
    refuted.
    """
    pred = Prediction(
        predicted_score_delta=-0.05,
        predicted_verdict_class_changes={"TLE": -5},
        confidence=0.7,
        rationale="should kill TLE",
    )
    status = refute_live(
        prediction=pred,
        actuals_so_far={"TLE": 8, "OK": 2},
        remaining_cases=5,
        prior_iter_counts={"TLE": 5, "OK": 5},
    )
    assert status.is_refuted is True
    assert "TLE" in status.refutation_reason


def test_refute_live_matches_actual_not_refuted() -> None:
    """Prediction matches what actually happened → not refuted."""
    pred = Prediction(
        predicted_score_delta=0.05,
        predicted_verdict_class_changes={"OK": 3, "TLE": -3},
        confidence=0.8,
        rationale="tighten",
    )
    # Prior: OK=5, TLE=5. We expect OK→8, TLE→2 (deltas match prediction).
    # All 10 cases done, 0 remaining, actuals exactly match prediction.
    status = refute_live(
        prediction=pred,
        actuals_so_far={"OK": 8, "TLE": 2},
        remaining_cases=0,
        prior_iter_counts={"OK": 5, "TLE": 5},
    )
    assert status.is_refuted is False
    assert status.refutation_reason == ""
    assert status.confidence_at_refute is None


def test_refute_live_empty_predicted_changes_not_refuted() -> None:
    """No verdict-class changes predicted → vacuously not refuted."""
    pred = Prediction(
        predicted_score_delta=0.02,
        predicted_verdict_class_changes={},
        confidence=0.5,
        rationale="tiny score bump",
    )
    status = refute_live(
        prediction=pred,
        actuals_so_far={"OK": 3, "WA": 2},
        remaining_cases=0,
        prior_iter_counts={"OK": 5, "WA": 0},
    )
    assert status.is_refuted is False
    assert status.refutation_reason == ""


def test_refute_live_zero_delta_entries_ignored() -> None:
    """Entries with predicted_delta == 0 should not contribute to refutation."""
    pred = Prediction(
        predicted_score_delta=0.0,
        predicted_verdict_class_changes={"OK": 0, "TLE": 0},
        confidence=0.4,
        rationale="no-op",
    )
    status = refute_live(
        prediction=pred,
        actuals_so_far={"OK": 100, "TLE": 100},
        remaining_cases=0,
        prior_iter_counts={"OK": 0, "TLE": 0},
    )
    assert status.is_refuted is False


# --------------------------------------------------------------------------- #
# Observability emission
# --------------------------------------------------------------------------- #

def test_emit_prediction_refuted_live_envelope_and_schema(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    pred = Prediction(
        predicted_score_delta=0.1,
        predicted_verdict_class_changes={"OK": 30},
        confidence=0.9,
        rationale="overconfident",
    )
    status = refute_live(
        prediction=pred,
        actuals_so_far={"OK": 5, "CE": 13},
        remaining_cases=2,
        prior_iter_counts={"OK": 0},
    )
    obs.prediction_refuted_live(iteration=4, status=status)

    events = _events_on(debug_file, CHANNEL_PREDICTION_REFUTED_LIVE)
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == CHANNEL_PREDICTION_REFUTED_LIVE
    assert ev["data"]["iteration"] == 4
    assert ev["data"]["is_refuted"] is True
    assert ev["data"]["remaining_cases"] == 2
    assert ev["data"]["session_id"] == obs.session_id
    assert ev["data"]["confidence_at_refute"] == pytest.approx(0.9)
    _validate_with_schema(ev, REFUTED_LIVE_SCHEMA)


def test_emit_prediction_refuted_live_null_confidence_when_not_refuted(
    debug_file: Path,
) -> None:
    """When not refuted, confidence_at_refute serialises as JSON null."""
    obs = AutobenchObservability(debug_file=debug_file)
    pred = Prediction(
        predicted_score_delta=0.0,
        predicted_verdict_class_changes={},
        confidence=0.5,
        rationale="",
    )
    status = refute_live(
        prediction=pred,
        actuals_so_far={},
        remaining_cases=0,
        prior_iter_counts={},
    )
    obs.prediction_refuted_live(iteration=1, status=status)

    events = _events_on(debug_file, CHANNEL_PREDICTION_REFUTED_LIVE)
    assert len(events) == 1
    ev = events[0]
    assert ev["data"]["is_refuted"] is False
    assert ev["data"]["confidence_at_refute"] is None
    _validate_with_schema(ev, REFUTED_LIVE_SCHEMA)


# --------------------------------------------------------------------------- #
# End-to-end integration: 3-iter mock loop, refute fires once during iter 1
# --------------------------------------------------------------------------- #

@dataclass
class _MockBenchmarkResult:
    aggregate_score: float = 0.0
    verdict_counts: dict[str, int] = field(default_factory=dict)
    case_results: list = field(default_factory=list)
    total_latency_ms: float = 0.0

    def pass_rate(self) -> float:
        return 0.0


@dataclass
class _MockHarnessResult:
    """Stand-in for HarnessResult; only needs a ``.verdict`` attribute."""

    verdict: Any = None
    latency_ms: float = 0.0


def test_e2e_live_refute_fires_once_during_iter1(debug_file: Path) -> None:
    """Iter 0 prediction is wildly wrong; live refute should fire ONCE in iter 1.

    Setup:
      * Iter 0 (prev): OK=5, TLE=5, score=0.5
      * Iter 0 emits Prediction(OK: +30, confidence=0.9) — only 10 cases per
        iter so this can never be true.
      * Iter 1 runs 10 cases; even before any complete, predicted +30 is
        unachievable (max OK reachable = 0 so_far + 10 remaining - 5 prior =
        +5). After the first case lands the refutation should fire.
      * Refutation must fire exactly once across all of iter 1.
    """
    from autobench.core import HarnessConfig, Verdict
    from autobench.rsi_loop import ImprovementDelta, SelfImprovingHarness

    # Build a mock evaluator whose ``_run_case`` returns a fixed verdict per
    # call (so the rsi_loop's monkey-patch wrapper exercises real interception).
    class _MockEvaluator:
        def __init__(self) -> None:
            self._iter = 0
            # Map iter idx to (verdict_counts, aggregate_score) used for the
            # eventual BenchmarkResult emission.
            self._iter_results = [
                # iter 0: 5 OK, 5 TLE
                ({"OK": 5, "TLE": 5}, 0.5),
                # iter 1: 6 OK, 4 TLE — far short of predicted +30 OK
                ({"OK": 6, "TLE": 4}, 0.55),
                # iter 2: plateau
                ({"OK": 6, "TLE": 4}, 0.555),
            ]
            self._case_idx = 0
            self._current_iter_verdicts: list[str] = []

        def _build_verdict_stream(self, counts: dict[str, int]) -> list[str]:
            out: list[str] = []
            for k, v in counts.items():
                out.extend([k] * v)
            return out

        def _run_case(self, harness, case, obs=None, iteration=0):
            verdict_str = self._current_iter_verdicts[self._case_idx]
            self._case_idx += 1
            return _MockHarnessResult(
                verdict=Verdict(verdict_str), latency_ms=1.0,
            )

        def run(self, harness, cases, obs=None):
            counts, score = self._iter_results[
                min(self._iter, len(self._iter_results) - 1)
            ]
            self._current_iter_verdicts = self._build_verdict_stream(counts)
            self._case_idx = 0
            # Drive the per-case loop so the live-refute wrapper sees each one.
            for case in cases:
                self._run_case(harness, case, obs=obs, iteration=self._iter)
            self._iter += 1
            return _MockBenchmarkResult(
                aggregate_score=score,
                verdict_counts=counts,
                case_results=[None] * sum(counts.values()),
            )

    iter_counter = {"n": 0}

    def _improver(h, r, iteration=0):
        delta = ImprovementDelta(improvement_summary=f"iter-{iter_counter['n']}")
        if iter_counter["n"] == 0:
            # Feasible at emission (4 <= max_achievable 5) but live-refuted at
            # case 8 of iter 1 when only 6 OK arrive and remaining headroom drops
            # below +4. Using {"OK": 30} would be caught by the feasibility clip
            # and never registered as a pending prediction.
            delta.prediction = Prediction(
                predicted_score_delta=0.1,
                predicted_verdict_class_changes={"OK": 4, "TLE": -4},
                confidence=0.95,
                rationale="expect 4 more OK",
            )
        iter_counter["n"] += 1
        return h, delta

    obs = AutobenchObservability(debug_file=debug_file)
    evaluator = _MockEvaluator()
    sih = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=evaluator,  # type: ignore[arg-type]
        max_iterations=3,
        improvement_threshold=0.001,
        default_improver="rule_based",
        obs=obs,
    )
    # 10 benchmark cases per iteration.
    sih.improve(benchmark_cases=[None] * 10, improver_fn=_improver)

    refuted_live_events = _events_on(debug_file, CHANNEL_PREDICTION_REFUTED_LIVE)
    assert len(refuted_live_events) == 1, (
        f"expected exactly 1 refuted_live event, got {len(refuted_live_events)}: "
        f"{refuted_live_events}"
    )
    ev = refuted_live_events[0]
    # iter 1 is the iteration whose actuals refuted the iter-0 prediction.
    assert ev["data"]["iteration"] == 1
    assert ev["data"]["is_refuted"] is True
    assert ev["data"]["confidence_at_refute"] == pytest.approx(0.95)
    assert "OK" in ev["data"]["refutation_reason"]
    assert ev["data"]["session_id"] == obs.session_id
    _validate_with_schema(ev, REFUTED_LIVE_SCHEMA)
