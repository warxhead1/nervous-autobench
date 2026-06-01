"""Tests for clip_prediction_to_feasible — nervous-bus-8d1d.

Motivating incident: cycle session 01KRSDHKD7M0JQ44AKFY8PR7FN iter 1's
improver predicted ``{OK: +8, CE: -12, WA: 0}`` with prior verdict counts
``{OK: 15, CE: 4, WA: 1}`` and 20 cases. CE cannot decrease by 12 when
only 4 CE exist — guaranteed-refute prediction polluting the calibration
ledger. The clip lifts the verifier's feasibility math (which already
existed in ``refute_live``) to PREDICTION TIME so impossible deltas are
bounded before they reach the bus.
"""

from __future__ import annotations

from autobench.audit.ahe import Prediction, clip_prediction_to_feasible


def test_no_clip_when_already_feasible():
    pred = Prediction(
        predicted_score_delta=0.05,
        predicted_verdict_class_changes={"OK": 2, "CE": -1, "WA": -1},
        confidence=0.7,
        rationale="reasonable",
    )
    prior = {"OK": 18, "CE": 1, "WA": 1}
    clipped, reasons = clip_prediction_to_feasible(pred, prior, num_cases=20)
    assert reasons == []
    assert clipped.predicted_verdict_class_changes == {"OK": 2, "CE": -1, "WA": -1}
    # Identity-preserving on the other fields.
    assert clipped.predicted_score_delta == 0.05
    assert clipped.confidence == 0.7
    assert clipped.rationale == "reasonable"


def test_cycle_5_exact_repro():
    """The cycle 01KRSDHKD7M0JQ44AKFY8PR7FN iter 1 prediction.

    Improver proposed CE:-12 against prior {CE:4}. Max negative delta is
    -4 (every remaining case lands elsewhere → CE drops to 0). Predicted
    OK:+8 against prior {OK:15}, num_cases=20 → max positive is +5
    (everything could be OK).
    """
    pred = Prediction(
        predicted_score_delta=0.08,
        predicted_verdict_class_changes={"OK": 8, "CE": -12, "WA": 0},
        confidence=0.85,
        rationale="reduce CE via stronger directive",
    )
    prior = {"OK": 15, "CE": 4, "WA": 1}
    clipped, reasons = clip_prediction_to_feasible(pred, prior, num_cases=20)
    assert clipped.predicted_verdict_class_changes["OK"] == 5
    assert clipped.predicted_verdict_class_changes["CE"] == -4
    assert clipped.predicted_verdict_class_changes["WA"] == 0
    assert len(reasons) == 2  # OK and CE both clipped, WA was 0 so untouched
    # Reasons should be human-readable.
    assert any("CE" in r and "-4" in r for r in reasons)
    assert any("OK" in r and "+5" in r for r in reasons)


def test_clip_negative_at_zero_prior():
    """Cannot predict a negative delta for a verdict that wasn't in the prior."""
    pred = Prediction(
        predicted_verdict_class_changes={"TLE": -3},
        confidence=0.6,
        rationale="reduce TLE",
    )
    # Prior had zero TLE — can't decrease below zero.
    clipped, reasons = clip_prediction_to_feasible(pred, {"OK": 20}, num_cases=20)
    assert clipped.predicted_verdict_class_changes["TLE"] == 0
    assert len(reasons) == 1


def test_clip_positive_at_num_cases():
    """Cannot predict an increase larger than (num_cases - prior)."""
    pred = Prediction(
        predicted_verdict_class_changes={"OK": 100},
        confidence=0.5,
        rationale="all OK",
    )
    clipped, reasons = clip_prediction_to_feasible(pred, {"OK": 5}, num_cases=20)
    assert clipped.predicted_verdict_class_changes["OK"] == 15  # 20 - 5
    assert len(reasons) == 1


def test_empty_prediction_passthrough():
    pred = Prediction()
    clipped, reasons = clip_prediction_to_feasible(pred, {"OK": 10}, num_cases=20)
    assert reasons == []
    assert clipped.predicted_verdict_class_changes == {}


def test_non_integer_delta_dropped():
    """An LLM that emits a string or null in verdict deltas — drop, not crash."""
    pred = Prediction(
        predicted_verdict_class_changes={"OK": "not-a-number", "CE": -2},
        confidence=0.5,
    )
    clipped, reasons = clip_prediction_to_feasible(
        pred, {"OK": 15, "CE": 4, "WA": 1}, num_cases=20,
    )
    # OK was unparseable → dropped from the clipped result.
    assert "OK" not in clipped.predicted_verdict_class_changes
    assert clipped.predicted_verdict_class_changes["CE"] == -2  # feasible
    assert any("OK" in r and "dropped" in r for r in reasons)


def test_zero_num_cases_clips_everything_to_zero():
    """Degenerate but defensible — no cases → no headroom for any delta."""
    pred = Prediction(
        predicted_verdict_class_changes={"OK": 5, "CE": -3},
        confidence=0.5,
    )
    clipped, reasons = clip_prediction_to_feasible(pred, {"OK": 0, "CE": 3}, num_cases=0)
    # OK: max headroom = 0 - 0 = 0; clip +5 → 0
    # CE: min headroom = -3; predicted -3 is feasible, NOT clipped.
    assert clipped.predicted_verdict_class_changes["OK"] == 0
    assert clipped.predicted_verdict_class_changes["CE"] == -3


def test_predicted_score_delta_and_confidence_preserved():
    """Score delta and confidence are NOT touched by the clip."""
    pred = Prediction(
        predicted_score_delta=0.123,
        predicted_verdict_class_changes={"OK": 50},  # will clip
        confidence=0.999,
        rationale="ambitious",
    )
    clipped, reasons = clip_prediction_to_feasible(pred, {"OK": 5}, num_cases=20)
    assert clipped.predicted_score_delta == 0.123
    assert clipped.confidence == 0.999
    assert clipped.rationale == "ambitious"
