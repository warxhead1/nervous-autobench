"""Tests for Bitloops-style guidance fact lifecycle management in AHE predictions.

Covers:
    * normalize_text — whitespace collapse, lowercase, trim
    * prediction_fingerprint — stable SHA-256 across functionally identical inputs
    * prediction_source_scope_key — ahe:{session}:{problem}:{iteration} format
    * compact_predictions — deduplication by fingerprint, highest-confidence retained
    * invalidate_prior_predictions — deactivation by scope
    * Prediction.active / parent_prediction_id / fact_fingerprint fields
    * PredictionVerification.lifecycle_status field
    * End-to-end: re-distillation for same scope invalidates prior predictions
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autobench.ahe import (
    Prediction,
    PredictionVerification,
    compact_predictions,
    invalidate_prior_predictions,
    normalize_text,
    prediction_fingerprint,
    prediction_source_scope_key,
    PlannedCompaction,
)
from autobench.invalidation import ahe_scope_key
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_PREDICTION,
)


# --------------------------------------------------------------------------- #
# normalize_text
# --------------------------------------------------------------------------- #

def test_normalize_text_lowercase() -> None:
    assert normalize_text("Hello WORLD") == "hello world"


def test_normalize_text_whitespace_collapse() -> None:
    assert normalize_text("hello   world\n\ttab") == "hello world tab"


def test_normalize_text_trim() -> None:
    assert normalize_text("  hello  ") == "hello"
    assert normalize_text("\n\thello\t\n") == "hello"


def test_normalize_text_empty() -> None:
    assert normalize_text("") == ""


# --------------------------------------------------------------------------- #
# prediction_fingerprint
# --------------------------------------------------------------------------- #

def test_prediction_fingerprint_stable_sha256() -> None:
    """Identical predictions produce the same SHA-256 hexdigest."""
    p1 = Prediction(
        predicted_score_delta=0.05,
        rationale="tighten time budget should recover 3 TLE cases",
        confidence=0.8,
    )
    p2 = Prediction(
        predicted_score_delta=0.05,
        rationale="tighten time budget should recover 3 TLE cases",
        confidence=0.8,
    )
    assert prediction_fingerprint(p1) == prediction_fingerprint(p2)


def test_prediction_fingerprint_whitespace_normalized() -> None:
    """Functionally identical text with different whitespace hashes the same."""
    p1 = Prediction(
        predicted_score_delta=0.05,
        rationale="tighten  time budget",  # double space
        confidence=0.9,
    )
    p2 = Prediction(
        predicted_score_delta=0.05,
        rationale="tighten time budget",  # single space
        confidence=0.9,
    )
    assert prediction_fingerprint(p1) == prediction_fingerprint(p2)


def test_prediction_fingerprint_different_delta_different_hash() -> None:
    """Different predicted_score_delta produces a different fingerprint."""
    p1 = Prediction(predicted_score_delta=0.05, rationale="same rationale", confidence=0.8)
    p2 = Prediction(predicted_score_delta=0.06, rationale="same rationale", confidence=0.8)
    assert prediction_fingerprint(p1) != prediction_fingerprint(p2)


def test_prediction_fingerprint_is_hex_sha256() -> None:
    fp = prediction_fingerprint(Prediction(predicted_score_delta=0.1, rationale="test"))
    assert len(fp) == 64  # SHA-256 hex is 64 chars
    assert re.fullmatch(r"[0-9a-f]{64}", fp) is not None


def test_prediction_fingerprint_uses_fact_fingerprint_when_present() -> None:
    """When fact_fingerprint is pre-set on the Prediction, it is used directly."""
    p = Prediction(
        predicted_score_delta=0.05,
        rationale="hello world",
        fact_fingerprint="abcd1234deadbeef",
    )
    assert prediction_fingerprint(p) == "abcd1234deadbeef"


# --------------------------------------------------------------------------- #
# prediction_source_scope_key
# --------------------------------------------------------------------------- #

def test_prediction_source_scope_key_format() -> None:
    key = prediction_source_scope_key("sess", "prob", 3)
    assert key == "ahe:sess:prob:3"


def test_ahe_scope_key_matches() -> None:
    """prediction_source_scope_key and ahe_scope_key produce identical output."""
    key_fn = prediction_source_scope_key("abc", "xyz", 7)
    key_imported = ahe_scope_key("abc", "xyz", 7)
    assert key_fn == key_imported == "ahe:abc:xyz:7"


# --------------------------------------------------------------------------- #
# compact_predictions
# --------------------------------------------------------------------------- #

def _make_pred(
    prediction_id: str,
    score_delta: float,
    rationale: str,
    confidence: float,
    active: bool = True,
    fingerprint: str | None = None,
) -> Prediction:
    return Prediction(
        prediction_id=prediction_id,
        predicted_score_delta=score_delta,
        rationale=rationale,
        confidence=confidence,
        active=active,
        fact_fingerprint=fingerprint,
    )


def test_compact_predictions_deduplicates_by_fingerprint() -> None:
    p1 = _make_pred("p1", 0.05, "recover TLE", 0.8)
    p2 = _make_pred("p2", 0.05, "recover TLE", 0.9)  # identical fingerprint, higher conf
    p3 = _make_pred("p3", 0.05, "recover TLE", 0.7)  # identical fingerprint, lower conf

    result = compact_predictions([p1, p2, p3])

    # p2 has highest confidence — it should be retained
    assert "p2" in result.retained
    assert "p1" in result.superseded
    assert "p3" in result.superseded
    assert len(result.retained) == 1
    assert len(result.superseded) == 2


def test_compact_predictions_marks_duplicates_inactive() -> None:
    p1 = _make_pred("p1", 0.05, "recover TLE", 0.8)
    p2 = _make_pred("p2", 0.05, "recover TLE", 0.9)

    compact_predictions([p1, p2])

    assert p1.active is False
    assert p1.lifecycle_status == "duplicate"
    assert p2.active is True


def test_compact_predictions_preserves_different_fingerprints() -> None:
    p1 = _make_pred("p1", 0.05, "recover TLE", 0.8)
    p2 = _make_pred("p2", 0.06, "different rationale", 0.7)

    result = compact_predictions([p1, p2])

    assert set(result.retained) == {"p1", "p2"}
    assert result.superseded == []


def test_compact_predictions_empty_list() -> None:
    result = compact_predictions([])
    assert result.retained == []
    assert result.superseded == []


def test_compact_predictions_single_prediction() -> None:
    p = _make_pred("p1", 0.05, "single", 0.8)
    result = compact_predictions([p])
    assert result.retained == ["p1"]
    assert result.superseded == []


def test_compact_predictions_uses_fact_fingerprint_when_set() -> None:
    """fact_fingerprint on the prediction short-circuits re-computation."""
    p1 = Prediction(prediction_id="p1", predicted_score_delta=0.05, rationale="tx",
                   confidence=0.8, fact_fingerprint="aaa")
    p2 = Prediction(prediction_id="p2", predicted_score_delta=0.05, rationale="ty",
                   confidence=0.9, fact_fingerprint="aaa")  # same fingerprint, diff text

    result = compact_predictions([p1, p2])

    assert "p2" in result.retained  # higher confidence wins
    assert "p1" in result.superseded
    assert p1.active is False
    assert p1.lifecycle_status == "duplicate"


# --------------------------------------------------------------------------- #
# invalidate_prior_predictions
# --------------------------------------------------------------------------- #

def test_invalidate_prior_predictions_deactivates_matching_scope(tmp_path: Path) -> None:
    store = {
        "p1": Prediction(prediction_id="p1", predicted_score_delta=0.05,
                          rationale="t1", confidence=0.8, active=True),
        "p2": Prediction(prediction_id="p2", predicted_score_delta=0.06,
                          rationale="t2", confidence=0.8, active=True),
    }
    # Simulate adding scope key to predictions
    for p in store.values():
        p.source_scope_key = "ahe:session:problem:3"

    count = invalidate_prior_predictions("session", "ahe:session:problem:3", store=store)

    assert count == 2
    assert all(not p.active for p in store.values())
    assert all(p.lifecycle_status == "superseded" for p in store.values())


def test_invalidate_prior_predictions_ignores_different_scope() -> None:
    p1 = Prediction(prediction_id="p1", predicted_score_delta=0.05,
                     rationale="t1", confidence=0.8, active=True)
    p1.source_scope_key = "ahe:session:problem:3"
    store = {"p1": p1}

    count = invalidate_prior_predictions("session", "ahe:session:problem:4", store=store)

    assert count == 0
    assert p1.active is True


def test_invalidate_prior_predictions_ignores_already_inactive() -> None:
    p1 = Prediction(prediction_id="p1", predicted_score_delta=0.05,
                     rationale="t1", confidence=0.8, active=False)
    p1.source_scope_key = "ahe:session:problem:3"
    store = {"p1": p1}

    count = invalidate_prior_predictions("session", "ahe:session:problem:3", store=store)

    assert count == 0


def test_invalidate_prior_predictions_empty_store() -> None:
    count = invalidate_prior_predictions("session", "ahe:session:problem:3", store={})
    assert count == 0


# --------------------------------------------------------------------------- #
# Prediction fields — defaults and wiring
# --------------------------------------------------------------------------- #

def test_prediction_default_active_is_true() -> None:
    p = Prediction(predicted_score_delta=0.05, rationale="test", confidence=0.5)
    assert p.active is True
    assert p.parent_prediction_id is None
    assert p.fact_fingerprint is None
    assert p.prediction_id == ""


def test_prediction_lifecycle_status_default() -> None:
    pv = PredictionVerification(
        predicted=Prediction(predicted_score_delta=0.05, rationale="t", confidence=0.5),
        actual_score_delta=0.04,
        actual_verdict_class_changes={},
        score_delta_error=0.01,
        verdict_match_ratio=0.8,
        outcome_label="confirmed",
        confidence_calibration=0.1,
    )
    assert pv.lifecycle_status == "active"


def test_prediction_lifecycle_status_superseded() -> None:
    pv = PredictionVerification(
        predicted=Prediction(predicted_score_delta=0.05, rationale="t", confidence=0.5),
        actual_score_delta=0.04,
        actual_verdict_class_changes={},
        score_delta_error=0.01,
        verdict_match_ratio=0.8,
        outcome_label="confirmed",
        confidence_calibration=0.1,
        lifecycle_status="superseded",
    )
    assert pv.lifecycle_status == "superseded"


# --------------------------------------------------------------------------- #
# Observability emission with new fields
# --------------------------------------------------------------------------- #

def test_improver_prediction_emits_lifecycle_fields(tmp_path: Path) -> None:
    """AutobenchObservability.improver_prediction includes active/fingerprint/prediction_id."""
    import os
    old_path = os.environ.get("PATH")
    os.environ["PATH"] = str(tmp_path / "empty-bin")

    try:
        debug_file = tmp_path / "debug.jsonl"
        obs = AutobenchObservability(debug_file=debug_file)

        pred = Prediction(
            prediction_id="pred-01",
            predicted_score_delta=0.05,
            rationale="tighten time budget",
            confidence=0.8,
            active=True,
            fact_fingerprint="abcd1234deadbeef",
            parent_prediction_id=None,
        )
        pred.source_scope_key = "ahe:session:problem:3"

        obs.improver_prediction(iteration=0, prediction=pred, model="test-model")

        lines = debug_file.read_text().strip().splitlines()
        event = json.loads(lines[0])

        assert event["type"] == CHANNEL_PREDICTION
        data = event["data"]
        assert data["active"] is True
        assert data["fact_fingerprint"] == "abcd1234deadbeef"
        assert data["prediction_id"] == "pred-01"
        assert data["parent_prediction_id"] is None
        assert data["source_scope_key"] == "ahe:session:problem:3"
    finally:
        if old_path is not None:
            os.environ["PATH"] = old_path
        else:
            os.environ.pop("PATH", None)


def test_improver_prediction_deactivated_flags_inactive(tmp_path: Path) -> None:
    """Predictions marked inactive are emitted with active=False."""
    import os
    old_path = os.environ.get("PATH")
    os.environ["PATH"] = str(tmp_path / "empty-bin")

    try:
        debug_file = tmp_path / "debug.jsonl"
        obs = AutobenchObservability(debug_file=debug_file)

        pred = Prediction(
            prediction_id="pred-02",
            predicted_score_delta=0.05,
            rationale="superseded",
            confidence=0.6,
            active=False,
            parent_prediction_id="pred-01",
        )

        obs.improver_prediction(iteration=1, prediction=pred, model="test-model")

        lines = debug_file.read_text().strip().splitlines()
        event = json.loads(lines[0])

        assert event["data"]["active"] is False
        assert event["data"]["parent_prediction_id"] == "pred-01"
    finally:
        if old_path is not None:
            os.environ["PATH"] = old_path
        else:
            os.environ.pop("PATH", None)


# --------------------------------------------------------------------------- #
# End-to-end: re-distillation invalidates prior predictions
# --------------------------------------------------------------------------- #

def test_redistillation_invalidates_prior_predictions(tmp_path: Path) -> None:
    """When a new prediction is emitted for the same scope, prior ones are deactivated."""
    import os
    old_path = os.environ.get("PATH")
    os.environ["PATH"] = str(tmp_path / "empty-bin")

    try:
        debug_file = tmp_path / "debug.jsonl"
        obs = AutobenchObservability(debug_file=debug_file)

        # First prediction for scope ahe:session:problem:3
        prior = Prediction(
            prediction_id="prior-01",
            predicted_score_delta=0.05,
            rationale="recover TLE",
            confidence=0.8,
            active=True,
        )
        prior.source_scope_key = "ahe:session:problem:3"
        prior.fact_fingerprint = prediction_fingerprint(prior)

        # Emit first prediction
        obs.improver_prediction(iteration=0, prediction=prior, model="test-model")

        # Simulate re-distillation: invalidate prior predictions for same scope
        store = {"prior-01": prior}
        invalidated = invalidate_prior_predictions(
            "session", "ahe:session:problem:3", store=store
        )
        assert invalidated == 1
        assert prior.active is False
        assert prior.lifecycle_status == "superseded"

        # New prediction for the same scope
        new_pred = Prediction(
            prediction_id="new-01",
            predicted_score_delta=0.07,
            rationale="recover TLE",
            confidence=0.85,
            active=True,
            parent_prediction_id="prior-01",
        )
        new_pred.source_scope_key = "ahe:session:problem:3"
        new_pred.fact_fingerprint = prediction_fingerprint(new_pred)

        obs.improver_prediction(iteration=1, prediction=new_pred, model="test-model")

        lines = debug_file.read_text().strip().splitlines()
        events = [json.loads(line) for line in lines]

        pred_events = [e for e in events if e["type"] == CHANNEL_PREDICTION]
        assert len(pred_events) == 2

        # Second emission should have parent_prediction_id set
        second = pred_events[1]["data"]
        assert second["prediction_id"] == "new-01"
        assert second["parent_prediction_id"] == "prior-01"
    finally:
        if old_path is not None:
            os.environ["PATH"] = old_path
        else:
            os.environ.pop("PATH", None)
