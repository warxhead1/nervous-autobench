"""Tests for the AHE (Agent Harness Evolution) prediction contract.

Covers:
    * parse_prediction_from_llm_response — well-formed, missing, malformed, partial
    * verify_prediction — confirmed / partial / refuted classification
    * confidence_calibration semantics
    * should_emit_warning for confidently-wrong predictions
    * End-to-end: a mock 3-iteration RSI run emits prediction.v1 at iter 0 and
      prediction.verified.v1 at iter 1 with matching session_id
    * Schema validation of every emitted event
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.ahe import (
    Prediction,
    PredictionVerification,
    parse_prediction_from_llm_response,
    verify_prediction,
    should_emit_warning,
)
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_PREDICTION,
    CHANNEL_PREDICTION_VERIFIED,
)



SCHEMA_FOR_CHANNEL = {
    CHANNEL_PREDICTION: SCHEMA_DIR / "autobench.improver.prediction.v1.json",
    CHANNEL_PREDICTION_VERIFIED: SCHEMA_DIR / "autobench.improver.prediction.verified.v1.json",
}


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
    """Validate an event against its JSON schema; assert on failure."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(event)


@dataclass
class _MockBenchmarkResult:
    """Stand-in for evaluator.BenchmarkResult with just the fields verify_prediction touches."""

    aggregate_score: float = 0.0
    verdict_counts: dict[str, int] = field(default_factory=dict)
    case_results: list = field(default_factory=list)
    total_latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return 0.0


# --------------------------------------------------------------------------- #
# parse_prediction_from_llm_response
# --------------------------------------------------------------------------- #

def test_parse_prediction_well_formed() -> None:
    raw = json.dumps({
        "rationale": "TLE pressure",
        "prediction": {
            "predicted_score_delta": 0.05,
            "predicted_verdict_class_changes": {"OK": 3, "TLE": -3},
            "confidence": 0.8,
            "rationale": "tighten time budget should recover TLE",
        },
    })
    p = parse_prediction_from_llm_response(raw)
    assert p is not None
    assert p.predicted_score_delta == pytest.approx(0.05)
    assert p.predicted_verdict_class_changes == {"OK": 3, "TLE": -3}
    assert p.confidence == pytest.approx(0.8)
    assert "tighten" in p.rationale


def test_parse_prediction_missing_prediction_returns_none() -> None:
    raw = json.dumps({"rationale": "no prediction here"})
    assert parse_prediction_from_llm_response(raw) is None


def test_parse_prediction_malformed_json_returns_none() -> None:
    assert parse_prediction_from_llm_response("not even close to JSON") is None
    # truncated JSON
    assert parse_prediction_from_llm_response('{"prediction": {"predicted_score_delta": 0.5,') is None


def test_parse_prediction_partial_fields_uses_defaults() -> None:
    raw = json.dumps({"prediction": {"predicted_score_delta": 0.1}})
    p = parse_prediction_from_llm_response(raw)
    assert p is not None
    assert p.predicted_score_delta == pytest.approx(0.1)
    assert p.confidence == 0.0  # default
    assert p.predicted_verdict_class_changes == {}
    assert p.rationale == ""


def test_parse_prediction_handles_code_fence() -> None:
    raw = '```json\n{"prediction": {"predicted_score_delta": 0.2, "confidence": 0.5}}\n```'
    p = parse_prediction_from_llm_response(raw)
    assert p is not None
    assert p.predicted_score_delta == pytest.approx(0.2)
    assert p.confidence == pytest.approx(0.5)


def test_parse_prediction_confidence_clamped() -> None:
    """Out-of-range confidence values are clamped to [0,1]."""
    raw = json.dumps({"prediction": {"confidence": 2.5}})
    p = parse_prediction_from_llm_response(raw)
    assert p is not None
    assert p.confidence == 1.0
    raw_neg = json.dumps({"prediction": {"confidence": -0.5}})
    p2 = parse_prediction_from_llm_response(raw_neg)
    assert p2 is not None
    assert p2.confidence == 0.0


def test_parse_prediction_empty_input() -> None:
    assert parse_prediction_from_llm_response("") is None


# --------------------------------------------------------------------------- #
# verify_prediction
# --------------------------------------------------------------------------- #

def test_verify_prediction_confirmed() -> None:
    """High-confidence prediction that matches actuals → confirmed, low calibration."""
    pred = Prediction(
        predicted_score_delta=0.05,
        predicted_verdict_class_changes={"OK": 3, "TLE": -3},
        confidence=0.9,
        rationale="should recover TLE",
    )
    prev = _MockBenchmarkResult(
        aggregate_score=0.50,
        verdict_counts={"OK": 5, "TLE": 5},
    )
    curr = _MockBenchmarkResult(
        aggregate_score=0.55,  # +0.05 exactly
        verdict_counts={"OK": 8, "TLE": 2},  # OK +3, TLE -3 exactly
    )
    v = verify_prediction(pred, prev, curr)
    assert v.outcome_label == "confirmed"
    assert v.verdict_match_ratio == pytest.approx(1.0)
    assert v.score_delta_error == pytest.approx(0.0, abs=1e-9)
    # Observed accuracy = 1.0 → calibration = |0.9 - 1.0| = 0.1
    assert v.confidence_calibration == pytest.approx(0.1, abs=1e-9)


def test_verify_prediction_refuted() -> None:
    """High-confidence prediction that's wrong → refuted, large calibration."""
    pred = Prediction(
        predicted_score_delta=0.10,
        predicted_verdict_class_changes={"OK": 4, "TLE": -4},
        confidence=0.9,
        rationale="should fix all TLE",
    )
    prev = _MockBenchmarkResult(
        aggregate_score=0.50,
        verdict_counts={"OK": 5, "TLE": 5},
    )
    curr = _MockBenchmarkResult(
        aggregate_score=0.30,  # -0.20, opposite direction
        verdict_counts={"OK": 2, "TLE": 8},  # OK -3, TLE +3 — wrong direction
    )
    v = verify_prediction(pred, prev, curr)
    assert v.outcome_label == "refuted"
    assert v.verdict_match_ratio == pytest.approx(0.0)
    # observed accuracy 0 → calibration = 0.9
    assert v.confidence_calibration == pytest.approx(0.9, abs=1e-9)
    assert should_emit_warning(v) is True


def test_verify_prediction_partial() -> None:
    """Half the verdict directions match but score-delta is off."""
    pred = Prediction(
        predicted_score_delta=0.30,  # very wrong magnitude
        predicted_verdict_class_changes={"OK": 4, "TLE": -4},
        confidence=0.6,
        rationale="optimistic",
    )
    prev = _MockBenchmarkResult(
        aggregate_score=0.50,
        verdict_counts={"OK": 5, "TLE": 5},
    )
    curr = _MockBenchmarkResult(
        aggregate_score=0.52,  # tiny improvement (score_delta_error large)
        verdict_counts={"OK": 7, "TLE": 5},  # OK went up (match), TLE same (no match)
    )
    v = verify_prediction(pred, prev, curr)
    # OK direction matches (+), TLE direction doesn't (predicted -, actual 0)
    assert v.verdict_match_ratio == pytest.approx(0.5)
    # 0.5 ≥ 0.5 threshold → partial (not confirmed since score_delta_error > tol)
    assert v.outcome_label == "partial"


def test_confidence_calibration_high_when_overconfident_and_refuted() -> None:
    """A confident-but-wrong prediction yields large calibration error."""
    pred = Prediction(
        predicted_score_delta=0.5,
        predicted_verdict_class_changes={"OK": 5, "TLE": -5},
        confidence=1.0,
        rationale="dead certain",
    )
    prev = _MockBenchmarkResult(aggregate_score=0.5, verdict_counts={"OK": 0, "TLE": 5})
    curr = _MockBenchmarkResult(aggregate_score=0.2, verdict_counts={"OK": 0, "TLE": 5})
    v = verify_prediction(pred, prev, curr)
    assert v.outcome_label == "refuted"
    assert v.confidence_calibration == pytest.approx(1.0, abs=1e-9)
    assert should_emit_warning(v) is True


def test_should_emit_warning_low_confidence_refuted_no_warning() -> None:
    """A low-confidence wrong prediction should NOT trigger the warning."""
    pred = Prediction(
        predicted_score_delta=0.5,
        predicted_verdict_class_changes={"OK": 5},
        confidence=0.2,  # explicit low confidence
        rationale="wild guess",
    )
    prev = _MockBenchmarkResult(aggregate_score=0.5, verdict_counts={"OK": 0})
    curr = _MockBenchmarkResult(aggregate_score=0.2, verdict_counts={"OK": 0})
    v = verify_prediction(pred, prev, curr)
    assert v.outcome_label == "refuted"
    assert should_emit_warning(v) is False


def test_verify_prediction_empty_changes_score_drives_outcome() -> None:
    """When no verdict changes are predicted, score_delta_error alone decides."""
    pred = Prediction(predicted_score_delta=0.05, confidence=0.7)
    prev = _MockBenchmarkResult(aggregate_score=0.5, verdict_counts={"OK": 5})
    curr = _MockBenchmarkResult(aggregate_score=0.55, verdict_counts={"OK": 5})
    v = verify_prediction(pred, prev, curr)
    # verdict_match_ratio defaults to 1.0 with no predictions, score_delta_error=0
    assert v.outcome_label == "confirmed"


# --------------------------------------------------------------------------- #
# Observability emission
# --------------------------------------------------------------------------- #

def test_emit_prediction_event_envelope_and_schema(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    pred = Prediction(
        predicted_score_delta=0.07,
        predicted_verdict_class_changes={"OK": 2, "WA": -2},
        confidence=0.65,
        rationale="hierarchical context should reduce WA",
    )
    obs.improver_prediction(iteration=3, prediction=pred, model="MiniMax-M2.7")
    events = _events_on(debug_file, CHANNEL_PREDICTION)
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == CHANNEL_PREDICTION
    assert ev["data"]["iteration"] == 3
    assert ev["data"]["model"] == "MiniMax-M2.7"
    assert ev["data"]["confidence"] == pytest.approx(0.65)
    assert ev["data"]["session_id"] == obs.session_id
    _validate_with_schema(ev, SCHEMA_FOR_CHANNEL[CHANNEL_PREDICTION])


def test_emit_prediction_verified_event_envelope_and_schema(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    pred = Prediction(
        predicted_score_delta=0.05,
        predicted_verdict_class_changes={"OK": 3, "TLE": -3},
        confidence=0.9,
        rationale="tighten time budget",
    )
    prev = _MockBenchmarkResult(aggregate_score=0.5, verdict_counts={"OK": 5, "TLE": 5})
    curr = _MockBenchmarkResult(aggregate_score=0.55, verdict_counts={"OK": 8, "TLE": 2})
    verification = verify_prediction(pred, prev, curr)
    obs.improver_prediction_verified(iteration=4, verification=verification)
    events = _events_on(debug_file, CHANNEL_PREDICTION_VERIFIED)
    assert len(events) == 1
    ev = events[0]
    assert ev["data"]["iteration"] == 4
    assert ev["data"]["outcome_label"] == "confirmed"
    assert ev["data"]["session_id"] == obs.session_id
    _validate_with_schema(ev, SCHEMA_FOR_CHANNEL[CHANNEL_PREDICTION_VERIFIED])


# --------------------------------------------------------------------------- #
# End-to-end: mock 3-iteration RSI run via SelfImprovingHarness.improve()
# --------------------------------------------------------------------------- #

def _make_rsi_harness_for_e2e(obs: AutobenchObservability):
    """Build a minimal SelfImprovingHarness wired up with mocks for an
    end-to-end emission test. The improver returns a Prediction on iter 0 and
    None thereafter; the evaluator returns BenchmarkResults that confirm the
    iter-0 prediction in iter 1."""
    from autobench.core import HarnessConfig
    from autobench.evaluator import BenchmarkEvaluator
    from autobench.rsi_loop import ImprovementDelta, SelfImprovingHarness

    base_harness = HarnessConfig()

    # Sequence of benchmark results returned by the mock evaluator, one per
    # iteration. We want prev_score=0.5 in iter 0, then 0.55 in iter 1 so the
    # iter-0 prediction (+0.05) is confirmed when verified at iter 1.
    results_sequence = [
        _MockBenchmarkResult(
            aggregate_score=0.50,
            verdict_counts={"OK": 5, "TLE": 5},
            case_results=[None] * 10,
        ),
        _MockBenchmarkResult(
            aggregate_score=0.55,
            verdict_counts={"OK": 8, "TLE": 2},
            case_results=[None] * 10,
        ),
        _MockBenchmarkResult(
            aggregate_score=0.555,  # plateau → loop should exit after this
            verdict_counts={"OK": 8, "TLE": 2},
            case_results=[None] * 10,
        ),
    ]

    class _MockEvaluator:
        def __init__(self):
            self._i = 0

        def run(self, harness, cases, obs=None):
            r = results_sequence[min(self._i, len(results_sequence) - 1)]
            self._i += 1
            return r

    iter_counter = {"n": 0}

    def _improver(h, r, iteration=0):
        delta = ImprovementDelta(improvement_summary=f"iter-{iter_counter['n']}")
        if iter_counter["n"] == 0:
            delta.prediction = Prediction(
                predicted_score_delta=0.05,
                predicted_verdict_class_changes={"OK": 3, "TLE": -3},
                confidence=0.9,
                rationale="tighten time budget",
            )
        iter_counter["n"] += 1
        return h, delta

    sih = SelfImprovingHarness(
        current_harness=base_harness,
        evaluator=_MockEvaluator(),  # type: ignore[arg-type]
        max_iterations=3,
        improvement_threshold=0.001,
        default_improver="rule_based",
        obs=obs,
    )
    return sih, _improver


def test_e2e_prediction_emitted_iter0_verified_iter1(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    sih, improver = _make_rsi_harness_for_e2e(obs)
    sih.improve(benchmark_cases=[], improver_fn=improver)

    pred_events = _events_on(debug_file, CHANNEL_PREDICTION)
    verified_events = _events_on(debug_file, CHANNEL_PREDICTION_VERIFIED)

    assert len(pred_events) == 1, f"expected 1 prediction emit, got {len(pred_events)}"
    assert len(verified_events) == 1, (
        f"expected 1 verified emit, got {len(verified_events)}"
    )

    pred_ev = pred_events[0]
    ver_ev = verified_events[0]
    assert pred_ev["data"]["iteration"] == 0
    assert ver_ev["data"]["iteration"] == 1
    # session_id correlation
    assert pred_ev["data"]["session_id"] == ver_ev["data"]["session_id"] == obs.session_id

    # Outcome should be confirmed given our sequence
    assert ver_ev["data"]["outcome_label"] == "confirmed"

    # Schema validation of every emitted event
    for ev in pred_events:
        _validate_with_schema(ev, SCHEMA_FOR_CHANNEL[CHANNEL_PREDICTION])
    for ev in verified_events:
        _validate_with_schema(ev, SCHEMA_FOR_CHANNEL[CHANNEL_PREDICTION_VERIFIED])
