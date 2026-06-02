"""Tests for the SACS diversity tracker.

Covers:
    * StructuralFingerprint shape stability.
    * Identical deltas → cosine similarity 1.0.
    * Disjoint-field deltas → similarity < 0.5 (and typically 0).
    * DiversityTracker warm-up (< min_bank) returns 0.0 penalty.
    * Threshold gating: low-similarity deltas → penalty 0.
    * Threshold gating: redundant deltas → penalty grows linearly.
    * Determinism: same trajectory → same penalty.
    * Schema validation of emitted events.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from autobench.evaluation.diversity import (
    DiversityTracker,
    FINGERPRINT_DIM,
    StructuralFingerprint,
    _cosine,
    sacs_similarity,
)
from autobench.observability import AutobenchObservability, CHANNEL_DIVERSITY
from autobench.rsi.loop import ImprovementDelta


from tests._paths import SCHEMA_DIR
SCHEMA_PATH = SCHEMA_DIR / "autobench.diversity.v1.json"


# --------------------------------------------------------------------------- #
# Fingerprint shape & content
# --------------------------------------------------------------------------- #


def test_fingerprint_has_23_dims() -> None:
    delta = ImprovementDelta(improvement_summary="test", budget_delta={"max_tokens": 6000})
    fp = StructuralFingerprint.from_delta(delta)
    assert fp.shape == (FINGERPRINT_DIM,)
    assert fp.dtype == np.float64


def test_fingerprint_empty_delta_is_almost_zero() -> None:
    delta = ImprovementDelta()
    fp = StructuralFingerprint.from_delta(delta)
    # The only non-zero entry will be the tanh-normalized summary length;
    # empty summary → tanh(-1.0) ≈ -0.7616. Everything else must be 0.
    structural = fp[:22]
    assert np.allclose(structural, 0.0)


def test_two_identical_deltas_cosine_one() -> None:
    d = ImprovementDelta(
        improvement_summary="reduced max_tokens to compress context",
        budget_delta={"max_tokens": 5500},
    )
    fp1 = StructuralFingerprint.from_delta(d)
    fp2 = StructuralFingerprint.from_delta(d)
    assert _cosine(fp1, fp2) == pytest.approx(1.0, abs=1e-9)


def test_disjoint_fields_low_similarity() -> None:
    """Deltas changing entirely different fields should have low SACS similarity."""
    d_budget = ImprovementDelta(
        improvement_summary="reduce tokens",
        budget_delta={"max_tokens": 5500},
    )
    d_context = ImprovementDelta(
        improvement_summary="switched to hierarchical context",
        context_manager_changed=True,
    )
    fp_a = StructuralFingerprint.from_delta(d_budget)
    fp_b = StructuralFingerprint.from_delta(d_context)
    fields_a = StructuralFingerprint.fields_changed(d_budget)
    fields_b = StructuralFingerprint.fields_changed(d_context)
    # No overlap on changed-field sets → SACS forces similarity to 0.
    sim = sacs_similarity(fp_a, fp_b, fields_a, fields_b)
    assert sim < 0.5
    assert sim == pytest.approx(0.0, abs=1e-9)


def test_fields_changed_covers_all_top_level_fields() -> None:
    d = ImprovementDelta(
        system_prompt_delta="add more examples",
        rollout_protocol_changed=True,
        context_manager_changed=True,
        tool_surface_delta="adjust tool list",
        budget_delta={"max_tokens": 5000, "max_time_seconds": 25},
    )
    changed = StructuralFingerprint.fields_changed(d)
    assert "system_prompt" in changed
    assert "rollout_protocol" in changed
    assert "context_manager" in changed
    assert "tool_surface" in changed
    assert "budget.max_tokens" in changed
    assert "budget.max_time_seconds" in changed


# --------------------------------------------------------------------------- #
# Tracker behaviour
# --------------------------------------------------------------------------- #


def test_tracker_warmup_returns_zero_penalty() -> None:
    tracker = DiversityTracker()
    d = ImprovementDelta(improvement_summary="x", budget_delta={"max_tokens": 5500})
    # min_bank=2 by default; with 0 entries, penalty must be 0.
    assert tracker.penalty_for(d) == 0.0
    tracker.record(d)
    # With 1 entry still below the warm-up bar.
    assert tracker.penalty_for(d) == 0.0


def test_tracker_below_thresholds_no_penalty() -> None:
    """Diverse-enough deltas (similarity below tau_max AND tau_mean) → 0 penalty."""
    tracker = DiversityTracker()
    # Three deltas changing entirely different fields → similarity is 0
    # (disjoint changed-field sets → SACS Jaccard overlap is 0).
    deltas = [
        ImprovementDelta(improvement_summary="a", budget_delta={"max_tokens": 5500}),
        ImprovementDelta(improvement_summary="b", context_manager_changed=True),
        ImprovementDelta(improvement_summary="c", rollout_protocol_changed=True),
    ]
    for d in deltas:
        tracker.record(d)
    candidate = ImprovementDelta(improvement_summary="d", tool_surface_delta="add foo")
    assert tracker.penalty_for(candidate) == 0.0


def test_tracker_above_thresholds_penalty_grows() -> None:
    """When candidate is structurally redundant with memory, penalty fires."""
    tracker = DiversityTracker()
    # Fill memory with many copies of the same delta.
    d_redundant = ImprovementDelta(
        improvement_summary="reduce tokens",
        budget_delta={"max_tokens": 5500},
    )
    for _ in range(5):
        tracker.record(d_redundant)
    # Candidate identical to memory entries → max sim ≈ 1.0, mean sim ≈ 1.0.
    penalty = tracker.penalty_for(d_redundant)
    # penalty = 0.15 * (1.0 - 0.6) + 0.85 * (1.0 - 0.3) = 0.06 + 0.595 = 0.655
    assert penalty == pytest.approx(0.655, abs=1e-3)


def test_tracker_penalty_growth_is_monotonic_in_similarity() -> None:
    tracker = DiversityTracker()
    base = ImprovementDelta(
        improvement_summary="x", budget_delta={"max_tokens": 5500},
    )
    for _ in range(3):
        tracker.record(base)
    p_identical = tracker.penalty_for(base)
    # Candidate touching a disjoint field has Jaccard overlap 0 → SACS = 0.
    p_distinct = tracker.penalty_for(
        ImprovementDelta(improvement_summary="y", rollout_protocol_changed=True)
    )
    assert p_distinct < p_identical


def test_tracker_determinism() -> None:
    """Same trajectory of deltas → same penalty (no hidden randomness)."""
    def run() -> float:
        t = DiversityTracker()
        for i in range(4):
            t.record(ImprovementDelta(
                improvement_summary=f"iter {i}",
                budget_delta={"max_tokens": 6000 - 100 * i},
            ))
        return t.penalty_for(
            ImprovementDelta(improvement_summary="cand", budget_delta={"max_tokens": 5500})
        )
    assert run() == run()


def test_tracker_diversity_score_drops_with_redundancy() -> None:
    t = DiversityTracker()
    base = ImprovementDelta(
        improvement_summary="x", budget_delta={"max_tokens": 5500},
    )
    for _ in range(5):
        t.record(base)
    # All entries identical → mean SACS = 1.0 → diversity_score = 0.0.
    assert t.current_diversity_score() == pytest.approx(0.0, abs=1e-6)


def test_tracker_memory_bounded() -> None:
    t = DiversityTracker(memory_size=3)
    for i in range(10):
        t.record(ImprovementDelta(
            improvement_summary=f"i{i}", budget_delta={"max_tokens": 6000 - i * 10},
        ))
    assert t.snapshot()["memory_size"] == 3


def test_apply_to_utility_subtracts_penalty() -> None:
    t = DiversityTracker()
    d = ImprovementDelta(improvement_summary="x", budget_delta={"max_tokens": 5500})
    for _ in range(3):
        t.record(d)
    raw = 0.85
    adjusted = t.apply_to_utility(raw, d)
    assert adjusted < raw
    assert adjusted == pytest.approx(raw - t.penalty_for(d))


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def test_diversity_event_validates_against_schema(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    import jsonschema

    with open(SCHEMA_PATH) as fh:
        schema = json.load(fh)

    debug_file = tmp_path / "debug.jsonl"
    os.environ["AUTOBENCH_OBS_DISABLE_PIPE"] = "1"
    obs = AutobenchObservability(debug_file=debug_file)

    # Synthesize a fingerprint (23 floats) for one iteration.
    delta = ImprovementDelta(
        improvement_summary="t", budget_delta={"max_tokens": 5500}
    )
    fp = StructuralFingerprint.from_delta(delta).tolist()
    obs.diversity_snapshot(
        iteration=2,
        fingerprint=fp,
        penalty=0.07,
        diversity_score=0.42,
        memory_size=5,
    )

    # Read back the emitted event and validate.
    contents = debug_file.read_text().strip().splitlines()
    diversity_events = [
        json.loads(line) for line in contents
        if json.loads(line).get("type") == CHANNEL_DIVERSITY
    ]
    assert len(diversity_events) == 1
    jsonschema.Draft202012Validator(schema).validate(diversity_events[0])
