"""Tests for the two-tier shader admission record assembler.

These are pure-logic tests over the tier-1 (PreAdmitResult) + tier-2 (GPU testbed
verdict JSON) fusion in :mod:`autobench.bus.shader_admission`. No GPU and no bus
are required — the emit path is not exercised here, only the record assembly and
the honesty contract (partial coverage surfaced, safe=None refused).
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from autobench.bus.shader_admission import (
    build_shader_admission,
    load_shadow_verdict,
)


@dataclass
class FakePreAdmit:
    safe: Optional[bool] = True
    verdict: str = "OK"


def _full_clean_shadow():
    return {
        "candidate": 202,
        "subset_context": [],
        "decision": "Admit",
        "reason": "Clean",
        "verdict": "PASS",
        "coverage": "full",
        "frames_stepped": 8,
        "quarantined": False,
        "violation_slots": [],
        "unobservable_slots": [],
    }


def _partial_shadow():
    # A clean subset run whose verdict could not observe every declared slot
    # (some slots live where the capture is blind) — the partial-coverage shape.
    return {
        "candidate": 644,
        "subset_context": [643, 645, 646, 647, 648, 654],
        "decision": "Admit",
        "reason": "CleanPartialCoverage",
        "verdict": "PASS",
        "coverage": "partial",
        "frames_stepped": 16,
        "quarantined": False,
        "violation_slots": [],
        "unobservable_slots": [466, 490, 493, 495, 498],
    }


def _reject_shadow():
    return {
        "candidate": 644,
        "subset_context": [],
        "decision": "Reject",
        "reason": "ContractViolation",
        "verdict": "FAIL",
        "coverage": "full",
        "frames_stepped": 8,
        "quarantined": True,
        "violation_slots": [20],
        "unobservable_slots": [],
    }


def test_full_clean_admits():
    data = build_shader_admission(
        work_type=202,
        shader_id="terrain@a",
        source_run_id="run-a",
        preadmit=FakePreAdmit(safe=True, verdict="OK"),
        shadow=_full_clean_shadow(),
    )
    assert data["admitted"] is True
    assert data["shadow_dispatch_verdict"] == "PASS"
    assert data["shadow_dispatch"]["coverage"] == "full"
    assert data["quarantined"] is False


def test_partial_coverage_admits_by_default_but_carries_the_blind_spot():
    """A partial-coverage PASS still admits under the default (permissive)
    policy, but the unobservable slots ride along so a strict consumer can see
    exactly what was NOT verified — this is the honesty contract."""
    data = build_shader_admission(
        work_type=644,
        shader_id="cand@b",
        source_run_id="run-b",
        preadmit=FakePreAdmit(safe=True, verdict="OK"),
        shadow=_partial_shadow(),
    )
    assert data["admitted"] is True
    assert data["shadow_dispatch"]["coverage"] == "partial"
    assert 466 in data["shadow_dispatch"]["unobservable_slots"]
    assert data["shadow_dispatch"]["subset_context"] == [643, 645, 646, 647, 648, 654]


def test_partial_coverage_blocked_under_require_full():
    """Strict gate: a partial-coverage PASS is NOT a fully-verified pass."""
    data = build_shader_admission(
        work_type=644,
        shader_id="cand@b",
        source_run_id="run-b",
        preadmit=FakePreAdmit(safe=True, verdict="OK"),
        shadow=_partial_shadow(),
        require_full_coverage=True,
    )
    assert data["admitted"] is False
    # The PASS verdict is unchanged; only the final gate decision tightens.
    assert data["shadow_dispatch_verdict"] == "PASS"


def test_reject_and_quarantine_not_admitted():
    data = build_shader_admission(
        work_type=644,
        shader_id="bad@c",
        source_run_id="run-c",
        preadmit=FakePreAdmit(safe=True, verdict="OK"),
        shadow=_reject_shadow(),
    )
    assert data["admitted"] is False
    assert data["shadow_dispatch_verdict"] == "FAIL"
    assert data["quarantined"] is True
    assert data["shadow_dispatch"]["violation_slots"] == [20]


def test_cpu_unsafe_not_admitted():
    data = build_shader_admission(
        work_type=202,
        shader_id="ce@d",
        source_run_id="run-d",
        preadmit=FakePreAdmit(safe=False, verdict="CE"),
        shadow=_full_clean_shadow(),
    )
    assert data["admitted"] is False
    assert data["cpu_prescreen"]["safe"] is False
    assert data["cpu_prescreen"]["verdict"] == "CE"


def test_no_shadow_means_null_verdict_and_no_admit():
    """tier-2 did not run → coarse verdict null, no shadow_dispatch block, and a
    candidate is not admitted on tier-1 alone."""
    data = build_shader_admission(
        work_type=202,
        shader_id="t1only@e",
        source_run_id="run-e",
        preadmit=FakePreAdmit(safe=True, verdict="OK"),
        shadow=None,
    )
    assert data["shadow_dispatch_verdict"] is None
    assert "shadow_dispatch" not in data
    assert data["admitted"] is False


def test_safe_none_refuses_to_assemble():
    """The dynamic gate could not run → no genuine tier-1 result → refuse rather
    than coerce a fake boolean into the record."""
    with pytest.raises(ValueError):
        build_shader_admission(
            work_type=202,
            shader_id="skip@f",
            source_run_id="run-f",
            preadmit=FakePreAdmit(safe=None, verdict="SKIP"),
            shadow=_full_clean_shadow(),
        )


def test_non_enum_verdict_coerced():
    """A non-enum tier-1 verdict (e.g. SKIP) with a concrete safe flag is coerced
    into the schema enum rather than emitted verbatim (which would dead-letter)."""
    data = build_shader_admission(
        work_type=202,
        shader_id="coerce@g",
        source_run_id="run-g",
        preadmit=FakePreAdmit(safe=True, verdict="SKIP"),
        shadow=_full_clean_shadow(),
    )
    assert data["cpu_prescreen"]["verdict"] == "OK"


def test_load_shadow_verdict_missing_returns_none():
    assert load_shadow_verdict("/nonexistent/path/verdict.json") is None


def test_assembled_record_validates_against_overlay_schema():
    """If the overlay schema is installed locally, the assembled record must
    validate against it (additionalProperties:false makes this a real check)."""
    import json
    import os

    schema_path = os.path.expanduser(
        "~/.config/nervous-bus/schemas/tengine.shader.admission.v1.json"
    )
    if not os.path.isfile(schema_path):
        pytest.skip("overlay schema not installed on this host")
    jsonschema = pytest.importorskip("jsonschema")

    data = build_shader_admission(
        work_type=644,
        shader_id="cand@schema",
        source_run_id="run-schema",
        preadmit=FakePreAdmit(safe=True, verdict="OK"),
        shadow=_partial_shadow(),
    )
    schema = json.load(open(schema_path))
    jsonschema.Draft202012Validator(schema).validate(data)
