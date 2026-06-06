"""Unit tests for autobench.bus.gpu_admission (nervous-bus-s0u3.7).

All tests run without a live Redis instance or GPU engine.  The heartbeat
read path is mocked at the ``_fetch_latest_heartbeat`` function level.

Test coverage:
  1. No heartbeat available → ALLOW (graceful fallback)
  2. Heartbeat with queue_depth <= threshold → ALLOW
  3. Heartbeat with queue_depth > threshold → DEFER
  4. Per-island in-flight quota exceeded → DEFER (even with no/low heartbeat)
  5. Multiple islands are tracked independently
  6. record_completed releases a slot correctly
  7. record_completed underflow is safe (no crash / no negative counts)
  8. should_submit respects BOTH conditions: queue deep AND quota full → DEFER
  9. _parse_xrevrange_lines correctly extracts data from redis-cli output
  10. _parse_xrevrange_lines with a field-count line (older redis-cli format)
  11. _parse_xrevrange_lines returns None for malformed/empty output
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from autobench.bus.gpu_admission import (
    DEFAULT_MAX_IN_FLIGHT_PER_ISLAND,
    DEFAULT_QUEUE_DEPTH_THRESHOLD,
    GPUAdmissionGate,
    HeartbeatSnapshot,
    SubmitDecision,
    _parse_xrevrange_lines,
)


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _hb(queue_depth: int, current_job_id: str | None = "job-abc") -> HeartbeatSnapshot:
    """Build a synthetic HeartbeatSnapshot for mocking."""
    return HeartbeatSnapshot(
        queue_depth=queue_depth,
        current_job_id=current_job_id,
        fencing_token="tok-001",
        holder="tengine/lab",
        ts="2026-06-05T12:00:00Z",
    )


def _gate(**kwargs) -> GPUAdmissionGate:
    """Build a GPUAdmissionGate with redis-cli lookups disabled by default."""
    g = GPUAdmissionGate(**kwargs)
    # Prevent any real redis-cli subprocess calls.
    g._redis_bin = None
    return g


def _gate_with_hb(heartbeat: HeartbeatSnapshot | None, **kwargs) -> GPUAdmissionGate:
    """Build a gate whose _latest_heartbeat() always returns *heartbeat*."""
    g = _gate(**kwargs)
    g._latest_heartbeat = lambda: heartbeat  # type: ignore[method-assign]
    return g


# ── Test 1: no heartbeat → ALLOW ─────────────────────────────────────────── #

def test_no_heartbeat_defaults_to_allow():
    """When no heartbeat is available, admission MUST default to ALLOW."""
    gate = _gate_with_hb(None)
    assert gate.should_submit("island_0") == SubmitDecision.ALLOW


# ── Test 2: low queue depth → ALLOW ──────────────────────────────────────── #

def test_low_queue_depth_allows():
    """queue_depth == threshold → ALLOW (threshold is exclusive upper bound)."""
    gate = _gate_with_hb(_hb(queue_depth=DEFAULT_QUEUE_DEPTH_THRESHOLD))
    assert gate.should_submit("island_0") == SubmitDecision.ALLOW


def test_zero_queue_depth_allows():
    gate = _gate_with_hb(_hb(queue_depth=0))
    assert gate.should_submit("island_0") == SubmitDecision.ALLOW


# ── Test 3: high queue depth → DEFER ─────────────────────────────────────── #

def test_high_queue_depth_defers():
    """queue_depth > threshold → DEFER."""
    gate = _gate_with_hb(_hb(queue_depth=DEFAULT_QUEUE_DEPTH_THRESHOLD + 1))
    assert gate.should_submit("island_0") == SubmitDecision.DEFER


def test_custom_threshold_respected():
    gate = _gate_with_hb(_hb(queue_depth=3), queue_depth_threshold=2)
    assert gate.should_submit("island_0") == SubmitDecision.DEFER

    gate2 = _gate_with_hb(_hb(queue_depth=2), queue_depth_threshold=2)
    assert gate2.should_submit("island_0") == SubmitDecision.ALLOW


# ── Test 4: per-island quota exceeded → DEFER ────────────────────────────── #

def test_island_quota_defers_when_full():
    """When in-flight >= quota, DEFER even with empty queue."""
    gate = _gate_with_hb(_hb(queue_depth=0), max_in_flight_per_island=2)
    gate.record_submitted("island_0")
    gate.record_submitted("island_0")
    # Now at quota limit.
    assert gate.should_submit("island_0") == SubmitDecision.DEFER


def test_island_quota_allows_below_limit():
    gate = _gate_with_hb(_hb(queue_depth=0), max_in_flight_per_island=2)
    gate.record_submitted("island_0")  # 1 in flight
    assert gate.should_submit("island_0") == SubmitDecision.ALLOW


# ── Test 5: multiple islands tracked independently ────────────────────────── #

def test_multiple_islands_independent():
    gate = _gate_with_hb(_hb(queue_depth=0), max_in_flight_per_island=1)
    gate.record_submitted("island_0")

    # island_0 is at quota but island_1 is not.
    assert gate.should_submit("island_0") == SubmitDecision.DEFER
    assert gate.should_submit("island_1") == SubmitDecision.ALLOW


def test_integer_island_id():
    """source_island can be an integer (the schema allows string | integer)."""
    gate = _gate_with_hb(_hb(queue_depth=0), max_in_flight_per_island=1)
    gate.record_submitted(0)  # integer island id
    assert gate.should_submit(0) == SubmitDecision.DEFER
    assert gate.should_submit(1) == SubmitDecision.ALLOW


# ── Test 6: record_completed releases a slot ─────────────────────────────── #

def test_record_completed_releases_slot():
    gate = _gate_with_hb(_hb(queue_depth=0), max_in_flight_per_island=1)
    gate.record_submitted("island_0")
    assert gate.should_submit("island_0") == SubmitDecision.DEFER

    gate.record_completed("island_0")
    assert gate.should_submit("island_0") == SubmitDecision.ALLOW


def test_in_flight_counter_accuracy():
    gate = _gate_with_hb(None, max_in_flight_per_island=10)
    for _ in range(3):
        gate.record_submitted("island_0")
    assert gate.in_flight("island_0") == 3

    gate.record_completed("island_0")
    assert gate.in_flight("island_0") == 2


# ── Test 7: underflow safety ─────────────────────────────────────────────── #

def test_record_completed_underflow_is_safe():
    """Calling record_completed with no jobs in flight must not raise or go negative."""
    gate = _gate_with_hb(None)
    gate.record_completed("island_0")  # no prior record_submitted
    assert gate.in_flight("island_0") == 0


# ── Test 8: combined — queue deep AND quota full ──────────────────────────── #

def test_defer_when_both_conditions_met():
    gate = _gate_with_hb(
        _hb(queue_depth=10),
        queue_depth_threshold=4,
        max_in_flight_per_island=1,
    )
    gate.record_submitted("island_0")
    assert gate.should_submit("island_0") == SubmitDecision.DEFER


# ── Test 9: _parse_xrevrange_lines with canonical redis-cli output ─────────── #

def test_parse_xrevrange_canonical():
    """Parses the flat redis-cli text format correctly."""
    data_payload = {
        "queue_depth": 3,
        "current_job_id": "job-xyz",
        "fencing_token": "tok-999",
        "holder": "tengine/lab",
        "ts": "2026-06-05T12:00:00Z",
    }
    lines = [
        "1749139200000-0",       # stream entry ID
        "data",
        json.dumps(data_payload),
    ]
    snap = _parse_xrevrange_lines(lines)
    assert snap is not None
    assert snap.queue_depth == 3
    assert snap.current_job_id == "job-xyz"
    assert snap.fencing_token == "tok-999"


# ── Test 10: _parse_xrevrange_lines with older field-count format ─────────── #

def test_parse_xrevrange_with_field_count_line():
    """Older redis-cli versions insert a field count after the entry ID."""
    data_payload = {
        "queue_depth": 1,
        "current_job_id": None,
        "fencing_token": "tok-42",
        "holder": "tengine/lab",
        "ts": "2026-06-05T12:00:00Z",
    }
    lines = [
        "1749139200000-0",   # stream entry ID
        "2",                 # field count (older redis-cli format)
        "data",
        json.dumps(data_payload),
    ]
    snap = _parse_xrevrange_lines(lines)
    assert snap is not None
    assert snap.queue_depth == 1
    assert snap.current_job_id is None


# ── Test 11: _parse_xrevrange_lines returns None for bad input ────────────── #

def test_parse_xrevrange_empty_returns_none():
    assert _parse_xrevrange_lines([]) is None


def test_parse_xrevrange_no_data_key_returns_none():
    lines = [
        "1749139200000-0",
        "some_other_key",
        "some_value",
    ]
    assert _parse_xrevrange_lines(lines) is None


def test_parse_xrevrange_invalid_json_returns_none():
    lines = [
        "1749139200000-0",
        "data",
        "{not valid json",
    ]
    assert _parse_xrevrange_lines(lines) is None


# ── Test 12: None source_island uses _default key ─────────────────────────── #

def test_none_island_uses_default_key():
    gate = _gate_with_hb(_hb(queue_depth=0), max_in_flight_per_island=1)
    gate.record_submitted(None)
    assert gate.in_flight(None) == 1
    assert gate.should_submit(None) == SubmitDecision.DEFER
    # Explicit "_default" string maps to the same bucket.
    gate.record_completed("_default")
    assert gate.in_flight(None) == 0


# ── Test 13: queue_depth() helper ─────────────────────────────────────────── #

def test_queue_depth_helper_returns_value():
    gate = _gate_with_hb(_hb(queue_depth=7))
    assert gate.queue_depth() == 7


def test_queue_depth_helper_returns_none_when_no_heartbeat():
    gate = _gate_with_hb(None)
    assert gate.queue_depth() is None


# ── Test 14: default constants are sensible ───────────────────────────────── #

def test_default_constants():
    assert DEFAULT_QUEUE_DEPTH_THRESHOLD == 4
    assert DEFAULT_MAX_IN_FLIGHT_PER_ISLAND == 2
