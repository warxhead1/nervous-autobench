"""Tests for racing_kernel.rollout_eval — GPU rollout evaluator.

All tests run without LLM, network, Redis, or real GPU.
Publisher, gate, and result-fetcher are mocked via injection.
Marked not-live.

Run:
    NBUS_ROOT=/home/eric/projects/nervous-bus \
    python -m pytest -m "not live" racing_kernel/tests/test_rollout_eval.py -q
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from autobench.racing_kernel.instance import generate_instance
from autobench.racing_kernel.oracle import evaluate_on_instance
from autobench.racing_kernel.rollout_eval import (
    _fitness_from_result,
    _parse_xrevrange_output,
    calibrate_ref_lap_time,
    evaluate_via_rollout,
)
from autobench.bus.gpu_admission import GPUAdmissionGate, SubmitDecision


# ── Helpers ────────────────────────────────────────────────────────────────────

_SIMPLE_POLICY = """\
def racing_line(u, curvature, half_width, speed_limit):
    lateral_offset = -curvature * half_width * 0.6
    lateral_offset = max(-half_width * 0.85, min(half_width * 0.85, lateral_offset))
    throttle = min(1.0, speed_limit / 22.0)
    return lateral_offset, throttle
"""


def _make_gate(decision: SubmitDecision) -> GPUAdmissionGate:
    """Return a GPUAdmissionGate whose should_submit always returns *decision*."""
    gate = MagicMock(spec=GPUAdmissionGate)
    gate.should_submit.return_value = decision
    # record_submitted / record_completed are no-ops in mocks.
    return gate


def _make_publisher(success: bool = True) -> MagicMock:
    """Return a mock GPUJobPublisher whose publish returns *success*."""
    pub = MagicMock()
    pub.publish.return_value = success
    return pub


def _make_fetcher(result: dict[str, Any] | None) -> MagicMock:
    """Return a callable that immediately returns *result* (bypasses redis poll)."""
    fetcher = MagicMock(return_value=result)
    return fetcher


# ── Core cases ─────────────────────────────────────────────────────────────────

class TestEvaluateViaRollout:

    def _inst(self) -> Any:
        return generate_instance("oval")

    # ── Case 1: ALLOW + result → fitness from GPU result ─────────────────── #

    def test_allow_with_result_returns_gpu_fitness(self):
        """ALLOW gate + valid GPU result → fitness derived from lap_time."""
        inst = self._inst()
        gpu_result = {
            "job_id": "some_id",
            "status": "completed",
            "verdict": "OK",
            "silo_tester_report": {
                "lap_time_ms": inst.ref_lap_time_s * 1000 * 1.05,  # 5% slower → good score
                "collisions": 0,
            },
        }
        fitness = evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=_make_publisher(True),
            gate=_make_gate(SubmitDecision.ALLOW),
            _result_fetcher=_make_fetcher(gpu_result),
        )
        assert fitness is not None
        # 5% slower than ref → speed_score ~0.64; collision_free=1.0 → weighted > 0.4
        assert 0.3 < fitness <= 1.0

    def test_allow_with_timed_out_result_returns_zero(self):
        """ALLOW + result with status=timed_out → 0.0 fitness."""
        inst = self._inst()
        gpu_result = {
            "job_id": "some_id",
            "status": "timed_out",
            "verdict": "TLE",
        }
        fitness = evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=_make_publisher(True),
            gate=_make_gate(SubmitDecision.ALLOW),
            _result_fetcher=_make_fetcher(gpu_result),
        )
        assert fitness == 0.0

    def test_allow_with_failed_result_returns_zero(self):
        """ALLOW + result with status=failed → 0.0 fitness."""
        inst = self._inst()
        gpu_result = {
            "job_id": "some_id",
            "status": "failed",
            "verdict": "RE",
        }
        fitness = evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=_make_publisher(True),
            gate=_make_gate(SubmitDecision.ALLOW),
            _result_fetcher=_make_fetcher(gpu_result),
        )
        assert fitness == 0.0

    # ── Case 2: DEFER → CPU oracle fallback ──────────────────────────────── #

    def test_defer_falls_back_to_cpu_oracle(self):
        """DEFER → evaluate_on_instance (CPU oracle) is called; publisher is NOT."""
        inst = self._inst()
        pub = _make_publisher(True)
        fitness = evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=pub,
            gate=_make_gate(SubmitDecision.DEFER),
        )
        # publisher.publish should NOT have been called
        pub.publish.assert_not_called()
        # fitness should match the CPU oracle
        cpu_fitness = evaluate_on_instance(_SIMPLE_POLICY, inst)
        assert fitness is not None
        assert cpu_fitness is not None
        assert abs(fitness - cpu_fitness) < 1e-9

    # ── Case 3: ALLOW + no result (timeout) → CPU fallback ───────────────── #

    def test_allow_no_result_falls_back_to_cpu(self):
        """ALLOW + publish succeeds, but no GPU result arrives → CPU fallback."""
        inst = self._inst()
        fitness = evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=_make_publisher(True),
            gate=_make_gate(SubmitDecision.ALLOW),
            _result_fetcher=_make_fetcher(None),  # simulate timeout
        )
        cpu_fitness = evaluate_on_instance(_SIMPLE_POLICY, inst)
        assert fitness is not None
        assert cpu_fitness is not None
        assert abs(fitness - cpu_fitness) < 1e-9

    # ── Case 4: publisher failure → CPU fallback ──────────────────────────── #

    def test_publisher_failure_falls_back_to_cpu(self):
        """publish() returns False → gate.record_submitted NOT called; CPU fallback."""
        inst = self._inst()
        gate = _make_gate(SubmitDecision.ALLOW)
        fitness = evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=_make_publisher(False),
            gate=gate,
            _result_fetcher=_make_fetcher(None),
        )
        gate.record_submitted.assert_not_called()
        cpu_fitness = evaluate_on_instance(_SIMPLE_POLICY, inst)
        assert fitness is not None
        assert cpu_fitness is not None
        assert abs(fitness - cpu_fitness) < 1e-9

    # ── Case 5: disabled → straight to CPU oracle ─────────────────────────── #

    def test_disabled_routes_to_cpu_oracle(self):
        """enabled=False → no gate, no publisher; CPU oracle result."""
        inst = self._inst()
        pub = _make_publisher(True)
        gate = _make_gate(SubmitDecision.ALLOW)
        fitness = evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=False,
            publisher=pub,
            gate=gate,
        )
        pub.publish.assert_not_called()
        cpu_fitness = evaluate_on_instance(_SIMPLE_POLICY, inst)
        assert fitness is not None
        assert cpu_fitness is not None
        assert abs(fitness - cpu_fitness) < 1e-9

    # ── Case 6: gate.record_completed called even on timeout ─────────────── #

    def test_record_completed_called_on_timeout(self):
        """gate.record_completed must be called even when result polling times out."""
        inst = self._inst()
        gate = _make_gate(SubmitDecision.ALLOW)
        evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=_make_publisher(True),
            gate=gate,
            _result_fetcher=_make_fetcher(None),
        )
        gate.record_completed.assert_called_once_with(0)

    def test_record_completed_called_on_success(self):
        """gate.record_completed must be called when GPU result is successfully consumed."""
        inst = self._inst()
        gpu_result = {
            "status": "completed",
            "verdict": "OK",
            "silo_tester_report": {"lap_time_ms": inst.ref_lap_time_s * 1000 * 1.1, "collisions": 0},
        }
        gate = _make_gate(SubmitDecision.ALLOW)
        evaluate_via_rollout(
            _SIMPLE_POLICY,
            inst,
            island=0,
            enabled=True,
            publisher=_make_publisher(True),
            gate=gate,
            _result_fetcher=_make_fetcher(gpu_result),
        )
        gate.record_completed.assert_called_once_with(0)


# ── _fitness_from_result unit tests ───────────────────────────────────────────

class TestFitnessFromResult:

    def _inst(self) -> Any:
        return generate_instance("oval")

    def test_ok_with_lap_time_returns_positive_fitness(self):
        inst = self._inst()
        result = {
            "status": "completed",
            "verdict": "OK",
            "silo_tester_report": {"lap_time_ms": inst.ref_lap_time_s * 1000 * 1.05, "collisions": 0},
        }
        f = _fitness_from_result(result, inst, inst.ref_lap_time_s)
        assert f > 0.0

    def test_timed_out_status_returns_zero(self):
        inst = self._inst()
        f = _fitness_from_result({"status": "timed_out", "verdict": "TLE"}, inst, inst.ref_lap_time_s)
        assert f == 0.0

    def test_failed_status_returns_zero(self):
        inst = self._inst()
        f = _fitness_from_result({"status": "failed", "verdict": "RE"}, inst, inst.ref_lap_time_s)
        assert f == 0.0

    def test_ce_verdict_returns_zero(self):
        inst = self._inst()
        f = _fitness_from_result({"status": "completed", "verdict": "CE"}, inst, inst.ref_lap_time_s)
        assert f == 0.0

    def test_collisions_reduce_track_term(self):
        """Result with collisions > 0 has lower fitness than collision-free."""
        inst = self._inst()
        lap_ms = inst.ref_lap_time_s * 1000 * 1.05
        result_clean = {
            "status": "completed",
            "verdict": "OK",
            "silo_tester_report": {"lap_time_ms": lap_ms, "collisions": 0},
        }
        result_crash = {
            "status": "completed",
            "verdict": "OK",
            "silo_tester_report": {"lap_time_ms": lap_ms, "collisions": 3},
        }
        f_clean = _fitness_from_result(result_clean, inst, inst.ref_lap_time_s)
        f_crash = _fitness_from_result(result_crash, inst, inst.ref_lap_time_s)
        assert f_clean > f_crash

    def test_no_lap_time_returns_failure_fitness(self):
        inst = self._inst()
        result = {"status": "completed", "verdict": "OK", "silo_tester_report": {}}
        f = _fitness_from_result(result, inst, inst.ref_lap_time_s)
        assert f == 0.0

    def test_fitness_clamped_to_unit_interval(self):
        inst = self._inst()
        # Give an implausibly fast lap time (faster than ref) — must not exceed 1.0
        result = {
            "status": "completed",
            "verdict": "OK",
            "silo_tester_report": {"lap_time_ms": 1.0, "collisions": 0},  # 1 ms lap
        }
        f = _fitness_from_result(result, inst, inst.ref_lap_time_s)
        assert 0.0 <= f <= 1.0


# ── calibrate_ref_lap_time unit tests ─────────────────────────────────────────

class TestCalibrateRefLapTime:

    def test_no_redis_returns_synthetic(self):
        """No redis_bin → falls back to instance.ref_lap_time_s."""
        inst = generate_instance("oval")
        result = calibrate_ref_lap_time(inst, redis_bin="nonexistent_redis_cli_xyz")
        assert result == inst.ref_lap_time_s

    def test_synthetic_fallback_is_positive(self):
        inst = generate_instance("chicane")
        result = calibrate_ref_lap_time(inst, redis_bin=None)
        assert result > 0.0


# ── _parse_xrevrange_output unit tests ────────────────────────────────────────

class TestParseXrevrangeOutput:

    def test_empty_output_returns_empty_list(self):
        assert _parse_xrevrange_output("") == []

    def test_single_entry_parses_data(self):
        """A minimal XREVRANGE entry with a 'data' JSON field."""
        raw = (
            "1748000000000-0\n"
            "2\n"
            "data\n"
            '{"job_id": "abc123", "status": "completed", "verdict": "OK"}\n'
        )
        entries = _parse_xrevrange_output(raw)
        assert len(entries) == 1
        assert entries[0]["job_id"] == "abc123"
        assert entries[0]["status"] == "completed"

    def test_malformed_json_data_skipped(self):
        raw = (
            "1748000000000-0\n"
            "2\n"
            "data\n"
            "not-valid-json\n"
        )
        entries = _parse_xrevrange_output(raw)
        assert entries == []

    def test_no_data_key_skipped(self):
        raw = (
            "1748000000000-0\n"
            "2\n"
            "type\n"
            "autobench.gpu_result.v1\n"
        )
        entries = _parse_xrevrange_output(raw)
        assert entries == []
