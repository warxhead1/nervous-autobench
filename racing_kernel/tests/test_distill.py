"""Tests for racing_kernel/distill.py — brain distillation (nervous-bus-71cn.6).

Covers:
  - LUT bake: shape, magic header, footprint_bytes
  - MLP bake: header fields, footprint_bytes
  - LUT/MLP query: output ranges, bilinear interpolation continuity
  - Tolerance measurement: MAE against seed oracle
  - distill_controller end-to-end: artifact written, DistillResult fields valid,
    event shape matches tengine.race.brain.v1 schema, footprint reported
  - emit_brain_event: writes a valid envelope to a temp debug.jsonl (mocked nervous)

All tests run without LLM, network, Redis, or real nervous binary — marked not-live.
The nervous publish subprocess is mocked via a captured environment check or
the emit_event=False flag.
"""

from __future__ import annotations

import json
import struct
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autobench.racing_kernel.instance import generate_instance, TRACK_LAYOUTS
from autobench.racing_kernel.oracle import SEED_RACING_PROGRAMS, _compile_policy
from autobench.racing_kernel.distill import (
    U_BINS,
    K_BINS,
    CURV_RANGE,
    MLP_HIDDEN,
    bake_lut,
    bake_mlp,
    query_lut,
    query_mlp,
    _measure_tolerance,
    _new_ulid,
    emit_brain_event,
    distill_controller,
    DistillResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oval_instance():
    return generate_instance("oval")


@pytest.fixture(scope="module")
def chicane_instance():
    return generate_instance("chicane")


@pytest.fixture(scope="module")
def two_instances(oval_instance, chicane_instance):
    return [oval_instance, chicane_instance]


@pytest.fixture(scope="module")
def seed_policy_fn():
    """Compile the first seed program into a callable."""
    _name, code = SEED_RACING_PROGRAMS[0]
    fn = _compile_policy(code)
    assert fn is not None, "Seed program failed to compile"
    return fn


# ---------------------------------------------------------------------------
# LUT bake tests
# ---------------------------------------------------------------------------

class TestLUTBake:
    def test_magic_header(self, two_instances, seed_policy_fn):
        blob = bake_lut(seed_policy_fn, two_instances)
        magic, u_bins, k_bins, k_range = struct.unpack_from("<4sIIf", blob, 0)
        assert magic == b"RLUT"
        assert u_bins == U_BINS
        assert k_bins == K_BINS
        assert abs(k_range - CURV_RANGE) < 1e-5

    def test_footprint_bytes(self, two_instances, seed_policy_fn):
        blob = bake_lut(seed_policy_fn, two_instances)
        header_size = struct.calcsize("<4sIIf")
        expected_body = U_BINS * K_BINS * 2 * 4  # float32
        assert len(blob) == header_size + expected_body
        # Typical: 16 + 8192 = 8208 bytes
        assert len(blob) < 9000, f"LUT too large: {len(blob)} bytes"

    def test_values_in_range(self, two_instances, seed_policy_fn):
        blob = bake_lut(seed_policy_fn, two_instances)
        header_size = struct.calcsize("<4sIIf")
        n_floats = U_BINS * K_BINS * 2
        values = struct.unpack_from(f"<{n_floats}f", blob, header_size)
        # lat_norm in [-1,1], throttle in [0,1]
        for i, v in enumerate(values):
            if i % 2 == 0:  # lat_norm
                assert -1.0 <= v <= 1.0, f"lat_norm out of range: {v} at index {i}"
            else:            # throttle
                assert 0.0 <= v <= 1.0, f"throttle out of range: {v} at index {i}"

    def test_requires_instances(self, seed_policy_fn):
        with pytest.raises(ValueError, match="at least one"):
            bake_lut(seed_policy_fn, [])

    def test_deterministic(self, two_instances, seed_policy_fn):
        """Same inputs → identical bytes."""
        blob1 = bake_lut(seed_policy_fn, two_instances)
        blob2 = bake_lut(seed_policy_fn, two_instances)
        assert blob1 == blob2


# ---------------------------------------------------------------------------
# LUT query tests
# ---------------------------------------------------------------------------

class TestLUTQuery:
    def test_output_ranges(self, two_instances, seed_policy_fn, oval_instance):
        blob = bake_lut(seed_policy_fn, two_instances)
        hw = oval_instance.half_width
        for u in [0.0, 0.25, 0.5, 0.75, 0.99]:
            for k in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                lat, thr = query_lut(blob, u, k, hw)
                assert -hw * 1.01 <= lat <= hw * 1.01, f"lat out of range: {lat}"
                assert 0.0 <= thr <= 1.0, f"thr out of range: {thr}"

    def test_continuity(self, two_instances, seed_policy_fn, oval_instance):
        """Small input perturbations produce small output changes (no huge jumps)."""
        blob = bake_lut(seed_policy_fn, two_instances)
        hw = oval_instance.half_width
        u = 0.3
        k0, k1 = 0.05, 0.051
        lat0, thr0 = query_lut(blob, u, k0, hw)
        lat1, thr1 = query_lut(blob, u, k1, hw)
        # Bilinear interpolation — tiny curvature step → tiny output step
        assert abs(lat1 - lat0) < hw * 0.1, "LUT lat not continuous"
        assert abs(thr1 - thr0) < 0.1, "LUT thr not continuous"

    def test_rejects_wrong_magic(self):
        bad = b"XXXX" + b"\x00" * 100
        with pytest.raises(ValueError, match="RLUT"):
            query_lut(bad, 0.0, 0.0, 5.0)


# ---------------------------------------------------------------------------
# MLP bake tests
# ---------------------------------------------------------------------------

class TestMLPBake:
    def test_magic_header(self, two_instances, seed_policy_fn):
        blob = bake_mlp(seed_policy_fn, two_instances)
        magic, n_in, n_hidden, n_out, k_range = struct.unpack_from("<4sIIIf", blob, 0)
        assert magic == b"RMLP"
        assert n_in == 2
        assert n_hidden == MLP_HIDDEN
        assert n_out == 2
        assert abs(k_range - CURV_RANGE) < 1e-5

    def test_footprint_bytes(self, two_instances, seed_policy_fn):
        blob = bake_mlp(seed_policy_fn, two_instances)
        header_size = struct.calcsize("<4sIIIf")
        # W1: hidden×2, b1: hidden, W2: 2×hidden, b2: 2
        expected_params = MLP_HIDDEN * 2 + MLP_HIDDEN + 2 * MLP_HIDDEN + 2
        expected_body = expected_params * 4  # float32
        assert len(blob) == header_size + expected_body
        # MLP is much smaller than LUT: < 1 KiB
        assert len(blob) < 1024, f"MLP too large: {len(blob)} bytes"

    def test_requires_instances(self, seed_policy_fn):
        with pytest.raises(ValueError, match="at least one"):
            bake_mlp(seed_policy_fn, [])


# ---------------------------------------------------------------------------
# MLP query tests
# ---------------------------------------------------------------------------

class TestMLPQuery:
    def test_output_ranges(self, two_instances, seed_policy_fn, oval_instance):
        blob = bake_mlp(seed_policy_fn, two_instances)
        hw = oval_instance.half_width
        for u in [0.0, 0.25, 0.5, 0.75, 0.99]:
            for k in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                lat, thr = query_mlp(blob, u, k, hw)
                assert -hw * 1.01 <= lat <= hw * 1.01, f"lat out of range: {lat}"
                assert 0.0 <= thr <= 1.0, f"thr out of range: {thr}"

    def test_rejects_wrong_magic(self):
        bad = b"XXXX" + b"\x00" * 200
        with pytest.raises(ValueError, match="RMLP"):
            query_mlp(bad, 0.0, 0.0, 5.0)


# ---------------------------------------------------------------------------
# Tolerance measurement
# ---------------------------------------------------------------------------

class TestTolerance:
    def test_lut_tolerance_low(self, two_instances, seed_policy_fn, oval_instance):
        """LUT approximation error should be < 10% of half_width for lateral."""
        blob = bake_lut(seed_policy_fn, two_instances)
        hw = oval_instance.half_width

        def _q(u, k, hw_):
            return query_lut(blob, u, k, hw_)

        tol = _measure_tolerance(seed_policy_fn, _q, two_instances)
        # lat_mae < 10% of half_width (a loose but meaningful bound)
        assert tol["lat_mae"] < hw * 0.10, (
            f"LUT lateral MAE too high: {tol['lat_mae']:.4f} > {hw * 0.10:.4f}"
        )
        assert tol["thr_mae"] < 0.10, f"LUT throttle MAE too high: {tol['thr_mae']:.4f}"

    def test_mlp_tolerance_reasonable(self, two_instances, seed_policy_fn, oval_instance):
        """MLP tolerance is looser (regression fit), but should be < 30% of half_width."""
        blob = bake_mlp(seed_policy_fn, two_instances)
        hw = oval_instance.half_width

        def _q(u, k, hw_):
            return query_mlp(blob, u, k, hw_)

        tol = _measure_tolerance(seed_policy_fn, _q, two_instances)
        assert tol["lat_mae"] < hw * 0.30, (
            f"MLP lateral MAE too high: {tol['lat_mae']:.4f}"
        )

    def test_tolerance_dict_keys(self, two_instances, seed_policy_fn, oval_instance):
        blob = bake_lut(seed_policy_fn, two_instances)

        def _q(u, k, hw_):
            return query_lut(blob, u, k, hw_)

        tol = _measure_tolerance(seed_policy_fn, _q, two_instances)
        assert set(tol.keys()) == {"lat_mae", "thr_mae", "lat_max", "thr_max"}


# ---------------------------------------------------------------------------
# emit_brain_event — event shape validation
# ---------------------------------------------------------------------------

class TestEmitBrainEvent:
    def test_writes_valid_envelope_to_jsonl(self, tmp_path):
        """emit_brain_event writes a valid tengine.race.brain.v1 envelope."""
        debug_log = tmp_path / "debug.jsonl"
        brain_id = _new_ulid()

        with patch("autobench.racing_kernel.distill._find_nervous_bin", return_value=None):
            with patch.object(Path, "home", return_value=tmp_path):
                emit_brain_event(
                    brain_id=brain_id,
                    status="distilled",
                    kind="lut",
                    artifact_uri="file:///tmp/test.lut",
                    footprint_bytes=8208,
                    source_run="run_test_001",
                    fitness=0.85,
                )

        # Find the written envelope
        written = None
        for candidate in [
            debug_log,
            tmp_path / ".cache" / "nervous-bus" / "debug.jsonl",
        ]:
            if candidate.exists():
                written = candidate
                break

        assert written is not None, "No debug.jsonl written"
        lines = [l.strip() for l in written.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1

        envelope = json.loads(lines[-1])
        # CloudEvents envelope fields
        assert envelope["specversion"] == "1.0"
        assert envelope["type"] == "tengine.race.brain.v1"
        assert envelope["source"] == "/autobench/race"
        assert "id" in envelope

        data = envelope["data"]
        assert data["brain_id"] == brain_id
        assert data["status"] == "distilled"
        assert data["kind"] == "lut"
        assert data["footprint_bytes"] == 8208
        assert data["fitness"] == pytest.approx(0.85)
        assert "created_at" in data

    def test_event_schema_required_fields(self, tmp_path):
        """Data block must satisfy tengine.race.brain.v1 required fields."""
        with patch("autobench.racing_kernel.distill._find_nervous_bin", return_value=None):
            with patch.object(Path, "home", return_value=tmp_path):
                emit_brain_event(
                    brain_id=_new_ulid(),
                    status="distilled",
                    kind="mlp",
                    artifact_uri="file:///tmp/test.mlp",
                    footprint_bytes=800,
                    source_run="run_x",
                    fitness=0.80,
                )

        log = tmp_path / ".cache" / "nervous-bus" / "debug.jsonl"
        assert log.exists()
        data = json.loads(log.read_text().splitlines()[-1])["data"]

        # Schema required: brain_id, status, kind
        for field in ("brain_id", "status", "kind"):
            assert field in data, f"Missing required field: {field}"

        # Status must be enum value
        assert data["status"] in ("distilled", "deployed", "retired")
        # Kind must be enum value
        assert data["kind"] in ("lut", "mlp", "hybrid", "kernel")


# ---------------------------------------------------------------------------
# distill_controller end-to-end
# ---------------------------------------------------------------------------

class TestDistillController:
    def test_lut_e2e(self, two_instances, tmp_path):
        """End-to-end LUT distillation: artifact written, DistillResult valid."""
        _name, code = SEED_RACING_PROGRAMS[0]
        out = tmp_path / "brain.lut"

        with patch("autobench.racing_kernel.distill._find_nervous_bin", return_value=None):
            result = distill_controller(
                code=code,
                instances=two_instances,
                output_path=out,
                kind="lut",
                source_run="test_run_lut",
                fitness=0.82,
                emit_event=False,  # suppress nervous call in tests
            )

        assert isinstance(result, DistillResult)
        assert result.kind == "lut"
        assert out.exists() or Path(result.artifact_path).exists()
        assert result.footprint_bytes > 0
        assert result.footprint_bytes < 10000  # must be tiny
        assert 0.0 <= result.fitness <= 1.0
        assert result.source_run == "test_run_lut"
        assert len(result.brain_id) == 26  # ULID length
        assert set(result.tolerance.keys()) == {"lat_mae", "thr_mae", "lat_max", "thr_max"}

    def test_mlp_e2e(self, two_instances, tmp_path):
        """End-to-end MLP distillation: artifact written, DistillResult valid."""
        _name, code = SEED_RACING_PROGRAMS[1]
        out = tmp_path / "brain.mlp"

        with patch("autobench.racing_kernel.distill._find_nervous_bin", return_value=None):
            result = distill_controller(
                code=code,
                instances=two_instances,
                output_path=out,
                kind="mlp",
                source_run="test_run_mlp",
                fitness=0.78,
                emit_event=False,
            )

        assert isinstance(result, DistillResult)
        assert result.kind == "mlp"
        artifact = Path(result.artifact_path)
        assert artifact.exists()
        assert result.footprint_bytes == artifact.stat().st_size
        assert result.footprint_bytes < 1024  # MLP must be sub-1 KiB

    def test_footprint_reported(self, two_instances, tmp_path):
        """footprint_bytes == actual file size on disk."""
        _name, code = SEED_RACING_PROGRAMS[0]
        out = tmp_path / "check.lut"
        with patch("autobench.racing_kernel.distill._find_nervous_bin", return_value=None):
            result = distill_controller(
                code=code, instances=two_instances,
                output_path=out, kind="lut",
                emit_event=False,
            )
        assert result.footprint_bytes == Path(result.artifact_path).stat().st_size

    def test_invalid_code_raises(self, two_instances, tmp_path):
        with pytest.raises(ValueError, match="Failed to compile"):
            distill_controller(
                code="not valid python !",
                instances=two_instances,
                output_path=tmp_path / "bad.lut",
                emit_event=False,
            )

    def test_empty_instances_raises(self, tmp_path):
        _name, code = SEED_RACING_PROGRAMS[0]
        with pytest.raises(ValueError, match="at least one"):
            distill_controller(
                code=code,
                instances=[],
                output_path=tmp_path / "empty.lut",
                emit_event=False,
            )

    def test_unknown_kind_raises(self, two_instances, tmp_path):
        _name, code = SEED_RACING_PROGRAMS[0]
        with pytest.raises(ValueError, match="Unknown kind"):
            distill_controller(
                code=code,
                instances=two_instances,
                output_path=tmp_path / "bad.xxx",
                kind="hybrid_future",
                emit_event=False,
            )

    def test_event_emitted_on_success(self, two_instances, tmp_path):
        """When emit_event=True the brain event is written to debug.jsonl."""
        _name, code = SEED_RACING_PROGRAMS[0]
        out = tmp_path / "evented.lut"

        with patch("autobench.racing_kernel.distill._find_nervous_bin", return_value=None):
            with patch.object(Path, "home", return_value=tmp_path):
                result = distill_controller(
                    code=code,
                    instances=two_instances,
                    output_path=out,
                    kind="lut",
                    source_run="test_event_run",
                    fitness=0.80,
                    emit_event=True,
                )

        log = tmp_path / ".cache" / "nervous-bus" / "debug.jsonl"
        assert log.exists(), "debug.jsonl not written"
        lines = [l for l in log.read_text().splitlines() if "tengine.race.brain.v1" in l]
        assert len(lines) >= 1, "tengine.race.brain.v1 event not found in debug.jsonl"

        env = json.loads(lines[-1])
        assert env["type"] == "tengine.race.brain.v1"
        data = env["data"]
        assert data["brain_id"] == result.brain_id
        assert data["status"] == "distilled"
        assert data["footprint_bytes"] == result.footprint_bytes
