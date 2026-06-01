"""Tests for the SDF topology oracle and island routing.

Covers:
    * TOPO_TARGETS dict has required instance keys
    * Analytical gyroid density measurement → ~0.17  (pure Python, no sandbox)
    * Analytical round_box density measurement → ~0.01 (pure Python, no sandbox)
    * Topology score discriminates: gyroid density >> round_box for gyroid target
    * Island routing: island 0 evaluates gyroid, island 1 evaluates round_box

Unit tests (2) use oracle_calibration._compute_sign_change_density which is
100% Python/numpy — no sandbox, no GPU required.

Mocked test (1) patches compile_and_run to return a fixed JSON result so that
compute_topology_score can be called without gVisor.

Integration test (1): SDFKernel.evaluate_fitness routing — patches ensure_executor
so the sandbox gate never fires.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Constants that the tests assert against (empirically measured values)
# ---------------------------------------------------------------------------

_GYROID_DENSITY_EXPECTED = 0.178    # calibrated target in TOPO_TARGETS
_GYROID_DENSITY_TOLERANCE = 0.04    # measured 0.1573, target 0.178 — allow ±~2σ

_ROUND_BOX_DENSITY_EXPECTED = 0.009  # calibrated target in TOPO_TARGETS
_ROUND_BOX_DENSITY_TOLERANCE = 0.015 # measured 0.0163 on pure-Python grid


# ---------------------------------------------------------------------------
# TOPO_TARGETS presence tests
# ---------------------------------------------------------------------------


def test_topo_targets_loaded():
    """TOPO_TARGETS must define calibrated targets for all three benchmark instances."""
    from autobench.sdf_kernel import TOPO_TARGETS

    for required in ("gyroid", "round_box", "warped_sphere"):
        assert required in TOPO_TARGETS, f"TOPO_TARGETS missing '{required}'"
        entry = TOPO_TARGETS[required]
        assert "target" in entry, f"TOPO_TARGETS['{required}'] missing 'target'"
        assert "sigma" in entry, f"TOPO_TARGETS['{required}'] missing 'sigma'"
        assert entry["target"] > 0.0
        assert entry["sigma"] > 0.0


# ---------------------------------------------------------------------------
# Analytical density tests — pure Python, no sandbox
# ---------------------------------------------------------------------------


def test_sign_change_density_analytical_gyroid():
    """Analytical gyroid sign_change_density on 24^3 grid should be ~0.17.

    Uses oracle_calibration._compute_sign_change_density (pure numpy).
    Gyroid is triply periodic → very high sign-change density.
    """
    from autobench.audit.oracle_calibration import _compute_sign_change_density
    from autobench.sdf_kernel import _sdf_gyroid

    density = _compute_sign_change_density(_sdf_gyroid, grid_size=24, lo=-1.5, hi=1.5)

    assert density > 0.10, f"Gyroid density too low: {density:.4f} (expected ~0.17)"
    assert density < 0.35, f"Gyroid density suspiciously high: {density:.4f}"
    # Also assert it's reasonably close to the documented target
    assert abs(density - _GYROID_DENSITY_EXPECTED) < 2 * _GYROID_DENSITY_TOLERANCE, (
        f"Gyroid density {density:.4f} is more than 2*tolerance from "
        f"expected {_GYROID_DENSITY_EXPECTED}"
    )


def test_sign_change_density_round_box():
    """Analytical round_box sign_change_density should be very low (~0.01).

    Round box is a simple closed convex surface — very few zero crossings on
    the 24^3 grid compared to gyroid.  The 20x ratio is the key discriminative signal.
    """
    from autobench.audit.oracle_calibration import _compute_sign_change_density
    from autobench.sdf_kernel import _sdf_round_box

    density = _compute_sign_change_density(_sdf_round_box, grid_size=24, lo=-1.5, hi=1.5)

    assert density < 0.05, f"Round-box density unexpectedly high: {density:.4f} (expected ~0.01)"
    assert density >= 0.0


def test_gyroid_has_higher_density_than_round_box():
    """Gyroid density must be at least 5x round_box — strong discriminative signal."""
    from autobench.audit.oracle_calibration import _compute_sign_change_density
    from autobench.sdf_kernel import _sdf_gyroid, _sdf_round_box

    gyroid_density = _compute_sign_change_density(_sdf_gyroid, 24, -1.5, 1.5)
    round_box_density = _compute_sign_change_density(_sdf_round_box, 24, -1.5, 1.5)

    assert round_box_density > 0, "round_box density should be positive"
    ratio = gyroid_density / round_box_density
    assert ratio > 5.0, (
        f"Expected gyroid/round_box density ratio > 5x for discrimination; "
        f"got {ratio:.1f}x (gyroid={gyroid_density:.4f}, round_box={round_box_density:.4f})"
    )


# ---------------------------------------------------------------------------
# Mocked topology score: compute_topology_score with patched compile_and_run
# ---------------------------------------------------------------------------


def _mock_executor(max_memory_mb: int = 256) -> MagicMock:
    """Return a minimal mock SandboxedExecutor."""
    executor = MagicMock()
    executor.max_memory_mb = max_memory_mb
    return executor


def _make_compile_and_run_mock(density: float):
    """Return a mock that pretends compile_and_run succeeded with given density."""
    from autobench.core import Verdict

    sign_changes = int(density * 3 * 24 ** 3)
    total = 3 * 24 ** 3
    stdout = json.dumps({
        "sign_changes": sign_changes,
        "total": total,
        "density": density,
    })

    def _mock(source, lang, constraints, stdin, executor):  # noqa: ARG001
        return (stdout, Verdict.OK, 0.0)

    return _mock


def test_topology_score_discriminates():
    """Gyroid density → high score vs gyroid target, near-zero score vs round_box target.

    Patches compile_and_run with gyroid-like density (~0.157).
    Asserts that:
      - score vs gyroid instance >> score vs round_box instance
    """
    from autobench.sdf_kernel import (
        compute_topology_score,
        generate_instance,
        TOPO_TARGETS,
    )

    gyroid_density = 0.157  # measured value close to gyroid target 0.178

    gyroid_inst = generate_instance("gyroid")
    round_box_inst = generate_instance("round_box")
    executor = _mock_executor()

    dummy_sdf = 'extern "C" float sdf(float x, float y, float z) { return x; }'

    with patch(
        "autobench.sdf_kernel.compile_and_run",
        side_effect=_make_compile_and_run_mock(gyroid_density),
    ):
        score_vs_gyroid, density_vs_gyroid = compute_topology_score(
            dummy_sdf, gyroid_inst, executor
        )
        score_vs_round_box, density_vs_round_box = compute_topology_score(
            dummy_sdf, round_box_inst, executor
        )

    # Gyroid density should score well vs gyroid target
    assert score_vs_gyroid > 0.7, (
        f"Score vs gyroid target should be high for gyroid density, got {score_vs_gyroid:.4f}"
    )
    # Same density should score near-zero vs round_box target (9x too high)
    assert score_vs_round_box < 0.01, (
        f"Score vs round_box target should be near-zero for gyroid density, "
        f"got {score_vs_round_box:.6f}"
    )
    # Discriminative ratio must be large
    assert score_vs_gyroid > 100 * score_vs_round_box or score_vs_round_box < 1e-6, (
        f"Topology oracle barely discriminates: "
        f"gyroid_score={score_vs_gyroid:.4f} round_box_score={score_vs_round_box:.6f}"
    )


# ---------------------------------------------------------------------------
# Island routing integration test
# ---------------------------------------------------------------------------


def test_sdf_island_routing():
    """With island_instance_assignment=True, island 0 → gyroid, island 1 → round_box.

    Patches ensure_executor to bypass the gVisor gate, then replaces
    evaluate_candidate with a spy that records which instance was used.
    """
    from autobench.kernel_base import KernelConfig, CandidateProgram

    dummy_executor = _mock_executor()

    with patch("autobench.sdf_kernel.ensure_executor", return_value=dummy_executor):
        from autobench.sdf_kernel import SDFKernel

        cfg = KernelConfig(
            instances=["gyroid", "round_box"],
            n_islands=2,
            allow_unsandboxed=True,
        )
        kernel = SDFKernel(cfg)

    instance_names_seen: list[str] = []

    original_evaluate = kernel.evaluate_candidate

    def spy_evaluate(code, instance):  # noqa: ARG001
        instance_names_seen.append(instance.name)
        return 0.5  # dummy fitness

    kernel.evaluate_candidate = spy_evaluate  # type: ignore[method-assign]

    # Island 0 should route to instances[0] = gyroid
    prog0 = CandidateProgram(id="p0", code="x", island=0, generation=0)
    kernel.evaluate_fitness(prog0)

    # Island 1 should route to instances[1] = round_box
    prog1 = CandidateProgram(id="p1", code="x", island=1, generation=0)
    kernel.evaluate_fitness(prog1)

    assert len(instance_names_seen) == 2
    assert instance_names_seen[0] == "gyroid", (
        f"Island 0 should evaluate gyroid, got '{instance_names_seen[0]}'"
    )
    assert instance_names_seen[1] == "round_box", (
        f"Island 1 should evaluate round_box, got '{instance_names_seen[1]}'"
    )
