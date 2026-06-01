"""Tests for the TSP kernel's sandbox autonomy gate + sandboxed evaluation.

Scope: the kernel's *additions* on top of autobench.sandbox — namely
  (1) ensure_sandboxed_executor refuses to hand back a non-isolating executor
      unless allow_unsandboxed is explicitly set, and
  (2) a pure-C++ baseline candidate compiles+runs through the executor and
      yields real, positive fitness (the end-to-end go/no-go).

Isolation properties (network/fs/timeout) are covered by test_sandbox*.py and
are intentionally not re-tested here.
"""

from __future__ import annotations

import shutil

import pytest

from autobench.tsp_kernel import (
    UnsafeSandboxError,
    ensure_sandboxed_executor,
    evaluate_on_instance,
    fetch_tsplib_instance,
    init_baseline_programs,
    KNOWN_OPTIMALS,
)
from autobench.engines.sandbox import SandboxedExecutor

_HAVE_GPP = shutil.which("g++") is not None


def _gvisor_available() -> bool:
    ex = SandboxedExecutor(sandbox_type="gvisor")
    return ex.sandbox_type == "gvisor"


class _DegradedExecutor:
    """Stand-in that mimics SandboxedExecutor silently downgrading to subprocess."""

    def __init__(self, sandbox_type="gvisor", max_memory_mb=512, cpu_limit=2):
        self.sandbox_type = "subprocess"  # the dangerous, non-isolating tier
        self.max_memory_mb = max_memory_mb
        self.cpu_limit = cpu_limit


def test_gate_refuses_when_isolation_degraded(monkeypatch):
    # ensure_sandboxed_executor was relocated to autobench.kernels.sandbox in
    # Phase 1 of the autobench restructuring (May 2026). The patch target moved
    # with it; the behaviour under test is unchanged.
    monkeypatch.setattr("autobench.kernels.sandbox.SandboxedExecutor", _DegradedExecutor)
    with pytest.raises(UnsafeSandboxError):
        ensure_sandboxed_executor(allow_unsandboxed=False)


def test_gate_allows_degraded_only_with_explicit_override(monkeypatch):
    monkeypatch.setattr("autobench.kernels.sandbox.SandboxedExecutor", _DegradedExecutor)
    ex = ensure_sandboxed_executor(allow_unsandboxed=True)
    assert ex.sandbox_type == "subprocess"  # override honored, eyes open


@pytest.mark.skipif(not _gvisor_available(), reason="gVisor not available on host")
def test_gate_accepts_isolating_sandbox():
    ex = ensure_sandboxed_executor(allow_unsandboxed=False)
    assert ex.sandbox_type in ("gvisor", "firecracker")


@pytest.mark.skipif(
    not (_HAVE_GPP and _gvisor_available()),
    reason="need g++ and gVisor for end-to-end sandboxed evaluation",
)
def test_baseline_evaluates_through_sandbox():
    """Go/no-go: a baseline compiles on host, runs under gVisor, scores > 0."""
    ex = ensure_sandboxed_executor(allow_unsandboxed=False)
    inst = fetch_tsplib_instance("berlin52")
    inst.optimal_tour_length = KNOWN_OPTIMALS["berlin52"]

    scored = 0
    for baseline in init_baseline_programs(island_id=0, generation=0):
        result = evaluate_on_instance(baseline.priority_code, inst, ex, run_timeout=15.0)
        assert result is not None, f"{baseline.id} failed to run through the sandbox"
        assert result.length > 0
        ratio = inst.optimal_tour_length / result.length
        assert 0.0 < ratio <= 1.0, f"implausible ratio {ratio} for {baseline.id}"
        assert len(result.tour) == inst.n, "tour must visit every node exactly once"
        scored += 1
    assert scored == 3
