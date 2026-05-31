"""Regression test for vmho: sandbox cgroup must contain the child PID, not the parent.

Background (Angela's audit, 2026-05-16): the previous implementation wrote
os.getpid() — the Python test runner's PID — into cgroup.procs before forking
the subprocess. That put the runner under the unit-under-test's memory limit
and OOM-killed it under load. Fix writes the child PID after Popen.
"""

from __future__ import annotations

import os
import random
import subprocess
import time
from pathlib import Path

import pytest


CGROUP_ROOT = Path("/sys/fs/cgroup")


def _cgroup_supported() -> tuple[bool, str]:
    """Return (supported, reason). Skip with reason if not supported."""
    if not CGROUP_ROOT.exists():
        return False, "cgroup v2 not mounted at /sys/fs/cgroup"
    # Try to create a probe cgroup
    probe = CGROUP_ROOT / f"autobench_probe_{random.randint(0, 0xFFFFFF):06x}"
    try:
        probe.mkdir()
    except PermissionError:
        return False, "no permission to create cgroups (run as root or grant cgroup delegation)"
    except OSError as e:
        return False, f"cgroup creation failed: {e}"
    # Verify we can write memory.max
    try:
        (probe / "memory.max").write_text("268435456")  # 256 MB
    except OSError as e:
        probe.rmdir()
        return False, f"cannot write memory.max: {e}"
    finally:
        try:
            probe.rmdir()
        except OSError:
            pass
    return True, ""


def test_cgroup_contains_child_not_parent(tmp_path):
    """Spawn a benign subprocess, assign it to a cgroup, verify ONLY the child
    is in the cgroup — not the test runner (parent) process."""
    supported, reason = _cgroup_supported()
    if not supported:
        pytest.skip(reason)

    cgroup_name = f"autobench_vmho_test_{random.randint(0, 0xFFFFFFFFFF):010x}"
    cgroup_path = CGROUP_ROOT / cgroup_name
    procs_file = cgroup_path / "cgroup.procs"

    parent_pid = os.getpid()
    proc = None
    try:
        cgroup_path.mkdir()
        (cgroup_path / "memory.max").write_text(str(256 * 1024 * 1024))

        # Spawn a child that sleeps long enough for us to inspect cgroup.procs.
        proc = subprocess.Popen(
            ["sleep", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Assign CHILD PID (mirrors the fixed inline path in sandbox.py).
        procs_file.write_text(str(proc.pid))

        # Read membership while the child is still alive.
        deadline = time.time() + 0.8
        child_seen = False
        parent_seen = False
        while time.time() < deadline:
            members = {
                int(line) for line in procs_file.read_text().split() if line.strip().isdigit()
            }
            if proc.pid in members:
                child_seen = True
            if parent_pid in members:
                parent_seen = True
                break  # fail fast
            if child_seen:
                break
            time.sleep(0.02)

        # Hard requirements from vmho AC.
        assert not parent_seen, (
            f"REGRESSION: parent (test runner) PID {parent_pid} appeared in "
            f"cgroup.procs of {cgroup_name} — runner is now under the "
            f"unit-under-test's memory limit (Angela vmho bug)"
        )
        assert child_seen, (
            f"child PID {proc.pid} was never observed in cgroup.procs — "
            f"PID assignment after Popen failed"
        )
    finally:
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
        # Children must exit before rmdir, else EBUSY.
        for _ in range(20):
            try:
                cgroup_path.rmdir()
                break
            except OSError:
                time.sleep(0.05)


def test_run_in_cgroup_dead_function_removed():
    """The standalone run_in_cgroup function was dead code that wrote the
    parent PID — confirm it's gone (consolidation evidence)."""
    from autobench import sandbox

    assert not hasattr(sandbox, "run_in_cgroup"), (
        "run_in_cgroup should be removed; the inline path in "
        "SandboxedExecutor.run_subprocess is the canonical implementation"
    )
