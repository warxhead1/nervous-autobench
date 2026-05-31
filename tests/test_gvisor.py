"""End-to-end tests for the gVisor sandbox path in autobench.

These tests exercise SandboxedExecutor with sandbox_type="gvisor" and the
helpers it relies on. Tests that require a working runsc binary are
skipped when /usr/local/bin/runsc (or /usr/bin/runsc) is not present so
the suite stays green on developer machines without gVisor installed.
"""

from __future__ import annotations

import os

import pytest

from autobench.sandbox import SandboxedExecutor
from autobench.core import Verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runsc_path() -> str | None:
    for p in ("/usr/local/bin/runsc", "/usr/bin/runsc"):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


RUNSC = _runsc_path()
requires_runsc = pytest.mark.skipif(
    RUNSC is None,
    reason="runsc not installed; skipping gVisor end-to-end tests",
)


# ---------------------------------------------------------------------------
# Unit tests — do not need runsc to actually run
# ---------------------------------------------------------------------------

class TestFindAndCheck:
    def test_find_runsc_finds_binary_when_present(self):
        """_find_runsc returns the path when runsc is installed."""
        if RUNSC is None:
            pytest.skip("runsc not installed")
        executor = SandboxedExecutor(sandbox_type="subprocess")
        found = executor._find_runsc()
        assert found == RUNSC

    def test_find_runsc_returns_none_when_path_missing(self, monkeypatch):
        """_find_runsc returns None when neither candidate path exists."""
        executor = SandboxedExecutor(sandbox_type="subprocess")
        # Force isfile to always return False so _find_runsc cannot resolve.
        import autobench.sandbox as sb

        monkeypatch.setattr(sb.os.path, "isfile", lambda _p: False)
        assert executor._find_runsc() is None

    def test_gvisor_cmd_prefix_shape(self):
        """The shared cmd prefix has the expected flags and ends with `do --`."""
        if RUNSC is None:
            pytest.skip("runsc not installed")
        executor = SandboxedExecutor(sandbox_type="subprocess")
        executor._runsc_path = RUNSC
        prefix = executor._gvisor_cmd_prefix()
        assert prefix[0] == RUNSC
        assert "--rootless" in prefix
        assert "--network=none" in prefix
        assert "--ignore-cgroups" in prefix
        assert "--platform=ptrace" in prefix
        assert prefix[-2:] == ["do", "--"]


@requires_runsc
class TestCheckGvisor:
    def test_check_gvisor_healthy(self):
        """check_gvisor returns (True, version, []) when runsc is healthy."""
        executor = SandboxedExecutor(sandbox_type="subprocess")
        executor._runsc_path = RUNSC
        available, version, issues = executor.check_gvisor()
        assert available is True, f"expected healthy gVisor, got issues={issues}"
        assert version  # non-empty
        assert issues == []

    def test_check_gvisor_python_subtest(self):
        """_test_gvisor_python independently passes when runsc + python3 work."""
        executor = SandboxedExecutor(sandbox_type="subprocess")
        executor._runsc_path = RUNSC
        ok, err = executor._test_gvisor_python()
        assert ok, f"python-under-runsc failed: {err}"


# ---------------------------------------------------------------------------
# End-to-end execution tests — need a real runsc
# ---------------------------------------------------------------------------

@requires_runsc
class TestGvisorEndToEnd:
    def test_simple_python_prints(self):
        """A trivial print(42) returns OK with '42' on stdout."""
        executor = SandboxedExecutor(sandbox_type="gvisor")
        # Sanity: gVisor should be the active sandbox (no fallback).
        assert executor.sandbox_type == "gvisor", (
            "Expected gVisor sandbox to be active; fell back to "
            f"{executor.sandbox_type}"
        )
        result = executor.execute("print(42)", "python")
        assert result.verdict == Verdict.OK, (result.verdict, result.stderr)
        assert "42" in result.stdout

    def test_runtime_error_returns_re(self):
        """A program that raises returns RE verdict."""
        executor = SandboxedExecutor(sandbox_type="gvisor")
        code = "raise ValueError('boom')"
        result = executor.execute(code, "python")
        assert result.verdict == Verdict.RE, (result.verdict, result.stderr)
        assert "ValueError" in result.stderr or "Traceback" in result.stderr

    def test_sleeping_program_triggers_tle(self):
        """A program that sleeps past max_time_seconds returns TLE."""
        executor = SandboxedExecutor(sandbox_type="gvisor")
        code = "import time\ntime.sleep(10)\nprint('done')\n"
        result = executor.execute(
            code,
            "python",
            constraints={"max_time_seconds": 2},
        )
        assert result.verdict == Verdict.TLE, (result.verdict, result.stderr)

    def test_network_is_blocked(self):
        """socket.create_connection to a public IP fails inside the sandbox."""
        executor = SandboxedExecutor(sandbox_type="gvisor")
        code = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
            "    print('CONNECTED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED:', e)\n"
        )
        result = executor.execute(
            code,
            "python",
            constraints={"max_time_seconds": 10},
        )
        # The program itself should complete (verdict OK) but report BLOCKED.
        assert "CONNECTED" not in result.stdout, (
            "Network access was NOT blocked by gVisor "
            f"(stdout={result.stdout!r})"
        )
        assert "BLOCKED" in result.stdout, (
            f"Expected network error inside sandbox, got stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
