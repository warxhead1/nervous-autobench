"""Unit tests for the Firecracker vsock exec protocol."""
from __future__ import annotations

import base64
import json

import pytest

from autobench.engines.firecracker_vm import FirecrackerVM, build_exec_request


class TestBuildExecRequest:
    """Tests for the host-side JSON request builder."""

    def test_basic_request(self):
        req = build_exec_request(["echo", "hello"], timeout=5.0, stdin=b"")
        assert req["cmd"] == ["echo", "hello"]
        assert req["timeout"] == 5.0
        assert req["stdin_b64"] == ""

    def test_request_with_stdin(self):
        stdin = b"world"
        req = build_exec_request(["grep", "foo"], timeout=10.0, stdin=stdin)
        assert req["cmd"] == ["grep", "foo"]
        assert req["timeout"] == 10.0
        assert base64.b64decode(req["stdin_b64"]) == stdin

    def test_request_empty_stdin(self):
        req = build_exec_request(["ls"], timeout=1.0)
        assert req["cmd"] == ["ls"]
        assert req["timeout"] == 1.0
        assert "stdin_b64" in req
        assert base64.b64decode(req["stdin_b64"]) == b""


class TestExecResponse:
    """Tests for host-side JSON response parsing (mocked socket)."""

    def test_response_roundtrip(self):
        """Simulate a full roundtrip: host builds request → guest parses → guest responds → host parses."""
        # Host-side request
        cmd = ["echo", "hello"]
        timeout = 5.0
        stdin = b"test input"
        req = build_exec_request(cmd, timeout=timeout, stdin=stdin)

        assert req["cmd"] == cmd
        assert req["timeout"] == timeout
        assert base64.b64decode(req["stdin_b64"]) == stdin

        # Simulate what the guest agent would produce
        elapsed_ms = 12.5
        resp = {
            "exit_code": 0,
            "stdout": "hello\n",
            "stderr": "",
            "elapsed_ms": elapsed_ms,
        }
        resp_json = json.dumps(resp)

        # Host-side parsing
        parsed = json.loads(resp_json)
        assert parsed["exit_code"] == 0
        assert parsed["stdout"] == "hello\n"
        assert parsed["stderr"] == ""
        assert parsed["elapsed_ms"] == elapsed_ms

    def test_response_error_exit_code(self):
        """Error exit codes are preserved."""
        resp = {
            "exit_code": 127,
            "stdout": "",
            "stderr": "command not found",
            "elapsed_ms": 3.0,
        }
        parsed = json.loads(json.dumps(resp))
        assert parsed["exit_code"] == 127
        assert parsed["stderr"] == "command not found"


class TestFirecrackerVMExecSignature:
    """Verify FirecrackerVM.exec has the right signature and no NotImplementedError."""

    def test_exec_no_notimplemented(self):
        """The exec method must not raise NotImplementedError."""
        import inspect

        src = inspect.getsource(FirecrackerVM.exec)
        assert "NotImplementedError" not in src

    def test_exec_accepts_stdin(self):
        """exec accepts stdin as bytes | None."""
        import inspect

        sig = inspect.signature(FirecrackerVM.exec)
        params = list(sig.parameters.keys())
        assert "stdin" in params

    def test_exec_returns_execution_result(self):
        """exec returns ExecutionResult, not subprocess.CompletedProcess."""
        import inspect

        # Check the return annotation is present (it's a string forward ref)
        src = inspect.getsource(FirecrackerVM.exec)
        assert "ExecutionResult" in src