"""Unit + integration tests for autobench.engines.firecracker_vm.

Unit tests use mocked sockets / mocked FirecrackerAPI — they run without a
firecracker binary, without /dev/kvm, and without network. The single
integration test is marked and skipped unless a real kernel + rootfs are
already present on disk at known paths.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from autobench.engines.firecracker_vm import (
    DEFAULT_GUEST_CID,
    DEFAULT_KERNEL_URL,
    DEFAULT_ROOTFS_URL,
    DEFAULT_VM_MEMORY_MB,
    DEFAULT_VM_VCPUS,
    FirecrackerAPI,
    FirecrackerError,
    FirecrackerPool,
    FirecrackerVM,
)


# --- Fake Firecracker server over unix socket ----------------------------


class _FakeFirecracker:
    """Minimal HTTP/1.1 server over a unix socket that records requests
    and returns canned responses keyed by (method, path)."""

    def __init__(self, socket_path: str, responses: dict[tuple[str, str], tuple[int, dict[str, Any] | None]]):
        self.socket_path = socket_path
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        self._sock.bind(socket_path)
        self._sock.listen(8)
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            try:
                self._handle(conn)
            finally:
                conn.close()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf.extend(chunk)
        header_end = buf.find(b"\r\n\r\n")
        header_blob = bytes(buf[:header_end]).decode("iso-8859-1")
        body_start = header_end + 4
        lines = header_blob.split("\r\n")
        method, path, _ = lines[0].split(" ", 2)
        content_length = 0
        for line in lines[1:]:
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
        while len(buf) - body_start < content_length:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
        body = bytes(buf[body_start:body_start + content_length])
        parsed_body: Any = None
        if body:
            try:
                parsed_body = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                parsed_body = body.decode("utf-8", errors="replace")
        self.requests.append({"method": method, "path": path, "body": parsed_body})
        status, response_body = self.responses.get((method, path), (204, None))
        if response_body is None:
            response_bytes = b""
        else:
            response_bytes = json.dumps(response_body).encode("utf-8")
        reason = {200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found"}.get(status, "Status")
        headers = [
            f"HTTP/1.1 {status} {reason}",
            f"Content-Length: {len(response_bytes)}",
        ]
        if response_bytes:
            headers.append("Content-Type: application/json")
        msg = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + response_bytes
        conn.sendall(msg)

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)
        self._sock.close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


@pytest.fixture
def fake_fc(tmp_path: Path):
    sock_path = str(tmp_path / "fc.sock")
    responses: dict[tuple[str, str], tuple[int, dict[str, Any] | None]] = {
        ("GET", "/machine-config"): (200, {"vcpu_count": 2, "mem_size_mib": 512, "smt": False}),
    }
    server = _FakeFirecracker(sock_path, responses)
    try:
        yield sock_path, server
    finally:
        server.close()


# --- FirecrackerAPI ------------------------------------------------------


class TestFirecrackerAPI:
    def test_get_200_returns_parsed_json(self, fake_fc):
        sock_path, _server = fake_fc
        api = FirecrackerAPI(sock_path, timeout=2.0)
        result = api.get("/machine-config")
        assert result["vcpu_count"] == 2
        assert result["mem_size_mib"] == 512

    def test_put_serializes_json_body(self, fake_fc):
        sock_path, server = fake_fc
        server.responses[("PUT", "/boot-source")] = (204, None)
        api = FirecrackerAPI(sock_path, timeout=2.0)
        api.put("/boot-source", {"kernel_image_path": "/tmp/vmlinux", "boot_args": "console=ttyS0"})
        match = [r for r in server.requests if r["method"] == "PUT" and r["path"] == "/boot-source"]
        assert len(match) == 1
        assert match[0]["body"]["kernel_image_path"] == "/tmp/vmlinux"
        assert match[0]["body"]["boot_args"] == "console=ttyS0"

    def test_204_no_content_returns_empty_dict(self, fake_fc):
        sock_path, server = fake_fc
        server.responses[("PUT", "/drives/rootfs")] = (204, None)
        api = FirecrackerAPI(sock_path, timeout=2.0)
        result = api.put("/drives/rootfs", {"drive_id": "rootfs"})
        assert result == {}

    def test_4xx_raises_firecracker_error(self, fake_fc):
        sock_path, server = fake_fc
        server.responses[("POST", "/actions")] = (400, {"fault_message": "no kernel"})
        api = FirecrackerAPI(sock_path, timeout=2.0)
        with pytest.raises(FirecrackerError) as ei:
            api.post("/actions", {"action_type": "InstanceStart"})
        assert "400" in str(ei.value)
        assert "no kernel" in str(ei.value)

    def test_parses_status_line_correctly(self):
        """Direct unit test of the response parser — no socket needed."""
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 13\r\n"
            b"\r\n"
            b'{"ok": true}\n'
        )
        result = FirecrackerAPI._parse_response("GET", "/", raw)
        assert result == {"ok": True}

    def test_parser_rejects_malformed_response(self):
        with pytest.raises(FirecrackerError):
            FirecrackerAPI._parse_response("GET", "/", b"not-http-at-all")


# --- FirecrackerVM.configure() ------------------------------------------


class TestFirecrackerVMConfigure:
    def test_configure_issues_correct_api_calls(self, tmp_path):
        vm = FirecrackerVM(
            vm_id="test-1",
            kernel_image="/path/to/kernel",
            rootfs="/path/to/rootfs",
            memory_mb=256,
            vcpus=1,
        )
        # Bypass artifact resolution + process spawn.
        vm._ensure_kernel = lambda: "/path/to/kernel"  # type: ignore[assignment]
        vm._ensure_rootfs = lambda: "/path/to/rootfs"  # type: ignore[assignment]
        vm._ensure_firecracker_binary = lambda: "/usr/bin/firecracker"  # type: ignore[assignment]

        mock_api = mock.MagicMock(spec=FirecrackerAPI)

        with mock.patch("autobench.engines.firecracker_vm.subprocess.Popen") as popen, \
             mock.patch("autobench.engines.firecracker_vm.os.path.exists", return_value=True), \
             mock.patch("autobench.engines.firecracker_vm.os.makedirs"), \
             mock.patch("autobench.engines.firecracker_vm.os.unlink"), \
             mock.patch("autobench.engines.firecracker_vm.FirecrackerAPI", return_value=mock_api):
            popen.return_value = mock.MagicMock(poll=lambda: None)
            vm.configure()

        # Pull out the PUT calls and verify each body shape.
        put_calls = {call.args[0]: call.args[1] for call in mock_api.put.call_args_list}

        assert "/boot-source" in put_calls
        assert put_calls["/boot-source"]["kernel_image_path"] == "/path/to/kernel"
        assert "console=ttyS0" in put_calls["/boot-source"]["boot_args"]

        assert "/drives/rootfs" in put_calls
        drive = put_calls["/drives/rootfs"]
        assert drive["drive_id"] == "rootfs"
        assert drive["path_on_host"] == "/path/to/rootfs"
        assert drive["is_root_device"] is True
        assert drive["is_read_only"] is False

        assert "/machine-config" in put_calls
        mc = put_calls["/machine-config"]
        assert mc["vcpu_count"] == 1
        assert mc["mem_size_mib"] == 256
        assert mc["smt"] is False

        assert "/vsock" in put_calls
        vsock = put_calls["/vsock"]
        assert vsock["guest_cid"] == DEFAULT_GUEST_CID
        assert vsock["uds_path"].endswith(".vsock")
        assert vsock["vsock_id"] == "vsock0"

        assert vm.state == "configured"

    def test_start_issues_instance_start(self):
        vm = FirecrackerVM(vm_id="test-2")
        mock_api = mock.MagicMock(spec=FirecrackerAPI)
        vm._api = mock_api
        vm.state = "configured"
        vm.start()
        # Firecracker v1.15.1 accepts PUT or POST on /actions; we use PUT to
        # avoid a known "Invalid HTTP Method" quirk after vsock setup.
        mock_api.put.assert_any_call("/actions", {"action_type": "InstanceStart"})
        assert vm.state == "running"

    def test_stop_sends_ctrl_alt_del(self):
        vm = FirecrackerVM(vm_id="test-3")
        mock_api = mock.MagicMock(spec=FirecrackerAPI)
        vm._api = mock_api
        vm.state = "running"
        vm._process = mock.MagicMock()
        vm._process.wait.return_value = 0
        vm.stop()
        # Verify graceful shutdown was attempted (via PUT /actions).
        assert any(
            call.args == ("/actions", {"action_type": "SendCtrlAltDel"})
            for call in mock_api.put.call_args_list
        )
        assert vm.state == "stopped"

    def test_exec_sends_vsock_request_and_parses_response(self):
        """exec() speaks JSON-RPC over vsock; mock the socket to verify the protocol."""
        vm = FirecrackerVM(vm_id="test-4")
        response_payload = json.dumps({
            "exit_code": 0,
            "stdout": "hello\n",
            "stderr": "",
            "elapsed_ms": 5.0,
        }).encode("utf-8") + b"\n"

        mock_sock = mock.MagicMock()
        mock_sock.recv.return_value = response_payload

        with mock.patch("socket.socket", return_value=mock_sock):
            result = vm.exec(["echo", "hello"])

        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.latency_ms == pytest.approx(5.0)
        # Verify we connected to the right CID and port
        mock_sock.connect.assert_called_once_with((vm.guest_cid, 8888))
        # Verify the request JSON was sent
        sent_bytes = b"".join(
            call.args[0] for call in mock_sock.sendall.call_args_list
        )
        req = json.loads(sent_bytes.decode("utf-8").strip())
        assert req["cmd"] == ["echo", "hello"]

    def test_looks_like_real_artifact_rejects_xml_404(self, tmp_path):
        bad = tmp_path / "vmlinux.bin"
        bad.write_bytes(b'<?xml version="1.0"?><Error>404</Error>')
        assert FirecrackerVM._looks_like_real_artifact(bad, min_bytes=1000) is False

    def test_looks_like_real_artifact_accepts_binary(self, tmp_path):
        good = tmp_path / "vmlinux.bin"
        good.write_bytes(b"\x7fELF" + b"\x00" * 2_000_000)
        assert FirecrackerVM._looks_like_real_artifact(good, min_bytes=1_000_000) is True


# --- FirecrackerPool -----------------------------------------------------


class TestFirecrackerPool:
    def test_has_kvm_true_when_dev_kvm_readable(self):
        with mock.patch("autobench.engines.firecracker_vm.os.path.exists", return_value=True), \
             mock.patch("autobench.engines.firecracker_vm.os.access", return_value=True), \
             mock.patch.object(FirecrackerPool, "_create_vm", side_effect=RuntimeError("no artifacts")):
            pool = FirecrackerPool(pool_size=2)
            assert pool.has_kvm is True
            assert len(pool._available) == 0  # graceful: no artifacts, no VMs, no crash

    def test_pool_fallback_when_kvm_missing(self):
        with mock.patch("autobench.engines.firecracker_vm.os.path.exists", return_value=False):
            pool = FirecrackerPool(pool_size=2)
            assert pool.has_kvm is False
            assert pool._available == []
            with pytest.raises(RuntimeError, match="KVM not available"):
                pool.acquire()

    def test_pool_fallback_when_kvm_not_accessible(self):
        with mock.patch("autobench.engines.firecracker_vm.os.path.exists", return_value=True), \
             mock.patch("autobench.engines.firecracker_vm.os.access", return_value=False):
            pool = FirecrackerPool(pool_size=1)
            assert pool.has_kvm is False
            assert pool._available == []

    def test_release_returns_healthy_vm_to_pool(self):
        with mock.patch("autobench.engines.firecracker_vm.os.path.exists", return_value=True), \
             mock.patch("autobench.engines.firecracker_vm.os.access", return_value=True), \
             mock.patch.object(FirecrackerPool, "_create_vm", side_effect=RuntimeError("no artifacts")):
            pool = FirecrackerPool(pool_size=1)
        vm = mock.MagicMock(spec=FirecrackerVM)
        vm.vm_id = "fake"
        vm.state = "running"
        vm.health_check.return_value = True
        pool.release(vm)
        assert vm in pool._available

    def test_release_discards_sick_vm(self):
        with mock.patch("autobench.engines.firecracker_vm.os.path.exists", return_value=True), \
             mock.patch("autobench.engines.firecracker_vm.os.access", return_value=True), \
             mock.patch.object(FirecrackerPool, "_create_vm", side_effect=RuntimeError("no artifacts")):
            pool = FirecrackerPool(pool_size=1)
        vm = mock.MagicMock(spec=FirecrackerVM)
        vm.vm_id = "fake"
        vm.state = "stopped"
        vm.health_check.return_value = False
        pool.release(vm)
        vm.delete.assert_called_once()
        assert vm not in pool._available


# --- Integration (skipped by default) -----------------------------------


_LOCAL_KERNEL = Path("/tmp/firecracker-vmlinux.bin")
_LOCAL_ROOTFS = Path("/tmp/firecracker-rootfs.ext4")


def _local_artifacts_present() -> bool:
    return (
        FirecrackerVM._looks_like_real_artifact(_LOCAL_KERNEL, min_bytes=4_000_000)
        and FirecrackerVM._looks_like_real_artifact(_LOCAL_ROOTFS, min_bytes=50_000_000)
        and os.path.exists("/dev/kvm")
        and os.access("/dev/kvm", os.R_OK | os.W_OK)
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _local_artifacts_present(),
    reason="Need /dev/kvm + valid kernel + rootfs at /tmp/firecracker-{vmlinux.bin,rootfs.ext4}",
)
def test_integration_real_vm_boots():
    """Actually launch firecracker, boot a VM, probe health, shut down."""
    vm = FirecrackerVM(
        vm_id=f"integration-{os.getpid()}",
        kernel_image=str(_LOCAL_KERNEL),
        rootfs=str(_LOCAL_ROOTFS),
    )
    try:
        vm.configure()
        vm.start()
        assert vm.state == "running"
        assert vm.health_check() is True
    finally:
        vm.delete()


# --- Module-level URL sanity --------------------------------------------


def test_default_artifact_urls_are_documented_constants():
    assert "spec.ccfc.min" in DEFAULT_KERNEL_URL
    assert "spec.ccfc.min" in DEFAULT_ROOTFS_URL
    assert DEFAULT_VM_MEMORY_MB > 0
    assert DEFAULT_VM_VCPUS > 0
