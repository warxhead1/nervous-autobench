"""Firecracker microVM integration for autobench sandbox.

Provides:
    FirecrackerAPI   — HTTP/1.1 client over the Firecracker unix socket
    FirecrackerVM    — single VM lifecycle: configure → start → exec → stop → delete
    FirecrackerPool  — pre-warmed pool for low-cold-start sandboxed exec

Firecracker control plane (real API, per upstream OpenAPI spec):
    PUT  /boot-source                  — kernel + boot_args
    PUT  /drives/{drive_id}            — block devices (rootfs etc.)
    PUT  /machine-config               — vCPUs + memory
    PUT  /vsock                        — vhost-vsock device (host UDS + guest CID)
    POST /actions  {InstanceStart}     — boot the VM
    POST /actions  {SendCtrlAltDel}    — graceful shutdown
    GET  /machine-config               — health probe
    GET  /                             — instance info

The old `/execute` and `/root-volume` endpoints in the previous revision do
not exist in Firecracker. Guest command execution is done over vsock to a
small in-guest agent (or serial console / network) — Firecracker itself
provides no guest-exec API.

Artifact URLs verified 2026-05-15:
    Kernel:  https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/x86_64/kernels/vmlinux.bin
    Rootfs:  https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/x86_64/rootfs/bionic.rootfs.ext4

The older `v1.x/latest/{kernels,roots}/...` paths return 404. CI images at
`firecracker-ci/v1.10/x86_64/` are also available but the rootfs there is
a squashfs (read-only) — we want a writable ext4, so we stick with the
quickstart artifacts.

Refs:
    https://github.com/firecracker-microvm/firecracker/blob/main/src/firecracker/swagger/firecracker.yaml
    https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md
    https://github.com/firecracker-microvm/firecracker/blob/main/docs/vsock.md
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sandbox import ExecutionResult

logger = logging.getLogger(__name__)


def build_exec_request(cmd: list[str], timeout: float, stdin: bytes | None = None) -> dict:
    """Build a JSON request dict for the in-guest agent over vsock."""
    return {
        "cmd": cmd,
        "stdin_b64": base64.b64encode(stdin or b"").decode("ascii"),
        "timeout": timeout,
    }

# Verified-working artifact URLs (see module docstring).
DEFAULT_KERNEL_URL = (
    "https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/x86_64/kernels/vmlinux.bin"
)
DEFAULT_ROOTFS_URL = (
    "https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/x86_64/rootfs/bionic.rootfs.ext4"
)
DEFAULT_VM_MEMORY_MB = 512
DEFAULT_VM_VCPUS = 2
POOL_SIZE = 4

# Firecracker boot args for serial console + read-write rootfs.
DEFAULT_BOOT_ARGS = (
    "console=ttyS0 reboot=k panic=1 pci=off "
    "i8042.noaux i8042.nomux i8042.nopnp i8042.dumbkbd"
)

# Vsock guest CID. CID 2 = host, CID 3+ = guests. We use 3 for the single VM.
DEFAULT_GUEST_CID = 3


class FirecrackerError(RuntimeError):
    """Raised when the Firecracker API returns a non-2xx response."""


class FirecrackerAPI:
    """HTTP/1.1 client over the Firecracker unix-socket control API.

    Parses status line + headers + body. Raises FirecrackerError on non-2xx.
    Returns parsed JSON on 2xx with a body, or {} on 204 / empty body.
    """

    def __init__(self, socket_path: str, timeout: float = 10.0) -> None:
        self.socket_path = socket_path
        self.timeout = timeout

    # --- public verb helpers ---------------------------------------------

    def get(self, path: str) -> dict[str, Any]:
        return self._do("GET", path, None)

    def put(self, path: str, body: Any | None = None) -> dict[str, Any]:
        return self._do("PUT", path, body)

    def patch(self, path: str, body: Any | None = None) -> dict[str, Any]:
        return self._do("PATCH", path, body)

    def post(self, path: str, body: Any | None = None) -> dict[str, Any]:
        return self._do("POST", path, body)

    # --- internals --------------------------------------------------------

    def _do(self, method: str, path: str, body: Any | None) -> dict[str, Any]:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
            body_bytes = b""
            headers = [f"{method} {path} HTTP/1.1", "Host: localhost", "Accept: application/json"]
            if body is not None:
                body_bytes = json.dumps(body).encode("utf-8")
                headers.append("Content-Type: application/json")
                headers.append(f"Content-Length: {len(body_bytes)}")
            else:
                headers.append("Content-Length: 0")
            request = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii")
            sock.sendall(request + body_bytes)
            raw = self._recv_response(sock)
        finally:
            sock.close()

        return self._parse_response(method, path, raw)

    @staticmethod
    def _recv_response(sock: socket.socket) -> bytes:
        # Read until we have full headers, then read Content-Length bytes.
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
        header_end = buf.find(b"\r\n\r\n")
        if header_end == -1:
            return bytes(buf)
        header_blob = bytes(buf[:header_end])
        body_start = header_end + 4
        # Parse Content-Length if present.
        content_length: int | None = None
        for line in header_blob.split(b"\r\n")[1:]:
            if line.lower().startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                except Exception:
                    content_length = None
                break
        if content_length is None:
            # Drain whatever else is available without blocking too long.
            sock.settimeout(0.2)
            try:
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
            except (socket.timeout, OSError):
                pass
            return bytes(buf)
        while len(buf) - body_start < content_length:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    @staticmethod
    def _parse_response(method: str, path: str, raw: bytes) -> dict[str, Any]:
        if not raw:
            raise FirecrackerError(f"{method} {path}: empty response from firecracker")
        # Split status line / headers / body.
        try:
            header_end = raw.index(b"\r\n\r\n")
        except ValueError:
            raise FirecrackerError(f"{method} {path}: malformed response (no header terminator)")
        header_blob = raw[:header_end].decode("iso-8859-1", errors="replace")
        body_blob = raw[header_end + 4:]
        status_line, _, _ = header_blob.partition("\r\n")
        # "HTTP/1.1 204 No Content"
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise FirecrackerError(f"{method} {path}: bad status line {status_line!r}")
        status = int(parts[1])
        if status >= 400:
            detail = body_blob.decode("utf-8", errors="replace").strip()
            raise FirecrackerError(f"{method} {path}: HTTP {status}: {detail}")
        if not body_blob.strip():
            return {}
        text = body_blob.decode("utf-8", errors="replace").strip()
        # Firecracker may return non-JSON bodies for some GETs; tolerate it.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}


class FirecrackerVM:
    """Single Firecracker microVM.

    Lifecycle:
        __init__ → configure() → start() → [exec()] → stop() → delete()

    `configure()` spawns `firecracker --api-sock <sock>` and then issues:
        PUT /boot-source, PUT /drives/rootfs, PUT /machine-config, PUT /vsock

    `start()` issues POST /actions InstanceStart.

    `exec()` is currently a TODO — Firecracker has no guest-exec API. The
    canonical solution is a tiny init-system inside the guest that listens on
    a vsock port and forks commands. We wire up the vsock device in
    configure(), but the guest-side agent is not part of this bead.
    """

    def __init__(
        self,
        vm_id: str,
        kernel_image: str | None = None,
        rootfs: str | None = None,
        memory_mb: int = DEFAULT_VM_MEMORY_MB,
        vcpus: int = DEFAULT_VM_VCPUS,
        guest_cid: int = DEFAULT_GUEST_CID,
        boot_args: str = DEFAULT_BOOT_ARGS,
    ) -> None:
        self.vm_id = vm_id
        self.kernel_image = kernel_image
        self.rootfs = rootfs
        self.memory_mb = memory_mb
        self.vcpus = vcpus
        self.guest_cid = guest_cid
        self.boot_args = boot_args
        self.socket_path = f"/tmp/firecracker-{vm_id}.sock"
        self.vsock_uds_path = f"/tmp/firecracker-{vm_id}.vsock"
        self.jailer_dir = f"/var/run/firecracker/{vm_id}"
        self.state = "stopped"
        self._api: FirecrackerAPI | None = None
        self._process: subprocess.Popen | None = None
        self._resolved_kernel: str | None = None
        self._resolved_rootfs: str | None = None

    # --- artifact resolution ---------------------------------------------

    def _ensure_firecracker_binary(self) -> str:
        for path in ("/usr/local/bin/firecracker", "/usr/bin/firecracker", "/opt/firecracker/firecracker"):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        raise RuntimeError(
            "firecracker binary not found. Install from "
            "https://github.com/firecracker-microvm/firecracker/releases"
        )

    @staticmethod
    def _looks_like_real_artifact(path: Path, min_bytes: int = 1_000_000) -> bool:
        """Sanity-check: a real kernel/rootfs is multi-MB and isn't ASCII XML."""
        if not path.is_file():
            return False
        try:
            if path.stat().st_size < min_bytes:
                return False
            with open(path, "rb") as f:
                head = f.read(16)
            # An S3 404 starts with "<?xml" or "<Error>".
            if head.startswith(b"<?xml") or head.startswith(b"<Error"):
                return False
            return True
        except OSError:
            return False

    def _ensure_artifact(self, supplied: str | None, url: str, name: str, min_bytes: int) -> str:
        if supplied and os.path.isfile(supplied):
            return supplied
        dest = Path(tempfile.gettempdir()) / f"firecracker-{name}"
        if self._looks_like_real_artifact(dest, min_bytes=min_bytes):
            return str(dest)
        # Stale / bad cached file — remove and re-fetch.
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        logger.info("Downloading Firecracker %s from %s -> %s", name, url, dest)
        try:
            subprocess.run(
                ["curl", "-fsSL", url, "-o", str(dest)],
                check=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"failed to download {name} from {url}: {e}") from e
        if not self._looks_like_real_artifact(dest, min_bytes=min_bytes):
            raise RuntimeError(
                f"downloaded {name} at {dest} does not look like a valid artifact "
                f"(too small or XML error body). URL may be stale: {url}"
            )
        return str(dest)

    def _ensure_kernel(self) -> str:
        return self._ensure_artifact(self.kernel_image, DEFAULT_KERNEL_URL, "vmlinux.bin", 4_000_000)

    def _ensure_rootfs(self) -> str:
        return self._ensure_artifact(self.rootfs, DEFAULT_ROOTFS_URL, "rootfs.ext4", 50_000_000)

    # --- lifecycle --------------------------------------------------------

    def configure(self) -> None:
        """Spawn firecracker process and push the full VM config via the API."""
        self._resolved_kernel = self._ensure_kernel()
        self._resolved_rootfs = self._ensure_rootfs()
        os.makedirs(self.jailer_dir, exist_ok=True)
        fc_path = self._ensure_firecracker_binary()

        # Clean stale socket — firecracker refuses to bind over an existing one.
        for stale in (self.socket_path, self.vsock_uds_path):
            try:
                os.unlink(stale)
            except FileNotFoundError:
                pass

        # Launch firecracker with just --api-sock; config comes via the socket.
        cmd = [fc_path, "--api-sock", self.socket_path, "--id", self.vm_id]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the API socket to appear.
        for _ in range(50):
            if os.path.exists(self.socket_path):
                break
            if self._process.poll() is not None:
                err = (self._process.stderr.read() or b"").decode("utf-8", errors="replace")
                raise RuntimeError(f"firecracker exited before opening api socket: {err}")
            time.sleep(0.05)
        else:
            raise RuntimeError(f"Firecracker socket {self.socket_path} not created within 2.5s")

        self._api = FirecrackerAPI(self.socket_path)

        # boot-source
        self._api.put(
            "/boot-source",
            {
                "kernel_image_path": self._resolved_kernel,
                "boot_args": self.boot_args,
            },
        )
        # drives — rootfs at /drives/rootfs
        self._api.put(
            "/drives/rootfs",
            {
                "drive_id": "rootfs",
                "path_on_host": self._resolved_rootfs,
                "is_root_device": True,
                "is_read_only": False,
            },
        )
        # machine-config
        self._api.put(
            "/machine-config",
            {
                "vcpu_count": self.vcpus,
                "mem_size_mib": self.memory_mb,
                "smt": False,
            },
        )
        # vsock — host UDS, guest CID 3. Not all kernels expose vhost-vsock,
        # so this is best-effort: we log and continue on failure.
        try:
            self._api.put(
                "/vsock",
                {
                    "guest_cid": self.guest_cid,
                    "uds_path": self.vsock_uds_path,
                    "vsock_id": "vsock0",
                },
            )
        except FirecrackerError as e:
            logger.warning("vsock configuration failed (continuing without guest-exec): %s", e)

        self.state = "configured"

    def start(self) -> None:
        if self.state == "running":
            return
        if self._api is None:
            self.configure()
        assert self._api is not None
        # Firecracker accepts both PUT and POST for /actions per the OpenAPI
        # spec, but v1.15.1 returns a misleading "Invalid HTTP Method" for
        # POST /actions after vsock setup. PUT works in all observed cases.
        self._api.put("/actions", {"action_type": "InstanceStart"})
        self.state = "running"

    def health_check(self) -> bool:
        """Return True if the firecracker API is responsive."""
        if self._api is None:
            return False
        try:
            self._api.get("/machine-config")
            return True
        except Exception:
            return False

    def exec(
        self, cmd: list[str], timeout: float = 30.0, stdin: bytes | None = None
    ) -> "ExecutionResult":
        """Execute a command inside the guest via vsock.

        Connects to the in-guest agent on vsock port 8888 and speaks a
        simple JSON-RPC-like protocol:
          Request:  {"cmd": [...], "stdin_b64": "...", "timeout": float}
          Response: {"exit_code": int, "stdout": str, "stderr": str, "elapsed_ms": float}
        """
        # Import ExecutionResult here to avoid circular import at module level.
        from .sandbox import ExecutionResult, Verdict

        vsock_port = 8888
        sock = socket.socket(40, socket.SOCK_STREAM)  # AF_VSOCK = 40
        sock.settimeout(max(timeout + 5.0, 35.0))
        try:
            sock.connect((self.guest_cid, vsock_port))
            req = {
                "cmd": cmd,
                "stdin_b64": base64.b64encode(stdin or b"").decode("ascii"),
                "timeout": timeout,
            }
            sock.sendall((json.dumps(req) + "\n").encode("utf-8"))

            # Read until newline.
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if data.endswith(b"\n"):
                    break

            if not data:
                raise FirecrackerError("vsock exec: empty response from guest agent")

            resp = json.loads(data.decode("utf-8"))
            elapsed_ms = resp.get("elapsed_ms", 0.0)
            exit_code = resp.get("exit_code", -1)
            verdict = Verdict.OK if exit_code == 0 else Verdict.ERROR
            return ExecutionResult(
                stdout=resp.get("stdout", ""),
                stderr=resp.get("stderr", ""),
                exit_code=exit_code,
                latency_ms=elapsed_ms,
                verdict=verdict,
            )
        finally:
            sock.close()

    def stop(self) -> None:
        """Graceful shutdown via SendCtrlAltDel, then kill the process."""
        if self._api is not None and self.state == "running":
            try:
                self._api.put("/actions", {"action_type": "SendCtrlAltDel"})
            except Exception as e:
                logger.debug("SendCtrlAltDel failed for %s: %s", self.vm_id, e)
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        self.state = "stopped"
        self._api = None

    def delete(self) -> None:
        self.stop()
        for path in (self.socket_path, self.vsock_uds_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        if os.path.isdir(self.jailer_dir):
            shutil.rmtree(self.jailer_dir, ignore_errors=True)


class FirecrackerPool:
    """Pool of pre-warmed Firecracker VMs.

    Gracefully degrades to an empty pool if /dev/kvm is unavailable OR if
    artifact downloads fail — callers should treat `has_kvm` and
    `len(pool._available)` as independent signals.
    """

    def __init__(
        self,
        pool_size: int = POOL_SIZE,
        kernel_image: str | None = None,
        rootfs: str | None = None,
        memory_mb: int = DEFAULT_VM_MEMORY_MB,
        vcpus: int = DEFAULT_VM_VCPUS,
    ) -> None:
        self.pool_size = pool_size
        self.kernel_image = kernel_image
        self.rootfs = rootfs
        self.memory_mb = memory_mb
        self.vcpus = vcpus
        self._available: list[FirecrackerVM] = []
        self._acquired: dict[str, FirecrackerVM] = {}
        self._vm_counter = 0
        self._has_kvm = os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK)

        if not self._has_kvm:
            logger.warning("FirecrackerPool: /dev/kvm not available; pool will be empty.")
            return

        for _ in range(pool_size):
            try:
                vm = self._create_vm()
                self._available.append(vm)
            except Exception as e:
                logger.warning("FirecrackerPool: failed to pre-warm VM: %s", e)

    def _create_vm(self) -> FirecrackerVM:
        self._vm_counter += 1
        vm = FirecrackerVM(
            vm_id=f"autobench-fc-{os.getpid()}-{self._vm_counter}",
            kernel_image=self.kernel_image,
            rootfs=self.rootfs,
            memory_mb=self.memory_mb,
            vcpus=self.vcpus,
            # Each VM needs its own guest CID. CID 2 reserved for host.
            guest_cid=DEFAULT_GUEST_CID + self._vm_counter,
        )
        vm.configure()
        vm.start()
        return vm

    def acquire(self, timeout: float = 30.0) -> FirecrackerVM:
        if not self._has_kvm:
            raise RuntimeError("FirecrackerPool: KVM not available on this host")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._available:
                vm = self._available.pop()
                self._acquired[vm.vm_id] = vm
                return vm
            if len(self._acquired) < self.pool_size:
                try:
                    vm = self._create_vm()
                    self._acquired[vm.vm_id] = vm
                    return vm
                except Exception as e:
                    logger.debug("on-demand VM create failed: %s", e)
            time.sleep(0.1)
        # Last-ditch: try one more spawn even past the pool size.
        vm = self._create_vm()
        self._acquired[vm.vm_id] = vm
        return vm

    def release(self, vm: FirecrackerVM) -> None:
        self._acquired.pop(vm.vm_id, None)
        if not self._has_kvm:
            return
        if vm.state == "running" and vm.health_check():
            self._available.append(vm)
            return
        # VM is sick — kill it and try to brew a replacement.
        try:
            vm.delete()
        except Exception:
            pass
        try:
            self._available.append(self._create_vm())
        except Exception as e:
            logger.warning("FirecrackerPool: failed to replace released VM: %s", e)

    def shutdown(self) -> None:
        for vm in list(self._available) + list(self._acquired.values()):
            try:
                vm.delete()
            except Exception:
                pass
        self._available.clear()
        self._acquired.clear()

    @property
    def has_kvm(self) -> bool:
        return self._has_kvm
