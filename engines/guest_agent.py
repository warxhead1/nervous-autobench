#!/usr/bin/env python3
"""Minimal guest agent — runs inside Firecracker VM, executes commands via vsock."""
import base64
import json
import socket
import subprocess
import time
import sys

VSOCK_PORT = 8888


def handle(conn):
    """Handle a single command execution request from the host."""
    data = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
        if data.endswith(b"\n"):
            break
    if not data:
        return

    try:
        req = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        return

    stdin_bytes = b""
    if req.get("stdin_b64"):
        stdin_bytes = base64.b64decode(req["stdin_b64"])

    start = time.monotonic()
    try:
        result = subprocess.run(
            req["cmd"],
            input=stdin_bytes,
            capture_output=True,
            timeout=req.get("timeout", 30.0),
        )
        exit_code = result.returncode
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        exit_code = 124
        stdout = ""
        stderr = "Timed out after {}s".format(req.get("timeout", 30.0))
    except Exception as e:
        exit_code = 1
        stdout = ""
        stderr = str(e)

    elapsed_ms = (time.monotonic() - start) * 1000
    resp = json.dumps({
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_ms": elapsed_ms,
    }) + "\n"
    conn.sendall(resp.encode())


def main():
    # AF_VSOCK = 40
    sock = socket.socket(40, socket.SOCK_STREAM)
    sock.bind((socket.VMADDR_CID_ANY, VSOCK_PORT))
    sock.listen(8)
    while True:
        conn, _ = sock.accept()
        handle(conn)
        conn.close()


if __name__ == "__main__":
    main()