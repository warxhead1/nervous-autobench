"""GPU result publisher for autobench.

Publishes GPUResult events to the nervous-bus via zellij pipes,
with a debug JSONL fallback when the pipe is unavailable.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .gpu_types import GPUResult

DEBUG_CACHE = Path.home() / ".cache" / "nervous-bus"
DEBUG_FILE = DEBUG_CACHE / "debug.jsonl"


def _ensure_debug_dir() -> None:
    DEBUG_CACHE.mkdir(parents=True, exist_ok=True)


class GPUResultPublisher:
    """Publishes GPUResult events to the nervous-bus autobench.gpu_result channel.

    Uses `zellij pipe -p nervous-bus -n autobench.gpu_result.v1` to emit
    CloudEvents-lite JSONL. Falls back to writing debug.jsonl if the pipe
    is unavailable.
    """

    def __init__(self) -> None:
        self._pipe_proc: Any = None

    def _try_zellij_pipe(self, payload: str) -> bool:
        """Attempt to write to zellij pipe. Returns True on success."""
        try:
            proc = subprocess.run(
                ["zellij", "pipe", "-p", "nervous-bus", "-n", "autobench.gpu_result.v1", "--"],
                input=payload.encode(),
                timeout=5,
                capture_output=True,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _write_debug(self, payload: str) -> None:
        """Append payload to the debug fallback file."""
        _ensure_debug_dir()
        with open(DEBUG_FILE, "a") as fh:
            fh.write(payload + "\n")

    def publish(self, result: GPUResult) -> bool:
        """Publish a GPUResult to the bus.

        Args:
            result: A GPUResult instance.

        Returns:
            True if published successfully (via zellij or debug fallback).
        """
        event = result.to_event()
        payload = json.dumps(event)

        if self._try_zellij_pipe(payload):
            return True

        # Fallback: write to debug file
        self._write_debug(payload)
        return True

    def close(self) -> None:
        """Close the publisher (no-op for zellij pipe, which is fire-and-forget)."""
        pass