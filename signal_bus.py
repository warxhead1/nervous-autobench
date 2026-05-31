"""Signal bus for autobench verdict events.

Publishes autobench results to the nervous-bus event system via zellij pipes,
and subscribes to receive them for deer-flow integration.

Classes:
    AutobenchResultPublisher — publishes HarnessResult events to autobench.result channel
    AutobenchResultSubscriber — receives result events and calls a callback
    DeerFlowResultSubscriber — subscribes to deer-flow.sandbox.result.v1 verdicts
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# ULID generation (compatible with the nervous shell SDK)
# --------------------------------------------------------------------------- #

def _ulid() -> str:
    """Generate a ULID-like identifier."""
    ts = int(time.time())
    hex_chars = "0123456789ABCDEF"
    import random
    rand_part = "".join(random.choice(hex_chars) for _ in range(16))
    return f"{ts:010d}{rand_part}"


def _iso_now() -> str:
    """Return current UTC time as RFC3339."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# Debug file fallback
# --------------------------------------------------------------------------- #

DEBUG_CACHE = Path.home() / ".cache" / "nervous-bus"
DEBUG_FILE = DEBUG_CACHE / "debug.jsonl"


def _ensure_debug_dir() -> None:
    DEBUG_CACHE.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Publisher
# --------------------------------------------------------------------------- #

class AutobenchResultPublisher:
    """Publishes HarnessResult events to the nervous-bus autobench.result channel.

    Uses `zellij pipe -p nervous-bus -n autobench.result` to emit CloudEvents-lite
    JSONL. Falls back to writing ~/.cache/nervous-bus/debug.jsonl if the pipe
    is unavailable.

    Args:
        harness_version: Version string for the harness configuration.
        benchmark_name: Name of the benchmark suite.
        iteration: RSI iteration number (0 if not running RSI).
    """

    def __init__(
        self,
        harness_version: str = "v0",
        benchmark_name: str = "default",
        iteration: int = 0,
    ) -> None:
        self.harness_version = harness_version
        self.benchmark_name = benchmark_name
        self.iteration = iteration
        self._pipe_proc: Any = None

    def _build_event(self, result: Any) -> dict[str, Any]:
        """Build a CloudEvents-lite envelope around the result data."""
        from .core import HarnessResult
        if not isinstance(result, HarnessResult):
            raise TypeError(f"expected HarnessResult, got {type(result).__name__}")

        problem_id = result.metadata.get("case_id", "unknown")
        error = result.error or ""

        data = {
            "problem_id": problem_id,
            "verdict": result.verdict.value,
            "p_score": result.p_score,
            "p_cost": result.p_cost,
            "p_time": result.p_time,
            "harness_version": self.harness_version,
            "benchmark_name": self.benchmark_name,
            "iteration": self.iteration,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "cost_dollars": result.cost_dollars,
            "error": error,
            "metadata": result.metadata,
        }

        return {
            "id": _ulid(),
            "source": "/autobench/evaluator",
            "type": "autobench.result.v1",
            "datacontenttype": "application/json",
            "time": _iso_now(),
            "data": data,
        }

    def _try_zellij_pipe(self, payload: str) -> bool:
        """Attempt to write to zellij pipe. Returns True on success."""
        try:
            proc = subprocess.run(
                ["zellij", "pipe", "-p", "nervous-bus", "-n", "autobench.result", "--"],
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

    def publish(self, result: Any) -> bool:
        """Publish a HarnessResult to the bus.

        Args:
            result: A HarnessResult instance.

        Returns:
            True if published successfully (via zellij or debug fallback).
        """
        event = self._build_event(result)
        payload = json.dumps(event)

        if self._try_zellij_pipe(payload):
            return True

        # Fallback: write to debug file
        self._write_debug(payload)
        return True  # Still returns True since we wrote to fallback

    def close(self) -> None:
        """Close the publisher (no-op for zellij pipe, which is fire-and-forget)."""
        pass


# --------------------------------------------------------------------------- #
# Deer-flow verdict subscriber
# --------------------------------------------------------------------------- #

class DeerFlowResultSubscriber:
    """Subscribes to deer-flow sandbox verdict events from the nervous-bus.

    Listens on the `deer-flow.sandbox.result.v1` channel (written by deer-flow's
    sandbox tools after each execution). Parses verdict signals and calls
    a callback with structured result data.

    The subscriber tails the debug.jsonl fallback file, filtering for
    deer-flow.sandbox.result.v1 events and extracting verdict/latency/exit data.

    Args:
        callback: Callable[[dict[str, Any]], None] — called for each verdict event.
            The dict contains: thread_id, case_id, verdict, latency_ms,
            exit_code, stderr (truncated to 500), stdout (truncated to 1000).
    """

    CHANNEL = "deer-flow.sandbox.result.v1"

    def __init__(
        self,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.callback = callback or (lambda e: None)
        self._running = False

    def _process_line(self, line: str) -> None:
        """Parse a deer-flow.sandbox.result.v1 line and dispatch to callback."""
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return

        if event.get("type") != self.CHANNEL:
            return

        data = event.get("data", {})
        parsed = {
            "thread_id": data.get("thread_id", "unknown"),
            "case_id": data.get("case_id", "bash"),
            "verdict": data.get("verdict", "OK"),
            "latency_ms": data.get("latency_ms", 0.0),
            "exit_code": data.get("exit_code", 0),
            "stderr": (data.get("stderr") or "")[:500],
            "stdout": (data.get("stdout") or "")[:1000],
        }
        self.callback(parsed)

    def tail(self) -> None:
        """Tail the debug.jsonl file and dispatch deer-flow sandbox events (blocking)."""
        if not DEBUG_FILE.exists():
            _ensure_debug_dir()
            DEBUG_FILE.touch()

        self._running = True
        last_size = DEBUG_FILE.stat().st_size

        while self._running:
            time.sleep(0.5)
            if not DEBUG_FILE.exists():
                continue
            current_size = DEBUG_FILE.stat().st_size
            if current_size > last_size:
                with open(DEBUG_FILE, "r") as fh:
                    fh.seek(last_size)
                    for line in fh:
                        self._process_line(line)
                last_size = current_size

    def stop(self) -> None:
        """Stop the tail loop."""
        self._running = False


# --------------------------------------------------------------------------- #
# Autobench subscriber
# --------------------------------------------------------------------------- #

class AutobenchResultSubscriber:
    """Subscribes to the autobench.result channel and calls a callback for each event.

    Reads events by tailing the debug.jsonl fallback file (for standalone use) OR
    subscribes via the zellij plugin pipe when available. Also emits to
    deer-flow.metaprobe.cycle for deer-flow integration.

    Args:
        callback: Callable[[dict[str, Any]], None] — called for each received event.
        deer_flow_emit: If True, re-emits events to deer-flow.metaprobe.cycle channel.
    """

    CHANNEL = "autobench.result.v1"

    def __init__(
        self,
        callback: Callable[[dict[str, Any]], None] | None = None,
        deer_flow_emit: bool = False,
    ) -> None:
        self.callback = callback or (lambda e: None)
        self.deer_flow_emit = deer_flow_emit
        self._running = False

    def _emit_to_deer_flow(self, event: dict[str, Any]) -> None:
        """Re-emit an autobench result event to deer-flow.metaprobe.cycle."""
        if not self.deer_flow_emit:
            return

        try:
            # Build metaprobe cycle event from the autobench result
            metaprobe_event = {
                "id": _ulid(),
                "source": "/autobench/evaluator",
                "type": "deer-flow.metaprobe.cycle.v1",
                "datacontenttype": "application/json",
                "time": _iso_now(),
                "data": {
                    "trigger": "autobench.result",
                    "autobench_event": event,
                },
            }
            payload = json.dumps(metaprobe_event)
            subprocess.run(
                ["zellij", "pipe", "-p", "nervous-bus", "-n", "deer-flow.metaprobe.cycle", "--"],
                input=payload.encode(),
                timeout=5,
                capture_output=True,
            )
        except Exception:
            # Non-fatal: deer-flow emit is best-effort
            pass

    def _process_line(self, line: str) -> None:
        """Parse and dispatch a single JSONL line."""
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return

        # Only process autobench.result events
        if event.get("type") != self.CHANNEL:
            return

        self.callback(event)
        self._emit_to_deer_flow(event)

    def tail_debug(self) -> None:
        """Tail the debug.jsonl file and dispatch events (blocking)."""
        if not DEBUG_FILE.exists():
            # Create empty file so we can tail it
            _ensure_debug_dir()
            DEBUG_FILE.touch()

        # Simple tail implementation using polling
        self._running = True
        last_size = DEBUG_FILE.stat().st_size

        while self._running:
            time.sleep(0.5)
            if not DEBUG_FILE.exists():
                continue
            current_size = DEBUG_FILE.stat().st_size
            if current_size > last_size:
                with open(DEBUG_FILE, "r") as fh:
                    fh.seek(last_size)
                    for line in fh:
                        self._process_line(line)
                last_size = current_size

    def stop(self) -> None:
        """Stop the tail loop."""
        self._running = False


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #

def make_publisher(
    harness_version: str = "v0",
    benchmark_name: str = "default",
    iteration: int = 0,
) -> AutobenchResultPublisher:
    """Create a pre-configured publisher."""
    return AutobenchResultPublisher(
        harness_version=harness_version,
        benchmark_name=benchmark_name,
        iteration=iteration,
    )


def make_subscriber(
    callback: Callable[[dict[str, Any]], None] | None = None,
    deer_flow_emit: bool = False,
) -> AutobenchResultSubscriber:
    """Create a pre-configured subscriber."""
    return AutobenchResultSubscriber(
        callback=callback,
        deer_flow_emit=deer_flow_emit,
    )


def make_deerflow_subscriber(
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> DeerFlowResultSubscriber:
    """Create a subscriber for deer-flow.sandbox.result.v1 events."""
    return DeerFlowResultSubscriber(callback=callback)