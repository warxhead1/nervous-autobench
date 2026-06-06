"""Heartbeat-driven GPU admission control + per-island fairness.

nervou-bus-s0u3.7: Producers gate GPU-job submits on the engine's lease
heartbeat (``tengine.gpu.lease.heartbeat.v1``) and respect per-island
in-flight quotas so no single evolution island can starve others.

Design principles:
  - **Graceful fallback**: if no heartbeat is present (engine not running,
    Redis down, or the stream simply hasn't been written to yet) admission
    ALWAYS returns ALLOW.  Producers must never dead-lock waiting for a
    heartbeat that may never come.
  - **Non-blocking**: ``should_submit`` is synchronous and never sleeps.
    A DEFER result signals the caller to do more CPU oracle work and retry
    later — the burden of retry scheduling is on the caller.
  - **Redis optional**: the heartbeat read path uses ``redis-cli XREVRANGE``
    via subprocess (mirrors how producers use ``nervous publish``).  If redis
    is unavailable the call logs a debug message and falls through to ALLOW.
  - **In-process quota**: per-island in-flight counters are kept in memory
    on the ``GPUAdmissionGate`` instance.  They are NOT persisted across
    restarts — on restart all quotas start empty (conservative: allows
    everything until jobs are observed in flight again).

Usage::

    from autobench.bus.gpu_admission import GPUAdmissionGate, SubmitDecision

    gate = GPUAdmissionGate(
        queue_depth_threshold=4,   # defer when queue_depth > this
        max_in_flight_per_island=2, # per-island concurrency cap
    )

    # Before submitting a GPU job:
    decision = gate.should_submit(source_island="island_0")
    if decision == SubmitDecision.ALLOW:
        publisher.publish(job)
        gate.record_submitted(source_island="island_0")
    else:
        # DEFER — spend this cycle on more CPU oracle work and retry
        do_more_cpu_work()

    # When a result arrives:
    gate.record_completed(source_island="island_0")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── Redis stream key for the GPU lease heartbeat ──────────────────────────────
HEARTBEAT_STREAM = "nbus:tengine.gpu.lease.heartbeat.v1"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_QUEUE_DEPTH_THRESHOLD = 4   # defer when queue_depth > this value
DEFAULT_MAX_IN_FLIGHT_PER_ISLAND = 2  # max concurrent GPU jobs per island


class SubmitDecision(str, Enum):
    """Return value of :meth:`GPUAdmissionGate.should_submit`."""

    ALLOW = "allow"
    """The gate has capacity — proceed with the GPU job submit."""

    DEFER = "defer"
    """The GPU queue is too deep or the island's in-flight quota is full.
    Caller should spend this cycle on CPU oracle work and retry later."""


@dataclass
class HeartbeatSnapshot:
    """Parsed fields from a single ``tengine.gpu.lease.heartbeat.v1`` envelope."""

    queue_depth: int
    current_job_id: str | None
    fencing_token: str
    holder: str
    ts: str


def _read_redis_bin() -> str | None:
    """Locate ``redis-cli`` on PATH."""
    import shutil
    return shutil.which("redis-cli")


def _fetch_latest_heartbeat(
    redis_bin: str | None,
    stream: str = HEARTBEAT_STREAM,
) -> HeartbeatSnapshot | None:
    """Read the most-recent heartbeat from *stream* via ``redis-cli XREVRANGE``.

    Returns ``None`` when:
    - ``redis-cli`` is not installed
    - Redis is unreachable / the stream does not exist
    - The envelope data cannot be parsed

    In all ``None`` cases the caller should treat absence as ALLOW (fallback).
    """
    if redis_bin is None:
        logger.debug("gpu_admission: redis-cli not found; heartbeat unavailable")
        return None

    try:
        proc = subprocess.run(
            [redis_bin, "XREVRANGE", stream, "+", "-", "COUNT", "1"],
            capture_output=True,
            timeout=2,
        )
    except Exception as exc:
        logger.debug("gpu_admission: redis-cli invocation failed: %s", exc)
        return None

    if proc.returncode != 0:
        logger.debug(
            "gpu_admission: redis-cli XREVRANGE failed (exit %d): %s",
            proc.returncode,
            proc.stderr.decode(errors="replace").strip()[:200],
        )
        return None

    # redis-cli outputs the XREVRANGE result in a flat text format:
    #   <entry-id>
    #   field_count
    #   key1
    #   value1
    #   ...
    # We parse the flat text into key/value pairs then locate the
    # "data" field (which holds the JSON CloudEvents data block).
    lines = proc.stdout.decode(errors="replace").splitlines()
    return _parse_xrevrange_lines(lines)


def _parse_xrevrange_lines(lines: list[str]) -> HeartbeatSnapshot | None:
    """Parse the flat ``redis-cli XREVRANGE`` text into a HeartbeatSnapshot.

    redis-cli (non-RESP3) emits entries as:
      <id>
      <field-count>       ← only in older Redis / redis-cli versions; sometimes absent
      key
      value
      key
      value
      ...

    We locate the ``data`` key and JSON-parse its value.
    """
    # Collect non-empty lines after the stream entry ID line.
    # The first non-empty line is the entry ID; everything after is k/v pairs.
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 3:
        # Not enough content to have an entry with a data field.
        return None

    # Build a dict from sequential key/value pairs (skip the entry ID at [0]).
    kv_lines = non_empty[1:]
    kv: dict[str, str] = {}
    i = 0
    while i + 1 < len(kv_lines):
        key = kv_lines[i].strip()
        val = kv_lines[i + 1].strip()
        # redis-cli sometimes inserts an integer "field count" line right after
        # the entry ID.  Skip any line that is purely a decimal integer that
        # immediately follows position 0 (the ID line was already skipped).
        if i == 0 and key.isdigit():
            # This is a field-count line; consume it and restart pairing.
            kv_lines = kv_lines[1:]
            continue
        kv[key] = val
        i += 2

    raw_data = kv.get("data")
    if raw_data is None:
        logger.debug("gpu_admission: heartbeat entry has no 'data' key")
        return None

    try:
        data: dict[str, Any] = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        logger.debug("gpu_admission: heartbeat data JSON parse failed: %s", exc)
        return None

    try:
        return HeartbeatSnapshot(
            queue_depth=int(data.get("queue_depth", 0)),
            current_job_id=data.get("current_job_id"),
            fencing_token=str(data.get("fencing_token", "")),
            holder=str(data.get("holder", "")),
            ts=str(data.get("ts", "")),
        )
    except (TypeError, ValueError) as exc:
        logger.debug("gpu_admission: heartbeat snapshot construction failed: %s", exc)
        return None


class GPUAdmissionGate:
    """Heartbeat-driven admission gate with per-island in-flight quotas.

    Thread safety:
        *Not thread-safe.*  Each island / loop should own its own
        ``GPUAdmissionGate`` instance, or callers must serialise access.

    Args:
        queue_depth_threshold: Defer submission when the heartbeat reports
            ``queue_depth > queue_depth_threshold``.  Default 4.
        max_in_flight_per_island: Maximum simultaneous in-flight GPU jobs per
            ``source_island``.  Default 2.
        heartbeat_stream: Redis stream key for the lease heartbeat.
            Defaults to ``nbus:tengine.gpu.lease.heartbeat.v1``.
    """

    def __init__(
        self,
        queue_depth_threshold: int = DEFAULT_QUEUE_DEPTH_THRESHOLD,
        max_in_flight_per_island: int = DEFAULT_MAX_IN_FLIGHT_PER_ISLAND,
        heartbeat_stream: str = HEARTBEAT_STREAM,
    ) -> None:
        self.queue_depth_threshold = queue_depth_threshold
        self.max_in_flight_per_island = max_in_flight_per_island
        self.heartbeat_stream = heartbeat_stream
        self._redis_bin: str | None = _read_redis_bin()
        # per-island in-flight counter: island_key → count
        self._in_flight: dict[str, int] = {}

    # ── Heartbeat read ────────────────────────────────────────────────────── #

    def _latest_heartbeat(self) -> HeartbeatSnapshot | None:
        """Fetch the most-recent heartbeat; returns None on any failure."""
        return _fetch_latest_heartbeat(
            redis_bin=self._redis_bin,
            stream=self.heartbeat_stream,
        )

    # ── Public API ────────────────────────────────────────────────────────── #

    def should_submit(self, source_island: str | int | None = None) -> SubmitDecision:
        """Decide whether a GPU job submit should proceed.

        Checks two conditions in order:

        1. **GPU queue depth** — reads the latest heartbeat from Redis.
           If ``queue_depth > queue_depth_threshold`` → DEFER.
           If no heartbeat is available (engine not running / Redis down) →
           ALLOW (never deadlock).

        2. **Per-island in-flight quota** — if the island already has
           ``max_in_flight_per_island`` jobs in flight → DEFER.

        Returns:
            :attr:`SubmitDecision.ALLOW` or :attr:`SubmitDecision.DEFER`.
        """
        island_key = str(source_island) if source_island is not None else "_default"

        # ── Check 1: queue depth from heartbeat ──────────────────────────── #
        heartbeat = self._latest_heartbeat()
        if heartbeat is not None:
            if heartbeat.queue_depth > self.queue_depth_threshold:
                logger.debug(
                    "gpu_admission: DEFER island=%s — queue_depth=%d > threshold=%d",
                    island_key, heartbeat.queue_depth, self.queue_depth_threshold,
                )
                return SubmitDecision.DEFER
        else:
            # No heartbeat → engine not running or Redis down — ALLOW (fallback).
            logger.debug(
                "gpu_admission: no heartbeat available — defaulting to ALLOW "
                "(island=%s)", island_key,
            )

        # ── Check 2: per-island in-flight quota ──────────────────────────── #
        in_flight = self._in_flight.get(island_key, 0)
        if in_flight >= self.max_in_flight_per_island:
            logger.debug(
                "gpu_admission: DEFER island=%s — in_flight=%d >= quota=%d",
                island_key, in_flight, self.max_in_flight_per_island,
            )
            return SubmitDecision.DEFER

        return SubmitDecision.ALLOW

    def record_submitted(self, source_island: str | int | None = None) -> None:
        """Increment the in-flight counter for *source_island*.

        Call this immediately after a successful ``GPUJobPublisher.publish``
        so the gate can track concurrency.
        """
        island_key = str(source_island) if source_island is not None else "_default"
        self._in_flight[island_key] = self._in_flight.get(island_key, 0) + 1

    def record_completed(self, source_island: str | int | None = None) -> None:
        """Decrement the in-flight counter for *source_island*.

        Call this when the matching ``GPUResult`` arrives (or on timeout/error)
        so the slot is freed for the next job.
        """
        island_key = str(source_island) if source_island is not None else "_default"
        current = self._in_flight.get(island_key, 0)
        if current <= 0:
            # Guard against underflow from mis-paired calls.
            self._in_flight[island_key] = 0
            logger.debug(
                "gpu_admission: record_completed called with no in-flight jobs "
                "for island=%s; ignoring", island_key,
            )
            return
        self._in_flight[island_key] = current - 1

    def in_flight(self, source_island: str | int | None = None) -> int:
        """Return the current in-flight count for *source_island*."""
        island_key = str(source_island) if source_island is not None else "_default"
        return self._in_flight.get(island_key, 0)

    def queue_depth(self) -> int | None:
        """Return the latest reported GPU queue depth, or None if unavailable."""
        hb = self._latest_heartbeat()
        return hb.queue_depth if hb is not None else None
