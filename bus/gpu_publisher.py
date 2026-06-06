"""GPU job + result publishers for autobench.

These events back the GPU work-queue — they are the queue-of-record, so they
MUST go through the nervous shell SDK (``nervous publish --json``). The SDK
handles the XADD to ``nbus:<type>`` + ``nbus:all``, the ``nbus:dedup:<id>``
idempotency claim, and schema validation against
``autobench.gpu_job.v1`` / ``autobench.gpu_result.v1``. Raw ``redis-cli XADD``
is deliberately NOT used here: it bypasses schema validation and dedup, which
would let a malformed job silently poison the queue.

Flow (matches ``kernels/base.py._publish``):
  1. Build the full CloudEvents-lite envelope (id/source/type/time/data).
  2. Write it to ``~/.cache/nervous-bus/debug.jsonl`` (durable history).
  3. Feed the SAME envelope to ``nervous publish --json`` on stdin for live
     delivery to Redis (dedup + schema-validated). ``NERVOUS_DEBUG_LOG`` is
     pointed at /dev/null so the SDK does not write a *second* debug.jsonl line
     (the durable record in step 2 is the single source of truth).

Failure behavior: if ``nervous`` is unavailable (binary not found) or the SDK
exits non-zero / errors, ``publish`` returns ``False`` and surfaces a warning.
We do NOT silently swallow SDK failures — the durable debug.jsonl write still
happened, but the caller is told the live/validated publish did not succeed so
it can retry or alert. (Note: bead nervous-bus-gzsv tracks a separate issue
where ``nervous`` can exit 0 on silent failure; we don't depend on its fix, and
we add no new silent-failure paths.)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .gpu_types import GPUJob, GPUResult

logger = logging.getLogger(__name__)

DEBUG_CACHE = Path.home() / ".cache" / "nervous-bus"
DEBUG_FILE = DEBUG_CACHE / "debug.jsonl"


def _ensure_debug_dir() -> None:
    DEBUG_CACHE.mkdir(parents=True, exist_ok=True)


def _find_nervous_bin() -> str | None:
    """Locate the ``nervous`` shell SDK — PATH first, then the known repo path."""
    found = shutil.which("nervous")
    if found:
        return found
    repo = Path.home() / "projects" / "nervous-bus" / "sdk" / "shell" / "nervous"
    return str(repo) if repo.is_file() else None


def _write_debug(payload: str) -> None:
    """Append the envelope to the durable debug fallback file."""
    _ensure_debug_dir()
    with open(DEBUG_FILE, "a") as fh:
        fh.write(payload + "\n")


def _publish_via_nervous(event: dict[str, Any], nervous_bin: str | None) -> bool:
    """Publish a pre-built CloudEvents envelope through the nervous shell SDK.

    Writes the durable debug.jsonl record first, then feeds the SAME envelope to
    ``nervous publish --json`` (which XADDs to ``nbus:<type>`` + ``nbus:all``,
    claims ``nbus:dedup:<id>``, and validates against the channel schema).

    Returns True only if the SDK accepted the event (exit 0). On a missing
    binary or non-zero/errored SDK call, returns False and logs a warning —
    the durable write above still stands, but the caller is told the validated
    live publish did not succeed (no silent swallow).
    """
    payload = json.dumps(event)

    # Durable history first — never lost even if the SDK is unavailable.
    try:
        _write_debug(payload)
    except Exception as e:  # pragma: no cover - disk failure is exceptional
        logger.warning("gpu publish: durable debug.jsonl write failed: %s", e)

    channel = event.get("type", "<unknown>")
    if not nervous_bin:
        logger.warning(
            "gpu publish: nervous SDK not found (PATH or "
            "~/projects/nervous-bus/sdk/shell/nervous); %s NOT delivered to "
            "Redis (durable record kept in debug.jsonl)",
            channel,
        )
        return False

    # Feed the ALREADY-BUILT envelope on stdin so nervous forwards it verbatim
    # (the non-json path would re-wrap it). NERVOUS_NO_ZELLIJ keeps pane fan-out
    # out of the hot path; NERVOUS_DEBUG_LOG=/dev/null avoids a duplicate
    # debug.jsonl line (durable record above is authoritative). We do NOT set
    # NERVOUS_NO_REDIS=1 — these events MUST land in Redis (queue-of-record).
    env = dict(os.environ)
    env["NERVOUS_NO_ZELLIJ"] = "1"
    env["NERVOUS_DEBUG_LOG"] = os.devnull
    try:
        proc = subprocess.run(
            [nervous_bin, "publish", "--json"],
            input=payload.encode(),
            timeout=10,
            capture_output=True,
            env=env,
        )
    except Exception as e:
        logger.warning("gpu publish: nervous SDK invocation failed for %s: %s",
                       channel, e)
        return False

    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        logger.warning("gpu publish: nervous SDK rejected %s (exit %d): %s",
                       channel, proc.returncode, stderr)
        return False
    return True


class GPUResultPublisher:
    """Publishes GPUResult events to the ``autobench.gpu_result.v1`` channel.

    Routes through the nervous shell SDK (``nervous publish --json``) so the
    result is schema-validated and dedup-claimed before it hits Redis. No raw
    ``redis-cli XADD`` and no zellij-pipe path — those bypass validation/dedup.
    """

    def __init__(self) -> None:
        self._nervous_bin = _find_nervous_bin()

    def publish(self, result: GPUResult) -> bool:
        """Publish a GPUResult to the bus via the nervous SDK.

        Args:
            result: A GPUResult instance.

        Returns:
            True if the nervous SDK accepted the event (schema-valid, XADD'd).
            False if the SDK was unavailable or rejected the payload — the
            durable debug.jsonl record is still written either way.
        """
        event = result.to_event()
        data = event.setdefault("data", {})
        # Enriched work-queue fields — set where the data is available; the rest
        # are TODOs until GPUResult carries them (see nervous-bus-s0u3.* for the
        # queue enrichment). job_id correlates a result back to its job;
        # correlation_id threads a job/result pair across the bus.
        # TODO(nervous-bus-s0u3): populate job_id/correlation_id once GPUResult
        # carries them through from the dispatched GPUJob. status is derived
        # from the verdict (OK verdict => succeeded, anything else => failed).
        data.setdefault("status", "succeeded" if result.verdict == "OK" else "failed")
        return _publish_via_nervous(event, self._nervous_bin)

    def close(self) -> None:
        """Close the publisher (no-op — the SDK call is per-publish)."""
        pass


class GPUJobPublisher:
    """Publishes GPUJob events to the ``autobench.gpu_job.v1`` channel.

    This is the GPU work-queue's enqueue path — the job event IS the queue
    entry of record, so it goes through the nervous SDK for schema validation
    and ``nbus:dedup:<id>`` idempotency. A malformed job is rejected at publish
    time rather than silently poisoning the queue.
    """

    SOURCE = "/autobench/gpu_broker"
    CHANNEL = "autobench.gpu_job.v1"

    def __init__(self) -> None:
        self._nervous_bin = _find_nervous_bin()

    def publish(self, job: GPUJob) -> bool:
        """Publish a GPUJob to the bus via the nervous SDK.

        Args:
            job: A GPUJob instance.

        Returns:
            True if the nervous SDK accepted the event (schema-valid, XADD'd).
            False if the SDK was unavailable or rejected the payload — the
            durable debug.jsonl record is still written either way.
        """
        from .idgen import iso_now, ulid

        data = job.to_dict()
        # Enriched work-queue fields — set where the data is available; leave a
        # TODO for the rest until GPUJob carries them through the dispatcher.
        # TODO(nervous-bus-s0u3): populate job_id/correlation_id/attempt/
        # max_attempts/priority/source_island once GPUJob carries them (the
        # dataclass currently exposes only silo_id/shader_artifact_path/case_id/
        # frames/timeout_s/reference_image_path). Until then a fresh job_id is
        # minted per publish so the queue entry is at least addressable.
        data.setdefault("job_id", ulid())
        event = {
            "id": ulid(),
            "source": self.SOURCE,
            "type": self.CHANNEL,
            "datacontenttype": "application/json",
            "time": iso_now(),
            "data": data,
        }
        return _publish_via_nervous(event, self._nervous_bin)

    def close(self) -> None:
        """Close the publisher (no-op — the SDK call is per-publish)."""
        pass
