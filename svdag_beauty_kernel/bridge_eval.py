"""Producer side of the generic engine-render contract.

Emits ``funsearch.engine_render.requested.v1`` for one evolved candidate and waits
for the matching ``funsearch.engine_render.completed.v1`` (paired by
correlation_id) by tailing the durable bus log (~/.cache/nervous-bus/debug.jsonl).
This is the CONFIRMATION render for chosen champions — never per-candidate fitness
— so a missing/slow engine consumer just yields None (dry mode), never blocking
evolution.

The contract is engine-agnostic on purpose: this file (and this repo) is PUBLIC,
so it names no specific engine. ANY engine adapter may answer the request; the
adapter that renders names itself in the reply's ``engine`` field.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_DEBUG_LOG = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"
REQUEST_CHANNEL = "funsearch.engine_render.requested.v1"
COMPLETED_CHANNEL = "funsearch.engine_render.completed.v1"


def render_enabled() -> bool:
    """Off by default: keeps runs self-contained until an engine consumes the contract."""
    return os.environ.get("AUTOBENCH_ENGINE_RENDER", "0") not in ("0", "", "false", "no")


def request_render(
    publish,
    candidate_code: str,
    candidate_id: str,
    *,
    run_id: str,
    kernel: str = "svdag",
    splice_target: str = "compute_density",
    instance: str | None = None,
    seed: float | None = None,
    timeout: float = 90.0,
    poll: float = 0.5,
) -> dict | None:
    """Ask any engine to render ``candidate_code``; return the completed data or None.

    ``publish`` is a callable (channel, data) -> bool (the kernel's ``_publish``).
    Pairs request → reply by ``correlation_id``. Returns the ``data`` block of the
    ``funsearch.engine_render.completed.v1`` event, or None on timeout (no engine
    answered) — the caller treats None as "skip the engine render, keep going".
    """
    correlation_id = uuid.uuid4().hex[:26]
    requested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Record our read offset BEFORE publishing so we only match fresh completions.
    start_offset = _DEBUG_LOG.stat().st_size if _DEBUG_LOG.exists() else 0

    data = {
        "correlation_id": correlation_id,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "kernel": kernel,
        "splice_target": splice_target,
        "candidate_code": candidate_code,
        "requested_at": requested_at,
    }
    if instance is not None:
        data["instance"] = instance
    if seed is not None:
        data["seed"] = seed

    publish(REQUEST_CHANNEL, data)

    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _scan_for_completion(start_offset, correlation_id)
        if result is not None:
            return result
        time.sleep(poll)
    logger.info("engine render timed out (no completion for %s)", correlation_id)
    return None


def _scan_for_completion(start_offset: int, correlation_id: str) -> dict | None:
    if not _DEBUG_LOG.exists():
        return None
    try:
        with open(_DEBUG_LOG) as f:
            f.seek(start_offset)
            for line in f:
                line = line.strip()
                if not line or COMPLETED_CHANNEL not in line:
                    continue
                try:
                    env = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if env.get("type") != COMPLETED_CHANNEL:
                    continue
                data = env.get("data", {})
                if data.get("correlation_id") == correlation_id:
                    return data
    except OSError as exc:
        logger.debug("engine render log scan failed: %s", exc)
    return None
