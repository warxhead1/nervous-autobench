"""Producer side of the tengine.shadergen.eval contract.

Emits ``tengine.shadergen.eval.requested.v1`` for one candidate and waits for the
matching ``tengine.shadergen.eval.completed.v1`` (paired by correlation_id) by
tailing the durable bus log (~/.cache/nervous-bus/debug.jsonl). This is the
CONFIRMATION render for the best candidate — never per-candidate fitness — so a
missing/slow tengine consumer just yields None (dry mode), never blocking
evolution.
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
REQUEST_CHANNEL = "tengine.shadergen.eval.requested.v1"
COMPLETED_CHANNEL = "tengine.shadergen.eval.completed.v1"


def render_enabled() -> bool:
    """Off by default: keeps runs self-contained until tengine consumes the contract."""
    return os.environ.get("AUTOBENCH_SVDAG_EVAL_RENDER", "0") not in ("0", "", "false", "no")


def request_render(
    publish,
    candidate_code: str,
    candidate_id: str,
    *,
    silo: str = "SvdagRacing",
    splice_target: str = "compute_density",
    domain: str = "svdag",
    timeout: float = 90.0,
    poll: float = 0.5,
) -> dict | None:
    """Ask tengine to render ``candidate_code``; return the eval.completed data or None.

    ``publish`` is a callable (channel, data) -> bool (the kernel's ``_publish``).
    """
    correlation_id = uuid.uuid4().hex[:26]
    requested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Record our read offset BEFORE publishing so we only match fresh completions.
    start_offset = _DEBUG_LOG.stat().st_size if _DEBUG_LOG.exists() else 0

    publish(REQUEST_CHANNEL, {
        "correlation_id": correlation_id,
        "silo": silo,
        "candidate_id": candidate_id,
        "domain": domain,
        "splice_target": splice_target,
        "candidate_code": candidate_code,
        "target_ref": "user/volcanic-reference",
        "requested_at": requested_at,
    })

    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _scan_for_completion(start_offset, correlation_id)
        if result is not None:
            return result
        time.sleep(poll)
    logger.info("eval render timed out (no tengine completion for %s)", correlation_id)
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
        logger.debug("eval log scan failed: %s", exc)
    return None
