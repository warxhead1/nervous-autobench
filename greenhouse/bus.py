"""bus — CloudEvents-lite envelope + dual-write publish for the greenhouse.

Mirrors ``kernels.base.FunSearchKernel._publish`` (durable ``debug.jsonl``
write first, then best-effort ``nervous publish --json``), reimplemented
standalone here rather than reused: the greenhouse is not a kernel subclass
and its CloudEvents ``source`` is always ``/autobench/greenhouse`` regardless
of which domain it evolved this cycle, whereas the kernel base's ``_publish``
derives ``source`` from ``BUS_CHANNEL_PREFIX``.

NEVER bypass this for greenhouse events — see nervous-bus/CLAUDE.md
"NEVER bypass `nervous publish`."
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

SOURCE = "/autobench/greenhouse"

DEFAULT_DEBUG_PATH = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"


def find_nervous_bin() -> str | None:
    """Locate the ``nervous`` shell SDK — PATH first, then the repo path."""
    found = shutil.which("nervous")
    if found:
        return found
    repo = Path.home() / "projects" / "nervous-bus" / "sdk" / "shell" / "nervous"
    return str(repo) if repo.is_file() else None


def build_envelope(channel: str, data: dict) -> dict:
    return {
        "specversion": "1.0",
        "id": uuid.uuid4().urn,
        "source": SOURCE,
        "type": channel,
        "datacontenttype": "application/json",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": data,
    }


def publish(
    channel: str,
    data: dict,
    *,
    debug_path: Path | None = None,
    nervous_bin: str | None = None,
) -> dict:
    """Build a CloudEvents envelope and dual-write it. Fail-silent; returns the envelope.

    Durable write to ``debug.jsonl`` happens unconditionally; the live
    ``nervous publish`` fork is best-effort and honours
    ``AUTOBENCH_OBS_DISABLE_PIPE`` (same knob the kernel loops use) so tests
    and offline runs never touch a subprocess.
    """
    envelope = build_envelope(channel, data)
    payload = json.dumps(envelope)

    dpath = debug_path or DEFAULT_DEBUG_PATH
    try:
        dpath.parent.mkdir(parents=True, exist_ok=True)
        with open(dpath, "a") as f:
            f.write(payload + "\n")
    except Exception:
        pass

    if os.environ.get("AUTOBENCH_OBS_DISABLE_PIPE", "").lower() in {"1", "true", "yes"}:
        return envelope

    nbin = nervous_bin if nervous_bin is not None else find_nervous_bin()
    if nbin:
        try:
            env = dict(os.environ)
            env["NERVOUS_NO_ZELLIJ"] = "1"
            env["NERVOUS_DEBUG_LOG"] = os.devnull
            proc = subprocess.Popen(
                [nbin, "publish", "--json"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            proc.communicate(payload.encode(), timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    return envelope
