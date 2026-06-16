"""Stateless helpers for autobench observability.

ULID / time generation, the debug-file fallback path, the divergence formatter
helpers, and the schema-validation helpers used by the convenience emitters.
This module imports only ``channels`` (constants) — never ``core`` or
``events`` — to keep the import graph acyclic.
"""

from __future__ import annotations

import json
import sys
import random
import time
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# ULID / time helpers (same style as signal_bus.py)
# --------------------------------------------------------------------------- #

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    """Generate a 26-char ULID-like identifier (Crockford base32, monotonic-ish)."""
    ts_ms = int(time.time() * 1000)
    # 10 chars time + 16 chars randomness
    time_part = ""
    n = ts_ms
    for _ in range(10):
        time_part = _CROCKFORD[n & 0x1F] + time_part
        n >>= 5
    rand_part = "".join(random.choice(_CROCKFORD) for _ in range(16))
    return time_part + rand_part


def _iso_now() -> str:
    """Return current UTC time as RFC3339 (millisecond precision)."""
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


# --------------------------------------------------------------------------- #
# Debug-file fallback (same path as the rest of the bus)
# --------------------------------------------------------------------------- #

DEBUG_CACHE = Path.home() / ".cache" / "nervous-bus"
DEBUG_FILE = DEBUG_CACHE / "debug.jsonl"


def _ensure_debug_dir() -> None:
    DEBUG_CACHE.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Truncation helper
# --------------------------------------------------------------------------- #

_TRUNCATE_MARKER = "[...truncated]"


def _truncate(s: str, max_len: int) -> str:
    """Truncate ``s`` to ``max_len`` chars, appending [...truncated] when cut.

    Guarantees the result is at most ``max_len`` chars even after the marker
    is appended.
    """
    if len(s) <= max_len:
        return s
    keep = max(0, max_len - len(_TRUNCATE_MARKER))
    return s[:keep] + _TRUNCATE_MARKER


# --------------------------------------------------------------------------- #
# Diff helper
# --------------------------------------------------------------------------- #

# Fields we consider "non-trivial" for divergence comparison. Order is the
# order they appear in the human-readable summary.
_DELTA_FIELDS = (
    "system_prompt_delta",
    "rollout_protocol_changed",
    "context_manager_changed",
    "tool_surface_delta",
    "budget_delta",
    "improvement_summary",
)


def _fmt_value(v: Any) -> str:
    """Render a delta field value compactly for the divergence summary."""
    if isinstance(v, dict):
        if not v:
            return "{}"
        inner = ", ".join(f"{k}: {v[k]}" for k in v)
        return "{" + inner + "}"
    if isinstance(v, str):
        # Trim long strings so the summary stays scannable.
        if len(v) > 40:
            return repr(v[:37] + "...")
        return repr(v)
    return str(v)


def _dict_diff(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Return a compact human-readable diff between two ImprovementDelta-shaped dicts.

    Example output::

        "system_prompt_delta: '' → 'add ex.'; budget_delta: {} → {max_tokens: 6553}"

    Returns ``""`` (falsy) when the two are equivalent on the comparable fields.
    Note: ``improvement_summary`` is informational text and may differ
    even when the structural mutation is identical; we skip it for the
    purpose of *deciding* divergence but DO include it in the summary
    when other fields already diverge.
    """
    parts: list[str] = []
    structural_diff = False
    for field_name in _DELTA_FIELDS:
        av = a.get(field_name, "" if "delta" in field_name and field_name != "budget_delta" else None)
        bv = b.get(field_name, "" if "delta" in field_name and field_name != "budget_delta" else None)
        # Normalise None vs default-empty so we don't false-positive on absence.
        if av is None and bv is None:
            continue
        if av == bv:
            continue
        if field_name != "improvement_summary":
            structural_diff = True
        parts.append(f"{field_name}: {_fmt_value(av)} → {_fmt_value(bv)}")

    if not structural_diff:
        # If the only difference was improvement_summary text, don't report
        # divergence — the harness mutation is identical.
        return ""
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Schema validation helpers (producer-triggered cycle channels, bus.notify)
# --------------------------------------------------------------------------- #

def _schemas_dir() -> Path:
    """Return the repo's ``schemas/`` directory.

    The observability package lives at ``<repo>/autobench/observability/``;
    schemas live at ``<repo>/schemas/<channel>.v<n>.json``. This file sits one
    directory deeper than the legacy ``observability.py``, so we walk up one
    extra parent to land on the same ``<repo>/schemas`` path as before.
    """
    import os
    if env := os.environ.get("NBUS_ROOT"):
        p = Path(env) / "schemas"
        if p.is_dir():
            return p
    pkg_root = Path(__file__).resolve().parents[1]
    if (pkg_root.parent / "schemas").is_dir():
        return pkg_root.parent / "schemas"
    sibling = pkg_root.parent / "nervous-bus"
    if (sibling / "schemas").is_dir():
        return sibling / "schemas"
    return Path(__file__).resolve().parents[2] / "schemas"


def _validate_data_payload(channel: str, data: dict[str, Any]) -> tuple[bool, str]:
    """Validate ``data`` against ``schemas/<channel>.json``'s ``data`` block.

    Returns ``(ok, error_message)``. ``ok`` is True when validation passes
    OR when the schema is unavailable / jsonschema isn't installed. We do
    NOT fail emission on validation problems — the bus contract is
    "best-effort + fall back to debug file" and we preserve that here. The
    error message is logged to stderr for observer-side debugging.
    """
    try:
        import jsonschema  # noqa: WPS433 — lazy: optional dep in some environments
    except Exception:  # noqa: BLE001
        return True, ""
    schema_path = _schemas_dir() / f"{channel}.json"
    if not schema_path.is_file():
        return True, ""
    try:
        schema = json.loads(schema_path.read_text())
        data_schema = schema.get("properties", {}).get("data", {})
        if not data_schema:
            return True, ""
        validator = jsonschema.Draft202012Validator(data_schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if errors:
            msgs = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:3])
            return False, msgs
        return True, ""
    except Exception as e:  # noqa: BLE001 — never raise from obs
        return True, f"validator setup failed: {e}"
