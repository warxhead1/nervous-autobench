"""Before/after diff of two HarnessConfig blobs.

Standalone helper kept in its own module so it can be imported by both
``autobench.observability`` (the emit path) and tests / dashboards without
dragging in unrelated emission machinery.

The output dict shape is the wire schema for ``autobench.improver.delta.diff.v1``:

    {
        "system_prompt_diff": str,        # unified diff or ""
        "tool_surface_diff": str,         # unified diff or ""
        "rollout_protocol_change": {"before": str|None, "after": str|None} | None,
        "context_manager_change":  {"before": str|None, "after": str|None} | None,
        "budget_changes": dict[str, {"before": v, "after": v}],
        "no_change": bool,
    }

The two ``*_change`` fields are ``None`` when unchanged (matching the schema's
``oneOf: [null, object]``); the budget_changes map is empty when unchanged.
``no_change`` is the rollup — ``True`` iff every tracked field is unchanged.
"""

from __future__ import annotations

import difflib
from typing import Any


def _unified_diff(before: str, after: str, label: str) -> str:
    """Return a unified-diff string for two text blobs, or "" if identical.

    Uses 3 lines of context (the difflib default). The ``label`` becomes the
    fromfile/tofile header (e.g. ``"system_prompt"`` → ``--- a/system_prompt``
    / ``+++ b/system_prompt``).
    """
    before_s = before or ""
    after_s = after or ""
    if before_s == after_s:
        return ""
    # splitlines(keepends=False) so difflib adds its own line markers; we then
    # join with newlines for a readable unified-diff text blob.
    a_lines = before_s.splitlines(keepends=True) or [""]
    b_lines = after_s.splitlines(keepends=True) or [""]
    # Ensure each line ends with newline so unified_diff doesn't emit the
    # noisy "\ No newline at end of file" trailer in the common case.
    if a_lines and not a_lines[-1].endswith("\n"):
        a_lines[-1] = a_lines[-1] + "\n"
    if b_lines and not b_lines[-1].endswith("\n"):
        b_lines[-1] = b_lines[-1] + "\n"
    diff_iter = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
        n=3,
    )
    return "".join(diff_iter)


def _enum_value(v: Any) -> str | None:
    """Render a HarnessConfig enum field as its ``.value`` string, or None."""
    if v is None:
        return None
    return getattr(v, "value", str(v))


def diff_harnesses(before: Any, after: Any) -> dict[str, Any]:
    """Compute the before/after diff payload for two ``HarnessConfig`` blobs.

    Accepts duck-typed objects with the standard HarnessConfig fields
    (``system_prompt``, ``rollout_protocol``, ``context_manager``,
    ``tool_surface``, ``budget``). The ``verifiers`` list is intentionally
    ignored — those are callables, not data, and diffing them is not useful.
    """
    sp_diff = _unified_diff(
        getattr(before, "system_prompt", "") or "",
        getattr(after, "system_prompt", "") or "",
        label="system_prompt",
    )
    ts_diff = _unified_diff(
        getattr(before, "tool_surface", "") or "",
        getattr(after, "tool_surface", "") or "",
        label="tool_surface",
    )

    rp_before = _enum_value(getattr(before, "rollout_protocol", None))
    rp_after = _enum_value(getattr(after, "rollout_protocol", None))
    rollout_change: dict[str, Any] | None
    if rp_before != rp_after:
        rollout_change = {"before": rp_before, "after": rp_after}
    else:
        rollout_change = None

    cm_before = _enum_value(getattr(before, "context_manager", None))
    cm_after = _enum_value(getattr(after, "context_manager", None))
    context_change: dict[str, Any] | None
    if cm_before != cm_after:
        context_change = {"before": cm_before, "after": cm_after}
    else:
        context_change = None

    before_budget = dict(getattr(before, "budget", {}) or {})
    after_budget = dict(getattr(after, "budget", {}) or {})
    budget_changes: dict[str, dict[str, Any]] = {}
    for key in sorted(set(before_budget) | set(after_budget)):
        b_val = before_budget.get(key)
        a_val = after_budget.get(key)
        if b_val != a_val:
            budget_changes[key] = {"before": b_val, "after": a_val}

    no_change = (
        sp_diff == ""
        and ts_diff == ""
        and rollout_change is None
        and context_change is None
        and not budget_changes
    )

    return {
        "system_prompt_diff": sp_diff,
        "tool_surface_diff": ts_diff,
        "rollout_protocol_change": rollout_change,
        "context_manager_change": context_change,
        "budget_changes": budget_changes,
        "no_change": no_change,
    }
