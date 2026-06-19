"""shader_admission — assemble + emit the two-tier shader admission record.

Closes the pre-admission loop between autobench's CPU pre-screen and a downstream
GPU shadow-dispatch gate. Two tiers gate an evolved shader/kernel before it is
admitted into a live engine work-type:

  tier 1 — autobench CPU pre-screen (:mod:`autobench.engines.preadmit`):
           compile check + bounded headless render. Cheap, GPU-light.
  tier 2 — a downstream GPU shadow-dispatch verdict, supplied to this module as a
           plain dict (the engine writes it out of band). This module is engine-
           agnostic: it forwards only the schema-known fields and never reasons
           about engine internals.

This module fuses the tier-2 verdict dict with the tier-1 :class:`PreAdmitResult`
into a ``tengine.shader.admission.v1`` envelope and publishes it on the bus
(an overlay/private schema; the field semantics live with that schema, not here).

Honesty contract:
  * ``shadow_dispatch.coverage`` is carried verbatim. A ``PASS`` with
    ``coverage != "full"`` is NOT a fully-verified pass — the gate ran clean but
    could not observe every declared slot. ``admitted`` can still be True under a
    permissive policy, but the partial coverage and the exact ``unobservable_slots``
    ride along so a strict consumer can gate harder. Nothing is hidden behind a
    green PASS.
  * ``cpu_prescreen.safe is None`` (the tier-1 dynamic gate could not run) means
    the candidate was never actually pre-screened — assembling a two-tier record
    would be a lie, so :func:`build_shader_admission` raises rather than coerce.

Best-effort + non-blocking on emit: any failure (missing SDK, bus down) is
swallowed, mirroring :func:`autobench.kernels.bridge.emit_render_evaluated`.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

# CPU pre-screen verdicts the overlay schema accepts (shader-applicable members
# of the autobench core.py Verdict enum; refactor verdicts RV/RD/RT excluded).
_CPU_VERDICT_ENUM: frozenset[str] = frozenset(
    {"CE", "RE", "TLE", "MLE", "WA", "OK", "VF"}
)


def _cpu_prescreen(preadmit: Any) -> dict[str, Any]:
    """Map a tier-1 :class:`PreAdmitResult` onto the schema's cpu_prescreen block.

    Raises ValueError when ``safe is None`` — the dynamic gate did not run, so
    there is no genuine tier-1 result to record. The schema's ``crash_risk`` is a
    derived *probability*, a different thing from PreAdmitResult's string label,
    so it is intentionally NOT mapped here (it is optional and left to the caller
    of a future risk model).
    """
    safe = getattr(preadmit, "safe", None)
    if safe is None:
        raise ValueError(
            "cpu pre-screen did not run (safe is None — no GPU for the dynamic "
            "gate); refusing to assemble a two-tier admission record"
        )

    verdict = str(getattr(preadmit, "verdict", "OK"))
    if verdict not in _CPU_VERDICT_ENUM:
        # SKIP or any non-enum value: coerce conservatively from the safe flag
        # rather than emit an out-of-enum verdict that would dead-letter.
        verdict = "OK" if safe else "RE"

    return {"safe": bool(safe), "verdict": verdict}


def _coarse_from_shadow(shadow: Optional[dict[str, Any]]) -> Optional[str]:
    """Resolve the coarse PASS/FAIL/None from a tier-2 verdict dict.

    The testbed already computes the coarse ``verdict`` field; trust it but fall
    back to the decision if absent. ``None`` ⟺ shadow dispatch did not run.
    """
    if not shadow:
        return None
    v = shadow.get("verdict")
    if v in ("PASS", "FAIL"):
        return v
    decision = shadow.get("decision")
    if decision == "Admit":
        return "PASS"
    if decision == "Reject":
        return "FAIL"
    return None


def _shadow_block(shadow: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Project the testbed JSON onto the schema's shadow_dispatch object, or None.

    Only the schema-known keys are forwarded (additionalProperties:false), so a
    richer testbed JSON never dead-letters the envelope.
    """
    if not shadow:
        return None
    block: dict[str, Any] = {}
    if "decision" in shadow:
        block["decision"] = shadow["decision"]
    if "reason" in shadow:
        block["reason"] = str(shadow["reason"])
    if "coverage" in shadow:
        block["coverage"] = shadow["coverage"]
    if "frames_stepped" in shadow:
        block["frames_stepped"] = int(shadow["frames_stepped"])
    if shadow.get("subset_context"):
        block["subset_context"] = [int(x) for x in shadow["subset_context"]]
    if shadow.get("violation_slots"):
        block["violation_slots"] = [int(x) for x in shadow["violation_slots"]]
    if shadow.get("unobservable_slots"):
        block["unobservable_slots"] = [int(x) for x in shadow["unobservable_slots"]]
    return block or None


def build_shader_admission(
    *,
    work_type: int,
    shader_id: str,
    source_run_id: str,
    preadmit: Any,
    shadow: Optional[dict[str, Any]] = None,
    slot_written: Optional[bool] = None,
    require_full_coverage: bool = False,
) -> dict[str, Any]:
    """Fuse tier-1 + tier-2 into a ``tengine.shader.admission.v1`` data block.

    Args:
        work_type: Engine work-type id the candidate is admitted against.
        shader_id: Stable id of the candidate (content hash / lineage id).
        source_run_id: ULID of the autobench evolution run that produced it.
        preadmit: tier-1 :class:`PreAdmitResult` (``safe`` must be concrete).
        shadow: tier-2 GPU shadow-dispatch verdict dict, or None if it did not run.
        slot_written: coverage-gate flag, if known.
        require_full_coverage: when True, a partial-coverage tier-2 PASS does NOT
            admit (strict gate). Default False mirrors a permissive gate policy.

    Returns:
        A dict conforming to the overlay schema's ``data`` block. ``admitted`` is
        the honest final decision: tier-1 safe AND tier-2 PASS AND not quarantined
        AND (full coverage OR not require_full_coverage).
    """
    cpu = _cpu_prescreen(preadmit)
    coarse = _coarse_from_shadow(shadow)
    shadow_block = _shadow_block(shadow)
    quarantined = bool(shadow.get("quarantined")) if shadow else False
    coverage = (shadow or {}).get("coverage")

    coverage_ok = (not require_full_coverage) or (coverage == "full")
    admitted = bool(
        cpu["safe"]
        and coarse == "PASS"
        and not quarantined
        and coverage_ok
    )

    data: dict[str, Any] = {
        "work_type": int(work_type),
        "shader_id": str(shader_id),
        "cpu_prescreen": cpu,
        "shadow_dispatch_verdict": coarse,
        "quarantined": quarantined,
        "admitted": admitted,
        "source_run_id": str(source_run_id),
    }
    if shadow_block is not None:
        data["shadow_dispatch"] = shadow_block
    if slot_written is not None:
        data["slot_written"] = bool(slot_written)
    return data


def emit_shader_admission(
    *,
    work_type: int,
    shader_id: str,
    source_run_id: str,
    preadmit: Any,
    shadow: Optional[dict[str, Any]] = None,
    slot_written: Optional[bool] = None,
    require_full_coverage: bool = False,
) -> Optional[dict[str, Any]]:
    """Build and fire-and-forget the ``tengine.shader.admission.v1`` envelope.

    Returns the envelope on a successful publish attempt, or None if it was
    skipped (assembly error, missing SDK, bus down). Never raises — the caller's
    evolution loop must not be affected by a telemetry hiccup.
    """
    try:
        from .envelope import build_event
        from ..kernels.bridge import _find_nervous_bin

        data = build_shader_admission(
            work_type=work_type,
            shader_id=shader_id,
            source_run_id=source_run_id,
            preadmit=preadmit,
            shadow=shadow,
            slot_written=slot_written,
            require_full_coverage=require_full_coverage,
        )

        envelope = build_event(
            source="/autobench/shader",
            type_="tengine.shader.admission.v1",
            data=data,
        )
        # Overlay schema validates specversion "1.0" like the public channels.
        envelope.setdefault("specversion", "1.0")
        payload = json.dumps(envelope)

        nervous_bin = _find_nervous_bin()
        if not nervous_bin:
            return None

        import subprocess

        env = dict(os.environ)
        env["NERVOUS_NO_ZELLIJ"] = "1"
        env["NERVOUS_DEBUG_LOG"] = os.devnull
        proc = subprocess.Popen(
            [nervous_bin, "publish", "--json"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            proc.communicate(payload.encode(), timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return envelope
    except Exception as exc:  # never touch the evolution loop
        try:
            print(f"[shader_admission] emit skipped: {exc}", file=sys.stderr)
        except Exception:
            pass
        return None


def load_shadow_verdict(path: str) -> Optional[dict[str, Any]]:
    """Parse a tier-2 testbed verdict JSON file, or None if absent/unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None
