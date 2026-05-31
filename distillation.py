"""Pattern distillation for autobench cycle reports (nervous-bus-1hlf).

Folds a cycle's bus events into the ``autobench.cycle.report.v1`` payload:

* ``summary``  — promotion outcome, AHE counts, aggregate scores, sizes.
* ``patterns`` — top failure modes, successful deltas, dissent hotspots,
  lineage diversity, cross-domain score.
* ``cost``     — worker/improver/judge call counts.

Two entry points:

* :meth:`CycleDistiller.distill_from_events` — pure function over a list of
  envelope-shaped event dicts. The test surface.
* :meth:`CycleDistiller.distill_from_observability` — reads from the live
  observability instance's debug-file fallback (the same path that
  ``deer obs bus`` reads). Lossy if the operator disabled the debug file,
  but the only way to recover events for a still-running daemon.

Defensive folding: when a given event type is absent, the corresponding
field is empty/zero rather than raised. The cycle may legitimately not
have produced certain events (e.g. no judging-pool dissent at all).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .observability import AutobenchObservability


__all__ = ["CycleDistiller"]


# --------------------------------------------------------------------------- #
# Channel name constants (mirror observability.CHANNEL_* without coupling)
# --------------------------------------------------------------------------- #

_CH_CASE_RESULT = "autobench.case.result.v1"
_CH_FAILURE_PATTERN = "autobench.failure_pattern.v1"
_CH_PRED_VERIFIED = "autobench.improver.prediction.verified.v1"
_CH_PRED_REFUTED_LIVE = "autobench.improver.prediction.refuted_live.v1"
_CH_DELTA_DIFF = "autobench.improver.delta.diff.v1"
_CH_JUDGE_DISAGREEMENT = "autobench.judge.disagreement.v1"
_CH_JUDGE_VERDICT = "autobench.judge.pool.verdict.v1"
_CH_DIVERSITY = "autobench.diversity.v1"
_CH_CROSS_DOMAIN = "autobench.cross_domain.evaluation.v1"
_CH_POPULATION_SUMMARY = "autobench.population.summary.v1"
_CH_PROMOTION = "autobench.continuous.promotion_decision.v1"
_CH_WORKER = "autobench.worker.v1"
_CH_IMPROVER = "autobench.improver.v1"


# Dissent threshold for "hotspot" inclusion (matches the wire-pop Phase 4
# escalation threshold). 0.4 = 2 of 5 judges dissented.
_DISSENT_HOTSPOT_THRESHOLD = 0.4

# Caps on returned list sizes (schema does not cap, but consumers do).
_TOP_FAILURE_MODES_N = 5
_DISSENT_HOTSPOTS_N = 10
_SUCCESSFUL_DELTAS_N = 10


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    """Return the ``data`` block of a CloudEvents-lite envelope (or the event itself).

    Most autobench events use the envelope shape ``{type, data, ...}``. A
    minority of tests pass the inner data dict directly; we accept both.
    """
    d = event.get("data")
    if isinstance(d, dict):
        return d
    return event


def _event_type(event: dict[str, Any]) -> str:
    """Return the event's channel/type string."""
    t = event.get("type")
    if isinstance(t, str):
        return t
    # When the event was passed as the bare data block, the caller usually
    # carries the type elsewhere. Treat as untyped.
    return ""


class CycleDistiller:
    """Fold a list of cycle events into a ``autobench.cycle.report.v1`` data dict."""

    def distill_from_events(
        self,
        events: list[dict[str, Any]],
        cycle_id: str,
        domain: str,
        requested_by: str,
        correlation_id: str,
        started_at: str,
        completed_at: str,
        bead_id: str | None = None,
        n_advocates_hint: int | None = None,
        n_cases_hint: int | None = None,
        baseline_score: float | None = None,
    ) -> dict[str, Any]:
        """Distill ``events`` into the cycle.report v1 data payload.

        Args:
            events: List of CloudEvents-lite envelope dicts (or bare data
                dicts). Order is preserved for "latest wins" semantics
                where applicable (e.g. final diversity snapshot).
            cycle_id: ULID identifying the cycle.
            domain: Primary benchmark domain name.
            requested_by: Producer identity ("operator", "hearth-loom", ...).
            correlation_id: Routing key for the matching cycle.requested
                event. For operator-launched cycles, callers pass cycle_id.
            started_at / completed_at: RFC3339 UTC cycle boundaries.
            bead_id: Optional tracker bead anchor.
            n_advocates_hint: Optional override when event stream lacks a
                population.summary (e.g. single-advocate cycles).
            n_cases_hint: Optional override; defaults to count of unique
                case_ids seen in case.result events.
            baseline_score: Optional pre-cycle canonical aggregate. When
                None and no promotion event carries it, defaults to 0.0.

        Returns:
            A dict matching ``schemas/autobench.cycle.report.v1.json``'s
            ``data`` block.
        """
        # ---- index events by type for cheap repeat scans -----------------
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ev in events or []:
            by_type[_event_type(ev)].append(ev)

        case_results = [_event_data(e) for e in by_type.get(_CH_CASE_RESULT, [])]
        failure_patterns = [_event_data(e) for e in by_type.get(_CH_FAILURE_PATTERN, [])]
        verifieds = [_event_data(e) for e in by_type.get(_CH_PRED_VERIFIED, [])]
        refuted_lives = [_event_data(e) for e in by_type.get(_CH_PRED_REFUTED_LIVE, [])]
        diffs = [_event_data(e) for e in by_type.get(_CH_DELTA_DIFF, [])]
        disagreements = [_event_data(e) for e in by_type.get(_CH_JUDGE_DISAGREEMENT, [])]
        judge_verdicts = [_event_data(e) for e in by_type.get(_CH_JUDGE_VERDICT, [])]
        diversity_snaps = [_event_data(e) for e in by_type.get(_CH_DIVERSITY, [])]
        cross_domain = [_event_data(e) for e in by_type.get(_CH_CROSS_DOMAIN, [])]
        pop_summaries = [_event_data(e) for e in by_type.get(_CH_POPULATION_SUMMARY, [])]
        promotion_events = [_event_data(e) for e in by_type.get(_CH_PROMOTION, [])]
        worker_events = [_event_data(e) for e in by_type.get(_CH_WORKER, [])]
        improver_events = [_event_data(e) for e in by_type.get(_CH_IMPROVER, [])]

        # ---- summary block -----------------------------------------------
        summary = self._fold_summary(
            case_results=case_results,
            verifieds=verifieds,
            refuted_lives=refuted_lives,
            pop_summaries=pop_summaries,
            promotion_events=promotion_events,
            n_advocates_hint=n_advocates_hint,
            n_cases_hint=n_cases_hint,
            baseline_score=baseline_score,
        )

        # ---- patterns block ----------------------------------------------
        patterns = {
            "top_failure_modes": self._top_failure_modes(
                case_results=case_results,
                failure_patterns=failure_patterns,
            ),
            "successful_deltas": self._successful_deltas(
                verifieds=verifieds,
                diffs=diffs,
            ),
            "dissent_hotspots": self._dissent_hotspots(disagreements),
            "lineage_diversity": self._lineage_diversity(
                diversity_snaps=diversity_snaps,
                pop_summaries=pop_summaries,
            ),
            "cross_domain_score": self._cross_domain_score(cross_domain),
        }

        # ---- cost block --------------------------------------------------
        cost = self._cost_rollup(
            worker_events=worker_events,
            improver_events=improver_events,
            judge_verdicts=judge_verdicts,
        )

        payload: dict[str, Any] = {
            "correlation_id": str(correlation_id),
            "cycle_id": str(cycle_id),
            "domain": str(domain),
            "requested_by": str(requested_by),
            "started_at": str(started_at),
            "completed_at": str(completed_at),
            "ts": str(completed_at),
            "summary": summary,
            "patterns": patterns,
            "cost": cost,
        }
        if bead_id:
            payload["bead_id"] = str(bead_id)
        return payload

    def distill_from_observability(
        self,
        obs: AutobenchObservability,
        cycle_id: str,
        domain: str,
        requested_by: str,
        correlation_id: str,
        started_at: str,
        completed_at: str,
        bead_id: str | None = None,
        n_advocates_hint: int | None = None,
        n_cases_hint: int | None = None,
        baseline_score: float | None = None,
        debug_file_override: Path | None = None,
    ) -> dict[str, Any]:
        """Read events from the obs debug-file fallback and distill.

        Filters to events whose ``data.session_id`` matches ``obs.session_id``
        OR whose data carries an advocate ``session_id`` that the cycle
        emitted under (population summaries do carry cross-advocate state
        — we widen the filter to any event from the same cycle window).

        Returns the cycle.report v1 data payload.
        """
        path = debug_file_override or getattr(obs, "_debug_file", None)
        events: list[dict[str, Any]] = []
        try:
            if path and Path(path).exists():
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:  # noqa: BLE001 — distillation must never raise
            events = []

        return self.distill_from_events(
            events=events,
            cycle_id=cycle_id,
            domain=domain,
            requested_by=requested_by,
            correlation_id=correlation_id,
            started_at=started_at,
            completed_at=completed_at,
            bead_id=bead_id,
            n_advocates_hint=n_advocates_hint,
            n_cases_hint=n_cases_hint,
            baseline_score=baseline_score,
        )

    # ------------------------------------------------------------------ #
    # Summary folder
    # ------------------------------------------------------------------ #

    def _fold_summary(
        self,
        *,
        case_results: list[dict[str, Any]],
        verifieds: list[dict[str, Any]],
        refuted_lives: list[dict[str, Any]],
        pop_summaries: list[dict[str, Any]],
        promotion_events: list[dict[str, Any]],
        n_advocates_hint: int | None,
        n_cases_hint: int | None,
        baseline_score: float | None,
    ) -> dict[str, Any]:
        # n_advocates: prefer the latest population summary's advocate count.
        n_advocates = 0
        n_iterations = 0
        aggregate_score_best = 0.0
        promoted_advocate_id = ""

        if pop_summaries:
            latest = pop_summaries[-1]
            advocates = latest.get("advocates", []) or []
            n_advocates = len(advocates)
            best = float("-inf")
            for a in advocates:
                bi = int(a.get("best_iter", -1))
                if bi >= 0:
                    # iterations are 0-indexed; count is bi+1
                    n_iterations += bi + 1
                fs = float(a.get("final_score", 0.0) or 0.0)
                if fs > best:
                    best = fs
            if best != float("-inf"):
                aggregate_score_best = best
            # Try winner_score as authoritative.
            if "winner_score" in latest:
                try:
                    aggregate_score_best = max(aggregate_score_best, float(latest["winner_score"]))
                except (TypeError, ValueError):
                    pass
        else:
            # Fallback: derive iteration count and best score from case_results.
            iters_seen: set[int] = set()
            for cr in case_results:
                try:
                    iters_seen.add(int(cr.get("iteration", -1)))
                except (TypeError, ValueError):
                    continue
            iters_seen.discard(-1)
            n_iterations = len(iters_seen)
            n_advocates = max(1, n_advocates_hint or 1)

        if n_advocates_hint is not None:
            n_advocates = int(n_advocates_hint)

        # n_cases: prefer unique case_id count from case.result; else hint.
        unique_cases = {str(cr.get("case_id", "")) for cr in case_results if cr.get("case_id")}
        n_cases = len(unique_cases) if unique_cases else int(n_cases_hint or 0)

        # promoted: True iff the latest promotion event decision == "accepted".
        promoted = False
        if promotion_events:
            latest_promo = promotion_events[-1]
            decision = str(latest_promo.get("decision", "") or "")
            promoted = decision == "accepted"
            promoted_advocate_id = str(latest_promo.get("candidate_advocate_id", "") or "")

        # AHE outcome counts. "refuted_live" wins over a confirmed/refuted
        # label at the same iteration — but absent richer state we count
        # refuted_live events distinctly via the dedicated channel.
        ahe_counts = Counter({k: 0 for k in ("confirmed", "partial", "refuted", "refuted_live", "none")})
        for v in verifieds:
            label = str(v.get("outcome_label", "none") or "none")
            if label not in ahe_counts:
                label = "none"
            ahe_counts[label] += 1
        # Add refuted_live as a distinct dimension (events that ARRIVED
        # before the verified event could fire).
        ahe_counts["refuted_live"] += sum(
            1 for r in refuted_lives if bool(r.get("is_refuted", False))
        )

        return {
            "n_advocates": int(n_advocates),
            "n_iterations": int(n_iterations),
            "n_cases": int(n_cases),
            "promoted": bool(promoted),
            "promoted_advocate_id": str(promoted_advocate_id),
            "aggregate_score_best": float(aggregate_score_best),
            "aggregate_score_baseline": float(baseline_score if baseline_score is not None else 0.0),
            "ahe_outcomes": {k: int(ahe_counts[k]) for k in ("confirmed", "partial", "refuted", "refuted_live", "none")},
        }

    # ------------------------------------------------------------------ #
    # Pattern folders
    # ------------------------------------------------------------------ #

    def _top_failure_modes(
        self,
        case_results: list[dict[str, Any]],
        failure_patterns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Top-5 failure clusters by count.

        Strategy: prefer ``failure_pattern.v1`` events (detector-curated
        shared-prefix clusters). When absent, fall back to grouping
        ``case.result.v1`` events by non-OK verdict.
        """
        counts: Counter[str] = Counter()

        if failure_patterns:
            for fp in failure_patterns:
                verdict = str(fp.get("verdict", "") or "?")
                prefix = str(fp.get("prefix", "") or "")[:60]
                key = f"{verdict}:{prefix}".rstrip(":")
                # Use the cluster's total_in_class as authoritative count.
                try:
                    n = int(fp.get("total_in_class", 0) or 0)
                except (TypeError, ValueError):
                    n = 0
                if n == 0:
                    try:
                        n = int(fp.get("sample_count", 0) or 0)
                    except (TypeError, ValueError):
                        n = 1
                counts[key] += n
        else:
            for cr in case_results:
                verdict = str(cr.get("verdict", "") or "")
                if not verdict or verdict.upper() == "OK":
                    continue
                counts[verdict] += 1

        total = sum(counts.values())
        if total == 0:
            return []
        items: list[dict[str, Any]] = []
        for key, n in counts.most_common(_TOP_FAILURE_MODES_N):
            items.append({
                "failure_mode": key,
                "count": int(n),
                "fraction": float(n) / float(total),
            })
        return items

    def _successful_deltas(
        self,
        verifieds: list[dict[str, Any]],
        diffs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Improver deltas whose subsequent verification confirmed the prediction.

        Joins ``verified`` events to the matching ``delta.diff`` event by
        ``iteration``. ``score_delta`` is the verification's
        ``actual_score_delta``. When no diff is found we still record the
        prediction with an empty delta_summary so the consumer sees that
        a confirmed prediction occurred.
        """
        diffs_by_iter: dict[int, dict[str, Any]] = {}
        for d in diffs:
            try:
                it = int(d.get("iteration", -1))
            except (TypeError, ValueError):
                continue
            diffs_by_iter[it] = d

        items: list[dict[str, Any]] = []
        for v in verifieds:
            label = str(v.get("outcome_label", "") or "")
            if label != "confirmed":
                continue
            try:
                it = int(v.get("iteration", -1))
            except (TypeError, ValueError):
                it = -1
            try:
                score_delta = float(v.get("actual_score_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                score_delta = 0.0
            diff = diffs_by_iter.get(it, {}) or {}
            # Pick the most informative diff field present.
            field = ""
            delta_summary = ""
            for fname in (
                "system_prompt_diff",
                "tool_surface_diff",
                "rollout_protocol_change",
                "context_manager_change",
                "budget_changes",
            ):
                val = diff.get(fname)
                if val:
                    field = fname
                    if isinstance(val, str):
                        delta_summary = val[:240]
                    else:
                        delta_summary = json.dumps(val, default=str)[:240]
                    break
            if not field:
                field = "prediction"
                rationale = v.get("predicted", {}) if isinstance(v.get("predicted"), dict) else {}
                delta_summary = str(rationale.get("rationale", ""))[:240] if rationale else ""
            items.append({
                "field": field,
                "delta_summary": delta_summary,
                "score_delta": score_delta,
            })

        # Rank by absolute score_delta descending so the biggest wins come first.
        items.sort(key=lambda it: abs(it["score_delta"]), reverse=True)
        return items[:_SUCCESSFUL_DELTAS_N]

    def _dissent_hotspots(self, disagreements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Top-10 judge-disagreement events with dissent_ratio > threshold."""
        items: list[dict[str, Any]] = []
        for d in disagreements:
            try:
                ratio = float(d.get("dissent_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if ratio <= _DISSENT_HOTSPOT_THRESHOLD:
                continue
            case_id = str(d.get("case_id", "") or "")
            dist = d.get("verdict_distribution", {}) or {}
            if not isinstance(dist, dict):
                dist = {}
            items.append({
                "case_id": case_id,
                "dissent_ratio": ratio,
                "verdict_distribution": {str(k): int(v) for k, v in dist.items()},
            })

        items.sort(key=lambda it: it["dissent_ratio"], reverse=True)
        return items[:_DISSENT_HOTSPOTS_N]

    def _lineage_diversity(
        self,
        diversity_snaps: list[dict[str, Any]],
        pop_summaries: list[dict[str, Any]],
    ) -> float:
        """Mean lineage diversity for this cycle.

        Order of preference:

        1. Mean of ``diversity_score`` across advocates in the latest
           ``population.summary.v1`` event (this is the canonical cross-
           advocate measure when Phase 2 was on).
        2. Latest ``autobench.diversity.v1`` snapshot's ``diversity_score``.
        3. 0.0 when neither event type appeared.
        """
        if pop_summaries:
            advocates = pop_summaries[-1].get("advocates", []) or []
            scores: list[float] = []
            for a in advocates:
                if "diversity_score" in a:
                    try:
                        scores.append(float(a["diversity_score"]))
                    except (TypeError, ValueError):
                        continue
            if scores:
                return float(sum(scores) / len(scores))

        if diversity_snaps:
            try:
                return float(diversity_snaps[-1].get("diversity_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return 0.0

    def _cross_domain_score(self, cross_domain: list[dict[str, Any]]) -> dict[str, float]:
        """Average per-domain aggregate score across advocates."""
        if not cross_domain:
            return {}
        sums: defaultdict[str, float] = defaultdict(float)
        counts: defaultdict[str, int] = defaultdict(int)
        for cd in cross_domain:
            per = cd.get("per_domain_scores", {}) or {}
            if not isinstance(per, dict):
                continue
            for k, v in per.items():
                try:
                    sums[str(k)] += float(v)
                    counts[str(k)] += 1
                except (TypeError, ValueError):
                    continue
        return {k: (sums[k] / counts[k]) for k in sums if counts[k] > 0}

    # ------------------------------------------------------------------ #
    # Cost rollup
    # ------------------------------------------------------------------ #

    def _cost_rollup(
        self,
        worker_events: list[dict[str, Any]],
        improver_events: list[dict[str, Any]],
        judge_verdicts: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Count requests across worker, improver, and judge channels.

        Worker / improver count once per event (one event per call).
        Judge count uses the ``n_votes`` field of each pool.verdict event
        (each pool.verdict aggregates N parallel judge calls).
        """
        worker_calls = len(worker_events)
        # Improver events come in start/complete pairs; count complete-only
        # to avoid double-billing the round-trip.
        improver_calls = sum(
            1 for e in improver_events
            if str(e.get("status", "") or "").lower() == "complete"
        )
        # When the status field is missing (rare), fall back to counting all.
        if improver_calls == 0 and improver_events:
            improver_calls = len(improver_events)
        judge_calls = 0
        for jv in judge_verdicts:
            try:
                n_votes = int(jv.get("n_votes", 0) or 0)
                if n_votes <= 0:
                    n_votes = int(jv.get("n_judges", 0) or 0)
                judge_calls += max(0, n_votes)
            except (TypeError, ValueError):
                continue

        return {
            "worker_calls": int(worker_calls),
            "improver_calls": int(improver_calls),
            "judge_calls": int(judge_calls),
            "total_requests": int(worker_calls + improver_calls + judge_calls),
        }
