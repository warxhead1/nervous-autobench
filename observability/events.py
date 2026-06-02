"""Standalone convenience emitters bound onto ``AutobenchObservability``.

These were authored as module-level functions and attached to the class via
assignment (``AutobenchObservability.foo = _foo``) so that sibling agents
editing the class body wouldn't collide on it. That contract is preserved
verbatim here: the functions take ``self`` as their first parameter and the
bindings run at import time. The package façade imports this module after
``core`` so the bindings are applied before anyone touches the class.
"""

from __future__ import annotations

import sys
from typing import Any

from .core import AutobenchObservability
from .channels import (
    CHANNEL_ADVERSARIAL_GENERATED,
    CHANNEL_ADVERSARIAL_ROUND,
    CHANNEL_BENCH_COMPLETED,
    CHANNEL_BUS_NOTIFY,
    CHANNEL_CONTINUOUS_DIGEST,
    CHANNEL_CONTINUOUS_SESSION,
    CHANNEL_CROSS_DOMAIN_EVALUATION,
    CHANNEL_CYCLE_REPORT,
    CHANNEL_CYCLE_REQUESTED,
    CHANNEL_DELTA_DIFF,
    CHANNEL_DIVERSITY,
    CHANNEL_FAILURE_CATEGORY,
    CHANNEL_FAILURE_PATTERN,
    CHANNEL_ITERATION_SUMMARY,
    CHANNEL_POPULATION_SUMMARY,
    CHANNEL_PREDICTION_REFUTED_LIVE,
    CHANNEL_PROMOTION_DECISION,
    CHANNEL_SANDBOX_STDERR,
    CHANNEL_WORKER,
    CHANNEL_WORKER_QUEUE_PRESSURE,
    SANDBOX_STDERR_EXCERPT_LEN,
)
from ._util import _iso_now, _validate_data_payload


# --------------------------------------------------------------------------- #
# SACS diversity channel (autobench.diversity.v1) — attached via assignment
# to stay out of the class body (sibling agents edit it concurrently).
# See autobench/diversity.py and research/diversity_penalty_2026.md.
# --------------------------------------------------------------------------- #

def _diversity_snapshot(
    self: AutobenchObservability,
    iteration: int,
    fingerprint: list[float],
    penalty: float,
    diversity_score: float,
    memory_size: int,
) -> None:
    """Emit a per-iteration snapshot of the SACS diversity tracker."""
    data: dict[str, Any] = {
        "iteration": int(iteration),
        "fingerprint": [float(x) for x in fingerprint],
        "penalty": float(penalty),
        "diversity_score": float(diversity_score),
        "memory_size": int(memory_size),
    }
    self._publish(CHANNEL_DIVERSITY, data)


AutobenchObservability.diversity_snapshot = _diversity_snapshot  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Adversarial dual co-evolution channels — Wave 5-X (nervous-bus-1rf).
# Code-A1 / SAGE pattern: the generator emits one curveball_generated per
# LLM call, and the dual emits one round_complete per generate→solve→judge
# round. Both methods are non-blocking and never raise, matching the rest
# of this module's contract. Attached via assignment at module load to
# stay out of the class body that sibling agents may be editing.
# --------------------------------------------------------------------------- #

def _adversarial_curveball_generated(
    self: AutobenchObservability,
    case_id: str,
    gotcha: str,
    target_failure_mode: str | None = None,
    generator_model: str = "",
    prompt_preview: str = "",
) -> None:
    """Emit a curveball-generated event for one LLM-produced adversarial case.

    ``prompt_preview`` is capped at 200 characters to keep the bus payload
    small; callers typically truncate before passing in but we defend in
    depth here.
    """
    preview = str(prompt_preview or "")[:200]
    data: dict[str, Any] = {
        "case_id": str(case_id),
        "gotcha": str(gotcha or ""),
        "generator_model": str(generator_model or ""),
        "prompt_preview": preview,
    }
    if target_failure_mode is not None:
        data["target_failure_mode"] = str(target_failure_mode)
    self._publish(CHANNEL_ADVERSARIAL_GENERATED, data)


def _adversarial_round_complete(
    self: AutobenchObservability,
    round_id: str,
    n_cases: int,
    verdict_counts: dict[str, int],
    failure_categories: dict[str, int],
    mean_score: float,
) -> None:
    """Emit a round-complete summary for one adversarial dual round.

    ``failure_categories`` is the histogram of ``target_failure_mode`` →
    count, restricted to cases the worker actually failed. An empty dict
    means the worker handled every steered trap (or no traps were steered).
    """
    data = {
        "round_id": str(round_id),
        "n_cases": int(n_cases),
        "verdict_counts": dict(verdict_counts or {}),
        "failure_categories": dict(failure_categories or {}),
        "mean_score": float(mean_score),
    }
    self._publish(CHANNEL_ADVERSARIAL_ROUND, data)


AutobenchObservability.adversarial_curveball_generated = (  # type: ignore[attr-defined]
    _adversarial_curveball_generated
)
AutobenchObservability.adversarial_round_complete = (  # type: ignore[attr-defined]
    _adversarial_round_complete
)


# --------------------------------------------------------------------------- #
# Continuous-mode daemon channels (autobench/continuous.py)
#
# session_complete  — one event per RSI session (initial vs final score + cost)
# digest            — one event per generated daily SurpriseDigest
#
# Attached to AutobenchObservability via assignment at module load so we can
# stay out of the class body (which sibling agents are also editing).
# --------------------------------------------------------------------------- #

def _continuous_session_complete(
    self: AutobenchObservability,
    initial_score: float,
    final_score: float,
    n_iterations: int,
    total_cost_usd: float,
    promoted: bool,
    benchmark_source: str = "",
    duration_seconds: float = 0.0,
) -> None:
    """Emit a continuous-mode session-complete event."""
    data: dict[str, Any] = {
        "initial_score": float(initial_score),
        "final_score": float(final_score),
        "n_iterations": int(n_iterations),
        "total_cost_usd": float(total_cost_usd),
        "promoted": bool(promoted),
        "benchmark_source": str(benchmark_source),
        "duration_seconds": float(duration_seconds),
    }
    self._publish(CHANNEL_CONTINUOUS_SESSION, data)


def _continuous_digest(
    self: AutobenchObservability,
    date: str,
    n_sessions: int,
    n_promotions: int,
    n_surprises: int,
    biggest_surprise_summary: str,
    digest_path: str = "",
) -> None:
    """Emit a continuous-mode daily-digest event."""
    data: dict[str, Any] = {
        "date": str(date),
        "n_sessions": int(n_sessions),
        "n_promotions": int(n_promotions),
        "n_surprises": int(n_surprises),
        "biggest_surprise_summary": str(biggest_surprise_summary),
        "digest_path": str(digest_path),
    }
    self._publish(CHANNEL_CONTINUOUS_DIGEST, data)


AutobenchObservability.continuous_session_complete = (  # type: ignore[attr-defined]
    _continuous_session_complete
)
AutobenchObservability.continuous_digest = (  # type: ignore[attr-defined]
    _continuous_digest
)


# --------------------------------------------------------------------------- #
# Cross-run promotion decision (nervous-bus-msqa / wire-pop Phase 5).
# Attached to AutobenchObservability via assignment at module load.
# --------------------------------------------------------------------------- #

def _promotion_decision(
    self: AutobenchObservability,
    cycle_id: str,
    candidate_advocate_id: str | None,
    candidate_session_id: str | None,
    candidate_score: float,
    candidate_adjusted_score: float,
    ahe_outcome: str,
    decision: str,
    decided_by: str,
    reason: str = "",
) -> None:
    """Emit a promotion-decision event for the autobench continuous loop."""
    data: dict[str, Any] = {
        "cycle_id": str(cycle_id),
        "candidate_advocate_id": (
            None if candidate_advocate_id is None else str(candidate_advocate_id)
        ),
        "candidate_session_id": (
            None if candidate_session_id is None else str(candidate_session_id)
        ),
        "candidate_score": float(candidate_score),
        "candidate_adjusted_score": float(candidate_adjusted_score),
        "ahe_outcome": str(ahe_outcome),
        "decision": str(decision),
        "decided_by": str(decided_by),
        "reason": str(reason),
    }
    self._publish(CHANNEL_PROMOTION_DECISION, data)


AutobenchObservability.promotion_decision = (  # type: ignore[attr-defined]
    _promotion_decision
)


# --------------------------------------------------------------------------- #
# Worker agent channel (autobench.worker.v1) — added with MiniMaxWorker.
# Attached to AutobenchObservability via assignment at module load so we
# can stay out of the class body (which sibling agents are also editing).
# --------------------------------------------------------------------------- #

def _worker_call(
    self: AutobenchObservability,
    case_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: float,
    code_preview: str = "",
    error: str | None = None,
) -> None:
    """Emit a worker-agent call event for one benchmark case.

    The worker agent (``MiniMaxWorker``) calls this once per ``generate()``
    so downstream consumers can attribute token cost and latency to the
    code-generation step (distinct from the improver step, which mutates
    the harness rather than producing code).

    ``code_preview`` is truncated to 512 chars at the caller so the bus
    payload stays small. ``error`` is non-None only when both the primary
    and fallback model failed.
    """
    data: dict[str, Any] = {
        "case_id": str(case_id),
        "model": str(model),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost_usd": float(cost_usd),
        "latency_ms": float(latency_ms),
        "code_preview": str(code_preview or "")[:512],
    }
    if error is not None:
        data["error"] = str(error)
    self._publish(CHANNEL_WORKER, data)


AutobenchObservability.worker_call = _worker_call  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Sandbox stderr channel (autobench.sandbox.stderr.v1) — bead nervous-bus-bns.
#
# When the sandbox returns a non-OK error-class verdict (CE/RE/TLE/MLE), the
# stderr text is already captured into HarnessResult.error (truncated at 500
# chars). This channel surfaces the first 200 chars as its own bus event so
# downstream consumers can correlate verdicts with the actual error message
# without parsing case_result.generated_code by hand.
#
# OK and WA never emit on this channel: WA is a wrong-answer outcome rather
# than an error-class signal, so its stderr is not actionable diagnostic data.
#
# Attached via assignment at module load to stay out of the class body that
# sibling agents may be editing concurrently.
# --------------------------------------------------------------------------- #

def _sandbox_stderr(
    self: AutobenchObservability,
    case_id: str,
    iteration: int,
    verdict: str,
    stderr_excerpt: str,
    exit_code: int | None = None,
    language: str = "",
) -> None:
    """Emit a sandbox-stderr event for one error-class verdict."""
    excerpt = str(stderr_excerpt or "")
    if len(excerpt) > SANDBOX_STDERR_EXCERPT_LEN:
        excerpt = excerpt[:SANDBOX_STDERR_EXCERPT_LEN]
    data: dict[str, Any] = {
        "case_id": str(case_id),
        "iteration": int(iteration),
        "verdict": str(verdict),
        "stderr_excerpt": excerpt,
        "exit_code": int(exit_code) if exit_code is not None else None,
        "language": str(language),
    }
    self._publish(CHANNEL_SANDBOX_STDERR, data)

    # Paired classification event — see autobench/stderr_classifier.py for the
    # category set. Self-defensive: if the classifier fails, the stderr event
    # is still on the bus, downstream just doesn't get a category for this case.
    try:
        from autobench.stderr_classifier import classify
        cls = classify(stderr=excerpt, verdict=str(verdict), language=str(language or "python"))
    except Exception:
        return
    cat_data: dict[str, Any] = {
        "case_id": str(case_id),
        "iteration": int(iteration),
        "verdict": str(verdict),
        "category": cls.get("category", "unknown"),
        "confidence": float(cls.get("confidence", 0.0)),
        "language": str(language or "python"),
    }
    if "subcategory" in cls:
        cat_data["subcategory"] = cls["subcategory"]
    if "hint" in cls:
        cat_data["hint"] = cls["hint"]
    if "matched_pattern" in cls:
        cat_data["matched_pattern"] = cls["matched_pattern"]
    self._publish(CHANNEL_FAILURE_CATEGORY, cat_data)


AutobenchObservability.sandbox_stderr = _sandbox_stderr  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Worker queue-pressure channel (autobench.worker.queue_pressure.v1) — bead nervous-bus-8vn.
# Rolling tokens/sec rate signal fired when MiniMaxWorker throughput degrades
# relative to its established baseline. Same assignment-at-module-end pattern.
# --------------------------------------------------------------------------- #

def _worker_queue_pressure(
    self: AutobenchObservability,
    model: str,
    current_rate_tps: float,
    baseline_tps: float,
    deviation_factor: float,
    recent_timeouts_count: int,
    latest_latency_ms: float,
    window_size: int,
) -> None:
    """Emit a queue-pressure signal for one worker session.

    Fired by ``MiniMaxWorker`` when current throughput drops below half the
    baseline OR the latest call's latency exceeds 2x the window mean.
    Debounced to at most one event per 30 seconds by the caller — this
    method does not enforce the debounce itself, it just emits.
    """
    data: dict[str, Any] = {
        "model": str(model),
        "current_rate_tps": float(current_rate_tps),
        "baseline_tps": float(baseline_tps),
        "deviation_factor": float(deviation_factor),
        "recent_timeouts_count": int(recent_timeouts_count),
        "latest_latency_ms": float(latest_latency_ms),
        "window_size": int(window_size),
    }
    self._publish(CHANNEL_WORKER_QUEUE_PRESSURE, data)


AutobenchObservability.worker_queue_pressure = (  # type: ignore[attr-defined]
    _worker_queue_pressure
)
# Iteration-summary rollup channel (autobench.iteration.summary.v1).
#
# Emitted at the completion of each RSI iteration so consumers (pulse
# dashboard, continuous.py surprise digest, deer-flow analytics) don't have
# to recompute aggregate_score / pass_rate / verdict_distribution from the
# raw case.result stream. Attached via assignment at module load to stay
# out of the class body (sibling agents edit it concurrently).
# See schemas/autobench.iteration.summary.v1.json and nervous-bus-91u.
# --------------------------------------------------------------------------- #

def _iteration_summary(
    self: AutobenchObservability,
    iteration: int,
    **kwargs: Any,
) -> None:
    """Emit a rollup summary event for one completed RSI iteration.

    The caller is responsible for assembling the payload via
    ``autobench.iteration_summary.build_iteration_summary`` (or an
    equivalent dict matching ``schemas/autobench.iteration.summary.v1.json``)
    and passing it as keyword arguments. We accept ``**kwargs`` here rather
    than a fixed signature so the schema can grow new fields without
    touching this attachment point.

    ``iteration`` is required and is normalised to an int so consumers can
    rely on its type. All other keys flow through as-is — the schema is
    enforced at the adapter edge per nervous-bus convention.
    """
    data: dict[str, Any] = dict(kwargs)
    data["iteration"] = int(iteration)
    self._publish(CHANNEL_ITERATION_SUMMARY, data)


AutobenchObservability.iteration_summary = _iteration_summary  # type: ignore[attr-defined]
# Failure-pattern channel (autobench.failure_pattern.v1) — nervous-bus-46v.
# Emits one event per detected shared-prefix failure cluster per RSI
# iteration. Pure detection lives in autobench/failure_pattern.py; this
# method is the bus boundary. Attached via assignment at module load to
# stay out of the class body (concurrent edits by sibling waves).
# --------------------------------------------------------------------------- #

def _failure_pattern(
    self: AutobenchObservability,
    iteration: int,
    pattern: Any,  # autobench.failure_pattern.FailurePattern
    prefix_len_chars: int = 20,
) -> None:
    """Emit one failure-pattern event for a detected shared-prefix cluster.

    ``pattern`` is an ``autobench.failure_pattern.FailurePattern`` instance.
    Carries no import dependency on the detector module — fields are pulled
    via ``getattr`` so the obs layer stays free of cycles. ``prefix_len_chars``
    is the detector's configured prefix-length parameter (the cap, not the
    realised length of this particular prefix, which may be shorter when the
    generated code itself was shorter).
    """
    sample_case_ids = list(getattr(pattern, "sample_case_ids", []) or [])
    # Schema requires 1..5 case_ids; defensively cap and skip empty clusters.
    sample_case_ids = [str(c) for c in sample_case_ids[:5] if c]
    if not sample_case_ids:
        # Fall back to a placeholder so the schema's minItems=1 holds; this
        # should not happen in practice because detect_failure_patterns
        # always populates the list, but defend in depth.
        sample_case_ids = ["<unknown>"]
    data: dict[str, Any] = {
        "iteration": int(iteration),
        "verdict": str(getattr(pattern, "verdict", "") or ""),
        "prefix": str(getattr(pattern, "prefix", "") or ""),
        "sample_count": int(getattr(pattern, "sample_count", 0) or 0),
        "total_in_class": int(getattr(pattern, "total_in_class", 0) or 0),
        "sample_case_ids": sample_case_ids,
        "prefix_len_chars": int(prefix_len_chars),
    }
    self._publish(CHANNEL_FAILURE_PATTERN, data)


AutobenchObservability.failure_pattern = _failure_pattern  # type: ignore[attr-defined]
# Live prediction refutation channel (nervous-bus-ykn).
#
# Emitted once per Prediction the moment iter N+1 actuals make it impossible,
# rather than waiting for the iter N+1 → N+2 boundary. Attached via assignment
# at module load to stay out of the class body (which sibling agents are also
# editing).
# --------------------------------------------------------------------------- #

def _prediction_refuted_live(
    self: AutobenchObservability,
    iteration: int,
    status: Any,  # autobench.ahe.LiveRefutationStatus
) -> None:
    """Emit a live-refutation event for the currently-pending prediction.

    ``iteration`` is iter N+1 — the iteration whose partial actuals proved the
    prediction unachievable. ``status`` is an ``autobench.ahe.LiveRefutationStatus``;
    serialised here so this method carries no import dependency on ``ahe``.
    """
    pred = getattr(status, "prediction", None)
    prediction_dict = {
        "predicted_score_delta": float(getattr(pred, "predicted_score_delta", 0.0)),
        "predicted_verdict_class_changes": dict(
            getattr(pred, "predicted_verdict_class_changes", {}) or {}
        ),
        "confidence": float(getattr(pred, "confidence", 0.0)),
        "rationale": str(getattr(pred, "rationale", "") or ""),
    }
    confidence_at_refute = getattr(status, "confidence_at_refute", None)
    data: dict[str, Any] = {
        "iteration": int(iteration),
        "prediction": prediction_dict,
        "actuals_so_far": dict(getattr(status, "actuals_so_far", {}) or {}),
        "remaining_cases": int(getattr(status, "remaining_cases", 0)),
        "is_refuted": bool(getattr(status, "is_refuted", False)),
        "refutation_reason": str(getattr(status, "refutation_reason", "") or ""),
        "confidence_at_refute": (
            float(confidence_at_refute) if confidence_at_refute is not None else None
        ),
    }
    self._publish(CHANNEL_PREDICTION_REFUTED_LIVE, data)


AutobenchObservability.prediction_refuted_live = (  # type: ignore[attr-defined]
    _prediction_refuted_live
)


# --------------------------------------------------------------------------- #
# Harness before/after diff channel (autobench.improver.delta.diff.v1) —
# nervous-bus-utm. Emits the actual APPLIED delta as a human-readable
# unified diff at the iter N→N+1 boundary so operators stop dumping two
# HarnessConfig blobs to eyeball what changed. Attached via assignment at
# module load — sibling agents are editing the class body concurrently.
# --------------------------------------------------------------------------- #

def _improver_delta_diff(
    self: AutobenchObservability,
    iteration: int,
    before: Any,
    after: Any,
) -> None:
    """Emit a before/after diff for the harness applied at iter N → N+1.

    ``iteration`` is the NEW iteration this delta produces (i.e. the
    iteration that will run with the AFTER harness). When the harnesses
    are identical we still emit (``no_change=true``) — "improver didn't
    actually change anything" is itself useful signal.
    """
    # Late import: harness_diff lives in autobench.audit; importing at module
    # top would tangle import order during partial refactors.
    from autobench.audit.harness_diff import diff_harnesses

    diff_payload = diff_harnesses(before, after)
    data: dict[str, Any] = {"iteration": int(iteration), **diff_payload}
    self._publish(CHANNEL_DELTA_DIFF, data)


AutobenchObservability.improver_delta_diff = (  # type: ignore[attr-defined]
    _improver_delta_diff
)


# --------------------------------------------------------------------------- #
# Population summary channel (autobench.population.summary.v1) — bead
# nervous-bus-6yut. Multi-advocate RSI spine: one event per population cycle
# summarising every advocate's best score and naming the winner. Attached via
# assignment at module load to stay out of the class body that sibling agents
# may be editing concurrently.
#
# advocate_id is also an OPTIONAL field on iteration.v1 / iteration.summary.v1
# starting with this bead — when present, downstream consumers can group
# iteration events by lineage. iteration.v1's schema is permissive (no
# additionalProperties:false) so adding the optional field is backward
# compatible without a schema bump. We do NOT add a separate
# autobench.advocate.iteration.v1 channel — the lineage is already identified
# by the per-advocate session_id, and a new channel would force downstream
# consumers to dual-subscribe.
# --------------------------------------------------------------------------- #

def _population_summary(
    self: AutobenchObservability,
    cycle_id: str,
    advocates_summary: list[dict[str, Any]],
    winner_id: str,
    winner_score: float,
    cycle_started_at: str,
    cycle_ended_at: str,
    adjusted_winner_id: str = "",
    diversity_weight: float = 0.0,
) -> None:
    """Emit one population.summary event at end of a multi-advocate cycle.

    ``advocates_summary`` is a list of dicts, each carrying:
        - advocate_id (str): e.g. "advocate-0"
        - session_id (str): the per-advocate ULID on the bus
        - final_score (float): best_score across all iterations
        - best_iter (int): iteration index that produced final_score (-1 if no iter ran)
        - diversity_score (float, optional, nervous-bus-bo86): mean pairwise
          lineage distance ∈ [0, 1] to sibling advocates this cycle.
        - adjusted_score (float, optional, nervous-bus-bo86): best_score
          rescaled by the cross-advocate diversity bonus.

    ``adjusted_winner_id`` (nervous-bus-bo86, optional): advocate_id of the
    highest *adjusted_score* lineage. Equal to ``winner_id`` when
    ``diversity_weight == 0`` or when both selections happen to coincide.
    ``diversity_weight`` is the post-cycle penalty weight used to compute
    adjusted_score (typically 0.10; 0.0 disables the bonus).
    """
    advocates_out: list[dict[str, Any]] = []
    for a in (advocates_summary or []):
        entry: dict[str, Any] = {
            "advocate_id": str(a.get("advocate_id", "")),
            "session_id": str(a.get("session_id", "")),
            "final_score": float(a.get("final_score", 0.0)),
            "best_iter": int(a.get("best_iter", -1)),
        }
        # Optional diversity fields — only emit when caller supplied them so
        # callers without bo86 wiring don't accidentally produce zero-valued
        # fields that downstream consumers might mistake for "fully convergent".
        if "diversity_score" in a:
            entry["diversity_score"] = float(a["diversity_score"])
        if "adjusted_score" in a:
            entry["adjusted_score"] = float(a["adjusted_score"])
        advocates_out.append(entry)

    data: dict[str, Any] = {
        "cycle_id": str(cycle_id),
        "advocates": advocates_out,
        "winner_id": str(winner_id),
        "winner_score": float(winner_score),
        "cycle_started_at": str(cycle_started_at),
        "cycle_ended_at": str(cycle_ended_at),
    }
    if adjusted_winner_id:
        data["adjusted_winner_id"] = str(adjusted_winner_id)
        data["diversity_weight"] = float(diversity_weight)
    self._publish(CHANNEL_POPULATION_SUMMARY, data)


AutobenchObservability.population_summary = (  # type: ignore[attr-defined]
    _population_summary
)


# --------------------------------------------------------------------------- #
# Cross-domain evaluation (autobench.cross_domain.evaluation.v1) — qp91.
# One event per advocate per population cycle when cross-domain aggregation
# is in play. Single-domain cycles do NOT emit — iteration.v1 +
# population.summary.v1 already cover them.
# --------------------------------------------------------------------------- #

def _cross_domain_evaluation_complete(
    self: AutobenchObservability,
    advocate_id: str,
    per_domain_scores: dict[str, float],
    aggregate_score: float,
    weights: dict[str, float],
    primary_domain: str = "",
    cycle_id: str = "",
) -> None:
    """Emit a cross-domain evaluation event for one advocate.

    Args:
        advocate_id: e.g. ``"advocate-0"``.
        per_domain_scores: Map of domain name → score in [0, 1].
        aggregate_score: Weighted mean across ``per_domain_scores`` using
            the renormalized ``weights``.
        weights: Map of domain name → renormalized weight. Sum is 1.0 for
            normalized weights, 0.0 for degenerate-weight defensive paths.
        primary_domain: Optional. Name of the domain RSI optimized against.
        cycle_id: Optional. Population cycle ULID for cross-cycle correlation.
    """
    data: dict[str, Any] = {
        "advocate_id": str(advocate_id),
        "per_domain_scores": {
            str(k): float(v) for k, v in (per_domain_scores or {}).items()
        },
        "aggregate_score": float(aggregate_score),
        "weights": {str(k): float(v) for k, v in (weights or {}).items()},
    }
    if primary_domain:
        data["primary_domain"] = str(primary_domain)
    if cycle_id:
        data["cycle_id"] = str(cycle_id)
    self._publish(CHANNEL_CROSS_DOMAIN_EVALUATION, data)


AutobenchObservability.cross_domain_evaluation_complete = (  # type: ignore[attr-defined]
    _cross_domain_evaluation_complete
)


# --------------------------------------------------------------------------- #
# Forge handshake — bus.bead.bench_completed.v1 (nervous-bus-e8x9).
#
# Emitted by ContinuousModeDaemon._emit_bench_completed when an accepted
# population promotion swaps the canonical harness. The bead_id binds the
# event to a tracker entry; deer-flow's Forge bus_consumer auto-stamps the
# `bench_delta` seal on that bead when this lands on the bus.
#
# Schema: schemas/bus.bead.bench_completed.v1.json — note the schema validates
# the DATA payload (flat shape), unlike the autobench.continuous.* channels
# which wrap the data block inside an envelope schema. Required data fields:
#   session_id, bead_id, baseline_metric, treatment_metric, delta, n,
#   passes_threshold, ts. Optional: ci_lower, ci_upper.
# --------------------------------------------------------------------------- #

def _bench_completed_promotion(
    self: AutobenchObservability,
    bead_id: str,
    baseline_metric: float,
    treatment_metric: float,
    delta: float,
    n: int,
    passes_threshold: bool,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> None:
    """Emit a bench_completed handshake event for an accepted promotion.

    Args:
        bead_id: Tracker bead this promotion is attributed to. Required —
            the channel is bead-keyed.
        baseline_metric: Pre-promotion canonical aggregate score.
        treatment_metric: Candidate (now-canonical) aggregate score.
        delta: ``treatment_metric - baseline_metric``.
        n: Sample size used to compute the metric. Schema requires >= 1; we
            floor here to keep emission safe even when callers miscount.
        passes_threshold: Always True at this callsite — we only fire when a
            promotion actually happened. Forge uses this as the gate signal.
        ci_lower, ci_upper: Optional confidence interval around ``delta``.
            Omitted from the payload when None so the schema's optional
            semantics are preserved.

    The ``session_id`` and ``ts`` fields required by the schema are stamped
    by the standard envelope path — ``session_id`` comes from this
    observability instance (the daemon's session ULID) and ``ts`` is added
    via the envelope's ``time`` field, then mirrored into the data dict
    below so downstream consumers reading the flat data payload still see
    both fields.
    """
    data: dict[str, Any] = {
        "bead_id": str(bead_id),
        "baseline_metric": float(baseline_metric),
        "treatment_metric": float(treatment_metric),
        "delta": float(delta),
        "n": max(1, int(n)),
        "passes_threshold": bool(passes_threshold),
        # Mirror the envelope ts into the data block so the flat-schema
        # validator (which doesn't see the envelope) finds it. The
        # observability layer uses millisecond precision; trim to seconds
        # for an RFC3339 UTC string per nervous-bus convention.
        "ts": _iso_now(),
    }
    if ci_lower is not None:
        data["ci_lower"] = float(ci_lower)
    if ci_upper is not None:
        data["ci_upper"] = float(ci_upper)
    self._publish(CHANNEL_BENCH_COMPLETED, data)


AutobenchObservability.bench_completed_promotion = (  # type: ignore[attr-defined]
    _bench_completed_promotion
)


# --------------------------------------------------------------------------- #
# Producer-triggered cycle channels — nervous-bus-1hlf.
#
# autobench.cycle.requested.v1 — inbound trigger from a producer (hearth-loom
#   loomie, tengine autoshader, tachyonac trade thesis, operator).
# autobench.cycle.report.v1    — outbound distilled report emitted at the
#   end of every cycle, triggered or operator-launched.
#
# Both validate their data payload against the corresponding schema before
# publishing — catches drift between the distillation folder and the schema
# at the adapter edge, per nervous-bus convention. Validation failure logs
# to stderr and falls back to emitting unvalidated (observability must
# never raise).
# --------------------------------------------------------------------------- #

def _cycle_requested(
    self: AutobenchObservability,
    correlation_id: str,
    requested_by: str,
    domain: str,
    ts: str | None = None,
    bead_id: str | None = None,
    n_advocates: int | None = None,
    max_iter: int | None = None,
    budget_seconds: float | None = None,
    target_skill: float | None = None,
    adversarial_ratio: float | None = None,
    judges_per_case: int | None = None,
    improver_strategy: str | None = None,
    notes: str | None = None,
) -> None:
    """Emit a producer-triggered cycle request.

    Used by tests and the operator-side trigger CLI; producers proper
    (hearth-loom, tengine, tachyonac) emit this through their own SDK
    paths. The TriggerDaemon subscribes to this channel and runs a cycle
    per event whose ``correlation_id`` it hasn't seen yet.
    """
    data: dict[str, Any] = {
        "correlation_id": str(correlation_id),
        "requested_by": str(requested_by),
        "domain": str(domain),
        "ts": str(ts) if ts else _iso_now(),
    }
    if bead_id is not None:
        data["bead_id"] = str(bead_id)
    if n_advocates is not None:
        data["n_advocates"] = int(n_advocates)
    if max_iter is not None:
        data["max_iter"] = int(max_iter)
    if budget_seconds is not None:
        data["budget_seconds"] = float(budget_seconds)
    if target_skill is not None:
        data["target_skill"] = float(target_skill)
    if adversarial_ratio is not None:
        data["adversarial_ratio"] = float(adversarial_ratio)
    if judges_per_case is not None:
        data["judges_per_case"] = int(judges_per_case)
    if improver_strategy is not None:
        data["improver_strategy"] = str(improver_strategy)
    if notes is not None:
        data["notes"] = str(notes)[:500]
    ok, msg = _validate_data_payload(CHANNEL_CYCLE_REQUESTED, data)
    if not ok:
        print(f"[obs] cycle_requested data fails schema: {msg}", file=sys.stderr)
    self._publish(CHANNEL_CYCLE_REQUESTED, data)


def _cycle_report(self: AutobenchObservability, data: dict[str, Any]) -> None:
    """Emit a cycle report. Validates ``data`` against the schema first.

    The distiller builds ``data`` to match
    ``schemas/autobench.cycle.report.v1.json``. We validate here so a
    drift between distiller and schema is caught at the bus edge rather
    than corrupting downstream consumers.
    """
    payload = dict(data) if isinstance(data, dict) else {}
    ok, msg = _validate_data_payload(CHANNEL_CYCLE_REPORT, payload)
    if not ok:
        print(f"[obs] cycle_report data fails schema: {msg}", file=sys.stderr)
    self._publish(CHANNEL_CYCLE_REPORT, payload)


AutobenchObservability.cycle_requested = _cycle_requested  # type: ignore[attr-defined]
AutobenchObservability.cycle_report = _cycle_report  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# bus.notify.v1 — nervous-bus-ibkg.
#
# Producer-agnostic notification channel. Producers emit; hearth (home-automation)
# subscribes and routes `channels` entries to actual transports (MQTT for phone,
# discord webhook, ntfy POST, session://<peer-id> for CCM). The bus carries the
# message; consumers decide which transports they have wired.
#
# This convenience method validates the data payload against the schema before
# publishing. Validation failure logs to stderr AND drops the emit (we don't
# want malformed notifies polluting downstream consumers' persistence). The
# emitter never raises — observability discipline.
#
# Sibling SDKs (Rust, Go, shell) emit on the same channel; this is the Python
# convenience surface for autobench-side producers.
# --------------------------------------------------------------------------- #

def _bus_notify(
    self: AutobenchObservability,
    priority: str,
    channels: list[str],
    summary: str,
    source_project: str,
    *,
    correlation_id: str | None = None,
    body: str | None = None,
    deep_link: str | None = None,
    source_event_type: str | None = None,
    dedup_key: str | None = None,
) -> None:
    """Emit a bus.notify.v1 event.

    Args:
        priority: One of ``info``, ``warn``, ``critical``. Consumers may filter
            on this — e.g. only ``critical`` routes to phone.
        channels: At least one transport — ``phone``, ``discord``, ``ntfy``,
            or ``session://<peer-id>``. Unwired transports are logged-and-dropped
            by the consumer; not fatal.
        summary: Lock-screen headline. Hard-capped at 140 chars by the schema;
            longer strings are rejected at validation and dropped.
        source_project: Lowercase project identifier — ``autobench``,
            ``hearth-loom``, ``nervous-bus``, etc.
        correlation_id: Optional ULID of the upstream event for tap-back routing.
        body: Optional longer detail (max 2000 chars).
        deep_link: Optional URI — PR, dugout, gateway path.
        source_event_type: Optional upstream channel name for traceability.
        dedup_key: Optional collapse key for consumers.

    Validation failure (bad enum, oversize summary, missing required field) is
    logged to stderr and the emit is **dropped** — bus consumers should never
    see malformed notifies. The method never raises.
    """
    try:
        data: dict[str, Any] = {
            "priority": str(priority),
            "channels": [str(c) for c in (channels or [])],
            "summary": str(summary),
            "source_project": str(source_project),
            "ts": _iso_now(),
        }
        if correlation_id is not None:
            data["correlation_id"] = str(correlation_id)
        if body is not None:
            data["body"] = str(body)
        if deep_link is not None:
            data["deep_link"] = str(deep_link)
        if source_event_type is not None:
            data["source_event_type"] = str(source_event_type)
        if dedup_key is not None:
            data["dedup_key"] = str(dedup_key)
        ok, msg = _validate_data_payload(CHANNEL_BUS_NOTIFY, data)
        if not ok:
            print(
                f"[obs] bus_notify data fails schema, dropping: {msg}",
                file=sys.stderr,
            )
            return
        self._publish(CHANNEL_BUS_NOTIFY, data)
    except Exception as e:  # noqa: BLE001 — observability must never raise
        print(f"[obs] bus_notify emit failed: {e}", file=sys.stderr)


AutobenchObservability.bus_notify = _bus_notify  # type: ignore[attr-defined]
