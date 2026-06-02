"""Core observability surface: the ``AutobenchObservability`` class + factory.

The class carries the stable per-run ``session_id`` and owns the non-blocking
emission path (``zellij pipe`` with a debug-file fallback). Standalone
convenience emitters that are attached to the class via assignment live in
``events`` and are bound onto the class in the package façade (``__init__``).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from .channels import (
    CHANNEL_AC_VERIFIED,
    CHANNEL_CASE_RESULT,
    CHANNEL_CHECKPOINT_REVERT,
    CHANNEL_CURRICULUM_CYCLE,
    CHANNEL_CURRICULUM_PROBLEM,
    CHANNEL_CURRICULUM_PROBLEM_REJECTED,
    CHANNEL_IMPROVER,
    CHANNEL_IMPROVER_DIVERGENCE,
    CHANNEL_IMPROVER_ENSEMBLE,
    CHANNEL_IMPROVER_REASONING,
    CHANNEL_ITERATION,
    CHANNEL_JUDGE_DISAGREEMENT,
    CHANNEL_JUDGE_POOL_VERDICT,
    CHANNEL_PHASE,
    CHANNEL_PREDICTION,
    CHANNEL_PREDICTION_CLIPPED,
    CHANNEL_PREDICTION_VERIFIED,
    CHANNEL_SANDBOX,
    CHANNEL_SCORING_ADAPTED,
    CHANNEL_SYMBOL_LINEAGE,
    CHANNEL_THRESHOLD_ADAPTED,
    GENERATED_CODE_TRUNCATE_LEN,
    SOURCE,
)
from ._util import (
    DEBUG_FILE,
    _dict_diff,
    _iso_now,
    _truncate,
    _ulid,
)


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class AutobenchObservability:
    """Central instrumentation surface for one autobench run.

    A single instance carries a stable ``session_id`` so a downstream consumer
    (pulse dashboard, deer-flow) can correlate every event from this run.

    All emission methods are **non-blocking and never raise**. On any failure
    (zellij missing, debug file unwritable, malformed data) the failure is
    logged to stderr and emission silently moves on.

    Args:
        session_id: Optional ULID to identify this run. Auto-generated when None.
        source: CloudEvents source URI. Defaults to ``/autobench``.
        debug_file: Optional override for the fallback JSONL path (used in tests).
    """

    def __init__(
        self,
        session_id: str | None = None,
        source: str = SOURCE,
        debug_file: Path | None = None,
    ) -> None:
        self.session_id = session_id or _ulid()
        self.source = source
        self._debug_file = Path(debug_file) if debug_file else DEBUG_FILE
        # Skip the pipe entirely if: env var says so, zellij binary missing,
        # or a previous probe in this instance hung/failed. Once flipped True,
        # all subsequent emits go straight to the debug file (cheap append).
        self._pipe_disabled: bool = (
            os.environ.get("AUTOBENCH_OBS_DISABLE_PIPE", "").lower() in {"1", "true", "yes"}
            or shutil.which("zellij") is None
        )

    # ------------------------------------------------------------------ #
    # Core emission
    # ------------------------------------------------------------------ #

    def _build_envelope(self, channel: str, data: dict[str, Any]) -> dict[str, Any]:
        """Build a CloudEvents-lite envelope around the data payload."""
        return {
            "specversion": "1.0",
            "id": _ulid(),
            "source": self.source,
            "type": channel,
            "datacontenttype": "application/json",
            "time": _iso_now(),
            "data": data,
        }

    def _try_zellij_pipe(self, channel: str, payload: str) -> bool:
        """Try to write to the zellij pipe. Returns True on success.

        After the first failure (timeout or non-zero exit) the pipe is disabled
        for the lifetime of this instance — falling back to debug-file writes
        which are cheap. This prevents N×timeout pathology when zellij is
        unreachable in a non-zellij shell or the WASM plugin isn't loaded.
        """
        if self._pipe_disabled:
            return False
        try:
            proc = subprocess.run(
                ["zellij", "pipe", "-p", "nervous-bus", "-n", channel, "--"],
                input=payload.encode(),
                timeout=0.5,
                capture_output=True,
            )
            if proc.returncode == 0:
                return True
            self._pipe_disabled = True
            return False
        except Exception:
            self._pipe_disabled = True
            return False

    def _write_debug(self, payload: str) -> None:
        """Append to the JSONL fallback file."""
        try:
            self._debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._debug_file, "a") as fh:
                fh.write(payload + "\n")
        except Exception as e:
            print(f"[obs] debug-file write failed: {e}", file=sys.stderr)

    def _publish(self, channel: str, data: dict[str, Any]) -> None:
        """Publish a single event. Never raises."""
        try:
            # Stamp the session_id on everything so consumers can correlate.
            data = dict(data)
            data.setdefault("session_id", self.session_id)

            event = self._build_envelope(channel, data)
            payload = json.dumps(event, default=str)

            if not self._try_zellij_pipe(channel, payload):
                self._write_debug(payload)
        except Exception as e:
            print(f"[obs] publish failed on {channel}: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Phase tracking
    # ------------------------------------------------------------------ #

    def phase_start(self, phase: str, **fields: Any) -> None:
        """Emit a phase-start event.

        ``phase`` is a free-form label such as ``benchmark``, ``sandbox_exec``,
        ``rsi_iteration``, ``improver_call``, ``pareto_update``.
        """
        data = {"phase": phase, "status": "start", "extra": fields or {}}
        self._publish(CHANNEL_PHASE, data)

    def phase_complete(self, phase: str, duration_ms: float, **fields: Any) -> None:
        """Emit a phase-complete event with wall-clock duration."""
        data = {
            "phase": phase,
            "status": "complete",
            "duration_ms": float(duration_ms),
            "extra": fields or {},
        }
        self._publish(CHANNEL_PHASE, data)

    def phase_error(self, phase: str, error: str, **fields: Any) -> None:
        """Emit a phase-error event with the captured exception text."""
        data = {
            "phase": phase,
            "status": "error",
            "error": str(error),
            "extra": fields or {},
        }
        self._publish(CHANNEL_PHASE, data)

    @contextlib.contextmanager
    def phase(self, phase: str, **fields: Any) -> Iterator[None]:
        """Context manager that emits start/complete or start/error automatically.

        ::

            with obs.phase("benchmark", suite="codeforces-easy"):
                run_benchmark()
        """
        self.phase_start(phase, **fields)
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            self.phase_error(phase, f"{type(exc).__name__}: {exc}",
                             duration_ms=duration_ms, **fields)
            raise
        else:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            self.phase_complete(phase, duration_ms, **fields)

    # ------------------------------------------------------------------ #
    # Iteration tracking
    # ------------------------------------------------------------------ #

    def iteration_start(
        self,
        iteration_num: int,
        harness_version: str,
        advocate_id: str | None = None,
    ) -> None:
        """Emit an iteration-start event for the RSI loop.

        ``advocate_id`` is optional (nervous-bus-6yut). When present, the
        event identifies which advocate's lineage produced this iteration in
        a multi-advocate population run. Single-lineage runs omit the field.
        """
        data: dict[str, Any] = {
            "iteration": int(iteration_num),
            "harness_version": str(harness_version),
            "status": "start",
        }
        if advocate_id is not None:
            data["advocate_id"] = str(advocate_id)
        self._publish(CHANNEL_ITERATION, data)

    def iteration_complete(
        self,
        iteration_num: int,
        aggregate_score: float,
        verdict_counts: dict[str, int],
        improvement_delta: dict[str, Any] | None = None,
        harness_version: str = "",
        advocate_id: str | None = None,
    ) -> None:
        """Emit an iteration-complete event with score + verdict breakdown.

        ``advocate_id`` is optional (nervous-bus-6yut) and identifies the
        population lineage. Omitted on single-lineage runs.
        """
        data: dict[str, Any] = {
            "iteration": int(iteration_num),
            "harness_version": str(harness_version),
            "status": "complete",
            "aggregate_score": float(aggregate_score),
            "verdict_counts": dict(verdict_counts),
            "improvement_delta": improvement_delta,
        }
        if advocate_id is not None:
            data["advocate_id"] = str(advocate_id)
        self._publish(CHANNEL_ITERATION, data)

    # ------------------------------------------------------------------ #
    # Sandbox tracking
    # ------------------------------------------------------------------ #

    def sandbox_dispatch(
        self,
        case_id: str,
        language: str,
        sandbox_type: str,
    ) -> None:
        """Emit a sandbox-dispatch event (we are about to run this case)."""
        data = {
            "case_id": str(case_id),
            "language": str(language),
            "sandbox_type": str(sandbox_type),
            "status": "dispatch",
        }
        self._publish(CHANNEL_SANDBOX, data)

    def sandbox_complete(
        self,
        case_id: str,
        verdict: str,
        latency_ms: float,
        exit_code: int,
        language: str = "",
        sandbox_type: str = "",
    ) -> None:
        """Emit a sandbox-complete event with the verdict + timing + exit code."""
        data = {
            "case_id": str(case_id),
            "language": str(language),
            "sandbox_type": str(sandbox_type),
            "status": "complete",
            "verdict": str(verdict),
            "latency_ms": float(latency_ms),
            "exit_code": int(exit_code),
        }
        self._publish(CHANNEL_SANDBOX, data)

    # ------------------------------------------------------------------ #
    # Per-case result (with captured generated code)
    # ------------------------------------------------------------------ #

    def case_result(
        self,
        case_id: str,
        iteration: int,
        language: str,
        verdict: str,
        p_score: float,
        latency_ms: float,
        generated_code: str,
        generated_code_length: int,
        attempt: int = 1,
        per_input_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit a per-case result event with the agent's generated code.

        ``generated_code`` is truncated to ``GENERATED_CODE_TRUNCATE_LEN``
        characters; ``generated_code_length`` retains the original length so
        downstream consumers can detect truncation. ``attempt`` is 1-indexed
        and >1 indicates the iterative rollout protocol re-ran this case
        after a verify_code failure (nervous-bus-x3os). Like all emission
        methods, this never raises and never blocks past the 0.5s pipe probe.
        """
        code = generated_code or ""
        if len(code) > GENERATED_CODE_TRUNCATE_LEN:
            code = code[:GENERATED_CODE_TRUNCATE_LEN]
        data = {
            "case_id": str(case_id),
            "iteration": int(iteration),
            "language": str(language),
            "verdict": str(verdict),
            "p_score": float(p_score),
            "latency_ms": float(latency_ms),
            "generated_code": code,
            "generated_code_length": int(generated_code_length),
            "attempt": max(1, int(attempt)),
        }
        if per_input_results:
            data["per_input_results"] = [
                {
                    "verdict": str(r.get("verdict", "")),
                    "p_score": float(r.get("p_score", 0.0)),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                }
                for r in per_input_results
            ]
        self._publish(CHANNEL_CASE_RESULT, data)

    # ------------------------------------------------------------------ #
    # Improver tracking
    # ------------------------------------------------------------------ #

    def improver_call_start(self, model: str, prompt_tokens: int) -> None:
        """Emit an improver-call-start event."""
        data = {
            "model": str(model),
            "status": "start",
            "prompt_tokens": int(prompt_tokens),
        }
        self._publish(CHANNEL_IMPROVER, data)

    def improver_call_complete(
        self,
        model: str,
        completion_tokens: int,
        delta_summary: str,
    ) -> None:
        """Emit an improver-call-complete event with the resulting delta."""
        data = {
            "model": str(model),
            "status": "complete",
            "completion_tokens": int(completion_tokens),
            "delta_summary": str(delta_summary),
        }
        self._publish(CHANNEL_IMPROVER, data)

    # ------------------------------------------------------------------ #
    # Improver deliberation (reasoning + divergence)
    # ------------------------------------------------------------------ #

    def improver_reasoning(
        self,
        model: str,
        iteration: int,
        prompt: str,
        raw_response: str,
        parsed_delta: dict[str, Any],
        parse_status: str,
        fallback_reason: str | None = None,
        latency_ms: float = 0.0,
        cost_dollars: float | None = None,
    ) -> None:
        """Emit a reasoning-capture event for one improver call.

        ``parse_status`` is one of ``"ok"``, ``"fell_back_to_rule_based"``,
        ``"no_change"``. ``parsed_delta`` should be an ``ImprovementDelta``
        rendered as a plain dict (use ``dataclasses.asdict``).
        """
        data: dict[str, Any] = {
            "iteration": int(iteration),
            "model": str(model),
            "prompt": str(prompt),
            "raw_response": str(raw_response),
            "parsed_delta": dict(parsed_delta) if parsed_delta else {},
            "parse_status": str(parse_status),
            "latency_ms": float(latency_ms),
        }
        if fallback_reason is not None:
            data["fallback_reason"] = str(fallback_reason)
        if cost_dollars is not None:
            data["cost_dollars"] = float(cost_dollars)
        self._publish(CHANNEL_IMPROVER_REASONING, data)

    def improver_divergence(
        self,
        iteration: int,
        llm_delta: dict[str, Any],
        heuristic_delta: dict[str, Any],
    ) -> None:
        """Emit a divergence event comparing LLM and rule-based proposals.

        Computes ``divergent`` (any non-trivial field differs) and
        ``divergence_summary`` (human-readable diff) from the two dicts.
        """
        llm = dict(llm_delta) if llm_delta else {}
        heur = dict(heuristic_delta) if heuristic_delta else {}
        summary = _dict_diff(llm, heur)
        divergent = bool(summary)
        data = {
            "iteration": int(iteration),
            "llm_delta": llm,
            "heuristic_delta": heur,
            "divergent": divergent,
            "divergence_summary": summary if summary else "no divergence",
        }
        self._publish(CHANNEL_IMPROVER_DIVERGENCE, data)

    def improver_threshold_adapted(
        self,
        effective_threshold: float,
        velocity: float,
        variance: float,
        iterations_used: int,
    ) -> None:
        """Emit convergence threshold adaptation event.

        Emitted when the RSI loop detects a plateau and relaxes the
        improvement threshold, or tightens it during high-velocity improvement.
        """
        data = {
            "session_id": self.session_id,
            "effective_threshold": float(effective_threshold),
            "velocity": float(velocity),
            "variance": float(variance),
            "iterations_used": int(iterations_used),
        }
        self._publish(CHANNEL_THRESHOLD_ADAPTED, data)

    def weights_adapted(
        self,
        iteration: int,
        weights: dict[str, float],
        score_variance: float,
        score_velocity: float,
        reason: str,
    ) -> None:
        """Emit weights adaptation event.

        Fired when the SICA scoring weights are recomputed based on observed
        score variance and velocity. ``iteration`` is the RSI iteration these
        weights were used for (i.e. the iteration that just completed).
        ``weights`` is the renormalized weight dict that was applied.
        ``score_variance`` is the measured variance of ``score_delta`` across
        iterations. ``score_velocity`` is the mean signed score delta over
        the same window. ``reason`` describes why adaptation fired
        (e.g. ``"plateau_detected"``, ``"high_variance"``, ``"early_iteration"``).
        """
        data = {
            "session_id": self.session_id,
            "iteration": int(iteration),
            "weights": {str(k): float(v) for k, v in weights.items()},
            "score_variance": float(score_variance),
            "score_velocity": float(score_velocity),
            "reason": str(reason),
        }
        self._publish(CHANNEL_SCORING_ADAPTED, data)

    # ------------------------------------------------------------------ #
    # Curriculum tracking
    # ------------------------------------------------------------------ #

    def curriculum_problem_generated(
        self,
        case_id: str,
        prompt: str,
        target_skills: list[str],
        difficulty_rating: int,
        generator_model: str,
        rationale: str,
        date: str,
        cycle_id: str = "",
    ) -> None:
        """Emit one curriculum.problem event for a freshly-generated case.

        ``prompt`` and ``rationale`` are truncated to ~280 chars for the bus
        payload — the full text lives in the cases.jsonl on disk.

        ``cycle_id`` lets downstream group all problems from one cycle —
        required for multi-cycle nightly burns (nervous-bus-9obz). Empty
        string for back-compat with single-cycle callers.
        """
        data = {
            "session_id": self.session_id,
            "case_id": str(case_id),
            "prompt_preview": str(prompt)[:280],
            "target_skills": [str(s) for s in (target_skills or [])],
            "difficulty_rating": int(difficulty_rating),
            "generator_model": str(generator_model),
            "rationale_preview": str(rationale or "")[:280],
            "date": str(date),
        }
        if cycle_id:
            data["cycle_id"] = str(cycle_id)
        self._publish(CHANNEL_CURRICULUM_PROBLEM, data)

    def curriculum_cycle_complete(
        self,
        cycle_id: str,
        n_problems_generated: int,
        n_problems_validated: int,
        goals_summary: dict[str, Any],
        date: str,
        generator_model: str = "",
        n_problems_rejected: int = 0,
    ) -> None:
        """Emit a curriculum.cycle event summarising one daily cycle."""
        data = {
            "cycle_id": str(cycle_id),
            "n_problems_generated": int(n_problems_generated),
            "n_problems_validated": int(n_problems_validated),
            "n_problems_rejected": int(n_problems_rejected),
            "goals_summary": dict(goals_summary or {}),
            "generator_model": str(generator_model),
            "date": str(date),
        }
        self._publish(CHANNEL_CURRICULUM_CYCLE, data)

    def curriculum_problem_rejected(
        self,
        reason: str,
        detail: str,
        generator_model: str,
        cycle_id: str = "",
        row_index: int | None = None,
        missing_keys: list[str] | None = None,
        raw_excerpt: str = "",
        stage: str | None = None,
        judge_detail: dict[str, Any] | None = None,
    ) -> None:
        """Emit one curriculum.problem.rejected event for a dropped problem.

        ``reason`` is the rejection taxonomy enum. Parse-stage values:
        json_decode_error, not_an_array, not_an_object, missing_required_keys,
        field_validation_error. Judge-stage values: judge_unsolvable,
        output_mismatch, skill_mismatch, judge_error. ``detail`` and
        ``raw_excerpt`` are capped at 1024 chars to bound payload.

        ``stage`` is "parse" or "judge"; absent on older callers. ``judge_detail``
        carries the judge's solution/skill comparison (see schema for shape).
        """
        data: dict[str, Any] = {
            "reason": str(reason),
            "detail": str(detail)[:1024],
            "generator_model": str(generator_model),
            "cycle_id": str(cycle_id),
            "row_index": row_index if row_index is None else int(row_index),
            "raw_excerpt": str(raw_excerpt)[:1024],
        }
        if missing_keys:
            data["missing_keys"] = [str(k) for k in missing_keys]
        if stage:
            data["stage"] = str(stage)
        if judge_detail:
            jd = dict(judge_detail)
            for k in ("judge_output", "claimed_output", "notes"):
                if k in jd:
                    jd[k] = str(jd[k])[:512]
            for k in ("actual_skills", "claimed_skills"):
                if k in jd and isinstance(jd[k], (list, tuple)):
                    jd[k] = [str(s) for s in jd[k]]
            data["judge_detail"] = jd
        self._publish(CHANNEL_CURRICULUM_PROBLEM_REJECTED, data)

    # ------------------------------------------------------------------ #
    # AHE prediction contract (arXiv:2604.25850)
    # ------------------------------------------------------------------ #

    def improver_prediction(
        self,
        iteration: int,
        prediction: Any,  # autobench.ahe.Prediction
        model: str,
    ) -> None:
        """Emit a falsifiable prediction the improver made about iteration+1.

        ``prediction`` is an ``autobench.ahe.Prediction`` dataclass instance.
        The dataclass is rendered to a dict at the call site so this method
        carries no import dependency on ``ahe``.
        """
        data = {
            "iteration": int(iteration),
            "model": str(model),
            "predicted_score_delta": float(getattr(prediction, "predicted_score_delta", 0.0)),
            "predicted_verdict_class_changes": dict(
                getattr(prediction, "predicted_verdict_class_changes", {}) or {}
            ),
            "confidence": float(getattr(prediction, "confidence", 0.0)),
            "rationale": str(getattr(prediction, "rationale", "") or ""),
            "active": bool(getattr(prediction, "active", True)),
            "parent_prediction_id": getattr(prediction, "parent_prediction_id", None),
            "fact_fingerprint": getattr(prediction, "fact_fingerprint", None),
            "prediction_id": getattr(prediction, "prediction_id", ""),
        }
        src_scope = getattr(prediction, "source_scope_key", None)
        if src_scope is not None:
            data["source_scope_key"] = str(src_scope)
        self._publish(CHANNEL_PREDICTION, data)

    def improver_prediction_clipped(
        self,
        iteration: int,
        original: Any,  # autobench.ahe.Prediction (pre-clip)
        clipped: Any,   # autobench.ahe.Prediction (post-clip)
        clip_reasons: list[str],
    ) -> None:
        """Emit when a prediction had at least one verdict-delta clipped to feasible range.

        nervous-bus-8d1d: improvers sometimes propose verdict-count deltas that
        can't physically happen (e.g. ``CE: -12`` when only 4 CE cases exist).
        We clip these to feasible bounds before persistence; this event records
        what was changed so the calibration ledger can downweight the
        improver's apparent accuracy on guaranteed-refute predictions.
        """
        data = {
            "iteration": int(iteration),
            "clip_reasons": list(clip_reasons or []),
            "original_verdict_changes": dict(
                getattr(original, "predicted_verdict_class_changes", {}) or {}
            ),
            "clipped_verdict_changes": dict(
                getattr(clipped, "predicted_verdict_class_changes", {}) or {}
            ),
            "original_confidence": float(getattr(original, "confidence", 0.0)),
        }
        self._publish(CHANNEL_PREDICTION_CLIPPED, data)

    def improver_prediction_verified(
        self,
        iteration: int,
        verification: Any,  # autobench.ahe.PredictionVerification
    ) -> None:
        """Emit the verification result for a previously-made prediction.

        ``verification`` is an ``autobench.ahe.PredictionVerification``. The
        ``iteration`` field is the iteration whose ACTUALS were compared
        against the prediction (i.e. the iteration AFTER the prediction-bearing
        one).
        """
        predicted = getattr(verification, "predicted", None)
        predicted_dict = {
            "predicted_score_delta": float(getattr(predicted, "predicted_score_delta", 0.0)),
            "predicted_verdict_class_changes": dict(
                getattr(predicted, "predicted_verdict_class_changes", {}) or {}
            ),
            "confidence": float(getattr(predicted, "confidence", 0.0)),
            "rationale": str(getattr(predicted, "rationale", "") or ""),
            "active": bool(getattr(predicted, "active", True)),
            "parent_prediction_id": getattr(predicted, "parent_prediction_id", None),
            "fact_fingerprint": getattr(predicted, "fact_fingerprint", None),
            "prediction_id": getattr(predicted, "prediction_id", ""),
        }
        data = {
            "iteration": int(iteration),
            "predicted": predicted_dict,
            "actual_score_delta": float(getattr(verification, "actual_score_delta", 0.0)),
            "actual_verdict_class_changes": dict(
                getattr(verification, "actual_verdict_class_changes", {}) or {}
            ),
            "score_delta_error": float(getattr(verification, "score_delta_error", 0.0)),
            "verdict_match_ratio": float(getattr(verification, "verdict_match_ratio", 0.0)),
            "outcome_label": str(getattr(verification, "outcome_label", "refuted")),
            "confidence_calibration": float(
                getattr(verification, "confidence_calibration", 0.0)
            ),
            "lifecycle_status": str(getattr(verification, "lifecycle_status", "active")),
        }
        self._publish(CHANNEL_PREDICTION_VERIFIED, data)

    def checkpoint_revert(
        self,
        iter_regressed: int,
        regression_delta: float,
        reverted_to_iter: int,
        variance_floor_used: float,
        best_score: float | None = None,
        iter_score: float | None = None,
    ) -> None:
        """Emit a best-iter-keep revert event (nervous-bus-sf0y).

        Fired when iteration ``iter_regressed`` scored more than
        ``variance_floor_used`` below the best-so-far iter and the RSI
        loop snapped its working harness back to the iter
        ``reverted_to_iter`` checkpoint. ``regression_delta`` is the
        signed delta (iter_score - best_score), so it's negative whenever
        a revert fires.
        """
        data = {
            "iter_regressed": int(iter_regressed),
            "regression_delta": float(regression_delta),
            "reverted_to_iter": int(reverted_to_iter),
            "variance_floor_used": float(variance_floor_used),
        }
        if best_score is not None:
            data["best_score"] = float(best_score)
        if iter_score is not None:
            data["iter_score"] = float(iter_score)
        self._publish(CHANNEL_CHECKPOINT_REVERT, data)

    # ------------------------------------------------------------------ #
    # JudgingPool consensus (nervous-bus-c48)
    # ------------------------------------------------------------------ #

    def judge_pool_verdict(
        self,
        case_id: str,
        iteration: int,
        n_judges: int,
        consensus_verdict: str,
        dissent_ratio: float,
        verdict_distribution: dict[str, int],
        consensus_p_score: float,
        consensus_p_cost: float = 0.0,
        consensus_p_time: float = 0.0,
        votes_summary: list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit the per-case consensus verdict from the anonymous JudgingPool.

        ``n_judges`` is the configured ensemble size; ``n_votes`` (derived
        from ``verdict_distribution`` sum) may be smaller if some judges
        errored. Like all emission methods, never raises and never blocks
        past the 0.5s pipe probe.
        """
        dist = {str(k): int(v) for k, v in (verdict_distribution or {}).items()}
        n_votes = sum(dist.values())
        data: dict[str, Any] = {
            "case_id": str(case_id),
            "iteration": int(iteration),
            "n_judges": int(n_judges),
            "n_votes": int(n_votes),
            "consensus_verdict": str(consensus_verdict),
            "dissent_ratio": float(dissent_ratio),
            "verdict_distribution": dist,
            "consensus_p_score": float(consensus_p_score),
            "consensus_p_cost": float(consensus_p_cost),
            "consensus_p_time": float(consensus_p_time),
        }
        if votes_summary:
            data["votes_summary"] = [
                {
                    "slot": int(v.get("slot", 0)),
                    "verdict": str(v.get("verdict", "")),
                    "p_score": float(v.get("p_score", 0.0)),
                }
                for v in votes_summary
            ]
        self._publish(CHANNEL_JUDGE_POOL_VERDICT, data)

    def judge_disagreement(
        self,
        case_id: str,
        iteration: int,
        consensus_verdict: str,
        dissent_ratio: float,
        dissent_threshold: float,
        verdict_distribution: dict[str, int],
        minority_verdicts: list[str] | None = None,
    ) -> None:
        """Emit the disagreement escalation signal (dissent_ratio > threshold).

        Mirrors the case-id and iteration of the matching
        autobench.judge.pool.verdict.v1 event so downstream consumers can
        join them by (session_id, case_id, iteration).
        """
        dist = {str(k): int(v) for k, v in (verdict_distribution or {}).items()}
        n_votes = sum(dist.values())
        if minority_verdicts is None:
            minority_verdicts = [v for v in dist if v != consensus_verdict]
        data: dict[str, Any] = {
            "case_id": str(case_id),
            "iteration": int(iteration),
            "n_votes": int(n_votes),
            "consensus_verdict": str(consensus_verdict),
            "dissent_ratio": float(dissent_ratio),
            "dissent_threshold": float(dissent_threshold),
            "verdict_distribution": dist,
            "minority_verdicts": [str(v) for v in minority_verdicts],
        }
        self._publish(CHANNEL_JUDGE_DISAGREEMENT, data)

    # ------------------------------------------------------------------ #
    # Multi-improver ensemble (nervous-bus-9xd, wire-pop Phase 6)
    # ------------------------------------------------------------------ #

    def improver_ensemble_complete(
        self,
        iteration: int,
        strategy: str,
        n_instances: int,
        instances: list[dict[str, Any]],
        vote_outcome: dict[str, Any],
    ) -> None:
        """Emit a multi-improver ensemble record (autobench.improver.ensemble.v1).

        Fired once per RSI iteration when the active improver fans out N
        anonymous MiniMax calls. ``strategy`` is "vote" (majority on field
        deltas) or "parallel" (best-arm by forward-eval score). ``instances``
        carries one dict per fan-out arm with ``instance_idx``,
        ``delta_summary``, optional ``score``, and ``selected`` bool.
        ``vote_outcome`` is the aggregator's per-field decision (vote) or
        ``selected_instance_idx`` (parallel).
        """
        data = {
            "iteration": int(iteration),
            "strategy": str(strategy),
            "n_instances": int(n_instances),
            "instances": list(instances or []),
            "vote_outcome": dict(vote_outcome or {}),
        }
        self._publish(CHANNEL_IMPROVER_ENSEMBLE, data)

    # ------------------------------------------------------------------ #
    # hearth-loom AC bullet evidence (nervous-bus-6kwv)
    # ------------------------------------------------------------------ #

    def ac_verified(
        self,
        bead_id: str,
        exec_id: str,
        ac_index: int,
        ac_text: str,
        command: str,
        exit_code: int,
        duration_ms: int,
        *,
        stdout: str | None = None,
        stderr: str | None = None,
        working_dir: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Emit one ``hearth-loom.ac.verified.v1`` event per AC bullet execution.

        Carries the evidence — extracted bash command + exit_code + truncated
        stdout/stderr + duration — for a single acceptance-criteria bullet
        verified during a hearth-loom loomie run. The primary producer is
        hearth-loom's Go executor; this Python surface exists so:

        * autobench-side post-cycle AC re-verification can emit on the same
          channel without going through the shell SDK
        * tests against the schema have an in-process emitter to exercise.

        Pure observability — never raises, validates payload shape (string vs
        int) and silently drops malformed input. Bound the stdout/stderr fields
        to ``GENERATED_CODE_TRUNCATE_LEN`` chars; oversize input is suffixed
        with ``[...truncated]``.

        Args:
            bead_id: Bead id this acceptance criterion was verified for.
            exec_id: Loomie execution identifier (correlates to lifecycle).
            ac_index: 0-based bullet index within the bead's AC list.
            ac_text: Prose AC bullet text (auto-truncated to 500 chars).
            command: Bash command extracted from the bullet (auto-truncated
                to 1000 chars).
            exit_code: Process exit code; 0 = pass.
            duration_ms: Wall-clock duration in milliseconds (>= 0).
            stdout: Optional captured stdout (truncated to 4096 chars).
            stderr: Optional captured stderr (truncated to 4096 chars).
            working_dir: Optional cwd the command ran in.
            correlation_id: Optional ULID linking to a producer-triggered
                subagent request.
        """
        try:
            data: dict[str, Any] = {
                "bead_id": str(bead_id),
                "exec_id": str(exec_id),
                "ac_index": int(ac_index),
                "ac_text": _truncate(str(ac_text), 500),
                "command": _truncate(str(command), 1000),
                "exit_code": int(exit_code),
                "duration_ms": max(0, int(duration_ms)),
                "ts": _iso_now(),
            }
            if stdout is not None:
                data["stdout"] = _truncate(str(stdout), 4096)
            if stderr is not None:
                data["stderr"] = _truncate(str(stderr), 4096)
            if working_dir is not None:
                data["working_dir"] = str(working_dir)
            if correlation_id is not None:
                data["correlation_id"] = str(correlation_id)
            # Reject obviously-malformed inputs silently — the AC index can't
            # be negative, and required strings can't be empty.
            if data["ac_index"] < 0 or not data["bead_id"] or not data["exec_id"]:
                return
            if not data["ac_text"] or not data["command"]:
                return
        except Exception as e:
            print(f"[obs] ac_verified payload build failed: {e}", file=sys.stderr)
            return
        self._publish(CHANNEL_AC_VERIFIED, data)

    # ------------------------------------------------------------------ #
    # Symbol lineage (Bitloops-style evidence graph)
    # ------------------------------------------------------------------ #

    def symbol_lineage(
        self,
        repo_id: str,
        session_id: str,
        checkpoint_id: str,
        lineage_kind: str,
        source_symbol_id: str,
        source_artefact_id: str,
        dest_symbol_id: str,
        dest_artefact_id: str,
        commit_sha: str = "",
        *,
        source_blob_sha: str | None = None,
        dest_blob_sha: str | None = None,
        agent: str = "",
    ) -> None:
        """Emit a symbol_evidence_lineage event (autobench.symbol.lineage.v1).

        Tracks symbol evolution across checkpoints. Mirrors Bitloops
        checkpoint_artefact_lineage. Lineage kinds: ``refactor``, ``extract``,
        ``inline``, ``rename_derived``, ``copy``.
        """
        data: dict[str, Any] = {
            "repo_id": str(repo_id),
            "session_id": str(session_id),
            "checkpoint_id": str(checkpoint_id),
            "lineage_kind": str(lineage_kind),
            "source_symbol_id": str(source_symbol_id),
            "source_artefact_id": str(source_artefact_id),
            "dest_symbol_id": str(dest_symbol_id),
            "dest_artefact_id": str(dest_artefact_id),
            "commit_sha": str(commit_sha),
        }
        if source_blob_sha is not None:
            data["source_blob_sha"] = str(source_blob_sha)
        if dest_blob_sha is not None:
            data["dest_blob_sha"] = str(dest_blob_sha)
        self._publish(CHANNEL_SYMBOL_LINEAGE, data)


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #

def make_observability(session_id: str | None = None) -> AutobenchObservability:
    """Create a pre-configured observability instance."""
    return AutobenchObservability(session_id=session_id)
