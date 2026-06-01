"""Live observability layer for autobench.

Emits phase / iteration / sandbox / improver events on the nervous-bus so that a
single autobench run is visible end-to-end in real time. Designed to be
**non-blocking and never raise** — observability must not corrupt the
correctness of the harness.

Channels:
    autobench.phase.v1      — start/complete/error for a named phase
    autobench.iteration.v1  — RSI iteration start/complete + aggregate scores
    autobench.sandbox.v1    — per-case sandbox dispatch + completion
    autobench.improver.v1   — improver model call boundaries

Emission mechanism mirrors AutobenchResultPublisher: try `zellij pipe`, fall
back to ~/.cache/nervous-bus/debug.jsonl.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator


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
# Channel constants
# --------------------------------------------------------------------------- #

CHANNEL_PHASE = "autobench.phase.v1"
CHANNEL_ITERATION = "autobench.iteration.v1"
CHANNEL_SANDBOX = "autobench.sandbox.v1"
CHANNEL_IMPROVER = "autobench.improver.v1"
CHANNEL_IMPROVER_REASONING = "autobench.improver.reasoning.v1"
CHANNEL_IMPROVER_DIVERGENCE = "autobench.improver.divergence.v1"
CHANNEL_CASE_RESULT = "autobench.case.result.v1"
CHANNEL_THRESHOLD_ADAPTED = "autobench.improver.convergence.threshold_adapted.v1"
CHANNEL_CURRICULUM_PROBLEM = "autobench.curriculum.problem.v1"
CHANNEL_CURRICULUM_PROBLEM_REJECTED = "autobench.curriculum.problem.rejected.v1"
CHANNEL_CURRICULUM_CYCLE = "autobench.curriculum.cycle.v1"
CHANNEL_PREDICTION = "autobench.improver.prediction.v1"
CHANNEL_PREDICTION_VERIFIED = "autobench.improver.prediction.verified.v1"
# nervous-bus-8d1d: emitted when an improver-proposed prediction was
# clipped to feasible verdict-count deltas before persistence.
CHANNEL_PREDICTION_CLIPPED = "autobench.improver.prediction.clipped.v1"
# nervous-bus-sf0y: emitted when an iteration regressed > variance_floor
# (default 2σ ≈ 0.027) below the best-so-far iter and the RSI loop
# reverted its working harness to the best-iter checkpoint.
CHANNEL_CHECKPOINT_REVERT = "autobench.rsi.checkpoint_revert.v1"
# nervous-bus-c48: anonymous N-judge pool wired into the live evaluator loop.
# pool.verdict fires per case, disagreement fires when dissent_ratio > threshold.
CHANNEL_JUDGE_POOL_VERDICT = "autobench.judge.pool.verdict.v1"
CHANNEL_JUDGE_DISAGREEMENT = "autobench.judge.disagreement.v1"
# nervous-bus-9xd (wire-pop Phase 6): emitted once per RSI iteration when
# the active improver is the multi-improver ensemble. Records each
# anonymous instance's delta summary + the aggregator's vote outcome.
CHANNEL_IMPROVER_ENSEMBLE = "autobench.improver.ensemble.v1"
# nervous-bus-6kwv: emitted once per acceptance-criteria bullet executed during
# a hearth-loom loomie run. Foundation for closing the phantom-AC-passes gap
# (sibling epic hearth-loom-loom-o4cc5). The Python emitter lives here for SDK
# convenience and for autobench-side post-cycle AC verification; the primary
# producer is hearth-loom's internal/executor/verify_ac_gate.go.
CHANNEL_AC_VERIFIED = "hearth-loom.ac.verified.v1"
CHANNEL_SYMBOL_LINEAGE = "autobench.symbol.lineage.v1"

# Maximum bytes of generated code retained per event. Keeping this small (4 KiB)
# bounds the bus payload and the debug-file growth while still being long enough
# for AST feature extraction over typical competitive-programming solutions.
GENERATED_CODE_TRUNCATE_LEN = 4096

SOURCE = "/autobench"


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
# Convenience factory
# --------------------------------------------------------------------------- #

def make_observability(session_id: str | None = None) -> AutobenchObservability:
    """Create a pre-configured observability instance."""
    return AutobenchObservability(session_id=session_id)


# --------------------------------------------------------------------------- #
# SACS diversity channel (autobench.diversity.v1) — attached via assignment
# to stay out of the class body (sibling agents edit it concurrently).
# See autobench/diversity.py and research/diversity_penalty_2026.md.
# --------------------------------------------------------------------------- #

CHANNEL_DIVERSITY = "autobench.diversity.v1"


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

CHANNEL_ADVERSARIAL_GENERATED = "autobench.adversarial.curveball_generated.v1"
CHANNEL_ADVERSARIAL_ROUND = "autobench.adversarial.round_complete.v1"


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

CHANNEL_CONTINUOUS_SESSION = "autobench.continuous.session_complete.v1"
CHANNEL_CONTINUOUS_DIGEST = "autobench.continuous.digest.v1"


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

CHANNEL_PROMOTION_DECISION = "autobench.continuous.promotion_decision.v1"


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

CHANNEL_WORKER = "autobench.worker.v1"


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

CHANNEL_SANDBOX_STDERR = "autobench.sandbox.stderr.v1"
CHANNEL_FAILURE_CATEGORY = "autobench.failure.category.v1"

# Maximum characters of stderr retained per event. Kept tight (200) so the
# bus payload stays scannable and one event fits comfortably on a single
# pulse-dashboard row. The full stderr (up to 500 chars) remains available
# on HarnessResult.error.
SANDBOX_STDERR_EXCERPT_LEN = 200


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

CHANNEL_WORKER_QUEUE_PRESSURE = "autobench.worker.queue_pressure.v1"


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

CHANNEL_ITERATION_SUMMARY = "autobench.iteration.summary.v1"


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

CHANNEL_FAILURE_PATTERN = "autobench.failure_pattern.v1"


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

CHANNEL_PREDICTION_REFUTED_LIVE = "autobench.improver.prediction.refuted_live.v1"
CHANNEL_SCORING_ADAPTED = "autobench.scoring.weights_adapted.v1"


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

CHANNEL_DELTA_DIFF = "autobench.improver.delta.diff.v1"


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

CHANNEL_POPULATION_SUMMARY = "autobench.population.summary.v1"


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

CHANNEL_CROSS_DOMAIN_EVALUATION = "autobench.cross_domain.evaluation.v1"


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

CHANNEL_BENCH_COMPLETED = "bus.bead.bench_completed.v1"


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

CHANNEL_CYCLE_REQUESTED = "autobench.cycle.requested.v1"
CHANNEL_CYCLE_REPORT = "autobench.cycle.report.v1"


def _schemas_dir() -> Path:
    """Return the repo's ``schemas/`` directory.

    The observability module lives at ``<repo>/autobench/observability.py``;
    schemas live at ``<repo>/schemas/<channel>.v<n>.json``.
    """
    return Path(__file__).resolve().parents[1] / "schemas"


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

CHANNEL_BUS_NOTIFY = "bus.notify.v1"


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
