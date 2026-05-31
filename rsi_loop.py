"""Recursive Self-Improvement (RSI) loop implementation for autobench.

Classes:
    SelfImprovingHarness — wraps HarnessConfig with improvement loop
    improve_harness(current_harness, benchmark_results) — generates improved harness config
    convergence_check(iteration_history) — checks if improvement has plateaued
"""

from __future__ import annotations

import copy
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .core import HarnessConfig, HarnessResult, RSILoop, Verdict
from .evaluator import BenchmarkEvaluator, BenchmarkResult
from .iteration_summary import build_iteration_summary, normalize_worker_usage
from .observability import AutobenchObservability


# nervous-bus-sf0y: default variance floor (2σ) below which we treat a
# score regression as noise. Measured at 0.0274 by tools/variance_floor.
# Override via env ``AUTOBENCH_VARIANCE_FLOOR``. The same value also
# replaces the legacy hardcoded 0.01 improvement_threshold default so
# the early-exit no longer chases noise.
DEFAULT_VARIANCE_FLOOR_2SIGMA = 0.027


def _default_variance_floor() -> float:
    """Read the 2σ variance floor from env, falling back to 0.027."""
    raw = os.environ.get("AUTOBENCH_VARIANCE_FLOOR")
    if raw is None or raw.strip() == "":
        return DEFAULT_VARIANCE_FLOOR_2SIGMA
    try:
        v = float(raw)
        if v < 0.0:
            return DEFAULT_VARIANCE_FLOOR_2SIGMA
        return v
    except ValueError:
        return DEFAULT_VARIANCE_FLOOR_2SIGMA


@dataclass
class ImprovementDelta:
    """Tracks what changed between harness versions."""

    system_prompt_delta: str = ""
    rollout_protocol_changed: bool = False
    context_manager_changed: bool = False
    tool_surface_delta: str = ""
    budget_delta: dict[str, Any] = field(default_factory=dict)
    improvement_summary: str = ""
    delta_score: float = 0.0
    # AHE prediction contract (arXiv:2604.25850) — optional Prediction the
    # improver attached describing what it expects to happen next iteration.
    # ``None`` when the improver did not supply one (rule-based path, missing
    # block, or malformed LLM JSON).
    prediction: Any = None  # autobench.ahe.Prediction | None


@dataclass
class SelfImprovingHarness:
    """A harness that wraps itself in an RSI loop.

    Wraps: RSILoop + BenchmarkEvaluator + improver_fn to produce
    a self-improving harness: g(C) = perf_metric(improver_agent(harness_v1))

    Attributes:
        current_harness: The current best harness configuration.
        evaluator: BenchmarkEvaluator used to assess harness quality.
        max_iterations: Max RSI iterations before halting.
        improvement_threshold: Minimum score delta to count as improvement.
            Defaults to the 2σ variance floor (env
            ``AUTOBENCH_VARIANCE_FLOOR``, else 0.027 — measured under SICA
            weights 2026-05-16). The legacy 0.01 default sat below the
            noise floor and caused the loop to read pure measurement noise
            as "convergence". See nervous-bus-sf0y.
        variance_floor: 2σ noise floor used by the best-iter-keep
            checkpoint logic (nervous-bus-sf0y). When iter N regresses by
            more than ``variance_floor`` below the best-so-far score, the
            loop reverts its working harness to the best-iter checkpoint
            and emits ``autobench.rsi.checkpoint_revert.v1``.
    """

    current_harness: HarnessConfig
    evaluator: BenchmarkEvaluator
    max_iterations: int = 10
    improvement_threshold: float = field(default_factory=_default_variance_floor)
    default_improver: str | None = "minimax"  # "minimax" | "anthropic" | "rule_based" | None
    obs: AutobenchObservability | None = None
    budget_guard: Any = None  # autobench.budget_guard.BudgetGuard | None
    diversity_tracker: Any = None  # autobench.diversity.DiversityTracker | None
    # nervous-bus-bo86 (Phase 2 of wire-pop): cross-advocate sibling context.
    # PopulationRunner threads a list of recent sibling ImprovementDeltas
    # so the improver_fn can render a SIBLINGS block in its prompt. The
    # list is snapshotted at each improver call. None / empty → no block.
    cross_advocate_context: list[ImprovementDelta] | None = None
    variance_floor: float = field(default_factory=_default_variance_floor)
    _rsi_loop: RSILoop = field(init=False)
    _iteration_history: list[tuple[HarnessConfig, BenchmarkResult, ImprovementDelta]] = field(
        default_factory=list
    )
    # nervous-bus-sf0y: best-iter-keep state. ``_best_harness`` is the
    # snapshot of the harness that produced ``_best_score`` at iter
    # ``_best_iter``. ``_revert_history`` is a list of dicts (one per
    # fired revert) that's exposed to the improver's next diagnosis call
    # so it can reconsider rather than doubling down on a regressed edit.
    _best_harness: HarnessConfig | None = field(default=None, init=False)
    _best_score: float = field(default=float("-inf"), init=False)
    _best_iter: int = field(default=-1, init=False)
    _revert_history: list[dict[str, Any]] = field(default_factory=list, init=False)
    # nervous-bus-msqa (wire-pop Phase 5): record AHE PredictionVerifications
    # as they're computed so PopulationRunner/continuous.py promotion can pick
    # a candidate by latest confirmed-AHE outcome. Each entry is the
    # ``PredictionVerification`` object produced at the top of an iteration
    # for the prediction made in the prior iteration. Verifications for
    # ``refuted_live`` events are recorded separately on
    # ``_live_refuted_predictions`` so we never promote a lineage whose
    # most-recent prediction was killed mid-evaluation.
    _verifications: list[Any] = field(default_factory=list, init=False)
    _live_refuted_iterations: list[int] = field(default_factory=list, init=False)
    # nervous-bus-8d1d: record every pre-emission clip so the caller/accumulator
    # can surface them to the improver's next diagnosis prompt.
    _prediction_clips: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._rsi_loop = RSILoop(
            max_iterations=self.max_iterations,
            improvement_threshold=self.improvement_threshold,
        )

    def _resolve_improver_fn(self, improver_fn):
        """Resolve improver_fn, using default_improver if None provided.

        Supports:
          - "minimax"          → MiniMaxLLMWrapper (default; one call/iter)
          - "minimax_ensemble" → MultiImproverEnsemble (nervous-bus-9xd, wire-pop
                                 Phase 6) — fans out N anonymous MiniMax calls
                                 per iteration. Aggregation strategy is read
                                 from env ``AUTOBENCH_IMPROVER_STRATEGY``
                                 (vote|parallel, default vote). Fan-out size
                                 from ``AUTOBENCH_IMPROVER_ENSEMBLE_N``
                                 (default 3).
          - "anthropic"        → AnthropicLLMWrapper (legacy, requires
                                 ANTHROPIC_API_KEY)
          - "rule_based" / None / unknown → in-module rule-based improve_harness

        The returned callable accepts an optional ``iteration`` kwarg so the
        RSI loop can thread iteration numbers into reasoning capture. When
        a revert has fired (nervous-bus-sf0y), ``self._revert_history`` is
        non-empty and the wrapper threads it into the diagnosis prompt so
        the improver can see "iter N regressed by X; reverted" and
        reconsider rather than doubling down.
        """
        if improver_fn is not None:
            return improver_fn
        if self.default_improver == "minimax":
            from .minimax_improver import MiniMaxLLMWrapper
            wrapper = MiniMaxLLMWrapper()
            def _minimax_improve(
                h, r, iteration=0, revert_history=None,
                cross_advocate_context=None,
            ):
                return wrapper.improve(
                    h, r, obs=self.obs, iteration=iteration,
                    revert_history=revert_history,
                    cross_advocate_context=cross_advocate_context,
                )
            return _minimax_improve
        if self.default_improver == "minimax_ensemble":
            # nervous-bus-9xd: fan out N anonymous MiniMax improvers per
            # iteration and aggregate by vote (default) or best-arm
            # (env-opt-in). Wrapper factory builds a fresh
            # MiniMaxLLMWrapper per call so each arm is independent.
            from .multi_improver import MultiImproverEnsemble
            ensemble = MultiImproverEnsemble()
            def _ensemble_improve(
                h, r, iteration=0, revert_history=None,
                cross_advocate_context=None,
            ):
                return ensemble.improve(
                    h, r, obs=self.obs, iteration=iteration,
                    revert_history=revert_history,
                    cross_advocate_context=cross_advocate_context,
                )
            return _ensemble_improve
        if self.default_improver == "anthropic":
            from .llm_improver import AnthropicLLMWrapper
            wrapper = AnthropicLLMWrapper()

            def _anthropic_improve(
                h, r, iteration=0, revert_history=None,
                cross_advocate_context=None,
            ):
                # AnthropicLLMWrapper may not accept cross_advocate_context;
                # degrade gracefully without breaking the legacy path.
                kwargs: dict[str, Any] = {
                    "obs": self.obs,
                    "iteration": iteration,
                    "revert_history": revert_history,
                }
                try:
                    res = wrapper.suggest_harness_improvements(
                        h, r, cross_advocate_context=cross_advocate_context,
                        **kwargs,
                    )
                except TypeError:
                    res = wrapper.suggest_harness_improvements(h, r, **kwargs)
                return res.suggested_harness, res.delta

            return _anthropic_improve
        return improve_harness

    def improve(
        self,
        benchmark_cases: list[Any],
        improver_fn: Callable[[HarnessConfig, BenchmarkResult], tuple[HarnessConfig, ImprovementDelta]] | None = None,
    ) -> tuple[HarnessConfig, BenchmarkResult, list[tuple[HarnessConfig, BenchmarkResult, ImprovementDelta]]]:
        """Run the full RSI loop to convergence.

        Args:
            benchmark_cases: List of benchmark cases to evaluate against.
            improver_fn: Function that takes (current_harness, benchmark_result)
                         and returns (improved_harness, delta).
                         If None, uses default_improver or rule-based improve_harness.

        Returns:
            Tuple of (final_harness, final_benchmark_result, full_history).
        """
        resolved_improver = self._resolve_improver_fn(improver_fn)
        harness = self.current_harness
        result: BenchmarkResult | None = None
        history: list[tuple[HarnessConfig, BenchmarkResult, ImprovementDelta]] = []

        # Model name for improver-call emission
        if self.default_improver == "minimax":
            model_name = "minimax-m2.7"
        elif self.default_improver == "minimax_ensemble":
            model_name = "minimax-m2.7-ensemble"
        elif self.default_improver == "anthropic":
            model_name = "anthropic"
        else:
            model_name = "rule_based"

        # AHE prediction contract: track the pending prediction from the
        # previous iteration so we can verify it against this iteration's
        # actuals. ``prev_result`` stores last iteration's BenchmarkResult.
        pending_prediction: Any = None  # autobench.ahe.Prediction | None
        prev_result: BenchmarkResult | None = None

        for i in range(self.max_iterations):
            prev_score = result.aggregate_score if result else 0.0

            if self.obs:
                self.obs.iteration_start(iteration_num=i, harness_version=f"v{i}")

            # AHE live refutation (nervous-bus-ykn): if we entered this
            # iteration with a pending prediction from the previous one, wrap
            # the evaluator's per-case loop so that we can call
            # ``refute_live`` after each case lands and emit
            # ``prediction.refuted_live.v1`` the moment the prediction
            # becomes mathematically unachievable. Emit at most once per
            # prediction. ``pending_prediction is None`` (iter 0) skips
            # silently.
            live_refute_state = _begin_live_refutation(
                pending_prediction=pending_prediction,
                prev_result=prev_result,
                benchmark_cases=benchmark_cases,
                evaluator=self.evaluator,
                obs=self.obs,
                iteration=i,
            )
            try:
                # Evaluate current harness
                result = self.evaluator.run(harness, benchmark_cases, obs=self.obs)
            finally:
                # nervous-bus-msqa: record whether the prior prediction was
                # live-refuted during this iteration's per-case evaluation so
                # cross-run promotion can skip lineages with refuted-live tails.
                if live_refute_state and live_refute_state.get("emitted"):
                    self._live_refuted_iterations.append(i)
                _end_live_refutation(live_refute_state)

            # Budget guard checkpoint (optional). Moved here (nervous-bus-k5ni)
            # to check BEFORE any new prediction is emitted for the next iter.
            # If budget is exceeded, no new prediction is emitted.
            if self.budget_guard is not None:
                self.budget_guard.record_iteration_complete()
                ok, reason = self.budget_guard.check()
                if not ok:
                    self.budget_guard.halt(reason)  # raises BudgetExceeded

            # nervous-bus-sf0y: best-iter-keep checkpoint + revert. After
            # the score lands, decide whether this iteration is the new
            # best or a >variance_floor regression. Two effects:
            #   * new best  → snapshot the harness we just evaluated
            #   * regress   → emit checkpoint_revert.v1 AND swap ``harness``
            #                 back to the best-iter snapshot so the improver
            #                 edits from the known-good baseline instead of
            #                 doubling down on the regressed config.
            curr_score = result.aggregate_score
            if self._best_harness is None or curr_score > self._best_score:
                self._best_harness = copy.deepcopy(harness)
                self._best_score = curr_score
                self._best_iter = i
            elif curr_score < self._best_score - self.variance_floor:
                regression_delta = curr_score - self._best_score  # negative
                revert_entry = {
                    "iter_regressed": i,
                    "iter_score": curr_score,
                    "best_score": self._best_score,
                    "best_iter": self._best_iter,
                    "regression_delta": regression_delta,
                    "variance_floor_used": self.variance_floor,
                }
                self._revert_history.append(revert_entry)
                if self.obs is not None:
                    try:
                        self.obs.checkpoint_revert(
                            iter_regressed=i,
                            regression_delta=regression_delta,
                            reverted_to_iter=self._best_iter,
                            variance_floor_used=self.variance_floor,
                            best_score=self._best_score,
                            iter_score=curr_score,
                        )
                    except Exception:  # noqa: BLE001 — observability never breaks the loop
                        pass
                # Swap the working harness back to the best-iter snapshot.
                # The improver call below will see this as its baseline AND
                # receive the revert context via ``self._revert_history``.
                harness = copy.deepcopy(self._best_harness)

            # Failure-pattern detection (nervous-bus-46v): surface shared
            # leading-prefix clusters among non-OK cases (e.g. all CE'd cases
            # starting with `<think>` prose) so the improver — and a human
            # watching the dashboard — sees the structural signal before the
            # mutation step runs. Pure detection; emission is fire-and-forget.
            if self.obs is not None:
                try:
                    from .failure_pattern import detect_failure_patterns
                    patterns = detect_failure_patterns(result.case_results)
                    for p in patterns:
                        self.obs.failure_pattern(iteration=i, pattern=p)
                except Exception:  # noqa: BLE001 — observability never breaks the loop
                    pass

            # AHE: verify the previous iteration's prediction against actuals.
            if pending_prediction is not None and prev_result is not None and self.obs:
                try:
                    from .ahe import verify_prediction
                    # nervous-bus: extract mean dissent_ratio across cases so
                    # contested iterations are downweighted in the verification.
                    dissent_ratios_list = (
                        (result.metadata or {}).get("judge_pool_dissent_ratios") or []
                    )
                    if dissent_ratios_list:
                        mean_dissent = sum(
                            float(d.get("dissent_ratio", 0.0))
                            for d in dissent_ratios_list
                        ) / len(dissent_ratios_list)
                    else:
                        mean_dissent = 0.0
                    verification = verify_prediction(
                        pending_prediction, prev_result, result,
                        dissent_ratio=mean_dissent,
                    )
                    self.obs.improver_prediction_verified(iteration=i, verification=verification)
                    # nervous-bus-msqa: stash for cross-run promotion in
                    # PopulationRunner. The list order matches iteration order;
                    # the last element is the most-recent verification.
                    self._verifications.append(verification)
                except Exception:  # noqa: BLE001 — never break the loop
                    pass
                pending_prediction = None

            # Attempt improvement
            if self.obs:
                self.obs.improver_call_start(model=model_name, prompt_tokens=0)

            # Snapshot the pre-improvement harness for the heuristic baseline.
            harness_before = harness
            t_improver_start = time.monotonic()
            # nervous-bus-sf0y: try threading revert_history through to any
            # improver_fn that accepts it. Fall back through progressively
            # smaller signatures so legacy improvers without the kwarg / iteration
            # kwarg still work.
            _revert_snapshot = list(self._revert_history)
            # nervous-bus-bo86: snapshot cross-advocate context at call time
            # (the PopulationRunner may append to the shared list after this
            # advocate finishes). None / empty list → no SIBLINGS block.
            _xa_context = (
                list(self.cross_advocate_context)
                if self.cross_advocate_context
                else None
            )
            # Inject prior-iteration prediction clips into _xa_context so the
            # improver sees them in its SIBLINGS block diagnosis prompt. Each
            # clip dict carries: iteration, clip_reasons (list of strings).
            # Render as a pseudo-ImprovementDelta with a human-readable summary.
            if self._prediction_clips:
                for clip in self._prediction_clips:
                    # Format: "Prediction clip: OK was +8 but max headroom was +4 (prior=16, num_cases=20)"
                    reasons_str = "; ".join(str(r) for r in clip.get("clip_reasons", []))
                    clip_delta = {
                        "improvement_summary": f"Prediction clip: {reasons_str}",
                    }
                    if _xa_context is None:
                        _xa_context = [clip_delta]
                    else:
                        _xa_context.append(clip_delta)
                # Clips are consumed by this call — they only feed the immediate
                # next iteration's improver.
                self._prediction_clips.clear()
            try:
                harness, delta = resolved_improver(
                    harness, result, iteration=i,
                    revert_history=_revert_snapshot,
                    cross_advocate_context=_xa_context,
                )
            except TypeError:
                try:
                    harness, delta = resolved_improver(
                        harness, result, iteration=i,
                        revert_history=_revert_snapshot,
                    )
                except TypeError:
                    try:
                        harness, delta = resolved_improver(harness_before, result, iteration=i)
                    except TypeError:
                        # Caller-supplied improver_fn without iteration kwarg — degrade.
                        harness, delta = resolved_improver(harness_before, result)
            t_improver_ms = (time.monotonic() - t_improver_start) * 1000.0

            if self.obs:
                self.obs.improver_call_complete(
                    model=model_name,
                    completion_tokens=0,
                    delta_summary=delta.improvement_summary,
                )

                # If the active improver is rule-based, emit its reasoning
                # directly here — there is no LLM wrapper to do it for us.
                if model_name == "rule_based":
                    strategy_tag = _rule_based_strategy_tag(result)
                    self.obs.improver_reasoning(
                        model="rule_based",
                        iteration=i,
                        prompt=_build_diagnosis_prompt(harness_before, result),
                        raw_response=f"matched strategy: {strategy_tag}",
                        parsed_delta=asdict(delta),
                        parse_status="ok",
                        latency_ms=t_improver_ms,
                        cost_dollars=0.0,
                    )

                # Always compute the heuristic baseline and emit divergence.
                _, heuristic_delta = improve_harness(harness_before, result)
                self.obs.improver_divergence(
                    iteration=i,
                    llm_delta=asdict(delta),
                    heuristic_delta=asdict(heuristic_delta),
                )

            # AHE: emit the prediction the improver just made (if any). It
            # describes what it expects to happen in iteration i+1; we'll verify
            # at the top of the next loop turn.
            #
            # Pre-emission feasibility check (nervous-bus-8d1d): clip predicted
            # verdict-count deltas to physically feasible bounds BEFORE the
            # prediction gets an ID. An improver that proposes "CE: -12" when
            # only 4 CE cases exist is wasting its prediction slot on a
            # guaranteed-refute. We emit the clip event and skip the raw
            # prediction emission entirely when clipping occurs; the improver
            # learns from the clip reason in its next diagnosis prompt.
            if getattr(delta, "prediction", None) is not None and self.obs:
                from .ahe import clip_prediction_to_feasible

                prior_counts = dict(getattr(result, "verdict_counts", {}) or {})
                num_cases = len(getattr(result, "case_results", []) or [])
                clipped_pred, clip_reasons = clip_prediction_to_feasible(
                    delta.prediction, prior_counts, num_cases,
                )
                if clip_reasons:
                    # Infeasible: emit clipped event BEFORE ID assignment,
                    # do NOT emit the raw prediction, record clip in ledger.
                    self.obs.improver_prediction_clipped(
                        iteration=i,
                        original=delta.prediction,
                        clipped=clipped_pred,
                        clip_reasons=clip_reasons,
                    )
                    # Stash clip in accumulator so caller can track.
                    self._prediction_clips.append(
                        {"iteration": i, "clip_reasons": clip_reasons}
                    )
                else:
                    # Feasible: assign identity and emit the prediction normally.
                    from .ahe import prediction_fingerprint as _pred_fingerprint
                    from .observability import _ulid

                    if not delta.prediction.prediction_id:
                        delta.prediction.prediction_id = _ulid()
                    delta.prediction.fact_fingerprint = _pred_fingerprint(
                        delta.prediction
                    )
                    if not getattr(delta.prediction, "source_scope_key", None):
                        _target = getattr(self, "_current_target", "default")
                        delta.prediction.source_scope_key = (
                            f"ahe:{self.obs.session_id}:{_target}:{i}"
                        )
                    self.obs.improver_prediction(
                        iteration=i, prediction=delta.prediction,
                        model=model_name,
                    )
                    pending_prediction = delta.prediction
            prev_result = result

            # Clean before/after harness diff at the iter N → N+1 boundary
            # (nervous-bus-utm). Always emit — no_change=true is signal too.
            if self.obs:
                try:
                    self.obs.improver_delta_diff(
                        iteration=i + 1, before=harness_before, after=harness,
                    )
                except Exception:  # noqa: BLE001 — observability must not break the loop
                    pass

            if self.diversity_tracker is not None:
                self.diversity_tracker.record(delta)

            # Detect a no-op improver delta (fallback path / malformed LLM
            # response / rule-based "no change"). Per nervous-bus-dxuq an
            # empty delta must NOT halt the run — the threshold-based early
            # exit below would otherwise misread "improver did nothing" as
            # "improvement has plateaued" and break after iter 0.
            is_noop_delta = (
                not delta.improvement_summary
                and not delta.system_prompt_delta
                and not delta.rollout_protocol_changed
                and not delta.context_manager_changed
                and not delta.tool_surface_delta
                and not delta.budget_delta
            )

            history.append((harness, result, delta))
            # Mirror live history so convergence_check can read deltas.
            self._iteration_history = history

            if self.obs:
                self.obs.iteration_complete(
                    iteration_num=i,
                    aggregate_score=result.aggregate_score,
                    verdict_counts=dict(result.verdict_counts),
                    improvement_delta={
                        "system_prompt_delta": delta.system_prompt_delta,
                        "rollout_protocol_changed": delta.rollout_protocol_changed,
                        "context_manager_changed": delta.context_manager_changed,
                        "tool_surface_delta": delta.tool_surface_delta,
                        "budget_delta": delta.budget_delta,
                        "improvement_summary": delta.improvement_summary,
                    },
                    harness_version=f"v{i}",
                )

                # iteration.summary.v1 rollup. Consumers (pulse dashboard,
                # continuous.py surprise digest, deer-flow analytics) use
                # this instead of re-aggregating from case.result events.
                # Worker cost/tokens come from evaluator's per-iteration
                # buffer of ._last_usage snapshots populated in _run_case.
                try:
                    worker_usage = (result.metadata or {}).get("worker_usage") or []
                    # Normalise each per-call usage dict so key-name drift across
                    # the worker's OpenAI vs Anthropic shapes (or any future
                    # producer using `cost` / `tokens` shorthand) does not silently
                    # drop the rollup to zero. See nervous-bus-ch4o.
                    normalised = [normalize_worker_usage(u) for u in worker_usage]
                    worker_costs = [u["cost_usd"] for u in normalised]
                    worker_tokens = [int(u["tokens"]) for u in normalised]
                    summary = build_iteration_summary(
                        iteration=i,
                        harness=harness_before,
                        result=result,
                        worker_call_costs=worker_costs or None,
                        worker_call_tokens=worker_tokens or None,
                        harness_version=f"v{i}",
                    )
                    if is_noop_delta:
                        summary["noop_iteration"] = True
                    self.obs.iteration_summary(**summary)
                except Exception:  # noqa: BLE001 — observability never breaks the loop
                    pass

            # Convergence check
            if self.convergence_check([(h, r) for h, r, _ in history]):
                break

            # Early exit if no meaningful improvement. Skip when the improver
            # produced a no-op delta (nervous-bus-dxuq): the threshold check
            # measures whether *attempted* improvements plateaued; an empty
            # delta means no attempt was made, so we must keep iterating.
            curr_score = result.aggregate_score
            if not is_noop_delta and abs(curr_score - prev_score) < self.improvement_threshold:
                break

        self.current_harness = harness
        self._iteration_history = history
        return harness, result, history

    def _compute_adaptive_threshold(
        self,
        deltas: list[float],
    ) -> tuple[float, float, float]:
        """Compute adaptive threshold from a sliding window of score deltas.

        Uses the last 5 iterations (or fewer if not available) to compute
        velocity (mean delta) and variance (std delta). The effective
        threshold is selected based on velocity/variance regime:

        - High velocity (>0.05) + low variance (<0.02): tight (0.015 above floor)
        - Moderate velocity (>0.02) + moderate variance (<0.05): floor (0.027)
        - Low velocity (<0.02) OR high variance (>0.05): relaxed (0.04 above floor)

        Returns (effective_threshold, velocity, variance).
        """
        floor = self.variance_floor
        if len(deltas) < 2:
            return floor, 0.0, 0.0

        velocity = statistics.mean(deltas)
        variance = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

        if velocity > 0.05 and variance < 0.02:
            effective = floor + 0.015
        elif velocity > 0.02 and variance < 0.05:
            effective = floor
        else:
            effective = floor + 0.04

        return effective, velocity, variance

    def convergence_check(
        self,
        iteration_history: list[tuple[HarnessConfig, BenchmarkResult]],
    ) -> bool:
        """Check if improvement has plateaued.

        Uses adjusted utility (raw aggregate_score - SACS diversity penalty)
        when a diversity_tracker is attached; falls back to raw scores
        otherwise. Per research/diversity_penalty_2026.md §4, the penalty
        lives here, NOT inside evaluator.score_harness — that contract stays
        a pure function of (harness, cases) so A/B comparisons remain valid.

        The convergence threshold is adaptive: when the sliding-window
        velocity is high and variance is low, a tighter threshold allows
        the loop to exit sooner. When improvement has plateaued (low
        velocity or high variance), the threshold is relaxed to allow
        more iterations before declaring convergence.
        """
        if len(iteration_history) < 3:
            return False

        tracker = self.diversity_tracker
        adjusted: list[float] = []
        for idx, (_, r) in enumerate(iteration_history[-3:]):
            penalty = 0.0
            if tracker is not None:
                hist_idx = len(iteration_history) - 3 + idx
                if 0 <= hist_idx < len(self._iteration_history):
                    _, _, d = self._iteration_history[hist_idx]
                    penalty = tracker.penalty_for(d)
            adjusted.append(r.aggregate_score - penalty)

        # Compute score deltas across the sliding window
        deltas = [adjusted[i] - adjusted[i - 1] for i in range(1, len(adjusted))]
        effective_threshold, velocity, variance = self._compute_adaptive_threshold(deltas)

        if self.obs is not None:
            self.obs.improver_threshold_adapted(
                effective_threshold=effective_threshold,
                velocity=velocity,
                variance=variance,
                iterations_used=len(deltas),
            )

        if all(abs(d) < effective_threshold for d in deltas):
            return True
        return False

    def run_single_iteration(
        self,
        benchmark_cases: list[Any],
        improver_fn: Callable[[HarnessConfig, BenchmarkResult], tuple[HarnessConfig, ImprovementDelta]] | None = None,
    ) -> tuple[HarnessConfig, BenchmarkResult, ImprovementDelta]:
        """Run a single RSI iteration (evaluate + improve).

        Returns:
            Tuple of (harness, result, delta) for this iteration.
        """
        resolved = self._resolve_improver_fn(improver_fn)
        # Evaluate
        result = self.evaluator.run(self.current_harness, benchmark_cases)

        # Improve
        harness, delta = resolved(self.current_harness, result)
        self.current_harness = harness

        return harness, result, delta


def improve_harness(
    current_harness: HarnessConfig,
    benchmark_result: BenchmarkResult,
    agent_fn: Callable[[str], str] | None = None,
) -> tuple[HarnessConfig, ImprovementDelta]:
    """Generate an improved harness configuration from benchmark results.

    This is the default improver_fn factory. It analyzes verdict distribution
    and score to produce targeted harness modifications.

    Args:
        current_harness: The current harness configuration.
        benchmark_result: Results from the last benchmark run.
        agent_fn: Optional LLM call (prompt -> str) to generate improvements.
                  If None, uses rule-based heuristics.

    Returns:
        Tuple of (improved_harness, delta).
    """
    verdict_counts = benchmark_result.verdict_counts
    total_cases = len(benchmark_result.case_results)

    # Analyze what went wrong
    ce_count = verdict_counts.get("CE", 0)
    re_count = verdict_counts.get("RE", 0)
    tle_count = verdict_counts.get("TLE", 0)
    mle_count = verdict_counts.get("MLE", 0)
    wa_count = verdict_counts.get("WA", 0)
    ok_count = verdict_counts.get("OK", 0)

    delta = ImprovementDelta()
    new_harness = HarnessConfig(
        system_prompt=current_harness.system_prompt,
        rollout_protocol=current_harness.rollout_protocol,
        context_manager=current_harness.context_manager,
        tool_surface=current_harness.tool_surface,
        verifiers=current_harness.verifiers,
        budget=current_harness.budget.copy(),
    )

    # Strategy: diagnose dominant verdict and adjust harness accordingly
    if agent_fn:
        # Use LLM to generate improvements
        diagnosis_prompt = _build_diagnosis_prompt(current_harness, benchmark_result)
        improvement_text = agent_fn(diagnosis_prompt)
        new_harness, delta = _parse_llm_improvement(current_harness, improvement_text)
    else:
        # Rule-based improvement
        if ce_count > 0 and ce_count >= total_cases * 0.3:
            # Many compilation errors → simplify code constraints
            new_harness.budget["max_tokens"] = int(current_harness.budget.get("max_tokens", 8192) * 0.8)
            delta.improvement_summary = f"Reduced max_tokens due to {ce_count} CE ({ce_count/total_cases:.0%})"
            delta.budget_delta = new_harness.budget.copy()

        elif tle_count > 0 and tle_count >= total_cases * 0.2:
            # Timeouts → tighten time budget and switch to iterative
            new_harness.budget["max_time_seconds"] = int(
                current_harness.budget.get("max_time_seconds", 30) * 0.8
            )
            if new_harness.rollout_protocol.value == "single":
                from .core import RolloutProtocol

                new_harness.rollout_protocol = RolloutProtocol.ITERATIVE
                delta.rollout_protocol_changed = True
            delta.improvement_summary = f"Tightened time budget due to {tle_count} TLE ({tle_count/total_cases:.0%})"

        elif wa_count > 0 and wa_count >= total_cases * 0.3:
            # Many wrong answers → improve system prompt with more examples
            from .core import ContextManager

            if new_harness.context_manager == ContextManager.FULL:
                new_harness.context_manager = ContextManager.HIERARCHICAL
                delta.context_manager_changed = True
            delta.improvement_summary = f"Switched to hierarchical context due to {wa_count} WA ({wa_count/total_cases:.0%})"

        elif re_count > 0 and re_count >= total_cases * 0.2:
            # Runtime errors → improve tool surface clarity
            delta.tool_surface_delta = "Added error-handling guidance to tool surface"
            new_harness.tool_surface = current_harness.tool_surface + "\n- Always handle exceptions"
            delta.improvement_summary = f"Enhanced tool surface due to {re_count} RE ({re_count/total_cases:.0%})"

        elif ok_count / total_cases >= 0.9:
            # Already very good — minor refinement
            delta.improvement_summary = "High pass rate; minor refinement"
            new_harness.budget["max_tokens"] = int(current_harness.budget.get("max_tokens", 8192) * 1.1)

        else:
            delta.improvement_summary = "Incremental improvement; balanced adjustments"
            # Proportional scaling
            new_harness.budget["max_tokens"] = int(current_harness.budget.get("max_tokens", 8192) * 0.95)

    delta.delta_score = benchmark_result.aggregate_score
    return new_harness, delta


def _rule_based_strategy_tag(result: BenchmarkResult) -> str:
    """Return a short tag identifying which rule-based branch fired.

    Mirrors the branching in :func:`improve_harness` so the reasoning event
    can name the strategy that produced the delta. Pure read-only inspection.
    """
    verdict_counts = result.verdict_counts
    total = len(result.case_results) or 1
    ce = verdict_counts.get("CE", 0)
    tle = verdict_counts.get("TLE", 0)
    wa = verdict_counts.get("WA", 0)
    re_ = verdict_counts.get("RE", 0)
    ok = verdict_counts.get("OK", 0)
    if ce > 0 and ce >= total * 0.3:
        return "ce_dominant_reduce_max_tokens"
    if tle > 0 and tle >= total * 0.2:
        return "tle_dominant_tighten_time_budget"
    if wa > 0 and wa >= total * 0.3:
        return "wa_dominant_hierarchical_context"
    if re_ > 0 and re_ >= total * 0.2:
        return "re_dominant_tool_surface_error_handling"
    if ok / total >= 0.9:
        return "ok_high_pass_rate_minor_refinement"
    return "balanced_proportional_scaling"


def _build_diagnosis_prompt(harness: HarnessConfig, result: BenchmarkResult) -> str:
    """Build a prompt for an LLM improver agent."""
    verdict_pct = {
        k: f"{v/len(result.case_results)*100:.0f}%" for k, v in result.verdict_counts.items()
    }

    # nervous-bus-19ur: full system_prompt/tool_surface (no truncation). Delta
    # is appended each iteration; the improver must see the accumulated text.
    return f"""You are improving a coding agent harness. Analyze the benchmark results and generate an improved harness configuration.

Current harness:
- system_prompt: {harness.system_prompt}
- rollout_protocol: {harness.rollout_protocol.value}
- context_manager: {harness.context_manager.value}
- tool_surface: {harness.tool_surface}
- budget: {harness.budget}

Benchmark results (aggregate score: {result.aggregate_score:.3f}):
- verdict_counts: {verdict_pct}
- total_latency_ms: {result.total_latency_ms:.0f}
- pass_rate: {result.pass_rate():.1%}

Based on this data, generate a JSON object with the fields you would change:
{{
  "system_prompt_changes": "what to add/change in system_prompt",
  "rollout_protocol": "single|iterative|self_revision|monte_carlo|keep",
  "context_manager": "full|budgeted|semantic|hierarchical|keep",
  "tool_surface_changes": "what to add/change in tool_surface",
  "budget_changes": {{"max_tokens": N, "max_time_seconds": N, "max_cost_dollars": N}},
  "rationale": "brief explanation of the changes"
}}

Return ONLY valid JSON, no markdown.
"""


def _parse_llm_improvement(
    current: HarnessConfig,
    improvement_text: str,
) -> tuple[HarnessConfig, ImprovementDelta]:
    """Parse LLM output into a new HarnessConfig + ImprovementDelta."""
    import json, re

    delta = ImprovementDelta()

    # Try to extract JSON from response
    json_match = re.search(r"\{[\s\S]*\}", improvement_text)
    if not json_match:
        return current, delta

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return current, delta

    from .core import ContextManager, RolloutProtocol, Verifier

    new_harness = HarnessConfig(
        system_prompt=current.system_prompt,
        rollout_protocol=current.rollout_protocol,
        context_manager=current.context_manager,
        tool_surface=current.tool_surface,
        verifiers=current.verifiers,
        budget=current.budget.copy(),
    )

    delta.system_prompt_delta = parsed.get("system_prompt_changes", "")
    if delta.system_prompt_delta:
        new_harness.system_prompt = current.system_prompt + "\n" + delta.system_prompt_delta

    rp = parsed.get("rollout_protocol", "keep")
    if rp != "keep":
        try:
            new_harness.rollout_protocol = RolloutProtocol(rp)
            delta.rollout_protocol_changed = True
        except ValueError:
            pass

    cm = parsed.get("context_manager", "keep")
    if cm != "keep":
        try:
            new_harness.context_manager = ContextManager(cm)
            delta.context_manager_changed = True
        except ValueError:
            pass

    ts = parsed.get("tool_surface_changes", "")
    if ts and ts.strip().lower() not in {"", "keep", "no change", "no_change", "nochange", "none", "null"}:
        # nervous-bus-ldd1: reject phantom tools — harness has no machinery
        # to materialize improver-proposed callables. See minimax_improver.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[ldd1] improver attempted to propose tool_surface change; "
            "ignored: %r", ts[:100],
        )

    bc = parsed.get("budget_changes", {})
    if bc:
        delta.budget_delta = bc
        new_harness.budget.update(bc)

    delta.improvement_summary = parsed.get("rationale", "LLM-generated improvement")
    return new_harness, delta


def convergence_check(
    iteration_history: list[tuple[HarnessConfig, BenchmarkResult]],
    window: int = 3,
    threshold: float = 0.01,
) -> bool:
    """Check if RSI improvement has plateaued.

    Args:
        iteration_history: List of (harness, result) from each iteration.
        window: Number of recent iterations to compare.
        threshold: Minimum delta to count as improvement.

    Returns:
        True if the last `window` iterations show no meaningful improvement.
    """
    if len(iteration_history) < window:
        return False

    recent_scores = [r.aggregate_score for _, r in iteration_history[-window:]]
    deltas = [abs(recent_scores[i] - recent_scores[i - 1]) for i in range(1, len(recent_scores))]

    return all(d < threshold for d in deltas)


# --------------------------------------------------------------------------- #
# AHE live refutation wiring (nervous-bus-ykn).
#
# These helpers monkey-patch the evaluator's ``_run_case`` for the duration of
# one ``evaluator.run()`` call so we can intercept each per-case verdict, fold
# it into a running actuals histogram, and call ``ahe.refute_live`` after each
# case. The moment a prediction becomes mathematically unachievable we emit
# ``autobench.improver.prediction.refuted_live.v1`` exactly once for that
# prediction.
#
# Implemented as module-level functions (rather than inside the class body) so
# sibling agents editing the class can avoid merge conflicts.
# --------------------------------------------------------------------------- #


def _begin_live_refutation(
    pending_prediction: Any,
    prev_result: Any,
    benchmark_cases: list[Any],
    evaluator: Any,
    obs: AutobenchObservability | None,
    iteration: int,
) -> dict[str, Any] | None:
    """Install a per-case interceptor on ``evaluator._run_case`` for live refute.

    Returns a state dict the caller passes to ``_end_live_refutation`` for
    cleanup, or ``None`` when there's nothing to do (no pending prediction,
    no observability, no prior result, missing ``_run_case``).
    """
    if (
        pending_prediction is None
        or prev_result is None
        or obs is None
        or not hasattr(evaluator, "_run_case")
    ):
        return None

    # Pull prior-iter verdict counts using the same helper ``verify_prediction``
    # uses, so prediction-class keys match (handles Verdict-enum normalisation).
    try:
        from .ahe import _verdict_counts, refute_live
    except Exception:  # noqa: BLE001
        return None

    try:
        prior_counts = _verdict_counts(prev_result)
    except Exception:  # noqa: BLE001
        prior_counts = {}

    state: dict[str, Any] = {
        "evaluator": evaluator,
        "original_run_case": evaluator._run_case,
        "actuals_so_far": {},
        "cases_done": 0,
        "total_cases": len(benchmark_cases),
        "emitted": False,
    }

    original_run_case = state["original_run_case"]

    def _wrapped_run_case(harness, case, obs=None, iteration=0, *args, **kwargs):
        result = original_run_case(
            harness, case, obs=obs, iteration=iteration, *args, **kwargs
        )
        # Best-effort: extract verdict letter and fold into actuals.
        try:
            verdict = getattr(result, "verdict", None)
            key = verdict.value if hasattr(verdict, "value") else (
                str(verdict) if verdict is not None else None
            )
            if key is not None:
                state["actuals_so_far"][key] = state["actuals_so_far"].get(key, 0) + 1
            state["cases_done"] += 1

            if not state["emitted"]:
                remaining = max(0, state["total_cases"] - state["cases_done"])
                status = refute_live(
                    pending_prediction,
                    state["actuals_so_far"],
                    remaining,
                    prior_counts,
                )
                if status.is_refuted:
                    obs_local = state.get("obs") or None
                    # ``obs`` from the wrapper signature is the per-call obs;
                    # we want the RSI-loop obs we captured in closure.
                    try:
                        _emit_live_refute(state["captured_obs"], iteration_n_plus_one=state["iteration"], status=status)
                    except Exception:  # noqa: BLE001
                        pass
                    state["emitted"] = True
        except Exception:  # noqa: BLE001 — never break the loop
            pass
        return result

    state["captured_obs"] = obs
    state["iteration"] = iteration

    # Install the wrapper. Restoration happens in _end_live_refutation.
    evaluator._run_case = _wrapped_run_case  # type: ignore[assignment]
    return state


def _end_live_refutation(state: dict[str, Any] | None) -> None:
    """Restore the evaluator's original ``_run_case`` after one evaluator.run."""
    if not state:
        return
    try:
        state["evaluator"]._run_case = state["original_run_case"]  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass


def _emit_live_refute(
    obs: AutobenchObservability,
    iteration_n_plus_one: int,
    status: Any,
) -> None:
    """Tiny shim so the call site stays one line; non-raising."""
    try:
        obs.prediction_refuted_live(iteration=iteration_n_plus_one, status=status)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m autobench.rsi_loop")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--benchmark", required=True)
    run_p.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    if args.cmd == "run":
        import json
        from pathlib import Path

        benchmark_path = Path(args.benchmark)
        if benchmark_path.is_dir():
            candidates = []
            for d in os.listdir(benchmark_path):
                p = benchmark_path / d
                if p.is_dir():
                    candidates.append(p / "cases.jsonl")
            all_candidates = [benchmark_path / "cases.jsonl"] + sorted(candidates)
            cases_file = None
            for c in all_candidates:
                if c.is_file():
                    cases_file = c
                    break
            if cases_file is None:
                sys.exit(f"No cases.jsonl found under {benchmark_path}")
        else:
            cases_file = benchmark_path

        from .core import HarnessConfig
        from .evaluator import BenchmarkCase, BenchmarkEvaluator

        cases = [BenchmarkCase(**json.loads(line)) for line in open(cases_file)]
        evaluator = BenchmarkEvaluator()
        harness = HarnessConfig()
        sih = SelfImprovingHarness(
            current_harness=harness,
            evaluator=evaluator,
            max_iterations=args.iterations,
            default_improver="rule_based",
        )
        result = sih.improve(cases)
        print(f"Score: {result[1].aggregate_score:.3f}, {len(result[2])} iterations")
