"""Evaluation with verdict-level signals for autobench.

Classes:
    BenchmarkEvaluator — runs harness against benchmark suite
    emit_verdict(code_output, stderr, runtime, memory) — CE/RE/TLE/MLE/WA/OK
    score_harness(harness_results[], utility_weights) — U = 0.5·score + 0.25·cost + 0.25·time (SICA formula)
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import HarnessConfig, HarnessResult, RolloutProtocol, Verdict


# nervous-bus-c48 (wire-pop Phase 7): default anonymous-judge ensemble size for
# every case verdict in the live evaluator loop. Each MiniMax-coding-plan
# request counts toward the 14250 req / 5h cap, so the default of 5 keeps the
# usual 20-case × 3-advocate × 5-iter cycle at 1500 judge calls — well under
# budget — while making judge-variance the dominant signal rather than a
# single noisy verdict. Override with the AUTOBENCH_JUDGES_PER_CASE env var;
# n=1 disables the pool entirely (bit-for-bit pre-wire behavior).
DEFAULT_JUDGES_PER_CASE = 5
DEFAULT_DISSENT_THRESHOLD = 0.4

# Cap retries for the iterative rollout protocol so a chronically-failing case
# can't burn the whole iteration budget. Each attempt emits its own
# case.result.v1 event with attempt=1,2,...,N (nervous-bus-x3os).
ITERATIVE_MAX_ATTEMPTS = 3

# Fallback per-case cost anchor when harness.budget has no max_cost_dollars.
# 0.10 USD/case is a conservative ceiling for a single MiniMax/Sonnet call
# (~10k tokens at current pricing). Used to normalize real worker cost into
# p_cost ∈ [0, 1] — see _normalize_p_cost (nervous-bus-569q).
DEFAULT_MAX_COST_PER_CASE_USD = 0.10
from .observability import GENERATED_CODE_TRUNCATE_LEN, AutobenchObservability
from .sandbox import ExecutionResult, SandboxedExecutor, compile_and_run, verify_output


# Verdict aggregation precedence (worst-wins) for multi-input cases
# (nervous-bus-uwjh). Mirrors emit_verdict's internal precedence so the
# per-input loop's aggregate matches a single-shot run when the case has
# one input. Refactor verdicts (RV/RD/RT) and the shader-only VF verdict
# are kept out — the python sandbox path never produces them, so they
# stay terminal-only via _run_shader_case / refactor_verifier.
_VERDICT_PRECEDENCE = [
    Verdict.CE,
    Verdict.RE,
    Verdict.TLE,
    Verdict.MLE,
    Verdict.VF,
    Verdict.WA,
    Verdict.OK,
]


def _build_revision_context(prior: HarnessResult) -> str:
    """Format a Reflexion-style revision prompt suffix from a failed attempt.

    SELF_REVISION (nervous-bus-dils) feeds this string back into generate_fn
    on the second pass so the worker sees its own execution feedback:
    verdict, stderr excerpt, observed stdout, expected stdout. Truncated to
    keep the revision prompt bounded.
    """
    md = prior.metadata or {}
    stderr = (md.get("stderr") or prior.error or "")[:800]
    stdout = (md.get("stdout") or "")[:400]
    expected = (md.get("expected_output") or "")[:400]
    parts = [
        f"Your prior solution received verdict {prior.verdict.value} (not OK).",
        f"Stderr:\n{stderr}" if stderr else "Stderr: <empty>",
        f"Observed stdout:\n{stdout}" if stdout else "Observed stdout: <empty>",
        f"Expected stdout:\n{expected}" if expected else "",
        "Produce a corrected solution. Output code only.",
    ]
    return "\n\n".join(p for p in parts if p)


def _worst_verdict(verdicts: Any) -> Verdict:
    """Return the worst (highest-precedence) verdict from an iterable."""
    seen = set(verdicts)
    if not seen:
        return Verdict.OK
    for v in _VERDICT_PRECEDENCE:
        if v in seen:
            return v
    # Unknown verdict — fall back to the first one we saw.
    return next(iter(seen))


# Default SICA utility weights.
#
# nervous-bus-isgo: cost term dropped to 0.0 because p_cost is no longer
# a meaningful signal — dq7l removed all hardcoded $/token tables (MiniMax
# coding plan bills by requests-per-5h, not dollars). Leaving cost weighted
# at 0.25 made every aggregate_score include a constant 0.25 baseline
# (since 1 - 0 = 1, scaled by 0.25), compressing dynamic range from [0,1]
# to [0.25,1] and deflating noise-floor analysis.
#
# Redistribution: 0.5 score / 0.5 time. Request-rate (the real billing
# unit) lives on RateBudgetGuard, NOT in the utility formula. Historical
# scores under the old weights are NOT directly comparable; the variance
# floor at 2026-05-16 must be re-measured under these weights before any
# new improver gain can be claimed.
DEFAULT_WEIGHTS = {
    "score": 0.5,
    "cost": 0.0,  # isgo: kept for back-compat but contributes nothing.
    "time": 0.5,
}


@dataclass
class BenchmarkCase:
    """A single benchmark test case.

    Attributes:
        id: Unique identifier for this case.
        prompt: The problem statement / coding prompt.
        language: Target language for the solution.
        expected_output: Expected stdout output (or JSON for structured).
            Used as the single broadcast expected output when
            ``expected_outputs`` is empty or len-mismatched against
            ``test_inputs``.
        expected_outputs: Optional per-input expected outputs (nervous-bus-tqhd).
            When non-empty AND ``len(expected_outputs) == len(test_inputs)``,
            the evaluator pairs ``(test_inputs[i], expected_outputs[i])``,
            unlocking asymmetric edge tests where different inputs yield
            different correct outputs. Otherwise falls back to the singular
            ``expected_output`` for every input (legacy behavior preserved).
        constraints: Execution constraints (time, memory).
        starter_code: Optional starter code skeleton.
        test_inputs: Optional list of inputs to run through the solution.
        metadata: Arbitrary extra data (difficulty, category, etc.).
    """

    id: str
    prompt: str
    language: str = "python"
    expected_output: str = ""
    expected_outputs: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(
        default_factory=lambda: {
            "max_time_seconds": 10,
            "max_memory_mb": 512,
        }
    )
    starter_code: str = ""
    test_inputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    shader_artifact_path: str = ""
    silo_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "language": self.language,
            "expected_output": self.expected_output,
            "expected_outputs": self.expected_outputs,
            "constraints": self.constraints,
            "starter_code": self.starter_code,
            "test_inputs": self.test_inputs,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Result of running a full benchmark (all cases).

    Attributes:
        case_results: Per-case results.
        aggregate_score: Weighted aggregate score (0.0–1.0).
        total_latency_ms: Total wall-clock time.
        verdict_counts: Dict mapping verdict types to counts.
        metadata: Extra data.
    """

    case_results: list[HarnessResult] = field(default_factory=list)
    aggregate_score: float = 0.0
    total_latency_ms: float = 0.0
    verdict_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for r in self.case_results if r.is_pass()) / len(self.case_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_results": [r.to_dict() for r in self.case_results],
            "aggregate_score": self.aggregate_score,
            "total_latency_ms": self.total_latency_ms,
            "verdict_counts": self.verdict_counts,
            "pass_rate": self.pass_rate(),
            "metadata": self.metadata,
        }


def _find_last_usage(fn: Any) -> dict[str, Any] | None:
    """Locate the ``_last_usage`` dict on a worker-like callable.

    MiniMaxWorker stores per-call usage on the instance. Callers commonly
    wrap the worker in a closure (``def _worker_callable(prompt, cfg): ...``)
    or a ``functools.partial`` for accounting, which hides instance
    attributes from a flat ``getattr`` on the outer callable. We probe:

    1. The object itself (worker instance passed directly).
    2. ``__self__`` (bound method).
    3. ``func`` (``functools.partial``) — recursively.
    4. ``__wrapped__`` (``functools.wraps``) — recursively.
    5. Each free variable in ``__closure__`` (capturing closures).

    First match wins. Returns ``None`` if nothing exposes ``_last_usage``.
    """
    seen: set[int] = set()

    def _probe(obj: Any) -> dict[str, Any] | None:
        if obj is None:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        usage = getattr(obj, "_last_usage", None)
        if isinstance(usage, dict):
            return usage

        bound_self = getattr(obj, "__self__", None)
        if bound_self is not None:
            found = _probe(bound_self)
            if found is not None:
                return found

        partial_func = getattr(obj, "func", None)
        if callable(partial_func):
            found = _probe(partial_func)
            if found is not None:
                return found

        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is not None:
            found = _probe(wrapped)
            if found is not None:
                return found

        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                found = _probe(val)
                if found is not None:
                    return found

        return None

    return _probe(fn)


def _normalize_p_cost(usage: dict[str, Any] | None, max_cost_per_case: float) -> float:
    """Convert real per-call worker cost into a normalized p_cost ∈ [0, 1].

    Formula (nervous-bus-569q):
        p_cost = min(1.0, cost_usd / max_cost_per_case)

    where ``cost_usd`` comes from the worker's ``_last_usage`` dict and
    ``max_cost_per_case`` is ``harness.budget["max_cost_dollars"] / num_cases``
    (computed in ``_run_inner``), falling back to
    ``DEFAULT_MAX_COST_PER_CASE_USD`` when the budget is missing or zero.

    Interpretation: higher cost → higher p_cost → lower utility (the SICA
    formula inverts cost as ``1 - avg_cost`` in ``score_harness``). When
    usage is missing (no worker, stub generate_fn), we return 0.0 rather
    than the legacy 0.5 stub — 0.0 means "this run consumed no measurable
    spend", which is accurate for offline/cached evaluations and removes
    the 0.0625-of-noise contribution Angela's audit flagged.
    """
    if not isinstance(usage, dict):
        return 0.0
    # Match the canonical keys produced by worker_agent._update_last_usage
    # plus the alternate keys handled in iteration_summary.normalize_worker_usage.
    raw_cost = (
        usage.get("cost_usd")
        if usage.get("cost_usd") is not None
        else usage.get("total_cost_usd")
        if usage.get("total_cost_usd") is not None
        else usage.get("cost")
        if usage.get("cost") is not None
        else usage.get("total_cost", 0.0)
    )
    try:
        cost = float(raw_cost or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if cost <= 0.0:
        return 0.0
    anchor = max_cost_per_case if max_cost_per_case > 0 else DEFAULT_MAX_COST_PER_CASE_USD
    return max(0.0, min(1.0, cost / anchor))


class BenchmarkEvaluator:
    """Runs a harness against a benchmark suite.

    The evaluator:
    1. Takes a HarnessConfig and a list of BenchmarkCases
    2. Calls the harness's improver_agent to generate code for each case
    3. Compiles and runs the generated code in a sandbox
    4. Emits verdict-level signals (CE/RE/TLE/MLE/WA/OK)
    5. Scores the results using the SICA formula

    The harness is intentionally agnostic to HOW code is generated —
    it calls `harness.generate_fn(prompt, config)` which the caller wires up
    to an agent (e.g., Claude API via sdk/claude, a local model, etc.).
    """

    def __init__(
        self,
        generate_fn: Any = None,  # Callable[[str, HarnessConfig], str] | None
        executor: SandboxedExecutor | None = None,
        weights: dict[str, float] | None = None,
        emit_signals: bool = False,
        obs: AutobenchObservability | None = None,
        judge_factory: Any = None,  # Callable[[str, dict], dict] | None
        judges_per_case: int | None = None,
        dissent_threshold: float | None = None,
        weights_fn: Any = None,  # Callable[[dict, float, float], dict[str, float]] | None
    ):
        """Initialize evaluator.

        Args:
            generate_fn: Function that takes (prompt: str, config: HarnessConfig)
                          and returns generated code (str).
                          If None, uses a stub that returns empty string.
            executor: SandboxedExecutor instance for running code.
            weights: Utility weights for SICA scoring. Only used when
                     ``weights_fn is None``. When ``weights_fn`` is set,
                     ``weights`` is the initial weights before adaptation.
            emit_signals: If True, emit each result to the nervous-bus via
                          AutobenchResultPublisher after each case is evaluated.
            judge_factory: Optional callable that scores a single worker output
                           in the JudgingPool protocol — signature
                           ``(prompt: str, context: dict) -> dict`` with keys
                           ``judge_id, verdict, p_score, p_cost, p_time, reasoning``.
                           When provided AND ``judges_per_case > 1``, the
                           evaluator fans out N anonymous calls per case via
                           threading, aggregates them through ``JudgingPool``,
                           and emits ``autobench.judge.pool.verdict.v1`` plus
                           an ``autobench.judge.disagreement.v1`` escalation
                           when the dissent ratio exceeds ``dissent_threshold``.
                           When None OR ``judges_per_case == 1``, the legacy
                           single-verdict path runs bit-for-bit unchanged
                           (no pool events, no metadata changes — nervous-bus-c48).
            judges_per_case: Anonymous-judge ensemble size. Falls back to the
                           ``AUTOBENCH_JUDGES_PER_CASE`` env var, then to
                           ``DEFAULT_JUDGES_PER_CASE`` (5). Clamped to ``>= 1``.
                           A value of 1 disables the pool regardless of
                           ``judge_factory``.
            dissent_threshold: Fraction in [0, 1]; when the per-case dissent
                           ratio strictly exceeds it, an
                           ``autobench.judge.disagreement.v1`` event fires.
                           Defaults to ``DEFAULT_DISSENT_THRESHOLD`` (0.4).
            weights_fn: Optional callable for adaptive SICA weights. Signature:
                        ``(harness: HarnessConfig, score_variance: float,
                           score_velocity: float) -> dict[str, float]``.
                        Receives the current harness budget, measured 2-sigma
                        variance of score_delta over the lookback window, and
                        mean signed score_delta (positive = improving).
                        ``score_variance`` and ``score_velocity`` are 0.0 on
                        the first iteration. When set, ``AUTOBENCH_ADAPTIVE_WEIGHTS``
                        env var must be 1 to activate (otherwise falls back
                        to ``weights``). The callable is invoked at the start
                        of each ``run()`` call; the resulting weights apply
                        to that iteration's ``aggregate_score``.
        """
        self.generate_fn = generate_fn or (lambda prompt, cfg: "")
        self.executor = executor or SandboxedExecutor()
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.emit_signals = emit_signals
        self.obs = obs
        self._publisher: Any = None

        # JudgingPool wiring (nervous-bus-c48). When judge_factory is None we
        # never construct a pool — every code path below short-circuits and
        # the legacy single-verdict behavior is preserved bit-for-bit.
        self.judge_factory = judge_factory
        if judges_per_case is None:
            env_val = os.environ.get("AUTOBENCH_JUDGES_PER_CASE")
            if env_val is not None:
                try:
                    judges_per_case = int(env_val)
                except ValueError:
                    judges_per_case = DEFAULT_JUDGES_PER_CASE
            else:
                judges_per_case = DEFAULT_JUDGES_PER_CASE
        self.judges_per_case = max(1, int(judges_per_case))

        # Default-factory injection: when callers want >1 judge but didn't
        # supply a factory, fall back to the MiniMax-backed default IF the
        # API key is present. Tests and explicit-factory callers see no
        # change. Set AUTOBENCH_DISABLE_DEFAULT_JUDGE=1 to opt out.
        if (
            self.judge_factory is None
            and self.judges_per_case > 1
            and os.environ.get("MINIMAX_API_KEY")
            and not os.environ.get("AUTOBENCH_DISABLE_DEFAULT_JUDGE")
        ):
            try:
                from autobench.default_judge_factory import make_minimax_judge_factory
                self.judge_factory = make_minimax_judge_factory()
            except Exception:
                # Never let factory construction prevent evaluator init —
                # judges silently stay off, which is the prior behavior.
                pass
        self.dissent_threshold = (
            float(dissent_threshold)
            if dissent_threshold is not None
            else DEFAULT_DISSENT_THRESHOLD
        )

        # Adaptive weights (nervous-bus-isgo continuation).
        # weights_fn is only activated when AUTOBENCH_ADAPTIVE_WEIGHTS=1.
        self._weights_fn = weights_fn
        self._adaptive_enabled = (
            os.environ.get("AUTOBENCH_ADAPTIVE_WEIGHTS", "").lower() in {"1", "true", "yes"}
            if weights_fn is not None
            else False
        )
        # Per-iteration state for adaptive weight computation.
        # score_history: list of (iteration, aggregate_score) from each run().
        self._score_history: list[tuple[int, float]] = []
        # Variance floor — same value used by RSI loop checkpoint logic.
        self._variance_floor = self._read_variance_floor()

    @staticmethod
    def _read_variance_floor() -> float:
        """Read 2σ variance floor, matching rsi_loop.py logic."""
        raw = os.environ.get("AUTOBENCH_VARIANCE_FLOOR")
        if raw is None or raw.strip() == "":
            return 0.027
        try:
            v = float(raw)
            if v < 0.0:
                return 0.027
            return v
        except ValueError:
            return 0.027

    def _compute_score_stats(self) -> tuple[float, float]:
        """Compute score_variance (2σ) and score_velocity over the lookback window.

        Returns (variance_2sigma, velocity_mean). Returns (0.0, 0.0) when
        fewer than 2 data points exist. The lookback window is the last 5
        iterations to keep the signal responsive without over-smoothing.
        """
        window = self._score_history[-5:]
        if len(window) < 2:
            return 0.0, 0.0
        scores = [s for _, s in window]
        # score_delta per iteration
        deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        # 2-sigma of deltas
        mean_delta = sum(deltas) / len(deltas)
        if len(deltas) == 1:
            variance_2sigma = abs(deltas[0])
        else:
            # population stdev * 2 (2-sigma)
            mean_sq = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
            import math
            variance_2sigma = 2.0 * math.sqrt(mean_sq)
        return variance_2sigma, mean_delta

    def _reason_from_stats(
        self,
        score_variance: float,
        score_velocity: float,
        harness: HarnessConfig,
    ) -> str:
        """Determine the adaptation reason string from current stats."""
        budget = harness.budget or {}
        remaining_time = float(budget.get("max_wall_time_seconds", 0.0) or 0.0)
        total_time = float(budget.get("max_wall_time_seconds", 0.0) or 3600.0)
        budget_exhausted = total_time > 0 and remaining_time / total_time < 0.1

        if score_variance > self._variance_floor:
            return "plateau_detected"
        elif score_variance > 0 and score_variance < self._variance_floor:
            return "high_variance"
        elif score_velocity > self._variance_floor:
            return "improving"
        elif budget_exhausted:
            return "budget_exhausted"
        else:
            return "early_iteration"

    def _compute_adaptive_weights(
        self,
        harness: HarnessConfig,
        iteration: int,
    ) -> dict[str, float]:
        """Compute adaptive SICA weights based on score variance and velocity.

        Strategy (nervous-bus-isgo):
        - High score variance (plateau): shift weight to cost and time, which
          carry real improvement signal when score gains are within noise.
        - High score velocity (improving): weight score heavily to amplify gains.
        - Early iterations (score_variance == 0.0): use conservative defaults.
        - Budget near exhaustion: shift toward cost/time to find efficiency gains.

        Returns renormalized weights dict summing to 1.0.
        """
        score_variance, score_velocity = self._compute_score_stats()
        budget = harness.budget or {}
        remaining_time = float(budget.get("max_wall_time_seconds", 0.0) or 0.0)
        total_time = float(budget.get("max_wall_time_seconds", 0.0) or 3600.0)
        budget_exhausted = total_time > 0 and remaining_time / total_time < 0.1

        # High variance plateau: shift 40% of score weight to cost/time.
        # score_variance > variance_floor means deltas are within noise,
        # so score changes are unreliable — cost/time become the signal.
        if score_variance > self._variance_floor:
            reason = "plateau_detected"
            w_score = 0.1   # almost ignore score when it's pure noise
            w_cost = 0.45  # cost has real signal even on plateaus
            w_time = 0.45  # time likewise
        elif score_variance > 0 and score_variance < self._variance_floor:
            # Noise-level variance but not zero — still a plateau but with
            # smaller deltas; split more conservatively.
            reason = "high_variance"
            w_score = 0.25
            w_cost = 0.375
            w_time = 0.375
        elif score_velocity > self._variance_floor:
            # Actively improving — amplify score signal.
            reason = "improving"
            w_score = 0.65
            w_cost = 0.175
            w_time = 0.175
        elif budget_exhausted:
            # Budget running low — find efficiency wins.
            reason = "budget_exhausted"
            w_score = 0.3
            w_cost = 0.4
            w_time = 0.3
        else:
            # Early iteration or neutral — balanced defaults.
            reason = "early_iteration"
            w_score = 0.4
            w_cost = 0.3
            w_time = 0.3

        weights = {"score": w_score, "cost": w_cost, "time": w_time}
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _emit_weights_adapted(
        self,
        iteration: int,
        weights: dict[str, float],
        score_variance: float,
        score_velocity: float,
        reason: str,
    ) -> None:
        """Emit weights_adapted event if obs is available."""
        if self.obs is not None:
            try:
                self.obs.weights_adapted(
                    iteration=iteration,
                    weights=weights,
                    score_variance=score_variance,
                    score_velocity=score_velocity,
                    reason=reason,
                )
            except Exception:
                # observability must never raise
                pass

    def _ensure_publisher(self, harness: HarnessConfig) -> Any:
        """Lazily create a publisher for the given harness version."""
        if self._publisher is None:
            try:
                from .signal_bus import AutobenchResultPublisher
            except Exception:
                return None
            self._publisher = AutobenchResultPublisher(
                harness_version=harness.version if hasattr(harness, "version") else "v0",
                benchmark_name=getattr(harness, "name", "default"),
                iteration=0,
            )
        return self._publisher

    def run(
        self,
        harness: HarnessConfig,
        cases: list[BenchmarkCase],
        obs: AutobenchObservability | None = None,
        iteration: int = 0,
    ) -> BenchmarkResult:
        """Run a harness against a list of benchmark cases.

        Args:
            harness: HarnessConfig with generation settings.
            cases: List of BenchmarkCase objects.
            obs: Optional AutobenchObservability instance overriding self.obs.
            iteration: RSI iteration number, threaded through to per-case
                observability emission so case.result events carry the correct
                iteration tag (otherwise AHE prediction verification cannot
                match actuals against the prior iteration's prediction).

        Returns:
            BenchmarkResult with per-case results and aggregates.
        """
        # Adaptive weights: compute and apply before the run when enabled.
        if self._adaptive_enabled and self._weights_fn is not None:
            score_variance, score_velocity = self._compute_score_stats()
            weights = self._weights_fn(harness, score_variance, score_velocity)
            # Renormalize to 1.0
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            reason = self._reason_from_stats(
                score_variance, score_velocity, harness
            )
            self._emit_weights_adapted(
                iteration, weights, score_variance, score_velocity, reason
            )
            self.weights = weights

        active_obs = obs if obs is not None else self.obs

        if active_obs is not None:
            with active_obs.phase("benchmark", num_cases=len(cases)):
                result = self._run_inner(harness, cases, active_obs, iteration)
        else:
            result = self._run_inner(harness, cases, None, iteration)

        # Record score for variance / velocity tracking in future iterations.
        self._score_history.append((iteration, result.aggregate_score))

        return result

    def _run_inner(
        self,
        harness: HarnessConfig,
        cases: list[BenchmarkCase],
        obs: AutobenchObservability | None,
        iteration: int = 0,
    ) -> BenchmarkResult:
        results: list[HarnessResult] = []
        total_latency = 0.0
        verdict_counts: dict[str, int] = {}
        # Per-iteration worker usage buffer (nervous-bus-91u plumbing fix).
        # MiniMaxWorker exposes ._last_usage with prompt_tokens/completion_tokens/
        # cost_usd after each call; we snapshot it post-generate_fn in _run_case
        # so iteration.summary.v1 has real cost/token rollups instead of zeros.
        self._iter_worker_usage: list[dict[str, Any]] = []

        is_iterative = harness.rollout_protocol == RolloutProtocol.ITERATIVE
        is_self_revision = harness.rollout_protocol == RolloutProtocol.SELF_REVISION
        # SELF_REVISION caps at 2 attempts (one original + one revision pass);
        # ITERATIVE caps at ITERATIVE_MAX_ATTEMPTS (nervous-bus-dils, x3os).
        max_attempts = (
            ITERATIVE_MAX_ATTEMPTS if is_iterative
            else (2 if is_self_revision else 1)
        )

        # Per-case cost anchor for p_cost normalization (nervous-bus-569q).
        # harness.budget["max_cost_dollars"] is the TOTAL budget for the
        # iteration; divide by num_cases to get a per-case ceiling. Fall
        # back to DEFAULT_MAX_COST_PER_CASE_USD when budget is unset/zero
        # or no cases (defensive — should not happen in practice).
        budget_total = float(
            (harness.budget or {}).get("max_cost_dollars", 0.0) or 0.0
        )
        num_cases = max(1, len(cases))
        max_cost_per_case = (
            budget_total / num_cases if budget_total > 0 else DEFAULT_MAX_COST_PER_CASE_USD
        )

        for case in cases:
            # Iterative rollout: re-run on non-pass verdicts up to max_attempts,
            # emitting one case.result.v1 event per attempt (attempt=1..N) so
            # downstream consumers can distinguish first-try-OK from retry-OK
            # (nervous-bus-x3os).
            result: HarnessResult | None = None
            revision_context = ""
            for attempt in range(1, max_attempts + 1):
                result = self._run_case(
                    harness, case, obs=obs, iteration=iteration, attempt=attempt,
                    max_cost_per_case=max_cost_per_case,
                    revision_context=revision_context,
                )
                total_latency += result.latency_ms
                if result.is_pass() or not (is_iterative or is_self_revision):
                    break
                # SELF_REVISION: build feedback for the (single) revision pass.
                # Reflexion-style — show the worker its own stderr, verdict,
                # and observed-vs-expected output so it can self-correct.
                if is_self_revision and attempt == 1:
                    revision_context = _build_revision_context(result)
            assert result is not None  # max_attempts >= 1
            results.append(result)

            if self.emit_signals:
                pub = self._ensure_publisher(harness)
                if pub is not None:
                    # Refresh iteration on the cached publisher so the event
                    # reflects the CURRENT RSI iteration, not the snapshot
                    # taken at evaluator construction (nervous-bus-4e0x).
                    pub.iteration = iteration
                    pub.publish(result)

            vc_key = result.verdict.value
            verdict_counts[vc_key] = verdict_counts.get(vc_key, 0) + 1

        aggregate_score = self.score_harness(results, self.weights)

        # nervous-bus-c48: surface judge-pool dissent on the envelope so
        # downstream consumers (AHE, promotion, calibration) can downweight
        # contested cases without re-walking every result.metadata dict.
        # Empty list when the pool was disabled (judges_per_case <= 1 or no
        # judge_factory) so the envelope shape stays stable.
        dissent_ratios: list[dict[str, Any]] = []
        contested_count = 0
        for r in results:
            jp = (r.metadata or {}).get("judge_pool")
            if not isinstance(jp, dict):
                continue
            dr = float(jp.get("dissent_ratio", 0.0))
            dissent_ratios.append({
                "case_id": (r.metadata or {}).get("case_id", ""),
                "dissent_ratio": dr,
                "consensus_verdict": str(jp.get("consensus_verdict", "")),
                "n_votes": int(jp.get("n_votes", 0)),
            })
            if dr > self.dissent_threshold:
                contested_count += 1

        envelope_metadata: dict[str, Any] = {
            "worker_usage": list(self._iter_worker_usage),
            "judge_pool_dissent_ratios": dissent_ratios,
            "judge_pool_contested_count": contested_count,
            "judge_pool_dissent_threshold": self.dissent_threshold,
            "judges_per_case": self.judges_per_case,
        }

        return BenchmarkResult(
            case_results=results,
            aggregate_score=aggregate_score,
            total_latency_ms=total_latency,
            verdict_counts=verdict_counts,
            metadata=envelope_metadata,
        )

    def _run_case(
        self,
        harness: HarnessConfig,
        case: BenchmarkCase,
        obs: AutobenchObservability | None = None,
        iteration: int = 0,
        attempt: int = 1,
        max_cost_per_case: float = DEFAULT_MAX_COST_PER_CASE_USD,
        revision_context: str = "",
    ) -> HarnessResult:
        """Run a single benchmark case.

        1. Generate code via self.generate_fn
        2. Execute in sandbox
        3. Emit verdict
        4. Normalize scores

        When `revision_context` is non-empty (SELF_REVISION protocol, attempt 2),
        the prior attempt's execution feedback (stderr, verdict, observed vs
        expected output) is appended to the prompt so the worker can produce a
        revised solution. The original prompt is preserved verbatim above.
        """
        start = time.perf_counter()
        code = ""
        error = ""
        usage: dict[str, Any] | None = None
        prompt = (
            f"{case.prompt}\n\n--- Prior attempt feedback ---\n{revision_context}"
            if revision_context
            else case.prompt
        )

        try:
            # Generate code
            code = self.generate_fn(prompt, harness)
            # Snapshot worker usage if generate_fn (or something it wraps)
            # exposes ._last_usage. MiniMaxWorker stashes it on the instance
            # after every call; run_first.py and the adversarial dual wrap
            # the worker in a closure/partial, so a flat getattr on the
            # outer callable returns None. We unwrap to find the real worker.
            usage = _find_last_usage(self.generate_fn)
            if isinstance(usage, dict) and hasattr(self, "_iter_worker_usage"):
                self._iter_worker_usage.append(dict(usage))
        except Exception as e:
            error = f"Generation error: {e}"
            return HarnessResult(
                verdict=Verdict.RE,
                error=error,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Shader benchmarks bypass SandboxedExecutor — they need a GL
        # context, and "sandboxing" for shader-as-code is a different
        # threat model (validated GLSL, no syscalls). Route to
        # ShaderExecutor when language=='glsl'.
        if case.language.lower() == "glsl":
            return self._run_shader_case(
                case, code, obs=obs, iteration=iteration, attempt=attempt,
            )

        # Execute in sandbox — iterate over ALL test_inputs so a case with
        # multiple inputs reports the worst-of verdict and a partial-credit
        # p_score (nervous-bus-uwjh). Empty test_inputs still runs once with
        # an empty stdin to preserve legacy single-shot behavior.
        constraints = {**harness.budget, **case.constraints}
        inputs = case.test_inputs if case.test_inputs else [""]

        # Per-input expected-output selection (nervous-bus-tqhd). When the
        # case carries an expected_outputs list whose length matches the
        # test_inputs list, pair element-wise so asymmetric edge tests
        # (different input → different correct output) work. Otherwise
        # broadcast the singular expected_output to every input — that is
        # the pre-tqhd behavior every existing tier-1 case relies on.
        use_per_input_expected = (
            bool(case.expected_outputs)
            and len(case.expected_outputs) == len(inputs)
        )

        per_input: list[dict[str, Any]] = []
        last_exec: ExecutionResult | None = None
        total_latency_ms = 0.0

        prev_executor_obs = getattr(self.executor, "obs", None)
        if obs is not None:
            self.executor.obs = obs
        try:
            for i, stdin_input in enumerate(inputs):
                exec_result = self.executor.execute(
                    code=code,
                    language=case.language,
                    constraints=constraints,
                    stdin=stdin_input,
                    case_id=case.id,
                )
                last_exec = exec_result
                expected = (
                    case.expected_outputs[i]
                    if use_per_input_expected
                    else case.expected_output
                )
                v = self.emit_verdict(
                    code_output=exec_result.stdout,
                    stderr=exec_result.stderr,
                    runtime_ms=exec_result.latency_ms,
                    memory_mb=constraints.get("max_memory_mb", 512),
                    exit_code=exec_result.exit_code,
                    expected_output=expected,
                    constraints=constraints,
                )
                total_latency_ms += exec_result.latency_ms
                per_input.append({
                    "verdict": v.value,
                    "p_score": 1.0 if v == Verdict.OK else 0.0,
                    "latency_ms": exec_result.latency_ms,
                })
        finally:
            if obs is not None:
                self.executor.obs = prev_executor_obs

        assert last_exec is not None  # inputs is always non-empty
        exec_result = last_exec  # surfaced fields below (stderr, stdout) for back-compat
        verdict = _worst_verdict(Verdict(r["verdict"]) for r in per_input)

        # Normalize p_time on TOTAL latency vs case budget (sum across inputs).
        max_time = constraints.get("max_time_seconds", 10)
        p_time = max(0.0, min(1.0, 1.0 - (total_latency_ms / 1000.0 / max_time)))

        # p_cost: derive from real per-call worker cost via _normalize_p_cost
        # (nervous-bus-569q). Pre-fix this was hardcoded 0.5, which the SICA
        # formula weights at 0.25 → 0.0625 of pure noise added to every
        # aggregate_score regardless of actual spend. The real cost flows
        # through MiniMaxWorker._last_usage → _find_last_usage(self.generate_fn).
        p_cost = _normalize_p_cost(usage, max_cost_per_case)

        # p_score: fraction of inputs that achieved OK (partial credit).
        p_score = sum(r["p_score"] for r in per_input) / len(per_input)

        # Capture the generated code so downstream tooling (AST feature
        # extraction, failure clustering, diagnoser) can inspect what the agent
        # actually produced. Truncate to GENERATED_CODE_TRUNCATE_LEN to bound
        # the per-result payload while keeping the original length as a summary.
        full_len = len(code)
        captured_code = code[:GENERATED_CODE_TRUNCATE_LEN] if code else ""

        result = HarnessResult(
            p_score=p_score,
            p_cost=p_cost,
            p_time=p_time,
            verdict=verdict,
            error=exec_result.stderr[:500] if exec_result.stderr else "",
            latency_ms=total_latency_ms,
            metadata={
                "case_id": case.id,
                "language": case.language,
                "generated_code": captured_code,
                "generated_code_length": full_len,
                "per_input_results": list(per_input),
                # SELF_REVISION (nervous-bus-dils) reads these on attempt 1
                # to build the revision_context fed to attempt 2.
                "stderr": exec_result.stderr or "",
                "stdout": exec_result.stdout or "",
                "expected_output": case.expected_output,
            },
        )

        # Emit a per-case result event mirroring sandbox_complete's pattern.
        # `obs` here is the active observability instance forwarded from
        # _run_inner (which already resolves self.obs vs the run() override).
        active_obs = obs if obs is not None else self.obs
        if active_obs is not None:
            try:
                active_obs.case_result(
                    case_id=case.id,
                    iteration=iteration,
                    language=case.language,
                    verdict=verdict.value,
                    p_score=p_score,
                    latency_ms=total_latency_ms,
                    generated_code=captured_code,
                    generated_code_length=full_len,
                    attempt=attempt,
                    per_input_results=per_input if len(per_input) > 1 else None,
                )
            except Exception:
                pass

            # Bead nervous-bus-bns: when the sandbox produces an error-class
            # verdict, additionally emit the first 200 chars of stderr so
            # downstream consumers can see WHY without parsing generated_code.
            # OK and WA are skipped — WA is wrong-answer, not an error signal.
            if verdict.value in {"CE", "RE", "TLE", "MLE"}:
                try:
                    active_obs.sandbox_stderr(
                        case_id=case.id,
                        iteration=iteration,
                        verdict=verdict.value,
                        stderr_excerpt=exec_result.stderr or "",
                        exit_code=exec_result.exit_code,
                        language=case.language,
                    )
                except Exception:
                    pass

        # nervous-bus-c48: anonymous JudgingPool fan-out for the live loop.
        # The pool runs ONLY when a judge_factory is configured AND
        # judges_per_case > 1 — otherwise the result above is the final
        # verdict, bit-for-bit identical to the pre-wire single-judge path.
        # _score_with_pool mutates `result.metadata["judge_pool"]` in place
        # and emits the two new bus events. It never raises.
        if self.judge_factory is not None and self.judges_per_case > 1:
            try:
                self._score_with_pool(
                    case=case,
                    result=result,
                    code=code,
                    expected_output=case.expected_output,
                    iteration=iteration,
                    obs=active_obs,
                )
            except Exception:
                # Observability MUST NOT break the harness — fall through to
                # the legacy single-verdict result on any pool failure.
                pass

        return result

    def _score_with_pool(
        self,
        case: BenchmarkCase,
        result: HarnessResult,
        code: str,
        expected_output: str,
        iteration: int,
        obs: AutobenchObservability | None,
    ) -> tuple[Verdict, float, list[JudgeVote]]:
        """Fan out N anonymous judges in parallel, aggregate, emit bus events.

        Each invocation of ``self.judge_factory`` is wrapped in a fresh
        anonymous slot wrapper so the factory cannot rely on cross-call
        state — slot index is the only identity exposed in emitted votes.
        Judges run on a thread pool (one ``threading.Thread`` per slot)
        so the total wall-clock for N MiniMax calls is dominated by the
        slowest single call, not their sum.

        Mutates ``result.metadata['judge_pool']`` with:
            n_judges, n_votes, consensus_verdict, dissent_ratio,
            verdict_distribution, consensus_p_score, votes_summary.

        Emits ``autobench.judge.pool.verdict.v1`` per case, plus
        ``autobench.judge.disagreement.v1`` when dissent_ratio strictly
        exceeds ``self.dissent_threshold``.

        Returns the (consensus_verdict, dissent_ratio, votes) triple for
        callers that want to act on the pool's view directly. The HarnessResult
        is NOT replaced — the sandbox verdict remains authoritative for
        verdict/p_score; the pool's verdict is a parallel signal recorded in
        metadata so downstream consumers (AHE, promotion) can join them.
        """
        if self.judge_factory is None or self.judges_per_case <= 1:
            # Defensive — callers already gate on this, but never trust them.
            return result.verdict, 0.0, []

        n = self.judges_per_case
        base_factory = self.judge_factory

        # Build N anonymous wrappers. Each wrapper owns its slot index, calls
        # the underlying factory fresh, and forces a unique judge_id in the
        # returned dict so cross-judge identity collisions can't sneak in.
        # Each wrapper is a brand-new function object — no module-level state
        # is shared across slots, satisfying "anonymous = fresh instance".
        def _make_anon_wrapper(slot_idx: int) -> Any:
            def _anon(prompt: str, context: dict[str, Any]) -> dict[str, Any]:
                raw = base_factory(prompt=prompt, context=context)
                if not isinstance(raw, dict):
                    return {"judge_id": f"anon-{slot_idx}", "verdict": "OK",
                            "p_score": 0.0, "p_cost": 0.0, "p_time": 0.0}
                out = dict(raw)
                out["judge_id"] = f"anon-{slot_idx}"
                return out
            return _anon

        anon_judges = [_make_anon_wrapper(i) for i in range(n)]
        pool = JudgingPool(judges=anon_judges, ensemble_size=n)

        # Build the judge prompt + context once. The judges see the same
        # information a human reviewer would: the original case prompt, the
        # expected output, the worker's code, its observed stdout/stderr,
        # and the sandbox-derived verdict (so they can either ratify it or
        # diverge). Truncate the code to keep prompts bounded.
        worker_output = ""
        worker_stderr = ""
        if isinstance(result.metadata, dict):
            worker_output = str(result.metadata.get("stdout", "") or "")[:2000]
            worker_stderr = str(result.metadata.get("stderr", "") or "")[:2000]
        judge_prompt = (
            f"Score the worker's solution for case {case.id}.\n\n"
            f"--- Problem ---\n{case.prompt[:2000]}\n\n"
            f"--- Worker code ---\n{(code or '')[:4000]}\n"
        )
        judge_context: dict[str, Any] = {
            "case_id": case.id,
            "language": case.language,
            "worker_output": worker_output,
            "worker_stderr": worker_stderr,
            "expected_output": (expected_output or "")[:2000],
            "sandbox_verdict": result.verdict.value,
            "sandbox_p_score": result.p_score,
        }

        # Parallel fan-out — one thread per anonymous slot. ``votes_by_slot``
        # is keyed by slot so the order is deterministic in the emitted
        # ``votes_summary`` regardless of which thread finished first.
        votes_by_slot: dict[int, JudgeVote] = {}
        lock = threading.Lock()

        def _worker(slot_idx: int, judge_callable: Any) -> None:
            vote = pool._score_single(judge_callable, judge_prompt, judge_context)
            if vote is None:
                return
            with lock:
                votes_by_slot[slot_idx] = vote

        threads = [
            threading.Thread(target=_worker, args=(i, anon_judges[i]), daemon=True)
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            # Bound the wait so a hung judge cannot stall the whole iteration.
            # 60s per case is generous — a single MiniMax call typically
            # returns in <20s; anything longer is almost certainly a hung
            # connection and we'd rather emit a partial-pool verdict.
            t.join(timeout=60.0)

        votes = [votes_by_slot[i] for i in sorted(votes_by_slot)]
        consensus_verdict, p_score_avg, p_cost_avg, p_time_avg = pool._aggregate_votes(votes)

        # Verdict distribution + dissent ratio. dissent = fraction of votes
        # that did NOT match the majority. With zero votes the ratio is 0.0
        # (degenerate — nothing to disagree about).
        verdict_distribution: dict[str, int] = {}
        for v in votes:
            verdict_distribution[v.verdict.value] = verdict_distribution.get(v.verdict.value, 0) + 1
        n_votes = len(votes)
        if n_votes > 0:
            consensus_count = verdict_distribution.get(consensus_verdict.value, 0)
            dissent_ratio = max(0.0, 1.0 - consensus_count / n_votes)
        else:
            dissent_ratio = 0.0

        votes_summary = [
            {"slot": i, "verdict": votes_by_slot[i].verdict.value,
             "p_score": votes_by_slot[i].p_score}
            for i in sorted(votes_by_slot)
        ]

        # Mutate the metadata in place so callers (BenchmarkResult assembly)
        # can surface dissent_ratio + consensus on the envelope.
        if not isinstance(result.metadata, dict):
            result.metadata = {}
        result.metadata["judge_pool"] = {
            "n_judges": n,
            "n_votes": n_votes,
            "consensus_verdict": consensus_verdict.value,
            "consensus_p_score": p_score_avg,
            "consensus_p_cost": p_cost_avg,
            "consensus_p_time": p_time_avg,
            "dissent_ratio": dissent_ratio,
            "verdict_distribution": dict(verdict_distribution),
            "votes_summary": list(votes_summary),
        }

        # Emit the pool verdict event + the disagreement escalation when
        # dissent strictly exceeds the threshold. Both emits are best-effort
        # — observability must not corrupt the harness.
        if obs is not None:
            try:
                obs.judge_pool_verdict(
                    case_id=case.id,
                    iteration=iteration,
                    n_judges=n,
                    consensus_verdict=consensus_verdict.value,
                    dissent_ratio=dissent_ratio,
                    verdict_distribution=dict(verdict_distribution),
                    consensus_p_score=p_score_avg,
                    consensus_p_cost=p_cost_avg,
                    consensus_p_time=p_time_avg,
                    votes_summary=votes_summary,
                )
            except Exception:
                pass

            if n_votes > 0 and dissent_ratio > self.dissent_threshold:
                minority_verdicts = [
                    v for v in verdict_distribution
                    if v != consensus_verdict.value
                ]
                try:
                    obs.judge_disagreement(
                        case_id=case.id,
                        iteration=iteration,
                        consensus_verdict=consensus_verdict.value,
                        dissent_ratio=dissent_ratio,
                        dissent_threshold=self.dissent_threshold,
                        verdict_distribution=dict(verdict_distribution),
                        minority_verdicts=minority_verdicts,
                    )
                except Exception:
                    pass

        return consensus_verdict, dissent_ratio, votes

    def _run_shader_case(
        self,
        case: BenchmarkCase,
        shader_src: str,
        obs: AutobenchObservability | None = None,
        iteration: int = 0,
        attempt: int = 1,
    ) -> HarnessResult:
        """Route a shader (language='glsl') case through ShaderExecutor.

        Returns a HarnessResult with:
            p_score   ← SSIM (continuous, in [0, 1] after clamping)
            verdict   ← OK / VF / WA / CE / RE / TLE (from ShaderExecutor)
            latency_ms← render_time_ms
        """
        from .shader_executor import ShaderExecutor  # local import: optional dep

        viewport_list = case.constraints.get("viewport", [512, 512])
        viewport = (int(viewport_list[0]), int(viewport_list[1]))
        target_ms = float(case.constraints.get("max_time_seconds", 5)) * 1000.0
        reference_path = case.expected_output

        # Sandbox-dispatch event for observability parity with the python
        # path. We tag sandbox_type='shader' so consumers can filter.
        if obs is not None:
            try:
                obs.sandbox_dispatch(
                    case_id=case.id,
                    language="glsl",
                    sandbox_type="shader",
                )
            except Exception:
                pass

        try:
            executor = ShaderExecutor(viewport=viewport, target_ms=target_ms)
            sresult = executor.run(
                shader_src=shader_src,
                reference_path=reference_path,
                viewport=viewport,
                target_ms=target_ms,
            )
        except NotImplementedError as e:
            return HarnessResult(
                verdict=Verdict.RE,
                error=f"shader backend unavailable: {e}",
                metadata={"case_id": case.id, "sandbox_type": "shader"},
            )

        # p_score is the SSIM (clamped to [0, 1]). For CE/RE/TLE it falls
        # back to 0.0 — caller has the verdict.
        if sresult.verdict in (Verdict.OK, Verdict.VF, Verdict.WA):
            p_score = max(0.0, min(1.0, sresult.ssim))
        else:
            p_score = 0.0

        p_time = max(0.0, min(1.0, 1.0 - (sresult.render_time_ms / target_ms))) if target_ms > 0 else 0.5

        if obs is not None:
            try:
                obs.sandbox_complete(
                    case_id=case.id,
                    language="glsl",
                    sandbox_type="shader",
                    verdict=sresult.verdict.value,
                    latency_ms=sresult.render_time_ms,
                    exit_code=0 if sresult.verdict != Verdict.CE else 1,
                )
            except Exception:
                pass

        full_len = len(shader_src or "")
        captured_code = (shader_src or "")[:GENERATED_CODE_TRUNCATE_LEN]

        result = HarnessResult(
            p_score=p_score,
            p_cost=0.5,  # cost not modelled for shader render
            p_time=p_time,
            verdict=sresult.verdict,
            error=sresult.error,
            latency_ms=sresult.render_time_ms,
            metadata={
                "case_id": case.id,
                "language": case.language,
                "sandbox_type": "shader",
                "ssim": sresult.ssim,
                "lpips": sresult.lpips,
                "frame_path": sresult.frame_path,
                "reference_path": str(reference_path),
                "compile_log_excerpt": sresult.compile_log[:512],
                "generated_code": captured_code,
                "generated_code_length": full_len,
            },
        )

        active_obs = obs if obs is not None else self.obs
        if active_obs is not None:
            try:
                active_obs.case_result(
                    case_id=case.id,
                    iteration=iteration,
                    language=case.language,
                    verdict=sresult.verdict.value,
                    p_score=p_score,
                    latency_ms=sresult.render_time_ms,
                    generated_code=captured_code,
                    generated_code_length=full_len,
                    attempt=attempt,
                )
            except Exception:
                pass

            # Bead nervous-bus-bns: surface error-class stderr for shader
            # cases too. ShaderExecutor stuffs the compile log / runtime
            # error into sresult.error; that's what downstream tooling
            # needs to see for CE/RE/TLE/MLE.
            if sresult.verdict.value in {"CE", "RE", "TLE", "MLE"}:
                try:
                    active_obs.sandbox_stderr(
                        case_id=case.id,
                        iteration=iteration,
                        verdict=sresult.verdict.value,
                        stderr_excerpt=sresult.error or "",
                        exit_code=None,
                        language=case.language,
                    )
                except Exception:
                    pass

        return result

    def emit_verdict(
        self,
        code_output: str,
        stderr: str,
        runtime_ms: float,
        memory_mb: float,
        exit_code: int = 0,
        expected_output: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> Verdict:
        """Emit a verdict from execution results.

        Precedence order (first match wins):
        CE > TLE > MLE > RE > WA > OK

        Args:
            code_output: stdout from the executed code.
            stderr: stderr from execution.
            runtime_ms: Actual runtime in milliseconds.
            memory_mb: Memory limit in MB.
            exit_code: Process exit code.
            expected_output: Expected stdout for WA comparison.
            constraints: Execution constraints.

        Returns:
            Verdict enum value.
        """
        constraints = constraints or {}

        # Compilation Error: non-zero exit and non-empty stderr with error indicators
        if exit_code != 0:
            ce_indicators = [
                "error:",
                "Error:",
                "ERROR",
                "cannot find symbol",
                "undefined reference",
                "syntax error",
                "SyntaxError",
                "ParseError",
                "javac:",
                "g++:",
                "gcc:",
            ]
            for indicator in ce_indicators:
                if indicator in stderr:
                    return Verdict.CE

        # Timeout
        max_time = constraints.get("max_time_seconds", 10)
        if runtime_ms / 1000.0 > max_time:
            return Verdict.TLE

        # Memory limit exceeded (heuristic)
        max_memory = constraints.get("max_memory_mb", 512)
        if self._looks_like_oom(stderr):
            return Verdict.MLE

        # Runtime error indicators in stderr
        re_indicators = [
            "Traceback",
            "panic:",
            "Exception in thread",
            "java.lang.RuntimeException",
            "runtime error:",
            "segmentation fault",
            "SIGSEGV",
            "SIGABRT",
            "index out of range",
            "key error",
            "null pointer",
            "undefined is not a function",
            "referenceerror",
            "typeerror",
            "attributeerror",
        ]
        for indicator in re_indicators:
            if indicator in stderr:
                return Verdict.RE

        # Wrong answer: exit code 0 but output mismatch
        if expected_output:
            actual = code_output.strip()
            expected = expected_output.strip()
            if actual != expected:
                # Check for partial match (e.g., output contains expected)
                if expected not in actual and actual not in expected:
                    return Verdict.WA

        # Default: OK
        return Verdict.OK

    def _looks_like_oom(self, stderr: str) -> bool:
        """Heuristic for out-of-memory conditions."""
        oom_indicators = [
            "out of memory",
            "OutOfMemoryError",
            "std::bad_alloc",
            "cannot allocate",
            "fatal: unable to allocate",
            "killed",
            "oom_kill",
            "memory limit exceeded",
            "MemoryError",
            "Java heap space",
        ]
        for indicator in oom_indicators:
            if indicator in stderr:
                return True
        return False

    @staticmethod
    def score_harness(
        harness_results: list[HarnessResult],
        utility_weights: dict[str, float] | None = None,
    ) -> float:
        """Score a set of harness results using the SICA formula.

        U = w_score * avg_p_score + w_cost * avg_p_cost + w_time * avg_p_time

        All p_* values are normalized [0.0, 1.0], so U is also [0.0, 1.0].

        Args:
            harness_results: List of HarnessResult objects.
            utility_weights: Dict with keys score, cost, time and float weights.
                              Weights should sum to 1.0 (normalized internally).

        Returns:
            Utility score in [0.0, 1.0].
        """
        if not harness_results:
            return 0.0

        utility_weights = utility_weights or DEFAULT_WEIGHTS.copy()

        # Normalize weights to sum to 1.0
        total_weight = sum(utility_weights.values())
        if total_weight == 0:
            total_weight = 1.0
        w_score = utility_weights.get("score", 0.5) / total_weight
        w_cost = utility_weights.get("cost", 0.25) / total_weight
        w_time = utility_weights.get("time", 0.25) / total_weight

        avg_score = sum(r.p_score for r in harness_results) / len(harness_results)
        avg_cost = sum(r.p_cost for r in harness_results) / len(harness_results)
        avg_time = sum(r.p_time for r in harness_results) / len(harness_results)

        # Higher cost/time = lower utility, so invert avg_cost and avg_time
        # (cost/time are normalized such that 1.0 = best = cheapest/fastest)
        utility = w_score * avg_score + w_cost * (1.0 - avg_cost) + w_time * (1.0 - avg_time)

        return max(0.0, min(1.0, utility))


def emit_verdict(
    code_output: str,
    stderr: str,
    runtime_ms: float,
    memory_mb: float,
    **kwargs,
) -> Verdict:
    """Standalone emit_verdict — delegates to BenchmarkEvaluator.emit_verdict."""
    evaluator = BenchmarkEvaluator()
    return evaluator.emit_verdict(
        code_output=code_output,
        stderr=stderr,
        runtime_ms=runtime_ms,
        memory_mb=memory_mb,
        **kwargs,
    )


def score_harness(
    harness_results: list[HarnessResult],
    utility_weights: dict[str, float] | None = None,
) -> float:
    """Standalone score_harness — delegates to BenchmarkEvaluator.score_harness."""
    return BenchmarkEvaluator.score_harness(harness_results, utility_weights)


# ---------------------------------------------------------------------------
# Collective LLM-as-judge (gap #10)
# ---------------------------------------------------------------------------


@dataclass
class JudgeVote:
    """A single judge's assessment of a worker output."""

    judge_id: str
    verdict: Verdict
    p_score: float
    p_cost: float
    p_time: float
    reasoning: str = ""


@dataclass
class JudgingPool:
    """Collective LLM-as-judge with anonymous aggregation.

    Assigns each worker output to N independent judges. Each judge scores
    in isolation — no cross-judge visibility (anonymous protocol). Aggregated
    verdict feeds into BenchmarkEvaluator.score_harness().

    Use JudgingPool when the judge variance dominates the measurement noise.
    Calibrate ensemble size by running noise_floor.py with 3, 5, 7 judges and
    finding where variance plateaus.

    Args:
        judges: List of judge client factories. Each factory is a callable
            that takes (prompt: str, context: dict) and returns a dict with
            keys: verdict (str), p_score (float), p_cost (float),
            p_time (float), reasoning (str).
        ensemble_size: How many judges to use. Defaults to len(judges).
        aggregation: "majority" (verdict) + mean (continuous) or
            "sica_weighted" (weighted by inverse variance).
    """

    judges: list[Any] = field(default_factory=list)
    ensemble_size: int | None = None
    aggregation: str = "majority"

    def __post_init__(self) -> None:
        if not self.judges:
            raise ValueError("JudgingPool requires at least one judge")
        self._size = self.ensemble_size or len(self.judges)

    def _score_single(
        self,
        judge_factory: Any,
        prompt: str,
        context: dict[str, Any],
    ) -> JudgeVote | None:
        """Ask one judge to score a single worker output."""
        try:
            result = judge_factory(prompt=prompt, context=context)
            if not isinstance(result, dict):
                return None
            verdict_str = result.get("verdict", "OK")
            try:
                verdict = Verdict(verdict_str)
            except ValueError:
                verdict = Verdict.OK
            return JudgeVote(
                judge_id=result.get("judge_id", "unknown"),
                verdict=verdict,
                p_score=float(result.get("p_score", 0.5)),
                p_cost=float(result.get("p_cost", 0.5)),
                p_time=float(result.get("p_time", 0.5)),
                reasoning=result.get("reasoning", ""),
            )
        except Exception:
            return None

    def _aggregate_votes(self, votes: list[JudgeVote]) -> tuple[Verdict, float, float, float]:
        """Aggregate a list of JudgeVote objects into a single verdict + scores."""
        if not votes:
            return Verdict.OK, 0.0, 0.5, 0.5

        # Majority vote for verdict
        verdict_counts: dict[Verdict, int] = {}
        for v in votes:
            verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1
        consensus_verdict = max(verdict_counts, key=verdict_counts.get)

        # Mean for continuous scores
        avg_p_score = sum(v.p_score for v in votes) / len(votes)
        avg_p_cost = sum(v.p_cost for v in votes) / len(votes)
        avg_p_time = sum(v.p_time for v in votes) / len(votes)

        return consensus_verdict, avg_p_score, avg_p_cost, avg_p_time

    def evaluate(
        self,
        prompt: str,
        context: dict[str, Any],
    ) -> tuple[Verdict, float, float, float, list[JudgeVote]]:
        """Evaluate a worker output using the full judging pool.

        Args:
            prompt: The evaluation prompt (e.g. "Score this code: ...")
            context: Arbitrary context passed to each judge factory
                    (e.g. {"worker_output": "...", "expected_output": "..."})

        Returns:
            Tuple of (verdict, p_score, p_cost, p_time, all_votes).
            The aggregated scores feed into score_harness().
        """
        active_judges = self.judges[: self._size]
        votes: list[JudgeVote] = []

        for judge_factory in active_judges:
            vote = self._score_single(judge_factory, prompt, context)
            if vote is not None:
                votes.append(vote)

        if not votes:
            return Verdict.OK, 0.0, 0.5, 0.5, []

        verdict, p_score, p_cost, p_time = self._aggregate_votes(votes)
        return verdict, p_score, p_cost, p_time, votes

    def calibrate_ensemble_size(
        self,
        calibration_cases: list[dict[str, Any]],
        judge_factory: Any,
    ) -> dict[int, float]:
        """Measure variance as a function of ensemble size.

        Args:
            calibration_cases: List of dicts with "prompt" and "context" keys.
            judge_factory: A single judge factory to sample from.

        Returns:
            Dict mapping ensemble_size -> variance (lower is better).
            Run with 3, 5, 7 judges over the same calibration cases to find
            where variance plateaus.
        """
        results: dict[int, list[float]] = {}
        for size in [1, 3, 5, 7]:
            scores: list[float] = []
            for case in calibration_cases:
                local_votes: list[JudgeVote] = []
                for _ in range(size):
                    vote = self._score_single(judge_factory, case["prompt"], case["context"])
                    if vote is not None:
                        local_votes.append(vote)
                if local_votes:
                    _, p_score, _, _ = self._aggregate_votes(local_votes)
                    scores.append(p_score)

            if scores:
                import statistics
                results[size] = statistics.variance(scores) if len(scores) > 1 else 0.0

        return results
