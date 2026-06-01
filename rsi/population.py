"""Multi-advocate RSI spine for autobench.

The :class:`PopulationRunner` orchestrates N parallel HarnessConfig
lineages ("advocates") through the RSI loop. Each advocate has its own
``session_id``, :class:`SelfImprovingHarness` instance, and
:class:`BudgetGuard`. End-of-cycle emits
``autobench.population.summary.v1`` and picks the raw-best + diversity-
adjusted winners.

``n_advocates == 1`` is indistinguishable from the legacy single-lineage
path (no summary event, single ``SelfImprovingHarness.improve()`` call).

Public surface: :class:`PopulationRunner`, :class:`PopulationResult`,
:class:`AdvocateResult`, :func:`select_promotion_candidate`.

Phase roadmap (6yut → bo86 → future shrinkage), sibling cross-context
wiring, and the diversity-adjustment formula are preserved in
``_checkpoints/architecture-history.md``.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..evaluation.registry import DEFAULT_DOMAIN, BenchmarkRegistry
from ..core import HarnessConfig
from ..evaluation.diversity import (
    LineageSignature,
    lineage_signature,
    pairwise_lineage_distance,
)
from ..evaluator import BenchmarkCase, BenchmarkEvaluator, BenchmarkResult
from ..observability import (
    CHANNEL_POPULATION_SUMMARY,
    AutobenchObservability,
    _iso_now,
    _ulid,
)
from ..rsi.loop import ImprovementDelta, SelfImprovingHarness


def _read_diversity_weight_env(default: float = 0.10) -> float:
    """Read ``AUTOBENCH_DIVERSITY_WEIGHT`` from env. Falls back to ``default``.

    Negative values are coerced to 0.0 (no penalty). Non-finite or unparseable
    values fall back to ``default``. A weight of 0.0 disables the cross-advocate
    diversity bonus entirely and reproduces Phase-1 winner-selection behavior
    bit-for-bit.
    """
    raw = os.environ.get("AUTOBENCH_DIVERSITY_WEIGHT")
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
        if not (v == v) or v in (float("inf"), float("-inf")):  # noqa: PLR0124
            return default
        return max(0.0, v)
    except ValueError:
        return default


def _read_n_advocates_env(default: int = 3) -> int:
    """Read ``AUTOBENCH_ADVOCATES`` from env. Falls back to ``default``.

    Values <= 0 are coerced to 1.
    """
    raw = os.environ.get("AUTOBENCH_ADVOCATES")
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
        if v < 1:
            return 1
        return v
    except ValueError:
        return default


@dataclass
class AdvocateResult:
    """Per-advocate output of one population run."""

    advocate_id: str           # e.g. "advocate-0", "advocate-1", "advocate-2"
    session_id: str
    final_harness: HarnessConfig
    final_result: BenchmarkResult | None
    history: list[tuple[HarnessConfig, BenchmarkResult, ImprovementDelta]]
    best_score: float
    best_iter: int
    error: str | None = None
    # nervous-bus-bo86 (Phase 2 of wire-pop): cross-advocate diversity fields.
    # diversity_score ∈ [0, 1] — mean pairwise lineage distance to siblings.
    # adjusted_score = best_score * (1 + diversity_weight * (diversity_score - 0.5))
    diversity_score: float = 0.0
    adjusted_score: float = 0.0
    # nervous-bus-msqa (Phase 5): AHE outcome label of the *latest* verified
    # prediction in this lineage. One of:
    #   "confirmed"     — last verified prediction's outcome_label == confirmed
    #   "partial"       — last verified prediction's outcome_label == partial
    #   "refuted"       — last verified prediction's outcome_label == refuted
    #   "refuted_live"  — the most recent iteration killed its prediction live
    #                     (mathematically unachievable mid-evaluation)
    #   "none"          — lineage made no verifiable prediction or errored
    # Cross-run promotion (continuous.py) NEVER selects refuted/refuted_live.
    ahe_outcome: str = "none"
    # nervous-bus-qp91 (Phase 6): per-domain scores from the cross-domain
    # registry. Always populated — for legacy single-domain runs this is
    # ``{DEFAULT_DOMAIN: best_score}``; aggregate_score == best_score in
    # that path. Promotion (Phase 5 selector) ranks by aggregate_score, NOT
    # best_score — cross-domain evidence is the new figure of merit.
    per_domain_scores: dict[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0


@dataclass
class PopulationResult:
    """Aggregate output of one population cycle."""

    advocates: list[AdvocateResult]
    winner_id: str
    winner_score: float
    cycle_started_at: str
    cycle_ended_at: str
    # nervous-bus-bo86: identity of the highest *adjusted_score* advocate.
    # Equal to winner_id when diversity_weight == 0 OR when raw-best and
    # diversity-adjusted-best happen to coincide.
    adjusted_winner_id: str = ""
    diversity_weight: float = 0.0
    cycle_id: str = field(default_factory=_ulid)


class PopulationRunner:
    """Run N parallel HarnessConfig lineages through the RSI loop sequentially.

    Args:
        n_advocates: Number of advocates to run. Must be >= 1.
        initial_harness_factory: Zero-arg callable returning a fresh
            :class:`HarnessConfig` for an advocate. Called once per advocate
            so factory-stateful mutations (rare) don't bleed across lineages.
        evaluator_factory: Zero-arg callable returning a fresh
            :class:`BenchmarkEvaluator`. Called once per advocate — the
            factory is expected to wire in a per-advocate
            :class:`AutobenchObservability` so events land on a distinct
            session_id.
        observability_factory: Optional zero-arg callable returning a fresh
            :class:`AutobenchObservability`. When None the runner constructs
            a default one per advocate. The same observability is passed to
            the :class:`SelfImprovingHarness` and is the source of the
            advocate's session_id.
        improver: Improver name passed through to
            :attr:`SelfImprovingHarness.default_improver`.
        max_iterations_per_advocate: RSI iteration cap per lineage.
        budget_per_advocate_seconds: Wall-clock cap for each advocate's
            :class:`BudgetGuard`. ``None`` disables the guard (each advocate
            runs without one — the underlying RSI loop already caps by
            iteration count). Budget choice for the cycle: budget is **per
            advocate** (not split across) — this matches the "isolation"
            principle and means total cycle budget scales with N. Operators
            who want a fixed total budget should pass
            ``budget_per_advocate_seconds = total_budget / n_advocates``.
        run_summary_obs: Optional :class:`AutobenchObservability` used to emit
            the ``autobench.population.summary.v1`` event. When None the
            runner makes a fresh one with a stable session_id of
            ``population-<cycle_id>``. The summary's ``session_id`` field on
            the bus envelope is therefore distinct from any one advocate's
            session_id (so consumers can correlate the summary back to the
            cycle, not a single lineage).
    """

    def __init__(
        self,
        n_advocates: int,
        initial_harness_factory: Callable[[], HarnessConfig],
        evaluator_factory: Callable[[], BenchmarkEvaluator],
        observability_factory: Callable[[], AutobenchObservability] | None = None,
        improver: str = "minimax",
        max_iterations_per_advocate: int = 5,
        budget_per_advocate_seconds: float | None = 600.0,
        improvement_threshold: float | None = None,
        budget_guard_factory: Callable[[AutobenchObservability], Any] | None = None,
        run_summary_obs: AutobenchObservability | None = None,
        diversity_penalty_weight: float | None = None,
        adversarial_ratio: float = 0.0,
        prior_result: "PopulationResult | None" = None,
        adversarial_obs: AutobenchObservability | None = None,
        registry: "BenchmarkRegistry | None" = None,
    ) -> None:
        if n_advocates < 1:
            raise ValueError(f"n_advocates must be >= 1, got {n_advocates}")
        self.n_advocates = n_advocates
        self.initial_harness_factory = initial_harness_factory
        self.evaluator_factory = evaluator_factory
        self.observability_factory = observability_factory
        self.improver = improver
        self.max_iterations_per_advocate = max_iterations_per_advocate
        self.budget_per_advocate_seconds = budget_per_advocate_seconds
        self.improvement_threshold = improvement_threshold
        self.budget_guard_factory = budget_guard_factory
        self.run_summary_obs = run_summary_obs
        # nervous-bus-bo86: explicit constructor arg wins; otherwise read env.
        # 0.0 disables the cross-advocate diversity bonus and preserves
        # Phase-1 winner-selection behavior bit-for-bit.
        if diversity_penalty_weight is None:
            self.diversity_penalty_weight = _read_diversity_weight_env(default=0.10)
        else:
            self.diversity_penalty_weight = max(0.0, float(diversity_penalty_weight))
        # Shared mutable list — fed into each advocate's SelfImprovingHarness
        # so the SIBLINGS prompt block can reference completed siblings.
        # Cleared at the start of every ``run()`` call.
        self._cross_advocate_context: list[ImprovementDelta] = []
        # nervous-bus-gdzo (wire-pop Phase 3): adversarial gotcha mix.
        # When > 0, this fraction of cases is replaced with adversarially-
        # generated curveballs each cycle, keyed to ``prior_result``'s
        # failure modes. 0.0 (default) disables the mix and preserves the
        # bit-for-bit Phase 1/2/4 path.
        self.adversarial_ratio = max(0.0, min(1.0, float(adversarial_ratio)))
        self.prior_result = prior_result
        self.adversarial_obs = adversarial_obs
        # nervous-bus-qp91 (Phase 6): cross-domain registry. None defers
        # construction to :meth:`run` and only when callers pass the new
        # dict shape — preserves bit-for-bit single-domain behavior.
        self.registry = registry

    # ------------------------------------------------------------------ #
    # Per-advocate helpers
    # ------------------------------------------------------------------ #

    def _advocate_id_for(self, index: int) -> str:
        return f"advocate-{index}"

    def _make_observability(self) -> AutobenchObservability:
        if self.observability_factory is not None:
            return self.observability_factory()
        return AutobenchObservability()

    def _make_budget_guard(self, obs: AutobenchObservability) -> Any:
        if self.budget_guard_factory is not None:
            return self.budget_guard_factory(obs)
        if self.budget_per_advocate_seconds is None:
            return None
        # Lazy import — budget_guard imports observability for its own emitter,
        # and we'd rather not pull it at module import time.
        from ..budget_guard import BudgetGuard
        return BudgetGuard(
            max_cost_dollars=0,
            max_wall_time_seconds=float(self.budget_per_advocate_seconds),
            session_id=obs.session_id,
        )

    def _run_one_advocate(
        self,
        index: int,
        benchmark_cases: list[BenchmarkCase],
        secondary_domains: dict[str, list[BenchmarkCase]] | None = None,
        primary_domain: str = DEFAULT_DOMAIN,
        registry: "BenchmarkRegistry | None" = None,
        cycle_id: str = "",
    ) -> AdvocateResult:
        """Run RSI for one advocate against the primary domain.

        After RSI completes, evaluate the final harness against every
        ``secondary_domains`` case set using the SAME evaluator to produce
        per-domain scores. The aggregate is computed via
        :meth:`BenchmarkRegistry.aggregate_score`.

        Args:
            index: Advocate ordinal (0-based).
            benchmark_cases: Primary-domain cases — RSI runs against these.
            secondary_domains: Optional ``{name: cases}`` for additional
                domains. Evaluated POST-RSI against the final harness.
            primary_domain: Name of the primary domain (used as the key in
                ``per_domain_scores``). Defaults to ``DEFAULT_DOMAIN`` for
                the legacy single-domain path.
            registry: When provided, used to compute ``aggregate_score``.
                When None, aggregate == primary score (legacy behavior).
        """
        advocate_id = self._advocate_id_for(index)
        obs = self._make_observability()
        harness = self.initial_harness_factory()
        evaluator = self.evaluator_factory()
        guard = self._make_budget_guard(obs)

        kwargs: dict[str, Any] = {
            "current_harness": harness,
            "evaluator": evaluator,
            "max_iterations": self.max_iterations_per_advocate,
            "default_improver": self.improver,
            "obs": obs,
            "budget_guard": guard,
            # nervous-bus-bo86: pass the SAME list reference so each advocate
            # sees the in-progress sibling state at improver-call time.
            "cross_advocate_context": self._cross_advocate_context,
        }
        if self.improvement_threshold is not None:
            kwargs["improvement_threshold"] = self.improvement_threshold

        sih = SelfImprovingHarness(**kwargs)

        final_harness: HarnessConfig = harness
        final_result: BenchmarkResult | None = None
        history: list[tuple[HarnessConfig, BenchmarkResult, ImprovementDelta]] = []
        error: str | None = None

        try:
            final_harness, final_result, history = sih.improve(benchmark_cases)
        except Exception as exc:  # noqa: BLE001 — one advocate's failure must not kill the cycle
            error = f"{type(exc).__name__}: {exc}"
            history = list(getattr(sih, "_iteration_history", []) or [])

        best_score: float = float("-inf")
        best_iter: int = -1
        for i, (_h, r, _d) in enumerate(history):
            score = float(getattr(r, "aggregate_score", 0.0))
            if score > best_score:
                best_score = score
                best_iter = i
        if best_score == float("-inf"):
            best_score = 0.0

        # nervous-bus-msqa: derive AHE outcome from the SelfImprovingHarness's
        # recorded verifications + live-refute trail. "refuted_live" trumps any
        # verified outcome at the same iteration: if the latest iteration
        # killed its prior prediction live, that's the operative signal even
        # if a later verify_prediction call recorded a different label.
        ahe_outcome = _derive_ahe_outcome(sih)

        # nervous-bus-qp91: cross-domain post-RSI evaluation. The primary
        # domain contributes ``best_score`` directly (the RSI loop already
        # maximized it). Each secondary domain runs the FINAL harness through
        # the same evaluator once and contributes its aggregate_score.
        per_domain: dict[str, float] = {primary_domain: float(best_score)}
        if secondary_domains:
            for dname, dcases in secondary_domains.items():
                if dname == primary_domain:
                    continue
                if not dcases:
                    # Empty cases contribute 0.0 (domain shipped but
                    # measured nothing this cycle).
                    per_domain[dname] = 0.0
                    continue
                try:
                    dres = evaluator.run(final_harness, dcases, obs=obs)
                    per_domain[dname] = float(getattr(dres, "aggregate_score", 0.0))
                except Exception as exc:  # noqa: BLE001
                    # A single domain's evaluation failure must not poison
                    # the whole advocate — log via error field and contribute 0.
                    error_msg = f"domain {dname} eval failed: {type(exc).__name__}: {exc}"
                    error = (error or "") + ("; " if error else "") + error_msg
                    per_domain[dname] = 0.0

        if registry is not None:
            aggregate = registry.aggregate_score(per_domain)
        else:
            # Legacy single-domain path — aggregate is the primary score.
            aggregate = float(best_score)

        # nervous-bus-qp91: emit cross-domain event when more than one domain
        # contributed. Uses the SAME obs instance that emitted iteration
        # events for this advocate so consumers can correlate via session_id.
        if registry is not None and len(per_domain) > 1:
            try:
                weight_map = registry.weights()
                used = {
                    n: max(0.0, float(weight_map.get(n, 0.0)))
                    for n in per_domain.keys()
                }
                total = sum(used.values())
                if total > 0.0:
                    used = {n: w / total for n, w in used.items()}
                obs.cross_domain_evaluation_complete(
                    advocate_id=advocate_id,
                    per_domain_scores=per_domain,
                    aggregate_score=aggregate,
                    weights=used,
                    primary_domain=primary_domain,
                    cycle_id=cycle_id,
                )
            except Exception:  # noqa: BLE001 — never break the cycle on emit
                pass

        return AdvocateResult(
            advocate_id=advocate_id,
            session_id=obs.session_id,
            final_harness=final_harness,
            final_result=final_result,
            history=history,
            best_score=best_score,
            best_iter=best_iter,
            error=error,
            ahe_outcome=ahe_outcome,
            per_domain_scores=per_domain,
            aggregate_score=aggregate,
        )

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #

    def run(
        self,
        benchmark_cases: "list[BenchmarkCase] | dict[str, list[BenchmarkCase]]",
    ) -> PopulationResult:
        """Run all advocates sequentially and pick a winner.

        Accepts either of two shapes (nervous-bus-qp91):

            * ``list[BenchmarkCase]`` — legacy single-domain shape. Wrapped
              as ``{DEFAULT_DOMAIN: cases}`` internally; aggregate_score
              equals best_score. Promotion semantics unchanged in this path.
            * ``dict[str, list[BenchmarkCase]]`` — multi-domain shape. The
              first key (by registry order, or insertion order for ad-hoc
              dicts) is the *primary domain* — RSI runs against it. Every
              other domain is evaluated against the FINAL harness only.
              ``aggregate_score`` is the weighted mean across all domains
              via :meth:`BenchmarkRegistry.aggregate_score`.

        Returns a :class:`PopulationResult`. Emits exactly one
        ``autobench.population.summary.v1`` event when ``n_advocates > 1``;
        when ``n_advocates == 1`` no summary event is emitted (the single
        advocate's own iteration events are the canonical record, matching
        the pre-population code path bit-for-bit).
        """
        cycle_started_at = _iso_now()
        cycle_id = _ulid()  # generated early so cross-domain events can correlate

        # Reset cross-advocate context for this cycle. Subsequent advocates
        # will see the most-recent ImprovementDelta from each predecessor.
        self._cross_advocate_context.clear()

        # ------------------------------------------------------------------ #
        # Cross-domain normalization (nervous-bus-qp91).
        # ------------------------------------------------------------------ #
        # Determine the primary domain (the case set RSI actually runs
        # against) and the secondary-domain map (post-RSI evaluations).
        if isinstance(benchmark_cases, dict):
            cases_by_domain: dict[str, list[BenchmarkCase]] = dict(benchmark_cases)
            if cases_by_domain:
                # Primary = first key (registry already orders by weight;
                # ad-hoc dicts use Python 3.7+ insertion order).
                primary_domain = next(iter(cases_by_domain))
                primary_cases = cases_by_domain[primary_domain]
            else:
                primary_domain = DEFAULT_DOMAIN
                primary_cases = []
            # The registry used for aggregation: prefer the one wired in
            # at construction time; otherwise build a default — this means
            # callers that pass a dict but no registry still get cross-
            # domain aggregation using DEFAULT_WEIGHTS.
            registry = self.registry or BenchmarkRegistry.default()
        else:
            # Legacy single-domain path — preserved bit-for-bit.
            primary_domain = DEFAULT_DOMAIN
            primary_cases = list(benchmark_cases or [])
            cases_by_domain = {primary_domain: primary_cases}
            registry = None  # signal legacy-aggregate semantics

        # nervous-bus-gdzo (wire-pop Phase 3): mix adversarial gotchas into
        # the assembled case set. No-op when adversarial_ratio == 0.0, which
        # preserves the Phase 1/2/4 path bit-for-bit. Adversarial mixing
        # operates on the PRIMARY domain only — secondary domains stay as
        # the registry shipped them so cross-domain comparisons remain
        # apples-to-apples across cycles.
        if self.adversarial_ratio > 0.0 and primary_cases:
            from ..evaluation.assembly import assemble_benchmark_cases
            primary_cases = assemble_benchmark_cases(
                base_cases=primary_cases,
                prior_result=self.prior_result,
                adversarial_ratio=self.adversarial_ratio,
                obs=self.adversarial_obs,
            )
            cases_by_domain[primary_domain] = primary_cases

        # Build the secondary-only map (everything except the primary).
        secondary_domains = {
            n: c for n, c in cases_by_domain.items() if n != primary_domain
        }

        advocates: list[AdvocateResult] = []
        for i in range(self.n_advocates):
            ar = self._run_one_advocate(
                i,
                primary_cases,
                secondary_domains=secondary_domains,
                primary_domain=primary_domain,
                registry=registry,
                cycle_id=cycle_id,
            )
            advocates.append(ar)
            # After this advocate finishes, append its FINAL ImprovementDelta
            # to the shared context so the next advocate's improver gets a
            # SIBLINGS block referencing this one. Advocates with no
            # iteration history contribute nothing.
            if ar.history:
                last_delta = ar.history[-1][2]
                self._cross_advocate_context.append(last_delta)

        cycle_ended_at = _iso_now()

        # ------------------------------------------------------------------ #
        # Cross-advocate diversity post-processing (nervous-bus-bo86).
        # ------------------------------------------------------------------ #
        # Build a LineageSignature per advocate, then compute each one's mean
        # pairwise distance to the others. Adjusted score:
        #     best_score * (1 + weight * (mean_dist - 0.5))
        # The 0.5 anchor means a "neutrally diverse" lineage gets a no-op
        # multiplier of 1.0, preserving Phase-1 winner-selection when
        # diversity is uniform across the population.
        weight = self.diversity_penalty_weight
        signatures: list[LineageSignature] = [
            lineage_signature([d for (_h, _r, d) in a.history])
            for a in advocates
        ]

        if len(advocates) > 1 and weight > 0.0:
            for i, a in enumerate(advocates):
                others = [signatures[j] for j in range(len(advocates)) if j != i]
                if others:
                    dists = [
                        pairwise_lineage_distance(signatures[i], s) for s in others
                    ]
                    mean_dist = sum(dists) / len(dists)
                else:
                    mean_dist = 0.0
                a.diversity_score = float(mean_dist)
                a.adjusted_score = float(
                    a.best_score * (1.0 + weight * (mean_dist - 0.5))
                )
        else:
            # Single-advocate cycle or weight disabled — adjusted_score is
            # the raw best_score and diversity_score is 0.0.
            for a in advocates:
                a.diversity_score = 0.0
                a.adjusted_score = float(a.best_score)

        # Raw-best winner (Phase-1 semantics, preserved for replay parity).
        winner = max(
            advocates,
            key=lambda a: (a.best_score, -int(a.advocate_id.rsplit("-", 1)[-1])),
        )
        # Diversity-adjusted winner (the operational winner — what
        # downstream cross-cycle promotion will key on, Phase 5).
        adj_winner = max(
            advocates,
            key=lambda a: (a.adjusted_score, -int(a.advocate_id.rsplit("-", 1)[-1])),
        )

        result = PopulationResult(
            advocates=advocates,
            winner_id=winner.advocate_id,
            winner_score=winner.best_score,
            cycle_started_at=cycle_started_at,
            cycle_ended_at=cycle_ended_at,
            adjusted_winner_id=adj_winner.advocate_id,
            diversity_weight=weight,
            cycle_id=cycle_id,
        )

        # Emit summary event when running with >1 advocate. Single-advocate
        # runs intentionally skip it: their behavior must match the legacy
        # single-lineage path exactly (no surprise bus event).
        if self.n_advocates > 1:
            try:
                summary_obs = self.run_summary_obs or AutobenchObservability(
                    session_id=f"pop-{result.cycle_id}"[:26],
                )
                summary_obs.population_summary(
                    cycle_id=result.cycle_id,
                    advocates_summary=[
                        {
                            "advocate_id": a.advocate_id,
                            "session_id": a.session_id,
                            "final_score": a.best_score,
                            "best_iter": a.best_iter,
                            # nervous-bus-bo86: per-advocate diversity fields.
                            # Schema is permissive (no additionalProperties:false)
                            # so this is a backward-compatible additive bump.
                            "diversity_score": a.diversity_score,
                            "adjusted_score": a.adjusted_score,
                        }
                        for a in advocates
                    ],
                    winner_id=result.winner_id,
                    winner_score=result.winner_score,
                    cycle_started_at=cycle_started_at,
                    cycle_ended_at=cycle_ended_at,
                    adjusted_winner_id=result.adjusted_winner_id,
                    diversity_weight=result.diversity_weight,
                )
            except Exception:  # noqa: BLE001 — observability never breaks the cycle
                pass

        return result


# --------------------------------------------------------------------------- #
# Cross-run promotion (nervous-bus-msqa / wire-pop Phase 5)
# --------------------------------------------------------------------------- #


_PROMOTABLE_AHE_OUTCOMES = {"confirmed", "partial"}
_REJECTED_AHE_OUTCOMES = {"refuted", "refuted_live"}


def _derive_ahe_outcome(sih: SelfImprovingHarness) -> str:
    """Read the AHE tail off a finished SelfImprovingHarness.

    Priority (most-recent wins):
        1. ``refuted_live`` if the FINAL iteration's prediction was killed
           mid-evaluation. This is the strongest negative signal — the lineage
           made a prediction that was mathematically unachievable by the time
           the next iteration finished its case loop.
        2. The ``outcome_label`` of the most recently verified prediction.
        3. ``"none"`` when the lineage produced no verifiable predictions.
    """
    verifs = list(getattr(sih, "_verifications", []) or [])
    live_refuted = list(getattr(sih, "_live_refuted_iterations", []) or [])

    last_verify_iter = -1
    last_verify_label = None
    if verifs:
        last_verify_label = str(getattr(verifs[-1], "outcome_label", "") or "")
        # PredictionVerification doesn't carry an iter index, but its position
        # in the list matches verification order; we treat it as "last".
        last_verify_iter = len(verifs) - 1

    last_live_refute_iter = max(live_refuted) if live_refuted else -1

    # If a live-refute happened AT OR AFTER the last verify, it's the operative
    # tail of the lineage.
    if last_live_refute_iter >= 0 and last_live_refute_iter >= last_verify_iter:
        return "refuted_live"
    if last_verify_label:
        return last_verify_label
    return "none"


def select_promotion_candidate(
    advocates: list[AdvocateResult],
) -> AdvocateResult | None:
    """Return the top confirmed-AHE advocate, or ``None``.

    Selection rules:

      * ``ahe_outcome`` MUST be in ``{"confirmed", "partial"}``.
        ``refuted`` / ``refuted_live`` / ``none`` advocates are NEVER selected.
      * Ranking key (nervous-bus-qp91):
        ``(ahe_outcome_rank, aggregate_score, adjusted_score, best_score, -index)``.
        ``confirmed`` outranks ``partial``; ties break on the **cross-domain
        aggregate_score** first (the new figure of merit), then on
        ``adjusted_score`` (diversity-aware best), then ``best_score``,
        then lowest advocate index (stable per Phase-1 contract).

        For legacy single-domain cycles ``aggregate_score == best_score``,
        so the cross-domain key is a no-op and Phase-5 ranking is preserved.
      * Returns ``None`` when no advocate has a promotable outcome. This is
        the "stage NOTHING" path — a feature, not a bug, preventing
        path-dependent drift on cycles where every prediction was wrong.
    """
    candidates = [a for a in advocates if a.ahe_outcome in _PROMOTABLE_AHE_OUTCOMES]
    if not candidates:
        return None

    def _rank(a: AdvocateResult) -> tuple[int, float, float, float, int]:
        outcome_rank = 2 if a.ahe_outcome == "confirmed" else 1
        try:
            idx = int(a.advocate_id.rsplit("-", 1)[-1])
        except ValueError:
            idx = 0
        return (
            outcome_rank,
            float(a.aggregate_score),
            float(a.adjusted_score),
            float(a.best_score),
            -idx,
        )

    return max(candidates, key=_rank)


__all__ = [
    "AdvocateResult",
    "PopulationResult",
    "PopulationRunner",
    "_read_n_advocates_env",
    "select_promotion_candidate",
]
