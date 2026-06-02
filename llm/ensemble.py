"""Multi-improver ensemble for the autobench RSI loop (wire-pop Phase 6).

Wraps N anonymous :class:`MiniMaxLLMWrapper` instances and aggregates their
proposed :class:`ImprovementDelta` objects into one chosen delta + harness
per iteration. Two strategies: ``"vote"`` (default, majority per field) and
``"parallel"`` (opt-in, requires an ``arm_evaluator`` callable).

Observability — when ``obs`` is supplied, emits an
``autobench.improver.ensemble.v1`` event per call. ``n_instances == 1``
short-circuits to a direct call with no ensemble event (zero behavioural
change vs. the single-improver path).

Strategy details and the nervous-bus-9xd derivation are preserved in
``_checkpoints/architecture-history.md``.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Callable

from ..core import HarnessConfig
from ..observability import AutobenchObservability
from ..rsi.loop import ImprovementDelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

DEFAULT_N_INSTANCES = 3
STRATEGY_ENV = "AUTOBENCH_IMPROVER_STRATEGY"
N_INSTANCES_ENV = "AUTOBENCH_IMPROVER_ENSEMBLE_N"
VALID_STRATEGIES = ("vote", "parallel")


def _read_strategy_env(default: str = "vote") -> str:
    """Read ``AUTOBENCH_IMPROVER_STRATEGY`` from env. Falls back to ``default``."""
    raw = os.environ.get(STRATEGY_ENV)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s not in VALID_STRATEGIES:
        logger.warning(
            "Invalid %s=%r; falling back to %r", STRATEGY_ENV, raw, default,
        )
        return default
    return s


def _read_n_instances_env(default: int = DEFAULT_N_INSTANCES) -> int:
    """Read ``AUTOBENCH_IMPROVER_ENSEMBLE_N`` from env. Falls back to ``default``."""
    raw = os.environ.get(N_INSTANCES_ENV)
    if raw is None:
        return default
    try:
        v = int(raw)
        if v < 1:
            return default
        return v
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Delta aggregation
# ---------------------------------------------------------------------------

# Fields we vote on. Order is the order they appear in the vote_outcome
# field_votes dict (stable for snapshot tests).
_VOTABLE_FIELDS = (
    "system_prompt_delta",
    "rollout_protocol_changed",
    "context_manager_changed",
    "tool_surface_delta",
    "budget_delta",
)


def _delta_value(delta: ImprovementDelta, field: str) -> Any:
    """Pull a field off an ImprovementDelta, normalising for hashability."""
    val = getattr(delta, field, None)
    # budget_delta is a dict — render as sorted tuple of items for hashing.
    if isinstance(val, dict):
        return tuple(sorted(val.items()))
    return val


def _is_field_noop(value: Any, field: str) -> bool:
    """True iff ``value`` represents 'no change' for the given field.

    For bool fields the no-op value is ``False``; for str fields it's an
    empty/whitespace string; for dict fields (rendered as tuple) it's empty.
    """
    if field in ("rollout_protocol_changed", "context_manager_changed"):
        return not bool(value)
    if field in ("system_prompt_delta", "tool_surface_delta"):
        return not (isinstance(value, str) and value.strip())
    if field == "budget_delta":
        return not value  # empty tuple from _delta_value, or empty dict
    return value is None or value == ""


def _summarise_delta(delta: ImprovementDelta) -> dict[str, Any]:
    """Compact, JSON-safe fingerprint of an ImprovementDelta for the bus.

    Drops ``prediction`` (verbose) and clips long strings. The schema declares
    ``delta_summary`` as additionalProperties:true so we have wiggle room.
    """
    summary_text = (getattr(delta, "improvement_summary", "") or "")[:200]
    sysprompt = (getattr(delta, "system_prompt_delta", "") or "")[:160]
    tool_delta = (getattr(delta, "tool_surface_delta", "") or "")[:160]
    return {
        "improvement_summary": summary_text,
        "system_prompt_delta": sysprompt,
        "rollout_protocol_changed": bool(
            getattr(delta, "rollout_protocol_changed", False)
        ),
        "context_manager_changed": bool(
            getattr(delta, "context_manager_changed", False)
        ),
        "tool_surface_delta": tool_delta,
        "budget_delta": dict(getattr(delta, "budget_delta", {}) or {}),
    }


def _summarise_value(value: Any, field: str) -> Any:
    """Render a winning vote value compactly for the bus payload."""
    if field == "budget_delta":
        # Was normalised to tuple-of-items; render back to dict.
        if isinstance(value, tuple):
            return dict(value)
        return dict(value or {})
    if isinstance(value, str):
        return value[:120]
    return value


def _rebuild_harness_from_pairs(
    pairs: list[tuple[HarnessConfig, ImprovementDelta]],
    winners: dict[str, Any],
    base: HarnessConfig,
) -> HarnessConfig:
    """Pick the arm-harness whose delta matches the most winning fields.

    This is the actual rebuild path used by ``aggregate_deltas``. The earlier
    helper without pairs is kept as a non-reachable fallback for callers that
    didn't have access to pairs.
    """
    # If every winner is a no-op, return baseline unchanged.
    any_change = any(not _is_field_noop(winners[f], f) for f in _VOTABLE_FIELDS)
    if not any_change:
        return copy.deepcopy(base)

    best_arm_idx = 0
    best_match = -1
    for idx, (_h, d) in enumerate(pairs):
        match = sum(
            1 for f in _VOTABLE_FIELDS if _delta_value(d, f) == winners[f]
        )
        if match > best_match:
            best_match = match
            best_arm_idx = idx
    return copy.deepcopy(pairs[best_arm_idx][0])


def _build_delta_from_winners(
    deltas: list[ImprovementDelta], winners: dict[str, Any],
) -> ImprovementDelta:
    """Assemble a fresh ImprovementDelta carrying the winning values."""
    merged = ImprovementDelta()
    # Restore raw budget_delta dict from the normalised tuple-of-items.
    raw_budget = winners["budget_delta"]
    if isinstance(raw_budget, tuple):
        merged.budget_delta = dict(raw_budget)
    else:
        merged.budget_delta = dict(raw_budget or {})

    merged.system_prompt_delta = str(winners["system_prompt_delta"] or "")
    merged.tool_surface_delta = str(winners["tool_surface_delta"] or "")
    merged.rollout_protocol_changed = bool(winners["rollout_protocol_changed"])
    merged.context_manager_changed = bool(winners["context_manager_changed"])

    # Pick the improvement_summary from the first non-empty delta that
    # contributed at least one winning field (so the human-readable rationale
    # comes from an arm that influenced the merged outcome). Fallback to the
    # first non-empty summary across all arms.
    chosen_summary = ""
    for d in deltas:
        if any(_delta_value(d, f) == winners[f] and not _is_field_noop(winners[f], f)
               for f in _VOTABLE_FIELDS):
            if getattr(d, "improvement_summary", ""):
                chosen_summary = d.improvement_summary
                break
    if not chosen_summary:
        for d in deltas:
            if getattr(d, "improvement_summary", ""):
                chosen_summary = d.improvement_summary
                break
    merged.improvement_summary = chosen_summary or "ensemble-vote (no-op majority)"

    # Carry through any prediction from the first arm that proposed one (so
    # the AHE verifier still has something to verify next iteration). The
    # ensemble does not synthesise a new prediction.
    for d in deltas:
        pred = getattr(d, "prediction", None)
        if pred is not None:
            merged.prediction = pred
            break

    return merged


def aggregate_deltas(
    pairs: list[tuple[HarnessConfig, ImprovementDelta]],
    strategy: str = "vote",
    *,
    arm_scores: list[float] | None = None,
    baseline_harness: HarnessConfig | None = None,
) -> tuple[HarnessConfig, ImprovementDelta, dict[str, Any]]:
    """Aggregate N (harness, delta) pairs into one chosen (harness, delta).

    See module docstring for strategy semantics. Returns
    ``(chosen_harness, chosen_delta, vote_outcome)``.
    """
    if not pairs:
        raise ValueError("aggregate_deltas requires at least one (harness, delta) pair")

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {VALID_STRATEGIES!r}, got {strategy!r}"
        )

    if strategy == "parallel":
        if arm_scores is None or len(arm_scores) != len(pairs):
            raise ValueError(
                "strategy='parallel' requires arm_scores of length == len(pairs)"
            )
        best_idx = 0
        best_score = arm_scores[0]
        for i in range(1, len(arm_scores)):
            if arm_scores[i] > best_score:
                best_idx = i
                best_score = arm_scores[i]
        chosen_harness, chosen_delta = pairs[best_idx]
        vote_outcome: dict[str, Any] = {
            "selected_instance_idx": best_idx,
            "selected_score": float(best_score),
            "arm_scores": [float(s) for s in arm_scores],
        }
        return chosen_harness, chosen_delta, vote_outcome

    # strategy == "vote"
    deltas = [d for (_h, d) in pairs]
    field_votes: dict[str, Any] = {}
    ties_broken: list[str] = []
    winners: dict[str, Any] = {}

    for field in _VOTABLE_FIELDS:
        values = [_delta_value(d, field) for d in deltas]
        tally: dict[Any, int] = {}
        for v in values:
            tally[v] = tally.get(v, 0) + 1
        sorted_items = sorted(
            tally.items(),
            key=lambda kv: (-kv[1], values.index(kv[0])),
        )
        top_value, top_count = sorted_items[0]
        contenders = [v for (v, c) in sorted_items if c == top_count]
        was_tie = len(contenders) > 1
        if was_tie:
            top_value = next(
                (v for v in values if v in contenders and not _is_field_noop(v, field)),
                contenders[0],
            )
            ties_broken.append(field)
        field_votes[field] = {
            "winner": _summarise_value(top_value, field),
            "count": int(tally.get(top_value, top_count)),
            "tied": was_tie,
            "total": len(values),
        }
        winners[field] = top_value

    base = baseline_harness if baseline_harness is not None else pairs[0][0]
    chosen_harness = _rebuild_harness_from_pairs(pairs, winners, base)
    chosen_delta = _build_delta_from_winners(deltas, winners)

    selected_mask = []
    for (_h, d) in pairs:
        match = all(_delta_value(d, f) == winners[f] for f in _VOTABLE_FIELDS)
        selected_mask.append(bool(match))

    vote_outcome = {
        "selected_instance_idx": None,
        "selected_score": None,
        "field_votes": field_votes,
        "ties_broken": ties_broken,
        "selected_mask": selected_mask,
    }
    return chosen_harness, chosen_delta, vote_outcome


# ---------------------------------------------------------------------------
# MultiImproverEnsemble — the public class wired into the RSI loop
# ---------------------------------------------------------------------------

class MultiImproverEnsemble:
    """Fan out N anonymous MiniMax improvers per iteration; aggregate deltas.

    Each instance gets a fresh :class:`MiniMaxLLMWrapper` (no shared HTTP
    client, no shared retry state) so the LLM samples are independent at the
    transport layer too. The wrapper factory is configurable via constructor
    arg ``wrapper_factory`` so tests can inject mocks.

    Usage::

        ens = MultiImproverEnsemble(n_instances=3, strategy="vote")
        new_harness, delta = ens.improve(
            harness, result, obs=obs, iteration=i,
        )

    On any partial failure (one instance raises) the offending arm is dropped
    and aggregation proceeds over the surviving deltas. If ALL instances fail
    the ensemble propagates the last error to the caller — the RSI loop's
    legacy fallback layer will catch it.
    """

    def __init__(
        self,
        n_instances: int | None = None,
        strategy: str | None = None,
        wrapper_factory: Callable[[], Any] | None = None,
        arm_evaluator: Callable[[HarnessConfig], float] | None = None,
    ) -> None:
        """Build an ensemble.

        Args:
            n_instances: How many anonymous improvers to fan out. Default
                read from ``AUTOBENCH_IMPROVER_ENSEMBLE_N`` env, else 3.
            strategy: "vote" (default) or "parallel". Default read from
                ``AUTOBENCH_IMPROVER_STRATEGY`` env, else "vote".
            wrapper_factory: Callable returning a fresh improver wrapper.
                Default builds a new :class:`MiniMaxLLMWrapper` per call.
                Tests inject mocks here.
            arm_evaluator: Strategy "parallel" only. Callable that takes a
                candidate :class:`HarnessConfig` and returns its aggregate
                score on one forward evaluation. Required when strategy is
                "parallel"; ignored otherwise.
        """
        self.n_instances = n_instances if n_instances is not None else _read_n_instances_env()
        if self.n_instances < 1:
            raise ValueError(f"n_instances must be >= 1, got {self.n_instances}")
        self.strategy = strategy if strategy is not None else _read_strategy_env()
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {VALID_STRATEGIES!r}, got {self.strategy!r}"
            )
        if wrapper_factory is None:
            # Late binding via the back-compat shim so tests that
            # ``monkeypatch.setattr(autobench.multi_improver,
            # "_default_wrapper_factory", ...)`` still take effect.
            import autobench.llm.ensemble
            self._wrapper_factory: Callable[[], Any] = (
                autobench.multi_improver._default_wrapper_factory
            )
        else:
            self._wrapper_factory = wrapper_factory
        self._arm_evaluator = arm_evaluator

    def improve(
        self,
        current_harness: HarnessConfig,
        benchmark_results: Any,
        *,
        obs: AutobenchObservability | None = None,
        iteration: int = 0,
        revert_history: list[dict[str, Any]] | None = None,
        cross_advocate_context: list[Any] | None = None,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        """Run the ensemble and return one merged (harness, delta) pair.

        When ``self.n_instances == 1`` this is a thin pass-through to a single
        wrapper.improve() call with no ensemble event emitted — preserving
        zero-regression on the single-improver path.
        """
        # Short-circuit n=1: behave identically to the legacy single-improver
        # path (no ensemble event, no aggregation overhead).
        if self.n_instances == 1:
            wrapper = self._wrapper_factory()
            return self._call_one(
                wrapper, current_harness, benchmark_results,
                obs=obs, iteration=iteration,
                revert_history=revert_history,
                cross_advocate_context=cross_advocate_context,
            )

        pairs: list[tuple[HarnessConfig, ImprovementDelta]] = []
        last_error: Exception | None = None
        for idx in range(self.n_instances):
            wrapper = self._wrapper_factory()
            try:
                h, d = self._call_one(
                    wrapper, current_harness, benchmark_results,
                    obs=obs, iteration=iteration,
                    revert_history=revert_history,
                    cross_advocate_context=cross_advocate_context,
                )
                pairs.append((h, d))
            except Exception as exc:  # noqa: BLE001 — drop arm, continue
                last_error = exc
                logger.warning(
                    "[9xd] ensemble arm %d/%d failed (%s); dropping",
                    idx + 1, self.n_instances, exc,
                )

        if not pairs:
            # Every arm died — re-raise the last error so the RSI loop's
            # legacy fallback (rule-based improver) catches it. We do NOT
            # emit an ensemble event when there's nothing to record.
            assert last_error is not None
            raise last_error

        # Aggregate. Strategy "parallel" needs per-arm forward scores.
        arm_scores: list[float] | None = None
        if self.strategy == "parallel":
            if self._arm_evaluator is None:
                logger.warning(
                    "[9xd] strategy='parallel' requested but no arm_evaluator; "
                    "falling back to 'vote' for this call",
                )
                strategy_used = "vote"
            else:
                strategy_used = "parallel"
                arm_scores = []
                for (cand_h, _d) in pairs:
                    try:
                        s = float(self._arm_evaluator(cand_h))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[9xd] arm_evaluator raised on candidate (%s); "
                            "scoring as -inf",
                            exc,
                        )
                        s = float("-inf")
                    arm_scores.append(s)
        else:
            strategy_used = "vote"

        chosen_harness, chosen_delta, vote_outcome = aggregate_deltas(
            pairs,
            strategy=strategy_used,
            arm_scores=arm_scores,
            baseline_harness=current_harness,
        )

        if obs is not None:
            _emit_ensemble_event(
                obs=obs,
                iteration=iteration,
                strategy=strategy_used,
                pairs=pairs,
                vote_outcome=vote_outcome,
                arm_scores=arm_scores,
            )

        return chosen_harness, chosen_delta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_one(
        self,
        wrapper: Any,
        current_harness: HarnessConfig,
        benchmark_results: Any,
        *,
        obs: AutobenchObservability | None,
        iteration: int,
        revert_history: list[dict[str, Any]] | None,
        cross_advocate_context: list[Any] | None,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        """Invoke one wrapper.improve() with graceful kwarg degradation.

        Mirrors the kwarg-degradation chain in
        :meth:`SelfImprovingHarness._resolve_improver_fn` so wrappers without
        all four kwargs still work.
        """
        try:
            return wrapper.improve(
                current_harness, benchmark_results,
                obs=obs, iteration=iteration,
                revert_history=revert_history,
                cross_advocate_context=cross_advocate_context,
            )
        except TypeError:
            try:
                return wrapper.improve(
                    current_harness, benchmark_results,
                    obs=obs, iteration=iteration,
                    revert_history=revert_history,
                )
            except TypeError:
                try:
                    return wrapper.improve(
                        current_harness, benchmark_results,
                        obs=obs, iteration=iteration,
                    )
                except TypeError:
                    return wrapper.improve(current_harness, benchmark_results)


def _default_wrapper_factory() -> Any:
    """Build a fresh :class:`MiniMaxLLMWrapper` per call.

    Imported lazily so tests can patch ``autobench.minimax_improver`` without
    forcing MINIMAX_API_KEY at import time. Lookup goes via the back-compat
    shim at the old root path so test monkeypatches still take effect.
    """
    import autobench.llm.minimax  # noqa: WPS433 — lazy for tests
    return autobench.minimax_improver.MiniMaxLLMWrapper()


# ---------------------------------------------------------------------------
# Observability emitter
# ---------------------------------------------------------------------------

def _emit_ensemble_event(
    obs: AutobenchObservability,
    iteration: int,
    strategy: str,
    pairs: list[tuple[HarnessConfig, ImprovementDelta]],
    vote_outcome: dict[str, Any],
    arm_scores: list[float] | None,
) -> None:
    """Emit ``autobench.improver.ensemble.v1`` once per ensemble call.

    Best-effort: any exception is swallowed so observability never breaks the
    loop. Routes through ``obs.improver_ensemble_complete`` when present
    (added by this same patch), falling back to ``obs._publish`` as a
    last-resort if an older observability layer is in play.
    """
    instances: list[dict[str, Any]] = []
    selected_mask = vote_outcome.get("selected_mask") or [False] * len(pairs)
    selected_idx = vote_outcome.get("selected_instance_idx")
    for idx, (_h, d) in enumerate(pairs):
        entry: dict[str, Any] = {
            "instance_idx": idx,
            "delta_summary": _summarise_delta(d),
            "score": (
                float(arm_scores[idx])
                if (arm_scores is not None and idx < len(arm_scores))
                else None
            ),
            "selected": bool(
                (selected_idx is not None and idx == selected_idx)
                or (selected_idx is None and idx < len(selected_mask) and selected_mask[idx])
            ),
        }
        instances.append(entry)

    try:
        emitter = getattr(obs, "improver_ensemble_complete", None)
        if emitter is not None:
            emitter(
                iteration=iteration,
                strategy=strategy,
                n_instances=len(pairs),
                instances=instances,
                vote_outcome=vote_outcome,
            )
            return
        # Fallback for legacy obs without the new emitter method — go direct.
        from ..observability import CHANNEL_IMPROVER_ENSEMBLE  # type: ignore[attr-defined]
        obs._publish(  # noqa: SLF001 — fallback
            CHANNEL_IMPROVER_ENSEMBLE,
            {
                "iteration": int(iteration),
                "strategy": str(strategy),
                "n_instances": len(pairs),
                "instances": instances,
                "vote_outcome": vote_outcome,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[9xd] ensemble event emission failed: %s", exc)


__all__ = [
    "MultiImproverEnsemble",
    "aggregate_deltas",
    "VALID_STRATEGIES",
    "DEFAULT_N_INSTANCES",
    "STRATEGY_ENV",
    "N_INSTANCES_ENV",
]
