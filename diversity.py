"""SACS (Skill-Aware Cosine Similarity) diversity penalty for the RSI loop.

Operationalizes R-Diverse (arXiv:2602.13103) Memory-Augmented Penalty (MAP)
over a rolling bank of ImprovementDelta signatures.

Design choice (see research/diversity_penalty_2026.md §4):
    The embedding is a 23-dim STRUCTURAL FINGERPRINT of which fields changed
    and in which direction — NOT a lexical embedding of the rationale text.
    Lexical embeddings recreate the "Surface Diversity Illusion" the penalty
    is meant to cure: two deltas with different prose but identical structural
    mutations should be flagged as redundant.

The SACS twist (vs raw cosine):
    sim = cos(fp_a, fp_b) * overlap_ratio
    overlap_ratio = |fields_changed_a ∩ fields_changed_b| /
                    |fields_changed_a ∪ fields_changed_b|
    This dampens spurious similarity when two deltas touch entirely different
    fields but happen to produce parallel direction vectors in the shared
    one-hot subspace.

Where it plugs in:
    - rsi_loop.SelfImprovingHarness instantiates a DiversityTracker.
    - After each improver call, the tracker records the delta.
    - convergence_check() uses adjusted_utility = aggregate_score - penalty
      to decide plateau (penalty applied here, NOT in evaluator.score_harness —
      that stays a pure function for A/B integrity).
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .rsi_loop import ImprovementDelta


# ---------------------------------------------------------------------------
# Fingerprint geometry (23 dimensions)
# ---------------------------------------------------------------------------
#
# Index  Dims  Meaning
# -----  ----  --------------------------------------------------------------
#   0     1    system_prompt_delta present?     (1.0 if non-empty else 0.0)
#   1     1    rollout_protocol_changed?        (1.0 / 0.0)
#   2     1    context_manager_changed?         (1.0 / 0.0)
#   3     1    tool_surface_delta present?      (1.0 / 0.0)
#   4-7   4    budget field presence            (max_tokens, max_time,
#                                                max_cost, max_memory)
#                                                1.0 if key in budget_delta
#                                                else 0.0
#   8-13  6    budget direction (sign)          for {max_tokens,
#                                                max_time_seconds,
#                                                max_cost_dollars,
#                                                max_memory_mb,
#                                                + two reserved slots for
#                                                  future budget fields:
#                                                  max_tool_calls,
#                                                  max_concurrent_cases}
#                                                values in {-1, 0, +1}
#  14-17  4    one-hot new context_manager      (FULL, BUDGETED, SEMANTIC,
#                                                HIERARCHICAL) — only set if
#                                                context_manager_changed.
#                                                Read from explicit
#                                                ``new_context_manager``
#                                                attr if present on delta,
#                                                else all zeros.
#  18-21  4    one-hot new rollout_protocol     (SINGLE, ITERATIVE,
#                                                SELF_REVISION, MONTE_CARLO)
#                                                same convention as above.
#  22     1    improvement_summary length       z-scored against a running
#                                                proxy mean (heuristic
#                                                normalisation: tanh of
#                                                (len - 60) / 60). Captures
#                                                "delta ambition" without
#                                                lexical content.
#
# Total: 23. Cosine similarity over this vector measures STRUCTURAL distance.
# Two deltas that both "shrink max_tokens by some amount" will have nearly
# identical fingerprints regardless of how their rationales read — exactly
# the desired property for skill-aware diversity.
# ---------------------------------------------------------------------------

FINGERPRINT_DIM = 23

# Budget keys, in fingerprint order (used for both presence + direction slots).
_BUDGET_KEYS = (
    "max_tokens",
    "max_time_seconds",
    "max_cost_dollars",
    "max_memory_mb",
    "max_tool_calls",      # reserved
    "max_concurrent_cases",  # reserved
)

# Context-manager one-hot order. Values come from autobench.core.ContextManager
# but we deliberately use string keys here to avoid an import cycle with
# rsi_loop (which is where ImprovementDelta lives).
_CTX_ORDER = ("full", "budgeted", "semantic", "hierarchical")

# Rollout-protocol one-hot order.
_PROTO_ORDER = ("single", "iterative", "self_revision", "monte_carlo")


def _norm_str(v: Any) -> str:
    """Normalize an enum-or-string to its lowercase value form."""
    if v is None:
        return ""
    if hasattr(v, "value"):
        return str(v.value).lower()
    return str(v).lower()


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


class StructuralFingerprint:
    """Compute a 23-dim structural fingerprint of an ImprovementDelta.

    Stateless utility; lives as a class so future variants (e.g., a
    code-encoder-backed signature) can subclass without changing call sites.
    """

    DIM = FINGERPRINT_DIM

    @staticmethod
    def from_delta(delta: "ImprovementDelta") -> np.ndarray:
        """Encode a delta as a 23-dim float vector.

        See the module-level docstring for the dimension layout.
        """
        vec = np.zeros(FINGERPRINT_DIM, dtype=np.float64)

        # 0..3: presence of textual / categorical changes.
        vec[0] = 1.0 if getattr(delta, "system_prompt_delta", "") else 0.0
        vec[1] = 1.0 if getattr(delta, "rollout_protocol_changed", False) else 0.0
        vec[2] = 1.0 if getattr(delta, "context_manager_changed", False) else 0.0
        vec[3] = 1.0 if getattr(delta, "tool_surface_delta", "") else 0.0

        budget_delta = getattr(delta, "budget_delta", {}) or {}

        # 4..7: presence of each of the four primary budget fields.
        for i, key in enumerate(_BUDGET_KEYS[:4]):
            vec[4 + i] = 1.0 if key in budget_delta else 0.0

        # 8..13: direction of change for each budget field.
        # We do not have prev_budget here (the delta only stores the new
        # values), so the "direction" is interpreted as the sign of the
        # value itself when it's encoded as a signed delta, OR the sign of
        # (new - some_implicit_baseline) if a default exists.
        #
        # In practice the rule-based improver writes the *new absolute value*
        # into budget_delta (not a relative delta), so we cannot recover the
        # direction without prev_budget. We treat presence as +1 (a change
        # happened) for slots where direction is unknown — this still
        # distinguishes "touched max_tokens" from "did not touch max_tokens"
        # but does not distinguish "more" from "less". When callers wish to
        # carry direction explicitly, they can attach a
        # ``budget_direction`` dict to the delta with keys in _BUDGET_KEYS
        # and values in {-1, 0, +1}.
        directions: dict[str, float] = {}
        explicit = getattr(delta, "budget_direction", None)
        if isinstance(explicit, dict):
            for k, v in explicit.items():
                try:
                    directions[k] = _sign(float(v))
                except (TypeError, ValueError):
                    directions[k] = 0.0
        for i, key in enumerate(_BUDGET_KEYS):
            if key in directions:
                vec[8 + i] = directions[key]
            elif key in budget_delta:
                # Default: presence implies "change in unknown direction".
                # We encode as +1 so two redundant "touch this field" deltas
                # are seen as similar (their primary information is the
                # field, not the sign).
                vec[8 + i] = 1.0
            else:
                vec[8 + i] = 0.0

        # 14..17: one-hot new context_manager (only meaningful when changed).
        new_ctx = _norm_str(getattr(delta, "new_context_manager", None))
        if vec[2] > 0.0 and new_ctx in _CTX_ORDER:
            vec[14 + _CTX_ORDER.index(new_ctx)] = 1.0

        # 18..21: one-hot new rollout_protocol (only meaningful when changed).
        new_proto = _norm_str(getattr(delta, "new_rollout_protocol", None))
        if vec[1] > 0.0 and new_proto in _PROTO_ORDER:
            vec[18 + _PROTO_ORDER.index(new_proto)] = 1.0

        # 22: heuristic z-score of improvement_summary length. tanh-normalized
        # so it stays in [-1, 1] regardless of summary verbosity. This is a
        # WEAK signal — its role is to be a tiebreaker between otherwise
        # identical structural fingerprints, not to drive cosine on its own.
        summary = getattr(delta, "improvement_summary", "") or ""
        vec[22] = math.tanh((len(summary) - 60.0) / 60.0)

        return vec

    @staticmethod
    def fields_changed(delta: "ImprovementDelta") -> frozenset[str]:
        """Return the set of *field names* this delta touches.

        Used by SACS to compute the overlap ratio that scales raw cosine
        similarity. Field granularity is coarse (one entry per top-level
        field), so two deltas touching the same field but in different
        directions still count as "overlapping" — direction is captured by
        the fingerprint vector itself.
        """
        changed: set[str] = set()
        if getattr(delta, "system_prompt_delta", ""):
            changed.add("system_prompt")
        if getattr(delta, "rollout_protocol_changed", False):
            changed.add("rollout_protocol")
        if getattr(delta, "context_manager_changed", False):
            changed.add("context_manager")
        if getattr(delta, "tool_surface_delta", ""):
            changed.add("tool_surface")
        for key in (getattr(delta, "budget_delta", {}) or {}).keys():
            changed.add(f"budget.{key}")
        return frozenset(changed)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 if either vector is the zero vector (no signal to compare).
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sacs_similarity(
    fp_a: np.ndarray,
    fp_b: np.ndarray,
    fields_a: frozenset[str],
    fields_b: frozenset[str],
) -> float:
    """Skill-Aware Cosine Similarity.

    sim = cos(fp_a, fp_b) * overlap_ratio
    overlap_ratio = |A ∩ B| / |A ∪ B|   (Jaccard over changed-field names)

    When neither delta changes anything, overlap is undefined → return 0.0.
    """
    cos_sim = _cosine(fp_a, fp_b)
    union = fields_a | fields_b
    if not union:
        return 0.0
    overlap_ratio = len(fields_a & fields_b) / len(union)
    return cos_sim * overlap_ratio


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lineage signature — cross-advocate aggregation (Phase 2 of wire-pop epic,
# nervous-bus-bo86). Aggregates a per-advocate sequence of ImprovementDeltas
# into a single "lineage signature" so the PopulationRunner can compute
# pairwise diversity between sibling advocates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineageSignature:
    """A single advocate's aggregated structural footprint over its iterations."""

    mean_fingerprint: np.ndarray
    fields_changed: frozenset[str]
    n_deltas: int


def lineage_signature(deltas: list["ImprovementDelta"]) -> LineageSignature:
    """Aggregate a sequence of ImprovementDeltas into a LineageSignature.

    Empty input → zero-vector signature with empty field set. Aggregation
    is by MEAN over per-delta fingerprints (keeps cosine well-scaled) and
    UNION over changed-field sets.
    """
    if not deltas:
        return LineageSignature(
            mean_fingerprint=np.zeros(FINGERPRINT_DIM, dtype=np.float64),
            fields_changed=frozenset(),
            n_deltas=0,
        )
    fps: list[np.ndarray] = []
    fields: set[str] = set()
    for d in deltas:
        fps.append(StructuralFingerprint.from_delta(d))
        fields |= set(StructuralFingerprint.fields_changed(d))
    mean_fp = np.mean(np.stack(fps), axis=0)
    return LineageSignature(
        mean_fingerprint=mean_fp,
        fields_changed=frozenset(fields),
        n_deltas=len(deltas),
    )


def pairwise_lineage_similarity(a: LineageSignature, b: LineageSignature) -> float:
    """SACS similarity between two lineage signatures (0.0 if either is empty)."""
    if a.n_deltas == 0 or b.n_deltas == 0:
        return 0.0
    return sacs_similarity(
        a.mean_fingerprint, b.mean_fingerprint, a.fields_changed, b.fields_changed
    )


def pairwise_lineage_distance(a: LineageSignature, b: LineageSignature) -> float:
    """Distance metric: 1.0 - similarity, clamped to [0.0, 1.0].

    Two empty lineages → 0.0 (haven't differentiated). One empty +
    one non-empty → 1.0 (max novelty).
    """
    if a.n_deltas == 0 and b.n_deltas == 0:
        return 0.0
    if a.n_deltas == 0 or b.n_deltas == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - pairwise_lineage_similarity(a, b)))


@dataclass
class DiversityTracker:
    """R-Diverse-style MAP penalty over a rolling bank of ImprovementDeltas.

    Args:
        memory_size: Number of recent deltas retained.
        beta: Mixing coefficient between max-similarity (anti-recycling) and
            mean-similarity (anti-clustering) terms. The research uses gamma
            for this; we follow the bead's naming.
        tau_max: Threshold above which the max-similarity term contributes.
        tau_mean: Threshold above which the mean-similarity term contributes.
        min_bank: Number of entries required before any penalty is computed
            (warm-up). Below this, penalty_for() returns 0.0.

    Defaults are slightly looser than R-Diverse's reported values
    (tau_max=0.5, tau_mean=0.25) because our 23-dim action space yields
    naturally higher cosines than a 768-dim text embedding.
    """

    memory_size: int = 20
    beta: float = 0.15
    tau_max: float = 0.6
    tau_mean: float = 0.3
    min_bank: int = 2
    _bank: deque = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        # deque(maxlen=) cannot be set via default_factory cleanly; rebuild.
        self._bank = deque(maxlen=self.memory_size)

    # -- Recording -----------------------------------------------------

    def record(self, delta: "ImprovementDelta") -> None:
        """Push a delta's fingerprint + changed-field set into memory."""
        fp = StructuralFingerprint.from_delta(delta)
        fields = StructuralFingerprint.fields_changed(delta)
        self._bank.append((fp, fields))

    # -- Querying ------------------------------------------------------

    def _similarities(self, fp: np.ndarray, fields: frozenset[str]) -> list[float]:
        return [sacs_similarity(fp, b_fp, fields, b_fields)
                for (b_fp, b_fields) in self._bank]

    def penalty_for(self, candidate_delta: "ImprovementDelta") -> float:
        """Compute the MAP penalty for a candidate delta against memory.

        Returns 0.0 during warm-up (|M| < min_bank). Otherwise:
            penalty = beta   * max(0, P_max  - tau_max)
                    + (1-beta) * max(0, P_mean - tau_mean)
        """
        if len(self._bank) < self.min_bank:
            return 0.0
        fp = StructuralFingerprint.from_delta(candidate_delta)
        fields = StructuralFingerprint.fields_changed(candidate_delta)
        sims = self._similarities(fp, fields)
        if not sims:
            return 0.0
        p_max = max(sims)
        p_mean = sum(sims) / len(sims)
        return (self.beta * max(0.0, p_max - self.tau_max)
                + (1.0 - self.beta) * max(0.0, p_mean - self.tau_mean))

    def current_diversity_score(self) -> float:
        """1.0 = fully diverse, 0.0 = fully redundant. Observability only.

        Computed as 1 - mean pairwise SACS similarity across memory. For
        |M| < 2 returns 1.0 (no evidence of redundancy yet).
        """
        if len(self._bank) < 2:
            return 1.0
        bank = list(self._bank)
        sims: list[float] = []
        for i, (fp_a, fields_a) in enumerate(bank):
            for fp_b, fields_b in bank[i + 1:]:
                sims.append(sacs_similarity(fp_a, fp_b, fields_a, fields_b))
        if not sims:
            return 1.0
        return 1.0 - (sum(sims) / len(sims))

    def apply_to_utility(
        self,
        raw_utility: float,
        candidate_delta: "ImprovementDelta",
    ) -> float:
        """Return ``raw_utility - penalty_for(candidate_delta)``.

        The penalty is intentionally NOT applied inside evaluator.score_harness
        (that would couple the evaluator to RSI state and break A/B integrity).
        Callers in rsi_loop apply it to compute an adjusted_utility used only
        for convergence detection and trajectory selection.
        """
        return raw_utility - self.penalty_for(candidate_delta)

    def snapshot(self) -> dict[str, Any]:
        """Return a small observability-friendly dict describing tracker state."""
        return {
            "memory_size": len(self._bank),
            "diversity_score": self.current_diversity_score(),
            "tau_max": self.tau_max,
            "tau_mean": self.tau_mean,
            "beta": self.beta,
            "capacity": self.memory_size,
        }


# ---------------------------------------------------------------------------
# Optional A/B helper
# ---------------------------------------------------------------------------


def run_ab_comparison(
    harness_factory,
    cases: list[Any],
    evaluator,
    iterations: int = 10,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the same RSI setup twice (with and without diversity tracker).

    ``harness_factory`` is a zero-arg callable that returns a fresh
    SelfImprovingHarness each call (so the two arms start from identical
    state but do not share mutation history).

    Returns a dict with arms ``without`` and ``with`` containing final
    aggregate_score and the per-iteration trajectory. Used by the future
    "A/B harness — diversity on vs off on CodeForces-20" bead.

    Note: this is a thin convenience wrapper; it does not perform statistical
    inference over multiple seeds — caller is expected to iterate.
    """
    import random

    if seed is not None:
        random.seed(seed)

    def _run(with_tracker: bool) -> dict[str, Any]:
        h = harness_factory()
        # Force the iteration count.
        h.max_iterations = iterations
        if with_tracker:
            h.diversity_tracker = DiversityTracker()
        else:
            h.diversity_tracker = None
        _, result, history = h.improve(cases)
        return {
            "final_score": result.aggregate_score if result else None,
            "trajectory": [r.aggregate_score for _, r, _ in history],
            "diversity_snapshot": (
                h.diversity_tracker.snapshot() if h.diversity_tracker else None
            ),
        }

    return {"without": _run(False), "with": _run(True)}
