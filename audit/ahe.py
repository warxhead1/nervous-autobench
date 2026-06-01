"""Agent Harness Evolution (AHE) — prediction contract for RSI.

Per arXiv:2604.25850 ("Agent Harness Evolution"), RSI gains 7.3pp on
Terminal-Bench (69.7% → 77.0% over 10 iterations) when every improver edit
ships a machine-checkable prediction about the *next* iteration's outcome,
verified against actuals. This turns RSI from a stochastic ratchet into a
falsifiable contract: when MiniMax says "this edit will recover 3 TLE cases
to OK with high confidence" and the next run shows TLE↑ and OK↓ we *see*
the antagonist lose, instead of merely seeing the score drift.

Surface:
    Prediction                  — improver's self-declared expectation
    PredictionVerification      — actual-vs-predicted outcome
    parse_prediction_from_llm_response — extract prediction JSON from LLM text
    verify_prediction           — verify a prediction against prev/curr results
    should_emit_warning         — flag confident-but-refuted predictions
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .invalidation import (
    InvalidationEngine,
    ahe_scope_key,
    get_invalidation_engine,
)

logger = logging.getLogger(__name__)


# Direction match thresholds.
_CONFIDENT_THRESHOLD = 0.7
_CONFIRMED_VERDICT_RATIO = 0.75
_PARTIAL_VERDICT_RATIO = 0.5
# Score-delta tolerance: an absolute error below this is treated as "correct
# magnitude" for purposes of outcome classification. Predictions live in score
# units (0..1 aggregate); 0.02 is one third of a "meaningful" 0.05 step.
_SCORE_DELTA_TOLERANCE = 0.05


@dataclass
class Prediction:
    """Improver's self-declared prediction about the next iteration.

    Attributes:
        predicted_score_delta: Expected change in ``aggregate_score`` (signed).
        predicted_verdict_class_changes: Map of verdict letter → signed count
            delta. e.g. ``{"TLE": -3, "OK": +3}`` means "I expect 3 TLE cases
            to flip to OK".
        confidence: Self-reported confidence in 0..1.
        rationale: One-sentence text explanation.
        active: Whether this prediction is currently the active one (default True).
            Inactive predictions are superseded or deduplicated.
        parent_prediction_id: ID of the prediction this one superseded (if any).
        fact_fingerprint: SHA-256 of normalized key+guidance for deduplication.
            Computed at emission time; None until then.
        prediction_id: Unique identifier for this prediction (ULID-like).
    """

    predicted_score_delta: float = 0.0
    predicted_verdict_class_changes: dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = ""
    active: bool = True
    parent_prediction_id: str | None = None
    fact_fingerprint: str | None = None
    prediction_id: str = ""


@dataclass
class PredictionVerification:
    """Outcome of verifying a Prediction against actual next-iteration results.

    Attributes:
        predicted: The Prediction we are verifying.
        actual_score_delta: Observed change in aggregate_score.
        actual_verdict_class_changes: Observed verdict-count deltas.
        score_delta_error: ``|predicted - actual|`` in score units.
        verdict_match_ratio: Fraction of predicted verdict changes whose sign
            matched the actual sign. ``1.0`` = every predicted shift went the
            right direction.
        outcome_label: ``"confirmed"`` | ``"partial"`` | ``"refuted"``.
        confidence_calibration: Gap between stated confidence and observed
            accuracy. ``0.0`` = perfectly calibrated; large = overconfident or
            underconfident relative to outcome.
        lifecycle_status: Lifecycle state — ``"active"`` (not yet verified),
            ``"superseded"`` (replaced by newer prediction for same scope),
            ``"duplicate"`` (deduplicated by fingerprint), or
            ``"rejected"`` (verification outcome was refuted).
        contested_score_multiplier: Downweight for contested cases (dissent_ratio
            > 0.4). Computed as ``1 - dissent_ratio`` so disputed cases count
            less toward confirming the prediction. ``1.0`` when the pool was
            disabled or no dissent data was available.
    """

    predicted: Prediction
    actual_score_delta: float
    actual_verdict_class_changes: dict[str, int]
    score_delta_error: float
    verdict_match_ratio: float
    outcome_label: str
    confidence_calibration: float
    lifecycle_status: Literal["active", "superseded", "duplicate", "rejected"] = "active"
    contested_score_multiplier: float = 1.0


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_prediction_from_llm_response(raw_response: str) -> Prediction | None:
    """Extract a ``prediction`` JSON block from LLM output.

    Looks for a top-level ``"prediction"`` key inside the first JSON object in
    the response. Tolerant of code fences, partial fields, and stray prose
    surrounding the JSON. Returns ``None`` if no parseable prediction is found.

    Examples
    --------
    >>> parse_prediction_from_llm_response('{"prediction": {"predicted_score_delta": 0.05, "confidence": 0.8}}')  # doctest: +ELLIPSIS
    Prediction(predicted_score_delta=0.05, ...)

    >>> parse_prediction_from_llm_response("no JSON here") is None
    True
    """
    if not raw_response:
        return None

    text = raw_response.strip()
    if text.startswith("```"):
        # Strip a leading fence line and trailing fence line if present.
        lines = text.splitlines()
        if len(lines) >= 2:
            # Drop first line ("```json" or "```")
            lines = lines[1:]
            # Drop trailing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

    # Find the first JSON object in the text.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    block = parsed.get("prediction")
    if not isinstance(block, dict):
        return None

    try:
        score_delta = float(block.get("predicted_score_delta", 0.0) or 0.0)
    except (TypeError, ValueError):
        score_delta = 0.0

    verdict_changes_raw = block.get("predicted_verdict_class_changes", {}) or {}
    verdict_changes: dict[str, int] = {}
    if isinstance(verdict_changes_raw, dict):
        for k, v in verdict_changes_raw.items():
            try:
                verdict_changes[str(k)] = int(v)
            except (TypeError, ValueError):
                continue

    try:
        confidence = float(block.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(block.get("rationale", "") or "")

    return Prediction(
        predicted_score_delta=score_delta,
        predicted_verdict_class_changes=verdict_changes,
        confidence=confidence,
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def _verdict_counts(result: Any) -> dict[str, int]:
    """Normalise verdict counts on a BenchmarkResult-like to ``{str: int}``.

    Handles either string keys ("OK","TLE") or Verdict-enum keys (.value lookup).
    Read-only.
    """
    raw = getattr(result, "verdict_counts", {}) or {}
    out: dict[str, int] = {}
    for k, v in raw.items():
        key = k.value if hasattr(k, "value") else str(k)
        try:
            out[key] = out.get(key, 0) + int(v)
        except (TypeError, ValueError):
            continue
    return out


def _verdict_diff(prev: dict[str, int], curr: dict[str, int]) -> dict[str, int]:
    """Return signed verdict-count delta (curr - prev) for the union of keys."""
    keys = set(prev) | set(curr)
    return {k: int(curr.get(k, 0)) - int(prev.get(k, 0)) for k in keys}


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def verify_prediction(
    prediction: Prediction,
    prev_result: Any,
    curr_result: Any,
    dissent_ratio: float = 0.0,
) -> PredictionVerification:
    """Verify a Prediction against the next iteration's actual results.

    ``prev_result`` is the iteration *during which* the prediction was made
    (its aggregate_score and verdict_counts are the baseline). ``curr_result``
    is the very next iteration (the one the prediction is *about*).
    ``dissent_ratio`` is the fraction of judges who dissented from the majority
    verdict for this iteration's benchmark run (0.0 when the judge pool was
    disabled or no dissent data was available).

    Outcome classification:
        - ``confirmed``  — verdict_match_ratio ≥ 0.75 AND score_delta_error ≤ tol
        - ``partial``    — verdict_match_ratio ≥ 0.5 OR score_delta_error ≤ tol
        - ``refuted``    — otherwise
    """
    prev_score = float(getattr(prev_result, "aggregate_score", 0.0))
    curr_score = float(getattr(curr_result, "aggregate_score", 0.0))
    actual_score_delta = curr_score - prev_score
    score_delta_error = abs(prediction.predicted_score_delta - actual_score_delta)

    prev_counts = _verdict_counts(prev_result)
    curr_counts = _verdict_counts(curr_result)
    actual_verdict_changes = _verdict_diff(prev_counts, curr_counts)

    # Compute verdict match ratio: fraction of predicted-nonzero entries whose
    # sign matches the actual sign. If no predictions were made, treat as 1.0
    # (vacuously true) — the score-delta term will still drive outcome.
    predicted_nonzero = {
        k: v for k, v in prediction.predicted_verdict_class_changes.items() if v != 0
    }
    if not predicted_nonzero:
        verdict_match_ratio = 1.0
    else:
        matches = 0
        for k, predicted_v in predicted_nonzero.items():
            actual_v = actual_verdict_changes.get(k, 0)
            if _sign(predicted_v) == _sign(actual_v):
                matches += 1
        verdict_match_ratio = matches / len(predicted_nonzero)

    score_ok = score_delta_error <= _SCORE_DELTA_TOLERANCE

    if verdict_match_ratio >= _CONFIRMED_VERDICT_RATIO and score_ok:
        outcome_label = "confirmed"
        observed_accuracy = 1.0
    elif verdict_match_ratio >= _PARTIAL_VERDICT_RATIO or score_ok:
        outcome_label = "partial"
        observed_accuracy = 0.5
    else:
        outcome_label = "refuted"
        observed_accuracy = 0.0

    confidence_calibration = abs(prediction.confidence - observed_accuracy)

    lifecycle_status = getattr(prediction, "lifecycle_status", "active") or "active"

    # nervous-bus: downweight contested cases. Cases with dissent_ratio > 0.4
    # are multiplied by (1 - dissent_ratio). For example, dissent=0.5 → 0.5.
    # Dissent <= 0.4 passes through at full weight (multiplier = 1.0).
    contested_multiplier = 1.0
    if dissent_ratio > 0.4:
        contested_multiplier = max(0.0, 1.0 - dissent_ratio)

    return PredictionVerification(
        predicted=prediction,
        actual_score_delta=actual_score_delta,
        actual_verdict_class_changes=actual_verdict_changes,
        score_delta_error=score_delta_error,
        verdict_match_ratio=verdict_match_ratio,
        outcome_label=outcome_label,
        confidence_calibration=confidence_calibration,
        lifecycle_status=lifecycle_status,
        contested_score_multiplier=contested_multiplier,
    )


def should_emit_warning(verification: PredictionVerification) -> bool:
    """Return True when the prediction was confidently wrong.

    A "confidently wrong" event is one we want to surface loudly: the improver
    asserted high confidence and the outcome was refuted. This is the
    falsifiability signal the AHE paper relies on — without it, the antagonist
    can stay confidently wrong forever.
    """
    return (
        verification.predicted.confidence >= _CONFIDENT_THRESHOLD
        and verification.outcome_label == "refuted"
    )


# --------------------------------------------------------------------------- #
# Serialization helpers (for emitting via observability)
# --------------------------------------------------------------------------- #

def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    """Render a Prediction as a JSON-safe dict (for bus payloads)."""
    out = asdict(prediction)
    # Omit empty string prediction_id from old callers for back-compat.
    if out.get("prediction_id") == "":
        del out["prediction_id"]
    return out


def verification_to_dict(verification: PredictionVerification) -> dict[str, Any]:
    """Render a PredictionVerification as a JSON-safe dict (for bus payloads)."""
    return {
        "predicted": prediction_to_dict(verification.predicted),
        "actual_score_delta": verification.actual_score_delta,
        "actual_verdict_class_changes": dict(verification.actual_verdict_class_changes),
        "score_delta_error": verification.score_delta_error,
        "verdict_match_ratio": verification.verdict_match_ratio,
        "outcome_label": verification.outcome_label,
        "confidence_calibration": verification.confidence_calibration,
        "lifecycle_status": verification.lifecycle_status,
        "contested_score_multiplier": verification.contested_score_multiplier,
    }


# --------------------------------------------------------------------------- #
# Live partial refutation (nervous-bus-ykn)
#
# For a long RSI iteration (30+ minutes), a wrong prediction sits unflagged
# until the iteration boundary even though cases arriving mid-flight already
# prove it impossible. ``refute_live`` answers: given the cases we have so far
# from iter N+1, is the prediction still achievable for the cases that remain?
# A prediction is refuted when, for any predicted verdict-count delta, the
# maximum still-achievable delta (assuming every remaining case lands on the
# best possible verdict) cannot match the predicted magnitude.
# --------------------------------------------------------------------------- #


@dataclass
class LiveRefutationStatus:
    """Snapshot of whether a Prediction can still be true given partial actuals.

    Attributes:
        prediction: The Prediction we are evaluating.
        actuals_so_far: Verdict-count histogram observed in iter N+1 to date.
        remaining_cases: Number of iter-N+1 cases not yet dispatched.
        is_refuted: True iff at least one predicted_verdict_class_changes entry
            has become mathematically unachievable.
        refutation_reason: Human-readable diagnosis (empty when not refuted).
        confidence_at_refute: Carries the original confidence at refutation
            time so downstream calibration tracking can correlate confident-
            yet-refuted predictions. None when not refuted.
    """

    prediction: Prediction
    actuals_so_far: dict[str, int]
    remaining_cases: int
    is_refuted: bool
    refutation_reason: str
    confidence_at_refute: float | None


def _compute_max_achievable_with_variance(
    actual_so_far: int,
    prior_count: int,
    remaining: int,
    z_score: float = 1.645,
) -> int:
    """Compute max achievable delta accounting for per-case distribution variance.

    Instead of assuming all remaining cases land on the predicted verdict (old
    optimistic bound), we use a one-sided upper confidence bound based on the
    prior verdict rate. This prevents false alarms when the prior distribution
    doesn't fully saturate a verdict class but remaining cases can still
    produce that verdict.

    Args:
        actual_so_far: Current count of the predicted verdict in iter N+1.
        prior_count: Count of the predicted verdict in iter N (baseline).
        remaining: Number of cases not yet evaluated.
        z_score: Number of standard deviations for the confidence interval.
            Defaults to 1.645 (one-sided 95% upper bound).

    Returns:
        Maximum achievable delta given distribution variance.
    """
    if remaining <= 0:
        return actual_so_far - prior_count

    # Estimate total cases from prior counts to derive prior rate.
    # We use a conservative estimate: assume at least prior_count cases existed.
    total_prior = max(prior_count, 1)
    prior_rate = prior_count / total_prior

    # Expected new verdicts from remaining cases based on prior rate.
    expected_new = remaining * prior_rate

    # Variance term for binomial: sqrt(n * p * (1 - p))
    # Using prior_rate as the probability estimate.
    variance = math.sqrt(max(0, remaining * prior_rate * (1 - prior_rate)))

    # One-sided upper confidence bound: expected + z * std_dev
    upper_bound = expected_new + z_score * variance

    return actual_so_far + int(math.floor(upper_bound)) - prior_count


def _compute_min_achievable_with_variance(
    actual_so_far: int,
    prior_count: int,
    remaining: int,
    z_score: float = 1.645,
) -> int:
    """Compute min achievable delta accounting for per-case distribution variance.

    For negative deltas, instead of assuming NO remaining cases produce the
    predicted verdict, we use a one-sided lower confidence bound based on the
    prior non-verdict rate. This accounts for cases that the prior didn't sample.

    Args:
        actual_so_far: Current count of the predicted verdict in iter N+1.
        prior_count: Count of the predicted verdict in iter N (baseline).
        remaining: Number of cases not yet evaluated.
        z_score: Number of standard deviations for the confidence interval.
            Defaults to 1.645 (one-sided 95% lower bound).

    Returns:
        Minimum achievable delta given distribution variance.
    """
    if remaining <= 0:
        return actual_so_far - prior_count

    # Estimate total cases from prior counts to derive prior rate.
    total_prior = max(prior_count, 1)
    prior_rate = prior_count / total_prior
    non_prior_rate = 1.0 - prior_rate

    # Expected remaining cases that will NOT produce the predicted verdict.
    expected_non_v = remaining * non_prior_rate

    # Variance for non-predicted verdict cases.
    variance = math.sqrt(max(0, remaining * non_prior_rate * prior_rate))

    # One-sided lower confidence bound: expected - z * std_dev
    lower_bound = expected_non_v - z_score * variance

    return actual_so_far - int(math.ceil(lower_bound)) - prior_count


def refute_live(
    prediction: Prediction,
    actuals_so_far: dict[str, int],
    remaining_cases: int,
    prior_iter_counts: dict[str, int],
) -> LiveRefutationStatus:
    """Decide whether ``prediction`` is still achievable given partial actuals.

    Args:
        prediction: The Prediction made about iter N+1 (from iter N).
        actuals_so_far: Verdict-count histogram of iter-N+1 cases completed so
            far. Keys are verdict letters (``"OK"``, ``"TLE"``, …); values are
            non-negative integer counts. Cases not yet dispatched are NOT
            included.
        remaining_cases: Number of iter-N+1 cases that have not yet completed.
            Non-negative.
        prior_iter_counts: Verdict-count histogram from iter N (the baseline
            against which the prediction's deltas are computed). Predictions
            are deltas (e.g. ``OK: +3``) relative to this baseline.

    Returns:
        ``LiveRefutationStatus``. ``is_refuted=True`` when at least one
        predicted delta cannot be achieved given the remaining headroom.

    Algorithm:
        For each ``(verdict, predicted_delta)`` in
        ``prediction.predicted_verdict_class_changes`` where the delta is
        non-zero:

        * ``actual_so_far`` = ``actuals_so_far.get(verdict, 0)``
        * ``prior`` = ``prior_iter_counts.get(verdict, 0)``
        * For positive deltas: max achievable is computed using a one-sided
          upper 95% confidence bound based on the prior verdict rate. This
          accounts for remaining-case distribution variance — not all remaining
          cases must land on the predicted verdict for the prediction to be
          achievable.
        * For negative deltas: min achievable is computed using a one-sided
          lower 95% confidence bound based on the prior non-verdict rate.

        Entries with ``predicted_delta == 0`` are ignored (they're not assertions).
    """
    pred_changes = prediction.predicted_verdict_class_changes or {}
    actuals = actuals_so_far or {}
    prior = prior_iter_counts or {}
    remaining = max(0, int(remaining_cases))

    refutation_reasons: list[str] = []

    for verdict, predicted_delta in pred_changes.items():
        try:
            pd = int(predicted_delta)
        except (TypeError, ValueError):
            continue
        if pd == 0:
            continue

        actual_so_far = int(actuals.get(verdict, 0))
        prior_count = int(prior.get(verdict, 0))

        if pd > 0:
            max_reachable_delta = _compute_max_achievable_with_variance(
                actual_so_far, prior_count, remaining
            )
            if pd > max_reachable_delta:
                refutation_reasons.append(
                    f"{verdict}: predicted +{pd} but max achievable is "
                    f"{max_reachable_delta:+d} (so_far={actual_so_far}, "
                    f"prior={prior_count}, remaining={remaining})"
                )
        else:  # pd < 0
            min_reachable_delta = _compute_min_achievable_with_variance(
                actual_so_far, prior_count, remaining
            )
            if pd < min_reachable_delta:
                refutation_reasons.append(
                    f"{verdict}: predicted {pd:+d} but min achievable is "
                    f"{min_reachable_delta:+d} (so_far={actual_so_far}, "
                    f"prior={prior_count}, remaining={remaining})"
                )

    is_refuted = bool(refutation_reasons)
    return LiveRefutationStatus(
        prediction=prediction,
        actuals_so_far=dict(actuals),
        remaining_cases=remaining,
        is_refuted=is_refuted,
        refutation_reason="; ".join(refutation_reasons),
        confidence_at_refute=float(prediction.confidence) if is_refuted else None,
    )


# --------------------------------------------------------------------------- #
# Feasibility-clip predictions (nervous-bus-8d1d)
#
# The LLM improver sometimes proposes verdict-count deltas that can't possibly
# happen — e.g. predicting "CE: -12" when iter N had only 1 CE case (you can't
# reduce CE by 12 when there's only 1 to remove). Cycle 01KRSDHKD7M0JQ44AKFY8PR7FN
# saw {OK:+8, CE:-12, WA:0} predicted with only 4 CE cases and 1 prior OK
# headroom — guaranteed-refute, polluting the prediction-outcome ledger.
#
# We already have the feasibility math in ``refute_live``; this is the same
# math applied at PREDICTION TIME so impossible deltas never get persisted as
# the improver's "stated bet."
# --------------------------------------------------------------------------- #


def clip_prediction_to_feasible(
    prediction: Prediction,
    prior_iter_counts: dict[str, int],
    num_cases: int,
) -> tuple[Prediction, list[str]]:
    """Clip a Prediction's verdict-count deltas to physically feasible ranges.

    Args:
        prediction: The Prediction as parsed from the LLM response.
        prior_iter_counts: Verdict-count histogram from iter N (the baseline
            against which the prediction's deltas are computed).
        num_cases: Total number of cases in iter N+1's benchmark. Used as the
            upper bound on any individual verdict count.

    Returns:
        ``(clipped_prediction, clip_reasons)``. The clipped Prediction has the
        same fields as the input except ``predicted_verdict_class_changes`` —
        each entry is bounded to ``[-prior, num_cases - prior]``. The
        ``clip_reasons`` list contains one human-readable string per delta
        that was actually changed; empty when the prediction was already
        feasible.

    Algorithm:
        For each ``(verdict, predicted_delta)``:
          * ``prior`` = ``prior_iter_counts.get(verdict, 0)``
          * Max positive delta: all ``num_cases`` could land on this verdict,
            so the ceiling is ``num_cases - prior``.
          * Max negative delta: zero cases could land on this verdict, so
            the floor is ``-prior``.
          * Clipped value = ``max(-prior, min(pd, num_cases - prior))``.

        ``predicted_score_delta`` and ``confidence`` are NOT clipped. The score
        delta is bounded by the [-1, +1] range of aggregate_score but is
        usually well within feasible bounds; if a future improver predicts a
        score delta outside [-1, +1] we can extend the clip then. Confidence
        is the improver's self-report and we don't editorialize it.
    """
    if not prediction or not prediction.predicted_verdict_class_changes:
        return prediction, []

    n = max(0, int(num_cases))
    prior = prior_iter_counts or {}
    clipped: dict[str, int] = {}
    reasons: list[str] = []

    for verdict, pd in prediction.predicted_verdict_class_changes.items():
        try:
            pd_int = int(pd)
        except (TypeError, ValueError):
            # Unparseable delta — drop it from the clipped prediction rather
            # than crash. The improver emitted garbage; we ignore it.
            reasons.append(f"{verdict}: dropped non-integer delta {pd!r}")
            continue

        prior_count = int(prior.get(verdict, 0))
        max_delta = n - prior_count
        min_delta = -prior_count

        if pd_int > max_delta:
            reasons.append(
                f"{verdict}: clipped +{pd_int} → +{max_delta} "
                f"(prior={prior_count}, num_cases={n}, max headroom={max_delta})"
            )
            clipped[verdict] = max_delta
        elif pd_int < min_delta:
            reasons.append(
                f"{verdict}: clipped {pd_int:+d} → {min_delta:+d} "
                f"(prior={prior_count}, min headroom={min_delta})"
            )
            clipped[verdict] = min_delta
        else:
            clipped[verdict] = pd_int

    clipped_prediction = Prediction(
        predicted_score_delta=prediction.predicted_score_delta,
        predicted_verdict_class_changes=clipped,
        confidence=prediction.confidence,
        rationale=prediction.rationale,
        active=prediction.active,
        parent_prediction_id=prediction.parent_prediction_id,
        fact_fingerprint=prediction.fact_fingerprint,
        prediction_id=prediction.prediction_id,
    )
    return clipped_prediction, reasons


# --------------------------------------------------------------------------- #
# Fact fingerprint + lifecycle management (Bitloops-style guidance tracking)
#
# The AHE paper (arXiv:2604.25850) turns RSI into a falsifiable contract by
# attaching machine-checkable predictions to every improver delta. This section
# implements Bitloops-style guidance fact lifecycle management: predictions are
# deduplicated by a SHA-256 fingerprint of their normalized content, superseded
# predictions are marked inactive, and a temporal scope key is used to
# invalidate prior predictions when a new iteration's prediction is distilled.
# --------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """Mirror Bitloops: split on whitespace, rejoin, lowercase, trim."""
    return " ".join(text.lower().split()).strip()


def prediction_fingerprint(pred: Prediction) -> str:
    """
    SHA-256 of: predicted_score_delta\n{rationale}

    Mirrors Bitloops guidance fact fingerprinting. The fingerprint is stable
    across functionally-identical predictions so that deduplication can find
    duplicates even when the LLM re-phrases the same idea.

    If the Prediction already has a non-None fact_fingerprint, returns it
    directly without re-computing.
    """
    if pred.fact_fingerprint is not None:
        return pred.fact_fingerprint
    key = (
        f"{normalize_text(str(pred.predicted_score_delta))}\n"
        f"{normalize_text(pred.rationale)}"
    )
    return hashlib.sha256(key.encode()).hexdigest()


def prediction_source_scope_key(
    session_id: str,
    problem_id: str,
    iteration: int,
) -> str:
    """Mirrors Bitloops: history_source:{session}:{problem_id}:{iteration}

    Used to scope prediction invalidation: when a new prediction is emitted
    for the same (session, problem, iteration) scope, all prior active
    predictions for that scope are deactivated.
    """
    return f"ahe:{session_id}:{problem_id}:{iteration}"


@dataclass
class PlannedCompaction:
    """Result of compacting a list of predictions by fingerprint.

    Attributes:
        retained: Prediction IDs that survived deduplication (one per fingerprint).
        superseded: Prediction IDs that were marked inactive as duplicates or
            superseded by a newer prediction for the same scope.
    """

    retained: list[str]
    superseded: list[str]


def compact_predictions(predictions: list[Prediction]) -> PlannedCompaction:
    """
    Deduplicate predictions by fingerprint, retaining the highest-confidence
    specimen per fingerprint group.

    Marked duplicates get ``active=False`` and ``lifecycle_status="duplicate"``.
    Marked superseded get ``active=False`` and ``lifecycle_status="superseded"``.
    The ``parent_prediction_id`` field is NOT set by this function — that is
    the caller's responsibility when the caller has explicit supersession context.
    Returns ``PlannedCompaction`` with retained vs superseded prediction IDs.
    """
    seen: dict[str, Prediction] = {}
    duplicates: list[str] = []

    for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
        fp = pred.fact_fingerprint or prediction_fingerprint(pred)
        if fp not in seen:
            seen[fp] = pred
        else:
            duplicates.append(pred.prediction_id)
            pred.active = False
            pred.lifecycle_status = "duplicate"

    return PlannedCompaction(
        retained=[seen[fp].prediction_id for fp in seen],
        superseded=duplicates,
    )


def invalidate_prior_predictions(
    repo_id: str,
    scope_key: str,
    store: dict[str, Prediction] | None = None,
) -> int:
    """
    Deactivate all active predictions for ``scope_key``.

    In the in-process version, ``store`` is a dict mapping prediction_id →
    Prediction (e.g. session_state.predictions). This function marks every
    active prediction whose ``source_scope_key`` matches as inactive and sets
    lifecycle_status to "superseded". Returns the count of predictions
    deactivated.

    In a distributed deployment, this function emits a
    ``autobench.improver.prediction.invalidated.v1`` event on the bus so the
    persistence layer can apply the invalidation asynchronously.
    """
    deactivated = 0
    for pred in (store or {}).values():
        source_scope = getattr(pred, "source_scope_key", None)
        if source_scope == scope_key and pred.active:
            pred.active = False
            pred.lifecycle_status = "superseded"
            deactivated += 1
    return deactivated


# --------------------------------------------------------------------------- #
# Bitloops-style source_scope_key invalidation registration
#
# Wire invalidate_prior_predictions into the shared InvalidationEngine so
# that re-distillation of an AHE scope key auto-deactivates prior predictions.
# --------------------------------------------------------------------------- #

def _register_ahe_invalidation() -> None:
    """Register AHE invalidation callback with the shared engine (called once)."""
    engine = get_invalidation_engine()
    # Register under the "ahe:" prefix — engine calls this when it detects
    # a re-distillation for any ahe:* scope key.
    engine.register(
        "ahe:",
        lambda sk: invalidate_prior_predictions(repo_id="", scope_key=sk, store={}),
        "autobench.improver.prediction.v1",
    )


_register_ahe_invalidation()
