"""Shared structural failure-pattern detector for autobench.

Surfaces clusters of non-OK cases whose generated code shares a common
leading prefix (after whitespace normalisation). The original motivating
example: every CE'd case in a wave begins with ``<think>`` prose because
the worker model leaked its reasoning into the code field.

This module is pure data — it does NOT emit on the bus. The RSI loop pulls
patterns out and forwards them through ``AutobenchObservability.failure_pattern``.
Keeping detection pure also makes the algorithm trivially testable without
plumbing the obs layer.

Algorithm:
    1. Group ``case_results`` by ``verdict`` (excluding OK).
    2. Within each non-OK class, take the first ``prefix_len`` chars of
       ``metadata["generated_code"]`` after whitespace normalisation
       (strip leading whitespace, replace newlines with ``|``).
    3. Count occurrences of each prefix. Emit a ``FailurePattern`` for any
       prefix occurring at least ``threshold`` times.
    4. Cap output at ``max_prefixes_per_verdict`` per verdict class
       (highest sample_count wins; tie-break by prefix string for
       deterministic ordering).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class FailurePattern:
    """One detected shared-prefix cluster of failing cases.

    Attributes:
        verdict: The verdict string (e.g. ``"CE"``, ``"RE"``, ``"WA"``).
        prefix: The normalised leading ``prefix_len_chars`` of generated code.
        sample_count: Number of cases in this prefix bucket.
        total_in_class: Number of cases in this verdict class total
            (i.e. how many non-OK cases share the same verdict).
        sample_case_ids: Up to 5 case_ids from the bucket, in encounter
            order, so a human can spot-check the cluster.
    """

    verdict: str
    prefix: str
    sample_count: int
    total_in_class: int
    sample_case_ids: list[str] = field(default_factory=list)


def _normalise_prefix(code: str, prefix_len: int) -> str:
    """Strip leading whitespace and collapse newlines to ``|`` for the prefix.

    The pipe character is chosen because it never appears as the first char
    of valid Python/C++/Go source, so a prefix containing ``|`` is a clear
    signal that the generated code had embedded line breaks early.
    """
    if not code:
        return ""
    stripped = code.lstrip()
    collapsed = stripped.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "|")
    return collapsed[:prefix_len]


def _case_id(result: Any) -> str:
    """Best-effort case_id extraction from a HarnessResult-shaped object."""
    md = getattr(result, "metadata", None) or {}
    cid = md.get("case_id") if isinstance(md, dict) else None
    return str(cid) if cid is not None else ""


def _generated_code(result: Any) -> str:
    """Best-effort generated_code extraction from a HarnessResult-shaped object."""
    md = getattr(result, "metadata", None) or {}
    if not isinstance(md, dict):
        return ""
    code = md.get("generated_code")
    return code if isinstance(code, str) else ""


def _verdict_str(result: Any) -> str:
    """Return the verdict as a plain string, regardless of enum/str shape."""
    v = getattr(result, "verdict", None)
    if v is None:
        return ""
    # autobench.core.Verdict is a (str, Enum); .value works, and str() also does.
    return getattr(v, "value", None) or str(v)


def detect_failure_patterns(
    case_results: Iterable[Any],
    prefix_len: int = 20,
    threshold: int = 3,
    max_prefixes_per_verdict: int = 3,
) -> list[FailurePattern]:
    """Detect shared-prefix clusters among non-OK cases.

    Args:
        case_results: Iterable of ``HarnessResult``-shaped objects. Each must
            expose ``verdict`` (str or enum with ``.value``) and a
            ``metadata`` dict carrying ``case_id`` and ``generated_code``.
        prefix_len: Number of characters from the (normalised) start of
            generated code to bucket by.
        threshold: Minimum bucket size to surface as a ``FailurePattern``.
            Buckets smaller than ``threshold`` are silently dropped.
        max_prefixes_per_verdict: Cap on patterns emitted per verdict class.
            When a verdict has more qualifying buckets than this cap, the
            buckets with highest ``sample_count`` win (deterministic
            tie-break by prefix string).

    Returns:
        List of ``FailurePattern`` instances, ordered by (verdict ASC,
        sample_count DESC, prefix ASC). Empty when nothing crosses the
        threshold.

    Notes:
        OK verdicts are excluded — we only care about failure clusters.
        Cases with empty ``generated_code`` are still bucketed (empty
        prefix); this surfaces "produced no code" as its own pattern,
        which is itself a useful signal.
    """
    if prefix_len <= 0 or threshold <= 0 or max_prefixes_per_verdict <= 0:
        return []

    # Group case_id lists by (verdict, prefix); also track per-verdict totals.
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    per_verdict_total: Counter[str] = Counter()

    for r in case_results:
        verdict = _verdict_str(r)
        if not verdict or verdict == "OK":
            continue
        per_verdict_total[verdict] += 1
        prefix = _normalise_prefix(_generated_code(r), prefix_len)
        buckets[(verdict, prefix)].append(_case_id(r))

    # Promote buckets that hit the threshold, then cap per verdict.
    per_verdict_candidates: dict[str, list[FailurePattern]] = defaultdict(list)
    for (verdict, prefix), case_ids in buckets.items():
        if len(case_ids) < threshold:
            continue
        per_verdict_candidates[verdict].append(
            FailurePattern(
                verdict=verdict,
                prefix=prefix,
                sample_count=len(case_ids),
                total_in_class=per_verdict_total[verdict],
                sample_case_ids=list(case_ids[:5]),
            )
        )

    out: list[FailurePattern] = []
    for verdict in sorted(per_verdict_candidates):
        candidates = per_verdict_candidates[verdict]
        # Highest sample_count first; deterministic tie-break by prefix.
        candidates.sort(key=lambda p: (-p.sample_count, p.prefix))
        out.extend(candidates[:max_prefixes_per_verdict])

    return out
