"""Evaluation types, constants, and helpers for autobench.

Split out of the former monolithic ``evaluator.py`` (behavior-preserving).
Holds:
    - module-level constants (DEFAULT_*, _VERDICT_PRECEDENCE, DEFAULT_WEIGHTS)
    - dataclasses BenchmarkCase, BenchmarkResult
    - helpers _build_revision_context, _worst_verdict, _find_last_usage,
      _normalize_p_cost
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import HarnessConfig, HarnessResult, RolloutProtocol, Verdict


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
from ..observability import GENERATED_CODE_TRUNCATE_LEN, AutobenchObservability
from ..engines.sandbox import ExecutionResult, SandboxedExecutor, compile_and_run, verify_output


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
