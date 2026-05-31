"""Iteration-summary rollup builder for the RSI loop.

Pure helper that turns a ``BenchmarkResult`` (plus per-iteration worker-call
cost/token data) into the data block published on
``autobench.iteration.summary.v1``.

Pulled into its own module so the rollup logic is unit-testable without
spinning up the full evaluator + observability stack. See
``nervous-bus-91u`` and the schema at
``schemas/autobench.iteration.summary.v1.json``.
"""

from __future__ import annotations

from typing import Any

from .core import HarnessConfig
from .evaluator import BenchmarkResult


def normalize_worker_usage(usage: dict[str, Any]) -> dict[str, float]:
    """Normalise a worker ``_last_usage`` dict to canonical ``cost_usd`` / ``tokens``.

    MiniMaxWorker today writes ``cost_usd`` / ``prompt_tokens`` / ``completion_tokens``
    (see ``worker_agent._update_last_usage``). But the worker's two response shapes
    (OpenAI vs Anthropic at ``/anthropic/v1/messages``) differ upstream, and any
    future producer wrapping the worker could plausibly emit ``cost``,
    ``total_cost``, ``tokens``, or Anthropic's ``input_tokens`` / ``output_tokens``
    keys. We accept the union, prefer the most explicit form, and never raise.

    Returns a dict with two float keys: ``cost_usd`` and ``tokens``.
    """
    cost = (
        usage.get("cost_usd")
        if usage.get("cost_usd") is not None
        else usage.get("total_cost_usd")
        if usage.get("total_cost_usd") is not None
        else usage.get("cost")
        if usage.get("cost") is not None
        else usage.get("total_cost", 0.0)
    )

    if usage.get("tokens") is not None:
        tokens = usage.get("tokens")
    else:
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        tokens = int(prompt) + int(completion)

    return {"cost_usd": float(cost or 0.0), "tokens": float(tokens or 0)}


def build_iteration_summary(
    iteration: int,
    harness: HarnessConfig,
    result: BenchmarkResult,
    worker_call_costs: list[float] | None = None,
    worker_call_tokens: list[int] | None = None,
    harness_version: str | None = None,
) -> dict[str, Any]:
    """Build the data block for ``autobench.iteration.summary.v1``.

    Args:
        iteration: RSI iteration number this summary describes.
        harness: The harness config the iteration ran against. Currently only
            used as a hint for ``harness_version`` when the caller doesn't
            pass one explicitly. ``HarnessConfig`` itself has no version
            field today; we default to ``"v{iteration}"`` to match the
            naming convention used by ``SelfImprovingHarness.improve``.
        result: ``BenchmarkResult`` from ``BenchmarkEvaluator.run``. We read
            ``aggregate_score``, ``total_latency_ms``, ``verdict_counts``,
            ``case_results`` and derive the rates from them.
        worker_call_costs: Per-call USD cost for every worker call made this
            iteration. Sum becomes ``total_cost_usd``. ``None`` or ``[]``
            yields 0.0 — see TODO in ``rsi_loop.improve`` re: full plumbing.
        worker_call_tokens: Per-call total token count (prompt+completion)
            for every worker call this iteration. Sum becomes
            ``total_tokens``. ``None`` or ``[]`` yields 0.
        harness_version: Optional explicit harness version tag. Defaults to
            ``f"v{iteration}"`` to mirror the RSI loop's tagging.

    Returns:
        A plain ``dict`` matching the ``data`` block of the schema.
    """
    case_results = list(result.case_results or [])
    num_cases = len(case_results)
    verdict_counts: dict[str, int] = {
        str(k): int(v) for k, v in (result.verdict_counts or {}).items()
    }

    # ce_rate / ok_rate are direct fractions over total cases. Division by
    # zero is avoided by checking num_cases up front — both rates fall back
    # to 0.0 on an empty case list (consistent with BenchmarkResult.pass_rate).
    if num_cases > 0:
        ok_count = verdict_counts.get("OK", 0)
        ce_count = verdict_counts.get("CE", 0)
        ok_rate = ok_count / num_cases
        ce_rate = ce_count / num_cases
        pass_rate = float(result.pass_rate())
    else:
        ok_rate = 0.0
        ce_rate = 0.0
        pass_rate = 0.0

    total_cost_usd = float(sum(worker_call_costs or []))
    total_tokens = int(sum(worker_call_tokens or []))

    version_tag = harness_version if harness_version is not None else f"v{int(iteration)}"

    return {
        "iteration": int(iteration),
        "aggregate_score": float(result.aggregate_score),
        "pass_rate": pass_rate,
        "total_latency_ms": float(result.total_latency_ms),
        "total_cost_usd": total_cost_usd,
        "total_tokens": total_tokens,
        "verdict_distribution": verdict_counts,
        "num_cases": num_cases,
        "harness_version": str(version_tag),
        "ce_rate": ce_rate,
        "ok_rate": ok_rate,
    }
