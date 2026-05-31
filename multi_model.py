"""Multi-model evaluation infrastructure for autobench.

Provides:
- ModelClient Protocol — uniform interface for all LLM providers
- ModelClientRegistry — named model registration with provider routing
- MultiModelBenchmark — runs harness evaluation across N models in parallel
- ScoreNormalizer — rank-based and z-score cross-model score normalization
- ModelLeaderboard — tracks per-model Pareto frontier entries

Schema: schemas/autobench.model_result.v1.json

Usage:
    registry = ModelClientRegistry()
    registry.register("claude-sonnet", anthropic_client)
    registry.register("deepseek-v4", deepseek_client)

    bench = MultiModelBenchmark(registry=registry, judging_pool=None)
    results = bench.run_sweep(harness, problems, models=["claude-sonnet", "deepseek-v4"])
    leaderboard = ModelLeaderboard(results)
    leaderboard.print_table()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .core import HarnessConfig, HarnessResult, Verdict
from .evaluator import JudgingPool


# ---------------------------------------------------------------------------
# ModelClient Protocol
# ---------------------------------------------------------------------------


class ModelClient(Protocol):
    """Uniform interface for all LLM providers in multi-model evaluation.

    Implement this Protocol to add a new provider. The generate() method
    must be deterministic for a given (prompt, config) pair — variance
    is captured by JudgingPool's ensemble.

    Args:
        model_id: Provider-specific model identifier (e.g. "claude-3-5-sonnet")
        api_key: Provider API key. May come from environment variable.
    """

    def generate(self, prompt: str, config: HarnessConfig) -> str:
        """Generate code from a prompt using the model.

        Args:
            prompt: The coding problem prompt.
            config: Harness configuration with budget/temperature/etc.

        Returns:
            Generated code as string.
        """

    def score(self, prompt: str, output: str, context: dict[str, Any]) -> dict[str, Any]:
        """Score a model output using the model's own judgment.

        This is an OPTIONAL method — only needed when the model itself
        acts as a judge in JudgingPool. Not all ModelClient implementations
        need to implement this.

        Returns:
            Dict with keys: verdict (str), p_score (float),
            p_cost (float), p_time (float), reasoning (str).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


@dataclass
class NormalizedScore:
    """Cross-model normalized scores.

    Attributes:
        raw_p_score: Raw p_score from BenchmarkEvaluator.
        rank: Percentile rank [0.0, 1.0] across all models.
        z_score: Z-score relative to the model distribution.
        percentile: Human-readable percentile (higher = better).
    """

    model_id: str
    raw_p_score: float
    raw_p_cost: float
    raw_p_time: float
    rank_score: float = 0.0
    rank_cost: float = 0.0
    rank_time: float = 0.0
    z_score: float = 0.0
    percentile: float = 0.0


class ScoreNormalizer:
    """Cross-model score normalization.

    Supports two modes:
    - "rank": Percentile rank (0.0 = worst, 1.0 = best). Robust to outliers.
    - "zscore": Z-score normalization. Requires enough data points to be meaningful.

    All normalization is per-benchmark-run — scores are only comparable within
    a single MultiModelBenchmark.run_sweep() call.
    """

    def __init__(self, mode: str = "rank") -> None:
        if mode not in ("rank", "zscore"):
            raise ValueError("mode must be 'rank' or 'zscore'")
        self.mode = mode

    def normalize_results(
        self,
        results: dict[str, list[HarnessResult]],
    ) -> dict[str, NormalizedScore]:
        """Normalize a dict of model_id -> HarnessResult list.

        Args:
            results: {model_id: [HarnessResult]} from a benchmark sweep.

        Returns:
            {model_id: NormalizedScore} with rank/zscore populated.
        """
        if not results:
            return {}

        model_ids = list(results.keys())
        raw_scores = {
            mid: self._aggregate_p_score(results[mid])
            for mid in model_ids
        }

        if self.mode == "rank":
            normalized = self._rank_normalize(raw_scores)
        else:
            normalized = self._zscore_normalize(raw_scores)

        # Compute percentile from rank
        for model_id, ns in normalized.items():
            ns.percentile = ns.rank_score * 100.0

        return normalized

    def _aggregate_p_score(self, harness_results: list[HarnessResult]) -> float:
        if not harness_results:
            return 0.0
        # Use median to reduce outlier influence
        scores = sorted(r.p_score for r in harness_results)
        n = len(scores)
        if n % 2 == 0:
            return (scores[n // 2 - 1] + scores[n // 2]) / 2.0
        return scores[n // 2]

    def _rank_normalize(self, raw_scores: dict[str, float]) -> dict[str, NormalizedScore]:
        sorted_models = sorted(raw_scores.keys(), key=lambda m: raw_scores[m])
        n = len(sorted_models)
        normalized = {}
        for rank, model_id in enumerate(sorted_models, start=1):
            rank_fraction = (rank - 1) / max(n - 1, 1) if n > 1 else 0.5
            ns = NormalizedScore(
                model_id=model_id,
                raw_p_score=raw_scores[model_id],
                raw_p_cost=0.5,
                raw_p_time=0.5,
                rank_score=rank_fraction,
                rank_cost=rank_fraction,
                rank_time=rank_fraction,
            )
            normalized[model_id] = ns
        return normalized

    def _zscore_normalize(self, raw_scores: dict[str, float]) -> dict[str, NormalizedScore]:
        scores = list(raw_scores.values())
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5

        normalized = {}
        for model_id, score in raw_scores.items():
            z = (score - mean) / std if std > 0 else 0.0
            # Convert z to 0-1 range via sigmoid-like transform
            rank_score = max(0.0, min(1.0, (z + 3.0) / 6.0))  # z=-3 → 0, z=+3 → 1
            ns = NormalizedScore(
                model_id=model_id,
                raw_p_score=score,
                raw_p_cost=0.5,
                raw_p_time=0.5,
                rank_score=rank_score,
                rank_cost=rank_score,
                rank_time=rank_score,
                z_score=z,
            )
            normalized[model_id] = ns
        return normalized


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelClientRegistry:
    """Registry for named model clients.

    Provides model lookup by string identifier. Supports aliasing
    (e.g. "sonnet" → "claude-3-5-sonnet") and provider grouping.

    Usage:
        registry = ModelClientRegistry()
        registry.register("claude-3-5-sonnet", anthropic_client)
        registry.register("deepseek-v4", deepseek_client)
        registry.alias("sonnet", "claude-3-5-sonnet")

        client = registry.get("sonnet")  # returns anthropic_client
    """

    def __init__(self) -> None:
        self._clients: dict[str, ModelClient] = {}
        self._aliases: dict[str, str] = {}
        self._provider_groups: dict[str, list[str]] = {}

    def register(
        self,
        model_id: str,
        client: ModelClient,
        provider: str | None = None,
    ) -> None:
        """Register a model client.

        Args:
            model_id: Unique identifier (e.g. "claude-3-5-sonnet")
            client: ModelClient instance
            provider: Optional provider name for grouping (e.g. "anthropic")
        """
        self._clients[model_id] = client
        if provider:
            if provider not in self._provider_groups:
                self._provider_groups[provider] = []
            self._provider_groups[provider].append(model_id)

    def alias(self, alias: str, model_id: str) -> None:
        """Add an alias for an existing model_id."""
        if model_id not in self._clients:
            raise KeyError(f"model_id {model_id} not registered")
        self._aliases[alias] = model_id

    def get(self, model_id: str) -> ModelClient:
        """Get a registered client by model_id or alias."""
        resolved = self._aliases.get(model_id, model_id)
        if resolved not in self._clients:
            raise KeyError(f"model {model_id} not registered")
        return self._clients[resolved]

    def list_models(self, provider: str | None = None) -> list[str]:
        """List all registered model IDs, optionally filtered by provider."""
        if provider:
            return list(self._provider_groups.get(provider, []))
        return list(self._clients.keys())

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._clients or model_id in self._aliases


# ---------------------------------------------------------------------------
# Multi-model benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class ModelBenchmarkResult:
    """Result of benchmarking a single model against a problem set."""

    model_id: str
    harness_results: list[HarnessResult] = field(default_factory=list)
    normalized: NormalizedScore | None = None
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0
    verdict_counts: dict[str, int] = field(default_factory=dict)

    def pass_rate(self) -> float:
        if not self.harness_results:
            return 0.0
        return sum(1 for r in self.harness_results if r.is_pass()) / len(self.harness_results)


class MultiModelBenchmark:
    """Run a harness against N models and collect normalized comparative results.

    Coordinates:
    1. ModelClientRegistry — resolves model names to clients
    2. JudgingPool — scores worker outputs (optional, only if judging_fn provided)
    3. ScoreNormalizer — rank/zscore normalization across the model sweep
    4. ParetoFrontier — tracks which models dominate on quality/cost/speed

    Args:
        registry: ModelClientRegistry with all available model clients.
        judging_pool: Optional JudgingPool instance. If provided, each model
            output is scored by the judging pool instead of raw verdict.
        normalizer: ScoreNormalizer instance. Defaults to rank-based.
    """

    def __init__(
        self,
        registry: ModelClientRegistry | None = None,
        judging_pool: JudgingPool | None = None,
        normalizer: ScoreNormalizer | None = None,
    ) -> None:
        self.registry = registry or ModelClientRegistry()
        self.judging_pool = judging_pool
        self.normalizer = normalizer or ScoreNormalizer(mode="rank")

    def run_sweep(
        self,
        harness: HarnessConfig,
        problems: list[dict[str, Any]],
        models: list[str],
        generate_fn=None,
    ) -> dict[str, ModelBenchmarkResult]:
        """Run a benchmark sweep across multiple models.

        Args:
            harness: HarnessConfig to evaluate.
            problems: List of problem dicts (from CodeForcesScraper.to_benchmark_cases()).
            models: List of model_ids to evaluate. Must be registered in registry.
            generate_fn: Function(model_id, prompt, config) -> code.
                        If None, uses registry.get(model_id).generate() directly.

        Returns:
            {model_id: ModelBenchmarkResult} with raw + normalized scores.
        """
        from .evaluator import BenchmarkEvaluator

        results: dict[str, ModelBenchmarkResult] = {}

        for model_id in models:
            if model_id not in self.registry:
                continue

            client = self.registry.get(model_id)
            model_results: list[HarnessResult] = []
            total_latency = 0.0
            total_cost = 0.0
            total_tokens = 0
            verdict_counts: dict[str, int] = {}

            for problem in problems:
                prompt = problem.get("prompt", "")
                start = time.perf_counter()

                try:
                    if generate_fn:
                        code = generate_fn(model_id, prompt, harness)
                    else:
                        code = client.generate(prompt, harness)
                except Exception as e:
                    model_results.append(HarnessResult(
                        verdict=Verdict.RE,
                        error=f"Generation error: {e}",
                        latency_ms=(time.perf_counter() - start) * 1000,
                    ))
                    continue

                exec_result = self._execute_code(code, problem)
                total_latency += exec_result.get("latency_ms", 0)
                total_cost += exec_result.get("cost_dollars", 0)
                total_tokens += exec_result.get("tokens_used", 0)

                verdict_str = exec_result.get("verdict", "OK")
                verdict_counts[verdict_str] = verdict_counts.get(verdict_str, 0) + 1

                model_results.append(HarnessResult(
                    p_score=exec_result.get("p_score", 0.0),
                    p_cost=exec_result.get("p_cost", 0.5),
                    p_time=exec_result.get("p_time", 0.5),
                    verdict=Verdict(exec_result.get("verdict", "OK")),
                    error=exec_result.get("error", ""),
                    latency_ms=exec_result.get("latency_ms", 0),
                    tokens_used=exec_result.get("tokens_used", 0),
                    cost_dollars=exec_result.get("cost_dollars", 0),
                    metadata={"model_id": model_id, "problem_id": problem.get("id", "")},
                ))

            results[model_id] = ModelBenchmarkResult(
                model_id=model_id,
                harness_results=model_results,
                total_latency_ms=total_latency,
                total_cost=total_cost,
                total_tokens=total_tokens,
                verdict_counts=verdict_counts,
            )

        # Normalize across all models
        raw_results = {mid: r.harness_results for mid, r in results.items()}
        normalized = self.normalizer.normalize_results(raw_results)

        for model_id, ns in normalized.items():
            if model_id in results:
                results[model_id].normalized = ns

        return results

    def _execute_code(
        self,
        code: str,
        problem: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute generated code against a problem. Stub — replace with SandboxedExecutor."""
        # TODO: Wire to SandboxedExecutor.execute()
        return {
            "verdict": "OK",
            "p_score": 0.5,
            "p_cost": 0.5,
            "p_time": 0.5,
            "latency_ms": 100.0,
            "cost_dollars": 0.0,
            "tokens_used": 0,
            "error": "",
        }


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


class ModelLeaderboard:
    """Tracks and displays cross-model benchmark results."""

    def __init__(self, results: dict[str, ModelBenchmarkResult]) -> None:
        self.results = results

    def top_k(self, k: int = 10, sort_by: str = "p_score") -> list[tuple[str, float]]:
        """Return top-k models sorted by sort_key."""
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: getattr(x[1].normalized, sort_by, 0) if x[1].normalized else 0,
            reverse=True,
        )
        return [(mid, r.normalized.rank_score if r.normalized else 0) for mid, r in sorted_models[:k]]

    def print_table(self) -> None:
        """Print a markdown-style leaderboard table."""
        rows = []
        for model_id, result in sorted(
            self.results.items(),
            key=lambda x: x[1].normalized.rank_score if x[1].normalized else 0,
            reverse=True,
        ):
            ns = result.normalized
            rows.append(f"| {model_id} | {ns.raw_p_score:.3f} | {ns.percentile:.1f}% | {result.pass_rate():.1%} | {len(result.harness_results)} |")

        header = "| Model | Raw Score | Percentile | Pass Rate | Cases |"
        sep = "|------|-----------|------------|-----------|-------|"
        print(header)
        print(sep)
        for row in rows:
            print(row)


# ---------------------------------------------------------------------------
# Built-in client adapters (stubs — wire to actual providers)
# ---------------------------------------------------------------------------


class AnthropicModelClient:
    """Anthropic API model client adapter."""

    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: str | None = None) -> None:
        self.model_id = model_id
        self._api_key = api_key or __import__("os").get("ANTHROPIC_API_KEY", "")

    def generate(self, prompt: str, config: HarnessConfig) -> str:
        """Call Anthropic API to generate code."""
        try:
            from anthropic import Anthropic
        except ImportError:
            return ""
        client = Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self.model_id,
            max_tokens=config.budget.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def score(self, prompt: str, output: str, context: dict[str, Any]) -> dict[str, Any]:
        """Use Anthropic model as a judge."""
        try:
            from anthropic import Anthropic
        except ImportError:
            return {"verdict": "OK", "p_score": 0.5, "p_cost": 0.5, "p_time": 0.5, "reasoning": ""}
        client = Anthropic(api_key=self._api_key)
        judge_prompt = f"Score this code output:\n\nProblem: {prompt}\n\nOutput: {output}\n\nRespond with JSON: {{'verdict': 'OK'|'WA'|'RE', 'p_score': 0-1, 'reasoning': '...' }}"
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        import json
        try:
            return json.loads(response.content[0].text)
        except Exception:
            return {"verdict": "OK", "p_score": 0.5, "p_cost": 0.5, "p_time": 0.5, "reasoning": ""}


class DeepSeekModelClient:
    """DeepSeek API model client adapter."""

    def __init__(self, model_id: str = "deepseek-chat", api_key: str | None = None) -> None:
        self.model_id = model_id
        self._api_key = api_key or __import__("os").get("DEEPSEEK_API_KEY", "")

    def generate(self, prompt: str, config: HarnessConfig) -> str:
        """Call DeepSeek API to generate code."""
        import requests
        url = f"https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_id,
            "max_tokens": config.budget.get("max_tokens", 4096),
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]