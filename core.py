"""Core harness types for autobench.

Classes:
    HarnessConfig — holds system_prompt, rollout_protocol, context_manager, tool_surface, verifiers, budget
    HarnessResult — p_score, p_cost, p_time, verdict (CE/RE/TLE/MLE/WA/OK)
    RSILoop — recursive self-improvement loop: g(C) = perf_metric(improver_agent(harness_v1))
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class Verdict(str, Enum):
    """Verdict types for code execution results.

    Standard outcomes:
        CE  — Compilation Error
        RE  — Runtime Error
        TLE — Time Limit Exceeded
        MLE — Memory Limit Exceeded
        WA  — Wrong Answer (binary fail)
        OK  — Correct / Accepted
        VF  — Visual Fidelity below OK threshold but above WA floor
              (continuous-valued partial credit for shader/image benchmarks
              where SSIM ∈ [0.80, 0.95) ⇒ VF, SSIM ≥ 0.95 ⇒ OK, SSIM < 0.80 ⇒ WA;
              p_score reports the SSIM directly when verdict is VF.)

    Refactor-family outcomes (autobench/refactor_verifier.py):
        RV  — Refactor Verified: symbol rename (or other refactor) passes
              AST-scope check, AST-drift check, and full test suite.
        RD  — Refactor Drift: tests may pass but AST diff shows changes
              beyond the declared refactor (e.g., default arg changed during
              a rename). Semantic equivalence is not preserved.
        RT  — Refactor Test Fail: refactor compiles but the project test
              suite regresses (e.g., a partial rename missed a call site).
    """

    CE = "CE"  # Compilation Error
    RE = "RE"  # Runtime Error
    TLE = "TLE"  # Time Limit Exceeded
    MLE = "MLE"  # Memory Limit Exceeded
    WA = "WA"  # Wrong Answer
    OK = "OK"  # Correct / Accepted
    VF = "VF"  # Visual Fidelity (continuous, below OK threshold, above WA floor)
    RV = "RV"  # Refactor Verified
    RD = "RD"  # Refactor Drift (AST changes beyond declared refactor)
    RT = "RT"  # Refactor Test Fail (test suite regressed)


class ContextManager(str, Enum):
    """Context management strategies."""

    FULL = "full"  # Full context window
    BUDGETED = "budgeted"  # Budget-limited context
    SEMANTIC = "semantic"  # Semantic chunking / retrieval
    HIERARCHICAL = "hierarchical"  # Hierarchical summary + drill-down


class RolloutProtocol(str, Enum):
    """Rollout / generation protocols."""

    SINGLE = "single"  # Single shot
    ITERATIVE = "iterative"  # Iterative refinement
    SELF_REVISION = "self_revision"  # Self-revision with feedback
    MONTE_CARLO = "monte_carlo"  # Monte Carlo tree search / rollouts


@dataclass
class Verifier:
    """A verifier callable that checks code output against expected results."""

    name: str
    check: Callable[[Any, Any, dict[str, Any]], bool]
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    """Configuration for a coding agent harness.

    Attributes:
        system_prompt: The system prompt / instructions for the agent.
        rollout_protocol: How the agent generates code (single/iterative/self_revision/monte_carlo).
        context_manager: Context management strategy.
        tool_surface: Description of available tools/functions.
        verifiers: List of Verifier callables for output validation.
        budget: Dict with max_tokens, max_time, max_cost budget limits.
    """

    system_prompt: str = ""
    rollout_protocol: RolloutProtocol = RolloutProtocol.SINGLE
    context_manager: ContextManager = ContextManager.FULL
    tool_surface: str = ""
    verifiers: list[Verifier] = field(default_factory=list)
    budget: dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 8192,
            "max_time_seconds": 30,
            # nervous-bus-dq7l: 0 = $ guard disabled. MiniMax coding plan
            # bills by requests-per-5h, not dollars. Tracked separately by
            # RateBudgetGuard; this field exists for back-compat only.
            "max_cost_dollars": 0,
            "max_memory_mb": 512,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "rollout_protocol": self.rollout_protocol.value,
            "context_manager": self.context_manager.value,
            "tool_surface": self.tool_surface,
            "verifiers": [v.name for v in self.verifiers],
            "budget": self.budget,
        }


@dataclass
class HarnessResult:
    """Result of running a harness against a test case.

    Attributes:
        p_score: Normalized score [0.0, 1.0].
        p_cost: Normalized cost [0.0, 1.0].
        p_time: Normalized time [0.0, 1.0].
        verdict: Execution verdict (CE/RE/TLE/MLE/WA/OK).
        error: Error message or traceback if any.
        latency_ms: Wall-clock latency in milliseconds.
        tokens_used: Number of tokens consumed.
        cost_dollars: Actual cost in dollars.
        metadata: Arbitrary extra data.
    """

    p_score: float = 0.0
    p_cost: float = 0.0
    p_time: float = 0.0
    verdict: Verdict = Verdict.OK
    error: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_dollars: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_pass(self) -> bool:
        return self.verdict == Verdict.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_score": self.p_score,
            "p_cost": self.p_cost,
            "p_time": self.p_time,
            "verdict": self.verdict.value,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost_dollars": self.cost_dollars,
            "metadata": self.metadata,
        }


@dataclass
class RSILoop:
    """Recursive Self-Improvement loop.

    Encapsulates the iteration: g(C) = perf_metric(improver_agent(harness_v1))

    Attributes:
        max_iterations: Maximum number of improvement iterations.
        improvement_threshold: Minimum delta to continue iterating.
        history: List of (harness_config, result) pairs from each iteration.
    """

    max_iterations: int = 10
    improvement_threshold: float = 0.01
    history: list[tuple[HarnessConfig, HarnessResult]] = field(default_factory=list)

    def iterate(
        self,
        harness: HarnessConfig,
        improver_fn: Callable[[HarnessConfig, list[HarnessResult]], HarnessConfig],
        benchmark_fn: Callable[[HarnessConfig], list[HarnessResult]],
    ) -> tuple[HarnessConfig, list[HarnessResult]]:
        """Run one improvement iteration.

        Args:
            harness: Current harness configuration.
            improver_fn: Function that takes (current_harness, benchmark_results)
                         and returns an improved HarnessConfig.
            benchmark_fn: Function that takes a harness config and returns results.

        Returns:
            Tuple of (improved_harness, results).
        """
        results = benchmark_fn(harness)
        improved = improver_fn(harness, results)
        self.history.append((harness, self._aggregate(results)))
        return improved, results

    def run(
        self,
        initial_harness: HarnessConfig,
        improver_fn: Callable[[HarnessConfig, list[HarnessResult]], HarnessConfig],
        benchmark_fn: Callable[[HarnessConfig], list[HarnessResult]],
    ) -> tuple[HarnessConfig, list[HarnessResult]]:
        """Run the full RSI loop to convergence or max_iterations.

        Args:
            initial_harness: Starting harness configuration.
            improver_fn: Function that improves a harness given benchmark results.
            benchmark_fn: Function that benchmarks a harness configuration.

        Returns:
            Tuple of (final_harness, final_results).
        """
        harness = initial_harness
        results: list[HarnessResult] = []

        for i in range(self.max_iterations):
            prev_score = self._latest_score()

            harness, results = self.iterate(harness, improver_fn, benchmark_fn)

            if self.convergence_check(self.history):
                break

            # Early exit if no improvement
            curr_score = self._latest_score()
            if prev_score is not None and abs(curr_score - prev_score) < self.improvement_threshold:
                break

        return harness, results

    def _latest_score(self) -> float | None:
        if not self.history:
            return None
        return self.history[-1][1].p_score

    def _aggregate(self, results: list[HarnessResult]) -> HarnessResult:
        """Aggregate a list of results into a single summary result."""
        if not results:
            return HarnessResult()

        # Weighted average of scores, cost, time
        avg_score = sum(r.p_score for r in results) / len(results)
        avg_cost = sum(r.p_cost for r in results) / len(results)
        avg_time = sum(r.p_time for r in results) / len(results)

        # Majority verdict
        verdict_counts: dict[Verdict, int] = {}
        for r in results:
            verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1
        majority_verdict = max(verdict_counts, key=verdict_counts.get)

        total_latency = sum(r.latency_ms for r in results)
        total_tokens = sum(r.tokens_used for r in results)
        total_cost = sum(r.cost_dollars for r in results)

        return HarnessResult(
            p_score=avg_score,
            p_cost=avg_cost,
            p_time=avg_time,
            verdict=majority_verdict,
            latency_ms=total_latency,
            tokens_used=total_tokens,
            cost_dollars=total_cost,
        )

    def convergence_check(self, iteration_history: list[tuple[HarnessConfig, HarnessResult]]) -> bool:
        """Check if improvement has plateaued.

        Returns True when the last 3 iterations show no meaningful improvement.
        """
        if len(iteration_history) < 3:
            return False

        recent = [h[1].p_score for h in iteration_history[-3:]]
        if all(abs(recent[i] - recent[i - 1]) < self.improvement_threshold for i in range(1, len(recent))):
            return True
        return False


# Convenience factory for building a default RSI loop
def make_rsi_loop(
    max_iterations: int = 10,
    improvement_threshold: float = 0.01,
) -> RSILoop:
    return RSILoop(
        max_iterations=max_iterations,
        improvement_threshold=improvement_threshold,
    )
