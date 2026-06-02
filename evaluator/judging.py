"""Collective LLM-as-judge primitives and standalone scoring functions.

Split out of the former monolithic ``evaluator.py`` (behavior-preserving).
Holds JudgeVote, JudgingPool, and the standalone emit_verdict / score_harness
delegators. The two standalone functions reference BenchmarkEvaluator (in
``engine.py``) via call-time imports to keep the package import graph acyclic
(types <- judging <- engine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import HarnessResult, Verdict


def emit_verdict(
    code_output: str,
    stderr: str,
    runtime_ms: float,
    memory_mb: float,
    **kwargs,
) -> Verdict:
    """Standalone emit_verdict — delegates to BenchmarkEvaluator.emit_verdict."""
    from .engine import BenchmarkEvaluator

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
    from .engine import BenchmarkEvaluator

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
