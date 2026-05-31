"""Tests for the collective LLM-as-judge JudgingPool."""

from __future__ import annotations

import statistics
from autobench.core import Verdict
from autobench.evaluator import JudgingPool, JudgeVote


def _stub_judge_factory(prompt: str, context: dict) -> dict:
    """Stub judge that returns deterministic scores based on context."""
    output = context.get("worker_output", "")
    # Simulate: higher word count → higher score, ties at 0.5
    words = len(output.split())
    p_score = min(1.0, words / 100.0)
    verdict = "OK" if p_score > 0.3 else "WA"
    return {
        "judge_id": "stub-a",
        "verdict": verdict,
        "p_score": p_score,
        "p_cost": 0.5,
        "p_time": 0.7,
        "reasoning": f"judged {words} words",
    }


def _stub_judge_b(prompt: str, context: dict) -> dict:
    """Second stub judge with different scoring logic."""
    output = context.get("worker_output", "")
    chars = len(output)
    p_score = min(1.0, chars / 500.0)
    verdict = "OK" if p_score > 0.2 else "WA"
    return {
        "judge_id": "stub-b",
        "verdict": verdict,
        "p_score": p_score,
        "p_cost": 0.6,
        "p_time": 0.5,
        "reasoning": f"judged {chars} chars",
    }


def test_judging_pool_rejects_empty():
    """JudgingPool raises if no judges provided."""
    try:
        JudgingPool(judges=[])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "at least one judge" in str(e)


def test_judging_pool_single_judge():
    """Single judge returns its own verdict and scores."""
    pool = JudgingPool(judges=[_stub_judge_factory], ensemble_size=1)
    verdict, p_score, p_cost, p_time, votes = pool.evaluate(
        prompt="score this",
        context={"worker_output": "hello world test output here word count now"},  # 9 words → p_score=0.09 still below threshold
    )
    assert verdict == Verdict.WA  # stub-a: 9/100=0.09 < 0.3 → WA
    assert p_score < 0.3
    assert len(votes) == 1
    assert votes[0].judge_id == "stub-a"


def test_judging_pool_majority_aggregates():
    """Two judges with conflicting verdicts — majority wins."""
    pool = JudgingPool(judges=[_stub_judge_factory, _stub_judge_b], ensemble_size=2)
    # stub-a gives WA (low words), stub-b gives WA (low chars) for empty string
    verdict, p_score, p_cost, p_time, votes = pool.evaluate(
        prompt="score",
        context={"worker_output": ""},
    )
    assert verdict == Verdict.WA  # unanimous
    assert len(votes) == 2


def test_judging_pool_aggregates_continuous_scores():
    """Continuous scores (p_cost, p_time) are averaged, not majority-voted."""
    pool = JudgingPool(judges=[_stub_judge_factory, _stub_judge_b], ensemble_size=2)
    verdict, p_score, p_cost, p_time, votes = pool.evaluate(
        prompt="score",
        context={"worker_output": "a" * 200},  # enough for both to give OK
    )
    # stub-a: p_cost=0.5, stub-b: p_cost=0.6 → average = 0.55
    assert abs(p_cost - 0.55) < 0.01
    # stub-a: p_time=0.7, stub-b: p_time=0.5 → average = 0.6
    assert abs(p_time - 0.6) < 0.01


def test_judging_pool_calibrate_variance():
    """calibrate_ensemble_size returns variance dict."""
    pool = JudgingPool(judges=[_stub_judge_factory, _stub_judge_b])

    calibration_cases = [
        {"prompt": "score", "context": {"worker_output": "short"}},
        {"prompt": "score", "context": {"worker_output": "medium length output here"}},
        {"prompt": "score", "context": {"worker_output": "much longer output with more content to evaluate properly"}},
    ]

    results = pool.calibrate_ensemble_size(calibration_cases, _stub_judge_factory)
    assert 1 in results
    assert 3 in results
    # Variance should be non-negative
    for size, var in results.items():
        assert var >= 0.0


def test_judge_vote_dataclass():
    """JudgeVote holds all fields correctly."""
    vote = JudgeVote(
        judge_id="test",
        verdict=Verdict.OK,
        p_score=0.9,
        p_cost=0.1,
        p_time=0.8,
        reasoning="looks good",
    )
    assert vote.judge_id == "test"
    assert vote.verdict == Verdict.OK
    assert vote.p_score == 0.9
    assert vote.p_cost == 0.1
    assert vote.p_time == 0.8
    assert vote.reasoning == "looks good"


def test_judging_pool_handles_judge_exception():
    """If a judge raises, JudgingPool continues and returns partial votes."""
    def _failing_judge(prompt: str, context: dict) -> dict:
        raise RuntimeError("judge error")

    pool = JudgingPool(judges=[_failing_judge, _stub_judge_factory], ensemble_size=2)
    verdict, p_score, p_cost, p_time, votes = pool.evaluate(
        prompt="score",
        context={"worker_output": "test output"},
    )
    # Should have 1 successful vote from stub_judge_factory
    assert len(votes) == 1
    assert votes[0].judge_id == "stub-a"


def test_judging_pool_unknown_verdict_defaults_to_ok():
    """Judge returns unknown verdict → defaults to OK."""

    def _weird_judge(prompt: str, context: dict) -> dict:
        return {
            "judge_id": "weird",
            "verdict": "NOT_A_REAL_VERDICT",
            "p_score": 0.5,
            "p_cost": 0.5,
            "p_time": 0.5,
            "reasoning": "",
        }

    pool = JudgingPool(judges=[_weird_judge], ensemble_size=1)
    verdict, _, _, _, votes = pool.evaluate(prompt="", context={})
    assert verdict == Verdict.OK
    assert votes[0].verdict == Verdict.OK