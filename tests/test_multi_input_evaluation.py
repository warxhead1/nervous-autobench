"""Multi-input case evaluation (nervous-bus-uwjh).

Angela the Auditor finding: evaluator.py used to only exercise
``case.test_inputs[0]``. A solution that passed input[0] but failed
input[1] reported verdict=OK. These tests verify that the loop now runs
every (test_input, expected_output) pair and aggregates with worst-wins
precedence + fractional p_score for partial credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autobench.core import HarnessConfig, Verdict
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator
from autobench.engines.sandbox import ExecutionResult


@dataclass
class _StubExecutor:
    """Returns a queued sequence of ExecutionResults, one per execute() call."""

    queue: list[ExecutionResult]
    obs: Any = None
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls = []

    def execute(self, code: str, language: str, constraints: dict[str, Any],
                stdin: str, case_id: str) -> ExecutionResult:
        self.calls.append(stdin)
        if not self.queue:
            raise AssertionError(f"_StubExecutor ran out of canned results at stdin={stdin!r}")
        return self.queue.pop(0)


def _ok(stdout: str = "hi") -> ExecutionResult:
    return ExecutionResult(stdout=stdout, stderr="", exit_code=0, latency_ms=10.0)


def _wa() -> ExecutionResult:
    # exit 0 but stdout mismatches expected — emit_verdict reports WA.
    return ExecutionResult(stdout="wrong", stderr="", exit_code=0, latency_ms=10.0)


def _ce() -> ExecutionResult:
    return ExecutionResult(
        stdout="", stderr="SyntaxError: bad token", exit_code=1, latency_ms=5.0,
    )


def _eval(executor: _StubExecutor) -> BenchmarkEvaluator:
    return BenchmarkEvaluator(
        generate_fn=lambda prompt, cfg: "print('hi')",
        executor=executor,
    )


def test_all_three_inputs_ok() -> None:
    """3 inputs, all pass — verdict=OK, p_score=1.0, latency is the sum."""
    executor = _StubExecutor(queue=[_ok(), _ok(), _ok()])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="multi-ok",
        prompt="echo",
        expected_output="hi",
        test_inputs=["a", "b", "c"],
    )
    res = ev.run(HarnessConfig(), [case])

    r = res.case_results[0]
    assert r.verdict == Verdict.OK
    assert r.p_score == 1.0
    assert r.latency_ms == 30.0  # sum of three 10ms runs
    assert executor.calls == ["a", "b", "c"]
    per_input = r.metadata["per_input_results"]
    assert len(per_input) == 3
    assert all(p["verdict"] == "OK" for p in per_input)


def test_partial_credit_one_wa() -> None:
    """3 inputs, input[1] fails WA — verdict=WA, p_score≈0.667."""
    executor = _StubExecutor(queue=[_ok(), _wa(), _ok()])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="multi-wa",
        prompt="echo",
        expected_output="hi",
        test_inputs=["a", "b", "c"],
    )
    res = ev.run(HarnessConfig(), [case])

    r = res.case_results[0]
    assert r.verdict == Verdict.WA
    assert abs(r.p_score - (2 / 3)) < 1e-9
    assert executor.calls == ["a", "b", "c"]
    verdicts = [p["verdict"] for p in r.metadata["per_input_results"]]
    assert verdicts == ["OK", "WA", "OK"]


def test_ce_outranks_wa() -> None:
    """Mixed CE+WA inputs — aggregate verdict is CE (CE > WA precedence)."""
    executor = _StubExecutor(queue=[_ce(), _wa(), _ok()])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="multi-ce",
        prompt="echo",
        expected_output="hi",
        test_inputs=["a", "b", "c"],
    )
    res = ev.run(HarnessConfig(), [case])

    r = res.case_results[0]
    assert r.verdict == Verdict.CE
    # Only the third input was OK → 1/3.
    assert abs(r.p_score - (1 / 3)) < 1e-9


def test_empty_test_inputs_single_shot() -> None:
    """Empty test_inputs runs sandbox once with empty stdin — legacy behavior."""
    executor = _StubExecutor(queue=[_ok()])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="empty-inputs",
        prompt="echo",
        expected_output="hi",
        test_inputs=[],
    )
    res = ev.run(HarnessConfig(), [case])

    r = res.case_results[0]
    assert r.verdict == Verdict.OK
    assert r.p_score == 1.0
    assert executor.calls == [""]
    assert len(r.metadata["per_input_results"]) == 1
