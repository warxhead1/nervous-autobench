"""Per-input expected outputs (nervous-bus-tqhd).

Before tqhd: ``BenchmarkCase`` carried a single ``expected_output`` that
the evaluator broadcast to every entry in ``test_inputs``. That forced
case authors to pick edge inputs where the canonical output happened to
match (e.g. all return 'YES'). After tqhd: an optional
``expected_outputs`` list lets a case carry distinct expected outputs
per input, unlocking asymmetric edge tests.

These tests exercise the three branches of the pairing logic:
1. Per-input list with matching length — pair (input[i], expected[i]).
2. Per-input list with mismatched length — silently fall back to the
   singular ``expected_output``, no crash.
3. Empty per-input list — singular broadcast (pre-tqhd behavior).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autobench.core import HarnessConfig, Verdict
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator
from autobench.engines.sandbox import ExecutionResult


@dataclass
class _StubExecutor:
    """Returns canned stdout strings as ExecutionResults, one per execute()."""

    stdouts: list[str]
    obs: Any = None
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls = []

    def execute(self, code: str, language: str, constraints: dict[str, Any],
                stdin: str, case_id: str) -> ExecutionResult:
        self.calls.append(stdin)
        if not self.stdouts:
            raise AssertionError(f"stub ran out at stdin={stdin!r}")
        return ExecutionResult(
            stdout=self.stdouts.pop(0),
            stderr="",
            exit_code=0,
            latency_ms=5.0,
        )


def _eval(executor: _StubExecutor) -> BenchmarkEvaluator:
    return BenchmarkEvaluator(
        generate_fn=lambda prompt, cfg: "print('hi')",
        executor=executor,
    )


def test_per_input_expected_outputs_match() -> None:
    """3 inputs, 3 distinct expected outputs — each input compared to its own."""
    # Stub returns the "correct" stdout for each input. If pairing were wrong
    # (e.g. broadcast singular), expected_output="42" would WA every input
    # whose stdout != "42".
    executor = _StubExecutor(stdouts=["1", "4", "9"])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="asymmetric-ok",
        prompt="square the input",
        expected_output="WILL_NOT_MATCH",  # singular is wrong on purpose
        expected_outputs=["1", "4", "9"],
        test_inputs=["1", "2", "3"],
    )
    res = ev.run(HarnessConfig(), [case])

    r = res.case_results[0]
    assert r.verdict == Verdict.OK, f"expected OK, got {r.verdict}"
    assert r.p_score == 1.0
    assert executor.calls == ["1", "2", "3"]
    per_input = r.metadata["per_input_results"]
    assert [p["verdict"] for p in per_input] == ["OK", "OK", "OK"]


def test_per_input_expected_outputs_wa_on_one() -> None:
    """Per-input pairing — middle input WAs against its OWN expected."""
    # Stub returns "1", "WRONG", "9" — middle input mismatches expected_outputs[1]="4".
    # Under broadcast (legacy), expected_output="1" would have made input[0] OK
    # but input[2]="9" also WA. The point is that the verdict is computed
    # element-wise against expected_outputs[i], not the singular.
    executor = _StubExecutor(stdouts=["1", "WRONG", "9"])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="asymmetric-wa",
        prompt="square the input",
        expected_output="UNUSED",
        expected_outputs=["1", "4", "9"],
        test_inputs=["1", "2", "3"],
    )
    res = ev.run(HarnessConfig(), [case])

    r = res.case_results[0]
    assert r.verdict == Verdict.WA
    assert abs(r.p_score - (2 / 3)) < 1e-9
    verdicts = [p["verdict"] for p in r.metadata["per_input_results"]]
    assert verdicts == ["OK", "WA", "OK"]


def test_mismatched_length_falls_back_to_singular() -> None:
    """expected_outputs len != test_inputs len — fall back to singular, no crash."""
    # 3 inputs, only 2 expected_outputs — must NOT pair (would IndexError).
    # Singular expected_output="hi" applies to every input.
    executor = _StubExecutor(stdouts=["hi", "hi", "hi"])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="len-mismatch",
        prompt="echo",
        expected_output="hi",
        expected_outputs=["only", "two"],  # 2 != 3
        test_inputs=["a", "b", "c"],
    )
    res = ev.run(HarnessConfig(), [case])  # must not raise

    r = res.case_results[0]
    assert r.verdict == Verdict.OK
    assert r.p_score == 1.0
    assert executor.calls == ["a", "b", "c"]


def test_empty_expected_outputs_falls_back_to_singular() -> None:
    """Empty expected_outputs — singular broadcast (pre-tqhd behavior)."""
    executor = _StubExecutor(stdouts=["YES", "YES", "YES"])
    ev = _eval(executor)
    case = BenchmarkCase(
        id="empty-list",
        prompt="watermelon",
        expected_output="YES",
        expected_outputs=[],  # explicit empty
        test_inputs=["8", "4", "100"],
    )
    res = ev.run(HarnessConfig(), [case])

    r = res.case_results[0]
    assert r.verdict == Verdict.OK
    assert r.p_score == 1.0


def test_to_dict_includes_expected_outputs() -> None:
    """BenchmarkCase.to_dict() must round-trip the new field."""
    case = BenchmarkCase(
        id="rt",
        prompt="p",
        expected_output="x",
        expected_outputs=["a", "b"],
        test_inputs=["1", "2"],
    )
    d = case.to_dict()
    assert d["expected_outputs"] == ["a", "b"]
    assert d["expected_output"] == "x"
