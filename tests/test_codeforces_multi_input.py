"""Verify codeforces tier-1 benchmark exercises the multi-input path (shgb).

uwjh shipped multi-input evaluation in `evaluator.py`, but every case in
`cases.jsonl` had exactly one test_input — so the new code path never ran
on the live benchmark. This test pins the floor: at least 5 cases must
have >=3 inputs, and the schema (single shared `expected_output`) must
remain consistent across all of them.

See `tests/test_multi_input_evaluation.py` for the stub-level coverage of
the worst-wins aggregation and `per_input_results` payload.
"""

from __future__ import annotations

import json
from pathlib import Path

CASES_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "codeforces_tier1"
    / "cases.jsonl"
)


def _load_cases() -> list[dict]:
    cases: list[dict] = []
    with CASES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def test_cases_jsonl_parses() -> None:
    """Every line in cases.jsonl is a JSON object with the expected fields."""
    cases = _load_cases()
    assert len(cases) >= 20, f"expected >=20 cases, got {len(cases)}"
    for c in cases:
        assert "id" in c and isinstance(c["id"], str)
        assert "test_inputs" in c and isinstance(c["test_inputs"], list)
        assert "expected_output" in c and isinstance(c["expected_output"], str)
        assert c["test_inputs"], f"{c['id']}: test_inputs must be non-empty"


def test_at_least_five_cases_have_multi_input() -> None:
    """uwjh's multi-input loop is exercised by >=5 cases (nervous-bus-shgb)."""
    cases = _load_cases()
    multi = [c for c in cases if len(c["test_inputs"]) >= 3]
    multi_ids = sorted(c["id"] for c in multi)
    assert len(multi) >= 5, (
        f"expected >=5 multi-input cases (uwjh path), got {len(multi)}: {multi_ids}"
    )


def test_multi_input_cases_share_one_expected_output() -> None:
    """Schema constraint: ALL test_inputs in a case share ONE expected_output.

    The evaluator pairs `(test_inputs[i], expected_output)` for every i (see
    `BenchmarkEvaluator._run_case` in `evaluator.py`). Until the schema gains
    a parallel `expected_outputs` list, every edge input MUST produce the
    same canonical output as the sample. This test guards against an
    accidental schema drift that silently breaks the contract.
    """
    cases = _load_cases()
    for c in cases:
        if len(c["test_inputs"]) < 2:
            continue
        # The case must have a non-empty expected_output that every input
        # is expected to produce. Empty expected_output combined with
        # multi-input is meaningless under the current schema.
        assert c["expected_output"], (
            f"{c['id']}: multi-input case has empty expected_output"
        )


def test_expanded_cases_include_known_edge_problems() -> None:
    """The shgb expansion targeted 5 problems with single-line-style outputs
    that are robust across many inputs. Pin the set so a regression that
    drops one of them is caught immediately.
    """
    cases = _load_cases()
    multi_ids = {c["id"] for c in cases if len(c["test_inputs"]) >= 3}
    expected = {"cf-4A", "cf-1A", "cf-6A", "cf-7A", "cf-8B"}
    missing = expected - multi_ids
    assert not missing, f"expected multi-input cases missing: {missing}"
