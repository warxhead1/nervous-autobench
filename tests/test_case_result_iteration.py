"""Regression test for nervous-bus-4e0x.

case.result.v1 events must carry the iteration at which the case ACTUALLY
ran, not a stale snapshot from evaluator construction. Before the fix, the
emitter hard-coded ``iteration=0`` in evaluator._run_case so every case
was labeled 0 even when rsi_loop bumped to iter 1+.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from autobench.core import HarnessConfig
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator
from autobench.observability import AutobenchObservability, CHANNEL_CASE_RESULT


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    return tmp_path / "debug.jsonl"


def _case(cid: str) -> BenchmarkCase:
    return BenchmarkCase(
        id=cid, prompt="print hi", expected_output="hi\n", language="python",
    )


def _iter_dist(path: Path) -> Counter:
    events = [
        json.loads(line)
        for line in path.read_text().splitlines() if line.strip()
    ]
    return Counter(
        e["data"].get("iteration")
        for e in events if e.get("type") == CHANNEL_CASE_RESULT
    )


def test_case_result_iteration_reflects_current_iter(debug_file: Path) -> None:
    """Simulate a 2-iteration RSI loop reusing the same evaluator instance.

    Iter 0: 3 cases. Iter 1: 2 cases. case.result.v1 events must carry
    iteration=0 three times and iteration=1 twice — never all-zero.
    """
    obs = AutobenchObservability(debug_file=debug_file)
    ev = BenchmarkEvaluator(generate_fn=lambda p, c: "print('hi')\n", obs=obs)

    ev.run(HarnessConfig(), [_case("a"), _case("b"), _case("c")], iteration=0)
    ev.run(HarnessConfig(), [_case("d"), _case("e")], iteration=1)

    dist = _iter_dist(debug_file)
    assert dist == Counter({0: 3, 1: 2}), f"got {dist!r}"


def test_case_result_iteration_default_is_zero(debug_file: Path) -> None:
    """When no iteration kwarg is passed, events default to 0 (back-compat)."""
    obs = AutobenchObservability(debug_file=debug_file)
    ev = BenchmarkEvaluator(generate_fn=lambda p, c: "print('hi')\n", obs=obs)
    ev.run(HarnessConfig(), [_case("a"), _case("b")])

    dist = _iter_dist(debug_file)
    assert dist == Counter({0: 2}), f"got {dist!r}"
