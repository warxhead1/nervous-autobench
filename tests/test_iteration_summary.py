"""Tests for the autobench.iteration.summary.v1 rollup event (nervous-bus-91u).

Verifies:

* ``build_iteration_summary`` produces a dict with the expected keys and
  values for known inputs (verdict rates, totals, cost/token rollup).
* Each emitted event validates against
  ``schemas/autobench.iteration.summary.v1.json``.
* When ``SelfImprovingHarness.improve`` runs N iterations with obs attached,
  exactly N ``autobench.iteration.summary.v1`` events are emitted with
  monotonically increasing iteration ids.
* ``ce_rate`` and ``ok_rate`` are correct for mixed-verdict input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.core import HarnessConfig, HarnessResult, Verdict
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator, BenchmarkResult
from autobench.iteration_summary import build_iteration_summary
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_ITERATION_SUMMARY,
)
from autobench.rsi_loop import ImprovementDelta, SelfImprovingHarness


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "autobench.iteration.summary.v1.json"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force pipe-disabled mode so emissions land in a per-test debug file."""
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_result(verdicts: list[Verdict], aggregate_score: float = 0.5) -> BenchmarkResult:
    """Build a BenchmarkResult with one HarnessResult per verdict entry."""
    case_results = [
        HarnessResult(
            p_score=1.0 if v == Verdict.OK else 0.0,
            verdict=v,
            latency_ms=50.0,
            metadata={"case_id": f"c{i}"},
        )
        for i, v in enumerate(verdicts)
    ]
    verdict_counts: dict[str, int] = {}
    for r in case_results:
        verdict_counts[r.verdict.value] = verdict_counts.get(r.verdict.value, 0) + 1
    return BenchmarkResult(
        case_results=case_results,
        aggregate_score=aggregate_score,
        total_latency_ms=sum(r.latency_ms for r in case_results),
        verdict_counts=verdict_counts,
    )


# --------------------------------------------------------------------------- #
# Pure builder tests
# --------------------------------------------------------------------------- #


def test_build_iteration_summary_basic_keys_and_values() -> None:
    result = _make_result(
        [Verdict.OK, Verdict.OK, Verdict.WA, Verdict.CE],
        aggregate_score=0.625,
    )
    summary = build_iteration_summary(
        iteration=3,
        harness=HarnessConfig(),
        result=result,
        worker_call_costs=[0.01, 0.02, 0.005, 0.015],
        worker_call_tokens=[100, 250, 50, 175],
        harness_version="v3",
    )

    expected_keys = {
        "iteration",
        "aggregate_score",
        "pass_rate",
        "total_latency_ms",
        "total_cost_usd",
        "total_tokens",
        "verdict_distribution",
        "num_cases",
        "harness_version",
        "ce_rate",
        "ok_rate",
    }
    assert set(summary.keys()) == expected_keys

    assert summary["iteration"] == 3
    assert summary["aggregate_score"] == pytest.approx(0.625)
    assert summary["num_cases"] == 4
    assert summary["pass_rate"] == pytest.approx(0.5)
    assert summary["ok_rate"] == pytest.approx(0.5)
    assert summary["ce_rate"] == pytest.approx(0.25)
    assert summary["total_latency_ms"] == pytest.approx(200.0)
    assert summary["total_cost_usd"] == pytest.approx(0.05)
    assert summary["total_tokens"] == 575
    assert summary["verdict_distribution"] == {"OK": 2, "WA": 1, "CE": 1}
    assert summary["harness_version"] == "v3"


def test_build_iteration_summary_empty_cost_lists_yield_zero() -> None:
    result = _make_result([Verdict.OK])
    summary = build_iteration_summary(
        iteration=0,
        harness=HarnessConfig(),
        result=result,
        worker_call_costs=None,
        worker_call_tokens=None,
    )
    assert summary["total_cost_usd"] == 0.0
    assert summary["total_tokens"] == 0
    assert summary["harness_version"] == "v0"  # default tag from iteration


def test_build_iteration_summary_empty_case_list_safe() -> None:
    result = BenchmarkResult(case_results=[], aggregate_score=0.0)
    summary = build_iteration_summary(
        iteration=7,
        harness=HarnessConfig(),
        result=result,
    )
    assert summary["num_cases"] == 0
    assert summary["pass_rate"] == 0.0
    assert summary["ok_rate"] == 0.0
    assert summary["ce_rate"] == 0.0


def test_build_iteration_summary_mixed_verdicts_rate_correctness() -> None:
    # 1 OK, 1 WA, 1 CE, 1 TLE, 1 RE → ok_rate=0.2, ce_rate=0.2, pass_rate=0.2
    result = _make_result(
        [Verdict.OK, Verdict.WA, Verdict.CE, Verdict.TLE, Verdict.RE]
    )
    summary = build_iteration_summary(
        iteration=1, harness=HarnessConfig(), result=result,
    )
    assert summary["ok_rate"] == pytest.approx(0.2)
    assert summary["ce_rate"] == pytest.approx(0.2)
    assert summary["pass_rate"] == pytest.approx(0.2)
    assert summary["verdict_distribution"] == {
        "OK": 1, "WA": 1, "CE": 1, "TLE": 1, "RE": 1,
    }


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def test_emitted_event_validates_against_schema(debug_file: Path) -> None:
    pytest.importorskip("jsonschema")
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    obs = AutobenchObservability(debug_file=debug_file)
    result = _make_result([Verdict.OK, Verdict.WA, Verdict.CE], aggregate_score=0.33)
    summary = build_iteration_summary(
        iteration=2,
        harness=HarnessConfig(),
        result=result,
        worker_call_costs=[0.01],
        worker_call_tokens=[120],
        harness_version="v2",
    )
    obs.iteration_summary(**summary)

    events = _events_on(debug_file, CHANNEL_ITERATION_SUMMARY)
    assert len(events) == 1
    validator.validate(events[0])  # raises on failure


# --------------------------------------------------------------------------- #
# Wiring tests (SelfImprovingHarness emits one summary per iteration)
# --------------------------------------------------------------------------- #


class _StubEvaluator:
    """Stand-in for BenchmarkEvaluator that returns a canned result."""

    def __init__(self, results: list[BenchmarkResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run(self, harness, cases, obs=None, iteration: int = 0) -> BenchmarkResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _noop_improver(harness, result, iteration: int = 0):
    return harness, ImprovementDelta(improvement_summary=f"noop@{iteration}")


def test_rsi_loop_emits_one_summary_per_iteration(debug_file: Path) -> None:
    # Two iterations with different aggregate scores so convergence won't
    # short-circuit the second one (delta >= threshold).
    iter0 = _make_result([Verdict.OK, Verdict.WA], aggregate_score=0.4)
    iter1 = _make_result([Verdict.OK, Verdict.OK], aggregate_score=0.9)
    stub_eval = _StubEvaluator([iter0, iter1])

    obs = AutobenchObservability(debug_file=debug_file)
    harness = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=stub_eval,  # type: ignore[arg-type]
        max_iterations=2,
        improvement_threshold=0.01,
        default_improver=None,
        obs=obs,
    )

    cases = [BenchmarkCase(id="c0", prompt="x"), BenchmarkCase(id="c1", prompt="y")]
    harness.improve(cases, improver_fn=_noop_improver)

    events = _events_on(debug_file, CHANNEL_ITERATION_SUMMARY)
    assert len(events) == 2, f"expected 2 iteration.summary events, got {len(events)}"

    iters = [e["data"]["iteration"] for e in events]
    assert iters == [0, 1]

    # First event mirrors iter0; second mirrors iter1.
    assert events[0]["data"]["aggregate_score"] == pytest.approx(0.4)
    assert events[0]["data"]["num_cases"] == 2
    assert events[1]["data"]["aggregate_score"] == pytest.approx(0.9)
    assert events[1]["data"]["ok_rate"] == pytest.approx(1.0)
    # Session id is stable across the run.
    assert events[0]["data"]["session_id"] == obs.session_id
    assert events[1]["data"]["session_id"] == obs.session_id


def test_rsi_loop_summary_absent_when_obs_none(debug_file: Path) -> None:
    """Without obs, no debug-file output at all."""
    iter0 = _make_result([Verdict.OK], aggregate_score=0.5)
    stub_eval = _StubEvaluator([iter0])

    harness = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=stub_eval,  # type: ignore[arg-type]
        max_iterations=1,
        default_improver=None,
        obs=None,
    )
    harness.improve([BenchmarkCase(id="c0", prompt="x")], improver_fn=_noop_improver)
    assert not debug_file.exists()


# --------------------------------------------------------------------------- #
# Worker-usage plumbing (nervous-bus-b3uz) — _last_usage must propagate
# through evaluator -> rsi_loop -> iteration.summary, including when the
# worker is wrapped in a closure (run_first.py shape) or a functools.partial.
# --------------------------------------------------------------------------- #


class _StubWorker:
    """Minimal MiniMaxWorker stand-in. Exposes ``_last_usage`` per call."""

    def __init__(self, usage: dict) -> None:
        self._last_usage = dict(usage)
        self.calls = 0

    def __call__(self, prompt: str, cfg: HarnessConfig) -> str:
        self.calls += 1
        return "print('hi')\n"


def _iteration_summary_for_generate_fn(
    generate_fn,
    debug_file: Path,
    n_cases: int = 2,
) -> dict:
    """Run a single RSI iteration with the supplied generate_fn and return
    the first iteration.summary event's ``data`` dict."""
    obs = AutobenchObservability(debug_file=debug_file)
    evaluator = BenchmarkEvaluator(generate_fn=generate_fn, obs=obs)

    harness = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=evaluator,
        max_iterations=1,
        default_improver=None,
        obs=obs,
    )
    cases = [
        BenchmarkCase(
            id=f"c{i}", prompt="print hi", expected_output="hi\n", language="python"
        )
        for i in range(n_cases)
    ]
    harness.improve(cases, improver_fn=_noop_improver)

    events = _events_on(debug_file, CHANNEL_ITERATION_SUMMARY)
    assert len(events) == 1, f"expected 1 summary event, got {len(events)}"
    return events[0]["data"]


def test_iteration_summary_reads_last_usage_from_direct_worker(debug_file: Path) -> None:
    """Worker instance passed directly as generate_fn — flat getattr works."""
    worker = _StubWorker({"prompt_tokens": 80, "completion_tokens": 43, "cost_usd": 0.045})

    data = _iteration_summary_for_generate_fn(worker, debug_file, n_cases=1)

    assert data["total_tokens"] == 123, data
    assert data["total_cost_usd"] == pytest.approx(0.045)


def test_iteration_summary_reads_last_usage_through_closure(debug_file: Path) -> None:
    """Worker wrapped in a closure (run_first.py shape) — flat getattr fails,
    closure-walk must locate ``_last_usage`` on the captured worker."""
    worker = _StubWorker({"prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.02})

    def _worker_callable(prompt: str, cfg: HarnessConfig) -> str:
        return worker(prompt, cfg)

    # Sanity-check that the bug shape exists: the closure itself does not
    # expose _last_usage, only the captured worker instance does.
    assert getattr(_worker_callable, "_last_usage", None) is None

    data = _iteration_summary_for_generate_fn(_worker_callable, debug_file, n_cases=2)

    # Two cases × 150 tokens/call = 300 ; two cases × $0.02 = $0.04
    assert data["total_tokens"] == 300, data
    assert data["total_cost_usd"] == pytest.approx(0.04)


def test_iteration_summary_reads_last_usage_shorthand_keys(debug_file: Path) -> None:
    """Worker setting ``_last_usage={'tokens':100,'cost':0.0042}`` — the shorthand
    shape (`cost` not `cost_usd`, `tokens` not split into prompt/completion) must
    still propagate via the normaliser. nervous-bus-ch4o regression guard.
    """
    worker = _StubWorker({"tokens": 100, "cost": 0.0042})

    data = _iteration_summary_for_generate_fn(worker, debug_file, n_cases=1)

    assert data["total_tokens"] == 100, data
    assert data["total_cost_usd"] == pytest.approx(0.0042)


def test_iteration_summary_reads_last_usage_anthropic_shape(debug_file: Path) -> None:
    """Anthropic ``input_tokens``/``output_tokens`` keys are normalised to
    ``prompt_tokens``+``completion_tokens`` equivalence in the rollup.
    """
    worker = _StubWorker(
        {"input_tokens": 60, "output_tokens": 40, "cost_usd": 0.0099}
    )

    data = _iteration_summary_for_generate_fn(worker, debug_file, n_cases=1)

    assert data["total_tokens"] == 100, data
    assert data["total_cost_usd"] == pytest.approx(0.0099)


def test_normalize_worker_usage_unit() -> None:
    """Direct unit test of the normaliser — accept all known key variants."""
    from autobench.iteration_summary import normalize_worker_usage

    # Canonical (current MiniMaxWorker) shape
    n = normalize_worker_usage(
        {"prompt_tokens": 80, "completion_tokens": 43, "cost_usd": 0.045}
    )
    assert n == {"cost_usd": 0.045, "tokens": 123.0}

    # Shorthand `cost`/`tokens` (bead AC reference shape)
    n = normalize_worker_usage({"tokens": 100, "cost": 0.05})
    assert n == {"cost_usd": 0.05, "tokens": 100.0}

    # Anthropic input/output tokens
    n = normalize_worker_usage(
        {"input_tokens": 10, "output_tokens": 5, "total_cost_usd": 0.001}
    )
    assert n == {"cost_usd": 0.001, "tokens": 15.0}

    # Missing everything — never raises, zeros out.
    assert normalize_worker_usage({}) == {"cost_usd": 0.0, "tokens": 0.0}


def test_iteration_summary_reads_last_usage_through_partial(debug_file: Path) -> None:
    """Worker wrapped in functools.partial — unwrap via ``.func``."""
    from functools import partial

    worker = _StubWorker({"prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.001})
    bound = partial(worker)  # partial wrapping the callable instance

    data = _iteration_summary_for_generate_fn(bound, debug_file, n_cases=1)

    assert data["total_tokens"] == 15, data
    assert data["total_cost_usd"] == pytest.approx(0.001)
