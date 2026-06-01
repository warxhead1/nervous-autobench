"""Tests for the multi-advocate RSI spine (nervous-bus-6yut).

Verifies:
    * ``PopulationRunner(n_advocates=1)`` produces single-advocate output
      indistinguishable from the legacy single-lineage code path (no
      summary event emitted, one advocate in the result).
    * ``PopulationRunner(n_advocates=2)`` emits two distinct session_ids on
      the bus.
    * Winner = highest best_score, ties broken by lowest advocate index.
    * One advocate's revert (sf0y) does not affect another advocate's harness.
    * The summary event lands on the bus with all expected fields and
      validates against schemas/autobench.population.summary.v1.json.
    * ``AUTOBENCH_ADVOCATES`` env var parsing.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.core import HarnessConfig, HarnessResult, Verdict
from autobench.evaluator import BenchmarkResult
from autobench.observability import (
    CHANNEL_ITERATION,
    CHANNEL_POPULATION_SUMMARY,
    AutobenchObservability,
)
from autobench.population import (
    AdvocateResult,
    PopulationResult,
    PopulationRunner,
    _read_n_advocates_env,
)
from autobench.rsi_loop import ImprovementDelta




# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #


def _dummy_case_result(verdict: Verdict = Verdict.OK) -> HarnessResult:
    return HarnessResult(p_score=1.0 if verdict == Verdict.OK else 0.0, verdict=verdict)


class _ScriptedEvaluator:
    """Returns scripted aggregate_score per call; advances index per ``run``."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.run_count = 0
        self.harnesses_seen: list[HarnessConfig] = []

    def run(
        self,
        harness: HarnessConfig,
        cases: list[Any],
        obs: Any = None,
    ) -> BenchmarkResult:
        self.harnesses_seen.append(harness)
        idx = min(self.run_count, len(self._scores) - 1)
        self.run_count += 1
        return BenchmarkResult(
            case_results=[_dummy_case_result()],
            aggregate_score=self._scores[idx],
            total_latency_ms=0.0,
            verdict_counts={"OK": 1},
            metadata={},
        )


def _make_marker_improver(marker_per_iter: dict[int, str]):
    """Build an improver that appends a per-iteration marker to system_prompt."""

    def _improver(
        harness: HarnessConfig,
        result: BenchmarkResult,
        iteration: int = 0,
        revert_history: list[Any] | None = None,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        marker = marker_per_iter.get(iteration, "")
        new = replace(
            harness,
            system_prompt=(harness.system_prompt or "") + f"|i{iteration}:{marker}",
        )
        delta = ImprovementDelta(
            improvement_summary=f"iter{iteration} edit",
            system_prompt_delta=f"i{iteration}:{marker}",
        )
        return new, delta

    return _improver


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force debug-file emission (no zellij in PATH)."""
    path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Env parsing
# --------------------------------------------------------------------------- #


def test_read_n_advocates_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOBENCH_ADVOCATES", raising=False)
    assert _read_n_advocates_env(default=3) == 3
    assert _read_n_advocates_env(default=1) == 1


def test_read_n_advocates_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_ADVOCATES", "5")
    assert _read_n_advocates_env() == 5


def test_read_n_advocates_clamps_min(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_ADVOCATES", "0")
    assert _read_n_advocates_env() == 1
    monkeypatch.setenv("AUTOBENCH_ADVOCATES", "-7")
    assert _read_n_advocates_env() == 1


def test_read_n_advocates_bad_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_ADVOCATES", "three")
    assert _read_n_advocates_env(default=3) == 3


# --------------------------------------------------------------------------- #
# n_advocates=1 — backwards-compat
# --------------------------------------------------------------------------- #


def test_n_equals_1_backwards_compat(debug_file: Path) -> None:
    """N=1 produces one advocate and emits NO population.summary event.

    The single-advocate path must be indistinguishable from the legacy
    SelfImprovingHarness.improve() call. The only checkable invariants:
    one advocate appears in the result; winner_id is that advocate; no
    population.summary event lands on the bus.
    """
    evaluator = _ScriptedEvaluator(scores=[0.30, 0.40])
    obs = AutobenchObservability(debug_file=debug_file)
    improver = _make_marker_improver({0: "A", 1: "B"})

    runner = PopulationRunner(
        n_advocates=1,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: evaluator,  # type: ignore[arg-type]
        observability_factory=lambda: obs,
        improver=None,  # use the supplied improver_fn instead
        max_iterations_per_advocate=2,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
    )

    # The runner doesn't accept a direct improver_fn — but default_improver=None
    # makes SelfImprovingHarness fall back to the in-module rule-based
    # improve_harness, which works against our ScriptedEvaluator's
    # single-OK-case result. That's fine for shape verification.
    result = runner.run(benchmark_cases=[])

    assert isinstance(result, PopulationResult)
    assert len(result.advocates) == 1
    assert result.winner_id == "advocate-0"
    assert result.advocates[0].session_id == obs.session_id

    # No summary event for N=1.
    events = _read_events(debug_file)
    summary_events = [e for e in events if e.get("type") == CHANNEL_POPULATION_SUMMARY]
    assert summary_events == [], "N=1 must not emit population.summary"


# --------------------------------------------------------------------------- #
# n_advocates=2 — produces distinct session_ids
# --------------------------------------------------------------------------- #


def test_n_equals_2_produces_two_sessions(debug_file: Path) -> None:
    """Two advocates → two distinct session_ids appear on the bus."""
    # Two separate obs/evaluator pairs so each advocate keeps its own bus signal.
    obs_a = AutobenchObservability(session_id="SESS-A-AAAAAAAAAAAAAAAAA", debug_file=debug_file)
    obs_b = AutobenchObservability(session_id="SESS-B-BBBBBBBBBBBBBBBBB", debug_file=debug_file)
    eval_a = _ScriptedEvaluator(scores=[0.30])
    eval_b = _ScriptedEvaluator(scores=[0.40])

    pending_obs = iter([obs_a, obs_b])
    pending_eval = iter([eval_a, eval_b])

    runner = PopulationRunner(
        n_advocates=2,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: next(pending_eval),  # type: ignore[arg-type]
        observability_factory=lambda: next(pending_obs),
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
    )

    result = runner.run(benchmark_cases=[])

    sessions = {a.session_id for a in result.advocates}
    assert sessions == {obs_a.session_id, obs_b.session_id}
    assert len(sessions) == 2

    # Each advocate's iteration events landed on its own session_id.
    iter_events = [
        e for e in _read_events(debug_file)
        if e.get("type") == CHANNEL_ITERATION
    ]
    sessions_seen = {e["data"]["session_id"] for e in iter_events}
    assert obs_a.session_id in sessions_seen
    assert obs_b.session_id in sessions_seen


# --------------------------------------------------------------------------- #
# Winner picking
# --------------------------------------------------------------------------- #


def test_winner_is_highest_score(debug_file: Path) -> None:
    """Highest best_score wins; ties broken by lowest advocate index."""
    # Three advocates scoring 0.30, 0.55, 0.40 → advocate-1 wins.
    scripts = [
        _ScriptedEvaluator(scores=[0.30]),
        _ScriptedEvaluator(scores=[0.55]),
        _ScriptedEvaluator(scores=[0.40]),
    ]
    obs_list = [
        AutobenchObservability(session_id=f"SESS-{i}" + "X" * 21, debug_file=debug_file)
        for i in range(3)
    ]
    pending_obs = iter(obs_list)
    pending_eval = iter(scripts)

    runner = PopulationRunner(
        n_advocates=3,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: next(pending_eval),  # type: ignore[arg-type]
        observability_factory=lambda: next(pending_obs),
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
    )
    result = runner.run(benchmark_cases=[])
    assert result.winner_id == "advocate-1"
    assert abs(result.winner_score - 0.55) < 1e-9


def test_winner_tie_break_by_lowest_index(debug_file: Path) -> None:
    """When two advocates tie, the lower-index advocate wins (deterministic)."""
    scripts = [
        _ScriptedEvaluator(scores=[0.50]),
        _ScriptedEvaluator(scores=[0.50]),
    ]
    obs_list = [
        AutobenchObservability(session_id=f"TIE-{i}" + "Y" * 22, debug_file=debug_file)
        for i in range(2)
    ]
    pending_obs = iter(obs_list)
    pending_eval = iter(scripts)

    runner = PopulationRunner(
        n_advocates=2,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: next(pending_eval),  # type: ignore[arg-type]
        observability_factory=lambda: next(pending_obs),
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
    )
    result = runner.run(benchmark_cases=[])
    assert result.winner_id == "advocate-0"


# --------------------------------------------------------------------------- #
# Isolation — revert on A does not affect B
# --------------------------------------------------------------------------- #


def test_isolation_advocate_a_revert_does_not_affect_b(debug_file: Path) -> None:
    """Advocate A regresses + reverts (sf0y); B's harness must be untouched."""
    # Advocate A: scores [0.70, 0.40] → regression >>variance_floor → revert.
    # Advocate B: scores [0.60, 0.65] → monotonic improvement, no revert.
    eval_a = _ScriptedEvaluator(scores=[0.70, 0.40, 0.50])
    eval_b = _ScriptedEvaluator(scores=[0.60, 0.65])
    obs_a = AutobenchObservability(session_id="ISO-A-" + "Z" * 21, debug_file=debug_file)
    obs_b = AutobenchObservability(session_id="ISO-B-" + "W" * 21, debug_file=debug_file)

    pending_obs = iter([obs_a, obs_b])
    pending_eval = iter([eval_a, eval_b])

    runner = PopulationRunner(
        n_advocates=2,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: next(pending_eval),  # type: ignore[arg-type]
        observability_factory=lambda: next(pending_obs),
        improver=None,
        max_iterations_per_advocate=2,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
    )
    result = runner.run(benchmark_cases=[])

    # Advocate A should have triggered a revert (best_score=0.70 from iter 0).
    assert result.advocates[0].best_score == pytest.approx(0.70)
    assert result.advocates[0].best_iter == 0

    # Advocate B should be unaffected: best_score=0.65 from iter 1.
    assert result.advocates[1].best_score == pytest.approx(0.65)
    assert result.advocates[1].best_iter == 1

    # B's evaluator only saw two harnesses, and both started from "BASE"
    # (no revert state spilled in from A's run).
    assert eval_b.harnesses_seen[0].system_prompt == "BASE"
    # B's second iteration evaluated a harness derived from BASE (improver-mutated).
    assert eval_b.harnesses_seen[1].system_prompt.startswith("BASE")


# --------------------------------------------------------------------------- #
# Summary event
# --------------------------------------------------------------------------- #


def test_population_summary_event_fires_and_validates(debug_file: Path) -> None:
    """N>=2 must emit one population.summary.v1 event with the expected shape."""
    scripts = [
        _ScriptedEvaluator(scores=[0.25]),
        _ScriptedEvaluator(scores=[0.65]),
    ]
    obs_list = [
        AutobenchObservability(session_id=f"SUMM-{i}" + "Q" * 21, debug_file=debug_file)
        for i in range(2)
    ]
    # The runner constructs its own summary obs by default (since
    # run_summary_obs=None) — we redirect that to our debug_file by
    # supplying one explicitly.
    summary_obs = AutobenchObservability(session_id="POP-SUM-XXXXXXXXXXXXXXXX", debug_file=debug_file)

    pending_obs = iter(obs_list)
    pending_eval = iter(scripts)

    runner = PopulationRunner(
        n_advocates=2,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: next(pending_eval),  # type: ignore[arg-type]
        observability_factory=lambda: next(pending_obs),
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
        run_summary_obs=summary_obs,
    )
    result = runner.run(benchmark_cases=[])

    # Find the summary event on the bus.
    events = _read_events(debug_file)
    summary_events = [e for e in events if e.get("type") == CHANNEL_POPULATION_SUMMARY]
    assert len(summary_events) == 1, f"expected 1 summary event, got {len(summary_events)}"
    ev = summary_events[0]

    # Envelope checks.
    for key in ("specversion", "id", "source", "type", "datacontenttype", "time", "data"):
        assert key in ev
    assert ev["specversion"] == "1.0"
    assert ev["type"] == CHANNEL_POPULATION_SUMMARY
    assert ev["datacontenttype"] == "application/json"

    data = ev["data"]
    assert data["cycle_id"] == result.cycle_id
    assert data["winner_id"] == "advocate-1"
    assert data["winner_score"] == pytest.approx(0.65)
    assert len(data["advocates"]) == 2
    advocate_ids = {a["advocate_id"] for a in data["advocates"]}
    assert advocate_ids == {"advocate-0", "advocate-1"}
    session_ids = {a["session_id"] for a in data["advocates"]}
    assert session_ids == {obs_list[0].session_id, obs_list[1].session_id}

    # Validate against the schema (Draft 2020-12).
    try:
        import jsonschema
    except ImportError:  # pragma: no cover — jsonschema is a runtime dep
        pytest.skip("jsonschema not installed")
    schema = json.loads((SCHEMA_DIR / "autobench.population.summary.v1.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(ev)


def test_iteration_event_accepts_advocate_id(debug_file: Path) -> None:
    """iteration.v1 events MAY carry an optional advocate_id field."""
    obs = AutobenchObservability(session_id="ADVTAG-XXXXXXXXXXXXXXXXXXXX", debug_file=debug_file)
    obs.iteration_start(iteration_num=0, harness_version="v0", advocate_id="advocate-2")
    obs.iteration_complete(
        iteration_num=0,
        aggregate_score=0.5,
        verdict_counts={"OK": 1},
        improvement_delta=None,
        harness_version="v0",
        advocate_id="advocate-2",
    )

    events = [
        e for e in _read_events(debug_file)
        if e.get("type") == CHANNEL_ITERATION
    ]
    assert len(events) == 2
    for e in events:
        assert e["data"]["advocate_id"] == "advocate-2"

    # Schema still validates (advocate_id is a documented optional field).
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")
    schema = json.loads((SCHEMA_DIR / "autobench.iteration.v1.json").read_text())
    for e in events:
        jsonschema.Draft202012Validator(schema).validate(e)


# --------------------------------------------------------------------------- #
# Failure isolation: one advocate raising must not kill the cycle
# --------------------------------------------------------------------------- #


class _RaisingEvaluator:
    def run(self, harness, cases, obs=None):  # noqa: D401
        raise RuntimeError("simulated worker explosion")


def test_failing_advocate_does_not_kill_cycle(debug_file: Path) -> None:
    """An advocate that raises mid-iteration must not poison its siblings."""
    eval_a = _RaisingEvaluator()
    eval_b = _ScriptedEvaluator(scores=[0.45])
    obs_a = AutobenchObservability(session_id="FAIL-A-" + "M" * 20, debug_file=debug_file)
    obs_b = AutobenchObservability(session_id="FAIL-B-" + "N" * 20, debug_file=debug_file)

    pending_obs = iter([obs_a, obs_b])
    pending_eval = iter([eval_a, eval_b])

    runner = PopulationRunner(
        n_advocates=2,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: next(pending_eval),  # type: ignore[arg-type]
        observability_factory=lambda: next(pending_obs),
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
    )
    result = runner.run(benchmark_cases=[])

    assert len(result.advocates) == 2
    # Advocate A failed: error field set, best_score=0 sentinel.
    assert result.advocates[0].error is not None
    assert "simulated worker" in result.advocates[0].error
    # Advocate B ran successfully.
    assert result.advocates[1].error is None
    assert result.advocates[1].best_score == pytest.approx(0.45)
    assert result.winner_id == "advocate-1"
