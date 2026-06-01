"""Tests for the counterfactual replay tool."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from autobench.core import (
    ContextManager,
    HarnessConfig,
    HarnessResult,
    RolloutProtocol,
    Verdict,
)
from autobench.evaluator import BenchmarkCase, BenchmarkResult
from autobench.replay import (
    CounterfactualRunner,
    ReplayComparison,
    ReplayLoader,
    filter_cases_by_id,
    harness_dict_to_config,
    load_cases_from_dir,
    merge_overrides,
    parse_override,
)


# --------------------------------------------------------------------------- #
# Helpers — synthetic event stream
# --------------------------------------------------------------------------- #

def _event(channel: str, data: dict[str, Any], eid: str = "01") -> dict[str, Any]:
    return {
        "specversion": "1.0",
        "id": eid,
        "source": "/autobench",
        "type": channel,
        "datacontenttype": "application/json",
        "time": "2026-05-16T00:00:00Z",
        "data": data,
    }


def _write_capture(path: Path, events: list[dict[str, Any]]) -> None:
    with open(path, "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _make_capture_2sessions_3iters_4channels(path: Path) -> None:
    """Synthetic capture: 2 sessions × 3 iterations × 4 channels."""
    events: list[dict[str, Any]] = []
    for sid in ("01SESS_A", "01SESS_B"):
        for it in range(3):
            events.append(_event("autobench.iteration.v1", {
                "session_id": sid,
                "iteration": it,
                "harness_version": f"v{it}",
                "status": "start",
            }))
            events.append(_event("autobench.phase.v1", {
                "session_id": sid,
                "phase": "benchmark",
                "status": "start",
                "extra": {"num_cases": 2},
            }))
            for cid in (f"case-{it}-A", f"case-{it}-B"):
                events.append(_event("autobench.sandbox.v1", {
                    "session_id": sid,
                    "case_id": cid,
                    "language": "python",
                    "sandbox_type": "subprocess",
                    "status": "dispatch",
                }))
                events.append(_event("autobench.sandbox.v1", {
                    "session_id": sid,
                    "case_id": cid,
                    "language": "python",
                    "sandbox_type": "subprocess",
                    "status": "complete",
                    "verdict": "OK",
                    "latency_ms": 12.5,
                    "exit_code": 0,
                }))
            events.append(_event("autobench.phase.v1", {
                "session_id": sid,
                "phase": "benchmark",
                "status": "complete",
                "duration_ms": 25.0,
                "extra": {},
            }))
            events.append(_event("autobench.improver.v1", {
                "session_id": sid,
                "model": "rule_based",
                "status": "start",
                "prompt_tokens": 0,
            }))
            events.append(_event("autobench.improver.v1", {
                "session_id": sid,
                "model": "rule_based",
                "status": "complete",
                "completion_tokens": 0,
                "delta_summary": f"iter {it} mutation",
            }))
            events.append(_event("autobench.iteration.v1", {
                "session_id": sid,
                "iteration": it,
                "harness_version": f"v{it}",
                "status": "complete",
                "aggregate_score": 0.8 + 0.05 * it,
                "verdict_counts": {"OK": 2},
                "improvement_delta": {
                    "system_prompt_delta": "",
                    "rollout_protocol_changed": False,
                    "context_manager_changed": False,
                    "tool_surface_delta": "",
                    "budget_delta": {"max_tokens": 4096 + it * 1024},
                    "improvement_summary": f"raised tokens (iter {it})",
                },
            }))
    _write_capture(path, events)


# --------------------------------------------------------------------------- #
# parse_override
# --------------------------------------------------------------------------- #

class TestParseOverride:

    def test_scalar(self):
        assert parse_override("context_manager=BUDGETED") == {"context_manager": "BUDGETED"}

    def test_nested(self):
        assert parse_override("budget.max_tokens=4096") == {"budget": {"max_tokens": 4096}}

    def test_deep_nest(self):
        assert parse_override("a.b.c=true") == {"a": {"b": {"c": True}}}

    def test_float(self):
        assert parse_override("budget.max_cost_dollars=0.25") == {"budget": {"max_cost_dollars": 0.25}}

    def test_string_fallback(self):
        assert parse_override("rollout_protocol=ITERATIVE") == {"rollout_protocol": "ITERATIVE"}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError):
            parse_override("not_an_override")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError):
            parse_override("=value")

    def test_merge(self):
        merged = merge_overrides([
            "budget.max_tokens=4096",
            "budget.max_time_seconds=60",
            "context_manager=BUDGETED",
        ])
        assert merged == {
            "budget": {"max_tokens": 4096, "max_time_seconds": 60},
            "context_manager": "BUDGETED",
        }


# --------------------------------------------------------------------------- #
# ReplayLoader
# --------------------------------------------------------------------------- #

class TestReplayLoader:

    def test_index_2_sessions_3_iterations_4_channels(self, tmp_path: Path):
        cap = tmp_path / "capture.jsonl"
        _make_capture_2sessions_3iters_4channels(cap)

        loader = ReplayLoader(cap)
        assert loader.sessions() == ["01SESS_A", "01SESS_B"]
        assert loader.iterations("01SESS_A") == [0, 1, 2]
        assert loader.iterations("01SESS_B") == [0, 1, 2]

        # Each iteration should have events from all 4 channels.
        for sid in loader.sessions():
            for it in loader.iterations(sid):
                evs = loader.events_for(sid, it)
                channels = {e["type"] for e in evs}
                assert channels == {
                    "autobench.iteration.v1",
                    "autobench.phase.v1",
                    "autobench.sandbox.v1",
                    "autobench.improver.v1",
                }

    def test_case_ids_and_verdicts(self, tmp_path: Path):
        cap = tmp_path / "cap.jsonl"
        _make_capture_2sessions_3iters_4channels(cap)
        loader = ReplayLoader(cap)
        assert loader.case_ids("01SESS_A", 1) == ["case-1-A", "case-1-B"]
        assert loader.original_verdicts("01SESS_A", 1) == {"case-1-A": "OK", "case-1-B": "OK"}
        assert loader.aggregate_score("01SESS_A", 1) == pytest.approx(0.85)

    def test_harness_at_replays_budget_deltas(self, tmp_path: Path):
        """harness_at at iter=2 should reflect budget_deltas from iters 0 and 1."""
        cap = tmp_path / "cap.jsonl"
        _make_capture_2sessions_3iters_4channels(cap)
        loader = ReplayLoader(cap)

        # At iter=0 (no completes before it), the base config should be untouched.
        h0 = loader.harness_at("01SESS_A", 0)
        assert h0["budget"]["max_tokens"] == 8192

        # At iter=2, the completes from iters 0 and 1 have applied.
        # iter 0: max_tokens = 4096
        # iter 1: max_tokens = 5120 (overwrites 4096)
        h2 = loader.harness_at("01SESS_A", 2)
        assert h2["budget"]["max_tokens"] == 5120
        assert h2["_unresolved_flips"] == []

    def test_harness_at_records_unresolved_flips(self, tmp_path: Path):
        """When an improver event only carries a *_changed flag, we tag it."""
        cap = tmp_path / "cap.jsonl"
        evs = [
            _event("autobench.iteration.v1", {
                "session_id": "S1",
                "iteration": 0,
                "harness_version": "v0",
                "status": "complete",
                "aggregate_score": 0.5,
                "verdict_counts": {"OK": 1},
                "improvement_delta": {
                    "system_prompt_delta": "tighter rubric",
                    "rollout_protocol_changed": True,
                    "context_manager_changed": False,
                    "tool_surface_delta": "",
                    "budget_delta": {},
                    "improvement_summary": "switched protocol",
                },
            }),
        ]
        _write_capture(cap, evs)
        loader = ReplayLoader(cap)
        h = loader.harness_at("S1", 1)
        assert "tighter rubric" in h["system_prompt"]
        assert any("rollout_protocol_changed" in s for s in h["_unresolved_flips"])

    def test_missing_file_is_empty(self, tmp_path: Path):
        loader = ReplayLoader(tmp_path / "does-not-exist.jsonl")
        assert loader.sessions() == []

    def test_ignores_garbage_lines(self, tmp_path: Path):
        cap = tmp_path / "cap.jsonl"
        with open(cap, "w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps(_event("autobench.iteration.v1", {
                "session_id": "S", "iteration": 0, "harness_version": "v0", "status": "start"
            })) + "\n")
            fh.write(json.dumps({"type": "some.other.channel", "data": {"session_id": "X"}}) + "\n")
        loader = ReplayLoader(cap)
        assert loader.sessions() == ["S"]


# --------------------------------------------------------------------------- #
# CounterfactualRunner — end-to-end
# --------------------------------------------------------------------------- #

class _MockEvaluator:
    """Hand-rolled BenchmarkEvaluator stand-in.

    Returns a deterministic BenchmarkResult based on a verdict map indexed by
    (context_manager, case_id). Lets us force a flip without spinning up a
    real sandbox.
    """

    def __init__(self, verdict_map: dict[tuple[str, str], Verdict]) -> None:
        self.verdict_map = verdict_map
        self.calls: list[HarnessConfig] = []

    def run(self, harness: HarnessConfig, cases: list[BenchmarkCase]) -> BenchmarkResult:
        self.calls.append(harness)
        case_results: list[HarnessResult] = []
        verdict_counts: dict[str, int] = {}
        ok_count = 0
        for case in cases:
            verdict = self.verdict_map.get(
                (harness.context_manager.value, case.id),
                Verdict.OK,
            )
            if verdict == Verdict.OK:
                ok_count += 1
            verdict_counts[verdict.value] = verdict_counts.get(verdict.value, 0) + 1
            case_results.append(HarnessResult(
                p_score=1.0 if verdict == Verdict.OK else 0.0,
                verdict=verdict,
                metadata={"case_id": case.id},
            ))
        agg = ok_count / len(cases) if cases else 0.0
        return BenchmarkResult(
            case_results=case_results,
            aggregate_score=agg,
            total_latency_ms=0.0,
            verdict_counts=verdict_counts,
        )


class TestCounterfactualRunner:

    def test_apply_override_context_manager(self):
        h = HarnessConfig()
        runner = CounterfactualRunner(evaluator=None)
        new = runner.apply_override(h, {"context_manager": "BUDGETED"})
        assert new.context_manager == ContextManager.BUDGETED
        # Original untouched.
        assert h.context_manager == ContextManager.FULL

    def test_apply_override_budget_merge(self):
        h = HarnessConfig()
        runner = CounterfactualRunner(evaluator=None)
        new = runner.apply_override(h, {"budget": {"max_tokens": 4096}})
        assert new.budget["max_tokens"] == 4096
        # Other budget keys preserved.
        assert new.budget["max_time_seconds"] == h.budget["max_time_seconds"]

    def test_apply_override_rollout_protocol(self):
        h = HarnessConfig()
        runner = CounterfactualRunner(evaluator=None)
        new = runner.apply_override(h, {"rollout_protocol": "ITERATIVE"})
        assert new.rollout_protocol == RolloutProtocol.ITERATIVE

    def test_end_to_end_verdict_flip(self):
        # Original harness: FULL → all OK
        # Replay harness: BUDGETED → case "cf-1" flips to WA
        case = BenchmarkCase(id="cf-1", prompt="solve it", language="python")
        verdict_map = {
            ("full", "cf-1"): Verdict.OK,
            ("budgeted", "cf-1"): Verdict.WA,
        }
        evaluator = _MockEvaluator(verdict_map)
        runner = CounterfactualRunner(evaluator)
        comparison = runner.run(
            original_harness=HarnessConfig(),
            override={"context_manager": "BUDGETED"},
            cases=[case],
            original_verdicts={"cf-1": "OK"},
            original_score=1.0,
        )
        assert comparison.replay_score == 0.0
        assert comparison.delta == pytest.approx(-1.0)
        assert len(comparison.flipped_cases) == 1
        assert comparison.flipped_cases[0] == ("cf-1", "OK", "WA")
        assert comparison.replay_verdicts["WA"] == 1
        # Replay harness should have BUDGETED applied.
        assert evaluator.calls[0].context_manager == ContextManager.BUDGETED

    def test_renders_text_report(self):
        case = BenchmarkCase(id="cf-1", prompt="solve it", language="python")
        evaluator = _MockEvaluator({("budgeted", "cf-1"): Verdict.WA})
        runner = CounterfactualRunner(evaluator)
        comparison = runner.run(
            original_harness=HarnessConfig(),
            override={"context_manager": "BUDGETED"},
            cases=[case],
            original_verdicts={"cf-1": "OK"},
            original_score=1.0,
        )
        comparison.session_id = "01SESS_X"
        comparison.iteration = 2
        text = comparison.render_text()
        assert "01SESS_X" in text
        assert "iteration 2" in text
        assert "BUDGETED" in text or "budgeted" in text
        assert "cf-1" in text
        assert "OK" in text and "WA" in text


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

class TestHelpers:

    def test_filter_cases_preserves_capture_order(self):
        a = BenchmarkCase(id="A", prompt="p", language="python")
        b = BenchmarkCase(id="B", prompt="p", language="python")
        c = BenchmarkCase(id="C", prompt="p", language="python")
        out = filter_cases_by_id([a, b, c], ["C", "A"])
        assert [x.id for x in out] == ["C", "A"]

    def test_harness_dict_to_config_roundtrip(self):
        d = {
            "system_prompt": "be careful",
            "rollout_protocol": "iterative",
            "context_manager": "budgeted",
            "budget": {"max_tokens": 1234},
        }
        cfg = harness_dict_to_config(d)
        assert cfg.system_prompt == "be careful"
        assert cfg.rollout_protocol == RolloutProtocol.ITERATIVE
        assert cfg.context_manager == ContextManager.BUDGETED
        assert cfg.budget["max_tokens"] == 1234

    def test_load_cases_from_dir(self, tmp_path: Path):
        d = tmp_path / "bench"
        d.mkdir()
        (d / "case_a.json").write_text(json.dumps({
            "id": "A", "prompt": "p", "language": "python",
        }))
        (d / "not-a-case.json").write_text(json.dumps({"foo": "bar"}))  # ignored
        cases = load_cases_from_dir(d)
        assert [c.id for c in cases] == ["A"]


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #

def test_cli_replay_help_exits_zero():
    """`autobench replay --help` must exit 0 with usage text."""
    proc = subprocess.run(
        [sys.executable, "-m", "autobench.cli", "replay", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "replay" in out.lower()
    assert "--session-id" in out
    assert "--iteration" in out
    assert "--override" in out
