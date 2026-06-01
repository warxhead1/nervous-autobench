"""Tests for autobench.trigger_daemon (nervous-bus-1hlf)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from autobench.observability import (
    CHANNEL_BENCH_COMPLETED,
    CHANNEL_CYCLE_REPORT,
    CHANNEL_CYCLE_REQUESTED,
    AutobenchObservability,
)
from autobench.daemons.trigger_daemon import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_ITER,
    DEFAULT_N_ADVOCATES,
    TriggerDaemon,
    build_cycle_config,
)


VALID_ULID = "01ABCDEFGHJKMNPQRSTVWXYZ00"


def _trigger_event(**overrides) -> dict:
    """Build a CloudEvents-lite envelope carrying a cycle.requested data block."""
    data = {
        "correlation_id": VALID_ULID,
        "requested_by": "operator",
        "domain": "codeforces_tier1",
        "ts": "2026-05-16T00:00:00Z",
    }
    data.update(overrides)
    return {
        "specversion": "1.0",
        "id": "01XXXXXXXXXXXXXXXXXXXXXXXX",
        "source": "/test",
        "type": CHANNEL_CYCLE_REQUESTED,
        "datacontenttype": "application/json",
        "time": "2026-05-16T00:00:00Z",
        "data": data,
    }


def _stub_runner_factory(events: list[dict], n_advocates: int = 2, n_cases: int = 3):
    """Build a runner factory returning a fixed event set for the daemon."""

    def _factory(config):
        return (
            events,
            {
                "cycle_id": "CYC" + "0" * 23,
                "n_advocates_hint": n_advocates,
                "n_cases_hint": n_cases,
            },
            "2026-05-16T00:00:00Z",
            "2026-05-16T00:30:00Z",
        )

    return _factory


def _make_obs(tmp_path: Path) -> AutobenchObservability:
    obs = AutobenchObservability(debug_file=tmp_path / "debug.jsonl")
    obs._pipe_disabled = True
    return obs


def _captured_reports(obs: AutobenchObservability) -> list[dict]:
    """Read all cycle.report events from the obs debug file."""
    path = obs._debug_file
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == CHANNEL_CYCLE_REPORT:
            out.append(ev["data"])
    return out


# --------------------------------------------------------------------------- #
# build_cycle_config
# --------------------------------------------------------------------------- #

def test_build_cycle_config_uses_defaults_when_overrides_absent():
    data = {
        "correlation_id": VALID_ULID,
        "requested_by": "hearth-loom",
        "domain": "codeforces_tier1",
    }
    cfg = build_cycle_config(data)
    assert cfg.correlation_id == VALID_ULID
    assert cfg.requested_by == "hearth-loom"
    assert cfg.domain == "codeforces_tier1"
    assert cfg.n_advocates == DEFAULT_N_ADVOCATES
    assert cfg.max_iter == DEFAULT_MAX_ITER
    assert cfg.budget_seconds == DEFAULT_BUDGET_SECONDS
    assert cfg.target_skill is None
    assert cfg.judges_per_case is None
    assert cfg.improver_strategy is None


def test_build_cycle_config_applies_overrides():
    data = {
        "correlation_id": VALID_ULID,
        "requested_by": "tengine",
        "domain": "shader_tier1",
        "bead_id": "test-bead-1",
        "n_advocates": 5,
        "max_iter": 8,
        "budget_seconds": 600,
        "target_skill": 0.75,
        "adversarial_ratio": 0.2,
        "judges_per_case": 7,
        "improver_strategy": "parallel",
        "notes": "exploration cycle",
    }
    cfg = build_cycle_config(data)
    assert cfg.bead_id == "test-bead-1"
    assert cfg.n_advocates == 5
    assert cfg.max_iter == 8
    assert cfg.budget_seconds == 600.0
    assert cfg.target_skill == 0.75
    assert cfg.adversarial_ratio == 0.2
    assert cfg.judges_per_case == 7
    assert cfg.improver_strategy == "parallel"
    assert cfg.notes == "exploration cycle"


# --------------------------------------------------------------------------- #
# handle_trigger
# --------------------------------------------------------------------------- #

def test_handle_trigger_valid_event_runs_cycle_and_emits_report(tmp_path):
    obs = _make_obs(tmp_path)
    factory = _stub_runner_factory(events=[], n_advocates=2, n_cases=3)
    daemon = TriggerDaemon(obs=obs, runner_factory=factory)

    event = _trigger_event(bead_id="test-bead-9", n_advocates=2, max_iter=2)
    report = daemon.handle_trigger(event)

    assert report["correlation_id"] == VALID_ULID
    assert report["domain"] == "codeforces_tier1"
    assert report["requested_by"] == "operator"
    assert report["bead_id"] == "test-bead-9"
    assert report["summary"]["promoted"] is False
    assert daemon.runs_handled == 1
    # The obs debug file should contain a cycle.report event.
    reports = _captured_reports(obs)
    assert len(reports) == 1
    assert reports[0]["correlation_id"] == VALID_ULID


def test_handle_trigger_invalid_event_emits_failure_report_without_running(tmp_path):
    obs = _make_obs(tmp_path)
    called = {"ran": False}

    def _factory(config):
        called["ran"] = True
        return ([], {}, "", "")

    daemon = TriggerDaemon(obs=obs, runner_factory=_factory)

    # correlation_id is wrong length → schema rejects.
    bad_event = _trigger_event(correlation_id="too-short")
    report = daemon.handle_trigger(bad_event)
    assert called["ran"] is False
    assert report["summary"]["promoted"] is False
    assert "trigger validation failed" in report["patterns"]["notes"]
    # Report still emitted so consumers see the rejection.
    reports = _captured_reports(obs)
    assert len(reports) == 1


def test_handle_trigger_missing_required_field_rejected(tmp_path):
    obs = _make_obs(tmp_path)
    daemon = TriggerDaemon(obs=obs, runner_factory=_stub_runner_factory([]))
    bad = _trigger_event()
    # Drop the domain field.
    del bad["data"]["domain"]
    report = daemon.handle_trigger(bad)
    assert "domain" in report["patterns"]["notes"]
    assert daemon.runs_handled == 0


def test_handle_trigger_correlation_id_flows_through_to_report(tmp_path):
    obs = _make_obs(tmp_path)
    daemon = TriggerDaemon(obs=obs, runner_factory=_stub_runner_factory([]))

    unique = "01ZZZZZZZZZZZZZZZZZZZZZZZZ"
    event = _trigger_event(correlation_id=unique)
    report = daemon.handle_trigger(event)
    assert report["correlation_id"] == unique


def test_handle_trigger_applies_bead_id_default_when_event_omits(tmp_path):
    obs = _make_obs(tmp_path)
    daemon = TriggerDaemon(
        bead_id_default="daemon-default-bead",
        obs=obs,
        runner_factory=_stub_runner_factory([]),
    )
    event = _trigger_event()
    report = daemon.handle_trigger(event)
    assert report["bead_id"] == "daemon-default-bead"


def test_handle_trigger_cycle_failure_emits_failure_report(tmp_path):
    obs = _make_obs(tmp_path)

    def _failing(config):
        raise RuntimeError("cycle blew up")

    daemon = TriggerDaemon(obs=obs, runner_factory=_failing)
    event = _trigger_event()
    report = daemon.handle_trigger(event)
    assert report["summary"]["promoted"] is False
    assert "cycle execution failed" in report["patterns"]["notes"]


# --------------------------------------------------------------------------- #
# listen() — mock_subprocess path
# --------------------------------------------------------------------------- #

def test_listen_mock_subprocess_processes_events(tmp_path):
    obs = _make_obs(tmp_path)
    daemon = TriggerDaemon(obs=obs, runner_factory=_stub_runner_factory([]))

    events = [_trigger_event(correlation_id=cid) for cid in (
        "01AAAAAAAAAAAAAAAAAAAAAAA1",
        "01AAAAAAAAAAAAAAAAAAAAAAA2",
        "01AAAAAAAAAAAAAAAAAAAAAAA3",
    )]
    lines = [json.dumps(e) for e in events]
    handled = daemon.listen(mock_subprocess=lines)
    assert handled == 3
    assert daemon.runs_handled == 3


def test_listen_max_runs_exits_after_n(tmp_path):
    obs = _make_obs(tmp_path)
    daemon = TriggerDaemon(obs=obs, runner_factory=_stub_runner_factory([]))

    lines = []
    for cid_suffix in range(5):
        cid = f"01{'A'*23}{cid_suffix}"
        lines.append(json.dumps(_trigger_event(correlation_id=cid)))
    handled = daemon.listen(mock_subprocess=lines, max_runs=2)
    assert handled == 2


def test_listen_skips_malformed_json(tmp_path):
    obs = _make_obs(tmp_path)
    daemon = TriggerDaemon(obs=obs, runner_factory=_stub_runner_factory([]))

    lines = [
        "not json {",
        json.dumps(_trigger_event(correlation_id="01AAAAAAAAAAAAAAAAAAAAAAA9")),
        "",
    ]
    handled = daemon.listen(mock_subprocess=lines)
    # Only one well-formed line processed.
    assert handled == 1


# --------------------------------------------------------------------------- #
# bench_completed_promotion emit (Fix 1)
# --------------------------------------------------------------------------- #

def test_handle_trigger_emits_bench_completed_when_bead_id_set(tmp_path):
    """When bead_id is present, bus.bead.bench_completed.v1 must be emitted."""
    obs = _make_obs(tmp_path)
    obs.bench_completed_promotion = MagicMock()

    factory = _stub_runner_factory(events=[], n_advocates=2, n_cases=5)
    daemon = TriggerDaemon(obs=obs, runner_factory=factory)

    event = _trigger_event(bead_id="bead-seal-test-001", n_advocates=2, max_iter=2)
    report = daemon.handle_trigger(event)

    assert report["bead_id"] == "bead-seal-test-001"
    obs.bench_completed_promotion.assert_called_once()
    kwargs = obs.bench_completed_promotion.call_args.kwargs
    assert kwargs["bead_id"] == "bead-seal-test-001"
    assert "baseline_metric" in kwargs
    assert "treatment_metric" in kwargs
    assert "delta" in kwargs
    assert "n" in kwargs
    assert "passes_threshold" in kwargs
    assert kwargs["ci_lower"] is None
    assert kwargs["ci_upper"] is None


def test_handle_trigger_no_bench_completed_when_bead_id_absent(tmp_path):
    """When bead_id is absent, bench_completed_promotion must NOT be called."""
    obs = _make_obs(tmp_path)
    obs.bench_completed_promotion = MagicMock()

    factory = _stub_runner_factory(events=[], n_advocates=2, n_cases=5)
    daemon = TriggerDaemon(obs=obs, runner_factory=factory)

    # Event without bead_id and daemon has no default.
    event = _trigger_event()
    daemon.handle_trigger(event)

    obs.bench_completed_promotion.assert_not_called()


def test_handle_trigger_no_bench_completed_when_bead_id_none(tmp_path):
    """Explicit bead_id=None in event and no daemon default → no emit."""
    obs = _make_obs(tmp_path)
    obs.bench_completed_promotion = MagicMock()

    factory = _stub_runner_factory(events=[], n_advocates=1, n_cases=2)
    daemon = TriggerDaemon(obs=obs, runner_factory=factory)

    event = _trigger_event()
    # Ensure data block doesn't carry a bead_id key at all.
    event["data"].pop("bead_id", None)
    daemon.handle_trigger(event)

    obs.bench_completed_promotion.assert_not_called()


# --------------------------------------------------------------------------- #
# Improver strategy wiring (Fix 2)
# --------------------------------------------------------------------------- #

def test_handle_trigger_improver_strategy_from_config_passed_to_runner(tmp_path):
    """improver_strategy from the trigger event must reach the runner factory."""
    obs = _make_obs(tmp_path)

    captured_configs = []

    def _capturing_factory(config):
        captured_configs.append(config)
        return (
            [],
            {"cycle_id": "CYC" + "0" * 23, "n_advocates_hint": 1, "n_cases_hint": 1},
            "2026-05-16T00:00:00Z",
            "2026-05-16T00:30:00Z",
        )

    daemon = TriggerDaemon(obs=obs, runner_factory=_capturing_factory)
    # Schema allows "vote" and "parallel" for improver_strategy.
    event = _trigger_event(improver_strategy="parallel")
    daemon.handle_trigger(event)

    assert len(captured_configs) == 1
    assert captured_configs[0].improver_strategy == "parallel"


def test_handle_trigger_improver_strategy_defaults_absent_when_not_in_event(tmp_path):
    """When improver_strategy is absent from the event, config has it as None."""
    obs = _make_obs(tmp_path)
    captured_configs = []

    def _capturing_factory(config):
        captured_configs.append(config)
        return (
            [],
            {"cycle_id": "CYC" + "0" * 23, "n_advocates_hint": 1, "n_cases_hint": 1},
            "2026-05-16T00:00:00Z",
            "2026-05-16T00:30:00Z",
        )

    daemon = TriggerDaemon(obs=obs, runner_factory=_capturing_factory)
    event = _trigger_event()
    daemon.handle_trigger(event)

    assert len(captured_configs) == 1
    assert captured_configs[0].improver_strategy is None


def test_run_cycle_with_population_runner_uses_improver_strategy(monkeypatch):
    """_run_cycle_with_population_runner resolves improver from config > env > default."""
    from autobench.daemons.trigger_daemon import _run_cycle_with_population_runner, CycleConfig
    from autobench.observability import _ulid

    captured_improver = {}

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            captured_improver["value"] = kwargs.get("improver")

        def run(self, cases):
            # Return a minimal result object.
            class _Result:
                cycle_id = _ulid()
                advocates = []
            return _Result()

    # PopulationRunner, BenchmarkRegistry, BenchmarkEvaluator are lazy-imported
    # inside _run_cycle_with_population_runner, so we patch their source modules.
    monkeypatch.setattr(
        "autobench.population.PopulationRunner", _FakeRunner
    )

    # The registry must return at least one case for the test domain so that the
    # function doesn't short-circuit before constructing the PopulationRunner.
    _DUMMY_CASE = object()

    class _FakeRegistry:
        @staticmethod
        def default():
            class _Reg:
                def load_all_cases(self):
                    return {"test_domain": [_DUMMY_CASE]}
            return _Reg()

    class _FakeEvaluator:
        def __init__(self, obs):
            pass

    monkeypatch.setattr("autobench.benchmark_registry.BenchmarkRegistry", _FakeRegistry)
    monkeypatch.setattr("autobench.evaluator.BenchmarkEvaluator", _FakeEvaluator)

    obs = MagicMock()
    obs._debug_file = None

    # config.improver_strategy="vote" → runner gets "vote"
    config = CycleConfig(
        correlation_id=VALID_ULID,
        requested_by="test",
        domain="test_domain",
        improver_strategy="vote",
    )
    _run_cycle_with_population_runner(config, obs)
    assert captured_improver.get("value") == "vote"

    # config.improver_strategy=None + no env → runner gets "minimax"
    config2 = CycleConfig(
        correlation_id=VALID_ULID,
        requested_by="test",
        domain="test_domain",
        improver_strategy=None,
    )
    monkeypatch.delenv("AUTOBENCH_DEFAULT_IMPROVER", raising=False)
    _run_cycle_with_population_runner(config2, obs)
    assert captured_improver.get("value") == "minimax"

    # config.improver_strategy=None + env="llm_vote" → runner gets "llm_vote"
    monkeypatch.setenv("AUTOBENCH_DEFAULT_IMPROVER", "llm_vote")
    _run_cycle_with_population_runner(config2, obs)
    assert captured_improver.get("value") == "llm_vote"
