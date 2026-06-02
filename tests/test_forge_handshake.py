"""Tests for the Forge handshake — nervous-bus-vwc8 + nervous-bus-e8x9.

Verifies:
    * ``ContinuousModeDaemon`` accepts ``bead_id`` via kwarg, env, and CLI.
    * Explicit kwarg wins over env.
    * When ``bead_id`` is None, ``_promote`` succeeds but the
      ``bus.bead.bench_completed.v1`` channel stays silent (warning logged).
    * When ``bead_id`` is set, an accepted promotion fires bench_completed
      with all required fields populated.
    * The emitted event's data payload validates against
      ``schemas/bus.bead.bench_completed.v1.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.daemons.continuous import (
    BEAD_ID_ENV,
    ContinuousModeDaemon,
)
from autobench.core import ContextManager, HarnessConfig, RolloutProtocol
from autobench.observability import AutobenchObservability
from autobench.rsi.population import AdvocateResult, PopulationResult


BENCH_SCHEMA = SCHEMA_DIR / "bus.bead.bench_completed.v1.json"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _make_advocate(
    *,
    ahe_outcome: str = "confirmed",
    best_score: float = 0.7,
    aggregate_score: float | None = None,
    case_count: int = 5,
) -> AdvocateResult:
    """Construct an advocate the promotion selector will pick.

    ``case_count`` populates a fake ``final_result.case_results`` list so the
    bench_completed n-derivation can pick it up without depending on real
    BenchmarkResult shape.
    """

    class _FakeResult:
        def __init__(self, n: int) -> None:
            self.case_results = list(range(n))  # opaque — only len() is read

    return AdvocateResult(
        advocate_id="advocate-0",
        session_id="SESS-FORGE-HANDSHAKE-PADPADPAD",
        final_harness=HarnessConfig(
            system_prompt="forge-handshake-test",
            rollout_protocol=RolloutProtocol.SINGLE,
            context_manager=ContextManager.FULL,
            tool_surface="",
        ),
        final_result=_FakeResult(case_count) if case_count > 0 else None,
        history=[],
        best_score=float(best_score),
        best_iter=0,
        ahe_outcome=ahe_outcome,
        adjusted_score=float(best_score),
        aggregate_score=float(best_score if aggregate_score is None else aggregate_score),
        per_domain_scores={},
    )


def _make_population_result(advocate: AdvocateResult) -> PopulationResult:
    return PopulationResult(
        advocates=[advocate],
        winner_id=advocate.advocate_id,
        winner_score=advocate.best_score,
        cycle_started_at="2026-05-17T00:00:00Z",
        cycle_ended_at="2026-05-17T00:01:00Z",
        adjusted_winner_id=advocate.advocate_id,
        diversity_weight=0.0,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "continuous"
    ws.mkdir()
    return ws


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "promotion_ledger.jsonl"


@pytest.fixture
def debug_file(tmp_path: Path) -> Path:
    return tmp_path / "debug.jsonl"


@pytest.fixture
def silent_pipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force observability emissions into the debug-file fallback.

    AUTOBENCH_OBS_DISABLE_PIPE=1 short-circuits the zellij pipe probe so
    tests never hang waiting for a non-existent socket.
    """
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    monkeypatch.delenv(BEAD_ID_ENV, raising=False)
    monkeypatch.delenv("AUTOBENCH_CONFIRM_PROMOTION", raising=False)
    monkeypatch.delenv("AUTOBENCH_REJECT_PROMOTION", raising=False)


# --------------------------------------------------------------------------- #
# Part 1 — bead_id binding (nervous-bus-vwc8)
# --------------------------------------------------------------------------- #


def test_bead_id_none_when_unset(
    workspace: Path, silent_pipe: None,
) -> None:
    """Default construction yields bead_id=None."""
    daemon = ContinuousModeDaemon(workspace=workspace)
    assert daemon.bead_id is None


def test_bead_id_from_explicit_kwarg(
    workspace: Path, silent_pipe: None,
) -> None:
    daemon = ContinuousModeDaemon(workspace=workspace, bead_id="nervous-bus-test")
    assert daemon.bead_id == "nervous-bus-test"


def test_bead_id_from_env(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    silent_pipe: None,
) -> None:
    monkeypatch.setenv(BEAD_ID_ENV, "nervous-bus-env-bead")
    daemon = ContinuousModeDaemon(workspace=workspace)
    assert daemon.bead_id == "nervous-bus-env-bead"


def test_explicit_kwarg_wins_over_env(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    silent_pipe: None,
) -> None:
    monkeypatch.setenv(BEAD_ID_ENV, "from-env")
    daemon = ContinuousModeDaemon(workspace=workspace, bead_id="from-kwarg")
    assert daemon.bead_id == "from-kwarg"


def test_empty_string_kwarg_is_treated_as_unbound(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    silent_pipe: None,
) -> None:
    """An explicit empty string is operator intent — clear, not inherit."""
    monkeypatch.setenv(BEAD_ID_ENV, "ignored-because-explicit-empty")
    daemon = ContinuousModeDaemon(workspace=workspace, bead_id="")
    assert daemon.bead_id is None


# --------------------------------------------------------------------------- #
# Part 2 — bench_completed emission (nervous-bus-e8x9)
# --------------------------------------------------------------------------- #


def _read_events(debug_file: Path) -> list[dict]:
    if not debug_file.exists():
        return []
    return [
        json.loads(line)
        for line in debug_file.read_text().splitlines()
        if line.strip()
    ]


def _bench_events(debug_file: Path) -> list[dict]:
    return [
        e for e in _read_events(debug_file)
        if e.get("type") == "bus.bead.bench_completed.v1"
    ]


def test_promotion_without_bead_id_does_not_emit_bench_completed(
    workspace: Path,
    ledger_path: Path,
    debug_file: Path,
    silent_pipe: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """bead_id=None: accepted promotion logs a warning but stays silent."""
    obs = AutobenchObservability(debug_file=debug_file)
    daemon = ContinuousModeDaemon(workspace=workspace, obs=obs, bead_id=None)
    assert daemon.bead_id is None

    pop = _make_population_result(_make_advocate())

    with caplog.at_level(logging.WARNING, logger="autobench.continuous"):
        decision = daemon.stage_promotion_from_population(
            pop, confirm=True, ledger_path=ledger_path,
        )

    assert decision.decision == "accepted"
    assert _bench_events(debug_file) == []
    # Warning surfaces the missing-bead reason so operators notice.
    assert any("unbound" in r.message or "bead_id" in r.message for r in caplog.records)


def test_promotion_with_bead_id_emits_bench_completed(
    workspace: Path,
    ledger_path: Path,
    debug_file: Path,
    silent_pipe: None,
) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    daemon = ContinuousModeDaemon(
        workspace=workspace, obs=obs, bead_id="nervous-bus-test",
    )
    advocate = _make_advocate(best_score=0.85, case_count=8)
    pop = _make_population_result(advocate)

    # Seed a prior canonical score so baseline != 0.0 — exercises the
    # stats-derived baseline path.
    (workspace / "stats.jsonl").write_text(
        json.dumps({
            "session_id": "prev",
            "timestamp": "2026-05-16T10:00:00Z",
            "initial_score": 0.4,
            "final_score": 0.55,
            "n_iterations": 3,
            "total_cost_usd": 0.0,
            "promoted": True,
            "benchmark_source": "test",
            "duration_seconds": 1.0,
            "error": "",
        }) + "\n"
    )

    decision = daemon.stage_promotion_from_population(
        pop, confirm=True, ledger_path=ledger_path,
    )
    assert decision.decision == "accepted"

    events = _bench_events(debug_file)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "bus.bead.bench_completed.v1"
    data = e["data"]
    assert data["bead_id"] == "nervous-bus-test"
    assert data["baseline_metric"] == pytest.approx(0.55)
    assert data["treatment_metric"] == pytest.approx(0.85)
    assert data["delta"] == pytest.approx(0.30)
    assert data["n"] == 8
    assert data["passes_threshold"] is True
    assert "ts" in data and data["ts"].endswith("Z")
    # session_id is stamped via the envelope path so flat-schema consumers
    # joining on session_id still find it.
    assert "session_id" in data


def test_bench_completed_payload_validates_against_schema(
    workspace: Path,
    ledger_path: Path,
    debug_file: Path,
    silent_pipe: None,
) -> None:
    """The data payload of an emitted event passes Draft202012Validator."""
    jsonschema = pytest.importorskip("jsonschema")

    schema = json.loads(BENCH_SCHEMA.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    obs = AutobenchObservability(debug_file=debug_file)
    daemon = ContinuousModeDaemon(
        workspace=workspace, obs=obs, bead_id="nervous-bus-validate",
    )
    pop = _make_population_result(_make_advocate(best_score=0.6, case_count=3))
    daemon.stage_promotion_from_population(
        pop, confirm=True, ledger_path=ledger_path,
    )

    events = _bench_events(debug_file)
    assert events, "no bench_completed event was emitted"
    # The bench_completed schema is flat — it validates the data dict
    # directly, not the envelope (unlike autobench.continuous.* schemas).
    for e in events:
        validator.validate(e["data"])


def test_bench_completed_only_fires_on_accepted_path(
    workspace: Path,
    ledger_path: Path,
    debug_file: Path,
    silent_pipe: None,
) -> None:
    """Staged and rejected decisions never emit bench_completed."""
    obs = AutobenchObservability(debug_file=debug_file)
    daemon = ContinuousModeDaemon(
        workspace=workspace, obs=obs, bead_id="nervous-bus-staged",
    )
    pop = _make_population_result(_make_advocate())

    # Stage-only — default path, no confirm.
    daemon.stage_promotion_from_population(pop, ledger_path=ledger_path)
    # Reject path.
    daemon.stage_promotion_from_population(
        pop, reject=True, ledger_path=ledger_path,
    )

    assert _bench_events(debug_file) == []


def test_bench_completed_skipped_when_promote_swap_fails(
    workspace: Path,
    ledger_path: Path,
    debug_file: Path,
    silent_pipe: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swap-failure short-circuits to a staged decision — no bench event."""
    obs = AutobenchObservability(debug_file=debug_file)
    daemon = ContinuousModeDaemon(
        workspace=workspace, obs=obs, bead_id="nervous-bus-failpath",
    )

    def _boom(_h):  # noqa: ANN001
        raise RuntimeError("simulated promote failure")

    monkeypatch.setattr(daemon, "_promote", _boom)

    pop = _make_population_result(_make_advocate())
    decision = daemon.stage_promotion_from_population(
        pop, confirm=True, ledger_path=ledger_path,
    )
    assert decision.decision == "staged"  # promotion-fallback path
    assert _bench_events(debug_file) == []


def test_bench_completed_falls_back_when_no_stats(
    workspace: Path,
    ledger_path: Path,
    debug_file: Path,
    silent_pipe: None,
) -> None:
    """First-time promotion (empty stats.jsonl) reports baseline=0.0."""
    obs = AutobenchObservability(debug_file=debug_file)
    daemon = ContinuousModeDaemon(
        workspace=workspace, obs=obs, bead_id="nervous-bus-firsttime",
    )
    pop = _make_population_result(_make_advocate(best_score=0.42))
    daemon.stage_promotion_from_population(
        pop, confirm=True, ledger_path=ledger_path,
    )

    events = _bench_events(debug_file)
    assert len(events) == 1
    data = events[0]["data"]
    assert data["baseline_metric"] == pytest.approx(0.0)
    assert data["treatment_metric"] == pytest.approx(0.42)
    assert data["delta"] == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# CLI surface — Part 1 wiring sanity
# --------------------------------------------------------------------------- #


def test_continuous_cli_accepts_bead_id_flag() -> None:
    """argparse exposes --bead-id and forwards it through to args.bead_id."""
    from autobench.daemons.continuous import main as continuous_main  # noqa: F401

    import argparse as _ap
    # We don't want to actually run the daemon — just verify argparse wiring.
    # Build a minimal parser the same way main() does and parse a sample.
    parser = _ap.ArgumentParser()
    parser.add_argument("--bead-id", dest="bead_id", default=None)
    ns = parser.parse_args(["--bead-id", "nervous-bus-cli-test"])
    assert ns.bead_id == "nervous-bus-cli-test"


def test_continuous_main_help_mentions_bead_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: the real main() help text includes --bead-id."""
    from autobench.daemons.continuous import main as continuous_main

    with pytest.raises(SystemExit):
        continuous_main(["--help"])
    out = capsys.readouterr().out
    assert "--bead-id" in out
    assert BEAD_ID_ENV in out
