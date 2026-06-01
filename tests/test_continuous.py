"""Tests for the autobench continuous-mode daemon.

Covers:
    * Workspace creation on first run.
    * ``run_one_session`` updates stats and emits the session-complete event,
      using a mocked evaluator + mocked improver.
    * Daemon backoff when ``rate_budget.check()`` returns blocked.
    * ``SurpriseDigest.generate_digest`` flags:
        - a synthetic confidently-wrong prediction
        - a divergence-win event (LLM diverged from heuristic AND won)
        - a score regression within a single session
    * CLI smoke: ``python3 -m autobench.continuous --help`` exits 0.
    * ``status`` subcommand does not crash on an empty workspace.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from autobench.daemons.continuous import (
    ContinuousModeDaemon,
    Digest,
    Surprise,
    SurpriseDigest,
    _serialise_harness,
    main as cli_main,
)
from autobench.core import HarnessConfig
from autobench.evaluator import BenchmarkCase, BenchmarkResult
from autobench.observability import AutobenchObservability


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "continuous"


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force AutobenchObservability into debug-file mode (no zellij)."""
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    return tmp_path / "debug.jsonl"


# --------------------------------------------------------------------------- #
# Workspace creation
# --------------------------------------------------------------------------- #


def test_workspace_created_on_first_use(workspace: Path) -> None:
    assert not workspace.exists()
    daemon = ContinuousModeDaemon(workspace=workspace)
    assert workspace.is_dir()
    assert (workspace / "harness.json").is_file()
    assert (workspace / "stats.jsonl").is_file()
    assert (workspace / "archive").is_dir()
    assert (workspace / "digests").is_dir()
    # Canonical loads cleanly.
    h = daemon.current_canonical_harness()
    assert isinstance(h, HarnessConfig)


# --------------------------------------------------------------------------- #
# run_one_session — mock evaluator + improver
# --------------------------------------------------------------------------- #


class _StubEvaluator:
    """Returns a BenchmarkResult with a configurable aggregate_score."""

    def __init__(self, score: float = 0.5) -> None:
        self.score = score
        self.calls = 0

    def run(self, harness: HarnessConfig, cases: list[Any], obs: Any = None) -> BenchmarkResult:
        from autobench.core import HarnessResult, Verdict
        self.calls += 1
        # Provide a populated case_results list so the rule-based improver
        # (which divides by len(case_results)) doesn't hit ZeroDivisionError.
        return BenchmarkResult(
            case_results=[
                HarnessResult(p_score=self.score, verdict=Verdict.OK, cost_dollars=0.0)
                for _ in (cases or [None])
            ],
            aggregate_score=self.score,
            total_latency_ms=10.0,
            verdict_counts={"OK": len(cases) or 1},
        )


def test_run_one_session_updates_stats(
    workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    debug = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug)

    # Patch the curriculum-source picker to return our stub cases.
    cases = [BenchmarkCase(id="t1", prompt="echo 1", language="python")]
    import autobench.daemons.continuous as cont_mod
    monkeypatch.setattr(cont_mod, "_pick_benchmark_source",
                        lambda ws: (Path("stub"), cases))

    # Patch improver inside SelfImprovingHarness via default_improver=rule_based
    # path so we never hit the network.
    daemon = ContinuousModeDaemon(
        workspace=workspace,
        evaluator=_StubEvaluator(score=0.7),
        improver="rule_based",
        max_iterations=2,
        obs=obs,
    )
    rec = daemon.run_one_session()
    # Stats appended
    rows = [
        line
        for line in (workspace / "stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["benchmark_source"] == "stub"
    assert payload["initial_score"] == pytest.approx(0.7)
    # final_score will also be ~0.7 since the stub returns the same score; the
    # important invariant is that the record exists and `error` is empty.
    assert rec.error == ""

    # session_complete emitted to debug file
    text = debug.read_text() if debug.is_file() else ""
    assert "autobench.continuous.session_complete.v1" in text


# --------------------------------------------------------------------------- #
# Daemon backoff under rate budget
# --------------------------------------------------------------------------- #


class _BlockedBudget:
    """Rate budget that always reports blocked."""

    window_seconds = 18000.0

    def check(self) -> tuple[bool, str]:
        return False, "synthetic block"

    def time_until_available(self) -> float:
        return 0.1  # quick retry for the test


def test_daemon_backs_off_when_budget_blocked(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = ContinuousModeDaemon(
        workspace=workspace,
        rate_budget=_BlockedBudget(),
    )
    ran: list[int] = []

    def _fake_run_one() -> Any:
        ran.append(1)
        raise AssertionError("should not run while blocked")

    monkeypatch.setattr(daemon, "run_one_session", _fake_run_one)

    # Stop after a brief wait so the loop exits the backoff path.
    import threading
    timer = threading.Timer(0.3, daemon.request_stop)
    timer.start()
    daemon.run_forever(max_sessions=5)
    timer.cancel()
    assert ran == []


# --------------------------------------------------------------------------- #
# SurpriseDigest — synthetic events
# --------------------------------------------------------------------------- #


def _evt(
    channel: str,
    data: dict[str, Any],
    *,
    when: dt.datetime | None = None,
) -> dict[str, Any]:
    """Helper that mints a CloudEvents-lite envelope for a synthetic event."""
    t = (when or dt.datetime.utcnow()).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {
        "specversion": "1.0",
        "id": "00000000000000000000000001",
        "source": "/autobench",
        "type": channel,
        "datacontenttype": "application/json",
        "time": t,
        "data": data,
    }


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def test_digest_flags_confidently_wrong_prediction(
    workspace: Path, tmp_path: Path
) -> None:
    debug = tmp_path / "debug.jsonl"
    events = [
        _evt(
            "autobench.improver.prediction.verified.v1",
            {
                "session_id": "sess-A",
                "iteration": 3,
                "model": "minimax-m2.7",
                "confidence": 0.92,
                "outcome_label": "refuted",
                "score_match_ratio": 0.15,
                "verdict_match_ratio": 0.2,
            },
        ),
    ]
    _write_events(debug, events)
    sd = SurpriseDigest(workspace=workspace, debug_file=debug)
    digest = sd.generate_digest(date="2026-05-16")
    assert any(s.kind == "confident_wrong" for s in digest.surprises)
    big = digest.biggest_surprise
    assert big is not None
    assert "REFUTED" in big.summary
    assert "0.92" in big.summary


def test_digest_flags_divergence_win(workspace: Path, tmp_path: Path) -> None:
    debug = tmp_path / "debug.jsonl"
    events = [
        # Iteration 0 baseline
        _evt(
            "autobench.iteration.v1",
            {
                "session_id": "sess-B",
                "iteration": 0,
                "status": "complete",
                "aggregate_score": 0.40,
                "verdict_counts": {"OK": 1},
            },
        ),
        # Iteration 1 — LLM diverged from heuristic
        _evt(
            "autobench.improver.divergence.v1",
            {
                "session_id": "sess-B",
                "iteration": 1,
                "divergent": True,
                "divergence_summary": "system_prompt_delta: '' → 'add ex.'",
            },
        ),
        _evt(
            "autobench.iteration.v1",
            {
                "session_id": "sess-B",
                "iteration": 1,
                "status": "complete",
                "aggregate_score": 0.65,  # +0.25 — clear divergence win
                "verdict_counts": {"OK": 1},
            },
        ),
    ]
    _write_events(debug, events)
    sd = SurpriseDigest(workspace=workspace, debug_file=debug)
    digest = sd.generate_digest(date="2026-05-16")
    assert any(s.kind == "divergence_win" for s in digest.surprises)
    win = next(s for s in digest.surprises if s.kind == "divergence_win")
    assert "+0.250" in win.summary or "0.25" in win.summary


def test_digest_flags_score_regression(workspace: Path, tmp_path: Path) -> None:
    debug = tmp_path / "debug.jsonl"
    events = [
        _evt(
            "autobench.iteration.v1",
            {
                "session_id": "sess-C",
                "iteration": 0,
                "status": "complete",
                "aggregate_score": 0.80,
                "verdict_counts": {"OK": 1},
            },
        ),
        _evt(
            "autobench.iteration.v1",
            {
                "session_id": "sess-C",
                "iteration": 1,
                "status": "complete",
                "aggregate_score": 0.50,
                "verdict_counts": {"OK": 1},
            },
        ),
    ]
    _write_events(debug, events)
    sd = SurpriseDigest(workspace=workspace, debug_file=debug)
    digest = sd.generate_digest(date="2026-05-16")
    assert any(s.kind == "regression" for s in digest.surprises)


def test_digest_writes_markdown(workspace: Path, tmp_path: Path) -> None:
    debug = tmp_path / "debug.jsonl"
    events = [
        _evt(
            "autobench.improver.prediction.verified.v1",
            {
                "session_id": "sess-X",
                "iteration": 1,
                "model": "minimax-m2.7",
                "confidence": 0.95,
                "outcome_label": "refuted",
                "score_match_ratio": 0.1,
                "verdict_match_ratio": 0.1,
            },
        ),
    ]
    _write_events(debug, events)
    sd = SurpriseDigest(workspace=workspace, debug_file=debug)
    digest = sd.generate_digest(date="2026-05-16")
    out = sd.write_digest(digest)
    assert out.is_file()
    md = out.read_text()
    assert "autobench continuous-mode digest" in md
    assert "Biggest surprise" in md


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "autobench.continuous", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "continuous-mode daemon" in result.stdout.lower() or "continuous" in result.stdout.lower()


def test_cli_status_on_empty_workspace(tmp_path: Path) -> None:
    # `status` is the default when no subcommand is provided.
    rc = cli_main(["--workspace", str(tmp_path / "ws")])
    assert rc == 0
    # Workspace was created.
    assert (tmp_path / "ws" / "harness.json").is_file()


def test_cli_digest_on_empty_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty debug.jsonl: the digest must still produce a digest with 0 surprises.
    debug = tmp_path / "debug.jsonl"
    debug.write_text("")
    monkeypatch.setattr(
        "autobench.continuous.DEBUG_FILE", debug
    )
    # Default subcommand `status` was tested above; explicitly run digest.
    rc = cli_main(["--workspace", str(tmp_path / "ws2"), "digest"])
    assert rc == 0
    out = tmp_path / "ws2" / "digests"
    assert out.is_dir()
    files = list(out.glob("*.md"))
    assert files
