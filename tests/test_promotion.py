"""Tests for cross-run promotion (nervous-bus-msqa / wire-pop Phase 5).

Verifies:
    * ``select_promotion_candidate`` never returns a refuted/refuted_live
      advocate.
    * It DOES return a confirmed advocate when one exists.
    * Stage-only decision does NOT call ``_promote``.
    * Confirm decision DOES call ``_promote`` and the canonical harness
      changes on disk.
    * Every code path appends one line to the promotion ledger.
    * The schema validates the emitted bus event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.continuous import (
    ContinuousModeDaemon,
    PromotionDecision,
    _resolve_promotion_ledger_path,
)
from autobench.core import ContextManager, HarnessConfig, RolloutProtocol
from autobench.population import (
    AdvocateResult,
    PopulationResult,
    select_promotion_candidate,
)
from autobench.rsi_loop import ImprovementDelta




# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_advocate(
    idx: int,
    *,
    ahe_outcome: str,
    best_score: float = 0.5,
    adjusted_score: float | None = None,
    aggregate_score: float | None = None,
    per_domain_scores: dict[str, float] | None = None,
) -> AdvocateResult:
    return AdvocateResult(
        advocate_id=f"advocate-{idx}",
        session_id=f"SESS-{idx:02d}-PADPADPADPADPADPAD",
        final_harness=HarnessConfig(
            system_prompt=f"prompt-{idx}",
            rollout_protocol=RolloutProtocol.SINGLE,
            context_manager=ContextManager.FULL,
            tool_surface="",
        ),
        final_result=None,
        history=[],
        best_score=best_score,
        best_iter=0,
        ahe_outcome=ahe_outcome,
        adjusted_score=float(best_score if adjusted_score is None else adjusted_score),
        aggregate_score=float(best_score if aggregate_score is None else aggregate_score),
        per_domain_scores=dict(per_domain_scores or {}),
    )


def _make_population_result(advocates: list[AdvocateResult]) -> PopulationResult:
    return PopulationResult(
        advocates=advocates,
        winner_id=advocates[0].advocate_id,
        winner_score=advocates[0].best_score,
        cycle_started_at="2026-05-16T00:00:00Z",
        cycle_ended_at="2026-05-16T00:01:00Z",
        adjusted_winner_id=advocates[0].advocate_id,
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


# --------------------------------------------------------------------------- #
# select_promotion_candidate
# --------------------------------------------------------------------------- #


def test_refuted_advocate_never_selected() -> None:
    """An advocate whose latest prediction was refuted is NEVER selected."""
    advocates = [
        _make_advocate(0, ahe_outcome="refuted", best_score=0.99),
        _make_advocate(1, ahe_outcome="refuted_live", best_score=0.95),
        _make_advocate(2, ahe_outcome="none", best_score=0.90),
    ]
    assert select_promotion_candidate(advocates) is None


def test_confirmed_advocate_is_selected() -> None:
    """A confirmed advocate IS selected even when a refuted one scores higher."""
    advocates = [
        _make_advocate(0, ahe_outcome="refuted", best_score=0.99),
        _make_advocate(1, ahe_outcome="confirmed", best_score=0.50),
        _make_advocate(2, ahe_outcome="refuted_live", best_score=0.80),
    ]
    picked = select_promotion_candidate(advocates)
    assert picked is not None
    assert picked.advocate_id == "advocate-1"
    assert picked.ahe_outcome == "confirmed"


def test_confirmed_outranks_partial() -> None:
    """Confirmed outranks partial even when partial has a higher score."""
    advocates = [
        _make_advocate(0, ahe_outcome="partial", best_score=0.99),
        _make_advocate(1, ahe_outcome="confirmed", best_score=0.30),
    ]
    picked = select_promotion_candidate(advocates)
    assert picked is not None
    assert picked.advocate_id == "advocate-1"


def test_empty_advocate_list_returns_none() -> None:
    assert select_promotion_candidate([]) is None


def test_promotion_ranks_by_aggregate_score_not_best_score() -> None:
    """nervous-bus-qp91: cross-domain aggregate trumps single-domain best_score.

    Both advocates are 'confirmed'. advocate-0 has higher best_score but
    LOWER aggregate_score (poor on the secondary domain). advocate-1 wins
    because its cross-domain evidence is stronger.
    """
    advocates = [
        _make_advocate(
            0,
            ahe_outcome="confirmed",
            best_score=0.90,
            adjusted_score=0.90,
            aggregate_score=0.50,  # poor on secondary domain
            per_domain_scores={"cf": 0.90, "multifile": 0.10},
        ),
        _make_advocate(
            1,
            ahe_outcome="confirmed",
            best_score=0.60,
            adjusted_score=0.60,
            aggregate_score=0.70,  # consistent across domains
            per_domain_scores={"cf": 0.60, "multifile": 0.80},
        ),
    ]
    picked = select_promotion_candidate(advocates)
    assert picked is not None
    assert picked.advocate_id == "advocate-1"
    assert picked.aggregate_score == 0.70


def test_partial_selected_when_no_confirmed() -> None:
    advocates = [
        _make_advocate(0, ahe_outcome="refuted", best_score=0.99),
        _make_advocate(1, ahe_outcome="partial", best_score=0.40),
        _make_advocate(2, ahe_outcome="partial", best_score=0.60, adjusted_score=0.65),
    ]
    picked = select_promotion_candidate(advocates)
    assert picked is not None
    assert picked.advocate_id == "advocate-2"


# --------------------------------------------------------------------------- #
# stage_promotion_from_population: gating
# --------------------------------------------------------------------------- #


def test_staged_decision_does_not_call_promote(
    workspace: Path, ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default path: candidate is logged, _promote is NOT called."""
    monkeypatch.delenv("AUTOBENCH_CONFIRM_PROMOTION", raising=False)
    monkeypatch.delenv("AUTOBENCH_REJECT_PROMOTION", raising=False)

    daemon = ContinuousModeDaemon(workspace=workspace)

    promote_calls: list[HarnessConfig] = []

    def _spy_promote(h: HarnessConfig) -> None:
        promote_calls.append(h)

    monkeypatch.setattr(daemon, "_promote", _spy_promote)

    advocates = [
        _make_advocate(0, ahe_outcome="confirmed", best_score=0.7),
    ]
    population_result = _make_population_result(advocates)

    decision = daemon.stage_promotion_from_population(
        population_result, ledger_path=ledger_path,
    )

    assert decision.decision == "staged"
    assert decision.decided_by == "default"
    assert decision.candidate_advocate_id == "advocate-0"
    assert decision.ahe_outcome == "confirmed"
    assert promote_calls == []


def test_accepted_decision_calls_promote(
    workspace: Path, ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """confirm=True → _promote fires; canonical harness is swapped on disk."""
    monkeypatch.delenv("AUTOBENCH_CONFIRM_PROMOTION", raising=False)
    monkeypatch.delenv("AUTOBENCH_REJECT_PROMOTION", raising=False)

    daemon = ContinuousModeDaemon(workspace=workspace)

    promote_calls: list[HarnessConfig] = []

    def _spy_promote(h: HarnessConfig) -> None:
        promote_calls.append(h)
        daemon.harness_path.write_text(
            json.dumps({"system_prompt": h.system_prompt}, indent=2)
        )

    monkeypatch.setattr(daemon, "_promote", _spy_promote)

    advocates = [
        _make_advocate(0, ahe_outcome="confirmed", best_score=0.7),
    ]
    population_result = _make_population_result(advocates)

    decision = daemon.stage_promotion_from_population(
        population_result, confirm=True, ledger_path=ledger_path,
    )

    assert decision.decision == "accepted"
    assert decision.decided_by == "cli"
    assert len(promote_calls) == 1
    assert promote_calls[0].system_prompt == "prompt-0"


def test_env_confirm_attributes_to_env(
    workspace: Path, ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUTOBENCH_CONFIRM_PROMOTION=1 → accepted/by-env."""
    monkeypatch.setenv("AUTOBENCH_CONFIRM_PROMOTION", "1")
    monkeypatch.delenv("AUTOBENCH_REJECT_PROMOTION", raising=False)

    daemon = ContinuousModeDaemon(workspace=workspace)
    monkeypatch.setattr(daemon, "_promote", lambda h: None)

    advocates = [_make_advocate(0, ahe_outcome="confirmed", best_score=0.7)]
    decision = daemon.stage_promotion_from_population(
        _make_population_result(advocates), ledger_path=ledger_path,
    )

    assert decision.decision == "accepted"
    assert decision.decided_by == "env"


def test_rejected_decision_does_not_call_promote(
    workspace: Path, ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reject=True → decision=rejected and _promote is NOT called."""
    monkeypatch.delenv("AUTOBENCH_CONFIRM_PROMOTION", raising=False)
    monkeypatch.delenv("AUTOBENCH_REJECT_PROMOTION", raising=False)

    daemon = ContinuousModeDaemon(workspace=workspace)
    promote_calls: list[HarnessConfig] = []
    monkeypatch.setattr(daemon, "_promote", lambda h: promote_calls.append(h))

    advocates = [_make_advocate(0, ahe_outcome="confirmed", best_score=0.7)]
    decision = daemon.stage_promotion_from_population(
        _make_population_result(advocates), reject=True, ledger_path=ledger_path,
    )

    assert decision.decision == "rejected"
    assert promote_calls == []


def test_no_promotable_candidate_auto_skips(
    workspace: Path, ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-refuted population → decision=staged, decided_by=auto-skip."""
    monkeypatch.delenv("AUTOBENCH_CONFIRM_PROMOTION", raising=False)
    monkeypatch.delenv("AUTOBENCH_REJECT_PROMOTION", raising=False)

    daemon = ContinuousModeDaemon(workspace=workspace)
    promote_calls: list[HarnessConfig] = []
    monkeypatch.setattr(daemon, "_promote", lambda h: promote_calls.append(h))

    advocates = [
        _make_advocate(0, ahe_outcome="refuted", best_score=0.7),
        _make_advocate(1, ahe_outcome="refuted_live", best_score=0.9),
        _make_advocate(2, ahe_outcome="none", best_score=0.5),
    ]
    decision = daemon.stage_promotion_from_population(
        _make_population_result(advocates),
        confirm=True,  # even with confirm, no candidate → auto-skip
        ledger_path=ledger_path,
    )

    assert decision.decision == "staged"
    assert decision.decided_by == "auto-skip"
    assert decision.candidate_advocate_id is None
    assert decision.ahe_outcome == "none"
    assert promote_calls == []


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_ledger_receives_entry_in_all_paths(
    workspace: Path, ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every decision path appends exactly one line to the ledger."""
    monkeypatch.delenv("AUTOBENCH_CONFIRM_PROMOTION", raising=False)
    monkeypatch.delenv("AUTOBENCH_REJECT_PROMOTION", raising=False)

    daemon = ContinuousModeDaemon(workspace=workspace)
    monkeypatch.setattr(daemon, "_promote", lambda h: None)

    confirmed = _make_population_result(
        [_make_advocate(0, ahe_outcome="confirmed", best_score=0.7)]
    )
    refuted = _make_population_result(
        [_make_advocate(0, ahe_outcome="refuted", best_score=0.9)]
    )

    # Path 1: stage-only
    daemon.stage_promotion_from_population(confirmed, ledger_path=ledger_path)
    # Path 2: accepted
    daemon.stage_promotion_from_population(
        confirmed, confirm=True, ledger_path=ledger_path,
    )
    # Path 3: rejected
    daemon.stage_promotion_from_population(
        confirmed, reject=True, ledger_path=ledger_path,
    )
    # Path 4: auto-skip
    daemon.stage_promotion_from_population(refuted, ledger_path=ledger_path)

    entries = _read_ledger(ledger_path)
    assert len(entries) == 4

    decisions = [e["decision"] for e in entries]
    deciders = [e["decided_by"] for e in entries]
    assert decisions == ["staged", "accepted", "rejected", "staged"]
    assert deciders == ["default", "cli", "cli", "auto-skip"]

    # All entries carry a timestamp.
    for e in entries:
        assert "ts" in e and e["ts"].endswith("Z")
        assert "cycle_id" in e
        assert "ahe_outcome" in e


def test_ledger_env_override(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUTOBENCH_PROMOTION_LEDGER env var routes ledger writes to that path."""
    target = tmp_path / "env_ledger.jsonl"
    monkeypatch.setenv("AUTOBENCH_PROMOTION_LEDGER", str(target))

    daemon = ContinuousModeDaemon(workspace=workspace)
    monkeypatch.setattr(daemon, "_promote", lambda h: None)

    advocates = [_make_advocate(0, ahe_outcome="confirmed", best_score=0.5)]
    daemon.stage_promotion_from_population(_make_population_result(advocates))

    assert target.exists()
    entries = _read_ledger(target)
    assert len(entries) == 1
    assert entries[0]["candidate_advocate_id"] == "advocate-0"


def test_resolve_ledger_path_default() -> None:
    """Default ledger path is tools/promotion_ledger.jsonl under repo root."""
    p = _resolve_promotion_ledger_path(None)
    assert p.name == "promotion_ledger.jsonl"
    assert p.parent.name == "tools"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_promotion_decision_schema_exists() -> None:
    schema_path = SCHEMA_DIR / "autobench.continuous.promotion_decision.v1.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text())
    assert schema["title"] == "autobench.continuous.promotion_decision v1"
    enum = schema["properties"]["data"]["properties"]["decision"]["enum"]
    assert set(enum) == {"staged", "accepted", "rejected"}


def test_emitted_event_matches_schema(
    workspace: Path, ledger_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bus event emitted by stage_promotion_from_population validates."""
    jsonschema = pytest.importorskip("jsonschema")

    schema_path = SCHEMA_DIR / "autobench.continuous.promotion_decision.v1.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    debug_file = tmp_path / "debug.jsonl"
    # Force fallback to debug-file emission (no zellij in PATH).
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    from autobench.observability import AutobenchObservability

    daemon = ContinuousModeDaemon(
        workspace=workspace,
        obs=AutobenchObservability(debug_file=debug_file),
    )
    monkeypatch.setattr(daemon, "_promote", lambda h: None)

    advocates = [_make_advocate(0, ahe_outcome="confirmed", best_score=0.5)]
    daemon.stage_promotion_from_population(
        _make_population_result(advocates), ledger_path=ledger_path,
    )

    assert debug_file.exists(), "expected an emitted event in the debug file"
    events = [json.loads(line) for line in debug_file.read_text().splitlines() if line.strip()]
    promotion_events = [
        e for e in events
        if e.get("type") == "autobench.continuous.promotion_decision.v1"
    ]
    assert promotion_events, "no promotion_decision event was emitted"
    for e in promotion_events:
        validator.validate(e)


# --------------------------------------------------------------------------- #
# PromotionDecision dataclass shape
# --------------------------------------------------------------------------- #


def test_promotion_decision_to_dicts() -> None:
    d = PromotionDecision(
        cycle_id="01HABCXYZ",
        candidate_advocate_id="advocate-2",
        candidate_session_id="SESS-X",
        candidate_score=0.71,
        candidate_adjusted_score=0.73,
        ahe_outcome="confirmed",
        decision="staged",
        decided_by="default",
        reason="awaiting approval",
    )
    ld = d.to_ledger_dict()
    assert ld["cycle_id"] == "01HABCXYZ"
    assert ld["decision"] == "staged"
    assert "ts" in ld and ld["ts"].endswith("Z")
    ed = d.to_event_data()
    assert "ts" not in ed  # event_data is the inner data block; ts lives in envelope
    assert ed["ahe_outcome"] == "confirmed"
