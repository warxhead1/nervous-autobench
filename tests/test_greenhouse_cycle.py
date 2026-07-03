"""Tests for greenhouse.cycle — goal selection, budget gating, and the dry-run pipeline.

No LLM calls, no network: dry-run substitutes a fixture results file for the
real kernel run. AUTOBENCH_OBS_DISABLE_PIPE=1 keeps bus.publish from forking
the `nervous` CLI; envelopes are inspected via debug_path/monkeypatch instead.
"""
from __future__ import annotations

import json

import jsonschema
import pytest

from greenhouse import bus, cycle, export
from greenhouse.goals import parse_manifest
from greenhouse.ledger import Ledger
from tests._paths import SCHEMA_DIR

_MANIFEST_RAW = {
    "version": 1,
    "budget": {"window_seconds": 18000, "window_max_requests": 1500, "per_cycle_max_requests": 300},
    "goals": [
        {"id": "terrain-goal", "domain": "terrain", "instances": ["rolling_hills"], "priority": 3, "want": 2},
        {"id": "noise-goal", "domain": "noise", "instances": ["perlin_like"], "priority": 1, "want": 2},
        {"id": "sdf-goal", "domain": "sdf", "instances": ["gyroid"], "priority": 1, "want": 1},
    ],
}


def _manifest():
    return parse_manifest(_MANIFEST_RAW)


@pytest.fixture(autouse=True)
def _disable_live_pipe(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")


# --------------------------------------------------------------------------- #
# select_goal
# --------------------------------------------------------------------------- #


def test_select_goal_skips_satisfied_want(tmp_path):
    manifest = _manifest()
    # Satisfy sdf-goal's want=1 by writing one drop for it.
    program = json.loads((cycle._FIXTURES_DIR / "dry_run_sdf.json").read_text())["top_programs"][0]
    export.export_candidate(domain="sdf", goal_id="sdf-goal", goal_notes="", goal_tags=[],
                             instance="gyroid", program=program, run_id="r0", drops_root=tmp_path)
    chosen_ids = {cycle.select_goal(manifest, tmp_path).id for _ in range(1)}
    assert "sdf-goal" not in chosen_ids


def test_select_goal_returns_none_when_all_satisfied(tmp_path):
    manifest = _manifest()
    for goal in manifest.goals:
        fixture_name = {"terrain": "dry_run_terrain.json", "noise": "dry_run_noise.json", "sdf": "dry_run_sdf.json"}[goal.domain]
        program = json.loads((cycle._FIXTURES_DIR / fixture_name).read_text())["top_programs"][0]
        for i in range(goal.want):
            export.export_candidate(domain=goal.domain, goal_id=goal.id, goal_notes="", goal_tags=[],
                                      instance=goal.instances[0], program={**program, "id": f"{program['id']}-{i}"},
                                      run_id=f"r{i}", drops_root=tmp_path)
    assert cycle.select_goal(manifest, tmp_path) is None


def test_select_goal_priority_weighted_round_robin(tmp_path):
    """A priority-3 goal should be selected roughly 3x as often as a priority-1 goal."""
    manifest = _manifest()
    picks = []
    for _ in range(8):
        goal = cycle.select_goal(manifest, tmp_path)
        if goal is None:
            break
        picks.append(goal.id)
        # Simulate a drop landing for whichever goal was picked.
        fixture_name = {"terrain": "dry_run_terrain.json", "noise": "dry_run_noise.json", "sdf": "dry_run_sdf.json"}[goal.domain]
        program = json.loads((cycle._FIXTURES_DIR / fixture_name).read_text())["top_programs"][0]
        export.export_candidate(domain=goal.domain, goal_id=goal.id, goal_notes="", goal_tags=[],
                                  instance=goal.instances[0], program={**program, "id": f"{program['id']}-{len(picks)}"},
                                  run_id=f"r{len(picks)}", drops_root=tmp_path)
    # terrain-goal (priority 3, want 2) should be exhausted well before the
    # priority-1 goals get their full share.
    assert picks[0] == "terrain-goal"
    assert picks.count("terrain-goal") == 2  # want satisfied, no more picks after


# --------------------------------------------------------------------------- #
# run_cycle: skip paths
# --------------------------------------------------------------------------- #


def test_run_cycle_skips_when_budget_exhausted(tmp_path):
    manifest = _manifest()
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    now = 1_000_000.0
    ledger.record(run_id="prev", goal_id="terrain-goal", requests=1490, ts=now - 10)  # leaves < MIN_CYCLE_REQUESTS
    result = cycle.run_cycle(
        dry_run=True, manifest=manifest, ledger=ledger, drops_root=tmp_path / "drops",
        lock_path=tmp_path / "cycle.lock", now=now,
    )
    assert result.skipped is True
    assert result.stop_reason == "budget_exhausted"


@pytest.mark.parametrize("scenario", ["lock_held", "budget_exhausted", "invalid_manifest"])
def test_cycle_completed_always_emitted_on_skip(scenario, tmp_path, monkeypatch):
    """The spec requires greenhouse.cycle.completed.v1 on EVERY outcome, not just successful runs."""
    debug_path = tmp_path / "debug.jsonl"
    monkeypatch.setattr(bus, "DEFAULT_DEBUG_PATH", debug_path)
    lock_path = tmp_path / "cycle.lock"
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    manifest = _manifest()

    if scenario == "lock_held":
        import fcntl
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = cycle.run_cycle(dry_run=True, manifest=manifest, ledger=ledger,
                                      drops_root=tmp_path / "drops", lock_path=lock_path)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
    elif scenario == "budget_exhausted":
        now = 1_000_000.0
        ledger.record(run_id="prev", goal_id="terrain-goal", requests=1490, ts=now - 10)
        result = cycle.run_cycle(dry_run=True, manifest=manifest, ledger=ledger,
                                  drops_root=tmp_path / "drops", lock_path=lock_path, now=now)
    else:
        monkeypatch.setenv("GREENHOUSE_GOALS", str(tmp_path / "does-not-exist.json"))
        result = cycle.run_cycle(dry_run=True, manifest=None, ledger=ledger,
                                  drops_root=tmp_path / "drops", lock_path=lock_path)

    assert result.skipped is True
    lines = [json.loads(l) for l in debug_path.read_text().splitlines() if l.strip()]
    completed = [e for e in lines if e["type"] == "greenhouse.cycle.completed.v1"]
    assert len(completed) == 1
    assert completed[0]["data"]["skipped"] is True

    schema = json.loads((SCHEMA_DIR / "greenhouse.cycle.completed.v1.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(completed[0])


def test_run_cycle_skips_when_lock_held(tmp_path):
    import fcntl
    manifest = _manifest()
    lock_path = tmp_path / "cycle.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = cycle.run_cycle(
            dry_run=True, manifest=manifest, ledger=Ledger(path=tmp_path / "ledger.jsonl"),
            drops_root=tmp_path / "drops", lock_path=lock_path,
        )
        assert result.skipped is True
        assert result.stop_reason == "lock_held"
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def test_run_cycle_skips_with_invalid_goals_manifest(tmp_path, monkeypatch):
    # Point GREENHOUSE_GOALS at a file that doesn't exist so this is hermetic
    # regardless of what's actually installed at the default manifest path.
    monkeypatch.setenv("GREENHOUSE_GOALS", str(tmp_path / "does-not-exist.json"))
    result = cycle.run_cycle(
        dry_run=True, manifest=None, ledger=Ledger(path=tmp_path / "ledger.jsonl"),
        drops_root=tmp_path / "drops", lock_path=tmp_path / "cycle.lock",
    )
    assert result.skipped is True
    assert "goals_manifest_error" in result.stop_reason


# --------------------------------------------------------------------------- #
# run_cycle: dry-run happy path, end to end
# --------------------------------------------------------------------------- #


def test_dry_run_cycle_exports_and_emits(tmp_path, monkeypatch):
    debug_path = tmp_path / "debug.jsonl"
    monkeypatch.setattr(bus, "DEFAULT_DEBUG_PATH", debug_path)

    manifest = _manifest()
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    result = cycle.run_cycle(
        dry_run=True, manifest=manifest, ledger=ledger, drops_root=tmp_path / "drops",
        lock_path=tmp_path / "cycle.lock", now=1_000_000.0,
    )

    assert result.skipped is False
    assert result.goal_id == "terrain-goal"  # highest priority, nothing dropped yet
    assert result.domain == "terrain"
    assert result.candidates_dropped == 1
    assert len(result.drop_paths) == 1
    assert result.drop_paths[0].is_file()
    assert result.llm_requests == 24  # from the terrain fixture
    assert result.window_requests_used == 24
    assert result.window_requests_budget == 1500

    # Ledger recorded the spend.
    assert ledger.window_used(18000, now=1_000_001.0) == 24

    # Bus emissions: exactly one candidate.ready + one cycle.completed, both schema-valid.
    lines = [json.loads(l) for l in debug_path.read_text().splitlines() if l.strip()]
    types = [e["type"] for e in lines]
    assert types == ["greenhouse.candidate.ready.v1", "greenhouse.cycle.completed.v1"]

    schemas = {
        "greenhouse.candidate.ready.v1": json.loads((SCHEMA_DIR / "greenhouse.candidate.ready.v1.json").read_text()),
        "greenhouse.cycle.completed.v1": json.loads((SCHEMA_DIR / "greenhouse.cycle.completed.v1.json").read_text()),
    }
    for envelope in lines:
        jsonschema.Draft202012Validator(schemas[envelope["type"]]).validate(envelope)


def test_dry_run_cycle_completed_always_emitted_even_when_export_fails(tmp_path, monkeypatch):
    """sdf-goal's fixture includes a GPU-incompatible candidate; cycle.completed must still fire."""
    debug_path = tmp_path / "debug.jsonl"
    monkeypatch.setattr(bus, "DEFAULT_DEBUG_PATH", debug_path)

    manifest_raw = json.loads(json.dumps(_MANIFEST_RAW))
    manifest_raw["goals"] = [g for g in manifest_raw["goals"] if g["domain"] == "sdf"]
    manifest_raw["goals"][0]["want"] = 2  # both fixture candidates would be attempted
    manifest = parse_manifest(manifest_raw)

    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    result = cycle.run_cycle(
        dry_run=True, manifest=manifest, ledger=ledger, drops_root=tmp_path / "drops",
        lock_path=tmp_path / "cycle.lock", now=2_000_000.0,
    )
    assert result.skipped is False
    assert result.candidates_dropped == 1  # only the valid sphere, not the static-array one
    assert result.export_errors  # the GPU-incompatible candidate is reported, not silently dropped

    lines = [json.loads(l) for l in debug_path.read_text().splitlines() if l.strip()]
    completed = [e for e in lines if e["type"] == "greenhouse.cycle.completed.v1"]
    assert len(completed) == 1
    assert completed[0]["data"]["candidates_dropped"] == 1


def test_dry_run_second_cycle_moves_to_next_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(bus, "DEFAULT_DEBUG_PATH", tmp_path / "debug.jsonl")
    manifest = _manifest()
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    drops_root = tmp_path / "drops"

    manifest_one_goal = parse_manifest({
        **_MANIFEST_RAW,
        "goals": [{"id": "terrain-goal", "domain": "terrain", "instances": ["rolling_hills"], "priority": 3, "want": 1}],
    })
    first = cycle.run_cycle(dry_run=True, manifest=manifest_one_goal, ledger=ledger, drops_root=drops_root,
                             lock_path=tmp_path / "cycle.lock", now=1_000_000.0)
    assert first.goal_id == "terrain-goal"

    second = cycle.run_cycle(dry_run=True, manifest=manifest, ledger=ledger, drops_root=drops_root,
                              lock_path=tmp_path / "cycle.lock", now=1_000_100.0)
    # terrain-goal's want (1) is now satisfied by the first cycle's drop -> selection moves on.
    assert second.goal_id != "terrain-goal"
