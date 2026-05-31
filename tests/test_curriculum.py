"""Tests for autobench.curriculum.

Coverage:
    * analyze_history identifies failure patterns (large-array TLE, dp-deep-state, neg-edge)
    * analyze_history identifies mastery patterns
    * synthesize_problems builds a prompt that includes the failure categories
    * Parsing — malformed JSON, missing keys, non-array — all degrade gracefully
    * save_problems writes valid BenchmarkCase JSONL + a manifest
    * run_once end-to-end with a mocked LLM produces files + emits events
    * Emitted events validate against the v1 schemas
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autobench.curriculum import (
    CurriculumAgent,
    CurriculumGoals,
    CurriculumScheduler,
    GeneratedProblem,
    _build_synthesis_prompt,
    _parse_problems,
)
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_CURRICULUM_PROBLEM,
    CHANNEL_CURRICULUM_CYCLE,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force fallback-to-debug-file emission."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


# ---------------------------------------------------------------------------
# analyze_history
# ---------------------------------------------------------------------------


def _make_case(
    case_id: str,
    verdict: str,
    prompt: str = "",
    tags: list[str] | None = None,
    rating: int = 1200,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "verdict": verdict,
        "prompt": prompt,
        "metadata": {"tags": tags or [], "rating": rating},
    }


def test_analyze_history_detects_large_array_timeouts() -> None:
    """5/5 TLEs on N>=1e5 should produce 'large-array-timeouts' failure."""
    cases = [
        _make_case(f"c{i}", "TLE", prompt=f"Given an array of N <= 10^6 integers, count pairs (i,j) such that a_i + a_j = K.")
        for i in range(5)
    ]
    history = [{"case_results": cases, "session_id": "S1"}]

    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    goals = agent.analyze_history(history)

    assert "large-array-timeouts" in goals.failure_categories
    assert "5/5" in goals.evidence["large-array-timeouts"]
    assert goals.n_sessions_analyzed == 1


def test_analyze_history_detects_dp_deep_state() -> None:
    cases = [
        _make_case("c1", "WA", tags=["dp"], rating=1600),
        _make_case("c2", "RE", tags=["dp"], rating=1700),
        _make_case("c3", "OK", tags=["dp"], rating=1500),
    ]
    history = [{"case_results": cases}]
    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    goals = agent.analyze_history(history)
    assert "dp-deep-state" in goals.failure_categories


def test_analyze_history_detects_negative_number_edge() -> None:
    cases = [
        _make_case("c1", "WA", prompt="The input contains negative integers; output ..."),
        _make_case("c2", "WA", prompt="Note that values may be negative and can range from -1e9..1e9."),
        _make_case("c3", "OK", prompt="Plain positive integers only."),
    ]
    history = [{"case_results": cases}]
    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    goals = agent.analyze_history(history)
    assert "negative-number-edge" in goals.failure_categories


def test_analyze_history_detects_mastery() -> None:
    cases = [
        _make_case(f"c{i}", "OK", tags=["implementation"], rating=900) for i in range(4)
    ] + [
        _make_case("c4", "WA", tags=["implementation"], rating=900),
    ]
    history = [{"case_results": cases}]
    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    goals = agent.analyze_history(history)
    assert "implementation-easy" in goals.mastery_categories


def test_analyze_history_empty_input() -> None:
    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    goals = agent.analyze_history([])
    assert goals.failure_categories == []
    assert goals.mastery_categories == []
    assert goals.n_sessions_analyzed == 0


def test_analyze_history_accepts_cloudevents_envelope() -> None:
    """The flatten helper should unwrap CloudEvents-style envelopes."""
    cases = [_make_case(f"c{i}", "TLE", prompt="N <= 10^6") for i in range(3)]
    envelope = {
        "type": "autobench.result.v1",
        "data": {"case_results": cases, "session_id": "S2"},
    }
    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    goals = agent.analyze_history([envelope])
    assert goals.n_sessions_analyzed >= 1
    # The 3 TLE-on-large-N should trigger
    assert "large-array-timeouts" in goals.failure_categories


# ---------------------------------------------------------------------------
# synthesize_problems — prompt + parsing
# ---------------------------------------------------------------------------


def test_synthesize_prompt_template_includes_goals() -> None:
    goals = CurriculumGoals(
        failure_categories=["large-array-timeouts"],
        mastery_categories=["math-easy"],
        evidence={"large-array-timeouts": "5/5 TLEs at N>=1e5"},
    )
    prompt = _build_synthesis_prompt(goals, n=3)
    assert "large-array-timeouts" in prompt
    assert "math-easy" in prompt
    assert "5/5 TLEs at N>=1e5" in prompt
    assert "exactly 3" in prompt


def test_parse_problems_happy_path() -> None:
    raw = json.dumps([
        {
            "id": "curr-001",
            "prompt": "Sum two integers.",
            "expected_output": "5",
            "sample_input": "2 3",
            "difficulty_rating": 800,
            "target_skills": ["math"],
            "rationale": "Drills basic IO.",
        }
    ])
    probs, _ = _parse_problems(raw, generator_model="test")
    assert len(probs) == 1
    assert probs[0].id == "curr-001"
    assert probs[0].test_inputs == ["2 3"]
    assert probs[0].generator_model == "test"


def test_parse_problems_strips_codefence() -> None:
    raw = "```json\n" + json.dumps([{
        "id": "x", "prompt": "p", "expected_output": "o",
        "sample_input": "i", "difficulty_rating": 1000,
        "target_skills": [], "rationale": "",
    }]) + "\n```"
    probs, _ = _parse_problems(raw, generator_model="m")
    assert len(probs) == 1


def test_parse_problems_rejects_malformed_json() -> None:
    probs, _ = _parse_problems("not json {{", generator_model="m")
    assert probs == []


def test_parse_problems_rejects_non_array() -> None:
    probs, _ = _parse_problems('{"id": "x"}', generator_model="m")
    assert probs == []


def test_parse_problems_skips_rows_missing_keys() -> None:
    raw = json.dumps([
        {"id": "ok", "prompt": "p", "expected_output": "o",
         "sample_input": "i", "difficulty_rating": 1000,
         "target_skills": [], "rationale": ""},
        {"id": "missing-most"},  # malformed — should be skipped
    ])
    probs, _ = _parse_problems(raw, generator_model="m")
    assert len(probs) == 1
    assert probs[0].id == "ok"


def test_synthesize_problems_retries_on_bad_response() -> None:
    """If first call returns garbage, agent retries; second call returns valid JSON."""
    calls: list[int] = []

    def caller(_sys: str, _user: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            return "garbage not json"
        return json.dumps([{
            "id": "curr-001", "prompt": "P", "expected_output": "O",
            "sample_input": "I", "difficulty_rating": 1000,
            "target_skills": ["math"], "rationale": "drill",
        }])

    agent = CurriculumAgent(llm_caller=caller)
    goals = CurriculumGoals(failure_categories=["math"])
    probs = agent.synthesize_problems(goals, n=1, max_attempts=3)
    assert len(probs) == 1
    assert len(calls) == 2  # retried once


def test_synthesize_problems_returns_empty_on_total_failure() -> None:
    """Never raises, even when all attempts fail."""
    def caller(_sys: str, _user: str) -> str:
        raise RuntimeError("network down")

    agent = CurriculumAgent(llm_caller=caller)
    probs = agent.synthesize_problems(CurriculumGoals(), n=5, max_attempts=2)
    assert probs == []


# ---------------------------------------------------------------------------
# save_problems
# ---------------------------------------------------------------------------


def test_save_problems_writes_valid_jsonl(tmp_path: Path) -> None:
    problems = [
        GeneratedProblem(
            id="curr-1", prompt="A", expected_output="B",
            test_inputs=["in"], target_skills=["math"],
            difficulty_rating=1100, rationale="r", generator_model="test",
        ),
        GeneratedProblem(
            id="curr-2", prompt="C", expected_output="D",
            test_inputs=["in2"], target_skills=["dp"],
            difficulty_rating=1500, rationale="r2", generator_model="test",
        ),
    ]
    agent = CurriculumAgent(output_dir=tmp_path, llm_caller=lambda s, u: "[]")
    day_dir = agent.save_problems(problems, date="2026-05-16")

    assert day_dir.exists()
    cases = day_dir / "cases.jsonl"
    manifest = day_dir / "manifest.json"
    assert cases.exists()
    assert manifest.exists()

    rows = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    for r in rows:
        # Must match BenchmarkCase shape
        for key in ("id", "prompt", "expected_output", "test_inputs", "constraints", "metadata"):
            assert key in r
        assert r["metadata"]["source"] == "curriculum"

    mf = json.loads(manifest.read_text())
    assert mf["n_problems"] == 2
    assert mf["generator_model"] == "MiniMax-M2.7"  # default


def test_save_problems_emits_per_problem_events(tmp_path: Path, debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    agent = CurriculumAgent(output_dir=tmp_path, llm_caller=lambda s, u: "[]", obs=obs)
    problems = [
        GeneratedProblem(
            id="curr-1", prompt="A", expected_output="B",
            test_inputs=["in"], target_skills=["math"],
            difficulty_rating=1100, rationale="r", generator_model="test",
        ),
    ]
    agent.save_problems(problems, date="2026-05-16")

    evs = _events_on(debug_file, CHANNEL_CURRICULUM_PROBLEM)
    assert len(evs) == 1
    assert evs[0]["data"]["case_id"] == "curr-1"
    assert evs[0]["data"]["target_skills"] == ["math"]


# ---------------------------------------------------------------------------
# run_once — end-to-end
# ---------------------------------------------------------------------------


def test_run_once_end_to_end(tmp_path: Path, debug_file: Path) -> None:
    # Seed debug.jsonl with some "yesterday" session events
    session_jsonl = tmp_path / "session.jsonl"
    cases = [_make_case(f"c{i}", "TLE", prompt=f"N <= 10^6 array problem #{i}") for i in range(4)]
    session_event = {
        "specversion": "1.0", "id": "X", "source": "/autobench",
        "type": "autobench.result.v1", "datacontenttype": "application/json",
        "time": "2099-01-01T00:00:00",  # future date — never falls before cutoff
        "data": {"case_results": cases, "session_id": "S1"},
    }
    session_jsonl.write_text(json.dumps(session_event) + "\n")

    canned_response = json.dumps([
        {
            "id": "curr-001",
            "prompt": "Given an array of N=10^6 integers, ...",
            "expected_output": "42",
            "sample_input": "3\n1 2 3",
            "difficulty_rating": 1400,
            "target_skills": ["arrays", "large-N"],
            "rationale": "Drills large-array-timeouts failure pattern.",
        }
    ])

    obs = AutobenchObservability(debug_file=debug_file)
    agent = CurriculumAgent(
        output_dir=tmp_path / "out",
        llm_caller=lambda s, u: canned_response,
        obs=obs,
    )
    sched = CurriculumScheduler(
        agent,
        daily_at_hour=6,
        debug_jsonl=session_jsonl,
        n_problems=1,
        obs=obs,
    )
    summary = sched.run_once()

    assert summary["n_validated"] == 1
    out_dir = Path(summary["day_dir"])
    assert (out_dir / "cases.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    # cycle event emitted
    cycle_evs = _events_on(debug_file, CHANNEL_CURRICULUM_CYCLE)
    assert len(cycle_evs) == 1
    assert cycle_evs[0]["data"]["n_problems_validated"] == 1

    # per-problem event emitted
    prob_evs = _events_on(debug_file, CHANNEL_CURRICULUM_PROBLEM)
    assert len(prob_evs) == 1


# ---------------------------------------------------------------------------
# Scheduler: hour scheduling logic
# ---------------------------------------------------------------------------


def test_scheduler_hour_validation() -> None:
    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    with pytest.raises(ValueError):
        CurriculumScheduler(agent, daily_at_hour=24)
    with pytest.raises(ValueError):
        CurriculumScheduler(agent, daily_at_hour=-1)


def test_scheduler_next_run_seconds() -> None:
    import datetime as dt

    agent = CurriculumAgent(llm_caller=lambda s, u: "[]")
    sched = CurriculumScheduler(agent, daily_at_hour=6)
    # At 05:00 → should be ~3600s until next run
    now = dt.datetime(2026, 5, 16, 5, 0, 0)
    secs = sched._seconds_until_next_run(now=now)
    assert 3599 <= secs <= 3601

    # At 07:00 → should be ~23h until tomorrow's 06:00
    now = dt.datetime(2026, 5, 16, 7, 0, 0)
    secs = sched._seconds_until_next_run(now=now)
    assert 23 * 3600 - 1 <= secs <= 23 * 3600 + 1


# ---------------------------------------------------------------------------
# Schema validation of emitted events
# ---------------------------------------------------------------------------


def test_curriculum_events_validate_against_schema(tmp_path: Path, debug_file: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    obs = AutobenchObservability(debug_file=debug_file)
    obs.curriculum_problem_generated(
        case_id="curr-1",
        prompt="x" * 500,  # ensure trunc applies
        target_skills=["math", "dp"],
        difficulty_rating=1500,
        generator_model="MiniMax-M2.7",
        rationale="drills dp-deep-state",
        date="2026-05-16",
    )
    obs.curriculum_cycle_complete(
        cycle_id="cycle-1",
        n_problems_generated=10,
        n_problems_validated=9,
        n_problems_rejected=1,
        goals_summary={"failure_categories": ["dp-deep-state"], "mastery_categories": [], "n_sessions_analyzed": 5},
        generator_model="MiniMax-M2.7",
        date="2026-05-16",
    )

    prob_schema = json.loads((SCHEMA_DIR / "autobench.curriculum.problem.v1.json").read_text())
    cycle_schema = json.loads((SCHEMA_DIR / "autobench.curriculum.cycle.v1.json").read_text())

    prob_evs = _events_on(debug_file, CHANNEL_CURRICULUM_PROBLEM)
    cycle_evs = _events_on(debug_file, CHANNEL_CURRICULUM_CYCLE)
    assert prob_evs and cycle_evs

    jsonschema.Draft202012Validator(prob_schema).validate(prob_evs[0])
    jsonschema.Draft202012Validator(cycle_schema).validate(cycle_evs[0])

    # Truncation worked
    assert len(prob_evs[0]["data"]["prompt_preview"]) <= 280


# ---------------------------------------------------------------------------
# Multi-cycle save (nervous-bus-9obz)
# ---------------------------------------------------------------------------


def _canned(start: int = 0):
    """Build a canned LLM caller emitting n problems with unique ids."""
    def make(n: int = 2) -> str:
        return json.dumps([
            {
                "id": f"curr-{start + i:03d}",
                "prompt": f"Problem {start + i}: solve it.",
                "expected_output": f"{start + i}",
                "sample_input": f"input-{start + i}",
                "difficulty_rating": 1000 + i * 100,
                "target_skills": ["math"],
                "rationale": f"drills #{start + i}",
            }
            for i in range(n)
        ])
    return make


def test_save_problems_multi_cycle_preserves_all(tmp_path: Path) -> None:
    """Three back-to-back cycles on the same date keep every problem."""
    agent = CurriculumAgent(output_dir=tmp_path, llm_caller=lambda s, u: "[]")
    ds = "2026-05-18"

    all_ids: list[str] = []
    cycle_ids: list[str] = []
    for k in range(3):
        problems = [
            GeneratedProblem(
                id=f"curr-c{k}-{i}",
                prompt=f"P-{k}-{i}",
                expected_output=f"O-{k}-{i}",
                test_inputs=[f"in-{k}-{i}"],
                target_skills=["math"],
                difficulty_rating=1000 + k * 100,
                rationale=f"r{k}{i}",
                generator_model="test",
            )
            for i in range(2)
        ]
        all_ids.extend(p.id for p in problems)
        cid = f"curr-{ds}-{1700000000 + k}"
        cycle_ids.append(cid)
        agent.save_problems(problems, date=ds, cycle_id=cid)

    day_dir = tmp_path / ds
    # Per-cycle shards all present
    for cid in cycle_ids:
        shard = day_dir / "cycles" / cid
        assert (shard / "cases.jsonl").exists(), f"missing shard for {cid}"
        assert (shard / "manifest.json").exists()
        shard_rows = [
            json.loads(line)
            for line in (shard / "cases.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(shard_rows) == 2
        for r in shard_rows:
            assert r["metadata"]["cycle_id"] == cid
            assert r["metadata"]["cycle_date"] == ds

    # Daily roll-up contains every problem from every cycle
    rollup = day_dir / "cases.jsonl"
    assert rollup.exists()
    rollup_rows = [
        json.loads(line)
        for line in rollup.read_text().splitlines()
        if line.strip()
    ]
    assert len(rollup_rows) == 6
    assert sorted(r["id"] for r in rollup_rows) == sorted(all_ids)

    # Every row is attributable to one of the cycles
    seen_cycle_ids = {r["metadata"]["cycle_id"] for r in rollup_rows}
    assert seen_cycle_ids == set(cycle_ids)

    # Daily manifest indexes all cycles
    daily_mf = json.loads((day_dir / "manifest.json").read_text())
    assert daily_mf["n_problems"] == 6
    assert len(daily_mf["cycles"]) == 3
    indexed_ids = {c["cycle_id"] for c in daily_mf["cycles"]}
    assert indexed_ids == set(cycle_ids)
    # current_cycle_id mirrors the most-recently-written cycle
    assert daily_mf["current_cycle_id"] == cycle_ids[-1]


def test_save_problems_solo_cycle_back_compat(tmp_path: Path) -> None:
    """Calling save_problems without cycle_id still produces a shard + roll-up."""
    agent = CurriculumAgent(output_dir=tmp_path, llm_caller=lambda s, u: "[]")
    problems = [
        GeneratedProblem(
            id="curr-x",
            prompt="P",
            expected_output="O",
            test_inputs=["in"],
            target_skills=["math"],
            difficulty_rating=1000,
            rationale="r",
            generator_model="test",
        )
    ]
    day_dir = agent.save_problems(problems, date="2026-05-18")
    assert (day_dir / "cases.jsonl").exists()
    assert (day_dir / "manifest.json").exists()
    # Exactly one shard created (with a synthetic solo-* id)
    shards = list((day_dir / "cycles").iterdir())
    assert len(shards) == 1
    assert shards[0].name.startswith("solo-")


def test_run_once_emits_cycle_id_in_metadata(tmp_path: Path) -> None:
    """CurriculumScheduler.run_once stamps cycle_id on every saved row."""
    canned = json.dumps([
        {
            "id": "curr-001",
            "prompt": "P",
            "expected_output": "O",
            "sample_input": "in",
            "difficulty_rating": 1000,
            "target_skills": ["math"],
            "rationale": "r",
        }
    ])
    agent = CurriculumAgent(
        output_dir=tmp_path / "out",
        llm_caller=lambda s, u: canned,
    )
    sched = CurriculumScheduler(
        agent,
        debug_jsonl=tmp_path / "no-such.jsonl",
        n_problems=1,
    )
    summary = sched.run_once()
    cid = summary["cycle_id"]
    day_dir = Path(summary["day_dir"])
    shard = day_dir / "cycles" / cid
    assert shard.exists()

    rollup_rows = [
        json.loads(line)
        for line in (day_dir / "cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rollup_rows
    for r in rollup_rows:
        assert r["metadata"]["cycle_id"] == cid
