"""Tests for the autobench observability layer.

Verifies that:
    * AutobenchObservability instantiates without zellij in PATH (falls back to
      debug-file emission).
    * Each emission method writes a well-formed CloudEvents-lite envelope.
    * The ``phase`` context manager emits start + complete on success, and
      start + error on exception.
    * ``session_id`` is stable across all emissions from one instance.
    * Every emitted event validates against the corresponding JSON schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.observability import (
    AutobenchObservability,
    CHANNEL_PHASE,
    CHANNEL_ITERATION,
    CHANNEL_SANDBOX,
    CHANNEL_IMPROVER,
)



SCHEMA_FOR_CHANNEL = {
    CHANNEL_PHASE: SCHEMA_DIR / "autobench.phase.v1.json",
    CHANNEL_ITERATION: SCHEMA_DIR / "autobench.iteration.v1.json",
    CHANNEL_SANDBOX: SCHEMA_DIR / "autobench.sandbox.v1.json",
    CHANNEL_IMPROVER: SCHEMA_DIR / "autobench.improver.v1.json",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a clean debug file and force zellij-pipe failure so we always fall
    back to the debug-file path."""
    path = tmp_path / "debug.jsonl"
    # Push a fake empty PATH so `zellij` cannot be found.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


# --------------------------------------------------------------------------- #
# Construction / fallback
# --------------------------------------------------------------------------- #

def test_instantiates_without_zellij(debug_file: Path) -> None:
    """AutobenchObservability must not raise even when zellij is missing."""
    obs = AutobenchObservability(debug_file=debug_file)
    assert obs.session_id
    assert len(obs.session_id) == 26  # ULID length


def test_auto_session_id_is_unique() -> None:
    a = AutobenchObservability()
    b = AutobenchObservability()
    assert a.session_id != b.session_id


def test_explicit_session_id_preserved(debug_file: Path) -> None:
    obs = AutobenchObservability(session_id="MYSESSION", debug_file=debug_file)
    assert obs.session_id == "MYSESSION"


# --------------------------------------------------------------------------- #
# Envelope shape
# --------------------------------------------------------------------------- #

def test_phase_start_writes_valid_envelope(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.phase_start("benchmark", suite="codeforces-easy")
    events = _events_on(debug_file, CHANNEL_PHASE)
    assert len(events) == 1
    ev = events[0]
    for key in ("specversion", "id", "source", "type", "datacontenttype", "time", "data"):
        assert key in ev, f"missing key {key}"
    assert ev["specversion"] == "1.0"
    assert ev["type"] == CHANNEL_PHASE
    assert ev["datacontenttype"] == "application/json"
    assert ev["source"] == "/autobench"
    assert ev["data"]["phase"] == "benchmark"
    assert ev["data"]["status"] == "start"
    assert ev["data"]["session_id"] == obs.session_id
    assert ev["data"]["extra"]["suite"] == "codeforces-easy"


def test_phase_complete_records_duration(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.phase_complete("benchmark", duration_ms=123.4)
    ev = _events_on(debug_file, CHANNEL_PHASE)[-1]
    assert ev["data"]["status"] == "complete"
    assert ev["data"]["duration_ms"] == pytest.approx(123.4)


def test_phase_error_records_error(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.phase_error("benchmark", "boom")
    ev = _events_on(debug_file, CHANNEL_PHASE)[-1]
    assert ev["data"]["status"] == "error"
    assert ev["data"]["error"] == "boom"


def test_iteration_events(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.iteration_start(3, harness_version="v0.5")
    obs.iteration_complete(
        3,
        aggregate_score=0.79,
        verdict_counts={"OK": 14, "TLE": 3, "WA": 3},
        improvement_delta={"score_delta": 0.05},
        harness_version="v0.5",
    )
    evs = _events_on(debug_file, CHANNEL_ITERATION)
    assert len(evs) == 2
    assert evs[0]["data"]["status"] == "start"
    assert evs[1]["data"]["status"] == "complete"
    assert evs[1]["data"]["aggregate_score"] == 0.79
    assert evs[1]["data"]["verdict_counts"]["OK"] == 14
    assert evs[1]["data"]["improvement_delta"] == {"score_delta": 0.05}


def test_sandbox_events(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.sandbox_dispatch("cf-1042-A", "python", "gvisor")
    obs.sandbox_complete(
        "cf-1042-A",
        verdict="OK",
        latency_ms=234.0,
        exit_code=0,
        language="python",
        sandbox_type="gvisor",
    )
    evs = _events_on(debug_file, CHANNEL_SANDBOX)
    assert len(evs) == 2
    assert evs[0]["data"]["status"] == "dispatch"
    assert evs[1]["data"]["status"] == "complete"
    assert evs[1]["data"]["verdict"] == "OK"
    assert evs[1]["data"]["latency_ms"] == 234.0
    assert evs[1]["data"]["exit_code"] == 0


def test_improver_events(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.improver_call_start("minimax-m2.7", prompt_tokens=412)
    obs.improver_call_complete(
        "minimax-m2.7",
        completion_tokens=187,
        delta_summary="Adjusted prompt template + raised temperature",
    )
    evs = _events_on(debug_file, CHANNEL_IMPROVER)
    assert len(evs) == 2
    assert evs[0]["data"]["status"] == "start"
    assert evs[0]["data"]["prompt_tokens"] == 412
    assert evs[1]["data"]["status"] == "complete"
    assert evs[1]["data"]["completion_tokens"] == 187
    assert "Adjusted" in evs[1]["data"]["delta_summary"]


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #

def test_phase_context_manager_success(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    with obs.phase("benchmark", suite="x"):
        pass
    evs = _events_on(debug_file, CHANNEL_PHASE)
    assert len(evs) == 2
    assert evs[0]["data"]["status"] == "start"
    assert evs[1]["data"]["status"] == "complete"
    assert "duration_ms" in evs[1]["data"]


def test_phase_context_manager_error(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    with pytest.raises(ValueError):
        with obs.phase("benchmark"):
            raise ValueError("boom")
    evs = _events_on(debug_file, CHANNEL_PHASE)
    assert len(evs) == 2
    assert evs[0]["data"]["status"] == "start"
    assert evs[1]["data"]["status"] == "error"
    assert "boom" in evs[1]["data"]["error"]


# --------------------------------------------------------------------------- #
# Session id stability
# --------------------------------------------------------------------------- #

def test_session_id_stable_across_calls(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.phase_start("a")
    obs.iteration_start(0, harness_version="v0")
    obs.sandbox_dispatch("c1", "python", "gvisor")
    obs.improver_call_start("m", 1)
    all_events = _read_events(debug_file)
    session_ids = {e["data"]["session_id"] for e in all_events}
    assert session_ids == {obs.session_id}


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

def test_all_schemas_load() -> None:
    """Each schema file is valid JSON Schema draft 2020-12."""
    jsonschema = pytest.importorskip("jsonschema")
    for channel, path in SCHEMA_FOR_CHANNEL.items():
        assert path.exists(), f"missing schema: {path}"
        schema = json.loads(path.read_text())
        # Smoke-check: validator construction succeeds
        jsonschema.Draft202012Validator.check_schema(schema)


def test_emitted_events_match_schema(debug_file: Path) -> None:
    """Every emission produces an event that validates against its schema."""
    jsonschema = pytest.importorskip("jsonschema")

    obs = AutobenchObservability(debug_file=debug_file)
    obs.phase_start("benchmark", suite="x")
    obs.phase_complete("benchmark", duration_ms=1.0)
    obs.phase_error("benchmark", "boom")
    obs.iteration_start(0, harness_version="v0")
    obs.iteration_complete(
        0,
        aggregate_score=0.5,
        verdict_counts={"OK": 1},
        improvement_delta=None,
        harness_version="v0",
    )
    obs.sandbox_dispatch("c", "python", "gvisor")
    obs.sandbox_complete("c", verdict="OK", latency_ms=10.0, exit_code=0,
                         language="python", sandbox_type="gvisor")
    obs.improver_call_start("m", 1)
    obs.improver_call_complete("m", 2, "delta")

    validators = {
        channel: jsonschema.Draft202012Validator(json.loads(path.read_text()))
        for channel, path in SCHEMA_FOR_CHANNEL.items()
    }

    events = _read_events(debug_file)
    assert events, "expected emissions to write events"
    for ev in events:
        channel = ev["type"]
        validator = validators.get(channel)
        assert validator is not None, f"no validator for {channel}"
        errors = sorted(validator.iter_errors(ev), key=lambda e: list(e.path))
        assert not errors, f"event failed schema {channel}: {[e.message for e in errors]}"


# --------------------------------------------------------------------------- #
# Never-raises invariant
# --------------------------------------------------------------------------- #

def test_publish_never_raises_on_bad_debug_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if the debug-file path is unwritable, emission must not raise."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    bad_path = tmp_path / "nonexistent" / "no" / "perm" / "x.jsonl"
    obs = AutobenchObservability(debug_file=bad_path)
    # Make it actually unwritable: create the file then chmod 0
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text("")
    os.chmod(bad_path, 0o000)
    try:
        obs.phase_start("safe")  # must not raise
    finally:
        os.chmod(bad_path, 0o644)
