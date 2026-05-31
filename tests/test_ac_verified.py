"""Tests for the hearth-loom.ac.verified.v1 schema + Python emitter (nervous-bus-6kwv).

Covers:
    * Schema self-validates (Draft 2020-12).
    * Pass case (exit_code=0, with stdout) validates against the schema.
    * Fail case (exit_code=1, with stderr) validates.
    * Minimal case (only required fields) validates.
    * Truncation case: stdout > 4096 chars produces a [...truncated] marker
      and the resulting envelope still validates (i.e. stdout stays ≤ 4096).
    * Emitter ``ac_verified`` exists, publishes on the right channel, stamps
      session_id, and includes ts.
    * Malformed input (negative ac_index, empty bead_id) is silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from autobench.observability import (
    AutobenchObservability,
    CHANNEL_AC_VERIFIED,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "hearth-loom.ac.verified.v1.json"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Clean debug file + empty PATH so zellij pipe falls back to JSONL."""
    path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


# --------------------------------------------------------------------------- #
# Schema self-validation
# --------------------------------------------------------------------------- #

def test_schema_self_validates() -> None:
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    # Sanity: declared type/channel matches the filename.
    assert schema["properties"]["type"]["const"] == "hearth-loom.ac.verified.v1"


# --------------------------------------------------------------------------- #
# Sample-validate canonical payloads
# --------------------------------------------------------------------------- #

def _envelope(data: dict) -> dict:
    return {
        "specversion": "1.0",
        "id": "01HXAC0000000000000000000A",
        "source": "/hearth-loom/executor",
        "type": "hearth-loom.ac.verified.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-16T12:00:00.000Z",
        "data": data,
    }


def test_pass_case_validates() -> None:
    ev = _envelope({
        "bead_id": "nervous-bus-6kwv",
        "exec_id": "exec-abc",
        "ac_index": 0,
        "ac_text": "schema self-validates + CI schema-lint passes",
        "command": "python -c 'import jsonschema, json; jsonschema.Draft202012Validator.check_schema(json.load(open(\"schemas/x.json\")))'",
        "exit_code": 0,
        "duration_ms": 123,
        "ts": "2026-05-16T12:00:00.000Z",
        "stdout": "schema OK\n",
        "working_dir": "/shuttle",
    })
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)


def test_fail_case_validates() -> None:
    ev = _envelope({
        "bead_id": "nervous-bus-6kwv",
        "exec_id": "exec-abc",
        "ac_index": 1,
        "ac_text": "go build ./... clean",
        "command": "go build ./...",
        "exit_code": 1,
        "duration_ms": 4321,
        "ts": "2026-05-16T12:00:01.000Z",
        "stderr": "internal/bus/publisher.go:42:5: undefined: ACVerifiedEvent\n",
        "working_dir": "/shuttle",
    })
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)


def test_minimal_required_only_validates() -> None:
    ev = _envelope({
        "bead_id": "nervous-bus-6kwv",
        "exec_id": "exec-abc",
        "ac_index": 0,
        "ac_text": "ok",
        "command": "true",
        "exit_code": 0,
        "duration_ms": 0,
        "ts": "2026-05-16T12:00:00.000Z",
    })
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)


def test_truncated_stdout_validates() -> None:
    # Simulate a long stdout that the emitter would truncate to 4096 with
    # the [...truncated] marker. The marker must still fit within maxLength.
    body = "x" * (4096 - len("[...truncated]"))
    truncated = body + "[...truncated]"
    assert len(truncated) == 4096
    ev = _envelope({
        "bead_id": "nervous-bus-6kwv",
        "exec_id": "exec-abc",
        "ac_index": 2,
        "ac_text": "loud command",
        "command": "yes",
        "exit_code": 0,
        "duration_ms": 50,
        "ts": "2026-05-16T12:00:02.000Z",
        "stdout": truncated,
    })
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)


def test_missing_required_field_fails_validation() -> None:
    # ac_index missing — should NOT validate.
    ev = _envelope({
        "bead_id": "nervous-bus-6kwv",
        "exec_id": "exec-abc",
        "ac_text": "ok",
        "command": "true",
        "exit_code": 0,
        "duration_ms": 0,
        "ts": "2026-05-16T12:00:00.000Z",
    })
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load_schema()).validate(ev)


# --------------------------------------------------------------------------- #
# Emitter
# --------------------------------------------------------------------------- #

def test_emitter_publishes_validating_event(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.ac_verified(
        bead_id="nervous-bus-6kwv",
        exec_id="exec-abc",
        ac_index=0,
        ac_text="schema self-validates",
        command="python -m pytest autobench/tests/test_ac_verified.py -xvs",
        exit_code=0,
        duration_ms=1234,
        stdout="all green\n",
        stderr="",
        working_dir="/shuttle",
        correlation_id="01HXCORR000000000000000000",
    )
    events = _events_on(debug_file, CHANNEL_AC_VERIFIED)
    assert len(events) == 1
    ev = events[0]
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)
    data = ev["data"]
    assert data["bead_id"] == "nervous-bus-6kwv"
    assert data["exec_id"] == "exec-abc"
    assert data["ac_index"] == 0
    assert data["exit_code"] == 0
    assert data["duration_ms"] == 1234
    assert data["stdout"] == "all green\n"
    assert data["working_dir"] == "/shuttle"
    assert data["correlation_id"] == "01HXCORR000000000000000000"
    # Session id stamped on every event by the base publish path.
    assert data["session_id"] == obs.session_id
    # Envelope shape sanity.
    assert ev["type"] == CHANNEL_AC_VERIFIED
    assert ev["specversion"] == "1.0"


def test_emitter_truncates_oversize_stdout(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    big = "y" * 10_000
    obs.ac_verified(
        bead_id="nervous-bus-6kwv",
        exec_id="exec-abc",
        ac_index=3,
        ac_text="loud",
        command="yes",
        exit_code=0,
        duration_ms=10,
        stdout=big,
    )
    events = _events_on(debug_file, CHANNEL_AC_VERIFIED)
    assert len(events) == 1
    ev = events[0]
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)
    out = ev["data"]["stdout"]
    assert len(out) <= 4096
    assert out.endswith("[...truncated]")


def test_emitter_drops_malformed_silently(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    # Negative ac_index — silently dropped.
    obs.ac_verified(
        bead_id="nervous-bus-6kwv",
        exec_id="exec-abc",
        ac_index=-1,
        ac_text="bad",
        command="true",
        exit_code=0,
        duration_ms=0,
    )
    # Empty bead_id — silently dropped.
    obs.ac_verified(
        bead_id="",
        exec_id="exec-abc",
        ac_index=0,
        ac_text="bad",
        command="true",
        exit_code=0,
        duration_ms=0,
    )
    # Empty command — silently dropped.
    obs.ac_verified(
        bead_id="nervous-bus-6kwv",
        exec_id="exec-abc",
        ac_index=0,
        ac_text="bad",
        command="",
        exit_code=0,
        duration_ms=0,
    )
    events = _events_on(debug_file, CHANNEL_AC_VERIFIED)
    assert events == []


def test_emitter_never_raises_on_garbage_input(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    # Pass non-int duration; emitter coerces but must not raise.
    try:
        obs.ac_verified(
            bead_id="nervous-bus-6kwv",
            exec_id="exec-abc",
            ac_index=0,
            ac_text="ok",
            command="true",
            exit_code=0,
            duration_ms="not-an-int",  # type: ignore[arg-type]
        )
    except Exception as e:  # pragma: no cover
        pytest.fail(f"ac_verified raised on garbage input: {e}")
    # Whether or not it published, the test's contract is no-raise.
