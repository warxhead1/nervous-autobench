"""Tests for bus.notify.v1 — schema + emitter (nervous-bus-ibkg).

Covers:
    * schema self-validates under Draft 2020-12.
    * Minimal + full sample events validate against the schema.
    * Required-field omission is caught by the schema.
    * AutobenchObservability.bus_notify exists and emits a well-formed envelope.
    * Malformed inputs (bad enum, oversize summary) are silently dropped — the
      emitter never raises and does not pollute the bus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.observability import (
    AutobenchObservability,
    CHANNEL_BUS_NOTIFY,
)


from tests._paths import SCHEMA_DIR
SCHEMA_PATH = SCHEMA_DIR / "bus.notify.v1.json"


# --------------------------------------------------------------------------- #
# Helpers — mirror test_observability.py
# --------------------------------------------------------------------------- #


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Clean debug file + force zellij-pipe failure so we always fall back
    to the debug-file path."""
    path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


# --------------------------------------------------------------------------- #
# Schema-level tests
# --------------------------------------------------------------------------- #


def test_schema_self_validates() -> None:
    """bus.notify.v1.json must be a valid Draft 2020-12 schema."""
    import jsonschema

    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_schema_has_required_metadata() -> None:
    schema = _load_schema()
    assert schema["$id"].endswith("bus.notify.v1.json")
    assert schema["$schema"].startswith("https://json-schema.org/draft/2020-12")
    assert "bus.notify" in schema["title"]
    assert schema["description"]


def test_minimal_event_validates() -> None:
    import jsonschema

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    event = {
        "specversion": "1.0",
        "id": "01JBVQ0DC9YBHJ8KCM2PXWFR3T",
        "source": "/autobench",
        "type": "bus.notify.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-17T12:00:00Z",
        "data": {
            "priority": "info",
            "channels": ["phone"],
            "summary": "autobench cycle complete",
            "source_project": "autobench",
            "ts": "2026-05-17T12:00:00Z",
        },
    }
    errors = list(validator.iter_errors(event))
    assert not errors, errors


def test_full_event_validates() -> None:
    import jsonschema

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    event = {
        "specversion": "1.0",
        "id": "01JBVQ0DC9YBHJ8KCM2PXWFR3U",
        "source": "/autobench",
        "type": "bus.notify.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-17T12:00:00Z",
        "data": {
            "priority": "critical",
            "channels": ["phone", "discord", "ntfy", "session://eric-pane-3"],
            "summary": "harness promotion landed — phone tap to inspect",
            "source_project": "autobench",
            "ts": "2026-05-17T12:00:00Z",
            "correlation_id": "01JBVQ0DC9YBHJ8KCM2PXWFR3T",
            "body": "baseline 0.42 -> treatment 0.55 (delta=0.13, n=42).",
            "deep_link": "https://github.com/warxhead1/nervous-bus/pull/123",
            "source_event_type": "bus.bead.bench_completed.v1",
            "dedup_key": "autobench:promotion:cycle-ABC",
        },
    }
    errors = list(validator.iter_errors(event))
    assert not errors, errors


def test_schema_rejects_bad_priority() -> None:
    import jsonschema

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    event = {
        "specversion": "1.0",
        "id": "x",
        "source": "/x",
        "type": "bus.notify.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-17T12:00:00Z",
        "data": {
            "priority": "urgent",  # not in enum
            "channels": ["phone"],
            "summary": "x",
            "source_project": "a",
            "ts": "2026-05-17T12:00:00Z",
        },
    }
    assert list(validator.iter_errors(event))


def test_schema_rejects_empty_channels() -> None:
    import jsonschema

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    event = {
        "specversion": "1.0",
        "id": "x",
        "source": "/x",
        "type": "bus.notify.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-17T12:00:00Z",
        "data": {
            "priority": "info",
            "channels": [],
            "summary": "x",
            "source_project": "a",
            "ts": "2026-05-17T12:00:00Z",
        },
    }
    assert list(validator.iter_errors(event))


def test_schema_rejects_oversize_summary() -> None:
    import jsonschema

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    event = {
        "specversion": "1.0",
        "id": "x",
        "source": "/x",
        "type": "bus.notify.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-17T12:00:00Z",
        "data": {
            "priority": "info",
            "channels": ["phone"],
            "summary": "x" * 141,  # 141 > 140 max
            "source_project": "a",
            "ts": "2026-05-17T12:00:00Z",
        },
    }
    assert list(validator.iter_errors(event))


def test_schema_rejects_missing_required() -> None:
    import jsonschema

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    # Missing `ts` from data
    event = {
        "specversion": "1.0",
        "id": "x",
        "source": "/x",
        "type": "bus.notify.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-17T12:00:00Z",
        "data": {
            "priority": "info",
            "channels": ["phone"],
            "summary": "x",
            "source_project": "a",
        },
    }
    assert list(validator.iter_errors(event))


def test_schema_accepts_session_channel() -> None:
    import jsonschema

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    event = {
        "specversion": "1.0",
        "id": "x",
        "source": "/x",
        "type": "bus.notify.v1",
        "datacontenttype": "application/json",
        "time": "2026-05-17T12:00:00Z",
        "data": {
            "priority": "info",
            "channels": ["session://eric-pane-3"],
            "summary": "peer ping",
            "source_project": "deer-flow",
            "ts": "2026-05-17T12:00:00Z",
        },
    }
    errors = list(validator.iter_errors(event))
    assert not errors, errors


# --------------------------------------------------------------------------- #
# Emitter tests
# --------------------------------------------------------------------------- #


def test_emitter_method_exists() -> None:
    obs = AutobenchObservability()
    assert callable(getattr(obs, "bus_notify", None))


def test_emitter_writes_valid_envelope(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.bus_notify(
        priority="info",
        channels=["phone"],
        summary="autobench cycle complete",
        source_project="autobench",
    )
    events = _read_events(debug_file)
    notify_events = [e for e in events if e.get("type") == CHANNEL_BUS_NOTIFY]
    assert len(notify_events) == 1
    ev = notify_events[0]
    assert ev["specversion"] == "1.0"
    assert ev["type"] == "bus.notify.v1"
    assert ev["data"]["priority"] == "info"
    assert ev["data"]["channels"] == ["phone"]
    assert ev["data"]["summary"] == "autobench cycle complete"
    assert ev["data"]["source_project"] == "autobench"
    assert ev["data"]["ts"]


def test_emitter_with_all_optionals(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.bus_notify(
        priority="critical",
        channels=["phone", "discord", "session://eric-pane-3"],
        summary="harness promotion landed",
        source_project="autobench",
        correlation_id="01JBVQ0DC9YBHJ8KCM2PXWFR3T",
        body="baseline 0.42 -> treatment 0.55",
        deep_link="https://github.com/warxhead1/nervous-bus/pull/123",
        source_event_type="bus.bead.bench_completed.v1",
        dedup_key="autobench:promotion:cycle-ABC",
    )
    events = _read_events(debug_file)
    notify_events = [e for e in events if e.get("type") == CHANNEL_BUS_NOTIFY]
    assert len(notify_events) == 1
    data = notify_events[0]["data"]
    assert data["priority"] == "critical"
    assert data["correlation_id"] == "01JBVQ0DC9YBHJ8KCM2PXWFR3T"
    assert data["body"] == "baseline 0.42 -> treatment 0.55"
    assert data["deep_link"].startswith("https://")
    assert data["source_event_type"] == "bus.bead.bench_completed.v1"
    assert data["dedup_key"] == "autobench:promotion:cycle-ABC"
    assert "session://eric-pane-3" in data["channels"]


def test_emitter_drops_malformed_priority(debug_file: Path) -> None:
    """Bad priority enum value should be silently dropped — no emit."""
    obs = AutobenchObservability(debug_file=debug_file)
    obs.bus_notify(
        priority="urgent",  # not in enum
        channels=["phone"],
        summary="x",
        source_project="autobench",
    )
    events = _read_events(debug_file)
    notify_events = [e for e in events if e.get("type") == CHANNEL_BUS_NOTIFY]
    assert notify_events == []


def test_emitter_drops_oversize_summary(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.bus_notify(
        priority="info",
        channels=["phone"],
        summary="x" * 200,
        source_project="autobench",
    )
    events = _read_events(debug_file)
    notify_events = [e for e in events if e.get("type") == CHANNEL_BUS_NOTIFY]
    assert notify_events == []


def test_emitter_drops_empty_channels(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.bus_notify(
        priority="info",
        channels=[],
        summary="x",
        source_project="autobench",
    )
    events = _read_events(debug_file)
    notify_events = [e for e in events if e.get("type") == CHANNEL_BUS_NOTIFY]
    assert notify_events == []


def test_emitter_never_raises(debug_file: Path) -> None:
    """Even with utterly garbage input, the emitter must not raise."""
    obs = AutobenchObservability(debug_file=debug_file)
    # None values for required strings — coerced to "None" string by str(),
    # which then fails schema validation and gets dropped. Must not raise.
    obs.bus_notify(
        priority=None,  # type: ignore[arg-type]
        channels=None,  # type: ignore[arg-type]
        summary=None,  # type: ignore[arg-type]
        source_project=None,  # type: ignore[arg-type]
    )
