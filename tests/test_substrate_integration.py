"""End-to-end substrate integration tests (nervous-bus-1hlf).

Pipes a synthetic ``autobench.cycle.requested.v1`` event into TriggerDaemon
via the mock_subprocess path, runs a tiny stubbed cycle, and asserts the
resulting ``autobench.cycle.report.v1`` is consistent with the request.
"""

from __future__ import annotations

import json
from pathlib import Path

from autobench.observability import (
    CHANNEL_CYCLE_REPORT,
    CHANNEL_CYCLE_REQUESTED,
    AutobenchObservability,
)
from autobench.trigger_daemon import TriggerDaemon


VALID_ULID = "01SBSTRATE00000000000000XX"


def _trigger_envelope(**overrides) -> dict:
    data = {
        "correlation_id": VALID_ULID,
        "requested_by": "hearth-loom",
        "domain": "codeforces_tier1",
        "ts": "2026-05-16T00:00:00Z",
    }
    data.update(overrides)
    return {
        "specversion": "1.0",
        "id": "01TRIGGEREVENT000000000000",
        "source": "/hearth-loom",
        "type": CHANNEL_CYCLE_REQUESTED,
        "datacontenttype": "application/json",
        "time": "2026-05-16T00:00:00Z",
        "data": data,
    }


def _tiny_case_result_events() -> list[dict]:
    """A handful of synthetic case.result events the stub cycle "produced"."""
    return [
        {
            "specversion": "1.0",
            "id": f"01CASE{i:020d}",
            "source": "/autobench",
            "type": "autobench.case.result.v1",
            "datacontenttype": "application/json",
            "time": "2026-05-16T00:10:00Z",
            "data": {
                "case_id": f"c{i}",
                "iteration": 0,
                "language": "python",
                "verdict": "OK" if i % 2 == 0 else "WA",
                "p_score": 1.0 if i % 2 == 0 else 0.0,
                "latency_ms": 50.0,
                "generated_code": "print()",
                "generated_code_length": 7,
                "attempt": 1,
            },
        }
        for i in range(4)
    ]


def test_end_to_end_request_to_report(tmp_path):
    """Mock subprocess → trigger daemon → tiny cycle → report consistent with request."""
    debug = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug)
    obs._pipe_disabled = True

    case_events = _tiny_case_result_events()

    def _stub_runner(config):
        # n_advocates=1, max_iter=1 — the smallest legal cycle.
        assert config.n_advocates == 1
        assert config.max_iter == 1
        assert config.domain == "codeforces_tier1"
        assert config.correlation_id == VALID_ULID
        assert config.bead_id == "test-bead-1hlf"
        return (
            case_events,
            {
                "cycle_id": "01TESTCYCLE0000000000000XX",
                "n_advocates_hint": 1,
                "n_cases_hint": 4,
                "baseline_score": 0.0,
            },
            "2026-05-16T00:00:00Z",
            "2026-05-16T00:05:00Z",
        )

    daemon = TriggerDaemon(obs=obs, runner_factory=_stub_runner)

    event = _trigger_envelope(
        bead_id="test-bead-1hlf",
        n_advocates=1,
        max_iter=1,
    )
    handled = daemon.listen(mock_subprocess=[json.dumps(event)], max_runs=1)
    assert handled == 1

    # Find the cycle.report event in the debug file.
    reports: list[dict] = []
    for line in debug.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == CHANNEL_CYCLE_REPORT:
            reports.append(ev)
    assert len(reports) == 1
    report_envelope = reports[0]
    data = report_envelope["data"]

    # Trace-correlation assertions: every routing field flows through.
    assert data["correlation_id"] == VALID_ULID
    assert data["domain"] == "codeforces_tier1"
    assert data["requested_by"] == "hearth-loom"
    assert data["bead_id"] == "test-bead-1hlf"
    assert data["cycle_id"] == "01TESTCYCLE0000000000000XX"
    assert data["started_at"] == "2026-05-16T00:00:00Z"
    assert data["completed_at"] == "2026-05-16T00:05:00Z"
    assert data["ts"] == data["completed_at"]

    # The four synthetic case events should have produced one failure mode
    # (two WAs) and four cases total.
    assert data["summary"]["n_cases"] == 4
    wa_modes = [m for m in data["patterns"]["top_failure_modes"] if m["failure_mode"] == "WA"]
    assert wa_modes and wa_modes[0]["count"] == 2

    # Envelope-level fields are well-formed.
    assert report_envelope["type"] == CHANNEL_CYCLE_REPORT
    assert report_envelope["specversion"] == "1.0"
    assert report_envelope["datacontenttype"] == "application/json"


def test_end_to_end_report_validates_against_schema(tmp_path):
    """The emitted report's data payload validates against the v1 schema."""
    import jsonschema

    debug = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug)
    obs._pipe_disabled = True

    def _stub_runner(config):
        return ([], {"cycle_id": "01CYC1000000000000000000XX"}, "2026-05-16T00:00:00Z", "2026-05-16T00:05:00Z")

    daemon = TriggerDaemon(obs=obs, runner_factory=_stub_runner)
    event = _trigger_envelope()
    daemon.listen(mock_subprocess=[json.dumps(event)], max_runs=1)

    reports = []
    for line in debug.read_text().splitlines():
        ev = json.loads(line) if line.strip() else None
        if ev and ev.get("type") == CHANNEL_CYCLE_REPORT:
            reports.append(ev)
    assert reports

    repo_root = Path(__file__).resolve().parents[2]
    schema = json.loads((repo_root / "schemas" / "autobench.cycle.report.v1.json").read_text())
    data_schema = schema["properties"]["data"]
    jsonschema.Draft202012Validator(data_schema).validate(reports[0]["data"])
