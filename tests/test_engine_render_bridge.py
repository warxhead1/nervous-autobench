"""Tests for the generic engine-render contract (Phase 2 producer side).

Exercises ``svdag_beauty_kernel.bridge_eval`` end-to-end against a synthetic bus
log, plus schema-validates a generated ``funsearch.artifact.v1`` engine_render
payload. No live services: the "engine" is simulated by appending a completed
event line to a temp debug.jsonl. Marked not-live (no network/sandbox).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

from autobench.svdag_beauty_kernel import bridge_eval

NBUS_SCHEMA_DIR = Path.home() / "projects" / "nervous-bus" / "schemas"
REQ_SCHEMA = NBUS_SCHEMA_DIR / "funsearch.engine_render.requested.v1.json"
DONE_SCHEMA = NBUS_SCHEMA_DIR / "funsearch.engine_render.completed.v1.json"
ART_SCHEMA = NBUS_SCHEMA_DIR / "funsearch.artifact.v1.json"


def _append_event(log: Path, channel: str, data: dict) -> None:
    env = {
        "specversion": "1.0",
        "id": uuid.uuid4().urn,
        "source": "/some/engine/adapter",
        "type": channel,
        "datacontenttype": "application/json",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": data,
    }
    with open(log, "a") as f:
        f.write(json.dumps(env) + "\n")


def _patch_log(monkeypatch, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    monkeypatch.setattr(bridge_eval, "_DEBUG_LOG", log)


def test_request_render_matches_synthetic_completion(monkeypatch, tmp_path):
    """A synthetic engine reply on the bus log is matched by correlation_id and
    its data block returned verbatim."""
    log = tmp_path / "debug.jsonl"
    _patch_log(monkeypatch, log)

    captured: dict = {}

    def publish(channel, data):
        # The producer writes the request to the bus; we also assert its shape and
        # then immediately simulate an engine answering on the same correlation_id.
        captured["channel"] = channel
        captured["data"] = data
        assert channel == bridge_eval.REQUEST_CHANNEL
        # validate the request against the public schema
        env = {
            "specversion": "1.0", "id": "urn:uuid:x", "source": "/autobench/svdag_kernel",
            "type": channel, "datacontenttype": "application/json",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data": data,
        }
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(REQ_SCHEMA.read_text())
        jsonschema.Draft202012Validator(schema).validate(env)
        # engine answers
        _append_event(log, bridge_eval.COMPLETED_CHANNEL, {
            "correlation_id": data["correlation_id"],
            "candidate_id": data["candidate_id"],
            "run_id": data["run_id"],
            "kernel": data["kernel"],
            "status": "ok",
            "engine": "fake-engine",
            "screenshot_path": str(tmp_path / "shot.png"),
            "render_ms": 42.0,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return True

    result = bridge_eval.request_render(
        publish, "float compute_density(...){return 0.0;}", "island0_gen3_llm0",
        run_id="01TESTRUN", kernel="svdag", instance="stratovolcano", seed=5151.0,
        timeout=5.0, poll=0.05,
    )
    assert result is not None
    assert result["status"] == "ok"
    assert result["engine"] == "fake-engine"
    assert result["screenshot_path"].endswith("shot.png")
    # request carried the new generic fields
    assert captured["data"]["run_id"] == "01TESTRUN"
    assert captured["data"]["kernel"] == "svdag"
    assert captured["data"]["splice_target"] == "compute_density"
    assert captured["data"]["instance"] == "stratovolcano"
    assert "requested_at" in captured["data"]


def test_request_render_graceful_timeout(monkeypatch, tmp_path):
    """No engine answers -> request_render returns None (run continues)."""
    log = tmp_path / "debug.jsonl"
    _patch_log(monkeypatch, log)

    def publish(channel, data):
        # never append a completion -> simulate no engine listening
        return True

    t0 = time.time()
    result = bridge_eval.request_render(
        publish, "code", "cand0", run_id="R", kernel="svdag",
        timeout=0.4, poll=0.05,
    )
    assert result is None
    assert time.time() - t0 >= 0.4  # actually waited out the deadline


def test_request_render_ignores_stale_and_mismatched_completions(monkeypatch, tmp_path):
    """Completions written before the request, or with a different
    correlation_id, are not matched."""
    log = tmp_path / "debug.jsonl"
    _patch_log(monkeypatch, log)

    # a stale completion from a previous render, written BEFORE we publish
    _append_event(log, bridge_eval.COMPLETED_CHANNEL, {
        "correlation_id": "STALE", "candidate_id": "old", "status": "ok",
        "screenshot_path": "/old.png", "completed_at": "2026-06-03T00:00:00Z",
    })

    def publish(channel, data):
        # a mismatched-correlation completion + the right one
        _append_event(log, bridge_eval.COMPLETED_CHANNEL, {
            "correlation_id": "OTHER", "candidate_id": "x", "status": "ok",
            "screenshot_path": "/wrong.png", "completed_at": "2026-06-03T00:00:00Z",
        })
        _append_event(log, bridge_eval.COMPLETED_CHANNEL, {
            "correlation_id": data["correlation_id"], "candidate_id": data["candidate_id"],
            "status": "ok", "screenshot_path": "/right.png",
            "completed_at": "2026-06-03T00:00:00Z",
        })
        return True

    result = bridge_eval.request_render(
        publish, "code", "cand0", run_id="R", kernel="svdag", timeout=5.0, poll=0.05,
    )
    assert result is not None
    assert result["screenshot_path"] == "/right.png"


def test_engine_render_artifact_payload_validates():
    """An engine_render funsearch.artifact.v1 payload validates against the public
    schema EXCEPT the render_type enum (which is stale upstream and does not yet
    list engine_render / svdag_voxel_iso). We assert: (a) the full payload is valid
    once render_type is widened, and (b) the categorization fields are present.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(ART_SCHEMA.read_text())

    data = {
        "kernel": "svdag",
        "run_id": "01TESTRUN",
        "generation": 7,
        "fitness": 0.7421,
        "instance": "stratovolcano",
        "artifact_path": "/run/out/shot.png",
        "render_type": "engine_render",
        # gallery-grouping discriminators (additionalProperties on data)
        "island": 2,
        "candidate_id": "island2_gen7_llm0",
        "metadata": {
            "pore_frac": 0.31, "rough": 0.52, "beta": 1.8, "relief": 0.66,
            "spectral_beta": 1.8, "code_length": 480,
            "best_program_code": "float compute_density(...){...}",
            "engine": "fake-engine", "render_ms": 42.0,
        },
        "time": "2026-06-03T00:00:00Z",
    }
    env = {
        "specversion": "1.0", "id": "urn:uuid:x", "source": "/autobench/svdag_kernel",
        "type": "funsearch.artifact.v1", "datacontenttype": "application/json",
        "time": "2026-06-03T00:00:00Z", "data": data,
    }

    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(env))
    # Only the render_type enum may legitimately fail (upstream enum is stale and
    # read-only here); everything else MUST validate.
    non_enum = [e for e in errors if list(e.absolute_path)[-1:] != ["render_type"]]
    assert not non_enum, f"unexpected schema errors: {[str(e) for e in non_enum]}"

    # And with render_type widened to a permitted value, the SAME payload is fully
    # valid — proving the structure (incl. island/candidate_id + metadata) conforms.
    data["render_type"] = "none"
    assert validator.is_valid(env), "payload invalid even with permitted render_type"

    # categorization metadata present for gallery grouping
    assert data["island"] == 2
    assert data["candidate_id"] == "island2_gen7_llm0"
    for k in ("pore_frac", "rough", "beta", "relief"):
        assert k in data["metadata"]
