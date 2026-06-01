"""Tests for signal_bus.py — AutobenchResultPublisher and AutobenchResultSubscriber."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from autobench.core import HarnessResult, HarnessResult, Verdict
from autobench.bus.idgen import iso_now, ulid
from autobench.bus.signal_bus import (
    AutobenchResultPublisher,
    AutobenchResultSubscriber,
    DEBUG_FILE,
    make_publisher,
    make_subscriber,
)


class TestUlid:
    def test_ulid_format(self):
        """ULID should be 26 characters: 10 digit timestamp + 16 hex chars."""
        uid = ulid()
        assert len(uid) == 26
        assert uid.isdigit() or uid[:10].isdigit()

    def test_ulid_unique(self):
        """Each ULID should be unique."""
        ids = {ulid() for _ in range(1000)}
        assert len(ids) == 1000


class TestIcoNow:
    def test_iso_now_format(self):
        """ISO timestamp should be RFC3339 (datetime-based form).

        Phase 2A unified the three prior RFC3339 helpers on
        ``datetime.now(timezone.utc).isoformat()`` — see
        ``autobench.bus.idgen.iso_now``. The resulting string is
        RFCEvents-compliant: it carries an explicit ``+00:00`` offset
        (instead of the prior ``Z`` shorthand) and includes microsecond
        precision when available.
        """
        ts = iso_now()
        # RFC3339 in the datetime.isoformat() form:
        #   YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00
        assert ts.endswith("+00:00")
        assert ts[10] == "T"
        # Prefix is YYYY-MM-DD — 10 chars.
        assert len(ts) >= 25
        # Prefix must be a valid date.
        from datetime import datetime
        # Tolerate optional microseconds by trimming before parse.
        head = ts.split("+", 1)[0]
        datetime.fromisoformat(head)


class TestAutobenchResultPublisher:
    def test_build_event_uses_harness_result_fields(self):
        """_build_event should extract problem_id, verdict, p_* from HarnessResult."""
        result = HarnessResult(
            p_score=0.85,
            p_cost=0.92,
            p_time=0.78,
            verdict=Verdict.OK,
            error="",
            latency_ms=1234.5,
            tokens_used=5000,
            cost_dollars=0.003,
            metadata={"case_id": "case-42", "extra": "data"},
        )

        pub = AutobenchResultPublisher(
            harness_version="v1.0",
            benchmark_name="humaneval",
            iteration=3,
        )
        event = pub._build_event(result)

        assert event["source"] == "/autobench/evaluator"
        assert event["type"] == "autobench.result.v1"
        assert event["datacontenttype"] == "application/json"
        assert "id" in event
        assert "time" in event

        data = event["data"]
        assert data["problem_id"] == "case-42"
        assert data["verdict"] == "OK"
        assert data["p_score"] == 0.85
        assert data["p_cost"] == 0.92
        assert data["p_time"] == 0.78
        assert data["harness_version"] == "v1.0"
        assert data["benchmark_name"] == "humaneval"
        assert data["iteration"] == 3
        assert data["latency_ms"] == 1234.5
        assert data["tokens_used"] == 5000
        assert data["cost_dollars"] == 0.003
        assert data["error"] == ""
        assert data["metadata"] == {"case_id": "case-42", "extra": "data"}

    def test_build_event_error_in_data(self):
        """Error field should appear in data when result.error is non-empty."""
        result = HarnessResult(
            p_score=0.0,
            p_cost=0.5,
            p_time=0.5,
            verdict=Verdict.RE,
            error="Traceback: index out of range",
            latency_ms=100.0,
            metadata={"case_id": "case-99"},
        )

        pub = AutobenchResultPublisher()
        event = pub._build_event(result)

        assert event["data"]["verdict"] == "RE"
        assert "index out of range" in event["data"]["error"]

    def test_build_event_rejects_non_harness_result(self):
        """Passing a non-HarnessResult should raise TypeError."""
        pub = AutobenchResultPublisher()
        with pytest.raises(TypeError):
            pub._build_event({"foo": "bar"})

    def test_try_zellij_pipe_succeeds(self):
        """When zellij pipe succeeds, _try_zellij_pipe returns True."""
        pub = AutobenchResultPublisher()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = pub._try_zellij_pipe('{"test": true}')
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[:4] == ["zellij", "pipe", "-p", "nervous-bus"]

    def test_try_zellij_pipe_falls_back_on_failure(self):
        """When zellij pipe fails, _try_zellij_pipe returns False."""
        pub = AutobenchResultPublisher()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            result = pub._try_zellij_pipe('{"test": true}')
            assert result is False

    def test_try_zellij_pipe_exc_on_timeout(self):
        """When zellij pipe throws, _try_zellij_pipe returns False."""
        pub = AutobenchResultPublisher()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)
            result = pub._try_zellij_pipe('{"test": true}')
            assert result is False

    def test_publish_writes_debug_on_zellij_failure(self, tmp_path):
        """When zellij fails, publish should write to debug.jsonl."""
        debug_dir = tmp_path / ".cache" / "nervous-bus"
        debug_file = debug_dir / "debug.jsonl"

        with patch("autobench.bus.signal_bus.DEBUG_FILE", debug_file):
            with patch("autobench.bus.signal_bus._ensure_debug_dir"):
                pub = AutobenchResultPublisher()
                with patch.object(pub, "_try_zellij_pipe", return_value=False):
                    with patch.object(pub, "_write_debug") as mock_write:
                        result = HarnessResult(
                            p_score=1.0,
                            verdict=Verdict.OK,
                            metadata={"case_id": "c1"},
                        )
                        pub.publish(result)
                        mock_write.assert_called_once()
                        written = mock_write.call_args[0][0]
                        assert "autobench.result.v1" in written

    def test_publish_returns_true_even_on_fallback(self, tmp_path):
        """publish() should return True when falling back to debug file."""
        debug_file = tmp_path / "debug.jsonl"

        with patch("autobench.bus.signal_bus.DEBUG_FILE", debug_file):
            pub = AutobenchResultPublisher()
            with patch.object(pub, "_try_zellij_pipe", return_value=False):
                result = HarnessResult(
                    p_score=1.0,
                    verdict=Verdict.OK,
                    metadata={"case_id": "c1"},
                )
                ok = pub.publish(result)
                assert ok is True


class TestAutobenchResultSubscriber:
    def test_process_line_dispatches_to_callback(self):
        """_process_line should call callback with parsed event."""
        received = []

        def cb(event):
            received.append(event)

        sub = AutobenchResultSubscriber(callback=cb)
        event = {
            "id": "abc123",
            "source": "/autobench/evaluator",
            "type": "autobench.result.v1",
            "datacontenttype": "application/json",
            "time": "2026-05-15T00:00:00Z",
            "data": {
                "problem_id": "case-1",
                "verdict": "OK",
                "p_score": 1.0,
                "p_cost": 0.9,
                "p_time": 0.8,
                "harness_version": "v0",
                "benchmark_name": "test",
                "iteration": 0,
            },
        }
        sub._process_line(json.dumps(event))
        assert len(received) == 1
        assert received[0]["data"]["problem_id"] == "case-1"

    def test_process_line_ignores_other_types(self):
        """_process_line should ignore events with type != autobench.result.v1."""
        received = []

        def cb(event):
            received.append(event)

        sub = AutobenchResultSubscriber(callback=cb)
        event = {
            "id": "abc123",
            "type": "some.other.event",
            "data": {},
        }
        sub._process_line(json.dumps(event))
        assert len(received) == 0

    def test_process_line_ignores_malformed_json(self):
        """_process_line should ignore lines that are not valid JSON."""
        received = []

        def cb(event):
            received.append(event)

        sub = AutobenchResultSubscriber(callback=cb)
        sub._process_line("not valid json")
        assert len(received) == 0

    def test_emit_to_deer_flow_called_when_enabled(self):
        """When deer_flow_emit=True, _emit_to_deer_flow should be called."""
        received = []

        def cb(event):
            received.append(event)

        sub = AutobenchResultSubscriber(callback=cb, deer_flow_emit=True)

        event = {
            "id": "abc123",
            "type": "autobench.result.v1",
            "source": "/autobench/evaluator",
            "data": {"problem_id": "case-1", "verdict": "OK"},
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            sub._process_line(json.dumps(event))

        assert len(received) == 1
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "deer-flow.metaprobe.cycle" in args

    def test_emit_to_deer_flow_not_called_when_disabled(self):
        """When deer_flow_emit=False, _emit_to_deer_flow should not be called."""
        sub = AutobenchResultSubscriber(callback=lambda e: None, deer_flow_emit=False)

        event = {
            "id": "abc123",
            "type": "autobench.result.v1",
            "source": "/autobench/evaluator",
            "data": {"problem_id": "case-1", "verdict": "OK"},
        }

        with patch("subprocess.run") as mock_run:
            sub._process_line(json.dumps(event))

        mock_run.assert_not_called()


class TestMakePublisher:
    def test_make_publisher_returns_configured_instance(self):
        """make_publisher should return a ready-to-use publisher."""
        pub = make_publisher(harness_version="v2", benchmark_name="mbpp", iteration=5)
        assert pub.harness_version == "v2"
        assert pub.benchmark_name == "mbpp"
        assert pub.iteration == 5


class TestMakeSubscriber:
    def test_make_subscriber_returns_configured_instance(self):
        """make_subscriber should return a ready-to-use subscriber."""
        cb = lambda e: None
        sub = make_subscriber(callback=cb, deer_flow_emit=True)
        assert sub.callback is cb
        assert sub.deer_flow_emit is True


class TestPublisherRoundTrip:
    def test_full_roundtrip_via_debug_file(self, tmp_path):
        """Publish a result and consume it via subscriber on the debug file."""
        debug_file = tmp_path / "debug.jsonl"

        # Produce
        with patch("autobench.bus.signal_bus.DEBUG_FILE", debug_file):
            pub = AutobenchResultPublisher(
                harness_version="v1",
                benchmark_name="test-bench",
                iteration=1,
            )
            with patch.object(pub, "_try_zellij_pipe", return_value=False):
                result = HarnessResult(
                    p_score=0.75,
                    p_cost=0.8,
                    p_time=0.65,
                    verdict=Verdict.WA,
                    error="output mismatch",
                    latency_ms=500.0,
                    tokens_used=3000,
                    cost_dollars=0.002,
                    metadata={"case_id": "roundtrip-case"},
                )
                pub.publish(result)

        # Verify file content
        lines = debug_file.read_text().strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["type"] == "autobench.result.v1"
        assert event["data"]["verdict"] == "WA"
        assert event["data"]["problem_id"] == "roundtrip-case"
        assert event["data"]["p_score"] == 0.75

        # Consume via subscriber
        received = []

        def cb(event):
            received.append(event)

        with patch("autobench.bus.signal_bus.DEBUG_FILE", debug_file):
            sub = AutobenchResultSubscriber(callback=cb)
            sub._process_line(lines[0])

        assert len(received) == 1
        assert received[0]["data"]["verdict"] == "WA"