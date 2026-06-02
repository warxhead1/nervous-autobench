"""Tests for the MiniMaxWorker queue-pressure rolling-tokens-per-second signal.

Bead: nervous-bus-8vn.

The signal fires when one of two pressure conditions holds *after* baseline
has been established from three successful calls:
    1. current_rate_tps < 0.5 * baseline_tps
    2. latest_latency_ms > 2 * mean(window_latencies)

Debounce: at most one event per 30 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from autobench.core import ContextManager, HarnessConfig, RolloutProtocol
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_WORKER_QUEUE_PRESSURE,
)
from autobench.llm.worker import MiniMaxWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_harness() -> HarnessConfig:
    return HarnessConfig(
        system_prompt="solve",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="",
        verifiers=[],
        budget={"max_tokens": 1024, "max_time_seconds": 10, "max_cost_dollars": 0.5},
    )


def _mock_resp(content: str, *, prompt_tokens: int = 50, completion_tokens: int = 50):
    return {
        "id": "resp",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _read_qp_events(debug_file: Path) -> list[dict]:
    events = [
        json.loads(line)
        for line in debug_file.read_text().splitlines()
        if line.strip()
    ]
    return [e for e in events if e["type"] == CHANNEL_WORKER_QUEUE_PRESSURE]


class _LatencyDriver:
    """Drives ``time.monotonic`` to produce deterministic latencies.

    Each call returns the next value from ``values`` (each value used twice —
    once for ``start`` and once for the latency computation in
    ``_attempt_with_model``). For queue-pressure timestamps we extend the
    sequence with monotonically-increasing wall times.
    """

    def __init__(self, latencies_seconds: list[float], gap_seconds: float = 0.1):
        # For each call we need (start, end) → end - start = latency.
        # Then queue-pressure code calls time.monotonic() once more after
        # _emit_worker_event for the "now" timestamp.
        self.values: list[float] = []
        t = 0.0
        for lat in latencies_seconds:
            self.values.append(t)         # generate() start
            t += lat
            self.values.append(t)         # latency_ms computation in _attempt_with_model
            t += gap_seconds
            self.values.append(t)         # _record_call_for_queue_pressure now
            t += gap_seconds
        self.idx = 0

    def __call__(self) -> float:
        v = self.values[self.idx]
        self.idx += 1
        return v


def _run_calls(
    worker: MiniMaxWorker,
    harness: HarnessConfig,
    latencies_seconds: list[float],
    completion_tokens_per_call: list[int],
    *,
    monotonic_driver: _LatencyDriver | None = None,
) -> None:
    """Execute one generate() per (latency, completion_tokens) pair."""
    driver = monotonic_driver or _LatencyDriver(latencies_seconds)
    mock_client = MagicMock()

    responses = []
    for ct in completion_tokens_per_call:
        r = MagicMock()
        r.json.return_value = _mock_resp("code", completion_tokens=ct)
        r.raise_for_status.return_value = None
        responses.append(r)
    mock_client.post.side_effect = responses

    # Install on the persistent client slot — no httpx.Client patch needed.
    worker._http_client = mock_client

    with patch("autobench.llm.worker.time.monotonic", side_effect=driver):
        for _ in range(len(latencies_seconds)):
            worker.generate("p", harness, case_id="c")


# ---------------------------------------------------------------------------
# (1) Baseline established after 3 calls
# ---------------------------------------------------------------------------

def test_baseline_established_after_three_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    obs = AutobenchObservability(debug_file=tmp_path / "d.jsonl")
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    # Three fast calls: 1s latency, 100 tokens each → tps = 100.
    _run_calls(w, harness, [1.0, 1.0, 1.0], [100, 100, 100])

    assert w._qp_baseline_tps is not None
    assert w._qp_baseline_tps == pytest.approx(100.0, rel=0.05)
    # No pressure events yet — baseline only just established.
    qp_events = _read_qp_events(tmp_path / "d.jsonl")
    assert qp_events == []


# ---------------------------------------------------------------------------
# (2) Fourth call at <0.5x baseline → event fires
# ---------------------------------------------------------------------------

def test_fourth_call_below_half_baseline_emits_event(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "d.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    # 3 baseline calls @ tps=100 (1s latency, 100 tokens), then 1 slow call
    # @ tps=30 (1s latency, 30 tokens) — well below 0.5x baseline.
    _run_calls(
        w,
        harness,
        latencies_seconds=[1.0, 1.0, 1.0, 1.0],
        completion_tokens_per_call=[100, 100, 100, 30],
    )

    qp_events = _read_qp_events(debug_file)
    assert len(qp_events) == 1
    data = qp_events[0]["data"]
    assert data["model"] == "MiniMax-M2.7"
    assert data["current_rate_tps"] == pytest.approx(30.0, rel=0.05)
    assert data["baseline_tps"] == pytest.approx(100.0, rel=0.05)
    assert data["deviation_factor"] == pytest.approx(0.3, rel=0.1)
    assert data["recent_timeouts_count"] == 0
    assert data["latest_latency_ms"] == pytest.approx(1000.0, rel=0.05)
    assert data["window_size"] >= 1


# ---------------------------------------------------------------------------
# (3) 4th call faster than baseline → no event
# ---------------------------------------------------------------------------

def test_fourth_call_faster_than_baseline_no_event(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "d.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    # 3 baseline calls @ tps=100, then 1 fast call @ tps=150 (1s latency,
    # 150 tokens). No pressure expected.
    _run_calls(
        w,
        harness,
        latencies_seconds=[1.0, 1.0, 1.0, 1.0],
        completion_tokens_per_call=[100, 100, 100, 150],
    )

    qp_events = _read_qp_events(debug_file)
    assert qp_events == []


# ---------------------------------------------------------------------------
# (4) Debounce — two slow calls within 30s emit only once
# ---------------------------------------------------------------------------

def test_debounce_within_30s_emits_once(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "d.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    # 3 fast baseline calls + 2 slow calls — both should detect pressure,
    # but only the first should emit (second is inside the 30s debounce
    # window because _LatencyDriver advances ~1.2s per call).
    _run_calls(
        w,
        harness,
        latencies_seconds=[1.0, 1.0, 1.0, 1.0, 1.0],
        completion_tokens_per_call=[100, 100, 100, 20, 20],
    )

    qp_events = _read_qp_events(debug_file)
    assert len(qp_events) == 1, (
        f"expected exactly one event under 30s debounce, got {len(qp_events)}"
    )


def test_debounce_releases_after_30s(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "d.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    # 3 fast baseline calls + 2 slow calls, but with a large gap_seconds so
    # the wall-clock between calls exceeds 30s — second slow call MUST emit.
    driver = _LatencyDriver(
        latencies_seconds=[1.0, 1.0, 1.0, 1.0, 1.0],
        gap_seconds=20.0,  # 20s between sub-events × 2 sub-events/call = 40s gap
    )
    _run_calls(
        w,
        harness,
        latencies_seconds=[1.0, 1.0, 1.0, 1.0, 1.0],
        completion_tokens_per_call=[100, 100, 100, 20, 20],
        monotonic_driver=driver,
    )

    qp_events = _read_qp_events(debug_file)
    assert len(qp_events) == 2


# ---------------------------------------------------------------------------
# (5) Timeout counter increments across retries
# ---------------------------------------------------------------------------

def test_timeout_counter_tracks_retries(monkeypatch, tmp_path):
    """One ReadTimeout retry followed by success → recent_timeouts_count >= 1."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "d.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, max_retries=3, endpoint_mode="openai")
    harness = _make_harness()

    # Build 4 calls' worth of responses. First three are clean (baseline).
    # Fourth call: ReadTimeout then a slow successful response.
    def make_clean():
        r = MagicMock()
        r.json.return_value = _mock_resp("ok", completion_tokens=100)
        r.raise_for_status.return_value = None
        return r

    def make_slow():
        r = MagicMock()
        r.json.return_value = _mock_resp("ok", completion_tokens=20)
        r.raise_for_status.return_value = None
        return r

    timeout_exc = httpx.ReadTimeout("readtimeout")

    mock_client = MagicMock()
    mock_client.post.side_effect = [
        make_clean(), make_clean(), make_clean(),  # baseline 1, 2, 3
        timeout_exc,                                # 4th call: 1st attempt times out
        make_slow(),                                # 4th call: 2nd attempt succeeds slowly
    ]

    # Install on the persistent client slot — no httpx.Client patch needed.
    w._http_client = mock_client

    driver = _LatencyDriver(
        latencies_seconds=[1.0, 1.0, 1.0, 1.0],
        gap_seconds=0.1,
    )

    with patch("autobench.llm.worker.time.monotonic", side_effect=driver), \
         patch("autobench.llm.worker.time.sleep"):
        for _ in range(4):
            w.generate("p", harness, case_id="c")

    qp_events = _read_qp_events(debug_file)
    assert len(qp_events) == 1
    assert qp_events[0]["data"]["recent_timeouts_count"] >= 1


# ---------------------------------------------------------------------------
# (6) Schema validation
# ---------------------------------------------------------------------------

def test_queue_pressure_event_validates_against_schema(monkeypatch, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")

    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    debug_file = tmp_path / "d.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)
    obs._pipe_disabled = True
    w = MiniMaxWorker(obs=obs, endpoint_mode="openai")
    harness = _make_harness()

    _run_calls(
        w,
        harness,
        latencies_seconds=[1.0, 1.0, 1.0, 1.0],
        completion_tokens_per_call=[100, 100, 100, 30],
    )

    qp_events = _read_qp_events(debug_file)
    assert len(qp_events) == 1

    from tests._paths import SCHEMA_DIR
    schema_path = SCHEMA_DIR / "autobench.worker.queue_pressure.v1.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(qp_events[0])
