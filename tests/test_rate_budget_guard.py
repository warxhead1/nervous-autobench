"""Tests for RateBudgetGuard and CompositeBudgetGuard.

The MiniMax coding plan caps requests at 15,000 / 5h. These tests use a
fake clock so the sliding window is exercised deterministically without
real sleeps. Schema validation is covered against
``schemas/autobench.budget.rate.v1.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.budget_guard import (
    BudgetExceeded,
    BudgetGuard,
    CompositeBudgetGuard,
    RateBudgetExceeded,
    RateBudgetGuard,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _MockPublisher:
    """Captures every emission for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, channel: str, payload: dict) -> None:
        self.events.append((channel, dict(payload)))

    def thresholds_fired(self) -> list[float]:
        return [p["warned_at_threshold"] for c, p in self.events if c == "autobench.budget.rate.v1"]

    def actions(self) -> list[str]:
        return [p["action"] for c, p in self.events if c == "autobench.budget.rate.v1"]


class _FakeClock:
    """Monotonic clock under test control."""

    def __init__(self, t0: float = 1_000_000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --------------------------------------------------------------------------- #
# Construction + safety margin
# --------------------------------------------------------------------------- #


def test_safety_margin_reduces_effective_cap():
    g = RateBudgetGuard(max_requests=15000, window_seconds=18000, safety_margin=0.05)
    # 15000 * (1 - 0.05) = 14250
    assert g._effective_max == 14250


def test_zero_safety_margin_keeps_full_cap():
    g = RateBudgetGuard(max_requests=100, window_seconds=10, safety_margin=0.0)
    assert g._effective_max == 100


def test_available_starts_at_effective_max():
    g = RateBudgetGuard(max_requests=10, window_seconds=10, safety_margin=0.0)
    assert g.available() == 10
    assert g.current_count() == 0


# --------------------------------------------------------------------------- #
# Sliding-window correctness
# --------------------------------------------------------------------------- #


def test_sliding_window_drops_old_timestamps():
    clock = _FakeClock()
    g = RateBudgetGuard(
        max_requests=10, window_seconds=2.0, safety_margin=0.0,
        publisher=_MockPublisher(), clock=clock,
    )
    for _ in range(5):
        g.record_request()
    assert g.current_count() == 5

    # Advance past the window — all timestamps should expire.
    clock.advance(2.001)
    assert g.current_count() == 0
    assert g.available() == 10


def test_partial_window_expiry():
    clock = _FakeClock()
    g = RateBudgetGuard(
        max_requests=100, window_seconds=10.0, safety_margin=0.0,
        publisher=_MockPublisher(), clock=clock,
    )
    g.record_request()        # t=0
    clock.advance(3.0)
    g.record_request()        # t=3
    g.record_request()        # t=3
    clock.advance(8.0)
    # Now t=11 — only the t=3 timestamps are still within the 10s window.
    assert g.current_count() == 2


def test_full_window_then_wait_returns_to_zero():
    """Spec acceptance: 100 requests + wait window_seconds -> 0."""
    clock = _FakeClock()
    g = RateBudgetGuard(
        max_requests=200, window_seconds=5.0, safety_margin=0.0,
        publisher=_MockPublisher(), clock=clock,
    )
    for _ in range(100):
        g.record_request()
    assert g.current_count() == 100
    clock.advance(5.001)
    assert g.current_count() == 0


# --------------------------------------------------------------------------- #
# check() + halt()
# --------------------------------------------------------------------------- #


def test_check_under_cap_returns_ok():
    g = RateBudgetGuard(
        max_requests=10, window_seconds=10, safety_margin=0.0,
        publisher=_MockPublisher(), clock=_FakeClock(),
    )
    for _ in range(5):
        g.record_request()
    ok, reason = g.check()
    assert ok is True
    assert reason is None


def test_check_at_cap_returns_not_ok():
    pub = _MockPublisher()
    g = RateBudgetGuard(
        max_requests=10, window_seconds=10, safety_margin=0.0,
        publisher=pub, clock=_FakeClock(),
    )
    for _ in range(10):
        g.record_request()
    ok, reason = g.check()
    assert ok is False
    assert reason is not None
    assert "rate" in reason.lower()


def test_halt_raises_rate_budget_exceeded():
    g = RateBudgetGuard(
        max_requests=5, window_seconds=10, safety_margin=0.0,
        publisher=_MockPublisher(), clock=_FakeClock(),
    )
    with pytest.raises(RateBudgetExceeded) as exc_info:
        g.halt("test reason")
    assert "test reason" in str(exc_info.value)


def test_halt_emits_event_with_action_halt():
    pub = _MockPublisher()
    g = RateBudgetGuard(
        max_requests=5, window_seconds=10, safety_margin=0.0,
        publisher=pub, clock=_FakeClock(),
    )
    with pytest.raises(RateBudgetExceeded):
        g.halt("manual halt")
    halt_events = [(c, p) for c, p in pub.events if p.get("action") == "halt"]
    assert len(halt_events) >= 1
    assert halt_events[0][0] == "autobench.budget.rate.v1"


# --------------------------------------------------------------------------- #
# Threshold emission (50 / 80 / 100%)
# --------------------------------------------------------------------------- #


def test_50_percent_threshold_emits():
    pub = _MockPublisher()
    g = RateBudgetGuard(
        max_requests=10, window_seconds=10, safety_margin=0.0,
        publisher=pub, clock=_FakeClock(),
    )
    for _ in range(5):
        g.record_request()
    fired = pub.thresholds_fired()
    assert 0.5 in fired
    assert 0.8 not in fired


def test_80_percent_threshold_emits():
    pub = _MockPublisher()
    g = RateBudgetGuard(
        max_requests=10, window_seconds=10, safety_margin=0.0,
        publisher=pub, clock=_FakeClock(),
    )
    for _ in range(8):
        g.record_request()
    fired = pub.thresholds_fired()
    assert 0.5 in fired
    assert 0.8 in fired
    assert 1.0 not in fired


def test_100_percent_threshold_fires_halt_action():
    pub = _MockPublisher()
    g = RateBudgetGuard(
        max_requests=10, window_seconds=10, safety_margin=0.0,
        publisher=pub, clock=_FakeClock(),
    )
    for _ in range(10):
        g.record_request()
    fired = pub.thresholds_fired()
    assert 0.5 in fired
    assert 0.8 in fired
    assert 1.0 in fired
    actions_at_100 = [
        p["action"] for c, p in pub.events
        if c == "autobench.budget.rate.v1" and p.get("warned_at_threshold") == 1.0
    ]
    assert "halt" in actions_at_100


def test_each_threshold_fires_only_once():
    pub = _MockPublisher()
    g = RateBudgetGuard(
        max_requests=20, window_seconds=10, safety_margin=0.0,
        publisher=pub, clock=_FakeClock(),
    )
    for _ in range(12):  # crosses 50% and stays in band
        g.record_request()
    fired = pub.thresholds_fired()
    assert fired.count(0.5) == 1


# --------------------------------------------------------------------------- #
# time_until_available
# --------------------------------------------------------------------------- #


def test_time_until_available_zero_when_under_cap():
    g = RateBudgetGuard(
        max_requests=10, window_seconds=10, safety_margin=0.0,
        publisher=_MockPublisher(), clock=_FakeClock(),
    )
    for _ in range(5):
        g.record_request()
    assert g.time_until_available() == 0.0


def test_time_until_available_accurate_when_full():
    clock = _FakeClock()
    g = RateBudgetGuard(
        max_requests=3, window_seconds=10.0, safety_margin=0.0,
        publisher=_MockPublisher(), clock=clock,
    )
    g.record_request()   # t=0
    clock.advance(2.0)
    g.record_request()   # t=2
    clock.advance(1.0)
    g.record_request()   # t=3 — now at cap
    # Oldest is t=0, window=10, current=3 → 7s until it leaves the window.
    assert g.time_until_available() == pytest.approx(7.0, abs=1e-6)
    clock.advance(4.0)   # now t=7
    # oldest still t=0, elapsed=7, window=10 → 3s left
    assert g.time_until_available() == pytest.approx(3.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# CompositeBudgetGuard
# --------------------------------------------------------------------------- #


def _make_composite(
    *,
    max_cost: float = 1.00,
    max_requests: int = 10,
    window: float = 10.0,
    margin: float = 0.0,
    pub: _MockPublisher | None = None,
    clock: _FakeClock | None = None,
) -> CompositeBudgetGuard:
    pub = pub or _MockPublisher()
    clock = clock or _FakeClock()
    cost = BudgetGuard(
        max_cost_dollars=max_cost,
        max_wall_time_seconds=1e9,  # effectively disable wall-time
        publisher=pub,
    )
    rate = RateBudgetGuard(
        max_requests=max_requests, window_seconds=window, safety_margin=margin,
        publisher=pub, clock=clock,
    )
    return CompositeBudgetGuard(cost_guard=cost, rate_guard=rate)


def test_composite_only_dollar_exceeded_halts_with_dollar_reason():
    g = _make_composite(max_cost=0.10, max_requests=1000)
    g.record_request(cost_usd=0.15)
    ok, reason = g.check()
    assert ok is False
    assert reason is not None
    assert "cost" in reason.lower()
    with pytest.raises(BudgetExceeded):
        g.halt(reason)


def test_composite_only_rate_exceeded_halts_with_rate_reason():
    g = _make_composite(max_cost=1000.0, max_requests=3)
    for _ in range(3):
        g.record_request(cost_usd=0.001)
    ok, reason = g.check()
    assert ok is False
    assert reason is not None
    assert "rate" in reason.lower()
    with pytest.raises(RateBudgetExceeded):
        g.halt(reason)


def test_composite_both_exceeded_rate_wins():
    """Rate is the binding constraint under free-tier plans → first-found wins."""
    g = _make_composite(max_cost=0.10, max_requests=3)
    for _ in range(3):
        g.record_request(cost_usd=0.05)  # crosses both
    ok, reason = g.check()
    assert ok is False
    # Rate is checked first.
    assert "rate" in reason.lower()


def test_composite_under_both_caps_returns_ok():
    g = _make_composite(max_cost=1.00, max_requests=100)
    for _ in range(5):
        g.record_request(cost_usd=0.01)
    ok, reason = g.check()
    assert ok is True
    assert reason is None


def test_composite_record_request_increments_both():
    pub = _MockPublisher()
    g = _make_composite(max_cost=10.0, max_requests=100, pub=pub)
    g.record_request(cost_usd=0.20)
    assert g.rate_guard.current_count() == 1
    assert g.cost_guard.current_cost_dollars == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def _schema_path() -> Path:
    from tests._paths import SCHEMA_DIR
    return SCHEMA_DIR / "autobench.budget.rate.v1.json"


def test_schema_file_exists():
    assert _schema_path().is_file()


def test_emitted_payload_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_schema_path().read_text())

    pub = _MockPublisher()
    g = RateBudgetGuard(
        max_requests=10, window_seconds=10, safety_margin=0.0,
        publisher=pub, clock=_FakeClock(),
    )
    for _ in range(10):
        g.record_request()

    # Find the 100% halt emission's payload and wrap it in a CloudEvents envelope
    # to validate against the full schema (which checks both envelope + data).
    rate_events = [(c, p) for c, p in pub.events if c == "autobench.budget.rate.v1"]
    assert rate_events, "no rate events emitted"
    channel, data = rate_events[-1]

    envelope = {
        "specversion": "1.0",
        "id": "01TESTULID00000000000000000",
        "source": "/autobench",
        "type": channel,
        "datacontenttype": "application/json",
        "time": "2026-05-16T12:00:00.000Z",
        "data": data,
    }
    jsonschema.Draft202012Validator(schema).validate(envelope)
