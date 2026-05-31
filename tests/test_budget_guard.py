"""Tests for autobench.budget_guard."""

from __future__ import annotations

import time

import pytest

from autobench.budget_guard import BudgetExceeded, BudgetGuard


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
        return [p["threshold"] for _, p in self.events]

    def actions(self) -> list[str]:
        return [p["action"] for _, p in self.events]


# --------------------------------------------------------------------------- #
# Cost accumulation
# --------------------------------------------------------------------------- #


def test_record_cost_accumulates():
    pub = _MockPublisher()
    g = BudgetGuard(max_cost_dollars=1.00, publisher=pub)
    g.record_cost(0.10)
    g.record_cost(0.05)
    assert g.current_cost_dollars == pytest.approx(0.15)


def test_record_cost_ignores_negatives_and_none():
    g = BudgetGuard(max_cost_dollars=1.00, publisher=_MockPublisher())
    g.record_cost(0.20)
    g.record_cost(-0.05)  # ignored
    g.record_cost(None)   # type: ignore[arg-type]
    assert g.current_cost_dollars == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# check() correctness
# --------------------------------------------------------------------------- #


def test_check_under_budget_returns_ok():
    g = BudgetGuard(max_cost_dollars=1.00, publisher=_MockPublisher())
    g.record_cost(0.40)
    ok, reason = g.check()
    assert ok is True
    assert reason is None


def test_check_at_cap_returns_not_ok():
    g = BudgetGuard(max_cost_dollars=1.00, publisher=_MockPublisher())
    g.record_cost(1.00)
    ok, reason = g.check()
    assert ok is False
    assert reason is not None
    assert "cost" in reason.lower()


def test_check_over_cap_returns_not_ok():
    g = BudgetGuard(max_cost_dollars=1.00, publisher=_MockPublisher())
    g.record_cost(1.50)
    ok, reason = g.check()
    assert ok is False


# --------------------------------------------------------------------------- #
# halt()
# --------------------------------------------------------------------------- #


def test_halt_raises_budget_exceeded():
    g = BudgetGuard(max_cost_dollars=1.00, publisher=_MockPublisher())
    with pytest.raises(BudgetExceeded) as exc_info:
        g.halt("test reason")
    assert "test reason" in str(exc_info.value)


def test_halt_emits_warning_event():
    pub = _MockPublisher()
    g = BudgetGuard(max_cost_dollars=1.00, publisher=pub)
    with pytest.raises(BudgetExceeded):
        g.halt("manual halt")
    # halt should emit at least one event with action='halt'
    halt_events = [(c, p) for c, p in pub.events if p.get("action") == "halt"]
    assert len(halt_events) >= 1
    assert halt_events[0][0] == "autobench.budget.warning.v1"


# --------------------------------------------------------------------------- #
# Threshold emission (50% / 80% / 100%)
# --------------------------------------------------------------------------- #


def test_50_percent_threshold_emits():
    pub = _MockPublisher()
    g = BudgetGuard(max_cost_dollars=1.00, publisher=pub)
    g.record_cost(0.50)
    fired = pub.thresholds_fired()
    assert 0.5 in fired
    assert 0.8 not in fired


def test_80_percent_threshold_emits():
    pub = _MockPublisher()
    g = BudgetGuard(max_cost_dollars=1.00, publisher=pub)
    g.record_cost(0.50)
    g.record_cost(0.31)  # cum 0.81 → crosses 80%
    fired = pub.thresholds_fired()
    assert 0.5 in fired
    assert 0.8 in fired
    assert 1.0 not in fired


def test_100_percent_threshold_emits_halt():
    pub = _MockPublisher()
    g = BudgetGuard(max_cost_dollars=1.00, publisher=pub)
    g.record_cost(1.20)
    fired = pub.thresholds_fired()
    assert 0.5 in fired
    assert 0.8 in fired
    assert 1.0 in fired
    # The 100% threshold should fire as 'halt'
    actions_at_100 = [p["action"] for _, p in pub.events if p.get("threshold") == 1.0]
    assert "halt" in actions_at_100


def test_each_threshold_fires_only_once():
    pub = _MockPublisher()
    g = BudgetGuard(max_cost_dollars=1.00, publisher=pub)
    g.record_cost(0.50)
    g.record_cost(0.05)  # still in 50% band
    g.record_cost(0.05)  # still in 50% band
    fired = pub.thresholds_fired()
    assert fired.count(0.5) == 1


# --------------------------------------------------------------------------- #
# Wall-time exhaustion
# --------------------------------------------------------------------------- #


def test_wall_time_exhaustion_triggers_halt_when_cost_is_low():
    pub = _MockPublisher()
    g = BudgetGuard(
        max_cost_dollars=100.00,
        max_wall_time_seconds=0.01,
        publisher=pub,
    )
    time.sleep(0.05)
    ok, reason = g.check()
    assert ok is False
    assert reason is not None
    assert "wall-time" in reason.lower() or "wall" in reason.lower()


def test_wall_fraction_triggers_threshold_emission():
    pub = _MockPublisher()
    g = BudgetGuard(
        max_cost_dollars=100.00,
        max_wall_time_seconds=0.01,
        publisher=pub,
    )
    time.sleep(0.05)
    # Record a 0-cost iteration to trigger _maybe_emit_threshold
    g.record_iteration_complete()
    fired = pub.thresholds_fired()
    # Should have fired at least 50% based on wall-clock
    assert any(t >= 0.5 for t in fired)


# --------------------------------------------------------------------------- #
# Integration sanity: record_iteration_complete checkpoints emission
# --------------------------------------------------------------------------- #


def test_record_iteration_complete_increments_counter():
    g = BudgetGuard(max_cost_dollars=1.00, publisher=_MockPublisher())
    assert g.iterations_completed == 0
    g.record_iteration_complete()
    g.record_iteration_complete()
    assert g.iterations_completed == 2
