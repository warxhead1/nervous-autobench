"""Tests for nervous-bus-7g77: run_first termination-message disambiguation.

Cycle 4 ran two full RSI iterations successfully but stderr still printed
``[run_first] no result produced (early halt before first iteration)``. The
three distinct termination paths are now disambiguated by
``_termination_message``:

1. early-halt (no iteration completed) -> "improver failed before first iteration"
2. budget-stop (>=1 iteration, then BudgetExceeded) -> "budget-stop after N iterations"
3. clean success                                    -> "complete - N iterations"
"""

from __future__ import annotations

from dataclasses import dataclass

from autobench.benchmarks.codeforces_tier1.run_first import _termination_message


@dataclass
class _FakeResult:
    aggregate_score: float


def test_early_halt_no_iterations() -> None:
    """final_result is None AND zero iterations -> improver-failed message."""
    msg = _termination_message(final_result=None, iteration_count=0, best_score=None)
    assert "improver failed before first iteration" in msg
    assert "no result produced" in msg


def test_budget_stop_after_iterations() -> None:
    """final_result is None BUT >=1 iteration ran -> budget-stop message."""
    msg = _termination_message(final_result=None, iteration_count=2, best_score=0.713)
    assert "budget-stop" in msg
    assert "2 iterations" in msg
    assert "0.7130" in msg
    # Crucial regression assertion: the old message must NOT appear.
    assert "no result produced" not in msg


def test_clean_success() -> None:
    """final_result present -> complete message with iteration count + score."""
    msg = _termination_message(
        final_result=_FakeResult(aggregate_score=0.8421),
        iteration_count=3,
        best_score=0.8421,
    )
    assert "complete" in msg
    assert "3 iterations" in msg
    assert "0.8421" in msg
    assert "no result produced" not in msg
    assert "budget-stop" not in msg


def test_budget_stop_missing_best_score() -> None:
    """Defensive: if best_score is None despite iteration_count>=1, render n/a."""
    msg = _termination_message(final_result=None, iteration_count=1, best_score=None)
    assert "budget-stop" in msg
    assert "n/a" in msg
