"""Back-compat shim. Prefer ``autobench.audit.budget_guard``."""

from autobench.audit.budget_guard import (  # noqa: F401
    BudgetExceeded,
    BudgetGuard,
    CompositeBudgetGuard,
    RateBudgetExceeded,
    RateBudgetGuard,
)

__all__ = [
    "BudgetExceeded",
    "BudgetGuard",
    "CompositeBudgetGuard",
    "RateBudgetExceeded",
    "RateBudgetGuard",
]
