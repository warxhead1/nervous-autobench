"""Back-compat shim. Prefer `autobench.bus.integration`.

Phase 2A of the autobench restructuring moved ``integration.py`` into the
``autobench.bus`` subpackage. This module re-exports the public API so
legacy ``from autobench.integration import …`` call sites keep working.
"""

from .bus.integration import (  # noqa: F401
    BudgetViolation,
    BudgetViolationMiddleware,
    DeerFlowEvaluator,
    NervousBusPublisher,
    RoleSpec,
)
