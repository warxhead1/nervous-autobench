"""Back-compat shim. Prefer `autobench.evaluation.distillation`.

Phase 2B of the autobench restructuring moved ``distillation.py`` into
the ``autobench.evaluation`` subpackage. This module re-exports the
public surface so legacy ``from autobench.distillation import …`` call
sites keep working.
"""

from .evaluation.distillation import (  # noqa: F401
    CycleDistiller,
)
