"""Back-compat shim. Prefer `autobench.evaluation.assembly`.

Phase 2B of the autobench restructuring moved ``benchmark_assembly.py``
into the ``autobench.evaluation`` subpackage. This module re-exports
the public surface so legacy ``from autobench.benchmark_assembly
import …`` call sites keep working.
"""

from .evaluation.assembly import (  # noqa: F401
    DEFAULT_ADVERSARIAL_RATIO,
    assemble_benchmark_cases,
)
