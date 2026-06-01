"""Back-compat shim. Prefer `autobench.bus.gpu_types`.

Phase 2A of the autobench restructuring moved ``gpu_types.py`` into the
``autobench.bus`` subpackage. This module re-exports the public API so
legacy ``from autobench.gpu_types import …`` call sites keep working.
"""

from .bus.gpu_types import (  # noqa: F401
    GPUJob,
    GPUResult,
)
