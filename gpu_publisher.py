"""Back-compat shim. Prefer `autobench.bus.gpu_publisher`.

Phase 2A of the autobench restructuring moved ``gpu_publisher.py`` into
the ``autobench.bus`` subpackage. This module re-exports the public API
so legacy ``from autobench.gpu_publisher import …`` call sites keep
working.
"""

from .bus.gpu_publisher import (  # noqa: F401
    GPUResultPublisher,
)
