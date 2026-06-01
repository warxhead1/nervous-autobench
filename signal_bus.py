"""Back-compat shim. Prefer `autobench.bus.signal_bus`.

Phase 2A of the autobench restructuring moved ``signal_bus.py`` into the
``autobench.bus`` subpackage. This module re-exports the public API so
legacy ``from autobench.signal_bus import …`` call sites keep working.
"""

from .bus.signal_bus import (  # noqa: F401
    AutobenchResultPublisher,
    AutobenchResultSubscriber,
    DeerFlowResultSubscriber,
    make_deerflow_subscriber,
    make_publisher,
    make_subscriber,
)
