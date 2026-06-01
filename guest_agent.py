"""Back-compat shim. Prefer `autobench.engines.guest_agent`.

Phase 2A of the autobench restructuring moved ``guest_agent.py`` into the
``autobench.engines`` subpackage. The in-VM stdlib-only agent is still
runnable as ``python -m autobench.engines.guest_agent``.

This module re-exports the public API so legacy
``from autobench.guest_agent import …`` call sites keep working.
"""

from .engines.guest_agent import (  # noqa: F401
    VSOCK_PORT,
)
