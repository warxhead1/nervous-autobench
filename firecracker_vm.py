"""Back-compat shim. Prefer `autobench.engines.firecracker_vm`.

Phase 2A of the autobench restructuring moved ``firecracker_vm.py`` into
the ``autobench.engines`` subpackage. This module re-exports the public
API so legacy ``from autobench.firecracker_vm import …`` call sites keep
working.
"""

from .engines.firecracker_vm import (  # noqa: F401
    FirecrackerAPI,
    FirecrackerError,
    FirecrackerPool,
    FirecrackerVM,
    build_exec_request,
)
