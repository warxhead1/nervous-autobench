"""Back-compat shim. Prefer `autobench.engines.sandbox`.

Phase 2A of the autobench restructuring moved ``sandbox.py`` into the
``autobench.engines`` subpackage. This module re-exports the public API
so legacy ``from autobench.sandbox import …`` call sites keep working.
"""

from .engines.sandbox import (  # noqa: F401
    ExecutionResult,
    SandboxedExecutor,
    Verdict,
    compile_and_run,
)
