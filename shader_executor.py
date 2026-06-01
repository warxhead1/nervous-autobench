"""Back-compat shim. Prefer `autobench.engines.shader_executor`.

Phase 2A of the autobench restructuring moved ``shader_executor.py`` into
the ``autobench.engines`` subpackage. This module re-exports the public
API so legacy ``from autobench.shader_executor import …`` call sites keep
working.
"""

from .engines.shader_executor import (  # noqa: F401
    ComputeRunResult,
    ShaderExecutor,
    ShaderRunResult,
    verdict_from_ssim,
)
