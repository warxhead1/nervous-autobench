"""Back-compat shim for the old ``autobench.kernel_base`` import path.

Phase 1 of the autobench kernels restructuring (May 2026) relocated the
FunSearch/EoH kernel base into ``autobench.kernels.base``. The old
``autobench.kernel_base`` module remains importable and re-exports the same
public names so that downstream code (hearth-loom, deer-flow, the per-kernel
``cli.py`` files, and any external scripts) keeps working.

New code should import from ``autobench.kernels`` directly.

Public names re-exported (mirrors ``autobench.kernels``):
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ConsolidatedPrior, new_ulid, make_local_llm_fn.
"""

from autobench.kernels import (  # noqa: F401
    CandidateProgram,
    ConsolidatedPrior,
    FunSearchKernel,
    Island,
    KernelConfig,
    make_local_llm_fn,
    new_ulid,
)
