"""autobench.kernels — FunSearch/EoH kernel base + per-kernel subpackage consolidation.

Phase 1 of the autobench restructuring. This subpackage used to live as
ad-hoc boilerplate in eight sibling kernel packages (``autobench/tsp_kernel``,
``autobench/sdf_kernel``, …). The consolidated home is here:

    autobench/kernels/
    ├── __init__.py            # public re-exports (this file)
    ├── base.py                # FunSearchKernel ABC + ConsolidatedPrior + helpers
    ├── config.py              # KernelConfig — single source of truth
    ├── sandbox.py             # ensure_sandboxed_executor + UnsafeSandboxError
    ├── bridge.py              # nervous_kernel_bridge (relocated, unchanged)
    ├── eval.py                # nervous_kernel_eval (relocated, unchanged)
    └── cli.py                 # unified CLI dispatcher (NEW)

Public re-exports — the same names that used to come from ``autobench.kernel_base``:

    from autobench.kernels import (
        FunSearchKernel,        # the ABC
        KernelConfig,           # the config dataclass
        CandidateProgram,       # the evolved-program record
        Island,                 # the island-model container
        ConsolidatedPrior,      # the cross-run T-vector store
        new_ulid,               # ULID generator
        make_local_llm_fn,      # CPU-fallback LLM factory
        ensure_sandboxed_executor,  # sandbox gate (replaces per-kernel ensure_executor)
        UnsafeSandboxError,
        KERNEL_REGISTRY,        # name → FunSearchKernel subclass (for the unified CLI)
        register_kernel,        # decorator — register a FunSearchKernel subclass
    )
"""

from .base import (
    KERNEL_REGISTRY,
    CandidateProgram,
    ConsolidatedPrior,
    FunSearchKernel,
    Island,
    make_local_llm_fn,
    new_ulid,
    register_kernel,
)
from .config import KernelConfig
from .sandbox import UnsafeSandboxError, ensure_sandboxed_executor

# bridge and eval are imported on demand (they pull in HTTP/urllib + GPU-bridge
# paths that not every consumer wants). Make them available as
# ``autobench.kernels.bridge`` / ``autobench.kernels.eval`` modules.
from . import bridge, eval  # noqa: F401

__all__ = [
    "CandidateProgram",
    "ConsolidatedPrior",
    "FunSearchKernel",
    "Island",
    "KERNEL_REGISTRY",
    "KernelConfig",
    "UnsafeSandboxError",
    "bridge",
    "ensure_sandboxed_executor",
    "eval",
    "make_local_llm_fn",
    "new_ulid",
    "register_kernel",
]
