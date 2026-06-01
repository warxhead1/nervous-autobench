"""autobench.engines — Sandboxed execution engines (sandbox, Firecracker, GPU).

Phase 2A of the autobench restructuring. This subpackage consolidates the
five execution backends that used to live as sibling modules at the
autobench package root:

    autobench/engines/
    ├── sandbox.py         # subprocess + gVisor + cgroup2 SandboxedExecutor
    ├── firecracker_vm.py  # Firecracker microVM pool (lifecycle, vsock exec)
    ├── guest_agent.py     # in-VM stdlib-only command agent (vsock port 8888)
    ├── shader_executor.py # moderngl/EGL headless GLSL renderer
    └── sdf_tracer.py      # g++ wrapper for SDF → PNG rendering

Public re-exports:

    from autobench.engines import (
        # sandbox.py
        SandboxedExecutor, Verdict, ExecutionResult, compile_and_run,
        # firecracker_vm.py
        FirecrackerVM, FirecrackerPool, FirecrackerAPI, FirecrackerError,
        build_exec_request,
        # guest_agent.py
        VSOCK_PORT,
    )
"""

from __future__ import annotations

# sandbox.py — multi-language execution + verdict types
from .sandbox import (
    ExecutionResult,
    SandboxedExecutor,
    Verdict,
    compile_and_run,
)

# firecracker_vm.py — microVM lifecycle + pool + vsock client
from .firecracker_vm import (
    FirecrackerAPI,
    FirecrackerError,
    FirecrackerPool,
    FirecrackerVM,
    build_exec_request,
)

# guest_agent.py — in-VM agent. VSOCK_PORT is the canonical vsock port (8888).
from .guest_agent import VSOCK_PORT

__all__ = [
    "ExecutionResult",
    "FirecrackerAPI",
    "FirecrackerError",
    "FirecrackerPool",
    "FirecrackerVM",
    "SandboxedExecutor",
    "VSOCK_PORT",
    "Verdict",
    "build_exec_request",
    "compile_and_run",
]
