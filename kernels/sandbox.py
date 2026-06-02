"""Sandbox gate — single source of truth for the untrusted-code executor.

Consolidated from autobench.tsp_kernel (Phase 1 of the kernel restructuring).
The 6 sibling kernels (latent, phase, sdf, sph, terrain, thermal) used to
`from ..tsp_kernel import ensure_sandboxed_executor` — a backwards edge
between siblings that left us with 6 near-identical `ensure_executor` shims
on top. They now import the function (and the error) from `autobench.kernels`.

The two kernels that need a custom executor build (TSP, SDF — the only two
that have historically been run with isolation on this stack) keep their
own variants; the default for the other six is the `tsp_kernel`-style
gVisor-or-firecracker gate, which is the right call for compile-and-run of
untrusted C++ candidates.

Note: this module deliberately does NOT import from `tsp_kernel`. That
would re-introduce the cycle we are trying to break.
"""

from __future__ import annotations

import logging

from ..engines.sandbox import SandboxedExecutor  # autobench.sandbox — package-relative

logger = logging.getLogger(__name__)


class UnsafeSandboxError(RuntimeError):
    """Raised when autonomy is requested but no isolating sandbox is available."""


def ensure_sandboxed_executor(
    *,
    allow_unsandboxed: bool = False,
    max_memory_mb: int = 512,
    cpu_limit: int = 2,
) -> SandboxedExecutor:
    """Build the executor used to compile+run untrusted candidates, gated.

    SandboxedExecutor silently downgrades ``gvisor`` -> ``subprocess`` when the
    gVisor health check fails. We therefore inspect the *post-init* sandbox_type:
    if it is not an isolating tier we refuse, because running unattended,
    LLM-generated C++ without network/filesystem isolation is the exact trust
    hazard this kernel exists to avoid. ``allow_unsandboxed`` is the explicit,
    operator-set override for trusted local experiments.
    """
    executor = SandboxedExecutor(
        sandbox_type="gvisor",
        max_memory_mb=max_memory_mb,
        cpu_limit=cpu_limit,
    )
    if executor.sandbox_type not in ("gvisor", "firecracker"):
        if not allow_unsandboxed:
            raise UnsafeSandboxError(
                "Kernel refuses to compile/run untrusted candidate code without "
                f"an isolating sandbox (got '{executor.sandbox_type}'). Install/enable "
                "gVisor (rootless runsc) or pass allow_unsandboxed=True for a trusted, "
                "attended run."
            )
        logger.warning(
            "SANDBOX DEGRADED: running untrusted candidate code under '%s' "
            "(no isolation) because allow_unsandboxed=True", executor.sandbox_type,
        )
    return executor
