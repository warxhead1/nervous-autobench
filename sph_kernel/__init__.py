"""SPH kernel — FunSearch evolution of smoothing kernels for real-time fluid simulation.

# Domain: SPH Smoothing Kernel W(r, h)

Smoothed Particle Hydrodynamics quality is entirely determined by the choice of
W(r, h): the weighting function that governs how each particle contributes to
field estimates at arbitrary positions. Current state-of-the-art (cubic spline,
Wendland C²) are hand-derived analytical solutions. FunSearch can discover
kernels that outperform these on specific scenarios (real-time density fields,
surface tension, turbulent flow) while respecting the physical invariants.

## Oracle: density-reconstruction MSE

The oracle places N particles in a 3D box with masses proportional to a known
multi-scale Gaussian density field ρ_true(x). At M off-lattice probe points the
evolved kernel reconstructs:

    ρ_est(x) = Σⱼ mⱼ * sph_kernel(|x - xⱼ|, h)

Fitness = 1/(1+MSE) where MSE = mean over probes of (ρ_est - ρ_true)².

This is a **generative membership oracle** — any kernel that correctly
interpolates density fields belongs to the class. A constant offset kernel or
a tent function both satisfy partition-of-unity but fail this oracle because
they don't correctly weight the off-lattice probes.

## Hard preconditions (reject if violated)
- Compact support: |W(h·1.001, h)| < 1e-3   (zero outside support)
- Positivity:       W(0, h) ≥ 0              (physical requirement)

## Why this beats partition+monotone+smooth as oracle
The cubic spline integrates to 1.0 and is monotone and C² by construction —
it scores ~1.0 on a math-properties oracle at gen 0, giving no gradient to
climb. The density-reconstruction oracle is hard because jittered particles
and off-lattice probes make kernel shape (not just normalization) load-bearing.
See: funsearch-sufficient-statistic-theory, funsearch-combined-oracle-sdf-gen0-0997.

## TEngine integration path
The evolved W(r,h) is a scalar function that plugs directly into a TEngine
compute shader for SPH particle simulation:
  for each probe particle: density += neighbor_mass * sph_kernel(dist, h)
No intermediate conversion needed — this is a shorter integration path than
the SDF kernel (which needs SVDAG bake + raymarch pipeline).

## Connection to SDF kernel
High-quality SPH fluid simulation requires correct boundary conditions from
scene SDFs. -∇SDF gives the contact normal; SDF < 0 triggers boundary forces.
The eikonal-valid evolved SDFs from sdf_kernel are the natural boundary
representation for this fluid simulation.

Usage:
    python -m autobench.sph_kernel run --instances gauss_blobs_3d \\
        --generations 60 --islands 6 --population 12 --allow-unsandboxed \\
        --plateau-generations 10 --candidates-per-island 1 --max-concurrent-llm 6

---

This package was decomposed into cohesive submodules (behavior-preserving file
split). The public surface is re-exported here unchanged:

* :mod:`.instance` — :class:`SPHInstance`, ``_INSTANCE_CONFIGS``,
  ``KNOWN_BASELINES``, ``generate_instance``, ``_rho``
* :mod:`.oracle`   — seed programs and ``get_seed_programs``
* :mod:`.scoring`  — C++ evaluator template, ``build_candidate_source``,
  ``evaluate_on_instance``
* :mod:`.loop`     — the registered :class:`SPHKernel`
"""
from __future__ import annotations

# Re-exported from ..kernels for back-compat: cli.py imports
# ``ensure_sandboxed_executor`` from this package namespace.
from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
)
from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run

from .instance import (
    SPHInstance,
    _INSTANCE_CONFIGS,
    KNOWN_BASELINES,
    _rho,
    generate_instance,
)
from .oracle import (
    _CUBIC_SPLINE_SEED,
    _WENDLAND_C2_SEED,
    _QUINTIC_SEED,
    SEED_SPH_PROGRAMS,
    get_seed_programs,
)
from .scoring import (
    _SPH_EVALUATOR_CPP,
    SPH_FUNCTION_SIGNATURE,
    build_candidate_source,
    evaluate_on_instance,
)
# Importing .loop registers SPHKernel under "sph" via @register_kernel.
from .loop import SPHKernel

__all__ = [
    # Kernel class (registered as "sph")
    "SPHKernel",
    # Instances
    "SPHInstance",
    "generate_instance",
    "_INSTANCE_CONFIGS",
    "KNOWN_BASELINES",
    # Scoring
    "evaluate_on_instance",
    "build_candidate_source",
    "SPH_FUNCTION_SIGNATURE",
    # Seeds / oracle
    "get_seed_programs",
    "SEED_SPH_PROGRAMS",
    # Sandbox helper re-exported for cli.py
    "ensure_sandboxed_executor",
]
