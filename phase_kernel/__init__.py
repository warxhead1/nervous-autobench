"""Phase field kernel — FunSearch evolution of matter-state transition driving forces.

# Domain: Allen-Cahn Phase Field  float reaction(float phi, float temp)

Evolves the bulk driving force of a phase field model against a 1D Allen-Cahn
time-stepper oracle.  The evolved function encodes the thermodynamic force
that drives matter transitions: liquid ↔ solid (water ↔ ice), vapour ↔ liquid,
or any two-phase system.

## The math — Allen-Cahn equation

  ∂φ/∂t = D·∇²φ + reaction(φ, T)

  φ ∈ [0,1]: order parameter (0 = liquid/disordered, 1 = solid/ordered)
  T ∈ [0,1]: dimensionless temperature (0 = absolute zero, 0.5 = melting point,
                                         1 = far above melting)

  The classical Allen-Cahn driving force is:
      reaction(φ, T) = −W′(φ) + m·(T − 0.5)
  where W(φ) = φ²(1−φ)²  (double-well potential)
  and W′(φ) = 2φ(1−φ)(2φ−1) = 4φ³ − 6φ² + 2φ

  Physical requirements:
    1. Equilibrium at φ=0 and φ=1 for T=0.5 (balanced at melting):
           reaction(0, 0.5) ≈ 0  and  reaction(1, 0.5) ≈ 0
    2. Correct temperature response:
           T > 0.5 → liquid preferred → pushes φ → 0 when φ > 0
           T < 0.5 → solid preferred  → pushes φ → 1 when φ < 1
    3. Interface convergence: from a sharp step, φ(x) should converge
       to a smooth tanh-profile within O(100) time steps.

## Oracle: 1D time-stepper

  1. Initialise φ[i]: step function (φ=0 for i<N/2, φ=1 for i≥N/2)
  2. Run N_steps of explicit Euler:
       φ[i] += dt · (D · ∇²φ[i] + reaction(φ[i], T))
  3. Measure interface quality:
       tanh_score = correlation of final φ(x) with ideal tanh profile
       equil_score = |φ[0:N/4]| + |1−φ[3N/4:N]|  (ends reach equilibrium)
       width_score = interface width vs target (too narrow or too wide = penalty)
  4. Fitness = 0.5·tanh_score + 0.3·equil_score + 0.2·width_score

  For the "melting" instance (T=0.7): solid half should shrink → φ→0.
  For the "freezing" instance (T=0.3): solid half should grow → φ→1.
  For the "balanced" instance (T=0.5): interface should sharpen without moving.

## Hard preconditions

  Stability: |reaction(φ, T)| < 50 for all φ∈[0,1], T∈[0,1].
  Finite: no NaN/Inf during time-stepping.

## TEngine integration path

  The evolved reaction(φ, T) function drops directly into a Slang compute
  shader that implements per-voxel phase field evolution for TEngine's
  temperature field layer:

    for each voxel:
      float T = temperature_field[voxel];
      float phi = phase_field[voxel];
      float dphi = D * laplacian(phi, voxel) + reaction(phi, T);
      phase_field[voxel] = clamp(phi + dt * dphi, 0.0, 1.0);

  The evolved function provides the thermodynamic driving force — the part
  that currently does not exist in TEngine.  This is the mathematical substrate
  for "walk up to a river and cast a freeze spell".

## Connection to other kernels

  SPH kernel   → fluid smoothing kernel (how particles reconstruct density)
  Terrain kernel → riverbed geometry that determines where water accumulates
  This kernel   → whether the water at each voxel is ice or liquid
  Phase field  → SDF kernel: ice boundary can be raymarched as evolved SDF
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Behavior-preserving file split. Implementation lives in submodules:
#   instance.py — PhaseInstance, _PHASE_INSTANCE_CONFIGS, generate_instance
#   oracle.py   — C++ evaluator, PHASE_FUNCTION_SIGNATURE, seeds,
#                 get_seed_programs, build_candidate_source
#   scoring.py  — evaluate_on_instance (sandbox compile/run + fitness)
#   loop.py     — PhaseKernel (registered via @register_kernel("phase"))
# This module re-exports the exact public surface cli.py imports from `.`.
# ---------------------------------------------------------------------------

from ..kernels import ensure_sandboxed_executor
from .instance import PhaseInstance, _PHASE_INSTANCE_CONFIGS, generate_instance
from .oracle import (
    PHASE_FUNCTION_SIGNATURE,
    SEED_PHASE_PROGRAMS,
    build_candidate_source,
    get_seed_programs,
)
from .scoring import evaluate_on_instance
from .loop import PhaseKernel  # noqa: F401 — import triggers @register_kernel("phase")

__all__ = [
    "PhaseKernel",
    "PhaseInstance",
    "evaluate_on_instance",
    "ensure_sandboxed_executor",
    "generate_instance",
    "get_seed_programs",
    "build_candidate_source",
    "_PHASE_INSTANCE_CONFIGS",
    "PHASE_FUNCTION_SIGNATURE",
    "SEED_PHASE_PROGRAMS",
]
