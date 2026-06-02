"""2D Thermal-Allen-Cahn kernel — FunSearch evolution of phase-field reaction forces.

# Domain: 2D Coupled Thermal Phase Field   float reaction(float phi, float temp)

Evolves the same reaction(φ, T) signature as the 1D phase_kernel, but evaluated
in a 2D spatial context: a spell creates a cold (or hot) zone; ice nucleates and
spreads outward.  This is the "freeze the river" oracle.

## Sign convention — CRITICAL

  m = 2*(0.5 − T)  →  positive when T < 0.5 (cold → solid preferred)
                   →  negative when T > 0.5 (hot  → liquid preferred)

  Classical: reaction(φ, T) = −W′(φ) + 2*(0.5 − T)
    φ=0, T=0.25: reaction = +0.5  → liquid nucleates into solid ✓
    φ=0, T=0.75: reaction = −0.5  → stays liquid ✓

  NOTE: The 1D phase_kernel used m = 2*(T − 0.5) which is BACKWARDS for
  directional instances.  Only phase_balanced (T=0.5, m=0) was unaffected.
  This kernel uses the correct convention throughout.

## Oracle: 2D time-stepper

  Grid N×N (default 64×64), temperature field fixed (externally imposed spell).

  For freeze_spot:
    T(x,y) = T_cold inside circle r < R_cold, else T_hot
    φ(x,y,0) = 0.02 (almost all liquid) + small nucleation seed at centre
    Run n_steps of explicit Euler, 5-point Laplacian, Neumann BC
    Stability: D·dt/dx² ≤ 0.25 (2D bound, half the 1D bound of 0.5)

  Fitness = 0.4·ice_coverage + 0.4·liquid_hold + 0.2·radial_corr

## Discriminative calibration rule

  Baselines MUST verify three conditions:
    correct-sign seed >> zero-reaction control  (gap ≥ 0.10)
    correct-sign seed >> wrong-sign seed        (gap ≥ 0.10)
  If gap < 0.10, oracle measures shape not dynamics — not yet useful.

------------------------------------------------------------------------------
This module was decomposed into cohesive submodules (behavior-preserving file
split). The public surface is re-exported here so existing imports keep working:

  instance.py  — ThermalInstance, _THERMAL_INSTANCE_CONFIGS, generate_instance
  scoring.py   — _THERMAL_EVALUATOR_CPP, THERMAL_FUNCTION_SIGNATURE,
                 build_candidate_source, evaluate_on_instance
  oracle.py    — _SEEDS, _DIAGNOSTIC_SEEDS, get_seed_programs, get_diagnostic_seeds
  loop.py      — ThermalKernel (registered via @register_kernel("thermal"))
"""
from __future__ import annotations

# Re-export sandbox executor helper expected by cli.py
# (`from . import ensure_sandboxed_executor as ensure_executor`).
from ..kernels import ensure_sandboxed_executor

from .instance import (
    ThermalInstance,
    _THERMAL_INSTANCE_CONFIGS,
    generate_instance,
)
from .scoring import (
    _THERMAL_EVALUATOR_CPP,
    THERMAL_FUNCTION_SIGNATURE,
    build_candidate_source,
    evaluate_on_instance,
)
from .oracle import (
    _SEEDS,
    _DIAGNOSTIC_SEEDS,
    get_seed_programs,
    get_diagnostic_seeds,
)
from .loop import ThermalKernel

__all__ = [
    "ThermalInstance",
    "_THERMAL_INSTANCE_CONFIGS",
    "generate_instance",
    "_THERMAL_EVALUATOR_CPP",
    "THERMAL_FUNCTION_SIGNATURE",
    "build_candidate_source",
    "evaluate_on_instance",
    "_SEEDS",
    "_DIAGNOSTIC_SEEDS",
    "get_seed_programs",
    "get_diagnostic_seeds",
    "ThermalKernel",
    "ensure_sandboxed_executor",
]
