"""Latent-heat-coupled 2D Allen-Cahn kernel.

# Domain: Coupled Phase-Field + Thermal Diffusion
# Function: float reaction(float phi, float temp, float lap_T)

Evolves the Allen-Cahn driving force in a setting where temperature is NOT fixed —
it evolves via coupled PDEs:

  ∂φ/∂t = D_φ·∇²φ + reaction(φ, T, ∇²T)
  ∂T/∂t = D_T·∇²T + L·∂φ/∂t

Latent heat L > 0: freezing (∂φ/∂t > 0) releases heat → T rises → drive weakens.
This is self-limiting: ice can only form until local T rises to the melting point 0.5.

## Why this oracle is hard

With fixed T (thermal_kernel), any correct-sign function trivially scores 1.0 because
the entire cold zone freezes and the hot zone melts.

With coupled T, the equilibrium coverage is:
  φ_eq = (0.5 − T_cold_initial) / L

For T_cold=0.25, L=0.4:  φ_eq = 0.625  (only 62.5% of cold zone freezes)
For T_cold=0.30, L=0.6:  φ_eq = 0.333  (only 33% freezes — very sensitive)

Functions that freeze too fast overshoot equilibrium → T rises above 0.5 → drive reverses
→ ice melts → oscillation → low score.
Functions that freeze at exactly the right rate hit the equilibrium cleanly → high score.

## Third argument: lap_T = ∇²T

This gives the function LOCAL THERMAL CONTEXT:
  lap_T > 0: T increasing at this point (heat flowing in, latent heat releasing nearby)
             → function can slow down to avoid overshoot
  lap_T < 0: T decreasing (heat flowing out, cold diffusing in)
             → function can speed up, more undercooling available
  lap_T ≈ 0: bulk region (already equilibrated)

Classical Allen-Cahn ignores lap_T. A function that uses it can stabilise the front.

## Oracle scoring (v2 — velocity-aware)

  T_balance_score  = Gaussian(mean_T_cold − 0.5, σ=0.10)     weight 0.30
  velocity_score   = exp(-(log(v/v_stefan))^2 / 0.25)         weight 0.30
                     v_stefan = D_T*(0.5-cold_temp)/(L*r_mid)
                     step(v>0): retreating fronts score 0
  sharpness_score  = narrow interface reward (target_width=3)  weight 0.25
  retention_score  = opposite zone stays in correct phase      weight 0.15

v2 changes vs v1:
  - D_phi reduced 10→4: natural AC width sqrt(4/0.5)≈2.8 cells, target_width=3 achievable
  - T_balance σ widened 0.06→0.10: correct equilibria near T=0.48-0.52 no longer penalized
  - T_balance weight 0.50→0.30: corroborating signal, not dominant
  - velocity_score added (0.30): rewards Stefan self-regulation; retreating fronts score 0
  - sharpness weight 0.35→0.25: less dominant; total still sums to 1.0
  - frontier_gradual instance added: cold_temp=0.30, L=0.6, φ_eq=0.333

Seed calibration (measured at D_phi=4, dt=0.020, freeze_latent):
  interface_thermal seed: 0.83 vs ZERO_REACTION 0.48 → gap +0.35  ✓
  classical_latent seed:  0.54 vs ZERO_REACTION 0.48 → gap +0.06  (marginal)
  gen21 discovered:       0.90 vs ZERO_REACTION 0.48 → gap +0.42  ✓
  WRONG_SIGN: collapsed → 0.001 (well below all correct-sign seeds) ✓

  Note: classical AC is a weak seed at D_phi=4 — its early-phase velocity≈0 because
  the undercooling is rapidly suppressed by latent heat before step n/4. This is physical
  (the oracle correctly penalizes slow classical AC) but means classical barely beats zero.
  Interface-amplified seeds (interface_thermal, gen21) show large gaps vs zero.

Decomposed into cohesive submodules (behaviour-preserving file split):
  - instance.py  — LatentInstance, _LATENT_INSTANCE_CONFIGS, generate_instance
  - scoring.py   — C++ evaluator source, LATENT_FUNCTION_SIGNATURE,
                   build_candidate_source, evaluate_on_instance
  - oracle.py    — _SEEDS, _DIAGNOSTIC_SEEDS, get_seed_programs,
                   get_diagnostic_seeds
  - loop.py      — LatentKernel (FunSearchKernel subclass); importing it fires
                   @register_kernel("latent")

This package re-exports the full public surface. `compile_and_run`,
`ensure_sandboxed_executor`, and `Verdict` live here as package attributes so
scoring.py (and any test patch) resolves them through this package at call time.
"""
from __future__ import annotations

import logging

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, register_kernel,
)
from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run

logger = logging.getLogger(__name__)

# Submodules — imported AFTER Verdict / compile_and_run are bound above, because
# scoring.py resolves those names back through this package at call time.
from .instance import (  # noqa: E402
    LatentInstance,
    _LATENT_INSTANCE_CONFIGS,
    generate_instance,
)
from .scoring import (  # noqa: E402
    _LATENT_EVALUATOR_CPP,
    LATENT_FUNCTION_SIGNATURE,
    build_candidate_source,
    evaluate_on_instance,
)
from .oracle import (  # noqa: E402
    _SEEDS,
    _DIAGNOSTIC_SEEDS,
    get_seed_programs,
    get_diagnostic_seeds,
)
from .loop import LatentKernel  # noqa: E402  (fires @register_kernel("latent"))

__all__ = [
    # sandbox / kernel base re-exports
    "FunSearchKernel", "KernelConfig", "CandidateProgram", "Island",
    "ensure_sandboxed_executor", "register_kernel", "Verdict",
    "SandboxedExecutor", "compile_and_run",
    # instance
    "LatentInstance", "_LATENT_INSTANCE_CONFIGS", "generate_instance",
    # scoring
    "_LATENT_EVALUATOR_CPP", "LATENT_FUNCTION_SIGNATURE",
    "build_candidate_source", "evaluate_on_instance",
    # oracle
    "_SEEDS", "_DIAGNOSTIC_SEEDS", "get_seed_programs", "get_diagnostic_seeds",
    # loop
    "LatentKernel",
]
