"""Terrain height-field kernel — FunSearch evolution of geological height functions.

# Domain: Terrain Height Field  float terrain(vec2 p)

Evolves a 2D height function against a *geological class-membership oracle*:
the oracle does NOT compare pixel values — it measures whether the evolved
function has the statistical signature of real terrain at a given target
Hurst exponent H.

## Oracle: Hurst-exponent + normalised-slope (sufficient statistic)

  1. Sample terrain(p) on a 32×32 grid over [-1,1]².
  2. Structure function SF(r) = E[|h(x+r)−h(x)|²] at lags r=1,2,3,4,6,8.
  3. Log-log regression slope = 2H  →  H_est.
  4. Normalised slope = mean(|∇h|) / std(h)  — scale-invariant roughness.

  Fitness = 0.6·hurst_score + 0.3·slope_score + 0.1·range_score

  hurst_score = exp(−8·(H_est − H_target)²)   — Gaussian gate around target
  slope_score = exp(−5·(ns − ns_target)² / (ns_target² + 0.01))
  range_score = min(1, max_h − min_h) / 0.3)  — degenerate-constant guard

This is the **generative membership oracle** for terrain: any function with the
correct fractal persistence belongs to the class, regardless of the specific
hash constants used. This fixes the SSIM hash-realization trap.

## Geological instances (target H values)

  rolling_hills    H=0.82  gentle, coherent — low-relief rural landscape
  mountain_peaks   H=0.66  rugged, persistent ridges — alpine terrain
  eroded_badlands  H=0.48  rough, channelled — high-erosion semiarid terrain
  river_valley     H=0.74  mixed — flat valley floor + steep flanking walls
  volcanic_plateau H=0.59  medium roughness with abrupt calderas / lava flows

## TEngine integration path

  float terrain(vec2 p)  maps directly to a HLSL/Slang compute shader.
  The terrain height function plugs into the height-map generator that feeds
  the SVDAG build pass.  Unlike the SDF kernel (which needs an SVDAG bake),
  terrain height is consumed by the terrain tessellation stage — integration
  is a 1-pass shader swap.

## Connection to other kernels

  SDF kernel  →  rock-formation shapes that cut into terrain height
  Noise kernel →  fine-detail overlay (normal perturbation) on terrain surface
  SPH kernel   →  water flowing over terrain (riverbed geometry from terrain)
  Phase kernel →  water→ice state transition driven by temperature field above terrain
"""
from __future__ import annotations

# Re-export the kernel sandbox factory so `cli.py` (and sibling code) can
# import it from this package namespace.
from ..kernels import ensure_sandboxed_executor

from .instance import (
    TerrainInstance,
    _TERRAIN_INSTANCE_CONFIGS,
    generate_instance,
)
from .oracle import (
    TERRAIN_FUNCTION_SIGNATURE,
    SEED_TERRAIN_PROGRAMS,
    get_seed_programs,
)
from .scoring import (
    _TERRAIN_EVALUATOR_CPP,
    build_candidate_source,
    evaluate_on_instance,
)
from .loop import TerrainKernel  # noqa: F401 — import triggers @register_kernel("terrain")

__all__ = [
    "TerrainKernel",
    "TerrainInstance",
    "evaluate_on_instance",
    "ensure_sandboxed_executor",
    "generate_instance",
    "get_seed_programs",
    "build_candidate_source",
    "TERRAIN_FUNCTION_SIGNATURE",
    "SEED_TERRAIN_PROGRAMS",
    "_TERRAIN_INSTANCE_CONFIGS",
    "_TERRAIN_EVALUATOR_CPP",
]
