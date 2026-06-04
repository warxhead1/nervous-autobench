"""Oasis kernel — FunSearch evolution of a shallow-water flow law into a
dynamic, breathing desert oasis.

# Domain: 2D shallow-water hydrology   float flux(float dhead, float depth, float visc)

The evolved function is the constitutive flow law of a thin water film over a
dune basin. The oracle drives the field with an artesian groundwater spring and
a day/night evaporation cycle, then measures whether the flow self-organises into
a stable pool that breathes — an oasis.

## The dynamic

  ∂w/∂t = -∇·flux(∇(H+w), w, visc) + spring(artesian) - evap(day, depth)

  H: dune-basin terrain (fixed per instance)   w: water depth (evolves)
  flux: evolved law — thin-film lubrication q = w³·∇head/(3·visc) is the reference

## Oracle (multiplicatively gated — every property must hold)

  basin_capture · (0.4+0.6·stability) · pool_fraction · (0.4+0.6·breathing)
    basin_capture: water pools in the low basin
    stability:     converges to a steady level (no drift / flood)
    pool_fraction: a pool a few % of the basin
    breathing:     bounded day/night shoreline oscillation — the *dynamic*

Calibrated derisk: lubrication seed ≈ 0.81, no-flow ≈ 0.00.

## Instances — viscosity × terrain

  clear_spring (low visc), muddy_seep (high visc), wide_pan (broad shallow).
"""
from __future__ import annotations

from ..kernels import ensure_sandboxed_executor
from .instance import OasisInstance, _OASIS_INSTANCE_CONFIGS, generate_instance
from .oracle import (
    OASIS_FUNCTION_SIGNATURE,
    SEED_OASIS_PROGRAMS,
    build_candidate_source,
    get_seed_programs,
)
from .scoring import evaluate_on_instance
from .render import render_oasis
from .loop import OasisKernel  # noqa: F401 — import triggers @register_kernel("oasis")

__all__ = [
    "OasisKernel",
    "OasisInstance",
    "evaluate_on_instance",
    "ensure_sandboxed_executor",
    "generate_instance",
    "get_seed_programs",
    "build_candidate_source",
    "render_oasis",
    "_OASIS_INSTANCE_CONFIGS",
    "OASIS_FUNCTION_SIGNATURE",
    "SEED_OASIS_PROGRAMS",
]
