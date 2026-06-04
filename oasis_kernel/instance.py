"""Oasis benchmark instances — viscosity × terrain scenarios.

The oracle builds a dune-basin terrain, drives a 2D shallow-water field with an
artesian spring + day/night evaporation, and measures whether the evolved flow
law `flux(dhead, depth, visc)` produces a stable, breathing oasis pool. Each
instance fixes a fluid viscosity and a basin shape; the evolved flux must work
across them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OasisInstance:
    name: str
    description: str
    viscosity: float     # kinematic viscosity passed to flux(); higher = sluggish
    spring: float        # artesian conductance at the basin spring
    artesian: float      # head the spring pushes toward (sets pool level)
    evap: float          # base evaporation rate (day-cycled, depth-shielded)
    day_amp: float       # day/night evaporation amplitude (drives the "breath")
    day_period: float    # steps per day/night cycle
    dune_amp: float      # dune ridge amplitude on the basin
    dune_freq: float     # dune spatial frequency
    grid_size: int
    n_steps: int


_OASIS_INSTANCE_CONFIGS: dict[str, dict] = {
    # Canonical oasis: clear, low-viscosity spring water in a dune basin.
    "clear_spring": {
        "viscosity": 0.02, "spring": 4.0, "artesian": 0.50, "evap": 0.0040,
        "day_amp": 0.45, "day_period": 150.0, "dune_amp": 0.07, "dune_freq": 6.5,
        "grid_size": 64, "n_steps": 700,
        "description": "Clear low-viscosity spring in a dune basin — round breathing pool",
    },
    # Sluggish muddy seep: high viscosity, stronger spring to compensate slower flow.
    "muddy_seep": {
        "viscosity": 0.05, "spring": 8.0, "artesian": 0.50, "evap": 0.0035,
        "day_amp": 0.45, "day_period": 150.0, "dune_amp": 0.08, "dune_freq": 6.0,
        "grid_size": 64, "n_steps": 700,
        "description": "Viscous muddy seep — sluggish spreading, slow shoreline breath",
    },
    # Broad shallow pan: flatter basin, higher water level → wider pool.
    "wide_pan": {
        "viscosity": 0.02, "spring": 5.0, "artesian": 0.60, "evap": 0.0050,
        "day_amp": 0.50, "day_period": 150.0, "dune_amp": 0.04, "dune_freq": 5.0,
        "grid_size": 64, "n_steps": 700,
        "description": "Broad shallow playa pan — wide pool, pronounced evaporative breath",
    },
}


def generate_instance(name: str) -> OasisInstance:
    cfg = _OASIS_INSTANCE_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown oasis instance: {name!r}. "
                         f"Available: {list(_OASIS_INSTANCE_CONFIGS)}")
    return OasisInstance(name=name, **cfg)
