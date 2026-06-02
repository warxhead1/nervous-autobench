"""Phase field benchmark instances — thermodynamic scenario configs.

Behavior-preserving split of phase_kernel/__init__.py: dataclass, instance
config table, and instance generation moved here verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Phase field benchmark instance
# ---------------------------------------------------------------------------

@dataclass
class PhaseInstance:
    """A thermodynamic scenario: temperature + diffusion + run length.

    The oracle initialises a 1D phase field with a sharp liquid/solid interface
    and runs the Allen-Cahn PDE for n_steps, measuring how well the evolved
    reaction() function drives the interface toward the correct equilibrium.
    """
    name: str
    description: str
    temperature: float     # T ∈ [0,1]; 0.5 = melting point
    D: float               # Diffusion coefficient
    n_steps: int           # Time-step count
    dt: float              # Time-step size
    grid_size: int         # 1D grid resolution
    target_width: float    # Target interface width (grid cells)


# ---------------------------------------------------------------------------
# Instance configurations
# ---------------------------------------------------------------------------

_PHASE_INSTANCE_CONFIGS: dict[str, dict] = {
    # Grid spacing dx=1; interface width ≈ sqrt(D) cells; stability: dt < 1/(2*D).
    # D=50  → interface width ≈ 7 cells;  dt_max = 0.010; use dt=0.006.
    # D=100 → interface width ≈ 10 cells; dt_max = 0.005; use dt=0.003.
    "water_ice_freezing": {
        "temperature": 0.30,     # Below melting: ice grows
        "D": 50.0,
        "dt": 0.006,
        "n_steps": 300,
        "grid_size": 64,
        "target_width": 7.0,
        "description": "Water supercooled at T=0.30: ice front grows (φ→1 spreads)",
    },
    "water_ice_melting": {
        "temperature": 0.70,     # Above melting: ice melts
        "D": 50.0,
        "dt": 0.006,
        "n_steps": 300,
        "grid_size": 64,
        "target_width": 7.0,
        "description": "Ice melting at T=0.70: liquid front grows (φ→0 spreads)",
    },
    "phase_balanced": {
        "temperature": 0.50,     # At melting: interface sharpens, doesn't move
        "D": 50.0,
        "dt": 0.006,
        "n_steps": 400,
        "grid_size": 64,
        "target_width": 7.0,
        "description": "Balanced at T=0.50: interface sharpens to equilibrium tanh",
    },
    "rapid_freeze": {
        "temperature": 0.10,     # Deep supercooling: very fast ice growth
        "D": 30.0,
        "dt": 0.008,
        "n_steps": 200,
        "grid_size": 64,
        "target_width": 5.0,
        "description": "Deep supercooling T=0.10: rapid ice growth, narrow interface",
    },
    "slow_melt": {
        "temperature": 0.65,     # Slow melt with wider interface
        "D": 100.0,
        "dt": 0.003,
        "n_steps": 500,
        "grid_size": 64,
        "target_width": 10.0,
        "description": "Slow melt T=0.65, wide interface: gradual phase boundary evolution",
    },
}


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_instance(name: str) -> PhaseInstance:
    cfg = _PHASE_INSTANCE_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown phase instance: {name!r}. "
                         f"Available: {list(_PHASE_INSTANCE_CONFIGS)}")
    return PhaseInstance(
        name=name,
        description=cfg["description"],
        temperature=cfg["temperature"],
        D=cfg["D"],
        dt=cfg["dt"],
        n_steps=cfg["n_steps"],
        grid_size=cfg["grid_size"],
        target_width=cfg["target_width"],
    )
