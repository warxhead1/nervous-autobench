"""Latent kernel problem instances.

LatentInstance dataclass, the instance config registry, and generate_instance.
Moved verbatim from the package __init__ (behaviour-preserving file split).
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------

@dataclass
class LatentInstance:
    name: str
    description: str
    grid_size: int
    cold_temp: float      # initial T in cold zone
    hot_temp: float       # initial T in hot zone
    cold_radius: float    # cold zone radius (cells)
    D_phi: float          # phase diffusivity
    D_T: float            # thermal diffusivity
    L: float              # latent heat coefficient
    dt: float
    n_steps: int
    initial_phi: float    # 0.02=all liquid, 0.98=all solid
    target_width: float   # target interface width (cells)

    @property
    def phi_eq(self) -> float:
        """Theoretical equilibrium coverage from energy balance."""
        if self.initial_phi < 0.5:
            return (0.5 - self.cold_temp) / self.L   # freeze: φ_eq = ΔT/L
        else:
            return (self.hot_temp - 0.5) / self.L    # melt: φ_melt = ΔT/L


# ---------------------------------------------------------------------------
# Instance registry
# ---------------------------------------------------------------------------

_LATENT_INSTANCE_CONFIGS: dict[str, dict] = {
    "freeze_latent": {
        "description": (
            "Freeze spot with latent heat: T rises as ice forms. "
            "Equilibrium coverage φ_eq=(0.5-0.25)/0.4=0.625 — only 62.5% of cold zone freezes. "
            "D_phi=4 gives natural interface width ~2.8 cells; target_width=3 is achievable."
        ),
        "grid_size": 64, "cold_temp": 0.25, "hot_temp": 0.75, "cold_radius": 18.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.4, "dt": 0.020,
        "n_steps": 500, "initial_phi": 0.02, "target_width": 3.0,
    },
    "melt_latent": {
        "description": (
            "Melt spot with latent heat: T drops as ice melts. "
            "Equilibrium melt coverage=(0.75-0.5)/0.4=0.625 — only 62.5% of hot zone melts. "
            "D_phi=4 matches freeze_latent for symmetric difficulty."
        ),
        "grid_size": 64, "cold_temp": 0.25, "hot_temp": 0.75, "cold_radius": 18.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.4, "dt": 0.020,
        "n_steps": 500, "initial_phi": 0.98, "target_width": 3.0,
    },
    "gentle_latent": {
        "description": (
            "Near-equilibrium freeze (T_cold=0.40) with latent heat. "
            "φ_eq=(0.5-0.40)/0.3=0.333 — tiny undercooling, very sensitive oracle."
        ),
        "grid_size": 64, "cold_temp": 0.40, "hot_temp": 0.60, "cold_radius": 20.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.3, "dt": 0.020,
        "n_steps": 700, "initial_phi": 0.02, "target_width": 3.0,
    },
    "frontier_gradual": {
        "description": (
            "Gradual frontier freeze: cold_temp=0.30, L=0.6 → φ_eq=0.333. "
            "Strong latent heat suppression forces gentle self-regulation. "
            "Velocity oracle is most discriminating here: v_stefan = D_T*(0.5-0.30)/(0.6*r_mid)."
        ),
        "grid_size": 64, "cold_temp": 0.30, "hot_temp": 0.70, "cold_radius": 18.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.6, "dt": 0.020,
        "n_steps": 500, "initial_phi": 0.02, "target_width": 3.0,
    },
}


def generate_instance(name: str) -> LatentInstance:
    if name not in _LATENT_INSTANCE_CONFIGS:
        raise ValueError(f"Unknown latent instance: {name!r}. "
                         f"Known: {list(_LATENT_INSTANCE_CONFIGS)}")
    cfg = _LATENT_INSTANCE_CONFIGS[name]
    return LatentInstance(name=name, **cfg)
