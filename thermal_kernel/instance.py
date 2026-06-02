"""Thermal instance dataclass + registry (split from thermal_kernel/__init__.py)."""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThermalInstance:
    """A 2D thermal scenario for the phase-field oracle."""
    name: str
    description: str
    grid_size: int
    cold_temp: float
    hot_temp: float
    cold_radius: float
    D: float
    dt: float
    n_steps: int
    initial_phi: float
    target_width: float


# ---------------------------------------------------------------------------
# Instance registry
# ---------------------------------------------------------------------------

_THERMAL_INSTANCE_CONFIGS: dict[str, dict] = {
    "freeze_spot": {
        # 200 steps so cold zone isn't trivially fully frozen — creates selection pressure
        "description": "Spell creates cold disk (T=0.25) in warm river (T=0.75). Ice grows outward.",
        "grid_size": 64, "cold_temp": 0.25, "hot_temp": 0.75, "cold_radius": 18.0,
        "D": 10.0, "dt": 0.018, "n_steps": 200, "initial_phi": 0.02, "target_width": 2.0,
    },
    "melt_spot": {
        "description": "Hot disk (T=0.75) melts frozen river (T=0.25). Liquid zone grows outward.",
        "grid_size": 64, "cold_temp": 0.25, "hot_temp": 0.75, "cold_radius": 18.0,
        "D": 10.0, "dt": 0.018, "n_steps": 200, "initial_phi": 0.98, "target_width": 2.0,
    },
    "deep_freeze": {
        "description": "Intense cold disk (T=0.10) in warm environment (T=0.80). Large driving force.",
        "grid_size": 64, "cold_temp": 0.10, "hot_temp": 0.80, "cold_radius": 15.0,
        "D": 10.0, "dt": 0.018, "n_steps": 150, "initial_phi": 0.02, "target_width": 2.0,
    },
    "gentle_freeze": {
        "description": "Near-equilibrium freeze (T=0.40) — tests sensitivity to small undercooling.",
        "grid_size": 64, "cold_temp": 0.40, "hot_temp": 0.60, "cold_radius": 20.0,
        "D": 10.0, "dt": 0.018, "n_steps": 400, "initial_phi": 0.02, "target_width": 3.0,
    },
}


def generate_instance(name: str) -> ThermalInstance:
    if name not in _THERMAL_INSTANCE_CONFIGS:
        raise ValueError(f"Unknown thermal instance: {name!r}. "
                         f"Known: {list(_THERMAL_INSTANCE_CONFIGS)}")
    cfg = _THERMAL_INSTANCE_CONFIGS[name]
    return ThermalInstance(name=name, **cfg)
