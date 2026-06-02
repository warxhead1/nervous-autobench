"""Terrain benchmark instance definitions and configurations."""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Terrain benchmark instance
# ---------------------------------------------------------------------------

@dataclass
class TerrainInstance:
    """A geological profile instance: target Hurst exponent + normalised slope.

    The oracle samples terrain(p) on a 32×32 grid and measures how closely
    the statistical properties match those of real terrain at the given H.
    """
    name: str
    description: str
    target_hurst: float        # Target Hurst exponent H ∈ (0,1)
    target_norm_slope: float   # Target mean(|∇h|)/std(h) — scale-invariant roughness
    grid_size: int = 32        # Sampling grid (N×N)


# ---------------------------------------------------------------------------
# Instance configurations
# ---------------------------------------------------------------------------

_TERRAIN_INSTANCE_CONFIGS: dict[str, dict] = {
    # Targets are offset from natural fBm cluster (fbm_standard H≈0.86, fbm_smooth H≈0.66,
    # ridge_fbm H≈0.79) so no single seed trivially saturates the oracle (>0.92 is fine;
    # >0.99 means evolution has nowhere to go and plateaus immediately).
    "rolling_hills": {
        "target_hurst": 0.88,        # Above fbm_standard (0.861) → needs even smoother
        "target_norm_slope": 0.17,
        "description": "Rolling countryside hills — very smooth, coherent, low-relief landscape",
    },
    "mountain_peaks": {
        "target_hurst": 0.63,        # Below fbm_smooth (0.661) → needs rougher
        "target_norm_slope": 0.30,
        "description": "Rugged mountain peaks — rough ridges, high persistence",
    },
    "eroded_badlands": {
        "target_hurst": 0.48,        # Far below all seeds → requires novel roughness structure
        "target_norm_slope": 0.40,
        "description": "Heavily eroded badlands — deep channels, maximum roughness",
    },
    "river_valley": {
        "target_hurst": 0.77,        # Between fbm_smooth (0.66) and fbm_standard (0.86)
        "target_norm_slope": 0.22,
        "description": "River valley — mixed relief: flat valley floor, steep flanking walls",
    },
    "volcanic_plateau": {
        "target_hurst": 0.57,        # Between eroded and mountain, pushes toward lower H
        "target_norm_slope": 0.33,
        "description": "Volcanic plateau — lava flows and calderas, medium-high roughness",
    },
}


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_instance(name: str) -> TerrainInstance:
    """Return a TerrainInstance from the named configuration."""
    cfg = _TERRAIN_INSTANCE_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown terrain instance: {name!r}. "
                         f"Available: {list(_TERRAIN_INSTANCE_CONFIGS)}")
    return TerrainInstance(
        name=name,
        description=cfg["description"],
        target_hurst=cfg["target_hurst"],
        target_norm_slope=cfg["target_norm_slope"],
    )
