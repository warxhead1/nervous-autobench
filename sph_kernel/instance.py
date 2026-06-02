"""SPH benchmark instances — density-field generation.

Moved verbatim from ``sph_kernel/__init__.py`` as part of a behavior-preserving
file split. Contains the :class:`SPHInstance` dataclass, the instance config
registry, and the reproducible instance generator.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# SPH benchmark instance
# ---------------------------------------------------------------------------

@dataclass
class SPHInstance:
    """A benchmark instance: a known density field sampled at particle+probe positions.

    Particles carry mass proportional to ρ_true at their jittered position.
    Probes are off-lattice points where we measure reconstruction accuracy.
    The evolved kernel must correctly estimate density at probe positions from
    the particle distribution — which requires the right kernel *shape*, not
    just correct normalization.
    """
    name: str
    description: str
    particles: list[tuple[float, float, float, float]]   # (x, y, z, mass)
    probes: list[tuple[float, float, float, float]]       # (x, y, z, rho_true)
    h: float                                              # smoothing length

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def n_probes(self) -> int:
        return len(self.probes)


# ---------------------------------------------------------------------------
# Instance generation: multi-scale Gaussian density blobs
# ---------------------------------------------------------------------------

# Blob parameters: (amplitude, cx, cy, cz, sigma) — deterministic, fixed layout
_INSTANCE_CONFIGS: dict[str, dict] = {
    "gauss_blobs_3d": {
        "description": "Three Gaussian density blobs at different scales, jittered particles",
        "blobs": [
            (2.5,  0.20,  0.18,  0.12, 0.14),
            (1.8, -0.22, -0.18,  0.08, 0.11),
            (3.1,  0.08, -0.28, -0.17, 0.09),
        ],
        "domain": 0.55,   # half-width of particle placement box
        "probe_domain": 0.45,  # slightly inset probe box (avoid boundary effects)
        "n_particles": 150,
        "n_probes": 80,
        "h": 0.40,
        "rng_seed": 42,
    },
    "sharp_gradients": {
        "description": "Narrow Gaussian blobs with steep density gradients — harder instance",
        "blobs": [
            (4.0,  0.15,  0.10,  0.05, 0.07),
            (3.0, -0.18, -0.12,  0.08, 0.06),
            (2.5,  0.05, -0.20, -0.15, 0.08),
        ],
        "domain": 0.40,
        "probe_domain": 0.35,
        "n_particles": 120,
        "n_probes": 60,
        "h": 0.35,
        "rng_seed": 137,
    },
    # ---- h-sweep directions: same blob field, different smoothing lengths ----
    # Small h (fine resolution): kernel must be tight to resolve structure
    "h020_fine": {
        "description": "Fine resolution h=0.20 — kernel must resolve sub-particle-spacing structure",
        "blobs": [
            (2.5,  0.20,  0.18,  0.12, 0.14),
            (1.8, -0.22, -0.18,  0.08, 0.11),
            (3.1,  0.08, -0.28, -0.17, 0.09),
        ],
        "domain": 0.45, "probe_domain": 0.38, "n_particles": 60,
        "n_probes": 40, "h": 0.20, "rng_seed": 201,
    },
    "h030_medium": {
        "description": "Medium-fine h=0.30 — intermediate smoothing regime",
        "blobs": [
            (2.5,  0.20,  0.18,  0.12, 0.14),
            (1.8, -0.22, -0.18,  0.08, 0.11),
            (3.1,  0.08, -0.28, -0.17, 0.09),
        ],
        "domain": 0.50, "probe_domain": 0.42, "n_particles": 100,
        "n_probes": 60, "h": 0.30, "rng_seed": 202,
    },
    "h050_coarse": {
        "description": "Coarse h=0.50 — dense particle overlap, kernel must spread smoothly",
        "blobs": [
            (2.5,  0.20,  0.18,  0.12, 0.14),
            (1.8, -0.22, -0.18,  0.08, 0.11),
            (3.1,  0.08, -0.28, -0.17, 0.09),
        ],
        "domain": 0.60, "probe_domain": 0.50, "n_particles": 200,
        "n_probes": 80, "h": 0.50, "rng_seed": 203,
    },
    "h065_very_coarse": {
        "description": "Very coarse h=0.65 — extreme overlap, tests kernel in liquid bulk limit",
        "blobs": [
            (2.5,  0.20,  0.18,  0.12, 0.14),
            (1.8, -0.22, -0.18,  0.08, 0.11),
            (3.1,  0.08, -0.28, -0.17, 0.09),
        ],
        "domain": 0.65, "probe_domain": 0.55, "n_particles": 250,
        "n_probes": 80, "h": 0.65, "rng_seed": 204,
    },
    # ---- Shape-diversity directions: different density field topology ----
    # Single large blob: tests partition-of-unity accuracy
    "single_blob": {
        "description": "Single large Gaussian blob — isolates partition-of-unity correctness",
        "blobs": [(3.0, 0.05, 0.05, 0.05, 0.22)],
        "domain": 0.55, "probe_domain": 0.45, "n_particles": 120,
        "n_probes": 60, "h": 0.40, "rng_seed": 301,
    },
    # Sparse particles: tests low-count regime (most sensitive to kernel shape)
    "sparse_h028": {
        "description": "Sparse particles (n=80) h=0.28 — ~7 neighbors in support, shape-sensitive",
        "blobs": [
            (3.0,  0.15,  0.10,  0.08, 0.18),
            (2.0, -0.20, -0.15, -0.10, 0.14),
        ],
        "domain": 0.45, "probe_domain": 0.38, "n_particles": 80,
        "n_probes": 40, "h": 0.28, "rng_seed": 302,
    },
    # Multi-scale: blobs spanning 3 decades of sigma
    "multi_scale": {
        "description": "Five blobs spanning wide-to-narrow sigma — tests multi-scale reconstruction",
        "blobs": [
            (1.5,  0.25,  0.20,  0.10, 0.22),
            (2.0, -0.20,  0.15, -0.10, 0.14),
            (3.0,  0.10, -0.20,  0.15, 0.09),
            (4.0, -0.05,  0.05, -0.20, 0.06),
            (2.5,  0.18, -0.08,  0.08, 0.17),
        ],
        "domain": 0.55, "probe_domain": 0.45, "n_particles": 150,
        "n_probes": 80, "h": 0.42, "rng_seed": 303,
    },
    # Bimodal: two separated medium blobs — tests kernel locality with measurable contrast
    "bimodal_contrast": {
        "description": "Two medium blobs with 10:1 amplitude contrast — tests no-leakage between peaks",
        "blobs": [
            (5.0,  0.22,  0.00,  0.00, 0.14),   # bright peak
            (0.5, -0.22,  0.00,  0.00, 0.14),   # dim peak
        ],
        "domain": 0.50, "probe_domain": 0.42, "n_particles": 120,
        "n_probes": 60, "h": 0.32, "rng_seed": 304,
    },
    # Layered sheet: high density in XY plane — fluid surface analogue
    "layered_sheet": {
        "description": "Density concentrated near z=0 plane — models fluid free surface",
        "blobs": [
            (4.0,  0.00,  0.00,  0.00, 0.30),  # wide in x,y; narrow implied by probe region
            (3.0,  0.15,  0.10,  0.00, 0.12),
            (2.5, -0.15, -0.10,  0.00, 0.10),
        ],
        "domain": 0.55, "probe_domain": 0.45, "n_particles": 130,
        "n_probes": 70, "h": 0.40, "rng_seed": 305,
    },
    # Asymmetric blob cluster: off-centre concentration — tests no-symmetry assumption
    "asymmetric_cluster": {
        "description": "Dense cluster in one corner — tests kernel in non-uniform spatial context",
        "blobs": [
            (5.0,  0.35,  0.30,  0.25, 0.12),
            (3.0,  0.28,  0.22,  0.20, 0.08),
            (1.5, -0.15, -0.10, -0.05, 0.18),
        ],
        "domain": 0.55, "probe_domain": 0.48, "n_particles": 130,
        "n_probes": 65, "h": 0.38, "rng_seed": 306,
    },
    # Very dense / high overlap: tests stability when many neighbours within h
    "dense_h060": {
        "description": "Dense packing h=0.60, 280 particles — stable kernel in bulk fluid regime",
        "blobs": [
            (2.0,  0.10,  0.08,  0.05, 0.20),
            (2.0, -0.10, -0.08, -0.05, 0.20),
            (2.0,  0.00,  0.15, -0.10, 0.18),
        ],
        "domain": 0.65, "probe_domain": 0.55, "n_particles": 280,
        "n_probes": 90, "h": 0.60, "rng_seed": 307,
    },
}

KNOWN_BASELINES: dict[str, float] = {
    "gauss_blobs_3d": 0.0,    # filled in at first run
    "sharp_gradients": 0.0,
}


def _rho(x: float, y: float, z: float,
         blobs: list[tuple[float, float, float, float, float]]) -> float:
    return sum(A * math.exp(-((x-cx)**2 + (y-cy)**2 + (z-cz)**2) / (2*s*s))
               for A, cx, cy, cz, s in blobs)


def generate_instance(name: str) -> SPHInstance:
    """Generate a reproducible SPH benchmark instance."""
    cfg = _INSTANCE_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown SPH instance: {name!r}. Available: {list(_INSTANCE_CONFIGS)}")

    rng = random.Random(cfg["rng_seed"])
    blobs = cfg["blobs"]
    dom = cfg["domain"]
    pdom = cfg["probe_domain"]
    n_part = cfg["n_particles"]
    n_prob = cfg["n_probes"]
    h = cfg["h"]
    vol = (2 * dom) ** 3          # box volume
    particle_vol = vol / n_part   # volume per particle

    # Jittered particle placement: 5×5×(n/25) near-lattice with ±15% jitter
    # Mass proportional to ρ_true at particle position
    particles: list[tuple[float, float, float, float]] = []
    for _ in range(n_part):
        x = rng.uniform(-dom, dom)
        y = rng.uniform(-dom, dom)
        z = rng.uniform(-dom, dom)
        mass = _rho(x, y, z, blobs) * particle_vol
        particles.append((x, y, z, float(mass)))

    # Off-lattice probe points — purposely not at particle positions
    probes: list[tuple[float, float, float, float]] = []
    for _ in range(n_prob):
        x = rng.uniform(-pdom, pdom)
        y = rng.uniform(-pdom, pdom)
        z = rng.uniform(-pdom, pdom)
        rho_t = _rho(x, y, z, blobs)
        probes.append((x, y, z, float(rho_t)))

    return SPHInstance(
        name=name,
        description=cfg["description"],
        particles=particles,
        probes=probes,
        h=h,
    )
