"""SVDAG volcanic-terrain benchmark instances.

Unlike the SDF kernel, there is no analytic ground-truth surface — "realism" is
a class, not a point. Each instance is a volcanic ARCHETYPE: a random seed (to
vary the realization the candidate is sampled at) plus a dict of target bands
the membership oracle scores against (see oracle.score_occupancy).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Tunable per-archetype targets for the membership oracle. Bands are deliberately
# wide (class membership, not a single point); calibrated so a decent fbm-cone
# seed lands ~0.65-0.78 and the flat/noise controls fall far below (derisking via
# the `baselines` CLI command).
def _targets(**over) -> dict:
    base = dict(
        solid_lo=0.18, solid_hi=0.46,     # fraction of voxels that are rock
        rough_half=0.012,                 # surface-gradient half-saturation
        steep_thresh=0.02, steep_half=0.04,  # steep-slope presence
        beta_mu=2.45, beta_sigma=0.60,    # heightfield spectral slope (rocky fractal)
        pore_lo=0.02, pore_hi=0.18,       # enclosed-void fraction (overhangs/caves)
        relief_half=0.08,                 # vertical relief (std of normalized height)
    )
    base.update(over)
    return base


@dataclass
class VolcanoInstance:
    name: str
    description: str
    seed: float
    targets: dict = field(default_factory=_targets)


# (name) -> (description, seed, target-overrides)
_INSTANCE_FACTORIES: dict[str, tuple[str, float, dict]] = {
    "stratovolcano": (
        "steep basalt cone, rugged fractal flanks, modest summit caves",
        1337.0, dict(solid_lo=0.16, solid_hi=0.40, pore_hi=0.16, relief_half=0.09),
    ),
    "lava_field": (
        "low vesicular basalt plain, high porosity, broken crust",
        4242.0, dict(solid_lo=0.26, solid_hi=0.52, beta_mu=2.20, pore_lo=0.05, pore_hi=0.26, relief_half=0.06),
    ),
    "caldera": (
        "collapsed summit ring, central depression, steep inner walls",
        909.0, dict(solid_lo=0.16, solid_hi=0.40, beta_mu=2.30, pore_hi=0.18),
    ),
    "eroded_badlands": (
        "deeply channeled rock, drainage gullies, hoodoos and overhangs",
        5151.0, dict(rough_half=0.014, beta_mu=2.55, pore_hi=0.22, relief_half=0.08),
    ),
}


def generate_instance(name: str) -> VolcanoInstance:
    if name not in _INSTANCE_FACTORIES:
        name = "stratovolcano"
    desc, seed, over = _INSTANCE_FACTORIES[name]
    return VolcanoInstance(name=name, description=desc, seed=seed, targets=_targets(**over))
