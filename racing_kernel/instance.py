"""Racing-kernel benchmark instances — seeded procedural track layouts.

Each RacingInstance is a closed-loop track defined by a sequence of control
points forming a 2-D spline, with a half-width that bounds the drivable
corridor.  Instances are fully deterministic given a seed so the FunSearch
loop can reproduce them without storing them (same contract as SDF instances).

Public surface:
  RacingInstance        — dataclass: track geometry + oracle calibration params
  generate_instance     — name/seed → RacingInstance
  TRACK_LAYOUTS         — dict of named layout seeds
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Track geometry helpers
# ---------------------------------------------------------------------------

def _catmull_rom(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Catmull-Rom spline — smooth interpolation between p1..p2 at param t."""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        2 * p1[0]
        + (-p0[0] + p2[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2 * p1[1]
        + (-p0[1] + p2[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return x, y


def _sample_centerline(
    control_pts: list[tuple[float, float]],
    samples_per_segment: int = 20,
) -> list[tuple[float, float]]:
    """Densely sample a closed Catmull-Rom spline from control_pts."""
    n = len(control_pts)
    pts: list[tuple[float, float]] = []
    for i in range(n):
        p0 = control_pts[(i - 1) % n]
        p1 = control_pts[i]
        p2 = control_pts[(i + 1) % n]
        p3 = control_pts[(i + 2) % n]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            pts.append(_catmull_rom(p0, p1, p2, p3, t))
    return pts


def _arc_lengths(pts: list[tuple[float, float]]) -> list[float]:
    """Cumulative arc-length at each sample point (starts at 0)."""
    arcs = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        arcs.append(arcs[-1] + math.hypot(dx, dy))
    return arcs


def _signed_curvature(pts: list[tuple[float, float]], i: int) -> float:
    """Signed curvature at index i (finite-difference approximation)."""
    n = len(pts)
    p0 = pts[(i - 1) % n]
    p1 = pts[i]
    p2 = pts[(i + 1) % n]
    dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
    dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
    cross = dx1 * dy2 - dy1 * dx2
    dot = dx1 * dx2 + dy1 * dy2
    denom = math.hypot(dx1, dy1) * math.hypot(dx2, dy2)
    if denom < 1e-9:
        return 0.0
    # Signed curvature via atan2 of the turning angle
    angle = math.atan2(cross, dot + denom)
    seg_len = 0.5 * (math.hypot(dx1, dy1) + math.hypot(dx2, dy2))
    return angle / (seg_len + 1e-9)


# ---------------------------------------------------------------------------
# RacingInstance dataclass
# ---------------------------------------------------------------------------

@dataclass
class RacingInstance:
    """One benchmark track for the racing-line FunSearch kernel.

    Attributes:
        name          — track identifier (e.g. "oval", "chicane").
        seed          — integer seed used to generate the layout.
        centerline    — N×2 list of (x,y) points uniformly spaced along the track.
        half_width    — half-width of the drivable corridor (world units).
        arc_lengths   — cumulative arc-length at each centerline sample.
        lap_length    — total lap distance (arc_lengths[-1]).
        curvatures    — signed curvature at each centerline sample (1/m).
        control_pts   — the raw procedural control points (for debug).

        # Oracle calibration (set from tengine.race.episode data when available;
        # synthetic defaults otherwise).
        ref_lap_time_s  — reference lap time for a competent controller (seconds).
        max_speed       — maximum achievable speed estimate (world units/s).
        speed_limit     — per-point speed limit derived from curvature (list, optional).
    """
    name: str
    seed: int
    centerline: list[tuple[float, float]]
    half_width: float
    arc_lengths: list[float]
    lap_length: float
    curvatures: list[float]
    control_pts: list[tuple[float, float]]
    ref_lap_time_s: float = 30.0
    max_speed: float = 20.0
    speed_limit: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.speed_limit:
            self.speed_limit = self._compute_speed_limits()

    def _compute_speed_limits(self) -> list[float]:
        """Per-point speed limit: v_max * sqrt(1 / (1 + k * |curvature|))."""
        k = 8.0  # lateral-g scaling constant
        return [
            self.max_speed * math.sqrt(1.0 / (1.0 + k * abs(c)))
            for c in self.curvatures
        ]


# ---------------------------------------------------------------------------
# Track layout generators
# ---------------------------------------------------------------------------

# Named track layouts — seed values used to reproduce each track
TRACK_LAYOUTS: dict[str, int] = {
    "oval":       0,
    "chicane":    1,
    "hairpin":    2,
    "complex":    3,
}


def _gen_control_points(
    seed: int,
    n_pts: int,
    radius: float = 50.0,
    perturbation: float = 20.0,
) -> list[tuple[float, float]]:
    """Generate a closed set of control points for a track layout."""
    rng = random.Random(seed)
    pts: list[tuple[float, float]] = []
    for i in range(n_pts):
        angle = 2 * math.pi * i / n_pts
        r = radius + rng.uniform(-perturbation, perturbation)
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return pts


_LAYOUT_PARAMS: dict[str, dict] = {
    "oval": {
        "n_pts": 6,
        "radius": 60.0,
        "perturbation": 8.0,
        "half_width": 8.0,
    },
    "chicane": {
        "n_pts": 10,
        "radius": 50.0,
        "perturbation": 22.0,
        "half_width": 6.0,
    },
    "hairpin": {
        "n_pts": 8,
        "radius": 45.0,
        "perturbation": 30.0,
        "half_width": 6.0,
    },
    "complex": {
        "n_pts": 14,
        "radius": 55.0,
        "perturbation": 25.0,
        "half_width": 7.0,
    },
}


def generate_instance(name: str, samples_per_segment: int = 24) -> RacingInstance:
    """Generate a RacingInstance for a named track layout.

    Args:
        name:                 One of ``TRACK_LAYOUTS`` keys.
        samples_per_segment:  Centerline density (higher → smoother oracle).

    Returns:
        RacingInstance with fully computed geometry and synthetic calibration.
    """
    if name not in TRACK_LAYOUTS:
        raise ValueError(f"Unknown track '{name}'. Available: {list(TRACK_LAYOUTS)}")
    seed = TRACK_LAYOUTS[name]
    params = _LAYOUT_PARAMS[name]
    ctrl = _gen_control_points(
        seed,
        n_pts=params["n_pts"],
        radius=params["radius"],
        perturbation=params["perturbation"],
    )
    centerline = _sample_centerline(ctrl, samples_per_segment=samples_per_segment)
    arcs = _arc_lengths(centerline)
    lap_len = arcs[-1]
    curvs = [_signed_curvature(centerline, i) for i in range(len(centerline))]
    half_w = params["half_width"]

    # Synthetic calibration: reference lap time is the physics-ceiling lap —
    # what a perfect controller (running at speed_limit everywhere) would achieve.
    # Seeds actually run slightly slower due to kinematic smoothing; the oracle's
    # speed_score term is computed as ref_lap / actual_lap, which is < 1 for seeds.
    # The score formula maps this ratio to the 0.65–0.80 range for seeds, and
    # rewards evolved controllers that approach or exceed the physics ceiling.
    max_speed = 22.0
    k_curv = 8.0
    tmp_speed_limits = [
        max_speed * math.sqrt(1.0 / (1.0 + k_curv * abs(c)))
        for c in curvs
    ]
    # Perfect lap time: integrate seg_len / speed_limit (physics ceiling)
    if len(centerline) > 1:
        perf_time = 0.0
        n_cl = len(centerline)
        for i in range(n_cl):
            ni = (i + 1) % n_cl
            dx = centerline[ni][0] - centerline[i][0]
            dy = centerline[ni][1] - centerline[i][1]
            seg = math.hypot(dx, dy)
            perf_time += seg / max(tmp_speed_limits[i], 0.5)
        ref_lap_time_s = perf_time  # the minimum achievable lap time
    else:
        ref_lap_time_s = lap_len / max_speed

    return RacingInstance(
        name=name,
        seed=seed,
        centerline=centerline,
        half_width=half_w,
        arc_lengths=arcs,
        lap_length=lap_len,
        curvatures=curvs,
        control_pts=ctrl,
        ref_lap_time_s=ref_lap_time_s,
        max_speed=max_speed,
    )
