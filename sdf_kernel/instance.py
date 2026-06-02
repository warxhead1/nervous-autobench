"""SDF benchmark instances — synthetic sample-point generation.

Analytic SDF target functions (Python side only — used to generate benchmark
samples), the SDFInstance dataclass, the instance factory registry, and
generate_instance().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# SDF benchmark instance: synthetic sample points
# ---------------------------------------------------------------------------

@dataclass
class SDFInstance:
    """A benchmark instance: a target SDF evaluated at N sample points.

    The Python side knows the analytic formula (used only at load time to
    generate samples). The C++ candidate receives only (x,y,z,expected_sdf)
    tuples and must recover the implicit surface without seeing the formula.

    Attributes:
        name:           Instance identifier (e.g. 'gyroid', 'round_box').
        description:    Human-readable description of the target shape.
        samples:        List of (x, y, z, expected_sdf_value).
        optimal_mse:    Always 0.0 (known optimal — an exact formula achieves it).
        bbox:           (min, max) of sampling domain (cube).
        n_samples:      Number of sample points.
    """
    name: str
    description: str
    samples: list[tuple[float, float, float, float]]
    optimal_mse: float = 0.0
    bbox: tuple[float, float] = (-2.0, 2.0)
    n_samples: int = 0

    def __post_init__(self) -> None:
        self.n_samples = len(self.samples)


def _make_samples(
    target_fn: Callable[[float, float, float], float],
    n: int,
    lo: float,
    hi: float,
    seed: int = 42,
) -> list[tuple[float, float, float, float]]:
    """Generate stratified random sample points for an SDF target."""
    import random
    rng = random.Random(seed)
    samples = []
    # Mix surface-near and volume-wide samples for a balanced signal.
    # 60% volume-wide, 40% surface-biased (within 0.4 of the zero set)
    volume = int(n * 0.6)
    near = n - volume
    for _ in range(volume):
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        z = rng.uniform(lo, hi)
        v = target_fn(x, y, z)
        samples.append((x, y, z, v))
    # Surface-biased: try to find points close to the zero set
    for _ in range(near):
        for _attempt in range(20):
            x = rng.uniform(lo, hi)
            y = rng.uniform(lo, hi)
            z = rng.uniform(lo, hi)
            v = target_fn(x, y, z)
            if abs(v) < 0.4:
                break
        samples.append((x, y, z, v))
    return samples


# ---------------------------------------------------------------------------
# Analytic SDF targets (Python side only — used to generate benchmark samples)
# ---------------------------------------------------------------------------

def _sdf_sphere(x: float, y: float, z: float, r: float = 1.0) -> float:
    return math.sqrt(x*x + y*y + z*z) - r


def _sdf_box(x: float, y: float, z: float, bx: float = 0.8, by: float = 0.5, bz: float = 0.6) -> float:
    qx, qy, qz = abs(x) - bx, abs(y) - by, abs(z) - bz
    mx, my, mz = max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)
    return math.sqrt(mx*mx + my*my + mz*mz) + min(max(qx, qy, qz), 0.0)


def _smooth_union(d1: float, d2: float, k: float = 0.3) -> float:
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h * h * k * 0.25


def _sdf_gyroid(x: float, y: float, z: float, scale: float = 2.5, thickness: float = 0.15) -> float:
    # Gyroid implicit: sin(sx)cos(sy) + sin(sy)cos(sz) + sin(sz)cos(sx) = 0
    sx, sy, sz = x * scale, y * scale, z * scale
    f = math.sin(sx) * math.cos(sy) + math.sin(sy) * math.cos(sz) + math.sin(sz) * math.cos(sx)
    # Approximate SDF via gradient magnitude: |grad f| ≈ scale * sqrt(3) at zero crossing
    grad_mag = scale * math.sqrt(3.0)
    return abs(f) / grad_mag - thickness


def _sdf_round_box(x: float, y: float, z: float) -> float:
    # Rounded box with different half-extents per axis and a corner radius
    bx, by, bz, r = 0.7, 0.4, 0.5, 0.15
    qx, qy, qz = abs(x) - bx, abs(y) - by, abs(z) - bz
    mx, my, mz = max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)
    return math.sqrt(mx*mx + my*my + mz*mz) + min(max(qx, qy, qz), 0.0) - r


def _sdf_warped_sphere(x: float, y: float, z: float) -> float:
    # Sphere with 3-axis sinusoidal domain warp — genuinely non-isometric distortion.
    # A plain sphere SDF cannot recover this without the warp terms; fitness < 1.0
    # for any seed that omits them. The warp parameters (amplitude=0.25, freq=3.0)
    # are chosen so the warp is large enough to score < 0.65 without it but the
    # surface is still a closed implicit in the sample bbox.
    xw = x + 0.25 * math.sin(3.0 * y)
    yw = y + 0.25 * math.sin(3.0 * z)
    zw = z + 0.25 * math.sin(3.0 * x)
    return math.sqrt(xw*xw + yw*yw + zw*zw) - 1.0


def _sdf_smooth_union_shape(x: float, y: float, z: float) -> float:
    # Smooth union of three spheres at different offsets — tests polynomial blending
    d1 = _sdf_sphere(x - 0.6, y, z, 0.5)
    d2 = _sdf_sphere(x + 0.6, y, z, 0.5)
    d3 = _sdf_sphere(x, y - 0.7, z, 0.4)
    return _smooth_union(_smooth_union(d1, d2, 0.3), d3, 0.25)


def _sdf_cloud_cluster(x: float, y: float, z: float) -> float:
    # Cumulus blob: true min-union of 7 spheres — eikonal-valid SDF.
    # Smooth_union was tested and discarded: it's not eikonal-valid, making it
    # impossible for the oracle to reward the correct structure.
    # True min-union IS eikonal-valid (|∇SDF|=1 almost everywhere).
    spheres = [
        (0.00,  0.00,  0.00,  0.55),
        (0.50,  0.15,  0.10,  0.45),
        (-0.45, 0.20,  0.05,  0.42),
        (0.15,  0.42, -0.22,  0.38),
        (-0.22, 0.45,  0.28,  0.35),
        (0.38,  0.52,  0.12,  0.30),
        (-0.10, 0.62, -0.08,  0.26),
    ]
    return min(math.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2) - r for cx, cy, cz, r in spheres)


def _sdf_torus_knot(x: float, y: float, z: float) -> float:
    # Trefoil (2,3) torus knot: p=2 winds around z-axis, q=3 through the hole
    # Tube of radius 0.15 swept along the knot curve
    # C(t) = ((R + r*cos(q*t))*cos(p*t), (R + r*cos(q*t))*sin(p*t), r*sin(q*t))
    R, r_minor, tube = 0.60, 0.35, 0.15
    N = 600
    best_d = float('inf')
    for i in range(N):
        t = 2 * math.pi * i / N
        rr = R + r_minor * math.cos(3 * t)
        kx = rr * math.cos(2 * t)
        ky = rr * math.sin(2 * t)
        kz = r_minor * math.sin(3 * t)
        d = math.sqrt((x-kx)**2 + (y-ky)**2 + (z-kz)**2)
        if d < best_d:
            best_d = d
    return best_d - tube


def _sdf_helix_tube(x: float, y: float, z: float) -> float:
    # Spring coil: helix around y-axis, R=0.7, pitch=0.5 (height per turn), 2 turns, tube r=0.15
    # C(t) = (R*cos(t), pitch*t/(2π), R*sin(t)), t ∈ [0, 4π]
    # N=800 samples gives spacing ≈ 0.011 << tube_radius=0.15 for accurate ground truth
    R, pitch, tube = 0.70, 0.50, 0.15
    N = 800
    best_d = float('inf')
    t_max = 2 * 2 * math.pi
    for i in range(N + 1):
        t = t_max * i / N
        kx = R * math.cos(t)
        ky = pitch * t / (2 * math.pi)
        kz = R * math.sin(t)
        d = math.sqrt((x-kx)**2 + (y-ky)**2 + (z-kz)**2)
        if d < best_d:
            best_d = d
    return best_d - tube


def _sdf_scherk_first(x: float, y: float, z: float) -> float:
    # Scherk's first minimal surface: exp(z)*cos(y) = cos(x) (doubly periodic)
    # Approximate SDF via |f| / |grad f| where f = exp(z)*cos(y) - cos(x)
    # grad: (sin(x), -exp(z)*sin(y), exp(z)*cos(y)), |grad|² = sin²(x) + exp(2z)
    # Valid in |x|, |y| < π/2 ≈ 1.57 (cos stays positive)
    ez = math.exp(max(-2.0, min(2.0, z)))
    f = ez * math.cos(y) - math.cos(x)
    grad2 = math.sin(x)**2 + ez**2
    return abs(f) / math.sqrt(max(grad2, 0.005)) - 0.08


# Known optimal MSE for each instance (0.0 = exact analytic formula exists)
KNOWN_OPTIMALS: dict[str, float] = {
    "gyroid": 0.0,
    "round_box": 0.0,
    "warped_sphere": 0.0,
    "smooth_union": 0.0,
    "sphere": 0.0,
    "cloud_cluster": 0.0,
    "torus_knot": 0.0,
    "helix_tube": 0.0,
    "scherk_first": 0.0,
}

# Instance factory: name → (target_fn, description, lo, hi, n_samples)
_INSTANCE_FACTORIES: dict[str, tuple[Callable, str, float, float, int]] = {
    "sphere": (
        _sdf_sphere,
        "Unit sphere: sqrt(x^2+y^2+z^2) - 1",
        -2.5, 2.5, 800,
    ),
    "round_box": (
        _sdf_round_box,
        "Rounded box with half-extents (0.7,0.4,0.5) and corner radius 0.15",
        -2.0, 2.0, 1000,
    ),
    "gyroid": (
        _sdf_gyroid,
        "Gyroid surface shell (scale=2.5, thickness=0.15) — triply periodic minimal surface",
        -1.5, 1.5, 1200,
    ),
    "warped_sphere": (
        _sdf_warped_sphere,
        "Sphere (r=1.0) with 3-axis sinusoidal domain warp (amplitude=0.25, freq=3.0) — breaks rotational symmetry, requires non-trivial distortion formula",
        -2.0, 2.0, 1000,
    ),
    "smooth_union": (
        _sdf_smooth_union_shape,
        "Smooth union of three spheres — tests polynomial blending (k=0.3/0.25)",
        -2.0, 2.0, 1000,
    ),
    "cloud_cluster": (
        _sdf_cloud_cluster,
        "Cumulus cloud blob: 7-sphere min-union, bottom-flat top-billowing, asymmetric",
        -2.0, 2.0, 1200,
    ),
    "torus_knot": (
        _sdf_torus_knot,
        "Trefoil (2,3) torus knot tube (R=0.60, r=0.35, tube=0.15) — helical knotted curve",
        -1.5, 1.5, 1000,
    ),
    "helix_tube": (
        _sdf_helix_tube,
        "Spring coil: helix around y-axis (R=0.70, pitch=0.50, 2 turns, tube=0.15)",
        -1.5, 1.5, 1000,
    ),
    "scherk_first": (
        _sdf_scherk_first,
        "Scherk's first minimal surface: exp(z)*cos(y)=cos(x), shell t=0.08 — doubly-periodic saddle",
        -1.2, 1.2, 1000,
    ),
}


def generate_instance(name: str) -> SDFInstance:
    """Generate a synthetic SDF benchmark instance from its analytic formula."""
    if name not in _INSTANCE_FACTORIES:
        raise ValueError(
            f"Unknown SDF instance '{name}'. Available: {sorted(_INSTANCE_FACTORIES)}"
        )
    fn, description, lo, hi, n = _INSTANCE_FACTORIES[name]
    samples = _make_samples(fn, n, lo, hi, seed=hash(name) & 0xFFFF)
    return SDFInstance(
        name=name,
        description=description,
        samples=samples,
        bbox=(lo, hi),
        n_samples=len(samples),
    )
