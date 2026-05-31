"""SDF FunSearch kernel — LLM-driven discovery of signed-distance-field functions.

# Domain choice: Signed Distance Fields (SDFs)

We chose SDF approximation over four other candidate domains (BVH split
heuristic, importance sampling, tone mapping, raymarching step) for these
reasons:

1. CRISP OPTIMUM, CLEAR FITNESS CONTRACT.
   The base class wants evaluate_candidate → (0,1], 1.0 = perfect. SDF gives
   exactly that: fitness = 1 / (1 + MSE) over N sampled points, so MSE=0 →
   fitness=1.0. BVH and raymarching have no crisp theoretical optimum — their
   "best" traversal-step count is instance-dependent, making the approx-ratio
   contract awkward.

2. SELF-CONTAINED EVALUATION — PURE CPU C++.
   The benchmark instances carry precomputed (x,y,z,target_value) sample
   points generated synthetically in Python at load time. No GPU, no mesh
   files, no external renderer needed. The C++ evaluator receives points as
   JSON on stdin, calls the evolved sdf(x,y,z), and returns MSE.

3. MATHEMATICALLY RICH.
   The functional space is enormous: polynomial primitives, trig/exponential
   combinations, smooth-union operators, domain-warping transformations,
   fractal noise. A 20-line sdf() body can represent structures that are
   very hard to discover by search. LLMs have substantial knowledge of
   analytical SDF representations, making the prior over useful mutations
   strong without constraining the search.

4. DIRECT TENGINE RELEVANCE.
   SDFs are the geometry primitive for tengine's raymarching / SVDAG pipeline.
   A kernel that discovers compact SDF representations of complex surfaces
   directly produces functions usable in shaders. The best found programs
   are potential drop-in replacements for hand-crafted artists' SDFs.

# Anti-cheat measure.
   Target values are stored *in the instance* (sample points with known
   correct distances). The C++ program only receives the sample coordinates
   and the expected distances — it cannot call a Python target function.
   Targets are composite shapes (smooth-union of multiple primitives with
   domain distortion) chosen to make the search non-trivial.

# Architecture
   ┌───────────────────────────────────────────────────────────────┐
   │                     FunSearch Loop                            │
   │                                                               │
   │  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐ │
   │  │  LLM Caller  │──▶│  sdf() fn     │──▶│  C++ Sandbox     │ │
   │  │  (deer-flow) │   │  Generator    │   │  (compile+run)   │ │
   │  └──────────────┘   └───────────────┘   └────────┬─────────┘ │
   │                                                    │          │
   │                          ┌────────────────────────┘          │
   │                          ▼                                    │
   │                   ┌──────────────┐                           │
   │                   │  Fitness     │                           │
   │                   │  1/(1+MSE)   │                           │
   │                   └──────┬───────┘                           │
   │                          │                                   │
   │         ┌────────────────┼────────────────┐                  │
   │         ▼                ▼                ▼                  │
   │  ┌───────────┐   ┌───────────┐   ┌───────────┐             │
   │  │  Island 0 │   │  Island 1 │ ...│ Island N │             │
   │  │  (20 progs)│   │  (20 progs)│   │  (20 progs)│          │
   │  └───────────┘   └───────────┘   └───────────┘             │
   │         │                │                │                  │
   │         └────────────────┼────────────────┘                  │
   │                    migration (every 10 gens)                 │
   └───────────────────────────────────────────────────────────────┘

Usage:
  python -m autobench.sdf_kernel run --instances gyroid,round_box,twisted_torus \\
           --generations 100 --islands 4 --population 20 --allow-unsandboxed
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..kernel_base import FunSearchKernel, KernelConfig, CandidateProgram, Island
from ..core import Verdict
from ..sandbox import SandboxedExecutor, compile_and_run
from ..tsp_kernel import ensure_sandboxed_executor, UnsafeSandboxError  # reuse sandbox gate

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Topology oracle targets (empirically measured from analytical functions)
# ---------------------------------------------------------------------------

TOPO_TARGETS: dict[str, dict[str, float]] = {
    "gyroid":        {"target": 0.178, "sigma": 0.060},
    "round_box":     {"target": 0.009, "sigma": 0.005},
    "warped_sphere": {"target": 0.018, "sigma": 0.008},
    # New instances — measured on 24³ grid in their bboxes (2026-05-30)
    "cloud_cluster": {"target": 0.010, "sigma": 0.005},  # measured: 0.0095 (7-sphere min-union)
    "torus_knot":    {"target": 0.021, "sigma": 0.010},  # measured: 0.0212 (trefoil tube)
    "helix_tube":    {"target": 0.019, "sigma": 0.008},  # measured: 0.0186 (spring coil)
    "scherk_first":  {"target": 0.070, "sigma": 0.025},  # measured: 0.0696 (doubly-periodic)
    # fallback for unknown instances
    "_default":      {"target": 0.050, "sigma": 0.050},
}

# Topology harness skeleton — LO, HI, GRID_N replaced per-instance at call time.
TOPO_SKELETON = r"""// Topology harness — counts sign changes on a 3D grid.
// LLM-evolved sdf() is appended below this skeleton.
#include <bits/stdc++.h>
using namespace std;

extern "C" float sdf(float x, float y, float z);
// LLM_SDF_PLACEHOLDER

int main() {
    double lo = LO_PLACEHOLDER;
    double hi = HI_PLACEHOLDER;
    int n = GRID_N_PLACEHOLDER;
    double step = (hi - lo) / n;
    long sign_changes = 0;
    long total = (long)n * n * n;

    // Sign changes along x-axis
    for (int iz = 0; iz < n; iz++) {
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n - 1; ix++) {
                float a = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                float b = sdf((float)(lo + (ix + 1) * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                if (isfinite(a) && isfinite(b) && ((a >= 0.0f) != (b >= 0.0f)))
                    sign_changes++;
            }
        }
    }
    // Sign changes along y-axis
    for (int iz = 0; iz < n; iz++) {
        for (int iy = 0; iy < n - 1; iy++) {
            for (int ix = 0; ix < n; ix++) {
                float a = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                float b = sdf((float)(lo + ix * step),
                              (float)(lo + (iy + 1) * step),
                              (float)(lo + iz * step));
                if (isfinite(a) && isfinite(b) && ((a >= 0.0f) != (b >= 0.0f)))
                    sign_changes++;
            }
        }
    }
    // Sign changes along z-axis
    for (int iz = 0; iz < n - 1; iz++) {
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n; ix++) {
                float a = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                float b = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + (iz + 1) * step));
                if (isfinite(a) && isfinite(b) && ((a >= 0.0f) != (b >= 0.0f)))
                    sign_changes++;
            }
        }
    }
    double density = (double)sign_changes / (3.0 * total);
    printf("{\"sign_changes\":%ld,\"total\":%ld,\"density\":%.6f}\n",
           sign_changes, 3 * total, density);
    return 0;
}
"""


def build_topology_source(sdf_code: str, instance: "SDFInstance", grid_n: int = 24) -> str:
    """Build C++ source for topology harness with bbox and sdf() baked in."""
    lo, hi = instance.bbox
    source = TOPO_SKELETON
    source = source.replace("LO_PLACEHOLDER", repr(float(lo)))
    source = source.replace("HI_PLACEHOLDER", repr(float(hi)))
    source = source.replace("GRID_N_PLACEHOLDER", str(grid_n))
    # Remove the placeholder comment and append the actual sdf code
    source = source.replace("// LLM_SDF_PLACEHOLDER", "")
    return source + "\n" + sdf_code + "\n"


def compute_topology_score(
    sdf_code: str,
    instance: "SDFInstance",
    executor: "SandboxedExecutor",
    run_timeout: float = 10.0,
    grid_n: int = 24,
) -> tuple[float, float]:
    """Compute topology score for sdf_code on instance.

    Compiles a C++ topology harness that evaluates sign_change_density on a
    grid_n^3 grid, then applies a Gaussian oracle against the calibrated target.

    Returns (topology_score, sign_change_density).
    On any failure, returns (0.0, 0.0) so the main oracle stays live.
    """
    source = build_topology_source(sdf_code, instance, grid_n=grid_n)
    try:
        stdout, verdict, _latency = compile_and_run(
            source,
            "cpp",
            constraints={"max_time_seconds": run_timeout, "max_memory_mb": executor.max_memory_mb},
            stdin="",
            executor=executor,
        )
        if verdict != Verdict.OK:
            logger.debug("Topology harness non-OK verdict %s on %s", verdict, instance.name)
            return 0.0, 0.0
        out = json.loads(stdout.strip())
        density = float(out["density"])
        if not math.isfinite(density) or density < 0:
            return 0.0, 0.0
    except Exception as exc:
        logger.debug("Topology score failed on %s: %s", instance.name, exc)
        return 0.0, 0.0

    params = TOPO_TARGETS.get(instance.name, TOPO_TARGETS["_default"])
    target = params["target"]
    sigma = params["sigma"]
    score = math.exp(-0.5 * ((density - target) / sigma) ** 2)
    return score, density


# ---------------------------------------------------------------------------
# C++ skeleton — fixed harness, LLM injects sdf(x, y, z)
# ---------------------------------------------------------------------------

CPP_SKELETON = r"""// Auto-generated SDF skeleton. LLM writes sdf() only.
// Reads sample points from stdin JSON; evaluates sdf() and returns MSE.
#include <bits/stdc++.h>
using namespace std;

// === LLM-EVOLVED FUNCTION — do not modify the signature ===
extern "C" float sdf(float x, float y, float z);
// ===========================================================

struct Sample { float x, y, z, expected; };

int main() {
    string s((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());

    // Hand-rolled JSON parser to avoid external dependencies.
    // Expects: {"name":"...","samples":[[x,y,z,expected],...]}
    int pos = 0;
    auto skip_ws = [&]() {
        while (pos < (int)s.size() && isspace((unsigned char)s[pos])) pos++;
    };
    auto expect_char = [&](char c) -> bool {
        skip_ws();
        if (pos < (int)s.size() && s[pos] == c) { pos++; return true; }
        return false;
    };
    auto parse_string = [&]() -> string {
        skip_ws();
        if (pos >= (int)s.size() || s[pos] != '"') return string();
        pos++; string out;
        while (pos < (int)s.size() && s[pos] != '"') {
            if (s[pos] == '\\') pos++;
            if (pos < (int)s.size()) out += s[pos++];
        }
        if (pos < (int)s.size()) pos++;
        return out;
    };
    auto parse_number = [&]() -> double {
        skip_ws();
        bool neg = false;
        if (pos < (int)s.size() && s[pos] == '-') { neg = true; pos++; }
        double v = 0;
        while (pos < (int)s.size() && isdigit((unsigned char)s[pos]))
            { v = v*10 + (s[pos]-'0'); pos++; }
        if (pos < (int)s.size() && s[pos] == '.') {
            pos++; double frac = 0.1;
            while (pos < (int)s.size() && isdigit((unsigned char)s[pos]))
                { v += (s[pos]-'0')*frac; frac*=0.1; pos++; }
        }
        if (pos < (int)s.size() && (s[pos]=='e'||s[pos]=='E')) {
            pos++; bool eneg = false;
            if (pos<(int)s.size()&&(s[pos]=='+'||s[pos]=='-')){eneg=(s[pos]=='-');pos++;}
            int exp=0;
            while(pos<(int)s.size()&&isdigit((unsigned char)s[pos])){exp=exp*10+(s[pos]-'0');pos++;}
            double mult=1; for(int i=0;i<exp;i++) mult*=10;
            if(eneg) v/=mult; else v*=mult;
        }
        return neg ? -v : v;
    };

    string inst_name;
    vector<Sample> samples;

    expect_char('{');
    while (true) {
        skip_ws();
        if (pos >= (int)s.size() || s[pos] == '}') { pos++; break; }
        string key = parse_string();
        expect_char(':');
        if (key == "name") {
            inst_name = parse_string();
        } else if (key == "samples") {
            expect_char('[');
            while (true) {
                skip_ws();
                if (pos>=(int)s.size()||s[pos]==']') { pos++; break; }
                expect_char('[');
                float x=(float)parse_number(); expect_char(',');
                float y=(float)parse_number(); expect_char(',');
                float z=(float)parse_number(); expect_char(',');
                float e=(float)parse_number();
                expect_char(']');
                samples.push_back({x,y,z,e});
                skip_ws(); if (pos<(int)s.size()&&s[pos]==',') pos++;
            }
        } else {
            // Skip unknown fields
            skip_ws();
            int depth = 0;
            if (pos<(int)s.size()&&(s[pos]=='{'||s[pos]=='[')) {
                while(pos<(int)s.size()) {
                    if(s[pos]=='{'||s[pos]=='[') depth++;
                    else if(s[pos]=='}'||s[pos]==']') { depth--; if(depth==0){pos++;break;} }
                    pos++;
                }
            } else if (pos<(int)s.size()&&s[pos]=='"') { parse_string(); }
            else { parse_number(); }
        }
        skip_ws(); if (pos<(int)s.size()&&s[pos]==',') pos++;
    }

    if (samples.empty()) {
        cerr << "ERROR: no samples parsed" << endl;
        return 1;
    }

    double mse = 0.0;
    int n = (int)samples.size();
    for (auto& smp : samples) {
        double got = (double)sdf(smp.x, smp.y, smp.z);
        double diff = got - (double)smp.expected;
        mse += diff * diff;
    }
    mse /= n;

    // Eikonal check: |∇sdf| should = 1 everywhere (valid signed-distance field).
    // Finite-difference gradient at every 5th sample point.
    const float h = 1e-3f;
    double grad_err_sum = 0.0;
    int n_grad = 0;
    for (int i = 0; i < n; i += 5) {
        auto& smp = samples[i];
        float dx = (sdf(smp.x+h,smp.y,smp.z) - sdf(smp.x-h,smp.y,smp.z)) / (2*h);
        float dy = (sdf(smp.x,smp.y+h,smp.z) - sdf(smp.x,smp.y-h,smp.z)) / (2*h);
        float dz = (sdf(smp.x,smp.y,smp.z+h) - sdf(smp.x,smp.y,smp.z-h)) / (2*h);
        float grad_mag = sqrtf(dx*dx + dy*dy + dz*dz);
        if (isfinite(grad_mag)) { grad_err_sum += fabs(grad_mag - 1.0f); n_grad++; }
    }
    double mean_grad_err = (n_grad > 0) ? grad_err_sum / n_grad : 1.0;

    printf("{\"mse\":%.9f,\"grad_err\":%.6f,\"n\":%d,\"instance\":\"%s\"}\n",
           mse, mean_grad_err, n, inst_name.c_str());
    return 0;
}
"""


SDF_FUNCTION_SIGNATURE = '''\
extern "C" float sdf(float x, float y, float z);

// Goal: return the signed distance from point (x,y,z) to the target surface.
// Positive outside the surface, negative inside, zero exactly on the surface.
// The returned value is compared against precomputed ground-truth distances.
// Fitness = 1 / (1 + MSE) — minimise MSE to maximise fitness.
//
// Available C math functions (no includes needed — skeleton provides them):
//   sqrtf, fabsf, fmaxf, fminf, sinf, cosf, atan2f, powf, expf, logf
//
// Useful SDF building blocks:
//   sphere(R):      sqrtf(x*x+y*y+z*z) - R
//   box(bx,by,bz):  q=max(|.|−b,0); length(q)+min(max(q.x,q.y,q.z),0)
//   torus(R,r):     q=sqrtf(x*x+y*y)-R; sqrtf(q*q+z*z)-r
//   smooth_union:   h=max(k-|d1-d2|,0)/k; min(d1,d2)-h*h*k/4
//   domain twist:   (xw,yw) = rotate(x,y, angle*z)
//   gyroid:         sin(s*x)*cos(s*y)+sin(s*y)*cos(s*z)+sin(s*z)*cos(s*x)
//   cloud blob:     fminf cascade of 5-8 spheres at asymmetric offsets
//   torus knot(p,q):((R+r*cos(q*t))*cos(p*t),(R+r*cos(q*t))*sin(p*t),r*sin(q*t)); min-dist tube
//   helix tube:     (R*cos(t), pitch*t/2pi, R*sin(t)); use atan2f+k-loop for nearest t
//   Scherk surf:    f=exp(z)*cos(y)-cos(x); SDF≈|f|/sqrt(sin²x+exp(2z))
'''


# ---------------------------------------------------------------------------
# Baseline seed programs — analytical solutions per instance family
# ---------------------------------------------------------------------------

SEED_SDF_PROGRAMS: dict[str, list[tuple[str, str]]] = {
    "generic": [
        ("sphere_baseline", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
        ("box_baseline", '''\
extern "C" float sdf(float x, float y, float z) {
    float qx = fabsf(x) - 0.7f;
    float qy = fabsf(y) - 0.5f;
    float qz = fabsf(z) - 0.6f;
    float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
    return sqrtf(mx*mx + my*my + mz*mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
}'''),
        ("torus_baseline", '''\
extern "C" float sdf(float x, float y, float z) {
    float q = sqrtf(x*x + y*y) - 0.9f;
    return sqrtf(q*q + z*z) - 0.3f;
}'''),
    ],
    "sphere": [
        ("sphere_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
        ("sphere_approx1", '''\
extern "C" float sdf(float x, float y, float z) {
    // Intentionally imperfect: Euclidean approximation
    float r2 = x*x + y*y + z*z;
    return sqrtf(r2) - 1.05f;
}'''),
        ("sphere_scaled", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 0.95f;
}'''),
    ],
    "gyroid": [
        ("gyroid_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    float s = 2.5f;
    float f = sinf(s*x)*cosf(s*y) + sinf(s*y)*cosf(s*z) + sinf(s*z)*cosf(s*x);
    float grad = s * 1.7320508f;  // sqrt(3)
    return fabsf(f) / grad - 0.15f;
}'''),
        ("gyroid_no_thickness", '''\
extern "C" float sdf(float x, float y, float z) {
    float s = 2.5f;
    float f = sinf(s*x)*cosf(s*y) + sinf(s*y)*cosf(s*z) + sinf(s*z)*cosf(s*x);
    return f * 0.23f;
}'''),
        ("gyroid_thick", '''\
extern "C" float sdf(float x, float y, float z) {
    float s = 2.5f;
    float f = sinf(s*x)*cosf(s*y) + sinf(s*y)*cosf(s*z) + sinf(s*z)*cosf(s*x);
    float grad = s * 1.7320508f;
    return fabsf(f) / grad - 0.25f;
}'''),
    ],
    "round_box": [
        ("round_box_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    float qx = fabsf(x) - 0.7f;
    float qy = fabsf(y) - 0.4f;
    float qz = fabsf(z) - 0.5f;
    float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
    return sqrtf(mx*mx + my*my + mz*mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f) - 0.15f;
}'''),
        ("box_no_rounding", '''\
extern "C" float sdf(float x, float y, float z) {
    float qx = fabsf(x) - 0.7f;
    float qy = fabsf(y) - 0.4f;
    float qz = fabsf(z) - 0.5f;
    float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
    return sqrtf(mx*mx + my*my + mz*mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
}'''),
        ("sphere_fallback", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 0.9f;
}'''),
    ],
    "warped_sphere": [
        ("sphere_no_warp", '''\
extern "C" float sdf(float x, float y, float z) {
    // Suboptimal: plain sphere (missing the warp terms) — scores ~0.62 fitness
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
        ("warped_sphere_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // Exact: sphere + 3-axis sinusoidal domain warp (amplitude=0.25, freq=3.0)
    float xw = x + 0.25f * sinf(3.0f * y);
    float yw = y + 0.25f * sinf(3.0f * z);
    float zw = z + 0.25f * sinf(3.0f * x);
    return sqrtf(xw*xw + yw*yw + zw*zw) - 1.0f;
}'''),
        ("warped_sphere_partial", '''\
extern "C" float sdf(float x, float y, float z) {
    // Partial: warp only x-axis — suboptimal, tests partial correction
    float xw = x + 0.25f * sinf(3.0f * y);
    return sqrtf(xw*xw + y*y + z*z) - 1.0f;
}'''),
    ],
    "smooth_union": [
        ("three_spheres_min", '''\
extern "C" float sdf(float x, float y, float z) {
    float d1 = sqrtf((x-0.6f)*(x-0.6f)+y*y+z*z) - 0.5f;
    float d2 = sqrtf((x+0.6f)*(x+0.6f)+y*y+z*z) - 0.5f;
    float d3 = sqrtf(x*x+(y-0.7f)*(y-0.7f)+z*z) - 0.4f;
    return fminf(fminf(d1, d2), d3);
}'''),
        ("smooth_union_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    float d1 = sqrtf((x-0.6f)*(x-0.6f)+y*y+z*z) - 0.5f;
    float d2 = sqrtf((x+0.6f)*(x+0.6f)+y*y+z*z) - 0.5f;
    float d3 = sqrtf(x*x+(y-0.7f)*(y-0.7f)+z*z) - 0.4f;
    // smooth_union with k=0.3
    float k = 0.3f;
    float h12 = fmaxf(k - fabsf(d1-d2), 0.0f) / k;
    float su12 = fminf(d1, d2) - h12*h12*k*0.25f;
    k = 0.25f;
    float h = fmaxf(k - fabsf(su12-d3), 0.0f) / k;
    return fminf(su12, d3) - h*h*k*0.25f;
}'''),
        ("single_sphere", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 0.8f;
}'''),
    ],
    "cloud_cluster": [
        ("cloud_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // True min-union of 7 spheres (eikonal-valid SDF)
    float d = sqrtf(x*x + y*y + z*z) - 0.55f;
    d = fminf(d, sqrtf((x-0.5f)*(x-0.5f)+(y-0.15f)*(y-0.15f)+(z-0.1f)*(z-0.1f))-0.45f);
    d = fminf(d, sqrtf((x+0.45f)*(x+0.45f)+(y-0.2f)*(y-0.2f)+(z-0.05f)*(z-0.05f))-0.42f);
    d = fminf(d, sqrtf((x-0.15f)*(x-0.15f)+(y-0.42f)*(y-0.42f)+(z+0.22f)*(z+0.22f))-0.38f);
    d = fminf(d, sqrtf((x+0.22f)*(x+0.22f)+(y-0.45f)*(y-0.45f)+(z-0.28f)*(z-0.28f))-0.35f);
    d = fminf(d, sqrtf((x-0.38f)*(x-0.38f)+(y-0.52f)*(y-0.52f)+(z-0.12f)*(z-0.12f))-0.30f);
    d = fminf(d, sqrtf((x+0.1f)*(x+0.1f)+(y-0.62f)*(y-0.62f)+(z+0.08f)*(z+0.08f))-0.26f);
    return d;
}'''),
        ("cloud_3sphere", '''\
extern "C" float sdf(float x, float y, float z) {
    // Partial: only 3 of 7 spheres — suboptimal, missing upper puffs
    float d = sqrtf(x*x + y*y + z*z) - 0.55f;
    d = fminf(d, sqrtf((x-0.5f)*(x-0.5f)+(y-0.15f)*(y-0.15f)+(z-0.1f)*(z-0.1f))-0.45f);
    d = fminf(d, sqrtf((x+0.45f)*(x+0.45f)+(y-0.2f)*(y-0.2f)+(z-0.05f)*(z-0.05f))-0.42f);
    return d;
}'''),
        ("cloud_sphere_fallback", '''\
extern "C" float sdf(float x, float y, float z) {
    // Worst-case: single large sphere
    return sqrtf(x*x + (y-0.2f)*(y-0.2f) + z*z) - 0.9f;
}'''),
    ],
    "torus_knot": [
        ("torus_knot_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // (2,3) torus knot: sample N=400 points on curve, find min distance
    const float R = 0.6f, rm = 0.35f, tube = 0.15f;
    const float pi2 = 6.28318530f;
    float best = 1e9f;
    for (int i = 0; i < 400; i++) {
        float t = pi2 * i / 400.0f;
        float rr = R + rm * cosf(3.0f * t);
        float kx = rr * cosf(2.0f * t);
        float ky = rr * sinf(2.0f * t);
        float kz = rm * sinf(3.0f * t);
        float d = sqrtf((x-kx)*(x-kx)+(y-ky)*(y-ky)+(z-kz)*(z-kz));
        if (d < best) best = d;
    }
    return best - tube;
}'''),
        ("torus_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    // Plain torus (missing the knot winding) — suboptimal seed
    float rho = sqrtf(x*x + y*y) - 0.6f;
    return sqrtf(rho*rho + z*z) - 0.25f;
}'''),
        ("sphere_fallback", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
    ],
    "helix_tube": [
        ("helix_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // Helix around y-axis: C(t) = (R*cos(t), pitch*t/(2π), R*sin(t)), 2 turns
    // N=800 for spacing << tube_radius
    const float R = 0.7f, pitch = 0.5f, tube = 0.15f;
    const float pi2 = 6.28318530f;
    float best = 1e9f;
    for (int i = 0; i <= 800; i++) {
        float t = 4.0f * pi2 * i / 800.0f;
        float kx = R * cosf(t);
        float ky = pitch * t / pi2;
        float kz = R * sinf(t);
        float d = sqrtf((x-kx)*(x-kx)+(y-ky)*(y-ky)+(z-kz)*(z-kz));
        if (d < best) best = d;
    }
    return best - tube;
}'''),
        ("torus_as_helix_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    // Torus (wrong topology, but similar scale) — suboptimal seed
    float rho = sqrtf(x*x + z*z) - 0.7f;
    return sqrtf(rho*rho + y*y) - 0.2f;
}'''),
        ("cylinder_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    // Vertical cylinder (worst-case approximation)
    return sqrtf(x*x + z*z) - 0.7f;
}'''),
    ],
    "scherk_first": [
        ("scherk_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // Scherk's first minimal surface: exp(z)*cos(y) = cos(x)
    // f = exp(z)*cos(y) - cos(x); |grad_f|^2 = sin^2(x) + exp(2z)
    float ez = expf(fmaxf(fminf(z, 1.5f), -1.5f));
    float f = ez * cosf(y) - cosf(x);
    float grad2 = sinf(x)*sinf(x) + ez*ez;
    return fabsf(f) / sqrtf(fmaxf(grad2, 0.005f)) - 0.08f;
}'''),
        ("scherk_saddle", '''\
extern "C" float sdf(float x, float y, float z) {
    // 2nd-order Taylor approx: z ≈ 0.5*(y^2 - x^2) near origin
    float f = z - 0.5f * (y*y - x*x);
    return fabsf(f) * 0.65f - 0.08f;
}'''),
        ("flat_plane", '''\
extern "C" float sdf(float x, float y, float z) {
    // Worst-case seed: flat plane at z=0
    return fabsf(z) - 0.12f;
}'''),
    ],
}


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    """Return seed programs for a given instance, falling back to generic seeds."""
    return SEED_SDF_PROGRAMS.get(instance_name, SEED_SDF_PROGRAMS["generic"])


# ---------------------------------------------------------------------------
# Sandbox gate
# ---------------------------------------------------------------------------

def ensure_executor(
    *,
    allow_unsandboxed: bool = False,
    max_memory_mb: int = 256,
    cpu_limit: int = 2,
) -> SandboxedExecutor:
    """Build and gate the executor. Raises UnsafeSandboxError if not isolated."""
    executor = SandboxedExecutor(
        sandbox_type="gvisor",
        max_memory_mb=max_memory_mb,
        cpu_limit=cpu_limit,
    )
    if executor.sandbox_type not in ("gvisor", "firecracker"):
        if not allow_unsandboxed:
            raise UnsafeSandboxError(
                "SDF kernel refuses to run untrusted candidate code without isolation "
                f"(got '{executor.sandbox_type}'). Install gVisor or pass "
                "allow_unsandboxed=True for a trusted, attended run."
            )
        logger.warning(
            "SANDBOX DEGRADED: running untrusted candidate code under '%s' "
            "(no isolation) because allow_unsandboxed=True", executor.sandbox_type,
        )
    return executor


def build_candidate_source(sdf_code: str) -> str:
    """Combine the fixed skeleton with the LLM-evolved sdf() implementation."""
    return CPP_SKELETON + "\n" + sdf_code + "\n"


def _instance_stdin(instance: SDFInstance) -> str:
    """Serialize an SDF instance to JSON for the C++ evaluator via stdin."""
    return json.dumps({
        "name": instance.name,
        "samples": [[x, y, z, e] for x, y, z, e in instance.samples],
    })


def evaluate_on_instance(
    sdf_code: str,
    instance: SDFInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 10.0,
) -> float | None:
    """Compile + run sdf_code against instance. Returns fitness in (0,1] or None.

    Combined fitness:
      combined = 0.6 * (1/(1+MSE)) * exp(-0.5*grad_err) + 0.4 * topology_score

    The eikonal term penalises sdf() functions that fit the zero-set but violate
    |∇f|=1 (not raymarcher-usable). The topology oracle measures sign-change
    density on a 24^3 grid and applies a Gaussian oracle against calibrated
    targets per instance — rewarding SDFs with the correct topological complexity
    (gyroid: very high density, round_box: very low, warped_sphere: low).

    On topology-harness failure the weight falls back to the eikonal-only score.
    """
    source = build_candidate_source(sdf_code)
    stdout, verdict, _latency = compile_and_run(
        source,
        "cpp",
        constraints={"max_time_seconds": run_timeout, "max_memory_mb": executor.max_memory_mb},
        stdin=_instance_stdin(instance),
        executor=executor,
    )
    if verdict != Verdict.OK:
        logger.debug("SDF candidate non-OK verdict %s on %s", verdict, instance.name)
        return None
    try:
        out = json.loads(stdout.strip())
        mse = float(out["mse"])
        if not math.isfinite(mse) or mse < 0:
            return None
        base = 1.0 / (1.0 + mse)
        grad_err = float(out.get("grad_err", 0.0))
        eikonal_penalty = math.exp(-0.5 * grad_err) if math.isfinite(grad_err) else 0.1
        eikonal_fitness = base * eikonal_penalty

        # Topology oracle — fail-safe: non-OK result yields 0.0/0.0 but
        # keeps the candidate alive on the weighted eikonal score.
        topology_score, sign_change_density = compute_topology_score(
            sdf_code, instance, executor, run_timeout=run_timeout
        )

        if topology_score > 0.0:
            fitness = 0.6 * eikonal_fitness + 0.4 * topology_score
        else:
            # topology harness failed — fall back to eikonal-only score
            fitness = eikonal_fitness

        # Stash per-instance diagnostics for publish_candidate to read back.
        # Avoids changing evaluate_candidate's return type (kernel_base expects float|None).
        instance._last_grad_err = grad_err
        instance._last_mse = mse
        instance._last_topology_score = topology_score
        instance._last_sign_change_density = sign_change_density
        return fitness
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.debug("SDF candidate output unparseable on %s: %s", instance.name, exc)
        return None


# ---------------------------------------------------------------------------
# Island personas + sketch library for SDF diversity
# ---------------------------------------------------------------------------

SDF_ISLAND_PERSONAS = [
    ("implicit surface geometer",
     "Think in terms of classical implicit surfaces: gyroid, Schwarz P/D, Neovius, torus knots. "
     "Use signed-distance combinations (fminf for union, fmaxf for intersection, smooth-union blends)."),
    ("domain-warp specialist",
     "Apply domain distortions before evaluating the base shape: sinusoidal twists, cylindrical folding, "
     "noise-based displacement. Warped domains often fit periodic/complex SDFs that analytic primitives miss."),
    ("polynomial approximator",
     "Use polynomial and rational function approximations: Taylor expansions around the surface, "
     "Chebyshev-style fits, or piecewise quadratics. Think like a numerical analyst fitting a curve."),
    ("trig-lattice explorer",
     "Exploit periodic structures: Fourier-style sums of sin/cos at multiple frequencies and axes. "
     "Gyroid and similar TPMS surfaces are naturally represented this way."),
    ("geometric combinator",
     "Combine simple primitives (spheres, boxes, cylinders, capsules) via smooth-min/max with "
     "varying blend radius k. Multi-stage smooth unions with different k values create rich surfaces."),
    ("volumetric blob sculptor",
     "Build organic shapes by smooth-union-blending 5-10 spheres at organic offsets with varying radii. "
     "Cloud, flesh, and cumulus structures emerge from asymmetric multi-sphere compositions with k=0.3-0.5. "
     "Position spheres to create flat bases and billowing tops: lower spheres wide, upper spheres small."),
    ("curve-tube sweeper",
     "Define a parametric curve C(t) = (R*cos(at), pitch*t, R*sin(at)) for helices, "
     "or C(t) = ((R+r*cos(qt))*cos(pt), (R+r*cos(qt))*sin(pt), r*sin(qt)) for torus knots. "
     "SDF = min_t dist(p, C(t)) - tube_radius. Unroll via atan2f to find nearest turn index."),
    ("minimal surface analyst",
     "Explore doubly- and triply-periodic minimal surfaces: gyroid, Schwarz-P, Scherk's saddle. "
     "Represent as f(x,y,z) = 0 where f is a sum of trig/exp terms, then approximate SDF as |f|/|∇f| "
     "using analytically computed gradient magnitude."),
]

SDF_PROMPT_SKETCHES = [
    ("smooth-union cascade",
     "Build from 2-3 simple primitives (spheres/boxes) combined with smooth-min (k=0.1–0.5). "
     "Vary k per level: tight blends near the surface, loose blends for large-scale shape."),
    ("trigonometric lattice",
     "Represent the surface as a linear combination of sin(a*x+b)*cos(c*y+d) terms across all 3 axes. "
     "Gyroid = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x); try variations with frequency scaling."),
    ("domain-twisted primitive",
     "Start from a known primitive (sphere radius r), then warp the domain: "
     "x' = x + 0.2*sin(4*y), y' = y + 0.2*sin(4*z), then evaluate the primitive at (x',y',z')."),
    ("Fourier series SDF",
     "Decompose the target into frequency components: use 2-4 terms of sin/cos per axis at "
     "frequencies 1, 2, 4. Combine additively with learned coefficients (try ±0.3–1.0 range)."),
    ("level-set blending",
     "Define multiple implicit surfaces f1, f2, f3 and blend: 0.4*f1 + 0.3*f2 + 0.3*f3. "
     "The blended level set at 0 is a weighted combination of their zero-crossings."),
    ("blob cluster cascade",
     "Place 5-9 spheres at organic offset positions with varying radii (0.2-0.6). "
     "Apply smooth-min (k=0.35-0.5) in a cascade: accumulate smallest-to-largest. "
     "For cloud/organic shapes: bias lower spheres wider, upper spheres smaller and offset upward."),
    ("parametric curve tube",
     "Define knot or helix curve C(t): for helix C(t)=(R*cos(t), pitch*t/2π, R*sin(t)); "
     "for (p,q) torus knot C(t)=((R+r*cos(qt))*cos(pt), (R+r*cos(qt))*sin(pt), r*sin(qt)). "
     "SDF = min_t ||p-C(t)||₂ - tube_r. Use atan2f(z,x) to find starting t, then loop ±3 turns."),
    ("exponential minimal surface",
     "Use the level-set form: f = exp(z)*cos(y) - cos(x) for Scherk, "
     "or f = cos(x) + cos(y) + cos(z) for Schwarz-P, f = sin(x)cos(y) + ... for gyroid. "
     "Approximate SDF = |f| / |∇f| where ∇f is computed analytically per term."),
]


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

def build_llm_prompt(
    island: Island,
    top_programs: list[CandidateProgram],
    generation: int,
    instance_names: list[str],
    hint: str = "",
) -> str:
    """Build the prompt for evolving a new sdf() function."""
    exemplars = ""
    for i, p in enumerate(sorted(top_programs, key=lambda x: -x.fitness)[:3]):
        exemplars += f"\n// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}\n"

    persona_name, persona_hint = SDF_ISLAND_PERSONAS[island.id % len(SDF_ISLAND_PERSONAS)]
    sketch_name, sketch_desc = SDF_PROMPT_SKETCHES[(island.id + generation) % len(SDF_PROMPT_SKETCHES)]

    if len(instance_names) == 1:
        instance_header = f"Target instance (this island evaluates ONLY this): **{instance_names[0]}**"
    else:
        instance_header = f"Benchmark instances: {', '.join(instance_names)}"

    hint_section = ""
    if hint:
        hint_section = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

    return f"""You are a shader math expert acting as a "{persona_name}".
{persona_hint}

Generate a new C++ `sdf(x, y, z)` function that approximates a target signed-distance field.

{SDF_FUNCTION_SIGNATURE}

{instance_header}
Island {island.id} — Generation {generation}

Approach hint ({sketch_name}): {sketch_desc}
{hint_section}
Top programs in this island:
{exemplars}

Your goal: write a sdf() function that MINIMISES the Mean Squared Error between its output and the ground-truth SDF values at a set of 3D sample points.

Fitness = 1 / (1 + MSE). Perfect match = fitness 1.0.

Rules:
- Return ONLY the C++ sdf() implementation in a single ```cpp code block
- Use `extern "C" float sdf(float x, float y, float z)`
- Include `<cmath>` functions (sinf, cosf, sqrtf, fabsf, fmaxf, fminf, powf)
- Do NOT redefine structs or main() — only provide sdf()
- Keep it under 40 lines
- Try to identify the geometric structure from the instance name and fitness signal
- Combining domain twists, smooth unions, and trig-based implicit surfaces usually beats axis-aligned primitives
"""


def parse_llm_response(response: str) -> str:
    """Extract a C++ sdf() function from an LLM response.

    Tries markdown fences first, then explicit markers, then bare function.
    Returns "" on failure — the caller treats that as a 0.0 fitness candidate.
    """
    if not response or not response.strip():
        return ""
    text = response.strip()

    # 1. Markdown code fence (the common case for all major models)
    fence = re.search(r"```[a-zA-Z0-9_+\-]*[ \t]*\n?(.*?)```", text, re.DOTALL)
    if fence and fence.group(1).strip():
        return fence.group(1).strip()

    # 2. Bare function definition starting with extern "C" or float sdf
    sig = re.search(r'(?:extern\s+"C"\s+)?(?:inline\s+)?float\s+sdf\s*\(', text)
    if sig:
        return text[sig.start():].strip()

    # 3. Last resort: strip stray fences and hope it compiles
    return text.replace("```cpp", "").replace("```c++", "").replace("```", "").strip()


# ---------------------------------------------------------------------------
# Main kernel class — real FunSearchKernel subclass (instance #2)
# ---------------------------------------------------------------------------

class SDFKernel(FunSearchKernel):
    """FunSearch-style SDF heuristic discovery kernel.

    Subclasses FunSearchKernel and implements the five abstract methods.
    This is the second concrete FunSearch kernel after TSP (instance #1).

    The kernel evolves compact C++ sdf(x,y,z) functions that approximate
    target signed-distance fields. Fitness = 1/(1+MSE), where MSE is
    computed over a fixed set of precomputed sample points per instance.
    The target function is never exposed to the LLM — it only sees sample
    coordinates and expected distances.
    """

    BUS_CHANNEL_PREFIX = "sdf"

    def __init__(self, config: KernelConfig):
        if not config.instances:
            config.instances = ["gyroid", "round_box", "warped_sphere"]

        # Enable island-per-instance routing: each island evaluates against
        # only one instance (island_id % n_instances), specialising its search.
        config.island_instance_assignment = True

        # Build executor BEFORE super().__init__() — fail fast if no sandbox.
        # (The base class does not build an executor; subclasses own this.)
        self.executor = ensure_executor(
            allow_unsandboxed=config.allow_unsandboxed,
            max_memory_mb=config.max_memory_mb,
        )
        logger.info("SDF kernel sandbox: %s", self.executor.sandbox_type)

        super().__init__(config)

        # The base class leaves self.problem_instances = [] (load_instances is
        # abstract and intentionally not called in __init__). We populate it now.
        self.problem_instances = self.load_instances()
        logger.info(
            "Loaded %d SDF instance(s): %s",
            len(self.problem_instances),
            [inst.name for inst in self.problem_instances],
        )

    # ------------------------------------------------------------------
    # FunSearchKernel abstract interface — all five methods
    # ------------------------------------------------------------------

    def load_instances(self) -> list[SDFInstance]:
        """Generate synthetic benchmark instances from config.instances list."""
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info(
                "Generated SDF instance '%s': %d samples, bbox=[%.1f,%.1f]",
                inst.name, inst.n_samples, inst.bbox[0], inst.bbox[1],
            )
            instances.append(inst)
        return instances

    def evaluate_candidate(self, code: str, instance: SDFInstance) -> float | None:
        """Evaluate sdf_code on one SDF instance. Returns fitness in (0,1] or None."""
        return evaluate_on_instance(
            code,
            instance,
            self.executor,
            run_timeout=self.config.run_timeout,
        )

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        """Build LLM prompt for evolving a sdf() function.

        With island_instance_assignment=True, each island evaluates only its
        assigned instance — pass only that name so the LLM knows its target.
        """
        if self.config.island_instance_assignment and self.problem_instances:
            inst_idx = island.id % len(self.problem_instances)
            instance_names = [self.problem_instances[inst_idx].name]
        else:
            instance_names = [inst.name for inst in self.problem_instances]
        return build_llm_prompt(
            island,
            top_programs,
            generation,
            instance_names=instance_names,
            hint=hint,
        )

    def parse_response(self, response: str) -> str:
        """Extract C++ sdf() from an LLM response."""
        return parse_llm_response(response)

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        """Return 3 analytical baseline programs for a new island.

        With island_instance_assignment=True, pick seeds relevant to the
        instance this island will specialise on.
        """
        if self.config.island_instance_assignment and self.problem_instances:
            inst_idx = island_id % len(self.problem_instances)
            first_name = self.problem_instances[inst_idx].name
        else:
            first_name = self.problem_instances[0].name if self.problem_instances else "generic"
        seeds = get_seed_programs(first_name)

        programs = []
        for variant_name, code in seeds:
            programs.append(CandidateProgram(
                id=f"island{island_id}_gen{generation}_{variant_name}",
                code=code,
                island=island_id,
                generation=generation,
                source="baseline",
            ))
        return programs

    def evaluate_fitness(self, program: CandidateProgram) -> tuple[float, float, float]:
        """Evaluate program against one or all instances.

        With island_instance_assignment=True, each island evaluates only the
        single instance whose index matches (island_id % n_instances).  This
        specialises the search so that islands compete for different instances
        instead of averaging over all three, which would blur the per-topology
        gradient signal.
        """
        t0 = time.time()
        if self.config.island_instance_assignment and self.problem_instances:
            inst_idx = program.island % len(self.problem_instances)
            instances = [self.problem_instances[inst_idx]]
        else:
            instances = self.problem_instances

        ratios: list[float] = []
        for inst in instances:
            r = self.evaluate_candidate(program.code, inst)
            ratios.append(r if r is not None else 0.0)
        elapsed = (time.time() - t0) * 1000

        if not ratios:
            return 0.0, 0.0, 0.0
        mean = sum(ratios) / len(ratios)
        var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        worst = min(ratios)
        program.fitness = mean
        program.fitness_variance = var
        program.worst_fitness = worst
        program.computation_time_ms = elapsed
        program.evaluated = True
        # Snapshot per-instance diagnostics onto the program now, while the instance
        # attributes are current for this evaluation. extract_t_vector() reads these
        # per-program attrs rather than the stale instance attrs, giving correct T-vectors
        # when iterating over multiple programs in the prior append loop.
        g_errs = [getattr(i, "_last_grad_err", None) for i in instances]
        t_scores = [getattr(i, "_last_topology_score", None) for i in instances]
        valid_ge = [g for g in g_errs if g is not None]
        valid_t = [t for t in t_scores if t is not None]
        program._eikonal_score = math.exp(-0.5 * sum(valid_ge) / len(valid_ge)) if valid_ge else 0.0
        program._topology_score = sum(valid_t) / len(valid_t) if valid_t else 0.0
        return mean, var, worst

    def extract_t_vector(self, program: CandidateProgram) -> dict:
        """Extract sufficient-statistic T-vector for this program.

        Reads eikonal and topology scores stashed per-program by evaluate_fitness(),
        giving correct per-program values rather than stale instance attributes.
        """
        return {
            "fitness": program.fitness,
            "eikonal_score": getattr(program, "_eikonal_score", 0.0),
            "topology_score": getattr(program, "_topology_score", 0.0),
            "code_length": len(program.code),
        }

    # ------------------------------------------------------------------
    # Bus publishing — emit on schema'd channels.
    # _publish_candidate: overridden below (adds topology + eikonal fields).
    # _publish_generation: not overridden — base class handles generation events.
    # ------------------------------------------------------------------

    def _find_nervous_bin(self) -> str | None:
        candidates = [
            Path(__file__).parent.parent.parent / "sdk" / "shell" / "nervous",
            Path.home() / "projects" / "nervous-bus" / "sdk" / "shell" / "nervous",
            Path("/usr/local/bin/nervous"),
            Path("/usr/bin/nervous"),
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
        return None

    def _publish(self, channel: str, data: dict) -> bool:
        """Publish to bus debug log and optionally the nervous CLI.

        Source is hardcoded to '/autobench/sdf_kernel' to match the schema
        const — the base class uses the class name dynamically, which would
        produce '/autobench/sdfkernel' (no underscore).
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/sdf_kernel",
            "type": channel,
            "datacontenttype": "application/json",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data": data,
        }
        payload = json.dumps(envelope)

        debug_path = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(debug_path, "a") as f:
                f.write(payload + "\n")
            if self.config.bus_verbose:
                logger.info("bus: published %s", channel)
        except Exception as e:
            logger.debug("bus: write to debug log failed: %s", e)

        if self._nervous_bin:
            try:
                env = dict(os.environ)
                env["NBUS_SKIP_VALIDATION"] = "1"
                env["NERVOUS_NO_ZELLIJ"] = "1"
                env["NERVOUS_NO_REDIS"] = "1"
                proc = subprocess.Popen(
                    [self._nervous_bin, "publish", channel, payload],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
                proc.wait(timeout=2)
            except Exception:
                pass

        return True

    def _render_best_program(self, best: "CandidateProgram", out_path: "Path") -> bool:
        """Render the evolved SDF via in-house CPU sphere tracer (fallback: GLSL probe)."""
        # Prefer the CPU tracer — it works headless with zero VRAM, and takes
        # the C++ code directly without translation. Fall back to the GLSL path
        # (ShaderExecutor) if the CPU tracer isn't available.
        if self.config.allow_unsandboxed:
            try:
                from ..sdf_tracer import render_sdf_cpp_to_png  # type: ignore
                from ..artifact_store import _INSTANCE_CAMERA_DIST  # type: ignore
                inst_name = (
                    self.problem_instances[best.island % len(self.problem_instances)].name
                    if self.problem_instances else ""
                )
                cam_dist = _INSTANCE_CAMERA_DIST.get(inst_name, 3.5)
                return render_sdf_cpp_to_png(
                    best.code, out_path, viewport=(256, 256), camera_dist=cam_dist
                )
            except Exception as e:
                logger.debug("CPU tracer unavailable (%s), trying GLSL path", e)

        from ..artifact_store import render_sdf_to_png  # type: ignore
        glsl = best.code.replace(
            "extern \"C\" float sdf(float x, float y, float z)",
            "float sdf(vec3 p)"
        )
        if "float x=p.x" not in glsl:
            glsl = glsl.replace(
                "float sdf(vec3 p) {",
                "float sdf(vec3 p) {\n    float x=p.x, y=p.y, z=p.z;"
            )
        return render_sdf_to_png(glsl, out_path)

    def _artifact_render_type(self) -> str:
        return "sdf_raymarch"

    def _publish_started(self) -> None:
        """Emit sdf.kernel.started.v1 when the run begins."""
        from ..kernel_base import _git_commit_short
        self._publish("sdf.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": [inst.name for inst in self.problem_instances],
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
            "sandbox_type": self.executor.sandbox_type,
        })

    def _publish_candidate(self, candidate: CandidateProgram) -> None:
        """Emit sdf.candidate.evaluated.v1 with eikonal and topology diagnostics."""
        if not candidate.evaluated or candidate.fitness <= 0:
            return
        # Collect per-instance diagnostics stashed by evaluate_on_instance.
        grad_errs = [getattr(inst, "_last_grad_err", None)
                     for inst in self.problem_instances]
        mses = [getattr(inst, "_last_mse", None)
                for inst in self.problem_instances]
        topo_scores = [getattr(inst, "_last_topology_score", None)
                       for inst in self.problem_instances]
        sign_densities = [getattr(inst, "_last_sign_change_density", None)
                          for inst in self.problem_instances]

        valid_ge = [g for g in grad_errs if g is not None]
        valid_mse = [m for m in mses if m is not None]
        valid_topo = [t for t in topo_scores if t is not None]
        valid_density = [d for d in sign_densities if d is not None]

        mean_grad_err = sum(valid_ge) / len(valid_ge) if valid_ge else None
        mean_mse = sum(valid_mse) / len(valid_mse) if valid_mse else None
        mean_topo = sum(valid_topo) / len(valid_topo) if valid_topo else None
        mean_density = sum(valid_density) / len(valid_density) if valid_density else None

        self._publish("sdf.candidate.evaluated.v1", {
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": candidate.id,
            "island": candidate.island,
            "fitness": round(candidate.fitness, 6),
            "fitness_variance": round(candidate.fitness_variance, 6),
            "worst_fitness": round(candidate.worst_fitness, 6),
            "eikonal_score": round(math.exp(-0.5 * mean_grad_err), 4) if mean_grad_err is not None else None,
            "mean_grad_err": round(mean_grad_err, 4) if mean_grad_err is not None else None,
            "mean_mse": round(mean_mse, 6) if mean_mse is not None else None,
            "topology_score": round(mean_topo, 4) if mean_topo is not None else None,
            "sign_change_density": round(mean_density, 6) if mean_density is not None else None,
            "source": candidate.source,
            "code_length": len(candidate.code),
            "computation_time_ms": round(candidate.computation_time_ms, 1),
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        """Emit sdf.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        # Include eikonal diagnostics in the best_program block.
        best_grad_err = None
        if best:
            grad_errs = [getattr(inst, "_last_grad_err", None)
                         for inst in self.problem_instances]
            valid = [g for g in grad_errs if g is not None]
            best_grad_err = sum(valid) / len(valid) if valid else None
        self._publish("sdf.kernel.completed.v1", {
            "run_id": self.run_id,
            "total_generations": self.generation,
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "best_program": {
                "id": best.id if best else "",
                "fitness": best.fitness if best else 0.0,
                "eikonal_score": round(math.exp(-0.5 * best_grad_err), 4)
                                 if best_grad_err is not None else None,
                "mean_grad_err": round(best_grad_err, 4) if best_grad_err is not None else None,
                "island": best.island if best else -1,
                "generation": best.generation if best else -1,
                "source": best.source if best else "",
                "sdf_code": best.code if best else "",
            } if best else None,
            "history": self.history,
        })

    def save_results(self, programs: list[CandidateProgram], path: Path | None = None) -> None:
        """Save results to JSON (mirrors TSP convention)."""
        if path is None and self.config.output_dir:
            path = self.config.output_dir / f"sdf_results_gen{self.generation}.json"
        if path is None:
            return
        best = programs[0] if programs else None
        output = {
            "config": {
                "instances": self.config.instances,
                "n_islands": self.config.n_islands,
                "population_per_island": self.config.population_per_island,
                "generations": self.generations,
            },
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "history": self.history,
            "best_program": {
                "id": best.id,
                "fitness": best.fitness,
                "sdf_code": best.code,
                "source": best.source,
                "island": best.island,
                "generation": best.generation,
            } if best else None,
            "top_programs": [
                {
                    "id": p.id,
                    "fitness": p.fitness,
                    "fitness_variance": p.fitness_variance,
                    "worst_fitness": p.worst_fitness,
                    "island": p.island,
                    "generation": p.generation,
                    "computation_time_ms": p.computation_time_ms,
                    "source": p.source,
                    "sdf_code": p.code,
                }
                for p in programs[:10]
            ],
        }
        path.write_text(json.dumps(output, indent=2))
        logger.info("Saved SDF results to %s", path)
