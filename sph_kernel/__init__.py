"""SPH kernel — FunSearch evolution of smoothing kernels for real-time fluid simulation.

# Domain: SPH Smoothing Kernel W(r, h)

Smoothed Particle Hydrodynamics quality is entirely determined by the choice of
W(r, h): the weighting function that governs how each particle contributes to
field estimates at arbitrary positions. Current state-of-the-art (cubic spline,
Wendland C²) are hand-derived analytical solutions. FunSearch can discover
kernels that outperform these on specific scenarios (real-time density fields,
surface tension, turbulent flow) while respecting the physical invariants.

## Oracle: density-reconstruction MSE

The oracle places N particles in a 3D box with masses proportional to a known
multi-scale Gaussian density field ρ_true(x). At M off-lattice probe points the
evolved kernel reconstructs:

    ρ_est(x) = Σⱼ mⱼ * sph_kernel(|x - xⱼ|, h)

Fitness = 1/(1+MSE) where MSE = mean over probes of (ρ_est - ρ_true)².

This is a **generative membership oracle** — any kernel that correctly
interpolates density fields belongs to the class. A constant offset kernel or
a tent function both satisfy partition-of-unity but fail this oracle because
they don't correctly weight the off-lattice probes.

## Hard preconditions (reject if violated)
- Compact support: |W(h·1.001, h)| < 1e-3   (zero outside support)
- Positivity:       W(0, h) ≥ 0              (physical requirement)

## Why this beats partition+monotone+smooth as oracle
The cubic spline integrates to 1.0 and is monotone and C² by construction —
it scores ~1.0 on a math-properties oracle at gen 0, giving no gradient to
climb. The density-reconstruction oracle is hard because jittered particles
and off-lattice probes make kernel shape (not just normalization) load-bearing.
See: funsearch-sufficient-statistic-theory, funsearch-combined-oracle-sdf-gen0-0997.

## TEngine integration path
The evolved W(r,h) is a scalar function that plugs directly into a TEngine
compute shader for SPH particle simulation:
  for each probe particle: density += neighbor_mass * sph_kernel(dist, h)
No intermediate conversion needed — this is a shorter integration path than
the SDF kernel (which needs SVDAG bake + raymarch pipeline).

## Connection to SDF kernel
High-quality SPH fluid simulation requires correct boundary conditions from
scene SDFs. -∇SDF gives the contact normal; SDF < 0 triggers boundary forces.
The eikonal-valid evolved SDFs from sdf_kernel are the natural boundary
representation for this fluid simulation.

Usage:
    python -m autobench.sph_kernel run --instances gauss_blobs_3d \\
        --generations 60 --islands 6 --population 12 --allow-unsandboxed \\
        --plateau-generations 10 --candidates-per-island 1 --max-concurrent-llm 6
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
)
from ..core import Verdict
from ..sandbox import SandboxedExecutor, compile_and_run

logger = logging.getLogger(__name__)


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
# C++ evaluator template
# The evolved sph_kernel() is injected at EVOLVED_CODE_MARKER.
# Input (stdin): JSON with particles, probes, h, name.
# Output: JSON line with mse, compact_ok, positive_ok, n, instance.
# ---------------------------------------------------------------------------

_SPH_EVALUATOR_CPP = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <array>

using namespace std;

// ===== EVOLVED_CODE_MARKER =====

struct Particle { float x, y, z, mass; };
struct Probe    { float x, y, z, rho;  };

// Minimal JSON parser (no dependencies)
struct P {
    const string& s; int pos;
    P(const string& src): s(src), pos(0) {}
    void ws(){ while(pos<(int)s.size()&&(s[pos]==' '||s[pos]=='\n'||s[pos]=='\r'||s[pos]=='\t'))pos++; }
    void expect(char c){ ws(); if(pos>=(int)s.size()||s[pos]!=c){fprintf(stderr,"expected '%c'\n",c);exit(1);} pos++; }
    string str(){ expect('"'); string r; while(pos<(int)s.size()&&s[pos]!='"')r+=s[pos++]; expect('"'); return r; }
    double num(){
        ws(); bool neg=pos<(int)s.size()&&s[pos]=='-'; if(neg)pos++;
        double v=0;
        while(pos<(int)s.size()&&isdigit((unsigned char)s[pos]))v=v*10+(s[pos++]-'0');
        if(pos<(int)s.size()&&s[pos]=='.'){pos++;double f=0.1;while(pos<(int)s.size()&&isdigit((unsigned char)s[pos])){v+=f*(s[pos++]-'0');f*=0.1;}}
        if(pos<(int)s.size()&&(s[pos]=='e'||s[pos]=='E')){pos++;bool en=s[pos]=='-';if(en||s[pos]=='+')pos++;int e=0;while(pos<(int)s.size()&&isdigit((unsigned char)s[pos]))e=e*10+(s[pos++]-'0');double m=1;for(int i=0;i<e;i++)m*=10;if(en)v/=m;else v*=m;}
        return neg?-v:v;
    }
    vector<array<float,4>> arr4(){
        expect('['); vector<array<float,4>> r; ws();
        while(pos<(int)s.size()&&s[pos]!=']'){
            expect('[');
            float a=(float)num(); expect(',');
            float b=(float)num(); expect(',');
            float c=(float)num(); expect(',');
            float d=(float)num();
            ws(); if(pos<(int)s.size()&&s[pos]==']')pos++;
            r.push_back({a,b,c,d});
            ws(); if(pos<(int)s.size()&&s[pos]==',')pos++;
            ws();
        }
        expect(']'); return r;
    }
    void skip(){ ws(); if(pos>=(int)s.size())return;
        if(s[pos]=='"'){str();return;}
        if(s[pos]=='['||s[pos]=='{'){int d=0;while(pos<(int)s.size()){if(s[pos]=='['||s[pos]=='{')d++;else if(s[pos]==']'||s[pos]=='}'){d--;pos++;if(!d)break;}else pos++;}return;}
        while(pos<(int)s.size()&&s[pos]!=','&&s[pos]!='}'&&s[pos]!=']')pos++;
    }
};

int main() {
    string inp((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());
    P p(inp);
    p.expect('{');

    string inst_name; float h=0.4f;
    vector<Particle> parts; vector<Probe> prbs;

    while(true){
        p.ws(); if(p.pos>=(int)inp.size()||inp[p.pos]=='}'){p.pos++;break;}
        string k=p.str(); p.expect(':');
        if(k=="name")      inst_name=p.str();
        else if(k=="h")    h=(float)p.num();
        else if(k=="particles"){ for(auto& t:p.arr4()) parts.push_back({t[0],t[1],t[2],t[3]}); }
        else if(k=="probes"){    for(auto& t:p.arr4()) prbs.push_back({t[0],t[1],t[2],t[3]}); }
        else p.skip();
        p.ws(); if(p.pos<(int)inp.size()&&inp[p.pos]==',')p.pos++;
    }

    if(parts.empty()||prbs.empty()){
        fprintf(stderr,"ERROR: no particles/probes parsed (%d/%d)\n",(int)parts.size(),(int)prbs.size());
        return 1;
    }

    // Hard preconditions
    int compact_ok = (fabsf(sph_kernel(h*1.001f, h)) < 1e-3f) ? 1 : 0;
    int positive_ok= (sph_kernel(0.0f, h) >= 0.0f) ? 1 : 0;

    if(!compact_ok || !positive_ok){
        printf("{\"mse\":100.0,\"compact_ok\":%d,\"positive_ok\":%d,\"n\":0,\"instance\":\"%s\"}\n",
               compact_ok, positive_ok, inst_name.c_str());
        return 0;
    }

    // Density reconstruction MSE
    double mse=0.0;
    int n=(int)prbs.size();
    for(int i=0;i<n;i++){
        Probe& q=prbs[i];
        float rho_est=0.0f;
        for(auto& pt:parts){
            float dx=q.x-pt.x, dy=q.y-pt.y, dz=q.z-pt.z;
            float r=sqrtf(dx*dx+dy*dy+dz*dz);
            rho_est += pt.mass * sph_kernel(r, h);
        }
        double err=(double)rho_est - (double)q.rho;
        mse += err*err;
    }
    mse /= n;

    // Monotonicity score: W(r,h) must be monotone decreasing r ∈ [0,h].
    // Violation means W has a local maximum away from r=0 → SPH pressure
    // gradient direction flips sign, breaking momentum conservation.
    // Sample 32 equidistant points; each non-monotone step costs 1/32.
    float mono_score = 1.0f;
    {
        int N_m = 32;
        float prev_w = sph_kernel(0.0f, h);
        int violations = 0;
        for (int k = 1; k <= N_m; k++) {
            float r_k = h * (float)k / N_m;
            float w_k = sph_kernel(r_k, h);
            if (w_k > prev_w + 1e-6f) violations++;
            prev_w = w_k;
        }
        mono_score = 1.0f - (float)violations / N_m;
    }

    // Gradient direction check: d/dr W(r,h) at r=h/2 should be negative.
    // Uses central difference. Kernels with W'(h/2) > 0 push particles apart
    // when they should attract → wrong pressure forces.
    float eps_r = h * 0.01f;
    float half_h = h * 0.5f;
    float dW_dr = (sph_kernel(half_h + eps_r, h) - sph_kernel(half_h - eps_r, h)) / (2.0f * eps_r);
    float grad_ok = (dW_dr < 0.0f) ? 1.0f : 0.0f;

    printf("{\"mse\":%.9f,\"compact_ok\":%d,\"positive_ok\":%d,\"mono_score\":%.4f,"
           "\"grad_ok\":%.1f,\"n\":%d,\"instance\":\"%s\"}\n",
           mse, compact_ok, positive_ok, mono_score, grad_ok, n, inst_name.c_str());
    return 0;
}
"""

SPH_FUNCTION_SIGNATURE = '''\
extern "C" float sph_kernel(float r, float h);

// Evolved SPH smoothing kernel W(r, h).
// r: distance between particle and evaluation point (r >= 0)
// h: smoothing length (compact support radius; W must be 0 for r >= h)
//
// PHYSICAL REQUIREMENTS (checked as hard preconditions):
//   Compact support: sph_kernel(h, h) == 0  (exactly zero outside support)
//   Positivity:      sph_kernel(0, h) >= 0
//
// WHAT MAKES A GOOD KERNEL:
//   The oracle places particles with mass ∝ ρ_true at jittered positions,
//   then checks how accurately Σ mⱼ·W(|x-xⱼ|,h) reconstructs ρ_true at
//   off-lattice probe points. Kernel *shape* determines accuracy — a flat
//   top-hat or tent function pass the preconditions but fail the oracle.
//
// Known kernels for reference:
//   Cubic spline:   σ·{6(q³-q²)+1 for q<0.5; 2(1-q)³ for 0.5≤q<1; 0 for q≥1}
//                   σ = 8/(πh³) in 3D
//   Wendland C²:    σ·(1-q)⁴(4q+1) for q<1; 0 otherwise
//                   σ = 21/(2πh³) in 3D
//   where q = r/h
//
// Available C math:  sqrtf, fabsf, fmaxf, fminf, sinf, cosf, powf, expf, logf
'''


# ---------------------------------------------------------------------------
# Seed programs — known-good baselines for the FunSearch prior
# ---------------------------------------------------------------------------

_CUBIC_SPLINE_SEED = '''\
extern "C" float sph_kernel(float r, float h) {
    float q = r / h;
    if (q >= 1.0f) return 0.0f;
    const float sigma = 8.0f / (3.14159265f * h * h * h);
    if (q <= 0.5f)
        return sigma * (6.0f * q * q * (q - 1.0f) + 1.0f);
    float d = 1.0f - q;
    return sigma * 2.0f * d * d * d;
}'''

_WENDLAND_C2_SEED = '''\
extern "C" float sph_kernel(float r, float h) {
    float q = r / h;
    if (q >= 1.0f) return 0.0f;
    const float sigma = 21.0f / (2.0f * 3.14159265f * h * h * h);
    float d = 1.0f - q;
    return sigma * d * d * d * d * (4.0f * q + 1.0f);
}'''

_QUINTIC_SEED = '''\
extern "C" float sph_kernel(float r, float h) {
    // Quintic spline — smoother than cubic, better for surface tension
    float q = r / h;
    if (q >= 1.0f) return 0.0f;
    const float sigma = 3.0f / (359.0f * 3.14159265f * h * h * h);
    float d1 = 1.0f - q;
    float d3 = (q <= 1.0f/3.0f) ? (1.0f/3.0f - q) : 0.0f;
    float d2 = (q <= 2.0f/3.0f) ? (2.0f/3.0f - q) : 0.0f;
    float v = d1*d1*d1*d1*d1
            - 6.0f * d2*d2*d2*d2*d2
            + 15.0f * d3*d3*d3*d3*d3;
    return sigma * v;
}'''

# Seed registry: name → (label, code)
SEED_SPH_PROGRAMS: dict[str, list[tuple[str, str]]] = {
    "generic": [
        ("cubic_spline",  _CUBIC_SPLINE_SEED),
        ("wendland_c2",   _WENDLAND_C2_SEED),
        ("quintic",       _QUINTIC_SEED),
    ],
    "gauss_blobs_3d": [
        ("cubic_spline",  _CUBIC_SPLINE_SEED),
        ("wendland_c2",   _WENDLAND_C2_SEED),
        ("quintic",       _QUINTIC_SEED),
    ],
}


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    return SEED_SPH_PROGRAMS.get(instance_name, SEED_SPH_PROGRAMS["generic"])


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


# ---------------------------------------------------------------------------
# C++ source construction
# ---------------------------------------------------------------------------

def build_candidate_source(code: str, instance: SPHInstance) -> tuple[str, str]:
    """Return (cpp_source, json_stdin) for one evaluation.

    The evolved sph_kernel() is inserted into the evaluator template;
    instance data is serialised as JSON for stdin.
    """
    cpp = _SPH_EVALUATOR_CPP.replace("// ===== EVOLVED_CODE_MARKER =====", code, 1)

    data = {
        "name": instance.name,
        "h": instance.h,
        "particles": [[p[0], p[1], p[2], p[3]] for p in instance.particles],
        "probes":    [[q[0], q[1], q[2], q[3]] for q in instance.probes],
    }
    return cpp, json.dumps(data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Single-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_on_instance(
    code: str,
    instance: SPHInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 30.0,
    compile_timeout: float = 20.0,
) -> Optional[float]:
    """Compile + run evolved kernel; return fitness in (0,1] or None on failure."""
    cpp_source, stdin_data = build_candidate_source(code, instance)

    stdout, verdict, _latency = compile_and_run(
        cpp_source,
        language="cpp",
        stdin=stdin_data,
        executor=executor,
    )
    if verdict != Verdict.OK or not stdout:
        return None

    try:
        out = json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None

    compact_ok  = out.get("compact_ok", 0)
    positive_ok = out.get("positive_ok", 0)
    if not compact_ok or not positive_ok:
        return None     # hard precondition violated

    mse = float(out.get("mse", 10.0))
    if not math.isfinite(mse) or mse < 0:
        return None

    density_fitness = 1.0 / (1.0 + mse)
    mono_score      = float(out.get("mono_score", 0.0))
    grad_ok         = float(out.get("grad_ok",    0.0))

    # Fitness = 0.70·density + 0.20·monotone + 0.10·gradient_direction
    # Monotone decreasing W ensures SPH pressure forces have correct sign.
    # A kernel with W'(h/2) > 0 would push particles apart under compression.
    return 0.70 * density_fitness + 0.20 * mono_score + 0.10 * grad_ok


# ---------------------------------------------------------------------------
# SPH FunSearch kernel
# ---------------------------------------------------------------------------

@register_kernel("sph")
class SPHKernel(FunSearchKernel):
    """FunSearch kernel that evolves SPH smoothing functions W(r, h).

    Oracle: density-reconstruction MSE — how accurately Σ mⱼ·W(|x-xⱼ|,h)
    recovers a known Gaussian blob density field at off-lattice probe points.
    Fitness = 1/(1+MSE); higher is better.
    """

    kernel_name = "sph"
    BUS_CHANNEL_PREFIX = "sph"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        logger.info("SPH kernel sandbox: %s", self.executor.sandbox_type)

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def load_instances(self) -> list[SPHInstance]:
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info("Generated SPH instance %r: %d particles, %d probes, h=%.3f",
                        name, inst.n_particles, inst.n_probes, inst.h)
            instances.append(inst)
        return instances

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(
            code, instance, self.executor,
            run_timeout=self.config.run_timeout,
        )

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        instance_names = [i.name for i in self.problem_instances]
        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )

        hint_block = ""
        if hint:
            hint_block = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        return (
            f"You are a numerical methods expert specialising in SPH fluid simulation kernels.\n"
            f"Evolve a better smoothing kernel W(r,h) for SPH density reconstruction.\n\n"
            f"{SPH_FUNCTION_SIGNATURE}\n\n"
            f"Target instance(s): {', '.join(instance_names)}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            f"Your goal: write a sph_kernel() that MINIMISES density-reconstruction MSE.\n"
            f"Fitness = 1/(1+MSE). Higher fitness = better.\n\n"
            f"Rules:\n"
            f"- Return ONLY the C++ sph_kernel() in a single ```cpp code block\n"
            f"- Signature: extern \"C\" float sph_kernel(float r, float h)\n"
            f"- Include <cmath> functions (sqrtf, fabsf, fmaxf, fminf, powf, expf)\n"
            f"- W(r,h) MUST be 0 for r >= h (compact support — checked as precondition)\n"
            f"- W(0,h) MUST be >= 0 (positivity — checked as precondition)\n"
            f"- No loops, no static arrays — must be evaluable in a single expression path\n"
            f"- Normalisation: 4π∫₀ʰ W(r,h)r²dr should ≈ 1 (partition of unity)\n"
            f"- Keep it under 20 lines\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "sph_kernel" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        seeds = get_seed_programs(
            self.config.instances[0] if self.config.instances else "gauss_blobs_3d"
        )
        programs = []
        for name, code in seeds:
            prog = CandidateProgram(
                id=str(uuid.uuid4()),
                code=code,
                island=island_id,
                generation=generation,
            )
            programs.append(prog)
        return programs

    # ------------------------------------------------------------------
    # Bus publishing — sph.kernel.* channels
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
        """Write event to bus debug log and optionally the nervous CLI.

        Source is hardcoded to '/autobench/sph_kernel' to match the schema
        const.
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/sph_kernel",
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

    def _publish_started(self) -> None:
        """Emit sph.kernel.started.v1 when the run begins."""
        from ..kernel_base import _git_commit_short
        self._publish("sph.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": self.config.instances,
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
            "sandbox_type": self.executor.sandbox_type,
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        """Emit sph.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        self._publish("sph.kernel.completed.v1", {
            "run_id": self.run_id,
            "total_generations": self.generation,
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "best_program": {
                "id": best.id if best else "",
                "fitness": best.fitness if best else 0.0,
                "island": best.island if best else -1,
                "generation": best.generation if best else -1,
                "source": best.source if best else "",
                "sph_code": best.code if best else "",
            } if best else None,
            "history": self.history,
        })

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def save_results(self, programs: list[CandidateProgram]) -> Optional[Path]:
        if not self.config.output_dir:
            return None
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        gen = self.generation
        out_path = out_dir / f"sph_results_gen{gen:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "sph",
            "run_id": self.run_id,
            "generation": gen,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {
                    "id": p.id,
                    "fitness": p.fitness,
                    "worst_fitness": p.worst_fitness,
                    "island": p.island,
                    "generation": p.generation,
                    "instance": getattr(p, "instance", ""),
                    "sph_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("SPH results saved to %s (top fitness=%.4f)", out_path,
                    top[0].fitness if top else 0)
        return out_path
