"""Terrain height-field kernel — FunSearch evolution of geological height functions.

# Domain: Terrain Height Field  float terrain(vec2 p)

Evolves a 2D height function against a *geological class-membership oracle*:
the oracle does NOT compare pixel values — it measures whether the evolved
function has the statistical signature of real terrain at a given target
Hurst exponent H.

## Oracle: Hurst-exponent + normalised-slope (sufficient statistic)

  1. Sample terrain(p) on a 32×32 grid over [-1,1]².
  2. Structure function SF(r) = E[|h(x+r)−h(x)|²] at lags r=1,2,3,4,6,8.
  3. Log-log regression slope = 2H  →  H_est.
  4. Normalised slope = mean(|∇h|) / std(h)  — scale-invariant roughness.

  Fitness = 0.6·hurst_score + 0.3·slope_score + 0.1·range_score

  hurst_score = exp(−8·(H_est − H_target)²)   — Gaussian gate around target
  slope_score = exp(−5·(ns − ns_target)² / (ns_target² + 0.01))
  range_score = min(1, max_h − min_h) / 0.3)  — degenerate-constant guard

This is the **generative membership oracle** for terrain: any function with the
correct fractal persistence belongs to the class, regardless of the specific
hash constants used. This fixes the SSIM hash-realization trap.

## Geological instances (target H values)

  rolling_hills    H=0.82  gentle, coherent — low-relief rural landscape
  mountain_peaks   H=0.66  rugged, persistent ridges — alpine terrain
  eroded_badlands  H=0.48  rough, channelled — high-erosion semiarid terrain
  river_valley     H=0.74  mixed — flat valley floor + steep flanking walls
  volcanic_plateau H=0.59  medium roughness with abrupt calderas / lava flows

## TEngine integration path

  float terrain(vec2 p)  maps directly to a HLSL/Slang compute shader.
  The terrain height function plugs into the height-map generator that feeds
  the SVDAG build pass.  Unlike the SDF kernel (which needs an SVDAG bake),
  terrain height is consumed by the terrain tessellation stage — integration
  is a 1-pass shader swap.

## Connection to other kernels

  SDF kernel  →  rock-formation shapes that cut into terrain height
  Noise kernel →  fine-detail overlay (normal perturbation) on terrain surface
  SPH kernel   →  water flowing over terrain (riverbed geometry from terrain)
  Phase kernel →  water→ice state transition driven by temperature field above terrain
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
from typing import Any, Optional

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
)
from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run

logger = logging.getLogger(__name__)


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
# C++ evaluator template
# GLSL-compatible types + structure-function oracle.
# Evolved terrain() is inserted at EVOLVED_CODE_MARKER.
# Input (stdin): JSON { "name":..., "target_hurst":..., "target_norm_slope":... }
# Output: JSON line with fitness, H_est, mean_slope, height_range, name.
# ---------------------------------------------------------------------------

_TERRAIN_EVALUATOR_CPP = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include <iostream>
#include <cfloat>

using namespace std;

// ── GLSL-compatible types ────────────────────────────────────────────────────
struct vec2 {
    float x, y;
    vec2() : x(0.f), y(0.f) {}
    vec2(float x_, float y_) : x(x_), y(y_) {}
    vec2 operator+(const vec2& b) const { return vec2(x+b.x, y+b.y); }
    vec2 operator-(const vec2& b) const { return vec2(x-b.x, y-b.y); }
    vec2 operator*(float s)        const { return vec2(x*s,   y*s);   }
    vec2 operator*(const vec2& b)  const { return vec2(x*b.x, y*b.y); }
    vec2& operator+=(const vec2& b){ x+=b.x; y+=b.y; return *this; }
};
inline vec2  operator*(float s, const vec2& v) { return vec2(s*v.x, s*v.y); }
inline vec2  operator+(float s, const vec2& v) { return vec2(s+v.x, s+v.y); }
inline vec2  operator-(float s, const vec2& v) { return vec2(s-v.x, s-v.y); }
inline float dot(vec2 a, vec2 b)   { return a.x*b.x + a.y*b.y; }
inline float length(vec2 v)        { return sqrtf(v.x*v.x + v.y*v.y); }
inline vec2  floor(const vec2& v)  { return vec2(floorf(v.x), floorf(v.y)); }
inline vec2  fract(const vec2& v)  { return vec2(v.x-floorf(v.x), v.y-floorf(v.y)); }
inline float fract(float x)        { return x - floorf(x); }
inline float mod(float x, float y) { return fmodf(x, y); }
// Note: use fabsf() for float abs — 'abs' conflicts with std::abs(int)
inline float sign(float x)         { return (x > 0.f) ? 1.f : (x < 0.f) ? -1.f : 0.f; }
inline float clamp(float x, float lo, float hi) { return fmaxf(lo, fminf(hi, x)); }
inline float mix(float a, float b, float t) { return a + t*(b-a); }
inline float smoothstep(float e0, float e1, float x) {
    float t = clamp((x-e0)/(e1-e0), 0.f, 1.f);
    return t*t*(3.f-2.f*t);
}
inline float step(float edge, float x) { return (x >= edge) ? 1.f : 0.f; }

// ===== EVOLVED_CODE_MARKER =====

// ── Grid sampling ────────────────────────────────────────────────────────────
#define GRID 32
static float H_GRID[GRID][GRID];

static bool sample_grid() {
    for (int iy = 0; iy < GRID; iy++) {
        for (int ix = 0; ix < GRID; ix++) {
            float px = -1.f + 2.f * (float)ix / (float)(GRID - 1);
            float py = -1.f + 2.f * (float)iy / (float)(GRID - 1);
            float v = terrain(vec2(px, py));
            if (!isfinite(v) || v < -1e6f || v > 1e6f) return false;
            H_GRID[iy][ix] = v;
        }
    }
    return true;
}

// Structure function at integer lag L (mean squared difference, both axes)
static double struct_func(int L) {
    double sum = 0.0; long cnt = 0;
    for (int iy = 0; iy < GRID; iy++)
        for (int ix = 0; ix + L < GRID; ix++) {
            double d = H_GRID[iy][ix+L] - H_GRID[iy][ix]; sum += d*d; cnt++;
        }
    for (int ix = 0; ix < GRID; ix++)
        for (int iy = 0; iy + L < GRID; iy++) {
            double d = H_GRID[iy+L][ix] - H_GRID[iy][ix]; sum += d*d; cnt++;
        }
    return cnt > 0 ? sum / cnt : 0.0;
}

// Hurst exponent via log-log regression on structure function
static float estimate_hurst() {
    static const int LAGS[] = {1, 2, 3, 4, 6, 8};
    const int NLAG = 6;
    float lx[6], ly[6]; int n = 0;
    for (int i = 0; i < NLAG; i++) {
        double sf = struct_func(LAGS[i]);
        if (sf > 1e-14) {
            lx[n] = logf((float)LAGS[i]);
            ly[n] = logf((float)sf);
            n++;
        }
    }
    if (n < 3) return -1.f;
    float sx=0,sy=0,sxy=0,sxx=0;
    for (int i=0;i<n;i++){sx+=lx[i];sy+=ly[i];sxy+=lx[i]*ly[i];sxx+=lx[i]*lx[i];}
    float D = (float)n*sxx - sx*sx;
    if (fabsf(D) < 1e-10f) return -1.f;
    return ((float)n*sxy - sx*sy) / D / 2.f;  // SF ~ r^(2H)
}

// Statistics over sampled grid
static void grid_stats(float* out_mean, float* out_std, float* out_range,
                       float* out_mean_slope) {
    double s=0, s2=0; float hmin=FLT_MAX, hmax=-FLT_MAX;
    int N = GRID*GRID;
    for (int i=0;i<GRID;i++) for (int j=0;j<GRID;j++){
        float v = H_GRID[i][j];
        s += v; s2 += (double)v*v;
        if (v < hmin) hmin = v;
        if (v > hmax) hmax = v;
    }
    *out_mean = (float)(s/N);
    float var = (float)(s2/N - (s/N)*(s/N));
    *out_std  = sqrtf(fmaxf(0.f, var));
    *out_range = hmax - hmin;
    // Mean gradient magnitude (central differences)
    double sg = 0.0; int cg = 0;
    for (int iy=1;iy<GRID-1;iy++) for (int ix=1;ix<GRID-1;ix++){
        float dx = (H_GRID[iy][ix+1] - H_GRID[iy][ix-1]) * 0.5f;
        float dy = (H_GRID[iy+1][ix] - H_GRID[iy-1][ix]) * 0.5f;
        sg += sqrtf(dx*dx + dy*dy); cg++;
    }
    *out_mean_slope = cg > 0 ? (float)(sg/cg) : 0.f;
}

int main() {
    string inp((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());

    // Parse JSON fields via simple search
    auto get_float = [&](const char* key, float def) -> float {
        string k = string("\"") + key + "\"";
        size_t p = inp.find(k);
        if (p == string::npos) return def;
        p += k.size();
        while (p < inp.size() && (inp[p]==' '||inp[p]==':')) p++;
        char* end; float v = strtof(inp.c_str()+p, &end);
        return (end != inp.c_str()+p) ? v : def;
    };
    auto get_str = [&](const char* key) -> string {
        string k = string("\"") + key + "\"";
        size_t p = inp.find(k);
        if (p == string::npos) return "";
        p += k.size();
        while (p < inp.size() && (inp[p]==' '||inp[p]==':')) p++;
        if (p < inp.size() && inp[p]=='"') {
            p++; string r;
            while (p < inp.size() && inp[p]!='"') r += inp[p++];
            return r;
        }
        return "";
    };

    string name          = get_str("name");
    float target_hurst   = get_float("target_hurst",      0.70f);
    float target_ns      = get_float("target_norm_slope", 1.50f);

    // Sample terrain and check for NaN/Inf
    if (!sample_grid()) {
        printf("{\"fitness\":0.001,\"H_est\":-1.0,\"valid\":false,\"name\":\"%s\"}\n",
               name.c_str());
        return 0;
    }

    float H_est = estimate_hurst();
    float mean_h, std_h, height_range, mean_slope;
    grid_stats(&mean_h, &std_h, &height_range, &mean_slope);

    float norm_slope = (std_h > 1e-6f) ? mean_slope / std_h : 0.f;

    // Hurst score: Gaussian around target H (width σ=0.35 → FWHM≈0.35)
    float hurst_score = (H_est > 0.0f && H_est < 2.0f) ?
        expf(-8.f * (H_est - target_hurst) * (H_est - target_hurst)) : 0.001f;

    // Normalised-slope score: relative Gaussian
    float ns_err = norm_slope - target_ns;
    float slope_score = (norm_slope > 0.f) ?
        expf(-5.f * ns_err*ns_err / (target_ns*target_ns + 0.01f)) : 0.001f;

    // Height-range guard: penalise near-constant functions
    float range_score = fminf(1.f, fmaxf(0.f, height_range / 0.2f));

    // Valley coherence score: real geological terrain has spatially correlated
    // slope-magnitude fields — valley floors are persistently flat, ridge crests
    // persistently steep, transitions smooth. Pure noise has uncorrelated gradients.
    // Measure: Pearson correlation of |∇h(i,j)| vs |∇h(i+1,j)| over all row pairs.
    float valley_score = 0.0f;
    {
        const int N = GRID;
        double sx=0,sy=0,sxy=0,sxx=0,syy=0; int cnt=0;
        for(int i=1;i<N-1;i++) for(int j=1;j<N-1;j++){
            // gradient magnitude at (i,j)
            float gx0 = 0.5f*(H_GRID[i][j+1]-H_GRID[i][j-1]);
            float gy0 = 0.5f*(H_GRID[i+1][j]-H_GRID[i-1][j]);
            float mag0 = sqrtf(gx0*gx0+gy0*gy0);
            // gradient magnitude at (i+1,j)
            if(i+1 >= N-1) continue;
            float gx1 = 0.5f*(H_GRID[i+1][j+1]-H_GRID[i+1][j-1]);
            float gy1 = 0.5f*(H_GRID[i+2][j]-H_GRID[i][j]);
            float mag1 = sqrtf(gx1*gx1+gy1*gy1);
            sx+=mag0; sy+=mag1; sxy+=mag0*mag1;
            sxx+=mag0*mag0; syy+=mag1*mag1; cnt++;
        }
        if(cnt>1){
            double D=sqrt((cnt*sxx-sx*sx)*(cnt*syy-sy*sy));
            valley_score = (D>1e-9) ? fmaxf(0.f,fminf(1.f,(float)((cnt*sxy-sx*sy)/D))) : 0.f;
        }
    }

    float fitness;
    if (height_range < 0.01f) {
        fitness = 0.001f;  // constant function → reject
    } else {
        // valley_score replaces 0.1·range_score: measures spatial coherence of
        // gradient field. Pure noise scores ~0; geological terrain scores ~0.6-0.9.
        fitness = 0.5f*hurst_score + 0.25f*slope_score
                + 0.15f*valley_score + 0.1f*range_score;
    }

    printf("{\"fitness\":%.6f,\"H_est\":%.4f,\"norm_slope\":%.4f,"
           "\"mean_slope\":%.4f,\"std_h\":%.4f,\"height_range\":%.4f,"
           "\"hurst_score\":%.4f,\"slope_score\":%.4f,\"valley_score\":%.4f,"
           "\"valid\":true,\"name\":\"%s\"}\n",
           fitness, H_est, norm_slope, mean_slope, std_h, height_range,
           hurst_score, slope_score, valley_score, name.c_str());
    return 0;
}
"""

TERRAIN_FUNCTION_SIGNATURE = '''\
float terrain(vec2 p);

// Evolved geological terrain height function.
// p: 2D position in world space (typically called at various scales)
// Returns: height value (any finite float — oracle normalises internally)
//
// AVAILABLE TYPES AND FUNCTIONS (GLSL-compatible):
//   vec2 type with .x / .y fields, +, -, * (scalar and component-wise)
//   dot(a,b), length(v), floor(v|f), fract(v|f), mix(a,b,t)
//   smoothstep(e0,e1,x), step(edge,x), mod(x,y), fabsf(x), sign(x)
//   clamp(x,lo,hi), sinf, cosf, sqrtf, powf, expf, logf
//
// WHAT THE ORACLE MEASURES:
//   Hurst exponent H (via log-log regression on the structure function)
//   and normalised slope (mean|∇h|/std(h)).  A function that returns
//   constant, linear, or purely periodic values scores near 0.
//   The oracle DOES NOT reward any specific hash constants — only the
//   *statistical shape* of the height distribution matters.
//
// TIPS FOR HIGH FITNESS:
//   • Use multiple frequency octaves (fBm-style) — correct H needs scale-invariance
//   • You may define private helper functions above terrain()
//   • Loops with fixed iteration count (≤ 10) are allowed
//   • No static mutable globals, no dynamic allocation
//   • Keep it under 50 lines total (helpers + terrain)
'''


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
# Seed programs — known-good terrain functions for the FunSearch prior
# ---------------------------------------------------------------------------

# Standard value-noise fBm with amp*=0.5 per octave.
# This gives H close to 0 by the 2D spectral theory
# but in practice measured H depends on hash + grid resolution.
# Serves as the baseline that evolution must beat.
_FBM_STANDARD_SEED = '''\
float _t_hash(float n) { return fract(sinf(n) * 43758.5453f); }
float _t_vnoise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x); f = f*f*(3.0f - 2.0f*f);
    float n00 = _t_hash(i.x + i.y * 57.0f);
    float n10 = _t_hash(i.x + 1.0f + i.y * 57.0f);
    float n01 = _t_hash(i.x + (i.y+1.0f) * 57.0f);
    float n11 = _t_hash(i.x + 1.0f + (i.y+1.0f) * 57.0f);
    return mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y);
}
float terrain(vec2 p) {
    float v = 0.0f, a = 0.5f;
    for (int i = 0; i < 6; i++) {
        v += a * _t_vnoise(p);
        p  = p * 2.0f;
        a *= 0.5f;
    }
    return v;
}'''

# Smoother fBm: slower amplitude decay (amp*=0.65) → higher H → smoother terrain.
_FBM_SMOOTH_SEED = '''\
float _ts_hash(float n) { return fract(sinf(n) * 43758.5453f); }
float _ts_vnoise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x); f = f*f*(3.0f - 2.0f*f);
    float n00 = _ts_hash(i.x + i.y * 57.0f);
    float n10 = _ts_hash(i.x + 1.0f + i.y * 57.0f);
    float n01 = _ts_hash(i.x + (i.y+1.0f) * 57.0f);
    float n11 = _ts_hash(i.x + 1.0f + (i.y+1.0f) * 57.0f);
    return mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y);
}
float terrain(vec2 p) {
    float v = 0.0f, a = 0.5f;
    for (int i = 0; i < 8; i++) {
        v += a * _ts_vnoise(p);
        p  = p * 2.0f;
        a *= 0.65f;   // slower decay → smoother, higher H
    }
    return v;
}'''

# Ridged multifractal: invert noise → sharp ridges, lower H.
_RIDGE_FBM_SEED = '''\
float _tr_hash(float n) { return fract(sinf(n) * 43758.5453f); }
float _tr_vnoise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x); f = f*f*(3.0f - 2.0f*f);
    float n00 = _tr_hash(i.x + i.y * 57.0f);
    float n10 = _tr_hash(i.x + 1.0f + i.y * 57.0f);
    float n01 = _tr_hash(i.x + (i.y+1.0f) * 57.0f);
    float n11 = _tr_hash(i.x + 1.0f + (i.y+1.0f) * 57.0f);
    return mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y);
}
float terrain(vec2 p) {
    float v = 0.0f, a = 0.5f, w = 1.0f;
    for (int i = 0; i < 7; i++) {
        float n = 1.0f - fabsf(2.0f * _tr_vnoise(p) - 1.0f);
        v += a * n * w;
        w  = clamp(n, 0.0f, 1.0f);
        p  = p * 2.17f;
        a *= 0.5f;
    }
    return v;
}'''

# Seed registry: instance name → list of (label, code) pairs
SEED_TERRAIN_PROGRAMS: dict[str, list[tuple[str, str]]] = {
    "generic": [
        ("fbm_standard", _FBM_STANDARD_SEED),
        ("fbm_smooth",   _FBM_SMOOTH_SEED),
        ("ridge_fbm",    _RIDGE_FBM_SEED),
    ],
}
# All instances share the generic seeds (oracle measures statistics, not hash-constants)
for _name in _TERRAIN_INSTANCE_CONFIGS:
    SEED_TERRAIN_PROGRAMS[_name] = SEED_TERRAIN_PROGRAMS["generic"]


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    return SEED_TERRAIN_PROGRAMS.get(instance_name, SEED_TERRAIN_PROGRAMS["generic"])


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


# ---------------------------------------------------------------------------
# C++ source construction
# ---------------------------------------------------------------------------

def build_candidate_source(code: str, instance: TerrainInstance) -> tuple[str, str]:
    """Return (cpp_source, json_stdin) for one evaluation."""
    cpp = _TERRAIN_EVALUATOR_CPP.replace("// ===== EVOLVED_CODE_MARKER =====", code, 1)
    data = {
        "name": instance.name,
        "target_hurst": instance.target_hurst,
        "target_norm_slope": instance.target_norm_slope,
    }
    return cpp, json.dumps(data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Single-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_on_instance(
    code: str,
    instance: TerrainInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 30.0,
) -> Optional[float]:
    """Compile + run evolved terrain function; return fitness ∈ (0,1] or None."""
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

    if not out.get("valid", False):
        return None

    fitness = float(out.get("fitness", 0.0))
    return fitness if math.isfinite(fitness) and fitness > 0 else None


# ---------------------------------------------------------------------------
# Terrain FunSearch kernel
# ---------------------------------------------------------------------------

@register_kernel("terrain")
class TerrainKernel(FunSearchKernel):
    """FunSearch kernel that evolves geological height functions terrain(vec2 p).

    Oracle: Hurst-exponent correctness + normalised slope — measures whether
    the evolved function has the statistical signature of real terrain at a
    specified geological class (rolling hills, mountain peaks, etc.).

    Fitness = 0.6·hurst_score + 0.3·slope_score + 0.1·range_score.
    """

    kernel_name = "terrain"
    BUS_CHANNEL_PREFIX = "terrain"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        logger.info("Terrain kernel: %d instances, sandbox=%s",
                    len(self.problem_instances), self.executor.sandbox_type)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def load_instances(self) -> list[TerrainInstance]:
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info("Terrain instance %r: H_target=%.2f, ns_target=%.2f",
                        name, inst.target_hurst, inst.target_norm_slope)
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
        instance = self.problem_instances[0] if self.problem_instances else None
        inst_desc = (f"{instance.name} (H_target={instance.target_hurst:.2f}, "
                     f"ns_target={instance.target_norm_slope:.2f}  — {instance.description})"
                     if instance else "unknown")

        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )

        hint_block = ""
        if hint:
            hint_block = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        return (
            f"You are a terrain generation expert and procedural noise researcher.\n"
            f"Evolve a better geological height function terrain(vec2 p).\n\n"
            f"{TERRAIN_FUNCTION_SIGNATURE}\n\n"
            f"Target: {inst_desc}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            f"Your goal: maximise Fitness = 0.6·hurst_score + 0.3·slope_score + 0.1·range_score\n"
            f"  hurst_score = exp(−8·(H_est − {instance.target_hurst if instance else 0.7:.2f})²)\n"
            f"  slope_score = exp(−5·(norm_slope − {instance.target_norm_slope if instance else 1.5:.2f})² / …)\n"
            f"  H_est is from structure-function log-log regression on a 32×32 sample grid.\n\n"
            f"Rules:\n"
            f"- Return ONLY the terrain() function (+ any helpers) in a single ```cpp code block\n"
            f"- Signature: float terrain(vec2 p)\n"
            f"- You MAY define private helper functions above terrain()\n"
            f"- Fixed-iteration loops (≤ 10 iters) are OK — needed for fBm octaves\n"
            f"- No static mutable globals, no dynamic allocation, no infinite loops\n"
            f"- All finite-float inputs must produce finite-float output\n"
            f"- Under 50 lines total\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+|glsl)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "terrain" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        seeds = get_seed_programs(
            self.config.instances[0] if self.config.instances else "rolling_hills"
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
    # Bus publishing — terrain.kernel.* channels
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

        Source is hardcoded to '/autobench/terrain_kernel' to match the schema
        const.
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/terrain_kernel",
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
        """Emit terrain.kernel.started.v1 when the run begins."""
        from ..kernels import _git_commit_short
        self._publish("terrain.kernel.started.v1", {
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
        """Emit terrain.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        self._publish("terrain.kernel.completed.v1", {
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
                "terrain_code": best.code if best else "",
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
        out_path = out_dir / f"terrain_results_gen{self.generation:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "terrain",
            "run_id": self.run_id,
            "generation": self.generation,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {
                    "id": p.id,
                    "fitness": p.fitness,
                    "worst_fitness": p.worst_fitness,
                    "island": p.island,
                    "generation": p.generation,
                    "terrain_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Terrain results → %s (best=%.4f)", out_path,
                    top[0].fitness if top else 0)

        # Publish best-of-generation to shader studio vault (fire-and-forget)
        if top:
            try:
                from autobench.kernels.bridge import NervousKernelBridge
                bridge = NervousKernelBridge()
                instance = self.config.instances[0] if self.config.instances else "terrain"
                bridge.publish_to_shader_vault(
                    top[0].code,
                    biome=instance,
                    fitness=top[0].fitness,
                    generation=self.generation,
                )
            except Exception as _e:
                pass  # vault publish is best-effort, never blocks evolution

        return out_path
