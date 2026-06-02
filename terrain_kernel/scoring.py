"""Terrain scoring: C++ structure-function oracle, source construction, evaluation."""
from __future__ import annotations

import json
import math
from typing import Optional

from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run
from .instance import TerrainInstance


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
