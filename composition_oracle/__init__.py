"""Composition oracle — proof-of-concept for multi-kernel physics integration.

This oracle tests whether the individually-evolved kernels are COMPOSABLE:
  - terrain(p)          → height field defines the riverbed geometry
  - W(r,h)              → SPH kernel places water where terrain is low
  - reaction(phi,T)     → Allen-Cahn freezes water when temperature drops

The composition test runs on a 32×32 grid (cheap, proof-of-concept):

  Step 1  Evaluate terrain height H(x,y) on 32×32 grid
  Step 2  Place "water" at cells where H < water_level (valley cells)
  Step 3  Assign initial phi=0.02 (liquid) to water cells, phi=0 to dry land
  Step 4  Create a Gaussian cold spot at grid center (spell T field)
  Step 5  Run Allen-Cahn PDE for n_steps using reaction(phi,T)
  Step 6  Score:
            valley_ice   = fraction of cold+wet cells that froze (phi > 0.5)
            dry_intact   = fraction of cold+dry cells that stayed dry (phi < 0.1)
            hot_water    = fraction of hot+wet cells that stayed liquid (phi < 0.5)

The composition oracle does NOT evolve new functions — it evaluates whether
a (terrain, reaction) pair produces physically coherent scene-level behavior.

## Usage

    from autobench.composition_oracle import evaluate_composition

    score = evaluate_composition(
        terrain_code="float terrain(vec2 p) { ... }",
        reaction_code="float reaction(float phi, float temp) { ... }",
        executor=executor,
    )
    # score ∈ [0,1]: 1.0 = ice formed only in cold valleys, not on warm dry land

## Connection to TEngine

This is the "proof of the rivers actually freeze" test before committing
to TEngine integration. When composition_score > 0.80 for the best evolved
(terrain, reaction) pair, the kernels are ready to ship to shadergen.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Optional

from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C++ composition evaluator
# ---------------------------------------------------------------------------

_COMPOSITION_CPP = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

// ── GLSL compatibility layer (for terrain) ──────────────────────────────────
struct vec2 {
    float x, y;
    vec2() : x(0), y(0) {}
    vec2(float x, float y): x(x), y(y) {}
    vec2 operator+(const vec2& o) const { return {x+o.x, y+o.y}; }
    vec2 operator-(const vec2& o) const { return {x-o.x, y-o.y}; }
    vec2 operator*(float s) const { return {x*s, y*s}; }
    vec2 operator*(const vec2& o) const { return {x*o.x, y*o.y}; }
    vec2 operator/(float s) const { return {x/s, y/s}; }
    vec2& operator+=(const vec2& o){ x+=o.x; y+=o.y; return *this; }
    float& operator[](int i) { return i==0?x:y; }
    const float& operator[](int i) const { return i==0?x:y; }
};
static inline vec2 operator*(float s, const vec2& v) { return {s*v.x, s*v.y}; }
static inline vec2 operator+(float s, const vec2& v) { return {s+v.x, s+v.y}; }
static inline vec2 operator-(float s, const vec2& v) { return {s-v.x, s-v.y}; }
static inline float dot(vec2 a, vec2 b) { return a.x*b.x+a.y*b.y; }
static inline float length(vec2 v) { return sqrtf(v.x*v.x+v.y*v.y); }
static inline vec2 normalize(vec2 v) { float l=length(v); return l>1e-9f?v*(1.f/l):vec2(0,0); }
static inline vec2 abs(vec2 v) { return {fabsf(v.x),fabsf(v.y)}; }
static inline vec2 floor(vec2 v) { return {floorf(v.x),floorf(v.y)}; }
static inline vec2 fract(vec2 v) { return {v.x-floorf(v.x),v.y-floorf(v.y)}; }
static inline float fract(float x) { return x-floorf(x); }
static inline vec2 min(vec2 a,vec2 b){ return {fminf(a.x,b.x),fminf(a.y,b.y)}; }
static inline vec2 max(vec2 a,vec2 b){ return {fmaxf(a.x,b.x),fmaxf(a.y,b.y)}; }
static inline vec2 clamp(vec2 v,float a,float b){ return {fmaxf(a,fminf(b,v.x)),fmaxf(a,fminf(b,v.y))}; }
static inline float mix(float a,float b,float t) { return a+t*(b-a); }
static inline vec2 mix(vec2 a,vec2 b,float t) { return {a.x+t*(b.x-a.x),a.y+t*(b.y-a.y)}; }
static inline float clamp(float x,float a,float b) { return fmaxf(a,fminf(b,x)); }
static inline float smoothstep(float e0,float e1,float x) {
    float t=clamp((x-e0)/(e1-e0),0.f,1.f); return t*t*(3.f-2.f*t); }
static inline float sign(float x) { return (x>0.f)-(x<0.f); }
static inline float mod(float x,float y) { return fmodf(x,y); }
static inline float step(float e,float x) { return x>=e?1.f:0.f; }

// ── TERRAIN CODE ─────────────────────────────────────────────────────────────
{TERRAIN_CODE}

// ── REACTION CODE ────────────────────────────────────────────────────────────
{REACTION_CODE}

int main() {
    const int N = 32;
    const float dt = 0.018f;
    const float D  = 10.0f;
    const int n_steps = 150;
    const float T_cold = 0.25f, T_hot = 0.75f;
    const float water_level = 0.0f;  // cells with terrain < water_level are "wet"

    // Step 1: terrain height field
    float H[N][N];
    float hmin = 1e9f, hmax = -1e9f;
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        float px = -1.f + 2.f*(float)j/(N-1);
        float py = -1.f + 2.f*(float)i/(N-1);
        float v = terrain(vec2(px,py));
        if(!isfinite(v)){printf("{\"valid\":false,\"reason\":\"terrain_nan\"}\n");return 0;}
        H[i][j]=v; hmin=fminf(hmin,v); hmax=fmaxf(hmax,v);
    }
    // Normalise height to [0,1] so water_level is a fraction
    float hrange = hmax-hmin;
    if(hrange < 0.01f){printf("{\"valid\":false,\"reason\":\"terrain_flat\"}\n");return 0;}
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) H[i][j]=(H[i][j]-hmin)/hrange;

    // Step 2: water mask (wet = low terrain), phi initialisation
    float phi[N][N], T_field[N][N];
    bool  wet_mask[N][N];
    float cx=(N-1)*0.5f, cy=(N-1)*0.5f;
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        bool wet = (H[i][j] < 0.4f);  // lower 40% of terrain is wet
        wet_mask[i][j] = wet;
        phi[i][j] = wet ? 0.05f : 0.0f;
        float r = sqrtf((i-cx)*(i-cx)+(j-cy)*(j-cy));
        T_field[i][j] = (r < (float)N*0.3f) ? T_cold : T_hot;
    }
    // Nucleation seed at centre
    for(int di=-2;di<=2;di++) for(int dj=-2;dj<=2;dj++){
        int ii=(int)cx+di, jj=(int)cy+dj;
        if(ii>=0&&ii<N&&jj>=0&&jj<N&&phi[ii][jj]>0.01f) phi[ii][jj]=0.65f;
    }

    // Step 3: Allen-Cahn PDE — only evolve wet cells (dry land is a fixed boundary)
    float phi_new[N][N];
    for(int s=0;s<n_steps;s++){
        for(int i=0;i<N;i++) for(int j=0;j<N;j++){
            if(!wet_mask[i][j]){ phi_new[i][j]=0.0f; continue; }
            float p=phi[i][j];
            // Neumann BC at wet/dry boundaries: treat dry neighbour as phi=0 (no flux into dry land)
            float pL=(i>0   && wet_mask[i-1][j])?phi[i-1][j]:p;
            float pR=(i<N-1 && wet_mask[i+1][j])?phi[i+1][j]:p;
            float pD=(j>0   && wet_mask[i][j-1])?phi[i][j-1]:p;
            float pU=(j<N-1 && wet_mask[i][j+1])?phi[i][j+1]:p;
            float lap=pL+pR+pD+pU-4.f*p;
            float r=reaction(p,T_field[i][j]);
            if(!isfinite(r)||fabsf(r)>50.f){printf("{\"valid\":false,\"reason\":\"reaction_diverged\"}\n");return 0;}
            phi_new[i][j]=p+dt*(D*lap+r);
            if(!isfinite(phi_new[i][j])){printf("{\"valid\":false,\"reason\":\"pde_nan\"}\n");return 0;}
        }
        for(int i=0;i<N;i++) for(int j=0;j<N;j++) phi[i][j]=phi_new[i][j];
    }

    // Step 4: Score
    float T_mid=(T_cold+T_hot)*0.5f;
    int cold_wet_total=0, cold_wet_frozen=0;
    int cold_dry_total=0, cold_dry_intact=0;
    int hot_wet_total=0,  hot_wet_liquid=0;

    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        bool wet  = (H[i][j] < 0.4f);
        bool cold = (T_field[i][j] <= T_mid);
        float p   = phi[i][j];
        if(cold && wet) {
            cold_wet_total++;
            if(p > 0.5f) cold_wet_frozen++;
        }
        if(cold && !wet) {
            cold_dry_total++;
            if(p < 0.1f) cold_dry_intact++;  // dry land stays dry
        }
        if(!cold && wet) {
            hot_wet_total++;
            if(p < 0.5f) hot_wet_liquid++;   // warm water stays liquid
        }
    }

    float valley_ice  = cold_wet_total >0 ? (float)cold_wet_frozen /cold_wet_total  : 0.f;
    float dry_intact  = cold_dry_total >0 ? (float)cold_dry_intact /cold_dry_total  : 1.f;
    float hot_water   = hot_wet_total  >0 ? (float)hot_wet_liquid  /hot_wet_total   : 1.f;

    // Composition fitness: ice in cold valleys, dry land stays dry, warm water liquid
    float fitness = 0.5f*valley_ice + 0.3f*dry_intact + 0.2f*hot_water;

    printf("{\"fitness\":%.6f,\"valley_ice\":%.4f,\"dry_intact\":%.4f,"
           "\"hot_water\":%.4f,\"cold_wet\":%d,\"valid\":true}\n",
           fitness, valley_ice, dry_intact, hot_water, cold_wet_total);
    return 0;
}
"""


def evaluate_composition(
    terrain_code: str,
    reaction_code: str,
    executor: SandboxedExecutor,
) -> Optional[float]:
    """Evaluate (terrain, reaction) pair on the composition oracle.

    Returns fitness ∈ [0,1] or None on failure:
      0.5·valley_ice  — ice formed in cold valleys
      0.3·dry_intact  — cold dry land stayed non-frozen
      0.2·hot_water   — warm water stayed liquid
    """
    cpp = _COMPOSITION_CPP.replace("{TERRAIN_CODE}", terrain_code)
    cpp = cpp.replace("{REACTION_CODE}", reaction_code)

    stdout, verdict, _ = compile_and_run(cpp, language="cpp", stdin="", executor=executor)
    if verdict != Verdict.OK or not stdout:
        return None

    try:
        out = json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None

    if not out.get("valid", False):
        logger.debug("composition invalid: %s", out.get("reason", "?"))
        return None

    fitness = float(out.get("fitness", 0.0))
    return fitness if math.isfinite(fitness) and fitness > 0 else None


def evaluate_best_pair(
    terrain_results_path: str,
    phase_results_path: str,
    executor: SandboxedExecutor,
) -> dict:
    """Load best terrain + phase programs, run composition oracle, return scores."""
    import json
    from pathlib import Path

    def load_best(path: str, code_field: str) -> tuple[str, float]:
        with open(path) as f:
            d = json.load(f)
        top = d.get("top_programs", [])
        if not top:
            raise ValueError(f"No programs in {path}")
        best = top[0]
        return best.get(code_field, ""), float(best.get("fitness", 0))

    terrain_code, terrain_fit = load_best(terrain_results_path, "terrain_code")
    reaction_code, phase_fit  = load_best(phase_results_path,   "reaction_code")

    composition_fit = evaluate_composition(terrain_code, reaction_code, executor)

    return {
        "terrain_fitness":     terrain_fit,
        "phase_fitness":       phase_fit,
        "composition_fitness": composition_fit,
        "terrain_code":        terrain_code[:200] + "..." if len(terrain_code) > 200 else terrain_code,
    }
