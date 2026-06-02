"""Terrain oracle: LLM prompt signature + seed programs (FunSearch prior)."""
from __future__ import annotations

from .instance import _TERRAIN_INSTANCE_CONFIGS


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
