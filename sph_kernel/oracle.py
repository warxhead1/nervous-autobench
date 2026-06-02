"""SPH seed programs — known-good baselines for the FunSearch prior.

Moved verbatim from ``sph_kernel/__init__.py`` as part of a behavior-preserving
file split.
"""
from __future__ import annotations


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
