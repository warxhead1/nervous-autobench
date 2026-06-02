"""NoiseInstance, NoiseKernelConfig, reference shaders, seeds, personas, and
shader-injection helpers for the noise kernel.

Moved verbatim from noise_kernel/__init__.py — no logic changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..kernels import KernelConfig

# ---------------------------------------------------------------------------
# NoiseKernelConfig — alias for KernelConfig with noise-friendly defaults
# ---------------------------------------------------------------------------

# Expose as NoiseKernelConfig so callers can import it by name.
# We subclass rather than alias so we can add noise-specific docstring/defaults.

@dataclass
class NoiseKernelConfig(KernelConfig):
    """KernelConfig pre-populated with sensible defaults for the noise kernel.

    Extra behaviour beyond KernelConfig:
        - Default instances are the three canonical noise benchmarks.
        - GLSL candidates run through the GPU driver; allow_unsandboxed
          is accepted for API compatibility but is not a safety gate
          (the driver provides isolation).
    """
    # Override default instances and plateau stop for noise
    def __post_init__(self) -> None:
        if not self.instances:
            # Default to single instance: the three reference types are pairwise
            # SSIM ~0.31 and RAPS cosine ~0.999 — they are mutually incompatible
            # as targets for a single function. Multi-instance noise runs set an
            # arithmetic ceiling of (1.0+0.31+0.30)/3 ≈ 0.536 that no evolution
            # can break. Single instance removes the ceiling entirely.
            # Use --instances value_noise_3d,perlin_like,fbm_2octave to override.
            self.instances = ["value_noise_3d"]
        if self.plateau_generations is None:
            self.plateau_generations = 15


# ---------------------------------------------------------------------------
# Benchmark instance descriptor
# ---------------------------------------------------------------------------

@dataclass
class NoiseInstance:
    """One noise benchmark instance.

    Attributes:
        name:              Identifier, e.g. 'value_noise_3d'.
        description:       Human-readable description.
        reference_shader:  Complete ShaderToy-body GLSL that renders the
                           reference (no #version, no uniform declarations —
                           those come from wrap_fragment_shader).  Must define
                           `float noise(vec3 p)` and `void mainImage(...)`.
        reference_png:     Path to the pre-rendered reference PNG (may not
                           exist yet; generated at kernel init via load_instances).
        reference_raps:    Pre-computed log-RAPS of the reference render.
                           None if the reference PNG hasn't been rendered yet.
        viewport:          (width, height) for render + SSIM.
    """
    name: str
    description: str
    reference_shader: str
    reference_png: Path
    reference_raps: np.ndarray | None = field(default=None, compare=False)
    viewport: tuple[int, int] = (256, 256)


# ---------------------------------------------------------------------------
# Reference GLSL shaders (ShaderToy-body format: no #version, no uniforms)
# wrap_fragment_shader adds: #version 330 core, out vec4 fragColor,
#                             uniform vec2 iResolution, uniform float iTime,
#                             void main() { mainImage(fragColor, gl_FragCoord.xy); }
# ---------------------------------------------------------------------------

# Probe template — noise(p) is injected before mainImage.
# The caller injects ONLY the noise() function definition here; mainImage is appended.
_PROBE_MAIN_IMAGE = """\
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;
    vec3 p = vec3(uv * 8.0, iTime * 0.1);
    float n = noise(p);
    n = clamp(n * 0.5 + 0.5, 0.0, 1.0);
    fragColor = vec4(n, n, n, 1.0);
}"""


# --- Reference 1: value_noise_3d ---
# Trilinear interpolation of a hash grid with quintic smoothstep.
# Range approximately [-1, 1].  This is the "canonical correct" version that
# candidates must match or beat.
_REF_VALUE_NOISE = """\
// Reference: trilinear hash-grid value noise with quintic fade
float hash1(float n) { return fract(sin(n) * 43758.5453123); }

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    // Quintic smoothstep (5th-order) for smoother gradients than Hermite
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = hash1(i.x + i.y * 57.0 + i.z * 113.0);
    float n100 = hash1(i.x + 1.0 + i.y * 57.0 + i.z * 113.0);
    float n010 = hash1(i.x + (i.y + 1.0) * 57.0 + i.z * 113.0);
    float n110 = hash1(i.x + 1.0 + (i.y + 1.0) * 57.0 + i.z * 113.0);
    float n001 = hash1(i.x + i.y * 57.0 + (i.z + 1.0) * 113.0);
    float n101 = hash1(i.x + 1.0 + i.y * 57.0 + (i.z + 1.0) * 113.0);
    float n011 = hash1(i.x + (i.y + 1.0) * 57.0 + (i.z + 1.0) * 113.0);
    float n111 = hash1(i.x + 1.0 + (i.y + 1.0) * 57.0 + (i.z + 1.0) * 113.0);
    float x00 = mix(n000, n100, u.x);
    float x10 = mix(n010, n110, u.x);
    float x01 = mix(n001, n101, u.x);
    float x11 = mix(n011, n111, u.x);
    float y0  = mix(x00, x10, u.y);
    float y1  = mix(x01, x11, u.y);
    return mix(y0, y1, u.z) * 2.0 - 1.0;
}
"""

# --- Reference 2: perlin_like ---
# Gradient noise with 5th-order fade and hash-derived gradients.
# Range approximately [-1, 1].
_REF_PERLIN_LIKE = """\
// Reference: gradient noise with 5th-order fade (Perlin-style)
vec3 hashGrad(vec3 p) {
    float h = dot(p, vec3(127.1, 311.7, 74.7));
    return normalize(vec3(
        fract(sin(h) * 43758.5453),
        fract(sin(h * 1.371) * 43758.5453),
        fract(sin(h * 2.617) * 43758.5453)
    ) * 2.0 - 1.0);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float v000 = dot(hashGrad(i + vec3(0,0,0)), f - vec3(0,0,0));
    float v100 = dot(hashGrad(i + vec3(1,0,0)), f - vec3(1,0,0));
    float v010 = dot(hashGrad(i + vec3(0,1,0)), f - vec3(0,1,0));
    float v110 = dot(hashGrad(i + vec3(1,1,0)), f - vec3(1,1,0));
    float v001 = dot(hashGrad(i + vec3(0,0,1)), f - vec3(0,0,1));
    float v101 = dot(hashGrad(i + vec3(1,0,1)), f - vec3(1,0,1));
    float v011 = dot(hashGrad(i + vec3(0,1,1)), f - vec3(0,1,1));
    float v111 = dot(hashGrad(i + vec3(1,1,1)), f - vec3(1,1,1));
    float x00 = mix(v000, v100, u.x);
    float x10 = mix(v010, v110, u.x);
    float x01 = mix(v001, v101, u.x);
    float x11 = mix(v011, v111, u.x);
    float y0  = mix(x00, x10, u.y);
    float y1  = mix(x01, x11, u.y);
    return mix(y0, y1, u.z) * 1.4;
}
"""

# --- Reference 3: fbm_2octave ---
# 2-octave fractional Brownian motion: characteristic 1/f^2 power spectrum.
_REF_FBM_2OCTAVE = """\
// Reference: 2-octave fBm built on value noise (H=0.5, lacunarity=2.0)
float _hashFbm(vec3 p) {
    float n = dot(p, vec3(127.1, 311.7, 74.7));
    return fract(sin(n) * 43758.5453);
}

float _valueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float v000 = _hashFbm(i + vec3(0,0,0));
    float v100 = _hashFbm(i + vec3(1,0,0));
    float v010 = _hashFbm(i + vec3(0,1,0));
    float v110 = _hashFbm(i + vec3(1,1,0));
    float v001 = _hashFbm(i + vec3(0,0,1));
    float v101 = _hashFbm(i + vec3(1,0,1));
    float v011 = _hashFbm(i + vec3(0,1,1));
    float v111 = _hashFbm(i + vec3(1,1,1));
    float x00 = mix(v000, v100, u.x);
    float x10 = mix(v010, v110, u.x);
    float x01 = mix(v001, v101, u.x);
    float x11 = mix(v011, v111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

float noise(vec3 p) {
    // fBm: octave 1 (amplitude 0.5, freq 1.0) + octave 2 (amplitude 0.25, freq 2.0)
    // H = 0.5 → amplitude halves each octave, freq doubles (lacunarity=2)
    float v = 0.5 * _valueNoise(p);
    v += 0.25 * _valueNoise(p * 2.0 + vec3(5.2, 1.3, 9.7));
    // normalize from [0, 0.75] → [-1, 1]
    return (v / 0.375) - 1.0;
}
"""

# Map instance name → reference GLSL body
_REFERENCE_SHADERS: dict[str, tuple[str, str]] = {
    "value_noise_3d": (
        _REF_VALUE_NOISE,
        "Trilinear hash-grid value noise with quintic smoothstep fade. Range ~[-1,1], smooth and isotropic.",
    ),
    "perlin_like": (
        _REF_PERLIN_LIKE,
        "Gradient noise (Perlin-style) with 5th-order fade and hash-derived gradients. Range ~[-1,1].",
    ),
    "fbm_2octave": (
        _REF_FBM_2OCTAVE,
        "2-octave fractional Brownian motion (H=0.5, lacunarity=2). Characteristic 1/f^2 power spectrum.",
    ),
}

# ---------------------------------------------------------------------------
# Seed programs (baselines the LLM must beat)
# ---------------------------------------------------------------------------

_SEED_1_VALUE = """\
// Seed 1: hash-based value noise (weak baseline)
float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n = i.x + i.y * 57.0 + 113.0 * i.z;
    return mix(mix(mix(fract(sin(n)*43758.5),fract(sin(n+1.0)*43758.5),f.x),
                   mix(fract(sin(n+57.0)*43758.5),fract(sin(n+58.0)*43758.5),f.x),f.y),
               mix(mix(fract(sin(n+113.0)*43758.5),fract(sin(n+114.0)*43758.5),f.x),
                   mix(fract(sin(n+170.0)*43758.5),fract(sin(n+171.0)*43758.5),f.x),f.y),f.z);
}"""

_SEED_2_SIMPLEX = """\
// Seed 2: simplex-inspired (medium baseline)
float noise(vec3 p) {
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);
    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = fract(sin(vec4(b.x,b.y,b.x,b.y) * 127.1 + b.z * 311.7) * 43758.5453);
    vec4 k2 = fract(sin(vec4(b.x,b.y,b.x,b.y) * 127.1 + (b.z+1.0) * 311.7) * 43758.5453);
    return mix(mix(mix(k1.x,k1.y,d.x),mix(k1.z,k1.w,d.x),d.y),
               mix(mix(k2.x,k2.y,d.x),mix(k2.z,k2.w,d.x),d.y), d.z) * 2.0 - 1.0;
}"""

_SEEDS = [
    ("value_noise_baseline", _SEED_1_VALUE),
    ("simplex_inspired_baseline", _SEED_2_SIMPLEX),
]

# ---------------------------------------------------------------------------
# Island personas (5 personalities)
# ---------------------------------------------------------------------------

_ISLAND_PERSONAS = [
    (
        "spectral engineer",
        "Focus on frequency content, minimize visible repetition patterns. "
        "Aim for a flat power spectrum — every frequency should have equal energy.",
    ),
    (
        "hash designer",
        "Design novel hash functions with period > 256 in all axes. "
        "Avoid visible grid artifacts from low-period hashes.",
    ),
    (
        "gradient artist",
        "Use gradient-based interpolation with smooth, continuous derivatives. "
        "The result should be visually smooth with no piecewise-linear banding.",
    ),
    (
        "domain warper",
        "Distort the input space before evaluating the base noise. "
        "Domain warping can break symmetry and create organic-looking patterns.",
    ),
    (
        "fbm composer",
        "Layer multiple octaves at different frequencies (lacunarity=2.0, "
        "H=0.5 or H=0.8). The composite should match the reference's 1/f spectrum.",
    ),
]

# ---------------------------------------------------------------------------
# Shader injection helpers
# ---------------------------------------------------------------------------

def build_probe_shader(noise_glsl: str) -> str:
    """Inject a `float noise(vec3 p)` implementation into the probe shader body.

    Returns the ShaderToy-body string (NO #version directive, NO uniform
    declarations — those are prepended by ShaderExecutor's wrap_fragment_shader).

    The returned string contains:
        1. The injected noise() function
        2. void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }

    ShaderExecutor wraps this in:
        #version 330 core
        out vec4 fragColor;
        uniform vec2 iResolution;
        uniform float iTime;
        <returned string>
        void main() { mainImage(fragColor, gl_FragCoord.xy); }
    """
    return noise_glsl.strip() + "\n\n" + _PROBE_MAIN_IMAGE


def build_reference_shader(instance_name: str) -> str:
    """Return the probe shader body for the reference noise function."""
    ref_glsl, _ = _REFERENCE_SHADERS[instance_name]
    return build_probe_shader(ref_glsl)
