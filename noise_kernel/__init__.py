"""NoiseKernel — FunSearch kernel for GPU-evaluated GLSL noise functions.

Evolves `float noise(vec3 p)` GLSL functions rendered headless via the
existing ShaderExecutor (moderngl/EGL). Fitness is SSIM against a
pre-rendered reference PNG produced by a known-good analytical noise
function for each benchmark instance.

# Benchmark instances
  value_noise_3d — trilinear hash-grid value noise, smooth, isotropic, [-1,1]
  perlin_like    — gradient-noise with 5th-order fade, [-1,1], spectral peak at f=1
  fbm_2octave    — 2-octave fractional Brownian motion, 1/f² spectrum

# GPU isolation note
  GLSL candidate code is evaluated by compiling and running through the GPU
  driver via moderngl/EGL. The driver provides its own sandbox boundary; no
  OS-level process sandbox is applied here (unlike the CPU C++ kernels).
  Only run this kernel on trusted input or in an attended setting.

# Evaluation pipeline
  1. Inject `float noise(vec3 p)` body into the probe shader template
     (only the ShaderToy-body portion — no #version, no uniforms; those are
     added by ShaderExecutor's wrap_fragment_shader)
  2. Validate with glslangValidator via validate_glsl()
  3. Render at 256×256, iTime=0.0 via ShaderExecutor.run()
  4. Score SSIM against the instance's reference PNG
  5. fitness = max(0.0, ssim_score), or None on compile/render failure
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..kernel_base import FunSearchKernel, KernelConfig, CandidateProgram, Island

logger = logging.getLogger(__name__)

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
# Spectral oracle — class-membership fitness (replaces 3x-SSIM oracle)
# ---------------------------------------------------------------------------

# Empirically measured parameters for the three canonical noise benchmarks.
# Calibrated at 256x256 renders using the reference shaders above.
NOISE_CLASS_PARAMS: dict[str, dict] = {
    "value_noise_3d": {
        "beta": 3.03,
        "beta_sigma": 0.4,
        "harmonic_ratio": 0.12,
        "hr_sigma": 0.08,
        "min_entropy": 2.5,
        "weights": (0.5, 0.3, 0.2),  # (beta_score, harmonic_score, entropy_guard)
    },
    "perlin_like": {
        "beta": 3.18,
        "beta_sigma": 0.4,
        "harmonic_ratio": 0.10,
        "hr_sigma": 0.08,
        "min_entropy": 2.5,
        "weights": (0.5, 0.3, 0.2),
    },
    "fbm_2octave": {
        "beta": 3.08,
        "beta_sigma": 0.4,
        "harmonic_ratio": 0.11,
        "hr_sigma": 0.08,
        "min_entropy": 2.5,
        "weights": (0.5, 0.3, 0.2),
    },
}


def compute_spectral_fitness(
    img_path: "Path",
    instance_name: str,
    _stash_target: "object | None" = None,
) -> "float | None":
    """Compute spectral class-membership fitness for a rendered noise image.

    Measures three properties of the radially-averaged power spectrum (RAPS)
    of the grayscale image and scores how well they match the calibrated
    targets for the given instance.

    Args:
        img_path:      Path to a rendered 256x256 PNG.
        instance_name: One of 'value_noise_3d', 'perlin_like', 'fbm_2octave'.
                       If unknown, falls back to None (caller should use SSIM).
        _stash_target: Optional object on which to stash _last_beta and
                       _last_hr for extract_t_vector().

    Returns:
        Float in [0, 1] on success, None on failure (missing file, bad params,
        PIL/numpy error).
    """
    params = NOISE_CLASS_PARAMS.get(instance_name)
    if params is None:
        logger.debug(
            "compute_spectral_fitness: unknown instance '%s', returning None", instance_name
        )
        return None

    try:
        from PIL import Image as _PILImage  # type: ignore
    except ImportError as e:
        logger.debug("compute_spectral_fitness: PIL unavailable: %s", e)
        return None

    if not Path(img_path).exists():
        logger.debug("compute_spectral_fitness: image not found: %s", img_path)
        return None

    try:
        arr = np.array(_PILImage.open(str(img_path)).convert("L"), dtype=np.float64)
    except Exception as e:
        logger.debug("compute_spectral_fitness: failed to load image %s: %s", img_path, e)
        return None

    # --- Radially-averaged power spectrum ---
    fft_shifted = np.fft.fftshift(np.fft.fft2(arr))
    magnitude_sq = np.abs(fft_shifted) ** 2
    h, w = magnitude_sq.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.mgrid[:h, :w]
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
    n_bins = min(h // 2, w // 2)
    raps = np.array(
        [magnitude_sq[r == i].mean() if (r == i).any() else 0.0 for i in range(n_bins)],
        dtype=np.float64,
    )

    # --- 1. Spectral beta: negative slope of log-log RAPS ---
    freqs = np.arange(1, n_bins)  # skip DC bin (r=0)
    log_freqs = np.log(freqs.astype(np.float64))
    log_raps = np.log(raps[1:] + 1e-12)
    slope, _ = np.polyfit(log_freqs, log_raps, 1)
    spectral_beta = -slope  # positive for 1/f noise

    # --- 2. Harmonic ratio ---
    base_freq = max(4, n_bins // 8)
    harmonic_ratio = raps[base_freq * 2] / (raps[base_freq] + 1e-10)

    # --- 3. Entropy guard ---
    # Histogram entropy of the grayscale image pixels in [0,255]
    pixel_counts, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = pixel_counts.sum()
    if total > 0:
        probs = pixel_counts[pixel_counts > 0] / total
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    entropy_guard = 1.0 if entropy >= params["min_entropy"] else 0.0

    # --- Score components ---
    beta_score = float(np.exp(
        -0.5 * ((spectral_beta - params["beta"]) / params["beta_sigma"]) ** 2
    ))
    hr_score = float(np.exp(
        -0.5 * ((harmonic_ratio - params["harmonic_ratio"]) / params["hr_sigma"]) ** 2
    ))

    w0, w1, w2 = params["weights"]
    fitness = w0 * beta_score + w1 * hr_score + w2 * entropy_guard

    # Stash for extract_t_vector
    if _stash_target is not None:
        _stash_target._last_beta = spectral_beta  # type: ignore[attr-defined]
        _stash_target._last_hr = harmonic_ratio  # type: ignore[attr-defined]
        _stash_target._last_raps_fitness = float(fitness)  # type: ignore[attr-defined]

    logger.debug(
        "spectral_fitness[%s]: beta=%.3f hr=%.3f entropy=%.2f "
        "-> beta_score=%.3f hr_score=%.3f entropy_guard=%.1f fitness=%.4f",
        instance_name, spectral_beta, harmonic_ratio, entropy,
        beta_score, hr_score, entropy_guard, fitness,
    )
    return float(np.clip(fitness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# RAPS oracle helpers — kept for future use / cosine-similarity fallback
# ---------------------------------------------------------------------------

def _radially_average(spectrum: np.ndarray) -> np.ndarray:
    """Radially-averaged power spectrum (RAPS) of a 2D magnitude array."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.mgrid[:h, :w]
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
    n_bins = min(h // 2, w // 2)
    raps = np.array(
        [spectrum[r == i].mean() if (r == i).any() else 0.0 for i in range(n_bins)],
        dtype=np.float64,
    )
    return raps


def _raps_from_image(img_path: Path) -> np.ndarray:
    """Load image, compute log-RAPS of its grayscale FFT magnitude."""
    from PIL import Image as _Image
    arr = np.array(_Image.open(str(img_path)).convert("L"), dtype=np.float64)
    fft_shifted = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.abs(fft_shifted) ** 2
    raps = _radially_average(magnitude)
    return np.log1p(raps)  # log-RAPS: spectral slope discrimination


def _raps_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two RAPS vectors (both already log-scaled)."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


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


# ---------------------------------------------------------------------------
# NoiseKernel
# ---------------------------------------------------------------------------

class NoiseKernel(FunSearchKernel):
    """FunSearch kernel for GPU-evaluated GLSL noise functions.

    Evolves `float noise(vec3 p)` candidates rendered headless on the GPU via
    ShaderExecutor (moderngl/EGL). Fitness is SSIM against a pre-rendered
    reference PNG for each benchmark instance.

    GLSL runs through the GPU driver, which provides its own isolation. This
    kernel does not apply an OS-level process sandbox (unlike the CPU kernels).

    Subclasses FunSearchKernel and implements the five abstract methods.
    """

    BUS_CHANNEL_PREFIX = "noise"

    def __init__(self, config: KernelConfig):
        super().__init__(config)

        # Lazy-init executor; only imported when needed so the module is cheap to import.
        self._executor: Any = None  # ShaderExecutor, created on first use

        # Populate instances immediately (reference PNGs generated here).
        self.problem_instances = self.load_instances()
        logger.info(
            "Loaded %d noise instance(s): %s",
            len(self.problem_instances),
            [inst.name for inst in self.problem_instances],
        )

    # ------------------------------------------------------------------
    # ShaderExecutor accessor (lazy init, graceful on no GPU)
    # ------------------------------------------------------------------

    def _get_executor(self) -> Any:
        """Return a ShaderExecutor, creating it on first call."""
        if self._executor is not None:
            return self._executor
        try:
            from ..shader_executor import ShaderExecutor  # type: ignore
            self._executor = ShaderExecutor(viewport=(256, 256))
            return self._executor
        except ImportError as e:
            logger.warning("ShaderExecutor import failed (moderngl/numpy missing?): %s", e)
            return None

    # ------------------------------------------------------------------
    # FunSearchKernel abstract interface
    # ------------------------------------------------------------------

    def load_instances(self) -> list[NoiseInstance]:
        """Load benchmark instances, generating reference PNGs if absent."""
        output_dir = self.config.output_dir or Path(tempfile.mkdtemp(prefix="noise_kernel_"))

        instances: list[NoiseInstance] = []
        for name in self.config.instances:
            if name not in _REFERENCE_SHADERS:
                raise ValueError(
                    f"Unknown noise instance '{name}'. "
                    f"Available: {sorted(_REFERENCE_SHADERS)}"
                )
            ref_glsl, description = _REFERENCE_SHADERS[name]
            ref_png = output_dir / f"{name}_reference.png"

            if not ref_png.exists():
                self._render_reference(name, ref_glsl, ref_png)

            ref_raps = _raps_from_image(ref_png) if ref_png.exists() else None

            instances.append(NoiseInstance(
                name=name,
                description=description,
                reference_shader=build_reference_shader(name),
                reference_png=ref_png,
                reference_raps=ref_raps,
            ))

        return instances

    def _render_reference(self, name: str, ref_glsl: str, out_path: Path) -> None:
        """Render the reference PNG for one instance. Logs warning on failure."""
        executor = self._get_executor()
        if executor is None:
            logger.warning(
                "No GPU executor available; skipping reference render for '%s'", name
            )
            return

        shader_body = build_probe_shader(ref_glsl)
        try:
            success, log, render_ms = executor.render(
                shader_body, out_path, viewport=(256, 256), i_time=0.0
            )
            if success:
                logger.info(
                    "Rendered reference '%s' → %s (%.1fms)", name, out_path, render_ms
                )
            else:
                logger.warning(
                    "Reference render failed for '%s': %s", name, log[:400]
                )
        except NotImplementedError as e:
            logger.warning("GPU not available for reference render '%s': %s", name, e)
        except Exception as e:
            logger.warning("Unexpected error rendering reference '%s': %s", name, e)

    def evaluate_candidate(self, code: str, instance: NoiseInstance) -> float | None:
        """Inject noise() into the probe shader, render once at t=0, score via spectral oracle.

        Fitness = spectral class-membership score in [0, 1] (see compute_spectral_fitness).
        Falls back to max(0.0, ssim) if the spectral oracle returns None (missing Pillow,
        unrecognised instance name, etc.).

        Returns None (not 0.0) if:
          - The reference PNG does not exist (no GPU at init time)
          - The candidate fails to compile (CE) or crashes (RE)
          - The GPU render backend is unavailable

        GLSL evaluation trusts the GPU driver's own isolation boundary.
        """
        if not instance.reference_png.exists():
            logger.debug(
                "Skipping evaluation of '%s': reference PNG missing at %s",
                instance.name, instance.reference_png,
            )
            return None

        executor = self._get_executor()
        if executor is None:
            return None

        shader_body = build_probe_shader(code)

        # Evaluate at 3 Z-slices (iTime=0, 3, 7 → Z=0.0, 0.3, 0.7) and average spectral
        # scores. A 2D function that only varies in XY will score poorly at non-zero Z.
        # Single Z=0 evaluation misses 2/3 of the 3D noise structure.
        _Z_SLICES = [0.0, 3.0, 7.0]
        spectral_scores: list[float] = []
        ssim_score = 0.0

        with tempfile.TemporaryDirectory(prefix="noise_eval_") as _tmp:
            from ..shader_executor import Verdict  # type: ignore

            for i_time in _Z_SLICES:
                candidate_png = Path(_tmp) / f"candidate_t{int(i_time*10):02d}.png"
                try:
                    result = executor.run(
                        shader_body,
                        reference_path=instance.reference_png,
                        viewport=(256, 256),
                        i_time=i_time,
                        out_path=candidate_png,
                    )
                except TypeError:
                    try:
                        result = executor.run(
                            shader_body,
                            reference_path=instance.reference_png,
                            viewport=(256, 256),
                            i_time=i_time,
                        )
                        candidate_png = None  # type: ignore[assignment]
                    except Exception as e:
                        logger.warning("Unexpected error on '%s' t=%.1f: %s", instance.name, i_time, e)
                        return None
                except Exception as e:
                    logger.warning("Unexpected error on '%s' t=%.1f: %s", instance.name, i_time, e)
                    return None

                if result.verdict in (Verdict.CE, Verdict.RE):
                    logger.debug("Candidate CE/RE on '%s': %s", instance.name, result.error[:200])
                    return None

                ssim_score = max(ssim_score, max(0.0, result.ssim))

                if candidate_png is not None and Path(candidate_png).exists():
                    spectral = compute_spectral_fitness(
                        candidate_png, instance.name, _stash_target=self
                    )
                    if spectral is not None:
                        spectral_scores.append(spectral)

            if spectral_scores:
                self._last_ssim_fallback = ssim_score
                return sum(spectral_scores) / len(spectral_scores)

            # Spectral oracle unavailable — fall back to SSIM
            logger.debug(
                "Spectral oracle unavailable for '%s'; using SSIM fallback=%.4f",
                instance.name, ssim_score,
            )
            self._last_beta = 0.0
            self._last_hr = 0.0
            self._last_raps_fitness = 0.0
            self._last_ssim_fallback = ssim_score
            return ssim_score

    def evaluate_fitness(self, program: CandidateProgram) -> tuple[float, float, float]:
        """Evaluate program and snapshot spectral features per-program.

        Calls the base-class evaluation loop, then stashes the spectral values
        that compute_spectral_fitness wrote onto self onto the program object.
        This lets extract_t_vector() return correct per-program T-vectors rather
        than kernel-level _last_* attributes that reflect the last evaluated program.
        """
        result = super().evaluate_fitness(program)
        program._spectral_beta = getattr(self, "_last_beta", 0.0)
        program._harmonic_ratio = getattr(self, "_last_hr", 0.0)
        return result

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        """Build the LLM prompt for evolving a new noise() function."""
        persona_name, persona_hint = _ISLAND_PERSONAS[island.id % len(_ISLAND_PERSONAS)]

        exemplars = ""
        for i, p in enumerate(sorted(top_programs, key=lambda x: -x.fitness)[:3]):
            exemplars += f"\n// Exemplar {i+1} (SSIM={p.fitness:.4f}):\n{p.code}\n"

        instance_names = [inst.name for inst in self.problem_instances]
        instance_desc = ", ".join(instance_names)

        hint_section = ""
        if hint:
            hint_section = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        return f"""You are a GLSL shader expert acting as a "{persona_name}".
{persona_hint}
{hint_section}
Write a new GLSL `float noise(vec3 p)` function.

## Signature and contract
```glsl
float noise(vec3 p);   // returns a value in [-1, 1]
```
- Input: a 3D point `p`.
- Output: continuous noise in [-1, 1]. Output range matters — values outside this
  will be clamped in rendering, flattening contrast.
- No textures, no samplers — pure computation only.
- **You may define private helper functions before `noise()`** (e.g. `float hash(vec3 p)`,
  `float valueNoise(vec3 p)`, `float fbm(vec3 p)`). Include them all in the same code block.
  This is required to implement fBm, Worley, or any multi-octave structure.

## Evaluation — spectral oracle (NOT pixel SSIM)
Your function is rendered at 256×256 at **three different Z slices** (iTime=0, 3, 7).
Fitness = average spectral slope match across all three slices:
- Target spectral slope β (power law exponent of log-log RAPS)
- Target harmonic ratio (ratio of 1st and 2nd spectral harmonics)
- Entropy guard: output must not be near-constant or near-binary

**Low-scoring failure modes:**
- All-black or all-white output (constant or clipped → no spectral structure)
- Identical output at all Z values (2D function disguised as 3D → fails Z-slice test)
- Visible axis-aligned grid lines or hard-edged repetition (wrong spectral slope)
- NaN/Inf output

## Benchmark instances
{instance_desc}

## Available GLSL builtins (no textures, no external functions)
`sin`, `cos`, `fract`, `floor`, `ceil`, `round`, `abs`, `mod`,
`mix`, `smoothstep`, `step`, `clamp`, `dot`, `cross`, `length`,
`normalize`, `sqrt`, `pow`, `exp`, `log`, `min`, `max`, `sign`

## Island {island.id} population (generation {generation})
{exemplars if exemplars else "  (no exemplars yet — this is the first generation)"}

## Your task
Write a SINGLE ```glsl code block. Include helper functions first, then `float noise(vec3 p)` last.
Do NOT include `void mainImage`, `#version`, `uniform`, or any other declarations.
Keep the total under 60 lines.
Aim to match the target spectral slope β — smooth, 3D-varying, multi-scale noise wins.
The Z-slice test means a function that ignores `p.z` will fail. Use all three components.
"""

    def parse_response(self, response: str) -> str:
        """Extract a GLSL noise() function body from an LLM response.

        Tries markdown fences first, then bare function definition.
        Returns '' on failure — treated as 0.0 fitness candidate.
        """
        if not response or not response.strip():
            return ""
        text = response.strip()

        # 1. Markdown fenced code block (glsl, glsl|GLSL, or generic)
        fence = re.search(r"```[a-zA-Z0-9_]*[ \t]*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence and fence.group(1).strip():
            code = fence.group(1).strip()
            # If it contains mainImage (LLM over-generated), extract just noise()
            if "void mainImage" in code:
                noise_part = _extract_noise_fn(code)
                if noise_part:
                    return noise_part
            return code

        # 2. Bare float noise( signature
        sig = re.search(r"float\s+noise\s*\(\s*vec3", text)
        if sig:
            return text[sig.start():].strip()

        # 3. Last-resort strip
        return text.replace("```glsl", "").replace("```GLSL", "").replace("```", "").strip()

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        """Return the two baseline seed programs for a fresh island."""
        programs = []
        for variant_name, code in _SEEDS:
            programs.append(CandidateProgram(
                id=f"island{island_id}_gen{generation}_{variant_name}",
                code=code,
                island=island_id,
                generation=generation,
                source="baseline",
            ))
        return programs

    # ------------------------------------------------------------------
    # T-vector extraction — spectral features for ConsolidatedPrior
    # ------------------------------------------------------------------

    def extract_t_vector(self, program: CandidateProgram) -> dict:
        """Return spectral features for this program as a T-vector dict.

        Reads features stashed per-program by evaluate_fitness(), giving
        correct per-program values rather than kernel-level _last_* attributes.
        Falls back to 0.0 for seed programs evaluated before the spectral oracle ran.
        """
        return {
            "fitness": program.fitness,
            "spectral_beta": getattr(program, "_spectral_beta", 0.0),
            "harmonic_ratio": getattr(program, "_harmonic_ratio", 0.0),
        }

    # ------------------------------------------------------------------
    # Candidate event publishing override — adds spectral fields
    # ------------------------------------------------------------------

    def _publish_candidate(self, program: CandidateProgram) -> None:
        """Emit noise.candidate.evaluated.v1 with spectral oracle fields."""
        spectral_beta = getattr(self, "_last_beta", 0.0)
        harmonic_ratio = getattr(self, "_last_hr", 0.0)
        raps_fitness = getattr(self, "_last_raps_fitness", 0.0)
        ssim_fallback = getattr(self, "_last_ssim_fallback", 0.0)

        self._publish(
            "noise.candidate.evaluated.v1",
            {
                "run_id": self.run_id,
                "program_id": program.id,
                "island": program.island,
                "generation": program.generation,
                "fitness": program.fitness,
                "source": program.source,
                "spectral_beta": spectral_beta,
                "harmonic_ratio": harmonic_ratio,
                "raps_fitness": raps_fitness,
                "ssim_fallback": ssim_fallback,
            },
        )

    # ------------------------------------------------------------------
    # Bus publishing — noise.kernel.* channels
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
        """Write event to bus debug log and optionally nervous CLI.

        Source is hardcoded to '/autobench/noise_kernel' to match the
        schema const (the base class derives it from the class name
        dynamically, producing '/autobench/noisekernel' without the underscore).
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/noise_kernel",
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
        """Render the best noise function to PNG."""
        executor = self._get_executor()
        if executor is None:
            return False
        try:
            shader_body = build_probe_shader(best.code)
            result = executor.render_only(shader_body, out_path=str(out_path),
                                          viewport=(512, 512), i_time=0.0)
            return bool(result.frame_path) and Path(result.frame_path).exists()
        except Exception as e:
            logger.debug("_render_best_program failed: %s", e)
            return False

    def _artifact_render_type(self) -> str:
        return "noise_render"

    def _publish_started(self) -> None:
        """Emit noise.kernel.started.v1 when the run begins."""
        from ..kernel_base import _git_commit_short
        self._publish("noise.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": [inst.name for inst in self.problem_instances],
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
            "gpu_backend": "moderngl/egl",
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        """Emit noise.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        self._publish("noise.kernel.completed.v1", {
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
                "noise_glsl": best.code if best else "",
            } if best else None,
            "history": self.history,
        })

    def save_results(self, programs: list[CandidateProgram], path: Path | None = None) -> None:
        """Save results to JSON."""
        if path is None and self.config.output_dir:
            path = self.config.output_dir / f"noise_results_gen{self.generation}.json"
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
                "noise_glsl": best.code,
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
                    "noise_glsl": p.code,
                }
                for p in programs[:10]
            ],
        }
        path.write_text(json.dumps(output, indent=2))
        logger.info("Saved noise results to %s", path)


# ---------------------------------------------------------------------------
# Helper: extract noise() from a response that also includes mainImage
# ---------------------------------------------------------------------------

def _extract_noise_fn(glsl: str) -> str:
    """Extract the `float noise(...)` function from a larger GLSL snippet.

    Returns the extracted function (including its closing brace) or '' if not found.
    """
    m = re.search(r"(float\s+noise\s*\(\s*vec3[^{]*\{)", glsl)
    if not m:
        return ""
    start = m.start()
    depth = 0
    i = m.start(1) + len(m.group(1)) - 1  # position of the opening brace
    while i < len(glsl):
        if glsl[i] == "{":
            depth += 1
        elif glsl[i] == "}":
            depth -= 1
            if depth == 0:
                return glsl[start:i + 1].strip()
        i += 1
    return glsl[start:].strip()
