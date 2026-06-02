"""NoiseKernel — FunSearch kernel for GPU-evaluated GLSL noise functions.

Evolves `float noise(vec3 p)` GLSL functions rendered headless via the
ShaderExecutor (moderngl/EGL). Fitness is a spectral class-membership score
(RAPS-based) against calibrated targets for each benchmark instance, with an
SSIM fallback.

The implementation is split across submodules:
    instance  — NoiseInstance, NoiseKernelConfig, reference shaders/seeds/
                personas, build_probe_shader / build_reference_shader
    spectral  — compute_spectral_fitness + RAPS helpers
    oracle    — LLM-response parsing helpers
    loop      — the NoiseKernel class (registered as "noise")

This module re-exports the public surface and triggers @register_kernel("noise").
"""

from __future__ import annotations

from ..kernels import CandidateProgram
from .instance import (
    NoiseInstance,
    NoiseKernelConfig,
    build_probe_shader,
    build_reference_shader,
    _ISLAND_PERSONAS,
    _PROBE_MAIN_IMAGE,
    _REFERENCE_SHADERS,
    _REF_FBM_2OCTAVE,
    _REF_PERLIN_LIKE,
    _REF_VALUE_NOISE,
    _SEED_1_VALUE,
    _SEED_2_SIMPLEX,
    _SEEDS,
)
from .spectral import (
    NOISE_CLASS_PARAMS,
    compute_spectral_fitness,
    _radially_average,
    _raps_cosine,
    _raps_from_image,
)
from .oracle import _extract_noise_fn
from .loop import NoiseKernel

__all__ = [
    "NoiseKernel",
    "NoiseKernelConfig",
    "NoiseInstance",
    "CandidateProgram",
    "compute_spectral_fitness",
    "build_probe_shader",
    "build_reference_shader",
    "NOISE_CLASS_PARAMS",
]
