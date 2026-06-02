"""Spectral oracle + RAPS helpers for the noise kernel.

compute_spectral_fitness scores a rendered noise image against calibrated
class-membership targets; the _raps_* helpers provide a cosine-similarity
fallback. Moved verbatim from noise_kernel/__init__.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

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
