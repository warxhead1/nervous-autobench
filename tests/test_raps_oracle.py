"""Tests for the RAPS (spectral) oracle in noise_kernel.

Covers:
    * Reference PNG → fitness > 0.85 (asset-dependent, skipped if PNG absent)
    * All-grey synthetic image → fitness < 0.1 (constant output rejected)
    * Phase invariance: two same-beta fractal images → |fitness_a - fitness_b| < 0.15
    * Spectral beta extraction: correct noise → beta in [2.0, 4.5]
    * NoiseKernel.extract_t_vector() always includes 'spectral_beta' key

Integration test (no GPU required):
    * NoiseKernel.extract_t_vector() constructs + returns spectral_beta key
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REF_DIR = pathlib.Path(__file__).parents[2] / "benchmarks" / "curriculum"

def _latest_reference_png(instance_name: str) -> pathlib.Path | None:
    """Return the most recent *_reference.png for instance_name, or None."""
    candidates = sorted(
        _REF_DIR.glob(f"**/{instance_name}_reference.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _make_fractal_noise_png(beta: float, seed: int, tmp_dir: pathlib.Path, size: int = 256) -> pathlib.Path:
    """Generate a synthetic fractal noise PNG with the given spectral slope.

    Uses inverse-FFT with 1/f^(beta/2) magnitude envelope and random phase.
    Guarantees spatial variety (non-constant, high entropy) so the entropy
    guard in compute_spectral_fitness passes.
    """
    from PIL import Image  # type: ignore

    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, (size, size))
    fx, fy = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size))
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0  # avoid divide-by-zero at DC
    magnitude = r ** (-beta / 2.0)
    magnitude[0, 0] = 0.0  # zero DC component
    spec = magnitude * np.exp(1j * phase)
    noise = np.real(np.fft.ifft2(spec))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-12) * 255.0
    arr = noise.astype(np.uint8)
    path = tmp_dir / f"noise_beta{beta:.1f}_seed{seed}.png"
    Image.fromarray(arr).save(str(path))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_spectral_fitness_with_reference_png():
    """Reference PNG scored against its own class should return fitness > 0.85."""
    from autobench.noise_kernel import compute_spectral_fitness

    png = _latest_reference_png("value_noise_3d")
    if png is None:
        pytest.skip("No value_noise_3d_reference.png found (GPU not available at build time)")

    fitness = compute_spectral_fitness(png, "value_noise_3d")
    assert fitness is not None, "compute_spectral_fitness returned None for real reference PNG"
    assert fitness > 0.85, f"Expected fitness > 0.85 for reference PNG, got {fitness:.4f}"


def test_spectral_fitness_rejects_constant():
    """All-grey (constant) image should score < 0.1 — entropy guard kills it."""
    from PIL import Image  # type: ignore
    from autobench.noise_kernel import compute_spectral_fitness

    with tempfile.TemporaryDirectory(prefix="raps_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        grey = np.full((256, 256), 128, dtype=np.uint8)
        img_path = tmp_path / "grey.png"
        Image.fromarray(grey).save(str(img_path))

        fitness = compute_spectral_fitness(img_path, "value_noise_3d")
        assert fitness is not None
        assert fitness < 0.1, f"Expected fitness < 0.1 for constant grey image, got {fitness:.4f}"


def test_spectral_fitness_is_phase_invariant():
    """Two fractal images with the same beta at different seeds → |fitness| diff < 0.15."""
    from autobench.noise_kernel import compute_spectral_fitness

    with tempfile.TemporaryDirectory(prefix="raps_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        path_a = _make_fractal_noise_png(beta=3.0, seed=42, tmp_dir=tmp_path)
        path_b = _make_fractal_noise_png(beta=3.0, seed=999, tmp_dir=tmp_path)

        fitness_a = compute_spectral_fitness(path_a, "value_noise_3d")
        fitness_b = compute_spectral_fitness(path_b, "value_noise_3d")

        assert fitness_a is not None
        assert fitness_b is not None
        diff = abs(fitness_a - fitness_b)
        assert diff < 0.15, (
            f"Phase invariance violated: fitness_a={fitness_a:.4f} fitness_b={fitness_b:.4f} "
            f"diff={diff:.4f} (should be < 0.15)"
        )


def test_spectral_beta_range():
    """A fractal image with beta~3.0 should have measured beta in [2.0, 4.5]."""
    from autobench.oracle_calibration import _compute_spectral_stats

    with tempfile.TemporaryDirectory(prefix="raps_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        path = _make_fractal_noise_png(beta=3.0, seed=7, tmp_dir=tmp_path)
        stats = _compute_spectral_stats(path)

        assert stats is not None
        measured_beta = stats["beta"]
        assert 2.0 <= measured_beta <= 4.5, (
            f"Expected beta in [2.0, 4.5] for fractal noise, got {measured_beta:.3f}"
        )


def test_noise_extract_t_vector_contains_spectral_beta():
    """NoiseKernel.extract_t_vector() must return a dict with 'spectral_beta' key.

    Integration test: constructs NoiseKernel without GPU (graceful degradation).
    extract_t_vector() falls back to 0.0 for any stash that hasn't been written.
    """
    from autobench.kernel_base import KernelConfig, CandidateProgram
    from autobench.noise_kernel import NoiseKernel

    with tempfile.TemporaryDirectory(prefix="raps_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        cfg = KernelConfig(instances=["value_noise_3d"], output_dir=tmp_path)
        kernel = NoiseKernel(cfg)

        prog = CandidateProgram(
            id="test_prog",
            code="float noise(vec3 p){ return 0.0; }",
            island=0,
            generation=0,
            fitness=0.5,
        )
        t_vector = kernel.extract_t_vector(prog)

        assert "spectral_beta" in t_vector, (
            f"extract_t_vector() must include 'spectral_beta' key; got keys: {list(t_vector.keys())}"
        )
        assert "harmonic_ratio" in t_vector
        assert "fitness" in t_vector
