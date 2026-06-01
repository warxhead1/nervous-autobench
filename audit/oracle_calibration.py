"""oracle_calibration — measure T-vector distributions for oracle parameter calibration.

Run: python -m autobench.oracle_calibration [--noise] [--sdf] [--all]

For noise: measures spectral beta and harmonic ratio from existing reference PNGs.
For SDF: measures sign-change density from analytical functions on a 24^3 grid.
Saves calibration JSON to ~/.cache/funsearch/calibration/<domain>.json.
Emits funsearch.calibration.v1 bus event on completion.

Within/between class variance ratios are reported per T-component.
A ratio > 0.1 means the component does NOT discriminate between classes.
For noise this is expected (all three noise types have similar beta/HR),
confirming that RAPS discriminates CORRECT noise from GARBAGE but not sub-types.
For SDF topology, sign_change_density spans a 20x range and ratios should be < 0.1.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_CACHE_ROOT = Path.home() / ".cache" / "funsearch" / "calibration"
_DEBUG_LOG = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"

# Default benchmark dir: look for reference PNGs in the most recent
# benchmarks/curriculum/<date>/ directory under the repo.
_REPO_ROOT = Path(__file__).parent.parent


def _find_reference_png_dir() -> Path | None:
    """Return the directory containing *_reference.png files, or None."""
    # Search heuristic: look for any *_reference.png under autobench/benchmarks
    # or benchmarks/ within the repo, then return the directory of the most
    # recently modified one.
    candidates = list(_REPO_ROOT.glob("autobench/benchmarks/**/*_reference.png"))
    candidates += list(_REPO_ROOT.glob("benchmarks/**/*_reference.png"))
    if not candidates:
        return None
    # Sort by mtime descending, return parent dir of the most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent


# ---------------------------------------------------------------------------
# Spectral statistics from a single image
# ---------------------------------------------------------------------------

def _compute_spectral_stats(img_path: Path) -> dict | None:
    """Compute spectral_beta, harmonic_ratio, and entropy from a PNG.

    Implements the same arithmetic as compute_spectral_fitness in noise_kernel,
    but returns raw T-vector components rather than a class-membership score.

    Returns dict with keys: beta, harmonic_ratio, entropy
    Returns None on I/O or import failure.
    """
    try:
        from PIL import Image as _PILImage  # type: ignore
    except ImportError as e:
        logger.error("PIL not available: %s", e)
        return None

    if not img_path.exists():
        logger.warning("Reference PNG not found: %s", img_path)
        return None

    try:
        arr = np.array(_PILImage.open(str(img_path)).convert("L"), dtype=np.float64)
    except Exception as e:
        logger.error("Failed to load %s: %s", img_path, e)
        return None

    # RAPS (radially-averaged power spectrum)
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

    # 1. Spectral beta: negative slope of log-log RAPS (skip DC bin)
    freqs = np.arange(1, n_bins)
    log_freqs = np.log(freqs.astype(np.float64))
    log_raps = np.log(raps[1:] + 1e-12)
    slope, _ = np.polyfit(log_freqs, log_raps, 1)
    spectral_beta = float(-slope)  # positive for 1/f noise

    # 2. Harmonic ratio: power at 2f vs f (base_freq = n_bins // 8, min 4)
    base_freq = max(4, n_bins // 8)
    harmonic_ratio = float(raps[base_freq * 2] / (raps[base_freq] + 1e-10))

    # 3. Shannon entropy of pixel histogram
    pixel_counts, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = pixel_counts.sum()
    if total > 0:
        probs = pixel_counts[pixel_counts > 0] / total
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    return {"beta": spectral_beta, "harmonic_ratio": harmonic_ratio, "entropy": entropy}


# ---------------------------------------------------------------------------
# Noise calibration
# ---------------------------------------------------------------------------

def calibrate_noise(ref_dir: Path | None = None) -> dict:
    """Measure spectral T-vector parameters from reference PNGs.

    For each *_reference.png found in ref_dir (or auto-detected), computes:
        spectral_beta, harmonic_ratio, entropy

    Since calibration typically has only one reference PNG per class, within-class
    variance is estimated by computing stats from 4 z-slices of the noise volume.
    Z-slices are approximated by cropping different horizontal strips of the 256x256
    render (the probe shader maps iTime→z, but we can simulate multiple z-samples
    by using image sub-regions).  This gives a rough but non-trivial within-class
    variance estimate without requiring GPU renders.

    Returns dict keyed by instance name:
        {
          "<name>": {
            "beta": float,       # mean across slices
            "beta_std": float,   # std across slices (within-class estimate)
            "hr": float,         # mean harmonic_ratio
            "hr_std": float,
            "entropy": float,    # mean entropy
            "entropy_std": float,
            "n_slices": int,
            "sources": [str],    # paths used
          }
        }
    """
    if ref_dir is None:
        ref_dir = _find_reference_png_dir()
    if ref_dir is None or not ref_dir.exists():
        logger.warning("No reference PNG directory found; noise calibration skipped.")
        return {}

    try:
        from PIL import Image as _PILImage  # type: ignore
    except ImportError as e:
        logger.error("PIL not available for noise calibration: %s", e)
        return {}

    results: dict[str, Any] = {}

    # Canonical instance names
    canonical = {"value_noise_3d", "perlin_like", "fbm_2octave"}

    png_files = sorted(ref_dir.glob("*_reference.png"))
    if not png_files:
        logger.warning("No *_reference.png files found in %s", ref_dir)
        return {}

    for png_path in png_files:
        # Extract instance name: strip '_reference.png' suffix
        stem = png_path.stem  # e.g. 'value_noise_3d_reference'
        if stem.endswith("_reference"):
            name = stem[: -len("_reference")]
        else:
            name = stem

        # Load the full image and split into 4 horizontal strips as z-slice proxies.
        # Each strip covers a different spatial frequency neighbourhood.
        try:
            img = np.array(_PILImage.open(str(png_path)).convert("L"), dtype=np.float64)
        except Exception as e:
            logger.warning("Could not load %s: %s", png_path, e)
            continue

        h = img.shape[0]
        # 4 overlapping strips: top-half, bottom-half, left-half, right-half
        # Using 2D spatial regions as independent samples.
        strips = [
            img[: h // 2, :],
            img[h // 2 :, :],
            img[:, : img.shape[1] // 2],
            img[:, img.shape[1] // 2 :],
        ]

        betas, hrs, entropies = [], [], []

        for strip in strips:
            # Pad strip to square if needed (polyfit needs consistent shape)
            sh, sw = strip.shape
            side = min(sh, sw)
            strip_sq = strip[:side, :side]

            fft_shifted = np.fft.fftshift(np.fft.fft2(strip_sq))
            magnitude_sq = np.abs(fft_shifted) ** 2
            sh2, sw2 = magnitude_sq.shape
            cy, cx = sh2 // 2, sw2 // 2
            y_idx, x_idx = np.mgrid[:sh2, :sw2]
            r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
            n_bins = min(sh2 // 2, sw2 // 2)
            raps = np.array(
                [magnitude_sq[r == i].mean() if (r == i).any() else 0.0
                 for i in range(n_bins)],
                dtype=np.float64,
            )

            freqs = np.arange(1, n_bins)
            log_freqs = np.log(freqs.astype(np.float64))
            log_raps = np.log(raps[1:] + 1e-12)
            if len(log_freqs) < 2:
                continue
            slope, _ = np.polyfit(log_freqs, log_raps, 1)
            betas.append(float(-slope))

            base_freq = max(4, n_bins // 8)
            hrs.append(float(raps[base_freq * 2] / (raps[base_freq] + 1e-10)))

            pixel_counts, _ = np.histogram(strip_sq, bins=256, range=(0, 256))
            total = pixel_counts.sum()
            if total > 0:
                probs = pixel_counts[pixel_counts > 0] / total
                entropies.append(float(-np.sum(probs * np.log2(probs))))

        if not betas:
            continue

        results[name] = {
            "beta": float(np.mean(betas)),
            "beta_std": float(np.std(betas)),
            "hr": float(np.mean(hrs)),
            "hr_std": float(np.std(hrs)),
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "entropy_std": float(np.std(entropies)) if entropies else 0.0,
            "n_slices": len(betas),
            "sources": [str(png_path)],
        }

    return results


def _noise_variance_ratios(noise_data: dict) -> dict:
    """Compute within/between class variance ratios for each T component.

    within_variance = mean of per-class variances (estimated from strips)
    between_variance = variance of per-class means across classes

    Returns dict: {"beta": float, "hr": float, "entropy": float}
    """
    components = ["beta", "hr", "entropy"]
    ratios = {}

    for comp in components:
        std_key = f"{comp}_std"
        means = [v[comp] for v in noise_data.values() if comp in v]
        stds = [v[std_key] for v in noise_data.values() if std_key in v]

        if len(means) < 2:
            ratios[comp] = None
            continue

        within_var = float(np.mean([s ** 2 for s in stds]))
        between_var = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0

        if between_var < 1e-12:
            # All class means identical → ratio is effectively infinite
            ratios[comp] = float("inf")
        else:
            ratios[comp] = within_var / between_var

    return ratios


# ---------------------------------------------------------------------------
# SDF topology calibration
# ---------------------------------------------------------------------------

# Documented targets from empirical measurement at 24^3 grid.
# These are the ground-truth values the calibration should reproduce.
_SDF_TOPO_TARGETS = {
    "gyroid": {"target": 0.1780, "sigma": 0.06},
    "round_box": {"target": 0.0091, "sigma": 0.006},
    "warped_sphere": {"target": 0.0179, "sigma": 0.012},
    "sphere": {"target": 0.0140, "sigma": 0.010},
    "smooth_union": {"target": 0.0250, "sigma": 0.015},
}


def _compute_sign_change_density(
    target_fn, grid_size: int = 24, lo: float = -1.5, hi: float = 1.5
) -> float:
    """Count sign changes (zero crossings) in a target_fn on a regular grid.

    Evaluates the analytical SDF on a grid_size^3 regular lattice.
    A voxel edge has a sign change when f(p0)*f(p1) < 0.
    sign_change_density = n_sign_change_edges / n_total_edges

    Args:
        target_fn:  Python callable (x, y, z) -> float (the analytical SDF).
        grid_size:  Number of grid points per axis.
        lo, hi:     Sampling domain bounds.

    Returns:
        float in [0, 1] — fraction of edges with a sign change.
    """
    # Build grid
    coords = np.linspace(lo, hi, grid_size)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")  # shape (N, N, N)

    # Evaluate analytical function on all grid points
    vals = np.zeros_like(xs)
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                vals[ix, iy, iz] = target_fn(
                    float(xs[ix, iy, iz]),
                    float(ys[ix, iy, iz]),
                    float(zs[ix, iy, iz]),
                )

    # Count sign-change edges along each axis direction
    n_sign_changes = 0
    n_total_edges = 0

    # x-direction edges
    v0 = vals[:-1, :, :]
    v1 = vals[1:, :, :]
    n_sign_changes += int(np.sum(v0 * v1 < 0))
    n_total_edges += v0.size

    # y-direction edges
    v0 = vals[:, :-1, :]
    v1 = vals[:, 1:, :]
    n_sign_changes += int(np.sum(v0 * v1 < 0))
    n_total_edges += v0.size

    # z-direction edges
    v0 = vals[:, :, :-1]
    v1 = vals[:, :, 1:]
    n_sign_changes += int(np.sum(v0 * v1 < 0))
    n_total_edges += v0.size

    return n_sign_changes / n_total_edges if n_total_edges > 0 else 0.0


def calibrate_sdf_topology(instances: list[str] | None = None) -> dict:
    """Measure sign_change_density for each SDF instance on a 24^3 grid.

    Uses the ANALYTICAL target functions directly (not LLM-generated code).

    Args:
        instances: List of instance names.  Defaults to all known instances.

    Returns dict:
        {
          "<name>": {
            "measured": float,     # measured sign_change_density
            "target": float,       # documented target value
            "sigma": float,        # documented tolerance sigma
            "within_target": bool, # |measured - target| < 2*sigma
            "ratio_to_round_box": float,  # discriminative power vs simplest instance
          }
        }
    """
    try:
        from autobench.sdf_kernel import _INSTANCE_FACTORIES  # type: ignore
    except ImportError as e:
        logger.error("Could not import sdf_kernel: %s", e)
        return {}

    if instances is None:
        instances = list(_INSTANCE_FACTORIES.keys())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_instances: list[str] = []
    for name in instances:
        if name not in seen:
            seen.add(name)
            unique_instances.append(name)

    results: dict[str, Any] = {}
    round_box_density: float | None = None

    for name in unique_instances:
        if name not in _INSTANCE_FACTORIES:
            logger.warning("Unknown SDF instance '%s'; skipping.", name)
            continue

        target_fn, _desc, lo, hi, _n = _INSTANCE_FACTORIES[name]

        # Use the domain defined by the instance factory, clamped to a
        # symmetric range suitable for the 24^3 grid measurement.
        domain_half = min(abs(lo), abs(hi), 1.5)
        measured = _compute_sign_change_density(
            target_fn, grid_size=24, lo=-domain_half, hi=domain_half
        )

        topo_target = _SDF_TOPO_TARGETS.get(name)
        target_val = topo_target["target"] if topo_target else None
        sigma_val = topo_target["sigma"] if topo_target else None
        within = (
            abs(measured - target_val) < 2 * sigma_val
            if (target_val is not None and sigma_val is not None)
            else None
        )

        if name == "round_box":
            round_box_density = measured

        results[name] = {
            "measured": measured,
            "target": target_val,
            "sigma": sigma_val,
            "within_target": within,
        }

    # Add ratio_to_round_box for discriminative power assessment
    if round_box_density and round_box_density > 0:
        for name, entry in results.items():
            entry["ratio_to_round_box"] = entry["measured"] / round_box_density
    elif round_box_density == 0:
        for entry in results.values():
            entry["ratio_to_round_box"] = float("inf")

    return results


def _sdf_variance_ratios(sdf_data: dict) -> dict:
    """Compute within/between variance ratio for sign_change_density.

    Since we have one measurement per instance, within-class variance is zero
    and between-class variance is entirely between-instance.  Reports the
    inter-instance coefficient of variation (std/mean) as a proxy.

    Returns dict: {"sign_change_density": float}
    """
    densities = [v["measured"] for v in sdf_data.values()]
    if len(densities) < 2:
        return {"sign_change_density": None}
    mean = float(np.mean(densities))
    std = float(np.std(densities, ddof=1))
    cv = std / mean if mean > 0 else float("inf")
    return {"sign_change_density": cv}  # high CV = good discriminative power


# ---------------------------------------------------------------------------
# Save / load calibration
# ---------------------------------------------------------------------------

def save_calibration(domain: str, data: dict, path: Path | None = None) -> Path:
    """Save calibration data as pretty-printed JSON.

    Args:
        domain: 'noise' or 'sdf'
        data:   The calibration dict to save.
        path:   Override save location.  Defaults to
                ~/.cache/funsearch/calibration/{domain}.json.

    Returns the path where the file was saved.
    """
    if path is None:
        path = _CACHE_ROOT / f"{domain}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s calibration to %s", domain, path)
    return path


def load_calibration(domain: str, path: Path | None = None) -> dict | None:
    """Load calibration data from JSON.

    Returns None if the file does not exist.
    """
    if path is None:
        path = _CACHE_ROOT / f"{domain}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to load calibration from %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Bus event emission
# ---------------------------------------------------------------------------

def _find_nervous_bin() -> str | None:
    candidates = [
        _REPO_ROOT / "sdk" / "shell" / "nervous",
        Path.home() / "projects" / "nervous-bus" / "sdk" / "shell" / "nervous",
        Path("/usr/local/bin/nervous"),
        Path("/usr/bin/nervous"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _emit_calibration_event(
    domain: str,
    measured_params: dict,
    within_between_ratios: dict,
    calibration_path: str,
) -> None:
    """Emit funsearch.calibration.v1 bus event.

    Mirrors the pattern from NoiseKernel._publish: appends to debug.jsonl
    and optionally invokes the nervous CLI with NBUS_SKIP_VALIDATION=1.
    """
    event = {
        "specversion": "1.0",
        "id": uuid.uuid4().urn,
        "source": "/autobench/oracle_calibration",
        "type": "funsearch.calibration.v1",
        "datacontenttype": "application/json",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {
            "domain": domain,
            "measured_params": measured_params,
            "within_between_ratios": within_between_ratios,
            "calibration_path": calibration_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    payload = json.dumps(event)

    # Append to debug log
    _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_DEBUG_LOG, "a") as f:
            f.write(payload + "\n")
    except Exception as e:
        logger.debug("Failed to write to debug log: %s", e)

    # Optionally publish via nervous CLI
    nervous_bin = _find_nervous_bin()
    if nervous_bin:
        try:
            env = dict(os.environ)
            env["NBUS_SKIP_VALIDATION"] = "1"
            env["NERVOUS_NO_ZELLIJ"] = "1"
            env["NERVOUS_NO_REDIS"] = "1"
            proc = subprocess.Popen(
                [nervous_bin, "publish", "funsearch.calibration.v1", payload],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            proc.wait(timeout=2)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

def _print_noise_results(noise_data: dict, ratios: dict) -> None:
    print("\n=== NOISE DOMAIN CALIBRATION ===")
    if not noise_data:
        print("  (no data — no reference PNGs found)")
        return

    print(f"\n  {'Instance':<20} {'beta':>8} {'beta_std':>10} {'hr':>8} {'hr_std':>9} "
          f"{'entropy':>9} {'n_slices':>9}")
    print("  " + "-" * 80)
    for name, d in sorted(noise_data.items()):
        print(
            f"  {name:<20} {d['beta']:8.3f} {d['beta_std']:10.3f} "
            f"{d['hr']:8.4f} {d['hr_std']:9.4f} "
            f"{d['entropy']:9.3f} {d['n_slices']:9d}"
        )

    print("\n  Within/between class variance ratios:")
    print(f"  {'Component':<20} {'ratio':>10}  {'verdict':>30}")
    print("  " + "-" * 65)
    for comp, ratio in sorted(ratios.items()):
        if ratio is None:
            verdict = "N/A (< 2 classes)"
        elif ratio == float("inf"):
            verdict = "WARNING: between-class variance ~ 0"
        elif ratio > 0.1:
            verdict = "WARNING (> 0.1) — does NOT discriminate sub-types"
        else:
            verdict = "OK — good sufficient statistic"
        ratio_str = f"{ratio:.4f}" if isinstance(ratio, float) and math.isfinite(ratio) else str(ratio)
        print(f"  {comp:<20} {ratio_str:>10}  {verdict}")

    print(
        "\n  NOTE: Noise RAPS discriminates CORRECT noise (beta~3.0, hr~0.1) from"
        "\n        GARBAGE, but NOT between noise sub-types. High within/between"
        "\n        ratios here are EXPECTED — they confirm this property."
    )


def _print_sdf_results(sdf_data: dict, ratios: dict) -> None:
    print("\n=== SDF TOPOLOGY CALIBRATION ===")
    if not sdf_data:
        print("  (no data)")
        return

    print(f"\n  {'Instance':<18} {'measured':>10} {'target':>9} {'sigma':>8} "
          f"{'within_2σ':>10} {'ratio/round_box':>16}")
    print("  " + "-" * 78)
    for name in sorted(sdf_data.keys()):
        d = sdf_data[name]
        measured = d["measured"]
        target = d.get("target")
        sigma = d.get("sigma")
        within = d.get("within_target")
        ratio = d.get("ratio_to_round_box")

        target_str = f"{target:.4f}" if target is not None else "N/A"
        sigma_str = f"{sigma:.4f}" if sigma is not None else "N/A"
        within_str = "YES" if within is True else ("NO" if within is False else "N/A")
        ratio_str = (
            f"{ratio:.1f}x" if ratio is not None and math.isfinite(ratio)
            else ("inf" if ratio == float("inf") else "N/A")
        )
        print(
            f"  {name:<18} {measured:10.4f} {target_str:>9} {sigma_str:>8} "
            f"{within_str:>10} {ratio_str:>16}"
        )

    print("\n  Inter-instance coefficient of variation (higher = better discriminator):")
    cv = ratios.get("sign_change_density")
    if cv is not None and math.isfinite(cv):
        if cv > 1.0:
            verdict = "EXCELLENT discriminative range (20x spread expected)"
        elif cv > 0.5:
            verdict = "GOOD"
        else:
            verdict = "WARNING: low spread — topology oracle may not discriminate"
        print(f"  sign_change_density CoV = {cv:.3f}  [{verdict}]")
    else:
        print(f"  sign_change_density CoV = {cv}")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Oracle calibration — measure T-vector distributions for oracle parameters."
    )
    parser.add_argument("--noise", action="store_true", help="Run noise calibration")
    parser.add_argument("--sdf", action="store_true", help="Run SDF topology calibration")
    parser.add_argument("--all", action="store_true", help="Run all calibrations")
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Directory containing *_reference.png files (noise calibration).",
    )
    parser.add_argument(
        "--sdf-instances",
        default=None,
        help="Comma-separated SDF instance names (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override cache directory for saved JSON files.",
    )
    args = parser.parse_args()

    run_noise = args.noise or args.all
    run_sdf = args.sdf or args.all

    if not (run_noise or run_sdf):
        parser.print_help()
        print(
            "\nNo calibration selected. Use --noise, --sdf, or --all.\n"
            "Example: python -m autobench.oracle_calibration --all"
        )
        return

    sdf_instances = (
        [s.strip() for s in args.sdf_instances.split(",")]
        if args.sdf_instances
        else None
    )

    # --- Noise calibration ---
    if run_noise:
        print(f"\nRunning noise calibration (ref_dir={args.ref_dir or 'auto-detect'})...")
        noise_data = calibrate_noise(ref_dir=args.ref_dir)
        noise_ratios = _noise_variance_ratios(noise_data)
        _print_noise_results(noise_data, noise_ratios)

        cal = {"instances": noise_data, "within_between_ratios": noise_ratios}
        path = save_calibration("noise", cal, args.output_dir / "noise.json" if args.output_dir else None)
        _emit_calibration_event(
            domain="noise",
            measured_params=noise_data,
            within_between_ratios=noise_ratios,
            calibration_path=str(path),
        )
        print(f"\n  Saved to: {path}")

    # --- SDF calibration ---
    if run_sdf:
        print(f"\nRunning SDF topology calibration (instances={sdf_instances or 'all'})...")
        sdf_data = calibrate_sdf_topology(instances=sdf_instances)
        sdf_ratios = _sdf_variance_ratios(sdf_data)
        _print_sdf_results(sdf_data, sdf_ratios)

        cal = {"instances": sdf_data, "within_between_ratios": sdf_ratios}
        path = save_calibration("sdf", cal, args.output_dir / "sdf.json" if args.output_dir else None)
        _emit_calibration_event(
            domain="sdf",
            measured_params=sdf_data,
            within_between_ratios=sdf_ratios,
            calibration_path=str(path),
        )
        print(f"\n  Saved to: {path}")

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
