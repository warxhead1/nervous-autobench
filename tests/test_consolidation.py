"""Tests for ConsolidatedPrior — cross-run T-vector accumulation.

Covers:
    * Round-trip: save T-vector, reload, verify contents
    * fit() returns sensible mean/std after 10+ vectors
    * diversity_score: a vector far from prior gets higher score than one near center
"""

from __future__ import annotations

import json
import tempfile
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prior(tmp_path: pathlib.Path, n_entries: int = 12):
    """Return a ConsolidatedPrior populated with n_entries T-vectors.

    Fitness ramps 0.05 → 0.60 so top-25% (n//4 entries) have highest fitness.
    T-vector values are spread linearly so the top group has non-zero variance.
    """
    from autobench.kernels import ConsolidatedPrior

    prior = ConsolidatedPrior("test", tmp_path / "prior.jsonl")
    for i in range(n_entries):
        fitness = (i + 1) * (0.6 / n_entries)  # 0.05..0.60
        prior.append(
            run_id=f"run_{i}",
            fitness=fitness,
            t_vector={
                "spectral_beta": 3.0 + i * 0.1,   # spread: 3.0 to 4.1
                "harmonic_ratio": 0.10 + i * 0.01, # spread: 0.10 to 0.21
            },
            code_hash=f"hash_{i}",
        )
    return prior


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


def test_consolidated_prior_roundtrip():
    """Append a T-vector, reload from file, verify the entry is preserved."""
    from autobench.kernels import ConsolidatedPrior

    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        prior_path = tmp_path / "prior.jsonl"

        prior = ConsolidatedPrior("test", prior_path)
        prior.append(
            run_id="run_abc",
            fitness=0.85,
            t_vector={"spectral_beta": 3.14, "harmonic_ratio": 0.11},
            code_hash="deadbeef",
        )

        # Reload from disk
        prior2 = ConsolidatedPrior("test", prior_path)
        count = prior2.load()

        assert count == 1, f"Expected 1 entry after reload, got {count}"
        entry = prior2._entries[0]
        assert entry["run_id"] == "run_abc"
        assert abs(entry["fitness"] - 0.85) < 1e-9
        assert abs(entry["t_vector"]["spectral_beta"] - 3.14) < 1e-9
        assert abs(entry["t_vector"]["harmonic_ratio"] - 0.11) < 1e-9
        assert entry["code_hash"] == "deadbeef"


def test_consolidated_prior_roundtrip_multiple_entries():
    """Multiple appends survive a reload — file is append-only JSONL."""
    from autobench.kernels import ConsolidatedPrior

    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        prior_path = tmp_path / "prior.jsonl"

        prior = ConsolidatedPrior("test", prior_path)
        for i in range(5):
            prior.append(f"run_{i}", float(i) * 0.2, {"x": float(i)}, f"h{i}")

        prior2 = ConsolidatedPrior("test", prior_path)
        count = prior2.load()
        assert count == 5
        # Verify ordering is preserved
        for i, entry in enumerate(prior2._entries):
            assert entry["run_id"] == f"run_{i}"
            assert abs(entry["t_vector"]["x"] - float(i)) < 1e-9


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------


def test_consolidated_prior_fit_returns_mean_and_std():
    """After ≥8 T-vectors, fit() returns {key: (mean, std)} for each component."""
    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        prior = _make_prior(tmp_path, n_entries=12)

        stats = prior.fit()

        assert isinstance(stats, dict), "fit() should return a dict"
        assert "spectral_beta" in stats, f"Expected 'spectral_beta' key; got {list(stats.keys())}"
        assert "harmonic_ratio" in stats

        for key, (mean, std) in stats.items():
            assert isinstance(mean, float), f"stats[{key}].mean should be float"
            assert isinstance(std, float), f"stats[{key}].std should be float"
            assert std >= 0.0, f"std must be non-negative, got {std} for {key}"


def test_consolidated_prior_fit_top_quartile_only():
    """fit() uses only the top-25% by fitness — mean reflects high-fitness values."""
    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        # 12 entries; fitness 0.05..0.60; top 3 have beta 3.9, 4.0, 4.1
        prior = _make_prior(tmp_path, n_entries=12)

        stats = prior.fit()
        mean_beta, _ = stats["spectral_beta"]

        # Full-population mean would be ~3.55; top-25% mean must be ~4.0
        assert mean_beta > 3.7, (
            f"fit() should use only top-25%, but spectral_beta mean={mean_beta:.3f} "
            f"is too low (expected ~4.0)"
        )


def test_consolidated_prior_fit_empty_returns_empty():
    """fit() on empty prior returns {}."""
    from autobench.kernels import ConsolidatedPrior

    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        prior = ConsolidatedPrior("test", tmp_path / "prior.jsonl")
        stats = prior.fit()
        assert stats == {}


# ---------------------------------------------------------------------------
# diversity_score tests
# ---------------------------------------------------------------------------


def test_diversity_score_far_is_higher_than_near():
    """A T-vector far from the prior centre gets a higher diversity score."""
    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        prior = _make_prior(tmp_path, n_entries=12)

        # Prior mean is around spectral_beta~4.0, harmonic_ratio~0.20
        near_center = {"spectral_beta": 4.0, "harmonic_ratio": 0.20}
        far_from_center = {"spectral_beta": 1.0, "harmonic_ratio": 0.50}

        score_near = prior.diversity_score(near_center)
        score_far = prior.diversity_score(far_from_center)

        assert score_far > score_near, (
            f"Vector far from prior should have higher diversity score: "
            f"near={score_near:.4f}, far={score_far:.4f}"
        )


def test_diversity_score_empty_prior_returns_one():
    """When no prior exists, diversity_score returns 1.0 (maximal novelty)."""
    from autobench.kernels import ConsolidatedPrior

    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        prior = ConsolidatedPrior("test", tmp_path / "prior.jsonl")
        score = prior.diversity_score({"spectral_beta": 3.0, "harmonic_ratio": 0.1})
        assert score == 1.0, f"Empty prior should return diversity 1.0, got {score}"


def test_diversity_score_at_center_near_zero():
    """A vector at the exact prior mean should have diversity score ~0."""
    with tempfile.TemporaryDirectory(prefix="prior_test_") as tmp:
        tmp_path = pathlib.Path(tmp)
        prior = _make_prior(tmp_path, n_entries=12)

        stats = prior.fit()
        mean_beta, _ = stats["spectral_beta"]
        mean_hr, _ = stats["harmonic_ratio"]

        at_center = {"spectral_beta": mean_beta, "harmonic_ratio": mean_hr}
        score = prior.diversity_score(at_center)

        assert score < 1e-9, (
            f"Vector at prior mean should have diversity ~0, got {score:.6f}"
        )
