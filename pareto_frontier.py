"""
Pareto frontier persistence layer for autobench.

Tracks which harness configurations are Pareto-optimal across quality/cost/speed
dimensions, across multiple benchmark runs.

Reference: AutoBench 2.0 leaderboard approach — track 30+ models on
quality/cost/speed, compute AAII (Area Above Ideal) correlation.
LiveCodeBench uses similar frontier analysis for multi-dimensional tradeoff evaluation.

Classes:
    ParetoFrontier — main persistence layer, stores and queries frontier configs
    FrontierConfig — a configuration on the Pareto frontier with metrics
    WeightSpace — evaluates frontier under different weight configurations

Functions:
    dominance_check(config_a, config_b) — check if A dominates B
    update_frontier(frontier, new_config) — add config, remove dominated ones
    query_frontier(frontier, cost_budget, time_budget) — find best config for budget
    compute_aaii(frontier) — Area Above Ideal curve for frontier quality

Usage:
    pf = ParetoFrontier(storage="json", path="frontier.json")
    pf.load()

    result = pf.add_result(benchmark_result, harness_config)
    # result.dominated = [configs that are now dominated]
    # result.frontier = current frontier

    best = pf.query(cost_budget=0.05, time_budget=10.0)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

# ULID-like generator (same as session_state.py)
def _generate_ulid() -> str:
    """Generate a ULID-like 26-char sortable identifier."""
    import random
    import time as _time
    timestamp_ms = int(_time.time() * 1000)
    random_bits = random.getrandbits(80)
    chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    result = []
    value = (timestamp_ms << 80) | random_bits
    for _ in range(26):
        result.append(chars[value & 31])
        value >>= 5
    return "".join(reversed(result))


def _rfc3339_now() -> str:
    """Return current time in RFC3339 format (UTC)."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core dominance and frontier logic
# ---------------------------------------------------------------------------

def dominance_check(
    config_a: dict[str, float],
    config_b: dict[str, float],
) -> bool:
    """
    Check if config_a dominates config_b across quality/cost/speed.

    Dominance means: A is >= B in all dimensions AND strictly > in at least one.

    Dimensions:
        quality: higher is better (maximize)
        cost:    lower is better (minimize — we invert internally)
        speed:   higher is better (maximize)

    Args:
        config_a: Dict with 'quality', 'cost', 'speed' keys (float values)
        config_b: Same structure

    Returns:
        True if config_a dominates config_b, False otherwise.
    """
    q_a = config_a.get("quality", 0.0)
    c_a = config_a.get("cost", 0.0)
    s_a = config_a.get("speed", 0.0)

    q_b = config_b.get("quality", 0.0)
    c_b = config_b.get("cost", 0.0)
    s_b = config_b.get("speed", 0.0)

    # Check all dimensions >= (higher quality/speed are better, lower cost is better)
    quality_ok = q_a >= q_b
    cost_ok = c_a <= c_b  # lower cost is better
    speed_ok = s_a >= s_b

    if not (quality_ok and cost_ok and speed_ok):
        return False

    # At least one strictly better
    strictly_better = (q_a > q_b) or (c_a < c_b) or (s_a > s_b)
    return strictly_better


def dominance_check_for_point(
    quality_a: float, cost_a: float, speed_a: float,
    quality_b: float, cost_b: float, speed_b: float,
) -> bool:
    """
    Scalar-version of dominance_check.
    Returns True if (quality_a, cost_a, speed_a) dominates (quality_b, cost_b, speed_b).
    """
    quality_ok = quality_a >= quality_b
    cost_ok = cost_a <= cost_b  # lower cost is better
    speed_ok = speed_a >= speed_b

    if not (quality_ok and cost_ok and speed_ok):
        return False

    strictly_better = (quality_a > quality_b) or (cost_a < cost_b) or (speed_a > speed_b)
    return strictly_better


def update_frontier(
    frontier: list[dict[str, Any]],
    new_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Add a new configuration to the frontier, removing any that become dominated.

    Args:
        frontier: Current list of frontier configurations.
                  Each config is a dict with 'quality', 'cost', 'speed' keys.
        new_config: New configuration to add (same structure).

    Returns:
        Tuple of (updated_frontier, list_of_newly_dominated_configs).
        The dominated configs are removed from the frontier and returned
        so callers can handle them (e.g., archive rather than delete).

    Algorithm:
        1. Check if new_config is dominated by any existing frontier point.
           If so, skip (no update needed).
        2. Find all existing frontier points that new_config dominates.
           Remove them (they are no longer on the frontier).
        3. Add new_config to frontier.
    """
    dominated = []

    # Step 1: Is new_config dominated by any existing frontier point?
    for existing in frontier:
        if dominance_check(existing, new_config):
            # new_config is dominated — no update needed
            return frontier, []
        # Also check the reverse (new dominates existing) to identify what to remove
        # We'll collect all dominated configs in step 2

    # Step 2: Find all configs that new_config dominates
    surviving = []
    for existing in frontier:
        if dominance_check(new_config, existing):
            dominated.append(existing)
        else:
            surviving.append(existing)

    # Step 3: Add new_config
    surviving.append(new_config)

    return surviving, dominated


def query_frontier(
    frontier: list[dict[str, Any]],
    cost_budget: float | None = None,
    time_budget: float | None = None,
    min_quality: float | None = None,
) -> list[dict[str, Any]]:
    """
    Query the frontier for configurations meeting budget constraints.

    Args:
        frontier: List of frontier configurations.
        cost_budget: Maximum cost (dollars) — only configs with cost <= this
        time_budget: Maximum time (seconds) — only configs with speed >= this
                     (speed stored as normalized value, so higher = faster)
        min_quality: Minimum quality score required

    Returns:
        List of frontier configs meeting ALL criteria, sorted by
        a composite score: quality + (1 - normalized_cost) + speed.
        Empty list if no configs match.
    """
    candidates = []

    for config in frontier:
        # Apply cost constraint (cost is lower = better)
        if cost_budget is not None:
            if config.get("cost", math.inf) > cost_budget:
                continue

        # Apply time constraint (speed is higher = better, so check if speed >= threshold)
        if time_budget is not None:
            # speed stored as normalized [0, 1]. time_budget of e.g. 10s maps to speed threshold
            # We map time_budget to a speed threshold: faster than budget = good
            # speed = 1.0 means fastest possible (0 time), 0.0 means slowest (infinite time)
            # Map time_budget to speed: if time_budget is 10s, we want speed >= 1/(1+time_budget)
            # Simplified: speed_threshold = 1.0 / (1.0 + time_budget) assumes time is normalized
            speed_threshold = 1.0 / (1.0 + time_budget) if time_budget > 0 else 0.5
            if config.get("speed", 0.0) < speed_threshold:
                continue

        # Apply quality constraint
        if min_quality is not None:
            if config.get("quality", 0.0) < min_quality:
                continue

        candidates.append(config)

    # Sort by composite score (higher is better)
    def composite_score(cfg: dict) -> float:
        q = cfg.get("quality", 0.0)
        c = cfg.get("cost", 0.0)
        s = cfg.get("speed", 0.0)
        # cost is [0, 1] normalized where 1 = cheapest. So 1 - c gives us "cost goodness"
        return q + (1.0 - c) + s

    candidates.sort(key=composite_score, reverse=True)
    return candidates


def compute_aaii(frontier: list[dict[str, Any]], samples: int = 100) -> float:
    """
    Compute Area Above Ideal (AAII) for a Pareto frontier.

    AAII measures how well-distributed and optimal a frontier is.
    Lower AAII = better frontier (more area covered near ideal point).

    The ideal point is (quality=1, cost=0, speed=1).
    The AAII is computed by sampling the cost-quality trade-off curve.

    Args:
        frontier: List of frontier configurations.
        samples: Number of samples for integration.

    Returns:
        AAII score (lower is better, 0 = ideal frontier).
    """
    if not frontier:
        return 1.0

    # Sort frontier by cost (ascending)
    sorted_frontier = sorted(frontier, key=lambda c: c.get("cost", math.inf))

    # Generate sample points along the cost axis
    min_cost = min(c.get("cost", 0) for c in frontier)
    max_cost = max(c.get("cost", 1) for c in frontier)

    if min_cost >= max_cost:
        # All configs have same cost — compute single point quality
        total_quality = sum(c.get("quality", 0) for c in frontier) / len(frontier)
        # AAII = 1 - area under quality curve (ideal would be quality=1 everywhere)
        return 1.0 - total_quality

    # Sample costs and find max quality at each
    delta = (max_cost - min_cost) / samples
    area = 0.0

    for i in range(samples):
        cost_at = min_cost + (i + 0.5) * delta
        # Find max quality at this cost
        max_q = 0.0
        for config in sorted_frontier:
            cfg_cost = config.get("cost", math.inf)
            if cfg_cost <= cost_at:
                max_q = max(max_q, config.get("quality", 0.0))
        # Rectangle: width=delta, height=(1 - max_q) — area above the frontier
        area += (1.0 - max_q) * delta

    # Normalize by total possible area (unit square)
    total_area = (max_cost - min_cost)
    if total_area > 0:
        return area / total_area
    return 1.0


# ---------------------------------------------------------------------------
# Weight space exploration
# ---------------------------------------------------------------------------

class WeightPreset(str, Enum):
    """Predefined weight configurations for multi-objective evaluation."""
    BALANCED = "balanced"       # 0.33 quality, 0.33 cost, 0.33 speed
    QUALITY_FIRST = "quality_first"  # 0.6 quality, 0.2 cost, 0.2 speed
    COST_FOCUSED = "cost_focused"    # 0.2 quality, 0.6 cost, 0.2 speed
    SPEED_FIRST = "speed_first"     # 0.2 quality, 0.2 cost, 0.6 speed
    QUALITY_COST = "quality_cost"   # 0.5 quality, 0.5 cost, 0.0 speed
    QUALITY_SPEED = "quality_speed" # 0.5 quality, 0.0 cost, 0.5 speed
    COST_SPEED = "cost_speed"       # 0.0 quality, 0.5 cost, 0.5 speed


WEIGHT_PRESETS: dict[WeightPreset, dict[str, float]] = {
    WeightPreset.BALANCED: {"quality": 0.33, "cost": 0.33, "speed": 0.34},
    WeightPreset.QUALITY_FIRST: {"quality": 0.6, "cost": 0.2, "speed": 0.2},
    WeightPreset.COST_FOCUSED: {"quality": 0.2, "cost": 0.6, "speed": 0.2},
    WeightPreset.SPEED_FIRST: {"quality": 0.2, "cost": 0.2, "speed": 0.6},
    WeightPreset.QUALITY_COST: {"quality": 0.5, "cost": 0.5, "speed": 0.0},
    WeightPreset.QUALITY_SPEED: {"quality": 0.5, "cost": 0.0, "speed": 0.5},
    WeightPreset.COST_SPEED: {"quality": 0.0, "cost": 0.5, "speed": 0.5},
}


@dataclass
class ScoredConfig:
    """A frontier configuration with a weighted utility score."""
    config: dict[str, Any]
    score: float
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "score": self.score,
            "weights": self.weights,
        }


def evaluate_weight_space(
    frontier: list[dict[str, Any]],
    preset: WeightPreset = WeightPreset.BALANCED,
    normalize: bool = True,
) -> list[ScoredConfig]:
    """
    Evaluate all frontier configs under a weight preset.

    Args:
        frontier: List of frontier configurations.
        preset: Weight preset to apply.
        normalize: If True, normalize weights to sum to 1.0.

    Returns:
        List of ScoredConfig sorted by score (highest first).
    """
    weights = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS[WeightPreset.BALANCED]).copy()

    if normalize:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    results = []
    for cfg in frontier:
        score = _compute_weighted_score(cfg, weights)
        results.append(ScoredConfig(config=cfg, score=score, weights=weights))

    results.sort(key=lambda x: x.score, reverse=True)
    return results


def _compute_weighted_score(config: dict[str, Any], weights: dict[str, float]) -> float:
    """
    Compute weighted utility score for a config.

    For cost: we use (1 - cost) because lower cost is better.
    For quality and speed: we use the values directly (higher is better).

    Returns score in [0, 1].
    """
    q = config.get("quality", 0.0)
    c = config.get("cost", 0.0)
    s = config.get("speed", 0.0)

    # Invert cost so higher = better
    c_goodness = 1.0 - c

    score = (
        weights.get("quality", 0.33) * q +
        weights.get("cost", 0.33) * c_goodness +
        weights.get("speed", 0.33) * s
    )
    return max(0.0, min(1.0, score))


def rank_frontier_by_preset(
    frontier: list[dict[str, Any]],
    preset: WeightPreset = WeightPreset.BALANCED,
) -> list[dict[str, Any]]:
    """
    Rank frontier configurations by a weight preset.

    Convenience wrapper that returns ranked configs (highest score first).
    """
    scored = evaluate_weight_space(frontier, preset)
    return [s.config for s in scored]


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------

class StorageBackend(str, Enum):
    """Supported storage backends for ParetoFrontier."""
    JSON = "json"
    SQLITE = "sqlite"


@dataclass
class FrontierConfig:
    """
    A configuration on the Pareto frontier.

    Attributes:
        config_id: Unique identifier (ULID-like).
        harness_config: Dict of harness settings (model, temperature, etc.).
        metrics: Dict with quality, cost, speed values.
        weights: Dict with quality, cost, speed weights used for scoring.
        added_at: RFC3339 timestamp when config was added.
        run_id: Identifier for the benchmark run that produced this config.
        metadata: Arbitrary extra data.
    """
    config_id: str
    harness_config: dict[str, Any]
    metrics: dict[str, float]  # quality, cost, speed
    weights: dict[str, float]  # quality, cost, speed weights
    added_at: str
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "harness_config": self.harness_config,
            "metrics": self.metrics,
            "weights": self.weights,
            "added_at": self.added_at,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrontierConfig:
        return cls(
            config_id=data["config_id"],
            harness_config=data["harness_config"],
            metrics=data["metrics"],
            weights=data["weights"],
            added_at=data["added_at"],
            run_id=data["run_id"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class UpdateResult:
    """Result of adding a new config to the frontier."""
    config_id: str
    was_added: bool
    was_dominated: bool  # True if new config was dominated by existing frontier
    newly_dominated: list[str]  # config_ids of configs that became dominated
    frontier_size: int
    aaii: float


class ParetoFrontier:
    """
    Persistence layer for Pareto frontier tracking.

    Stores frontier configurations to disk (JSON or SQLite), supports
    dominance checking, weight space evaluation, and budget-based queries.

    Usage:
        pf = ParetoFrontier(storage="sqlite", path="frontier.db")
        pf.load()

        result = pf.add_result(benchmark_result, harness_config)

        best_configs = pf.query(cost_budget=0.05, time_budget=10.0)
        ranked = pf.rank(preset=WeightPreset.QUALITY_FIRST)
    """

    def __init__(
        self,
        storage: StorageBackend = StorageBackend.JSON,
        path: str | Path = "pareto_frontier.json",
        auto_save: bool = True,
        transfer_coefficient: float = 1.0,
    ):
        """
        Initialize Pareto frontier.

        Args:
            storage: Storage backend (JSON or SQLite).
            path: File path for storage.
            auto_save: If True, save after every modification.
            transfer_coefficient: Cross-model transfer multiplier (arXiv 2604.14004:
                                   memory transfer achieves ~1.037× improvement on held-out models).
                                   Set to 1.0 until tuned with live multi-model data.
        """
        self.storage = storage
        self.path = Path(path)
        self.auto_save = auto_save
        self._configs: list[FrontierConfig] = []
        self._dominated_ids: set[str] = set()  # Archive of removed configs
        self.transfer_coefficient = transfer_coefficient

    # ---- Public API ----

    def load(self) -> None:
        """Load frontier from storage."""
        if self.storage == StorageBackend.JSON:
            self._load_json()
        else:
            self._load_sqlite()

    def save(self) -> None:
        """Save frontier to storage."""
        if self.storage == StorageBackend.JSON:
            self._save_json()
        else:
            self._save_sqlite()

    @property
    def frontier(self) -> list[dict[str, Any]]:
        """Return current frontier as list of config dicts (excludes dominated)."""
        active = [cfg for cfg in self._configs if cfg.config_id not in self._dominated_ids]
        return [cfg.to_dict() for cfg in active]

    @property
    def all_configs(self) -> list[dict[str, Any]]:
        """Return all configs including dominated (archive)."""
        return [cfg.to_dict() for cfg in self._configs]

    @property
    def size(self) -> int:
        """Number of active frontier configs."""
        return len([c for c in self._configs if c.config_id not in self._dominated_ids])

    def add_result(
        self,
        benchmark_result: Any,  # BenchmarkResult from evaluator.py
        harness_config: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UpdateResult:
        """
        Add a benchmark result to the frontier.

        Args:
            benchmark_result: BenchmarkResult from evaluator.py.
                              Must have aggregate_score, total_latency_ms, verdict_counts.
            harness_config: Dict of harness settings.
            weights: Optional weights dict. If None, uses BALANCED.
            run_id: Optional run identifier. Auto-generated if None.
            metadata: Optional extra data.

        Returns:
            UpdateResult with details of the update.
        """
        weights = weights or {"quality": 0.33, "cost": 0.33, "speed": 0.34}
        if harness_config is None:
            harness_config = {}

        # Extract metrics from BenchmarkResult
        metrics = self._extract_metrics(benchmark_result)

        new_config_dict = {
            "quality": metrics["quality"],
            "cost": metrics["cost"],
            "speed": metrics["speed"],
        }

        # Current active frontier as dicts
        active_dicts = [
            cfg.to_dict() for cfg in self._configs
            if cfg.config_id not in self._dominated_ids
        ]

        # Use raw metrics dicts for dominance check
        active_metric_dicts = [
            {"quality": c["metrics"]["quality"], "cost": c["metrics"]["cost"], "speed": c["metrics"]["speed"]}
            for c in active_dicts
        ]

        # Check if new config is dominated
        is_dominated = False
        for existing in active_metric_dicts:
            if dominance_check(existing, new_config_dict):
                is_dominated = True
                break

        # Find newly dominated configs
        newly_dominated_ids = []
        surviving_configs = []

        if not is_dominated:
            for cfg in self._configs:
                if cfg.config_id in self._dominated_ids:
                    continue
                existing_metrics = {
                    "quality": cfg.metrics["quality"],
                    "cost": cfg.metrics["cost"],
                    "speed": cfg.metrics["speed"],
                }
                if dominance_check(new_config_dict, existing_metrics):
                    newly_dominated_ids.append(cfg.config_id)
                else:
                    surviving_configs.append(cfg)

        if is_dominated:
            # Don't add — config is dominated
            return UpdateResult(
                config_id="",
                was_added=False,
                was_dominated=True,
                newly_dominated=[],
                frontier_size=len(surviving_configs) if surviving_configs else self.size,
                aaii=self._compute_frontier_aaii(),
            )

        # Create new FrontierConfig
        config_id = _generate_ulid()
        now = _rfc3339_now()
        run_id = run_id or _generate_ulid()

        fc = FrontierConfig(
            config_id=config_id,
            harness_config=harness_config,
            metrics=metrics,
            weights=weights,
            added_at=now,
            run_id=run_id,
            metadata=metadata or {},
        )

        # Remove newly dominated from active
        dominated_set = set(newly_dominated_ids)
        surviving = [cfg for cfg in self._configs if cfg.config_id not in dominated_set]
        surviving.append(fc)

        self._configs = surviving
        self._dominated_ids.update(newly_dominated_ids)

        if self.auto_save:
            self.save()

        return UpdateResult(
            config_id=config_id,
            was_added=True,
            was_dominated=False,
            newly_dominated=newly_dominated_ids,
            frontier_size=self.size,
            aaii=self._compute_frontier_aaii(),
        )

    def add_config(
        self,
        metrics: dict[str, float],
        harness_config: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UpdateResult:
        """
        Add a config directly with raw metrics dict.

        Args:
            metrics: Dict with 'quality', 'cost', 'speed' keys (all float, 0-1).
            harness_config: Dict of harness settings.
            weights: Optional weights dict.
            run_id: Optional run identifier.
            metadata: Optional extra data.
        """
        weights = weights or {"quality": 0.33, "cost": 0.33, "speed": 0.34}
        if harness_config is None:
            harness_config = {}

        new_config_dict = {
            "quality": metrics["quality"],
            "cost": metrics["cost"],
            "speed": metrics["speed"],
        }

        # Active frontier as metric dicts for dominance check
        active_metric_dicts = [
            {
                "quality": cfg.metrics["quality"],
                "cost": cfg.metrics["cost"],
                "speed": cfg.metrics["speed"],
            }
            for cfg in self._configs
            if cfg.config_id not in self._dominated_ids
        ]

        # Is the new config dominated by any existing point?
        is_dominated = False
        for existing in active_metric_dicts:
            if dominance_check(existing, new_config_dict):
                is_dominated = True
                break

        if is_dominated:
            return UpdateResult(
                config_id="",
                was_added=False,
                was_dominated=True,
                newly_dominated=[],
                frontier_size=self.size,
                aaii=self._compute_frontier_aaii(),
            )

        # Find newly dominated configs
        newly_dominated_ids: list[str] = []
        for cfg in self._configs:
            if cfg.config_id in self._dominated_ids:
                continue
            existing_metrics = {
                "quality": cfg.metrics["quality"],
                "cost": cfg.metrics["cost"],
                "speed": cfg.metrics["speed"],
            }
            if dominance_check(new_config_dict, existing_metrics):
                newly_dominated_ids.append(cfg.config_id)

        # Build the FrontierConfig with the metrics as provided
        config_id = _generate_ulid()
        now = _rfc3339_now()
        run_id = run_id or _generate_ulid()

        fc = FrontierConfig(
            config_id=config_id,
            harness_config=harness_config,
            metrics=dict(metrics),
            weights=weights,
            added_at=now,
            run_id=run_id,
            metadata=metadata or {},
        )

        # Keep dominated configs in _configs (archive), just mark them dominated.
        self._configs.append(fc)
        self._dominated_ids.update(newly_dominated_ids)

        if self.auto_save:
            self.save()

        return UpdateResult(
            config_id=config_id,
            was_added=True,
            was_dominated=False,
            newly_dominated=newly_dominated_ids,
            frontier_size=self.size,
            aaii=self._compute_frontier_aaii(),
        )

    def query(
        self,
        cost_budget: float | None = None,
        time_budget: float | None = None,
        min_quality: float | None = None,
        preset: WeightPreset | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query frontier for configs meeting budget and quality constraints.

        Args:
            cost_budget: Maximum cost in dollars.
            time_budget: Maximum time in seconds (maps to speed threshold).
            min_quality: Minimum quality score.
            preset: Optional weight preset to rank results.

        Returns:
            List of matching configs sorted by composite score.
        """
        active = [cfg for cfg in self._configs if cfg.config_id not in self._dominated_ids]
        # Build flat shadow dicts for filtering, paired with the full dict
        flat_pairs = [
            (
                {
                    "quality": cfg.metrics.get("quality", 0.0),
                    "cost": cfg.metrics.get("cost", 0.0),
                    "speed": cfg.metrics.get("speed", 0.0),
                },
                cfg.to_dict(),
            )
            for cfg in active
        ]

        flat_filtered = query_frontier(
            [flat for flat, _ in flat_pairs], cost_budget, time_budget, min_quality
        )

        # Map back to full dicts, preserving order from flat_filtered
        filtered_full = []
        for flat in flat_filtered:
            for f, full in flat_pairs:
                if f is flat:
                    filtered_full.append(full)
                    break

        if preset:
            # Rank uses _compute_weighted_score on flat dicts; rebuild pairing
            flat_filtered_pairs = [
                (
                    {
                        "quality": full["metrics"].get("quality", 0.0),
                        "cost": full["metrics"].get("cost", 0.0),
                        "speed": full["metrics"].get("speed", 0.0),
                    },
                    full,
                )
                for full in filtered_full
            ]
            ranked_flat = rank_frontier_by_preset(
                [flat for flat, _ in flat_filtered_pairs], preset
            )
            ranked_full = []
            for flat in ranked_flat:
                for f, full in flat_filtered_pairs:
                    if f is flat:
                        ranked_full.append(full)
                        break
            return ranked_full

        return filtered_full

    def rank(
        self,
        preset: WeightPreset = WeightPreset.BALANCED,
    ) -> list[dict[str, Any]]:
        """
        Rank all frontier configs by weight preset.

        Args:
            preset: Weight preset to use for ranking.

        Returns:
            List of configs sorted by weighted score.
        """
        active = [cfg for cfg in self._configs if cfg.config_id not in self._dominated_ids]
        flat_pairs = [
            (
                {
                    "quality": cfg.metrics.get("quality", 0.0),
                    "cost": cfg.metrics.get("cost", 0.0),
                    "speed": cfg.metrics.get("speed", 0.0),
                },
                cfg.to_dict(),
            )
            for cfg in active
        ]
        ranked_flat = rank_frontier_by_preset(
            [flat for flat, _ in flat_pairs], preset
        )
        ranked_full = []
        for flat in ranked_flat:
            for f, full in flat_pairs:
                if f is flat:
                    ranked_full.append(full)
                    break
        return ranked_full

    def aaii(self) -> float:
        """Compute AAII for current frontier."""
        return self._compute_frontier_aaii()

    def clear(self) -> None:
        """Clear all configs (both active and dominated archive)."""
        self._configs = []
        self._dominated_ids = set()
        if self.auto_save:
            self.save()

    def remove_config(self, config_id: str) -> bool:
        """Remove a config by ID."""
        original_len = len(self._configs)
        self._configs = [c for c in self._configs if c.config_id != config_id]
        if len(self._configs) < original_len:
            if self.auto_save:
                self.save()
            return True
        return False

    def get_config(self, config_id: str) -> dict[str, Any] | None:
        """Get a config by ID."""
        for cfg in self._configs:
            if cfg.config_id == config_id:
                return cfg.to_dict()
        return None

    # ---- Private helpers ----

    def _extract_metrics(self, result: Any) -> dict[str, float]:
        """
        Extract quality/cost/speed metrics from a BenchmarkResult.

        BenchmarkResult has:
            - aggregate_score (0-1) -> quality
            - total_latency_ms -> speed (normalize: 1.0 at 0ms, 0.0 at some max)
            - verdict_counts -> pass rate as quality supplement

        For cost: we use a placeholder normalization. Wire up to actual cost
        tracking when harness reports cost.
        """
        quality = getattr(result, "aggregate_score", 0.0)

        # Normalize latency to speed: 0ms = 1.0, 30s = 0.0
        latency_ms = getattr(result, "total_latency_ms", 0.0)
        max_latency_ms = 30000.0  # 30s
        speed = max(0.0, min(1.0, 1.0 - latency_ms / max_latency_ms))

        # Cost: placeholder. When harness reports actual cost, update this.
        # For now, use verdict distribution as quality proxy
        verdict_counts = getattr(result, "verdict_counts", {}) or {}
        total = sum(verdict_counts.values()) or 1
        ok_count = verdict_counts.get("OK", 0)
        # p_cost = 1 - (ok_count / total)  # more OK = more expensive (proxy)
        # Simpler: cost = 0.5 as default placeholder
        cost = 0.5

        return {
            "quality": quality,
            "cost": cost,
            "speed": speed,
        }

    def _compute_frontier_aaii(self) -> float:
        """Compute AAII for current active frontier."""
        active = [cfg for cfg in self._configs if cfg.config_id not in self._dominated_ids]
        metric_dicts = [
            {"quality": c.metrics["quality"], "cost": c.metrics["cost"], "speed": c.metrics["speed"]}
            for c in active
        ]
        return compute_aaii(metric_dicts)

    # ---- JSON persistence ----

    def _load_json(self) -> None:
        """Load from JSON file."""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self._configs = [FrontierConfig.from_dict(c) for c in data.get("configs", [])]
            self._dominated_ids = set(data.get("dominated_ids", []))
        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted file — start fresh
            self._configs = []
            self._dominated_ids = set()

    def _save_json(self) -> None:
        """Save to JSON file."""
        data = {
            "configs": [cfg.to_dict() for cfg in self._configs],
            "dominated_ids": list(self._dominated_ids),
            "saved_at": _rfc3339_now(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    # ---- SQLite persistence ----

    def _load_sqlite(self) -> None:
        """Load from SQLite database."""
        if not self.path.exists():
            return
        try:
            conn = sqlite3.connect(str(self.path))
            conn.row_factory = sqlite3.Row

            # Load configs
            rows = conn.execute("SELECT * FROM configs").fetchall()
            self._configs = []
            for row in rows:
                self._configs.append(FrontierConfig(
                    config_id=row["config_id"],
                    harness_config=json.loads(row["harness_config"]),
                    metrics=json.loads(row["metrics"]),
                    weights=json.loads(row["weights"]),
                    added_at=row["added_at"],
                    run_id=row["run_id"],
                    metadata=json.loads(row["metadata"] or "{}"),
                ))

            # Load dominated IDs
            dominated_rows = conn.execute("SELECT config_id FROM dominated").fetchall()
            self._dominated_ids = set(row["config_id"] for row in dominated_rows)

            conn.close()
        except sqlite3.OperationalError:
            self._configs = []
            self._dominated_ids = set()

    def _save_sqlite(self) -> None:
        """Save to SQLite database."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row

        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS configs (
                config_id TEXT PRIMARY KEY,
                harness_config TEXT NOT NULL,
                metrics TEXT NOT NULL,
                weights TEXT NOT NULL,
                added_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dominated (
                config_id TEXT PRIMARY KEY,
                dominated_at TEXT NOT NULL
            )
        """)

        # Clear and replace configs
        conn.execute("DELETE FROM configs")
        for cfg in self._configs:
            conn.execute(
                """INSERT INTO configs
                   (config_id, harness_config, metrics, weights, added_at, run_id, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    cfg.config_id,
                    json.dumps(cfg.harness_config),
                    json.dumps(cfg.metrics),
                    json.dumps(cfg.weights),
                    cfg.added_at,
                    cfg.run_id,
                    json.dumps(cfg.metadata),
                )
            )

        # Clear and replace dominated
        conn.execute("DELETE FROM dominated")
        for cid in self._dominated_ids:
            conn.execute(
                "INSERT INTO dominated (config_id, dominated_at) VALUES (?, ?)",
                (cid, _rfc3339_now()),
            )

        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Integration with BenchmarkResult
# ---------------------------------------------------------------------------

def benchmark_result_to_metrics(result: Any) -> dict[str, float]:
    """
    Convert a BenchmarkResult to quality/cost/speed metrics dict.

    Args:
        result: BenchmarkResult from evaluator.py

    Returns:
        Dict with 'quality', 'cost', 'speed' keys (all float 0-1).
    """
    quality = getattr(result, "aggregate_score", 0.0)

    latency_ms = getattr(result, "total_latency_ms", 0.0)
    max_latency_ms = 30000.0
    speed = max(0.0, min(1.0, 1.0 - latency_ms / max_latency_ms))

    verdict_counts = getattr(result, "verdict_counts", {}) or {}
    total = sum(verdict_counts.values()) or 1
    ok_count = verdict_counts.get("OK", 0)

    # Cost proxy: pass rate indicator
    # In a real system, cost would come from harness reporting
    cost = 0.5

    return {
        "quality": quality,
        "cost": cost,
        "speed": speed,
    }


def frontier_config_from_benchmark_result(
    result: Any,
    harness_config: dict[str, Any],
    weights: dict[str, float] | None = None,
    run_id: str | None = None,
) -> FrontierConfig:
    """
    Create a FrontierConfig from a BenchmarkResult and HarnessConfig.

    Args:
        result: BenchmarkResult from evaluator.py
        harness_config: HarnessConfig (or dict representation)
        weights: Optional weights dict.
        run_id: Optional run identifier.

    Returns:
        FrontierConfig ready to be added to a ParetoFrontier.
    """
    if hasattr(harness_config, "to_dict"):
        hc = harness_config.to_dict()
    else:
        hc = dict(harness_config)

    metrics = benchmark_result_to_metrics(result)
    weights = weights or {"quality": 0.33, "cost": 0.33, "speed": 0.34}

    return FrontierConfig(
        config_id=_generate_ulid(),
        harness_config=hc,
        metrics=metrics,
        weights=weights,
        added_at=_rfc3339_now(),
        run_id=run_id or _generate_ulid(),
        metadata={},
    )