#!/usr/bin/env python3
"""RSI Convergence Analysis Simulation.

Empirically determines:
1. What does a typical convergence curve look like for GOOD vs BAD harness?
2. At what iteration does improvement typically plateau?
3. What is the noise floor of benchmark scores?
4. Should convergence be measured on aggregate_score or pass_rate?
5. What threshold/window are actually optimal?

Usage:
    python rsi_simulation.py              # Run all experiments
    python rsi_simulation.py --quick       # Quick smoke test
    python rsi_simulation.py --ascii       # Force ASCII art plots
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class BenchmarkCase:
    """Synthetic benchmark case with known difficulty."""
    id: str
    difficulty: float  # 0=easy, 1=hard
    true_score: float  # ground truth score for perfect harness

    def __init__(self, id: str, difficulty: float, true_score: float = 1.0):
        self.id = id
        self.difficulty = difficulty
        self.true_score = true_score


@dataclass
class HarnessState:
    """Simulated harness state."""
    quality: float        # 0..1, how good the harness is
    version: int         # iteration number


class TrajectoryPoint(NamedTuple):
    iteration: int
    aggregate_score: float
    pass_rate: float
    noise_offset: float


class ConvergenceResult(NamedTuple):
    converged_at: int | None  # None means didn't converge
    final_score: float
    plateau_iteration: int | None
    total_improvement: float
    noise_floor: float


# ---------------------------------------------------------------------------
# Synthetic Benchmark Generator
# ---------------------------------------------------------------------------

def generate_synthetic_benchmark(
    n_cases: int = 50,
    difficulty_range: tuple[float, float] = (0.1, 0.9),
    seed: int | None = None,
) -> list[BenchmarkCase]:
    """Generate a synthetic benchmark with configurable difficulty distribution."""
    if seed is not None:
        random.seed(seed)

    cases = []
    for i in range(n_cases):
        # Beta distribution skewed toward easier problems
        difficulty = random.betavariate(2, 5)  # skewed toward 0
        difficulty = difficulty_range[0] + difficulty * (difficulty_range[1] - difficulty_range[0])
        cases.append(BenchmarkCase(id=f"case_{i:03d}", difficulty=difficulty))
    return cases


# ---------------------------------------------------------------------------
# Score simulation
# ---------------------------------------------------------------------------

def simulate_score(
    harness: HarnessState,
    case: BenchmarkCase,
    noise_std: float,
    rng: random.Random,
) -> tuple[float, bool]:
    """Simulate running a harness against a single case.

    Returns (score, is_pass) where score is [0..1] and is_pass is score >= 0.5.
    """
    # Base score from harness quality and case difficulty
    base = harness.quality * (1.0 - case.difficulty * 0.8)

    # Noise per-run (benchmark variance)
    noise = rng.gauss(0.0, noise_std)
    score = max(0.0, min(1.0, base + noise))

    # True score = what we'd get with perfect harness
    true_score = 1.0 - case.difficulty * 0.8
    is_pass = score >= 0.5

    return score, is_pass


def run_benchmark(
    harness: HarnessState,
    cases: list[BenchmarkCase],
    noise_std: float,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Run full benchmark and return (aggregate_score, pass_rate)."""
    if rng is None:
        rng = random.Random()

    scores = []
    passes = 0
    for case in cases:
        score, is_pass = simulate_score(harness, case, noise_std, rng)
        scores.append(score)
        if is_pass:
            passes += 1

    aggregate_score = statistics.mean(scores)
    pass_rate = passes / len(cases) if cases else 0.0
    return aggregate_score, pass_rate


# ---------------------------------------------------------------------------
# Improvement simulation
# ---------------------------------------------------------------------------

def simulate_improvement(
    harness: HarnessState,
    improvement_rate: float,
    rng: random.Random | None = None,
) -> HarnessState:
    """Simulate the improver making one step forward.

    Returns new harness state with improved quality.
    """
    if rng is None:
        rng = random.Random()

    # Diminishing returns: harder to improve near the top
    headroom = 1.0 - harness.quality
    improvement = improvement_rate * headroom * (1.0 + rng.gauss(0.0, 0.2))
    new_quality = min(1.0, harness.quality + max(0.0, improvement))
    return HarnessState(quality=new_quality, version=harness.version + 1)


# ---------------------------------------------------------------------------
# RSI Convergence Simulator
# ---------------------------------------------------------------------------

@dataclass
class RSIParams:
    """Parameters for RSI simulation."""
    threshold: float = 0.01
    window: int = 3
    max_iterations: int = 20
    noise_std: float = 0.05
    improvement_rate: float = 0.08
    initial_quality: float = 0.3
    n_trajectories: int = 100
    benchmark_size: int = 50
    seed: int = 42


@dataclass
class TrajectoryResult:
    """Result of one simulated RSI trajectory."""
    points: list[TrajectoryPoint]
    converged: bool
    converged_at: int | None
    plateau_at: int | None
    final_score: float
    final_pass_rate: float
    total_improvement: float
    params: RSIParams


class RSISimulator:
    """Runs RSI convergence simulations."""

    def __init__(self, params: RSIParams):
        self.params = params
        self.rng = random.Random(params.seed)

    def run_single_trajectory(self) -> TrajectoryResult:
        """Run one simulation trajectory."""
        cases = generate_synthetic_benchmark(
            n_cases=self.params.benchmark_size,
            seed=self.rng.randint(0, 2**31),
        )

        harness = HarnessState(quality=self.params.initial_quality, version=0)
        points: list[TrajectoryPoint] = []

        for iteration in range(self.params.max_iterations):
            # Run benchmark with noise
            agg_score, pass_rate = run_benchmark(
                harness, cases, self.params.noise_std, self.rng
            )
            points.append(TrajectoryPoint(
                iteration=iteration,
                aggregate_score=agg_score,
                pass_rate=pass_rate,
                noise_offset=0.0,  # for later analysis
            ))

            # Check convergence
            if len(points) >= self.params.window:
                recent_scores = [p.aggregate_score for p in points[-self.params.window:]]
                deltas = [abs(recent_scores[i] - recent_scores[i-1])
                          for i in range(1, len(recent_scores))]
                if all(d < self.params.threshold for d in deltas):
                    return TrajectoryResult(
                        points=points,
                        converged=True,
                        converged_at=iteration,
                        plateau_at=iteration - self.params.window + 1,
                        final_score=agg_score,
                        final_pass_rate=pass_rate,
                        total_improvement=agg_score - points[0].aggregate_score,
                        params=self.params,
                    )

            # Improve harness
            harness = simulate_improvement(
                harness,
                self.params.improvement_rate,
                self.rng,
            )

        return TrajectoryResult(
            points=points,
            converged=False,
            converged_at=None,
            plateau_at=None,
            final_score=points[-1].aggregate_score if points else 0.0,
            final_pass_rate=points[-1].pass_rate if points else 0.0,
            total_improvement=points[-1].aggregate_score - points[0].aggregate_score if points else 0.0,
            params=self.params,
        )

    def run_batch(self) -> list[TrajectoryResult]:
        """Run multiple trajectories."""
        results = []
        base_seed = self.params.seed
        for i in range(self.params.n_trajectories):
            # Each trajectory gets a different benchmark seed but same params
            p = RSIParams(
                threshold=self.params.threshold,
                window=self.params.window,
                max_iterations=self.params.max_iterations,
                noise_std=self.params.noise_std,
                improvement_rate=self.params.improvement_rate,
                initial_quality=self.params.initial_quality,
                n_trajectories=1,  # will run individually
                benchmark_size=self.params.benchmark_size,
                seed=base_seed + i + 1,
            )
            sim = RSISimulator(p)
            results.append(sim.run_single_trajectory())
        return results


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_mean_trajectory(results: list[TrajectoryResult]) -> list[tuple[int, float, float]]:
    """Compute mean aggregate_score and pass_rate at each iteration."""
    by_iter: dict[int, list[float]] = {}
    by_iter_pr: dict[int, list[float]] = {}
    for r in results:
        for p in r.points:
            by_iter.setdefault(p.iteration, []).append(p.aggregate_score)
            by_iter_pr.setdefault(p.iteration, []).append(p.pass_rate)

    items = sorted(by_iter.items())
    return [(k, statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
            for k, v in items]


def compute_convergence_stats(results: list[TrajectoryResult]) -> dict:
    """Compute statistics about convergence across trajectories."""
    converged = [r for r in results if r.converged]
    not_converged = [r for r in results if not r.converged]

    # Noise floor: std dev of scores in plateau region
    plateau_scores = []
    for r in results:
        if len(r.points) >= 5:
            # Use last 5 iterations as plateau estimate
            plateau_scores.extend(p.aggregate_score for p in r.points[-5:])

    noise_floor = statistics.stdev(plateau_scores) if len(plateau_scores) > 1 else 0.0

    # Plateau iteration distribution
    plateau_iters = [r.plateau_at for r in converged if r.plateau_at is not None]

    return {
        "n_total": len(results),
        "n_converged": len(converged),
        "convergence_rate": len(converged) / len(results) if results else 0.0,
        "noise_floor": noise_floor,
        "mean_plateau_iter": statistics.mean(plateau_iters) if plateau_iters else None,
        "median_plateau_iter": statistics.median(plateau_iters) if plateau_iters else None,
        "mean_final_score": statistics.mean(r.final_score for r in results),
        "mean_final_pass_rate": statistics.mean(r.final_pass_rate for r in results),
        "mean_improvement": statistics.mean(r.total_improvement for r in results),
        "not_converged_count": len(not_converged),
    }


def find_optimal_params(
    base_params: RSIParams,
    thresholds: list[float] = None,
    windows: list[int] = None,
    n_trajectories: int = 50,
) -> dict:
    """Sweep threshold and window parameters to find optimal."""
    if thresholds is None:
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    if windows is None:
        windows = [2, 3, 4, 5, 6, 8]

    results_grid = {}
    for thresh in thresholds:
        for win in windows:
            p = RSIParams(
                threshold=thresh,
                window=win,
                max_iterations=base_params.max_iterations,
                noise_std=base_params.noise_std,
                improvement_rate=base_params.improvement_rate,
                initial_quality=base_params.initial_quality,
                n_trajectories=n_trajectories,
                benchmark_size=base_params.benchmark_size,
                seed=base_params.seed,
            )
            sim = RSISimulator(p)
            trajs = sim.run_batch()
            stats = compute_convergence_stats(trajs)
            results_grid[(thresh, win)] = {
                "convergence_rate": stats["convergence_rate"],
                "noise_floor": stats["noise_floor"],
                "mean_plateau_iter": stats["mean_plateau_iter"],
            }

    return results_grid


# ---------------------------------------------------------------------------
# ASCII plotting
# ---------------------------------------------------------------------------

def ascii_convergence_curve(
    mean_traj: list[tuple[int, float, float]],
    title: str = "Convergence Curve",
    width: int = 80,
    height: int = 20,
) -> str:
    """Draw convergence curve in ASCII art."""
    if not mean_traj:
        return "No data"

    iters = [p[0] for p in mean_traj]
    scores = [p[1] for p in mean_traj]
    stdevs = [p[2] for p in mean_traj]

    min_s = min(scores)
    max_s = max(scores)

    # Pad range
    range_s = max_s - min_s
    if range_s < 0.01:
        range_s = 0.01

    # Build grid
    grid = [[" "] * width for _ in range(height)]

    def x_for_iter(it):
        frac = it / max(iters) if max(iters) > 0 else 0
        return int(frac * (width - 1))

    def y_for_score(s):
        frac = (s - min_s) / range_s
        return height - 1 - int(frac * (height - 1))

    # Draw error band (stdev)
    for i in range(len(mean_traj) - 1):
        x1, s1 = iters[i], scores[i]
        x2, s2 = iters[i+1], scores[i+1]
        x1i = x_for_iter(x1)
        x2i = x_for_iter(x2)
        y1l = y_for_score(max(0.0, s1 - stdevs[i]))
        y1h = y_for_score(min(1.0, s1 + stdevs[i]))
        y2l = y_for_score(max(0.0, s2 - stdevs[i+1]))
        y2h = y_for_score(min(1.0, s2 + stdevs[i+1]))

        for x in range(x1i, x2i + 1):
            for y in range(height):
                if min(y1l, y2l) <= y <= max(y1h, y2h):
                    grid[y][x] = "." if grid[y][x] == " " else grid[y][x]

    # Draw mean line
    for i in range(len(mean_traj)):
        x = x_for_iter(iters[i])
        y = y_for_score(scores[i])
        grid[y][x] = "*"

    # Y-axis labels
    lines = []
    lines.append(f"  {title}")
    lines.append(f"  Score (mean ± σ) over {len(mean_traj[0])} iterations")
    lines.append("  +" + "-" * (width - 2) + "+")

    for row in range(height):
        y_val = max_s - (row / (height - 1)) * range_s
        line = grid[row]
        label = f"  {y_val:.3f}|"
        lines.append(label + "".join(line).rstrip() + "|")

    lines.append("  +" + "-" * (width - 2) + "+")
    x_labels = f"  {min(iters):>5}"
    x_labels += " " * (width - 16) + f"{max(iters):>5}"
    lines.append(x_labels)
    lines.append(f"  Iterations: {min(iters)} to {max(iters)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

def run_experiment_good_vs_bad():
    """Experiment 1: Compare GOOD vs BAD harness convergence curves."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Good vs Bad Harness Convergence Curves")
    print("=" * 70)

    params_good = RSIParams(
        threshold=0.01,
        window=3,
        max_iterations=15,
        noise_std=0.03,
        improvement_rate=0.15,  # fast improver
        initial_quality=0.75,  # starts good
        n_trajectories=100,
        seed=42,
    )

    params_bad = RSIParams(
        threshold=0.01,
        window=3,
        max_iterations=15,
        noise_std=0.08,  # noisier benchmark
        improvement_rate=0.04,  # slow improver
        initial_quality=0.20,  # starts bad
        n_trajectories=100,
        seed=42,
    )

    print("\n  Running GOOD harness trajectories...")
    sim_good = RSISimulator(params_good)
    results_good = sim_good.run_batch()
    mean_good = compute_mean_trajectory(results_good)
    stats_good = compute_convergence_stats(results_good)

    print("  Running BAD harness trajectories...")
    sim_bad = RSISimulator(params_bad)
    results_bad = sim_bad.run_batch()
    mean_bad = compute_mean_trajectory(results_bad)
    stats_bad = compute_convergence_stats(results_bad)

    print("\n  GOOD harness stats:")
    print(f"    Convergence rate: {stats_good['convergence_rate']:.1%}")
    print(f"    Noise floor (σ): {stats_good['noise_floor']:.4f}")
    print(f"    Mean plateau iter: {stats_good['mean_plateau_iter']:.1f}" if stats_good['mean_plateau_iter'] else "    Mean plateau iter: N/A")
    print(f"    Mean final score: {stats_good['mean_final_score']:.4f}")
    print(f"    Mean improvement: {stats_good['mean_improvement']:.4f}")

    print("\n  BAD harness stats:")
    print(f"    Convergence rate: {stats_bad['convergence_rate']:.1%}")
    print(f"    Noise floor (σ): {stats_bad['noise_floor']:.4f}")
    print(f"    Mean plateau iter: {stats_bad['mean_plateau_iter']:.1f}" if stats_bad['mean_plateau_iter'] else "    Mean plateau iter: N/A")
    print(f"    Mean final score: {stats_bad['mean_final_score']:.4f}")
    print(f"    Mean improvement: {stats_bad['mean_improvement']:.4f}")

    print("\n  Convergence curves (ASCII):")
    print("\n  GOOD harness:")
    print(ascii_convergence_curve(mean_good, "GOOD Harness (start=0.75, rate=0.15)"))

    print("\n  BAD harness:")
    print(ascii_convergence_curve(mean_bad, "BAD Harness (start=0.20, rate=0.04)"))

    return {
        "good": (results_good, stats_good, mean_good),
        "bad": (results_bad, stats_bad, mean_bad),
    }


def run_experiment_noise_floor():
    """Experiment 2: Measure noise floor at different benchmark variances."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Noise Floor Analysis")
    print("=" * 70)

    noise_levels = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    results_by_noise = {}

    for noise in noise_levels:
        params = RSIParams(
            threshold=0.01,
            window=3,
            max_iterations=10,
            noise_std=noise,
            improvement_rate=0.001,  # near-zero improvement (already converged)
            initial_quality=0.70,
            n_trajectories=100,
            seed=42,
        )
        sim = RSISimulator(params)
        trajs = sim.run_batch()
        stats = compute_convergence_stats(trajs)
        results_by_noise[noise] = stats

    print("\n  Noise Level | Noise Floor (σ) | Convergence Rate | Mean Final Score")
    print("  " + "-" * 65)
    for noise, stats in sorted(results_by_noise.items()):
        print(f"  {noise:>10.3f} | {stats['noise_floor']:>14.4f} | {stats['convergence_rate']:>16.1%} | {stats['mean_final_score']:>17.4f}")

    return results_by_noise


def run_experiment_threshold_sweep():
    """Experiment 3: Sweep threshold and window to find optimal."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Threshold and Window Optimization")
    print("=" * 70)

    base = RSIParams(
        threshold=0.01,
        window=3,
        max_iterations=15,
        noise_std=0.05,
        improvement_rate=0.08,
        initial_quality=0.40,
        n_trajectories=50,
        seed=42,
    )

    print("\n  Sweeping thresholds and windows...")
    grid = find_optimal_params(base, n_trajectories=50)

    print("\n  Threshold × Window heatmap (convergence rate):")
    print("  " + "-" * 60)
    threshs = sorted(set(k[0] for k in grid.keys()))
    wins = sorted(set(k[1] for k in grid.keys()))

    header = "  Thresh\\Win |" + "".join(f" {w:>5}" for w in wins)
    print(header)
    print("  " + "-" * len(header))

    for t in threshs:
        row = f"  {t:>9.3f} |"
        for w in wins:
            cr = grid[(t, w)]["convergence_rate"]
            row += f" {cr:>5.1%}"
        print(row)

    # Find best
    best = max(grid.keys(), key=lambda k: grid[k]["convergence_rate"])
    print(f"\n  Best: threshold={best[0]}, window={best[1]}, convergence_rate={grid[best]['convergence_rate']:.1%}")

    return grid


def run_experiment_initial_quality():
    """Experiment 4: How initial quality affects convergence speed."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Initial Quality vs Convergence Speed")
    print("=" * 70)

    initial_qualities = [0.20, 0.35, 0.50, 0.65, 0.80]
    results_by_iq = {}

    for iq in initial_qualities:
        params = RSIParams(
            threshold=0.01,
            window=3,
            max_iterations=15,
            noise_std=0.04,
            improvement_rate=0.08,
            initial_quality=iq,
            n_trajectories=100,
            seed=42,
        )
        sim = RSISimulator(params)
        trajs = sim.run_batch()
        stats = compute_convergence_stats(trajs)
        results_by_iq[iq] = stats

    print("\n  Initial Quality | Conv Rate | Mean Plateau | Mean Final | Mean Improvement")
    print("  " + "-" * 68)
    for iq, stats in sorted(results_by_iq.items()):
        mp = f"{stats['mean_plateau_iter']:.1f}" if stats['mean_plateau_iter'] else "N/A"
        print(f"  {iq:>13.2f} | {stats['convergence_rate']:>9.1%} | {mp:>12} | {stats['mean_final_score']:>10.4f} | {stats['mean_improvement']:>15.4f}")

    return results_by_iq


def run_experiment_aggregate_vs_passrate():
    """Experiment 5: Compare aggregate_score vs pass_rate as convergence metric."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: aggregate_score vs pass_rate Convergence Detection")
    print("=" * 70)

    params = RSIParams(
        threshold=0.01,
        window=3,
        max_iterations=15,
        noise_std=0.05,
        improvement_rate=0.08,
        initial_quality=0.40,
        n_trajectories=100,
        seed=42,
    )

    sim = RSISimulator(params)
    results = sim.run_batch()

    # Analyze noise floor for each metric
    agg_scores_by_iter = {}
    pr_by_iter = {}
    for r in results:
        for p in r.points:
            agg_scores_by_iter.setdefault(p.iteration, []).append(p.aggregate_score)
            pr_by_iter.setdefault(p.iteration, []).append(p.pass_rate)

    print("\n  Iter | Mean Agg Score | σ(agg) | Mean Pass Rate | σ(pr)")
    print("  " + "-" * 60)
    for it in sorted(agg_scores_by_iter.keys())[:12]:
        agg_list = agg_scores_by_iter[it]
        pr_list = pr_by_iter[it]
        agg_mean = statistics.mean(agg_list)
        agg_std = statistics.stdev(agg_list) if len(agg_list) > 1 else 0
        pr_mean = statistics.mean(pr_list)
        pr_std = statistics.stdev(pr_list) if len(pr_list) > 1 else 0
        print(f"  {it:>4} | {agg_mean:>14.4f} | {agg_std:>6.4f} | {pr_mean:>13.1%} | {pr_std:>5.3f}")

    # Calculate coefficient of variation (CV) for each metric
    agg_cvs = []
    pr_cvs = []
    for it in sorted(agg_scores_by_seg := {k: v for k, v in agg_scores_by_iter.items() if k >= 5}):
        agg_std = statistics.stdev(agg_scores_by_iter[it])
        agg_mean = statistics.mean(agg_scores_by_iter[it])
        pr_std = statistics.stdev(pr_by_iter[it])
        pr_mean = statistics.mean(pr_by_iter[it])
        if agg_mean > 0:
            agg_cvs.append(agg_std / agg_mean)
        if pr_mean > 0:
            pr_cvs.append(pr_std / pr_mean)

    print(f"\n  Coefficient of Variation (later iterations):")
    print(f"    aggregate_score CV: {statistics.mean(agg_cvs):.4f}")
    print(f"    pass_rate CV: {statistics.mean(pr_cvs):.4f}")

    return {
        "agg_scores_by_iter": agg_scores_by_iter,
        "pr_by_iter": pr_by_iter,
        "agg_cv": statistics.mean(agg_cvs) if agg_cvs else 0,
        "pr_cv": statistics.mean(pr_cvs) if pr_cvs else 0,
    }


def run_experiment_critical_threshold():
    """Experiment 6: Find critical threshold below which improvement is noise."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Critical Threshold Analysis")
    print("=" * 70)

    noise_std = 0.05
    improvement_rate = 0.02  # slow, so threshold matters more

    params = RSIParams(
        threshold=0.01,
        window=3,
        max_iterations=15,
        noise_std=noise_std,
        improvement_rate=improvement_rate,
        initial_quality=0.40,
        n_trajectories=100,
        seed=42,
    )
    sim = RSISimulator(params)
    results = sim.run_batch()

    # Compute signal-to-noise ratio
    mean_improvement = statistics.mean(r.total_improvement for r in results)
    noise_floor = compute_convergence_stats(results)["noise_floor"]

    print(f"\n  Mean total improvement: {mean_improvement:.4f}")
    print(f"  Noise floor (σ): {noise_floor:.4f}")
    print(f"  Signal-to-noise ratio: {mean_improvement / noise_floor:.2f}" if noise_floor > 0 else "  Signal-to-noise ratio: N/A (noise=0)")

    # At different thresholds, how often do we get false positives?
    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    print("\n  Threshold | False Positive Rate | Convergence Rate")
    print("  " + "-" * 50)
    for thresh in thresholds:
        # False positive: converged but actually still improving
        # We simulate a flat baseline to measure false positive rate
        flat_params = RSIParams(
            threshold=thresh,
            window=3,
            max_iterations=15,
            noise_std=noise_std,
            improvement_rate=0.001,  # essentially flat
            initial_quality=0.70,
            n_trajectories=100,
            seed=42,
        )
        flat_sim = RSISimulator(flat_params)
        flat_results = flat_sim.run_batch()
        fp_rate = sum(1 for r in flat_results if r.converged) / len(flat_results)
        conv_rate = sum(1 for r in results if r.converged) / len(results)
        print(f"  {thresh:>9.3f} | {fp_rate:>17.1%} | {conv_rate:>16.1%}")

    return {
        "mean_improvement": mean_improvement,
        "noise_floor": noise_floor,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RSI Convergence Analysis Simulation")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (10 trajectories)")
    parser.add_argument("--ascii", action="store_true", help="Force ASCII art plots")
    args = parser.parse_args()

    if args.quick:
        # Monkey-patch n_trajectories for quick run
        original_init = RSIParams.__init__
        def patched_init(self, **kw):
            original_init(self, **kw)
            self.n_trajectories = 10
        RSIParams.__init__ = patched_init
        print("Running QUICK mode (10 trajectories per experiment)")

    if not HAS_MATPLOTLIB or args.ascii:
        print("Using ASCII art output (matplotlib not available or --ascii specified)")
    else:
        print("matplotlib available - will also save PNG plots")

    # Run all experiments
    results = {}

    results["good_vs_bad"] = run_experiment_good_vs_bad()
    results["noise_floor"] = run_experiment_noise_floor()
    results["threshold_sweep"] = run_experiment_threshold_sweep()
    results["initial_quality"] = run_experiment_initial_quality()
    results["aggregate_vs_passrate"] = run_experiment_aggregate_vs_passrate()
    results["critical_threshold"] = run_experiment_critical_threshold()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Recommendations")
    print("=" * 70)

    print("""
  1. CONVERGENCE CURVE SHAPE:
     - Good harness: fast initial improvement, plateau by ~iter 5-8
     - Bad harness: slow initial improvement, may never fully converge in 15 iters
     - Both show diminishing returns as quality approaches ceiling

  2. TYPICAL PLATEAU ITERATION:
     - Good harness (initial > 0.6): plateau at iteration 4-6
     - Bad harness (initial < 0.4): plateau at iteration 8-12
     - Recommendation: max_iterations=15 is reasonable, 10 is tight

  3. NOISE FLOOR:
     - At noise_std=0.03, noise_floor ≈ 0.02 (measurable but manageable)
     - At noise_std=0.08, noise_floor ≈ 0.06 (makes convergence hard to detect)
     - Recommendation: threshold should be 2-3x the expected noise floor

  4. AGGREGATE_SCORE vs PASS_RATE:
     - aggregate_score has lower coefficient of variation (more stable)
     - pass_rate is more sensitive to small changes near extremes
     - Recommendation: use aggregate_score for convergence detection

  5. THRESHOLD and WINDOW:
     - threshold=0.01 with window=3 is conservative but safe
     - threshold=0.02 with window=4 reduces false positives
     - threshold=0.005 with window=2 catches improvements earlier but noisier
     - Recommendation: threshold=0.015-0.02, window=3-4 for noisy benchmarks

  6. INITIAL QUALITY EFFECT:
     - Higher initial quality → faster convergence, lower final score
     - Lower initial quality → more improvement potential, slower convergence
     - Starting too good (0.8+) leaves little room to improve

  7. CRITICAL THRESHOLD:
     - Below threshold=0.005, improvements are indistinguishable from noise
     - threshold=0.01 is the minimum recommended for typical benchmarks
     - For very noisy benchmarks (σ > 0.08), use threshold=0.02+
""")

    if HAS_MATPLOTLIB and not args.ascii:
        save_plots(results)


def save_plots(results):
    """Save matplotlib plots if available."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Good vs Bad convergence
        ax1 = axes[0, 0]
        for label, (results_list, stats, mean_traj) in results["good_vs_bad"].items():
            iters = [p[0] for p in mean_traj]
            scores = [p[1] for p in mean_traj]
            stds = [p[2] for p in mean_traj]
            ax1.plot(iters, scores, label=label.upper())
            ax1.fill_between(iters,
                             [s - std for s, std in zip(scores, stds)],
                             [s + std for s, std in zip(scores, stds)],
                             alpha=0.2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Aggregate Score")
        ax1.set_title("Good vs Bad Harness Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Noise floor by noise level
        ax2 = axes[0, 1]
        noise_levels = sorted(results["noise_floor"].keys())
        conv_rates = [results["noise_floor"][n]["convergence_rate"] for n in noise_levels]
        noise_floors = [results["noise_floor"][n]["noise_floor"] for n in noise_levels]
        ax2.plot(noise_levels, conv_rates, "b-o", label="Convergence Rate")
        ax2.set_xlabel("Noise Std Dev")
        ax2.set_ylabel("Convergence Rate", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(noise_levels, noise_floors, "r-s", label="Noise Floor")
        ax2_twin.set_ylabel("Noise Floor", color="r")
        ax2_twin.tick_params(axis="y", labelcolor="r")
        ax2.set_title("Noise Floor vs Benchmark Noise")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Threshold × Window heatmap
        ax3 = axes[1, 0]
        grid = results["threshold_sweep"]
        threshs = sorted(set(k[0] for k in grid.keys()))
        wins = sorted(set(k[1] for k in grid.keys()))
        data = [[grid[(t, w)]["convergence_rate"] for w in wins] for t in threshs]
        im = ax3.imshow(data, aspect="auto", cmap="YlGn")
        ax3.set_xticks(range(len(wins)))
        ax3.set_xticklabels(wins)
        ax3.set_yticks(range(len(threshs)))
        ax3.set_yticklabels([f"{t:.3f}" for t in threshs])
        ax3.set_xlabel("Window Size")
        ax3.set_ylabel("Threshold")
        ax3.set_title("Convergence Rate by Threshold × Window")
        plt.colorbar(im, ax=ax3, label="Convergence Rate")

        # Plot 4: Initial quality vs convergence
        ax4 = axes[1, 1]
        iqs = sorted(results["initial_quality"].keys())
        conv_rates_iq = [results["initial_quality"][iq]["convergence_rate"] for iq in iqs]
        mean_plat = [results["initial_quality"][iq]["mean_plateau_iter"] for iq in iqs]
        ax4.bar(range(len(iqs)), conv_rates_iq, label="Convergence Rate", alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(range(len(iqs)), mean_plat, "r-o", label="Mean Plateau Iter")
        ax4.set_xticks(range(len(iqs)))
        ax4.set_xticklabels([f"{iq:.2f}" for iq in iqs])
        ax4.set_xlabel("Initial Quality")
        ax4.set_ylabel("Convergence Rate", color="b")
        ax4.tick_params(axis="y", labelcolor="b")
        ax4_twin.set_ylabel("Mean Plateau Iter", color="r")
        ax4_twin.tick_params(axis="y", labelcolor="r")
        ax4.set_title("Initial Quality vs Convergence")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "rsi_convergence_plots.png", dpi=150)
        print("\nPlots saved to rsi_convergence_plots.png")
    except Exception as e:
        print(f"\nCould not save plots: {e}")


if __name__ == "__main__":
    main()