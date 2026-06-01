"""Recursive Self-Improvement (RSI) subpackage.

Public surface (root-level ``autobench.rsi_loop`` etc. are deprecated
back-compat shims):

  - :class:`SelfImprovingHarness`, :class:`ImprovementDelta`,
    :func:`improve_harness`, :func:`convergence_check` (from ``loop``)
  - :class:`RSISimulator` + experiment runners (from ``simulation``)
  - :class:`AdversarialGenerator`, :class:`AdversarialCase` (from ``adversarial``)
  - :class:`PopulationRunner`, :class:`PopulationResult`,
    :func:`select_promotion_candidate` (from ``population``)
  - :class:`ReplayLoader`, :class:`ReplayComparison`,
    :class:`CounterfactualRunner` (from ``replay``)
"""

from __future__ import annotations

from .adversarial import (
    AdversarialCase,
    AdversarialGenerator,
    AdversarialRoundResult,
    generate_adversarial_case_mix,
    mine_failure_modes_from_result,
)
from .loop import (
    DEFAULT_VARIANCE_FLOOR_2SIGMA,
    ImprovementDelta,
    SelfImprovingHarness,
    convergence_check,
    improve_harness,
)
from .population import (
    AdvocateResult,
    PopulationResult,
    PopulationRunner,
    select_promotion_candidate,
)
from .replay import (
    CounterfactualRunner,
    ReplayComparison,
    ReplayLoader,
)
from .simulation import (
    RSISimulator,
    TrajectoryResult,
    ascii_convergence_curve,
    compute_convergence_stats,
    compute_mean_trajectory,
    find_optimal_params,
    run_benchmark,
    run_experiment_good_vs_bad,
    run_experiment_noise_floor,
    simulate_improvement,
    simulate_score,
)

__all__ = [
    "DEFAULT_VARIANCE_FLOOR_2SIGMA",
    "AdversarialCase",
    "AdversarialGenerator",
    "AdversarialRoundResult",
    "AdvocateResult",
    "CounterfactualRunner",
    "ImprovementDelta",
    "PopulationResult",
    "PopulationRunner",
    "RSISimulator",
    "RSISimulatorClass",
    "ReplayComparison",
    "ReplayLoader",
    "RSISimulator",
    "SelfImprovingHarness",
    "TrajectoryResult",
    "ascii_convergence_curve",
    "compute_convergence_stats",
    "compute_mean_trajectory",
    "convergence_check",
    "find_optimal_params",
    "generate_adversarial_case_mix",
    "improve_harness",
    "mine_failure_modes_from_result",
    "run_benchmark",
    "run_experiment_good_vs_bad",
    "run_experiment_noise_floor",
    "select_promotion_candidate",
    "simulate_improvement",
    "simulate_score",
]
