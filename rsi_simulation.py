"""Back-compat shim. Prefer `autobench.rsi.simulation`.

Phase 2B of the autobench restructuring moved ``rsi_simulation.py``
into the ``autobench.rsi`` subpackage. This module re-exports the
public surface so legacy ``from autobench.rsi_simulation import …``
call sites keep working.
"""

from .rsi.simulation import (  # noqa: F401
    RSISimulator,
    TrajectoryResult,
    compute_convergence_stats,
    compute_mean_trajectory,
    find_optimal_params,
    run_benchmark,
    run_experiment_good_vs_bad,
    run_experiment_noise_floor,
    simulate_improvement,
    simulate_score,
)
