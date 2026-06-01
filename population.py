"""Back-compat shim. Prefer `autobench.rsi.population`.

Phase 2B of the autobench restructuring moved ``population.py`` into
the ``autobench.rsi`` subpackage. This module re-exports the public
surface so legacy ``from autobench.population import …`` call sites
keep working.
"""

from .rsi.population import (  # noqa: F401
    AdvocateResult,
    PopulationResult,
    PopulationRunner,
    _read_diversity_weight_env,
    _read_n_advocates_env,
    select_promotion_candidate,
)
