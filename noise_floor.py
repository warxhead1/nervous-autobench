"""Back-compat shim. Prefer `autobench.evaluation.noise_floor`.

Phase 2B of the autobench restructuring moved ``noise_floor.py`` into
the ``autobench.evaluation`` subpackage. This module re-exports the
public surface so legacy ``from autobench.noise_floor import …`` call
sites keep working.
"""

from .evaluation.noise_floor import (  # noqa: F401
    LanguageStats,
    NoiseFloorReport,
    NoiseFloorResult,
    main,
    print_report,
    run_noise_floor_measurement,
)
