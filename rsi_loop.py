"""Back-compat shim. Prefer `autobench.rsi.loop`.

Phase 2B of the autobench restructuring moved ``rsi_loop.py`` into the
``autobench.rsi`` subpackage. This module re-exports the public surface
so legacy ``from autobench.rsi_loop import …`` call sites keep working.
"""

from .rsi.loop import (  # noqa: F401
    DEFAULT_VARIANCE_FLOOR_2SIGMA,
    ImprovementDelta,
    SelfImprovingHarness,
    _build_diagnosis_prompt,
    _default_variance_floor,
    _parse_llm_improvement,
    convergence_check,
    improve_harness,
)
