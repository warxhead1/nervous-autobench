"""Back-compat shim. Prefer `autobench.evaluation.curriculum`.

Phase 2B of the autobench restructuring moved ``curriculum.py`` into
the ``autobench.evaluation`` subpackage. This module re-exports the
public surface so legacy ``from autobench.curriculum import …`` call
sites keep working.
"""

from .evaluation.curriculum import (  # noqa: F401
    DEFAULT_CACHE_DIR,
    CurriculumAgent,
    CurriculumGoals,
    CurriculumScheduler,
    GeneratedProblem,
    JudgeVerdict,
    _build_synthesis_prompt,
    _difficulty_for_target_skill,
    _parse_problems,
    daily_synthesis,
)
