"""Back-compat shim. Prefer `autobench.rsi.replay`.

Phase 2B of the autobench restructuring moved ``replay.py`` into the
``autobench.rsi`` subpackage. This module re-exports the public surface
so legacy ``from autobench.replay import …`` call sites keep working.
"""

from .rsi.replay import (  # noqa: F401
    CounterfactualRunner,
    ReplayComparison,
    ReplayLoader,
    filter_cases_by_id,
    harness_dict_to_config,
    load_cases_from_dir,
    merge_overrides,
    parse_override,
)
