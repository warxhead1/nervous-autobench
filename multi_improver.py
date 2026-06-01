"""Back-compat shim. Prefer `autobench.llm.ensemble`.

Phase 2B of the autobench restructuring moved ``multi_improver.py``
into the ``autobench.llm`` subpackage. This module re-exports the
public surface so legacy ``from autobench.multi_improver import …``
call sites keep working.
"""

from .llm.ensemble import (  # noqa: F401
    DEFAULT_N_INSTANCES,
    MultiImproverEnsemble,
    _default_wrapper_factory,
    aggregate_deltas,
)
