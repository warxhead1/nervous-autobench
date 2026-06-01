"""Back-compat shim. Prefer `autobench.llm.judge`.

Phase 2B of the autobench restructuring moved ``default_judge_factory.py``
into the ``autobench.llm`` subpackage. This module re-exports the
public surface so legacy ``from autobench.default_judge_factory import …``
call sites keep working.
"""

from .llm.judge import (  # noqa: F401
    _parse_judge_response,
    make_minimax_judge_factory,
)
