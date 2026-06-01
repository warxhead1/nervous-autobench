"""Back-compat shim. Prefer `autobench.llm.minimax`.

Phase 2B of the autobench restructuring moved ``minimax_improver.py``
into the ``autobench.llm`` subpackage. This module re-exports the
public surface so legacy ``from autobench.minimax_improver import …``
call sites keep working.
"""

from .llm.minimax import (  # noqa: F401
    MiniMaxLLMWrapper,
    _format_revert_history_block,
    _format_siblings_block,
    _is_no_op_value,
    _parse_response,
)
