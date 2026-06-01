"""Back-compat shim. Prefer `autobench.llm.anthropic`.

Phase 2B of the autobench restructuring moved ``llm_improver.py`` into
the ``autobench.llm`` subpackage. This module re-exports the public
surface so legacy ``from autobench.llm_improver import …`` call sites
keep working.
"""

from .llm.anthropic import (  # noqa: F401
    AnthropicLLMWrapper,
)
from .llm.base import (  # noqa: F401
    LLMImprovementResult,
    _build_evidence_section,
    _tolerant_json_loads,
)
