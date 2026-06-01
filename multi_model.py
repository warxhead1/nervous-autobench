"""Back-compat shim. Prefer `autobench.llm.models`.

Phase 2B of the autobench restructuring moved ``multi_model.py`` into
the ``autobench.llm`` subpackage. This module re-exports the public
surface so legacy ``from autobench.multi_model import …`` call sites
keep working.
"""

from .llm.models import (  # noqa: F401
    AnthropicModelClient,
    DeepSeekModelClient,
    ModelBenchmarkResult,
    ModelClient,
    ModelClientRegistry,
    ModelLeaderboard,
    MultiModelBenchmark,
    NormalizedScore,
    ScoreNormalizer,
)
