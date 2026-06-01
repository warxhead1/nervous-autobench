"""LLM improver subpackage: Anthropic / MiniMax / ensemble / worker / models / judge.

Public surface (import these — the root-level ``autobench.llm_improver`` etc.
are deprecated back-compat shims):

  - :class:`AnthropicLLMWrapper`  (from ``anthropic``)
  - :class:`MiniMaxLLMWrapper`    (from ``minimax``)
  - :class:`MultiImproverEnsemble` (from ``ensemble``)
  - :class:`MiniMaxWorker`         (from ``worker``)
  - :class:`ModelClient` / :class:`ModelClientRegistry` (from ``models``)
  - :func:`make_minimax_judge_factory` (from ``judge``)
  - :class:`LLMImprovementResult`, :class:`BaseLLMImprover` (from ``base``)
"""

from __future__ import annotations

from .anthropic import AnthropicLLMWrapper
from .base import (
    BaseLLMImprover,
    LLMImprovementResult,
    build_evidence_section,
    tolerant_json_loads,
)
from .ensemble import (
    DEFAULT_N_INSTANCES,
    MultiImproverEnsemble,
    aggregate_deltas,
)
from .judge import make_minimax_judge_factory
from .minimax import MiniMaxLLMWrapper
from .models import (
    ModelBenchmarkResult,
    ModelClient,
    ModelClientRegistry,
    ModelLeaderboard,
    MultiModelBenchmark,
)
from .worker import MiniMaxWorker, WorkerResult

__all__ = [
    "AnthropicLLMWrapper",
    "BaseLLMImprover",
    "DEFAULT_N_INSTANCES",
    "LLMImprovementResult",
    "MiniMaxLLMWrapper",
    "MiniMaxWorker",
    "ModelBenchmarkResult",
    "ModelClient",
    "ModelClientRegistry",
    "ModelLeaderboard",
    "MultiImproverEnsemble",
    "MultiModelBenchmark",
    "WorkerResult",
    "aggregate_deltas",
    "build_evidence_section",
    "make_minimax_judge_factory",
    "tolerant_json_loads",
]
