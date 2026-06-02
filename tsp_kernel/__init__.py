"""TSP FunSearch kernel — LLM-driven heuristic discovery for routing.

Decomposed into cohesive submodules (behavior-preserving file split):
  - ``instance``  — TSPInstance/TSPLIB loading, CandidateProgram, Island, ULIDs
  - ``scoring``   — sandboxed C++ compile/run + fitness evaluation
  - ``oracle``    — LLM prompt build, response parse, mutation
  - ``loop``      — TSPKernel class + island/evolution helpers

This package re-exports the public surface so existing
``from autobench.tsp_kernel import ...`` imports keep working, and importing
the package registers the ``tsp`` kernel via ``@register_kernel`` on TSPKernel.

Reference:
  FunSearch (DeepMind, 2024): https://github.com/google-deepmind/funsearch
  EoH (ICML 2024): https://arxiv.org/abs/2401.02051
"""

from __future__ import annotations

from .instance import (
    KNOWN_OPTIMALS,
    TSPLIB_URL_BASE,
    CandidateProgram,
    Island,
    TSPInstance,
    Tourevaluation,
    fetch_tsplib_instance,
    new_ulid,
)
from .scoring import (
    CPP_MAIN_TEMPLATE,
    CPP_SKELETON,
    FunSearchKernel,
    KernelConfig,
    UnsafeSandboxError,
    build_candidate_source,
    ensure_sandboxed_executor,
    evaluate_fitness,
    evaluate_on_instance,
    register_kernel,
)
from .oracle import (
    ISLAND_PERSONAS,
    PRIORITY_SIGNATURE,
    PROMPT_SKETCHES,
    build_llm_prompt,
    mutate_priority,
    parse_llm_response,
)
from .loop import (
    TSPKernel,
    evaluate_island,
    init_baseline_programs,
    initialize_islands,
    migrate,
)

__all__ = [
    "KNOWN_OPTIMALS",
    "TSPLIB_URL_BASE",
    "CandidateProgram",
    "Island",
    "TSPInstance",
    "Tourevaluation",
    "fetch_tsplib_instance",
    "new_ulid",
    "CPP_MAIN_TEMPLATE",
    "CPP_SKELETON",
    "FunSearchKernel",
    "KernelConfig",
    "UnsafeSandboxError",
    "build_candidate_source",
    "ensure_sandboxed_executor",
    "evaluate_fitness",
    "evaluate_on_instance",
    "register_kernel",
    "ISLAND_PERSONAS",
    "PRIORITY_SIGNATURE",
    "PROMPT_SKETCHES",
    "build_llm_prompt",
    "mutate_priority",
    "parse_llm_response",
    "TSPKernel",
    "evaluate_island",
    "init_baseline_programs",
    "initialize_islands",
    "migrate",
]
