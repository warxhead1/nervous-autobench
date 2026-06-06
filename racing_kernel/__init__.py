"""Racing FunSearch kernel — LLM-driven evolution of racing-line controllers.

Evolves a Python ``racing_line(u, curvature, half_width, speed_limit)`` policy
function using a generative-membership oracle.  One island per track instance
(island-per-track, same pattern as sdf_kernel's island-per-instance).

Module layout:
  - instance.py  — RacingInstance, track geometry, generate_instance
  - oracle.py    — generative oracle, seed programs, prompt/parse helpers
  - loop.py      — RacingKernel (FunSearchKernel subclass + @register_kernel)

Public surface re-exported here so ``from autobench.racing_kernel import ...``
works as expected.
"""

from __future__ import annotations

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
)
from ..core import Verdict

from .instance import (  # noqa: F401
    RacingInstance,
    generate_instance,
    TRACK_LAYOUTS,
    TRACK_LAYOUTS as KNOWN_TRACKS,
)
from .oracle import (  # noqa: F401
    SEED_RACING_PROGRAMS,
    ISLAND_PERSONAS,
    PROMPT_SKETCHES,
    build_llm_prompt,
    parse_llm_response,
    evaluate_on_instance,
)
from .loop import RacingKernel  # noqa: F401  (fires @register_kernel("racing") on import)

__all__ = [
    # kernel base re-exports
    "FunSearchKernel", "KernelConfig", "CandidateProgram", "Island",
    "ensure_sandboxed_executor", "UnsafeSandboxError", "register_kernel",
    "Verdict",
    # instance
    "RacingInstance", "generate_instance", "TRACK_LAYOUTS", "KNOWN_TRACKS",
    # oracle
    "SEED_RACING_PROGRAMS", "ISLAND_PERSONAS", "PROMPT_SKETCHES",
    "build_llm_prompt", "parse_llm_response", "evaluate_on_instance",
    # loop
    "RacingKernel",
]
