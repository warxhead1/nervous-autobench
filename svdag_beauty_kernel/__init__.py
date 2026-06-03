"""svdag_beauty FunSearch kernel — evolves compute_density for volcanic SVDAG terrain.

Submodules:
  - instance.py     — VolcanoInstance archetypes + target bands
  - oracle.py       — C density harness, seed programs, membership oracle, prompts
  - bridge_eval.py  — producer side of the tengine.shadergen.eval contract (render)
  - loop.py         — SVDAGBeautyKernel (FunSearchKernel subclass)

Re-exports compile_and_run / ensure_executor / Verdict as package attributes so
oracle.evaluate_on_instance resolves them through this namespace (test-patchable).
"""

from __future__ import annotations

import logging

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
)
from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run

ensure_executor = ensure_sandboxed_executor

logger = logging.getLogger(__name__)

from .instance import (  # noqa: E402
    VolcanoInstance, generate_instance, _INSTANCE_FACTORIES,
)
from .oracle import (  # noqa: E402
    SVDAG_SKELETON, COMPUTE_DENSITY_SIGNATURE, SEED_PROGRAMS, CONTROL_PROGRAMS,
    get_seed_programs, build_candidate_source, evaluate_on_instance,
    score_occupancy, build_llm_prompt, parse_llm_response,
)
from .loop import SVDAGBeautyKernel  # noqa: E402  (fires @register_kernel)

__all__ = [
    "FunSearchKernel", "KernelConfig", "CandidateProgram", "Island",
    "ensure_sandboxed_executor", "ensure_executor", "UnsafeSandboxError",
    "register_kernel", "Verdict", "SandboxedExecutor", "compile_and_run",
    "VolcanoInstance", "generate_instance", "_INSTANCE_FACTORIES",
    "SVDAG_SKELETON", "COMPUTE_DENSITY_SIGNATURE", "SEED_PROGRAMS", "CONTROL_PROGRAMS",
    "get_seed_programs", "build_candidate_source", "evaluate_on_instance",
    "score_occupancy", "build_llm_prompt", "parse_llm_response",
    "SVDAGBeautyKernel",
]
