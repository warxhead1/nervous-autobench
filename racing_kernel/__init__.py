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

Import note: uses absolute imports so the package is importable both as
``autobench.racing_kernel`` (installed) and as a standalone path during
pytest collection (``pytest racing_kernel``).
"""

from __future__ import annotations

# Use absolute imports so this package is importable from pytest racing_kernel
# (which loads it as top-level 'racing_kernel') AND as autobench.racing_kernel.
try:
    from autobench.kernels import (  # noqa: F401
        FunSearchKernel, KernelConfig, CandidateProgram, Island,
        ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
    )
    from autobench.core import Verdict  # noqa: F401
    from autobench.racing_kernel.instance import (  # noqa: F401
        RacingInstance,
        generate_instance,
        TRACK_LAYOUTS,
    )
    from autobench.racing_kernel.oracle import (  # noqa: F401
        SEED_RACING_PROGRAMS,
        ISLAND_PERSONAS,
        PROMPT_SKETCHES,
        build_llm_prompt,
        parse_llm_response,
        evaluate_on_instance,
    )
    from autobench.racing_kernel.loop import RacingKernel  # noqa: F401
    KNOWN_TRACKS = TRACK_LAYOUTS
    # rollout_eval — additive export (nervous-bus-71cn.5)
    from autobench.racing_kernel.rollout_eval import (  # noqa: F401
        evaluate_via_rollout,
        calibrate_ref_lap_time,
    )
except ImportError:
    # Fallback: imported standalone (e.g. pytest racing_kernel without install).
    # Sub-modules use absolute imports; __init__ can stay mostly empty here.
    pass

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
    # rollout_eval (nervous-bus-71cn.5)
    "evaluate_via_rollout",
    "calibrate_ref_lap_time",
]
