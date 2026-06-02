"""SDF FunSearch kernel — LLM-driven discovery of signed-distance-field functions.

Decomposed into cohesive submodules (behaviour-preserving file split):
  - instance.py  — SDFInstance, analytic `_sdf_*` targets, _INSTANCE_FACTORIES,
                    generate_instance, _make_samples, KNOWN_OPTIMALS
  - topology.py  — TOPO_TARGETS, build_topology_source, compute_topology_score
  - oracle.py    — C++ skeleton, seed programs, personas/sketches, prompt build,
                   parse_llm_response, evaluate_on_instance
  - loop.py      — SDFKernel (FunSearchKernel subclass) + bus/evolution helpers

This package re-exports the full public surface. `compile_and_run`,
`ensure_executor`, and `Verdict` live here as package attributes so tests can
patch `autobench.sdf_kernel.compile_and_run` / `.ensure_executor`.
"""

from __future__ import annotations

import logging

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
)
from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run

# Back-compat: tests and external code patch `autobench.sdf_kernel.ensure_executor`.
# The local copy used to be a near-duplicate of kernels.ensure_sandboxed_executor
# with max_memory_mb=256; that distinction is now in KernelConfig.max_memory_mb.
ensure_executor = ensure_sandboxed_executor

logger = logging.getLogger(__name__)

# Submodules — imported AFTER ensure_executor / compile_and_run / Verdict are
# bound above, because they resolve those names back through this package at
# call time (so test patches on this package take effect).
from .instance import (  # noqa: E402
    SDFInstance,
    _make_samples,
    _sdf_sphere,
    _sdf_box,
    _smooth_union,
    _sdf_gyroid,
    _sdf_round_box,
    _sdf_warped_sphere,
    _sdf_smooth_union_shape,
    _sdf_cloud_cluster,
    _sdf_torus_knot,
    _sdf_helix_tube,
    _sdf_scherk_first,
    KNOWN_OPTIMALS,
    _INSTANCE_FACTORIES,
    generate_instance,
)
from .topology import (  # noqa: E402
    TOPO_TARGETS,
    TOPO_SKELETON,
    build_topology_source,
    compute_topology_score,
)
from .oracle import (  # noqa: E402
    CPP_SKELETON,
    SDF_FUNCTION_SIGNATURE,
    SEED_SDF_PROGRAMS,
    get_seed_programs,
    build_candidate_source,
    _instance_stdin,
    evaluate_on_instance,
    SDF_ISLAND_PERSONAS,
    SDF_PROMPT_SKETCHES,
    build_llm_prompt,
    parse_llm_response,
)
from .loop import SDFKernel  # noqa: E402  (fires @register_kernel("sdf") on import)

__all__ = [
    # sandbox / kernel base re-exports
    "FunSearchKernel", "KernelConfig", "CandidateProgram", "Island",
    "ensure_sandboxed_executor", "ensure_executor", "UnsafeSandboxError",
    "register_kernel", "Verdict", "SandboxedExecutor", "compile_and_run",
    # instance
    "SDFInstance", "_make_samples", "generate_instance",
    "_sdf_sphere", "_sdf_box", "_smooth_union", "_sdf_gyroid", "_sdf_round_box",
    "_sdf_warped_sphere", "_sdf_smooth_union_shape", "_sdf_cloud_cluster",
    "_sdf_torus_knot", "_sdf_helix_tube", "_sdf_scherk_first",
    "KNOWN_OPTIMALS", "_INSTANCE_FACTORIES",
    # topology
    "TOPO_TARGETS", "TOPO_SKELETON", "build_topology_source",
    "compute_topology_score",
    # oracle
    "CPP_SKELETON", "SDF_FUNCTION_SIGNATURE", "SEED_SDF_PROGRAMS",
    "get_seed_programs", "build_candidate_source", "_instance_stdin",
    "evaluate_on_instance", "SDF_ISLAND_PERSONAS", "SDF_PROMPT_SKETCHES",
    "build_llm_prompt", "parse_llm_response",
    # loop
    "SDFKernel",
]
