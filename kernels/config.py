"""KernelConfig — single source of truth for FunSearch kernel configuration.

Consolidated from autobench.kernel_base (Phase 1 of the kernel restructuring).
The duplicate definition that used to live in tsp_kernel/__init__.py is gone —
all kernels import `KernelConfig` from `autobench.kernels`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Callable


@dataclass
class KernelConfig:
    """Configuration for any FunSearch-style kernel."""
    # Problem instances (kernel subclass interprets these names)
    instances: list[str] = field(default_factory=list)
    # Island model
    n_islands: int = 4
    population_per_island: int = 10
    generations: int = 100
    migration_interval: int = 5
    temperature: float = 0.9
    # Timeouts
    compile_timeout: float = 30.0
    run_timeout: float = 10.0
    llm_timeout: float = 120.0
    # Throughput
    candidates_per_island: int = 1
    max_concurrent_llm: int = 8
    # Horizon governance (stop before generation cap when any is hit)
    target_fitness: float | None = None
    max_requests: int | None = None
    max_wall_seconds: float | None = None
    plateau_generations: int | None = None
    plateau_epsilon: float = 1e-4
    plateau_hint: bool = True          # call deer for strategic advice mid-plateau
    plateau_hint_model: str = os.environ.get("MINIMAX_MODEL", "minimax-m2.7").lower()
    # I/O
    llm_call_fn: Callable[[str], str] | None = None
    output_dir: Path | None = None
    # Sandbox
    allow_unsandboxed: bool = False
    max_memory_mb: int = 512
    # Bus
    nervous_bin: str | None = None
    bus_verbose: bool = False
    # Island routing
    island_instance_assignment: bool = False  # per-island instance routing (SDF uses True)
    # Consolidated prior
    use_consolidated_prior: bool = True   # load/save T-vectors across runs
    prior_top_k: int = 5                  # how many programs to extract T-vectors from
