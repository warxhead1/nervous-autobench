"""Unified CLI dispatcher for the autobench FunSearch kernels.

Replaces the eight near-identical per-kernel ``cli.py`` files with a single
parser that dispatches via the registry in ``kernels/base.py``. The per-kernel
``cli.py`` files are kept as thin back-compat shims that call into here.

Usage:
    python -m autobench.kernels run         --kernel sdf  --instances gyroid,round_box
    python -m autobench.kernels baselines   --kernel sdf  --instance gyroid
    python -m autobench.kernels instances   --kernel sdf
    python -m autobench.kernels compile     --kernel sdf  --code-file my_sdf.cpp --instance gyroid
    python -m autobench.kernels eval        --kernel sdf  --results-file sdf_results_gen50.json
    python -m autobench.kernels eval-bridge --code-file terrain.c

The ``compile`` and ``eval`` subcommands are wired to the per-kernel custom
handlers when those handlers exist on the kernel subclass (registered via
``register_kernel_compile_handler`` and ``register_eval_handler``), and fall
back to a generic message otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

from .base import KERNEL_REGISTRY, FunSearchKernel

# Eagerly import the 8 kernel subpackages so their @register_kernel decorators
# fire. Without this, ``python -m autobench.kernels run --kernel sdf`` would
# not know about any kernel — the user would have to remember to import the
# per-kernel subpackage first. Importing the subpackage has the side-effect
# of populating KERNEL_REGISTRY.
#
# Order: noise last (it pulls in moderngl via the noise_kernel import side
# effects in some environments; everything else is pure-CPU/import-cheap).
for _kernel_pkg in (
    "autobench.tsp_kernel",
    "autobench.sdf_kernel",
    "autobench.latent_kernel",
    "autobench.phase_kernel",
    "autobench.sph_kernel",
    "autobench.terrain_kernel",
    "autobench.thermal_kernel",
    "autobench.noise_kernel",
):
    try:
        __import__(_kernel_pkg)
    except Exception as _exc:
        # Don't fail the CLI just because one kernel couldn't import (e.g. no
        # GPU for noise_kernel). Print to stderr so the user knows.
        print(f"[autobench.kernels] skipping {_kernel_pkg}: {_exc}",
              file=sys.stderr)

# ---------------------------------------------------------------------------
# Per-kernel custom command handlers — opt-in via decorators below.
# Used by sdf/tsp for `compile`/`eval`, by terrain for `eval-bridge`.
# ---------------------------------------------------------------------------

_COMPILE_HANDLERS: dict[str, Callable[[argparse.Namespace, "FunSearchKernel"], int]] = {}
_EVAL_HANDLERS: dict[str, Callable[[argparse.Namespace, "FunSearchKernel"], int]] = {}
_EVAL_BRIDGE_HANDLERS: dict[str, Callable[[argparse.Namespace, "FunSearchKernel"], int]] = {}


def register_compile_handler(name: str):
    """Decorator: register a kernel-specific `compile` command handler."""
    def deco(fn: Callable[[argparse.Namespace, "FunSearchKernel"], int]):
        _COMPILE_HANDLERS[name] = fn
        return fn
    return deco


def register_eval_handler(name: str):
    """Decorator: register a kernel-specific `eval` command handler."""
    def deco(fn: Callable[[argparse.Namespace, "FunSearchKernel"], int]):
        _EVAL_HANDLERS[name] = fn
        return fn
    return deco


def register_eval_bridge_handler(name: str):
    """Decorator: register a kernel-specific `eval-bridge` command handler (TEngine hot-swap)."""
    def deco(fn: Callable[[argparse.Namespace, "FunSearchKernel"], int]):
        _EVAL_BRIDGE_HANDLERS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def _resolve_kernel(name: str) -> type[FunSearchKernel]:
    if name not in KERNEL_REGISTRY:
        registered = ", ".join(sorted(KERNEL_REGISTRY))
        raise SystemExit(
            f"Unknown kernel '{name}'. Registered: [{registered}]\n"
            f"  (kernels register themselves on import — is '{name}' imported somewhere?)"
        )
    return KERNEL_REGISTRY[name]


def _default_output_dir(kernel: str, arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from datetime import date
    base = Path(__file__).resolve().parents[2] / "benchmarks" / "curriculum" / date.today().isoformat()
    base.mkdir(parents=True, exist_ok=True)
    return base


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full FunSearch loop on the named kernel."""
    kernel_cls = _resolve_kernel(args.kernel)
    instances = args.instances.split(",")
    from .config import KernelConfig
    config = KernelConfig(
        instances=instances,
        n_islands=args.islands,
        population_per_island=args.population,
        generations=args.generations,
        migration_interval=args.migration_interval,
        temperature=args.temperature,
        output_dir=_default_output_dir(args.kernel, args.output_dir),
        allow_unsandboxed=args.allow_unsandboxed,
        candidates_per_island=args.candidates_per_island,
        max_concurrent_llm=args.max_concurrent_llm,
        target_fitness=args.target_fitness,
        max_requests=args.max_requests,
        max_wall_seconds=args.max_wall_seconds,
        plateau_generations=args.plateau_generations,
    )
    kernel = kernel_cls(config)
    print(f"[{args.kernel} kernel] Starting FunSearch loop")
    print(f"  instances: {instances}")
    print(f"  islands: {args.islands} × {args.population} programs")
    print(f"  generations: {args.generations}")
    programs = kernel.run()
    if hasattr(kernel, "save_results"):
        kernel.save_results(programs)
    print(f"\nStopped: {kernel.stop_reason}")
    print(f"  generations: {kernel.generation} | LLM requests: {kernel.llm_requests}")
    best = programs[0] if programs else None
    if best:
        print(f"\nBest: {best.id}  fitness={best.fitness:.6f}")
        print(f"  island={best.island} gen={best.generation}")
        print(f"\n{best.code}")
    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    """List available instances for the named kernel by instantiating it briefly.

    Implemented in-kernel: each kernel has its own ``cmd_instances`` semantics,
    so we delegate to the kernel class if it exposes one, else print a hint.
    """
    kernel_cls = _resolve_kernel(args.kernel)
    # Most kernels expose a list-style constant after import. We import the
    # module that contains the kernel so its side-effects (constants like
    # `_INSTANCE_CONFIGS`, `_INSTANCE_FACTORIES`) populate.
    module = sys.modules.get(kernel_cls.__module__)
    if module is None:
        import importlib
        module = importlib.import_module(kernel_cls.__module__)
    # Try the common per-kernel instance-list attributes
    for attr in ("_INSTANCE_CONFIGS", "_INSTANCE_FACTORIES", "_LATENT_INSTANCE_CONFIGS",
                 "_PHASE_INSTANCE_CONFIGS", "_TERRAIN_INSTANCE_CONFIGS",
                 "_THERMAL_INSTANCE_CONFIGS", "KNOWN_OPTIMALS", "_REFERENCE_SHADERS",
                 "TSP_INSTANCE_DIR", "instances"):
        if hasattr(module, attr):
            val = getattr(module, attr)
            if isinstance(val, dict):
                print(f"[{args.kernel} kernel] Available instances ({attr}):")
                for name in sorted(val):
                    print(f"  {name}")
                return 0
    print(f"[{args.kernel} kernel] No instance listing available — "
          f"the kernel does not export a *_INSTANCE_CONFIGS dict.")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Test-compile a candidate. Per-kernel custom logic wins if registered."""
    kernel_cls = _resolve_kernel(args.kernel)
    handler = _COMPILE_HANDLERS.get(args.kernel)
    if handler is None:
        print(f"[{args.kernel} kernel] `compile` subcommand is not implemented "
              f"for this kernel. Use the per-kernel CLI "
              f"(python -m autobench.{args.kernel}_kernel compile ...).")
        return 1
    return handler(args, kernel_cls)


def cmd_eval(args: argparse.Namespace) -> int:
    """Re-evaluate saved programs from a results JSON. Per-kernel custom logic wins if registered."""
    kernel_cls = _resolve_kernel(args.kernel)
    handler = _EVAL_HANDLERS.get(args.kernel)
    if handler is None:
        print(f"[{args.kernel} kernel] `eval` subcommand is not implemented "
              f"for this kernel. Use the per-kernel CLI "
              f"(python -m autobench.{args.kernel}_kernel eval ...).")
        return 1
    return handler(args, kernel_cls)


def cmd_eval_bridge(args: argparse.Namespace) -> int:
    """Hot-swap a FunSearch terrain candidate through the TEngine bridge."""
    kernel_cls = _resolve_kernel(args.kernel)
    handler = _EVAL_BRIDGE_HANDLERS.get(args.kernel)
    if handler is None:
        print(f"[{args.kernel} kernel] `eval-bridge` subcommand is not implemented "
              f"for this kernel. (Only the `terrain` kernel uses the TEngine bridge.)")
        return 1
    return handler(args, kernel_cls)


def cmd_baselines(args: argparse.Namespace) -> int:
    """Run the baseline evaluation for the named kernel.

    Like `instances`, the per-kernel cmd_baselines lives in the per-kernel CLI
    today; the unified CLI is a thin dispatcher that prints a hint.
    """
    print(f"[{args.kernel} kernel] `baselines` runs in the per-kernel CLI — "
          f"call it as: python -m autobench.{args.kernel}_kernel baselines --instance {args.instance}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="autobench FunSearch kernels — unified CLI",
        epilog="kernels register themselves on import. Pass --kernel NAME to dispatch.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    run_p = sub.add_parser("run", help="Run FunSearch evolution loop on a registered kernel")
    run_p.add_argument("--kernel", required=True, help="Kernel name (e.g. sdf, tsp, sph, latent, phase, terrain, thermal, noise)")
    run_p.add_argument("--instances", required=True,
                       help="Comma-separated instance names (kernel-specific; try 'default' or see per-kernel CLI)")
    run_p.add_argument("--generations", type=int, default=50)
    run_p.add_argument("--islands", type=int, default=4)
    run_p.add_argument("--population", type=int, default=20)
    run_p.add_argument("--migration-interval", type=int, default=10)
    run_p.add_argument("--temperature", type=float, default=0.9)
    run_p.add_argument("--candidates-per-island", type=int, default=1)
    run_p.add_argument("--max-concurrent-llm", type=int, default=8)
    run_p.add_argument("--target-fitness", type=float, default=None)
    run_p.add_argument("--max-requests", type=int, default=None)
    run_p.add_argument("--max-wall-seconds", type=float, default=None)
    run_p.add_argument("--plateau-generations", type=int, default=None)
    run_p.add_argument("--output-dir", default=None)
    run_p.add_argument("--allow-unsandboxed", action="store_true",
                       help="DANGER: run untrusted candidate code without isolation")
    run_p.add_argument("-v", "--verbose", action="store_true")

    # baselines
    base_p = sub.add_parser("baselines", help="Evaluate seed programs for an instance (delegates to per-kernel CLI)")
    base_p.add_argument("--kernel", required=True)
    base_p.add_argument("--instance", default=None)
    base_p.add_argument("--allow-unsandboxed", action="store_true")
    base_p.add_argument("-v", "--verbose", action="store_true")

    # instances
    inst_p = sub.add_parser("instances", help="List available instances for a registered kernel")
    inst_p.add_argument("--kernel", required=True)

    # compile
    compile_p = sub.add_parser("compile", help="Test-compile a candidate (currently: sdf, tsp)")
    compile_p.add_argument("--kernel", required=True)
    compile_p.add_argument("--code-file", required=True)
    compile_p.add_argument("--instance", default=None)
    compile_p.add_argument("--output", default=None)
    compile_p.add_argument("--allow-unsandboxed", action="store_true")
    compile_p.add_argument("-v", "--verbose", action="store_true")

    # eval
    eval_p = sub.add_parser("eval", help="Re-evaluate saved programs (currently: sdf, tsp)")
    eval_p.add_argument("--kernel", required=True)
    eval_p.add_argument("--results-file", required=True)
    eval_p.add_argument("--instances", default=None)
    eval_p.add_argument("--top-n", type=int, default=3)
    eval_p.add_argument("--allow-unsandboxed", action="store_true")
    eval_p.add_argument("-v", "--verbose", action="store_true")

    # eval-bridge
    eb_p = sub.add_parser("eval-bridge", help="Hot-swap a FunSearch candidate through the TEngine bridge (terrain)")
    eb_p.add_argument("--kernel", required=True)
    eb_p.add_argument("--code-file", required=True)
    eb_p.add_argument("--biome", default="rolling_hills")
    eb_p.add_argument("--allow-unsandboxed", action="store_true")
    eb_p.add_argument("-v", "--verbose", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))

    dispatch = {
        "run": cmd_run,
        "baselines": cmd_baselines,
        "instances": cmd_instances,
        "compile": cmd_compile,
        "eval": cmd_eval,
        "eval-bridge": cmd_eval_bridge,
    }
    handler = dispatch.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
