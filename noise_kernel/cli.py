"""NoiseKernel CLI entry point.

Usage:
    python -m autobench.noise_kernel run \\
        --instances value_noise_3d,perlin_like,fbm_2octave \\
        --generations 50 --islands 4 --population 20 \\
        --allow-unsandboxed

    python -m autobench.noise_kernel baselines \\
        --instance value_noise_3d --allow-unsandboxed

    python -m autobench.noise_kernel instances
        -- list available benchmark instances
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    NoiseKernel,
    NoiseKernelConfig,
    NoiseInstance,
    _REFERENCE_SHADERS,
    _SEEDS,
    build_probe_shader,
)
from ..kernel_base import KernelConfig, CandidateProgram


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _default_output_dir(arg: str | None) -> Path:
    """Return a persistent output dir for this run.

    Honours --output-dir if set; otherwise defaults to
    autobench/benchmarks/curriculum/<YYYY-MM-DD>/ (the convention for
    dated benchmark runs).
    """
    if arg:
        return Path(arg)
    date = datetime.date.today().isoformat()
    base = Path(__file__).resolve().parents[2] / "benchmarks" / "curriculum" / date
    base.mkdir(parents=True, exist_ok=True)
    return base


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full FunSearch evolution loop."""
    instances = args.instances.split(",")
    config = NoiseKernelConfig(
        instances=instances,
        n_islands=args.islands,
        population_per_island=args.population,
        generations=args.generations,
        migration_interval=args.migration_interval,
        temperature=args.temperature,
        output_dir=_default_output_dir(args.output_dir),
        allow_unsandboxed=args.allow_unsandboxed,
        candidates_per_island=args.candidates_per_island,
        max_concurrent_llm=args.max_concurrent_llm,
        target_fitness=args.target_fitness,
        max_requests=args.max_requests,
        max_wall_seconds=args.max_wall_seconds,
        plateau_generations=args.plateau_generations,
    )

    kernel = NoiseKernel(config)

    print("[noise kernel] Starting FunSearch loop")
    print(f"  instances: {instances}")
    print(f"  islands: {args.islands} x {args.population} programs")
    print(f"  generations: {args.generations}")
    print("  backend: moderngl/EGL (GPU)")

    programs = kernel.run()
    kernel.save_results(programs)

    print(f"\nStopped: {kernel.stop_reason}")
    print(f"  generations run: {kernel.generation} | LLM requests: {kernel.llm_requests}")

    best = programs[0] if programs else None
    if best:
        print(f"\nBest noise(): {best.id}")
        print(f"  fitness (SSIM): {best.fitness:.6f}")
        print(f"  worst-case fitness: {best.worst_fitness:.6f}")
        print(f"  island: {best.island}, gen: {best.generation}")
        print(f"\nnoise() function:\n{best.code}")

    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Evaluate all baseline seed programs for a given instance."""
    instance_name = args.instance or "value_noise_3d"
    if instance_name not in _REFERENCE_SHADERS:
        print(f"[ERROR] Unknown instance '{instance_name}'. Available: {sorted(_REFERENCE_SHADERS)}")
        return 1

    import tempfile
    output_dir = Path(tempfile.mkdtemp(prefix="noise_baselines_"))
    config = NoiseKernelConfig(
        instances=[instance_name],
        output_dir=output_dir,
        allow_unsandboxed=args.allow_unsandboxed,
    )

    try:
        kernel = NoiseKernel(config)
    except Exception as e:
        print(f"[ERROR] Kernel init failed: {e}")
        return 1

    inst = kernel.problem_instances[0] if kernel.problem_instances else None
    if inst is None:
        print("[ERROR] No instances loaded")
        return 1

    print(f"[noise kernel] Baseline evaluation for instance: {instance_name}")
    print(f"  Description: {inst.description}")
    print(f"  Reference PNG: {inst.reference_png}")
    if not inst.reference_png.exists():
        print("  [WARNING] Reference PNG not rendered (no GPU?); SSIM scores will be None")
    print()

    for variant_name, code in _SEEDS:
        fitness = kernel.evaluate_candidate(code, inst)
        if fitness is None:
            print(f"  {variant_name}: SKIPPED (no GPU or compile error)")
        else:
            print(f"  {variant_name}: SSIM={fitness:.6f}")

    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    """List available noise benchmark instances."""
    print("[noise kernel] Available benchmark instances:")
    for name, (_, description) in _REFERENCE_SHADERS.items():
        print(f"  {name:<20} {description}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Noise FunSearch kernel — GPU-evaluated GLSL noise evolution"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run ---
    run_parser = sub.add_parser("run", help="Run FunSearch evolution loop")
    run_parser.add_argument(
        "--instances", default="value_noise_3d,perlin_like,fbm_2octave",
        help="Comma-separated instance names (default: all 3). "
             "Run 'instances' subcommand to list available."
    )
    run_parser.add_argument("--generations", type=int, default=50, help="Max generations")
    run_parser.add_argument("--islands", type=int, default=4, help="Number of islands")
    run_parser.add_argument("--population", type=int, default=20, help="Programs per island")
    run_parser.add_argument("--migration-interval", type=int, default=10,
                            help="Generations between migrations")
    run_parser.add_argument("--temperature", type=float, default=0.9, help="LLM temperature")
    run_parser.add_argument("--candidates-per-island", type=int, default=1,
                            help="LLM candidates each island generates per generation")
    run_parser.add_argument("--max-concurrent-llm", type=int, default=8,
                            help="Max concurrent LLM calls")
    # Horizon governance
    run_parser.add_argument("--target-fitness", type=float, default=None,
                            help="Stop when best SSIM >= this (1.0 = exact match)")
    run_parser.add_argument("--max-requests", type=int, default=None,
                            help="LLM request budget (billing unit)")
    run_parser.add_argument("--max-wall-seconds", type=float, default=None,
                            help="Wall-clock budget in seconds")
    run_parser.add_argument("--plateau-generations", type=int, default=None,
                            help="Stop after N gens with no SSIM improvement")
    run_parser.add_argument("--output-dir", default=None, help="Directory for results JSON")
    run_parser.add_argument(
        "--allow-unsandboxed", action="store_true",
        help="Accept GLSL candidates without OS-level sandboxing (driver-only isolation)"
    )
    run_parser.add_argument("-v", "--verbose", action="store_true")

    # --- baselines ---
    baselines_parser = sub.add_parser(
        "baselines", help="Evaluate all seed baseline programs for an instance"
    )
    baselines_parser.add_argument(
        "--instance", default="value_noise_3d",
        help="Noise instance name (default: value_noise_3d)"
    )
    baselines_parser.add_argument(
        "--allow-unsandboxed", action="store_true",
        help="Accept GLSL evaluation without OS-level sandboxing"
    )
    baselines_parser.add_argument("-v", "--verbose", action="store_true")

    # --- instances ---
    sub.add_parser("instances", help="List available noise benchmark instances")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    dispatch = {
        "run": cmd_run,
        "baselines": cmd_baselines,
        "instances": cmd_instances,
    }
    handler = dispatch.get(args.cmd)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
