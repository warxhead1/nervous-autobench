"""SPH kernel CLI entry point.

Usage:
    # Run FunSearch evolution on density reconstruction oracle:
    python -m autobench.sph_kernel run --instances gauss_blobs_3d \\
        --generations 60 --islands 6 --population 12 --allow-unsandboxed

    # Verify seed kernels score in expected 0.70-0.90 range (derisking gate):
    python -m autobench.sph_kernel baselines --instance gauss_blobs_3d --allow-unsandboxed

    # List available instances:
    python -m autobench.sph_kernel instances
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    SPHKernel,
    SPHInstance,
    evaluate_on_instance,
    ensure_sandboxed_executor as ensure_executor,
    generate_instance,
    get_seed_programs,
    _INSTANCE_CONFIGS,
)
from ..kernels import KernelConfig


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _default_output_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    date = datetime.date.today().isoformat()
    base = Path(__file__).resolve().parents[2] / "benchmarks" / "curriculum" / date
    base.mkdir(parents=True, exist_ok=True)
    return base


def cmd_run(args: argparse.Namespace) -> int:
    instances = args.instances.split(",")
    config = KernelConfig(
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

    kernel = SPHKernel(config)

    print("[SPH kernel] Starting FunSearch loop")
    print(f"  instances: {instances}")
    print(f"  islands: {args.islands} × {args.population} programs")
    print(f"  generations: {args.generations}")
    print(f"  sandbox: {kernel.executor.sandbox_type}")

    programs = kernel.run()
    results_path = kernel.save_results(programs)

    print(f"\nStopped: {kernel.stop_reason}")
    print(f"  generations run: {kernel.generation} | LLM requests: {kernel.llm_requests}")

    best = programs[0] if programs else None
    if best:
        print(f"\nBest sph_kernel(): {best.id}")
        print(f"  fitness (1/(1+MSE)): {best.fitness:.6f}")
        print(f"  worst-case fitness:  {best.worst_fitness:.6f}")
        print(f"  island: {best.island}, gen: {best.generation}")
        print(f"\nsph_kernel() function:\n{best.code}")

    if results_path:
        print(f"\nResults: {results_path}")

    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Evaluate all seed programs — derisking gate for oracle difficulty.

    Expected scores: 0.70–0.90. If seeds score >0.95 the oracle is too easy;
    tighten the instance (reduce n_particles or use sharp_gradients).
    If seeds score <0.50 the oracle is too hard; increase n_particles or h.
    """
    instance_name = args.instance or "gauss_blobs_3d"

    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        inst = generate_instance(instance_name)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"[SPH kernel] Baseline evaluation: {instance_name}")
    print(f"  particles: {inst.n_particles}  probes: {inst.n_probes}  h={inst.h:.3f}")
    print(f"  {inst.description}")
    print()

    seeds = get_seed_programs(instance_name)
    max_score = 0.0
    for name, code in seeds:
        fitness = evaluate_on_instance(code, inst, executor, run_timeout=30.0)
        if fitness is None:
            print(f"  {name:<20}: FAILED TO COMPILE/RUN")
        else:
            mse = 1.0 / fitness - 1.0
            flag = ""
            if fitness > 0.95:
                flag = "  ⚠ oracle too easy (>0.95)"
            elif fitness < 0.50:
                flag = "  ⚠ oracle too hard (<0.50)"
            print(f"  {name:<20}: fitness={fitness:.6f}  MSE={mse:.6f}{flag}")
            max_score = max(max_score, fitness)

    print()
    if max_score > 0.95:
        print("WARNING: oracle is too easy — seed scores >0.95. "
              "Try --instance sharp_gradients or reduce n_particles in _INSTANCE_CONFIGS.")
        return 1
    if max_score < 0.50:
        print("WARNING: oracle is too hard — best seed <0.50. "
              "Increase n_particles or h in _INSTANCE_CONFIGS.")
        return 1
    print(f"Oracle difficulty OK (best seed={max_score:.4f}, target 0.70–0.90).")
    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    print("[SPH kernel] Available benchmark instances:")
    for name, cfg in _INSTANCE_CONFIGS.items():
        print(f"  {name:<24} particles={cfg['n_particles']}  probes={cfg['n_probes']}  "
              f"h={cfg['h']:.2f}")
        print(f"    {cfg['description']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SPH FunSearch kernel — smoothing kernel evolution for fluid simulation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    run_p = sub.add_parser("run", help="Run FunSearch evolution loop")
    run_p.add_argument("--instances", default="gauss_blobs_3d",
                       help="Comma-separated instance names (default: gauss_blobs_3d)")
    run_p.add_argument("--generations", type=int, default=60)
    run_p.add_argument("--islands", type=int, default=6)
    run_p.add_argument("--population", type=int, default=12)
    run_p.add_argument("--migration-interval", type=int, default=10)
    run_p.add_argument("--temperature", type=float, default=0.9)
    run_p.add_argument("--candidates-per-island", type=int, default=1)
    run_p.add_argument("--max-concurrent-llm", type=int, default=6)
    run_p.add_argument("--target-fitness", type=float, default=None)
    run_p.add_argument("--max-requests", type=int, default=None)
    run_p.add_argument("--max-wall-seconds", type=float, default=None)
    run_p.add_argument("--plateau-generations", type=int, default=12)
    run_p.add_argument("--output-dir", default=None)
    run_p.add_argument("--allow-unsandboxed", action="store_true",
                       help="DANGER: run untrusted code without isolation (trusted/attended only)")
    run_p.add_argument("-v", "--verbose", action="store_true")

    # baselines
    base_p = sub.add_parser("baselines", help="Evaluate seed programs (oracle derisking gate)")
    base_p.add_argument("--instance", default="gauss_blobs_3d")
    base_p.add_argument("--allow-unsandboxed", action="store_true")
    base_p.add_argument("-v", "--verbose", action="store_true")

    # instances
    sub.add_parser("instances", help="List available SPH benchmark instances")

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
