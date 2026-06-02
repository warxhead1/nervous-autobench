"""Terrain kernel CLI.

Usage:
    python -m autobench.terrain_kernel run --instances rolling_hills \\
        --generations 60 --islands 4 --population 10 --allow-unsandboxed

    python -m autobench.terrain_kernel baselines --instance rolling_hills --allow-unsandboxed

    python -m autobench.terrain_kernel instances
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    TerrainKernel,
    TerrainInstance,
    evaluate_on_instance,
    ensure_sandboxed_executor as ensure_executor,
    generate_instance,
    get_seed_programs,
    _TERRAIN_INSTANCE_CONFIGS,
)
from ..kernels import KernelConfig


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
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

    kernel = TerrainKernel(config)

    print("[Terrain kernel] Starting FunSearch loop")
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
        print(f"\nBest terrain(): {best.id}")
        print(f"  fitness:       {best.fitness:.6f}")
        print(f"  worst_fitness: {best.worst_fitness:.6f}")
        print(f"  island: {best.island}, gen: {best.generation}")
        print(f"\nterrain() function:\n{best.code}")

    if results_path:
        print(f"\nResults: {results_path}")

    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Evaluate seed functions — oracle calibration gate.

    Reports H_est alongside fitness so you can tune target_hurst values.
    Expected scores: 0.50–0.90. If all seeds score >0.95 the oracle is too
    easy; tighten target_hurst. If all seeds score <0.30 the oracle is too
    hard; relax target_hurst toward the measured H_est of the best seed.
    """
    instance_name = args.instance or "rolling_hills"

    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        inst = generate_instance(instance_name)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"[Terrain kernel] Baseline evaluation: {instance_name}")
    print(f"  target_hurst: {inst.target_hurst:.3f}  target_norm_slope: {inst.target_norm_slope:.3f}")
    print(f"  {inst.description}")
    print()

    from . import build_candidate_source
    from ..core import Verdict
    from ..engines.sandbox import compile_and_run

    seeds = get_seed_programs(instance_name)
    max_score = 0.0
    for name, code in seeds:
        cpp_source, stdin_data = build_candidate_source(code, inst)
        stdout, verdict, _lat = compile_and_run(cpp_source, language="cpp",
                                                stdin=stdin_data, executor=executor)
        if verdict != Verdict.OK or not stdout:
            print(f"  {name:<20}: FAILED TO COMPILE/RUN (verdict={verdict})")
            continue
        try:
            out = json.loads(stdout.strip().split("\n")[-1])
        except Exception:
            print(f"  {name:<20}: BAD OUTPUT: {stdout[:80]!r}")
            continue

        if not out.get("valid", False):
            print(f"  {name:<20}: INVALID (NaN/Inf in terrain output)")
            continue

        fitness = float(out.get("fitness", 0.0))
        H_est   = float(out.get("H_est", -1.0))
        ns      = float(out.get("norm_slope", -1.0))
        hr      = float(out.get("height_range", -1.0))
        hs      = float(out.get("hurst_score", -1.0))
        ss      = float(out.get("slope_score", -1.0))
        flag = ""
        if fitness > 0.95:
            flag = "  ⚠ oracle too easy (>0.95)"
        elif fitness < 0.30:
            flag = "  ⚠ oracle too hard (<0.30)"
        print(f"  {name:<20}: fitness={fitness:.4f}  H_est={H_est:.3f}  "
              f"ns={ns:.3f}  hr={hr:.3f}  "
              f"hurst_s={hs:.3f}  slope_s={ss:.3f}{flag}")
        max_score = max(max_score, fitness)

    print()
    if max_score > 0.95:
        print("WARNING: oracle too easy — adjust target_hurst in _TERRAIN_INSTANCE_CONFIGS.")
        return 1
    if max_score < 0.30:
        print("WARNING: oracle too hard — adjust target_hurst to match measured H_est.")
        return 1
    print(f"Oracle calibration OK (best seed={max_score:.4f}, target 0.50–0.90).")
    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    print("[Terrain kernel] Available geological instances:")
    for name, cfg in _TERRAIN_INSTANCE_CONFIGS.items():
        print(f"  {name:<20} H_target={cfg['target_hurst']:.2f}  "
              f"ns_target={cfg['target_norm_slope']:.2f}")
        print(f"    {cfg['description']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Terrain FunSearch kernel — geological height function evolution"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    run_p = sub.add_parser("run", help="Run FunSearch evolution loop")
    run_p.add_argument("--instances", default="rolling_hills",
                       help="Comma-separated instance names (default: rolling_hills)")
    run_p.add_argument("--generations", type=int, default=60)
    run_p.add_argument("--islands", type=int, default=4)
    run_p.add_argument("--population", type=int, default=10)
    run_p.add_argument("--migration-interval", type=int, default=8)
    run_p.add_argument("--temperature", type=float, default=0.9)
    run_p.add_argument("--candidates-per-island", type=int, default=1)
    run_p.add_argument("--max-concurrent-llm", type=int, default=4)
    run_p.add_argument("--target-fitness", type=float, default=None)
    run_p.add_argument("--max-requests", type=int, default=None)
    run_p.add_argument("--max-wall-seconds", type=float, default=None)
    run_p.add_argument("--plateau-generations", type=int, default=12)
    run_p.add_argument("--output-dir", default=None)
    run_p.add_argument("--allow-unsandboxed", action="store_true")
    run_p.add_argument("-v", "--verbose", action="store_true")

    # baselines
    base_p = sub.add_parser("baselines", help="Evaluate seed functions (oracle calibration gate)")
    base_p.add_argument("--instance", default="rolling_hills")
    base_p.add_argument("--allow-unsandboxed", action="store_true")
    base_p.add_argument("-v", "--verbose", action="store_true")

    # instances
    sub.add_parser("instances", help="List available geological instances")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    dispatch = {"run": cmd_run, "baselines": cmd_baselines, "instances": cmd_instances}
    handler = dispatch.get(args.cmd)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
