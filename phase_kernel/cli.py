"""Phase field kernel CLI.

Usage:
    python -m autobench.phase_kernel run --instances water_ice_freezing \\
        --generations 60 --islands 4 --population 10 --allow-unsandboxed

    python -m autobench.phase_kernel baselines --instance water_ice_freezing --allow-unsandboxed

    python -m autobench.phase_kernel instances
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    PhaseKernel,
    PhaseInstance,
    evaluate_on_instance,
    ensure_executor,
    generate_instance,
    get_seed_programs,
    build_candidate_source,
    _PHASE_INSTANCE_CONFIGS,
)
from ..kernel_base import KernelConfig
from ..core import Verdict
from ..sandbox import compile_and_run


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

    kernel = PhaseKernel(config)

    print("[Phase kernel] Starting FunSearch loop")
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
        print(f"\nBest reaction(): {best.id}")
        print(f"  fitness:       {best.fitness:.6f}")
        print(f"  worst_fitness: {best.worst_fitness:.6f}")
        print(f"  island: {best.island}, gen: {best.generation}")
        print(f"\nreaction() function:\n{best.code}")

    if results_path:
        print(f"\nResults: {results_path}")

    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Evaluate seed programs — oracle calibration gate.

    Expected scores: 0.50–0.90.
    Reports per-component scores (tanh, equil, width) so you can see
    which part of the oracle the seeds are failing.
    """
    instance_name = args.instance or "water_ice_freezing"

    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        inst = generate_instance(instance_name)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"[Phase kernel] Baseline evaluation: {instance_name}")
    print(f"  T={inst.temperature:.2f}  D={inst.D:.2f}  "
          f"steps={inst.n_steps}  grid={inst.grid_size}")
    print(f"  {inst.description}")
    print()

    seeds = get_seed_programs(instance_name)
    max_score = 0.0
    for name, code in seeds:
        cpp_source, stdin_data = build_candidate_source(code, inst)
        stdout, verdict, _lat = compile_and_run(cpp_source, language="cpp",
                                                stdin=stdin_data, executor=executor)
        if verdict != Verdict.OK or not stdout:
            print(f"  {name:<22}: FAILED (verdict={verdict})")
            continue
        try:
            out = json.loads(stdout.strip().split("\n")[-1])
        except Exception:
            print(f"  {name:<22}: BAD OUTPUT: {stdout[:80]!r}")
            continue

        if not out.get("valid", False):
            reason = out.get("reason", "unknown")
            print(f"  {name:<22}: INVALID ({reason})")
            continue

        fitness = float(out.get("fitness", 0.0))
        tc = float(out.get("tanh_score", -1.0))
        es = float(out.get("equil_score", -1.0))
        ws = float(out.get("width_score", -1.0))
        pl = float(out.get("phi_left", -1.0))
        pm = float(out.get("phi_mid",  -1.0))
        pr = float(out.get("phi_right", -1.0))

        flag = ""
        if fitness > 0.95:
            flag = "  ⚠ too easy"
        elif fitness < 0.30:
            flag = "  ⚠ too hard"
        print(f"  {name:<22}: fitness={fitness:.4f}  "
              f"tanh={tc:.3f}  equil={es:.3f}  width={ws:.3f}  "
              f"φ(L/M/R)={pl:.3f}/{pm:.3f}/{pr:.3f}{flag}")
        max_score = max(max_score, fitness)

    print()
    if max_score < 0.30:
        print("WARNING: oracle too hard — check PDE stability (reduce dt or increase D).")
        return 1
    if max_score > 0.95:
        print("WARNING: oracle too easy — tighten target_width or increase n_steps.")
        return 1
    print(f"Oracle calibration OK (best seed={max_score:.4f}).")
    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    print("[Phase kernel] Available thermodynamic instances:")
    for name, cfg in _PHASE_INSTANCE_CONFIGS.items():
        print(f"  {name:<25} T={cfg['temperature']:.2f}  D={cfg['D']:.2f}  "
              f"steps={cfg['n_steps']}")
        print(f"    {cfg['description']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase FunSearch kernel — Allen-Cahn driving force evolution"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    run_p = sub.add_parser("run", help="Run FunSearch evolution loop")
    run_p.add_argument("--instances", default="water_ice_freezing",
                       help="Comma-separated instance names (default: water_ice_freezing)")
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
    base_p = sub.add_parser("baselines", help="Evaluate seed programs (oracle calibration)")
    base_p.add_argument("--instance", default="water_ice_freezing")
    base_p.add_argument("--allow-unsandboxed", action="store_true")
    base_p.add_argument("-v", "--verbose", action="store_true")

    # instances
    sub.add_parser("instances", help="List available thermodynamic instances")

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
