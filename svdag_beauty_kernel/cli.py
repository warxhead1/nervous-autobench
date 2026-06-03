"""svdag_beauty kernel CLI.

    python -m autobench.svdag_beauty_kernel baselines --instance stratovolcano --allow-unsandboxed
    python -m autobench.svdag_beauty_kernel run --instances stratovolcano,lava_field \\
        --generations 30 --islands 4 --population 12 --allow-unsandboxed
    python -m autobench.svdag_beauty_kernel instances
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

from . import (
    SVDAGBeautyKernel, generate_instance, evaluate_on_instance,
    ensure_executor, SEED_PROGRAMS, CONTROL_PROGRAMS, _INSTANCE_FACTORIES,
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


def cmd_baselines(args: argparse.Namespace) -> int:
    """Derisking gate: seeds should score ~0.65-0.80, controls far below."""
    inst = generate_instance(args.instance)
    executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
    print(f"[svdag_beauty] baselines on '{inst.name}' — {inst.description}")
    print(f"  sandbox: {executor.sandbox_type}\n  seeds (want 0.65-0.80):")
    from . import score_occupancy  # noqa: F401  (ensure import path warm)
    for name, code in SEED_PROGRAMS:
        f = evaluate_on_instance(code, inst, executor, run_timeout=30.0)
        diag = getattr(inst, "_last_diag", {})
        sub = diag.get("subscores", {}) if isinstance(diag, dict) else {}
        print(f"    {name:<16} fitness={_fmt(f)}  {sub}")
    print("  controls (want << seeds):")
    for name, code in CONTROL_PROGRAMS:
        f = evaluate_on_instance(code, inst, executor, run_timeout=30.0)
        print(f"    {name:<16} fitness={_fmt(f)}")
    return 0


def _fmt(f) -> str:
    return f"{f:.4f}" if isinstance(f, float) else "FAIL"


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
    kernel = SVDAGBeautyKernel(config)
    print("[svdag_beauty] Starting FunSearch loop")
    print(f"  instances: {instances}")
    print(f"  islands: {args.islands} x {args.population} | generations: {args.generations}")
    print(f"  sandbox: {kernel.executor.sandbox_type}")
    programs = kernel.run()
    kernel.save_results(programs)
    print(f"\nStopped: {kernel.stop_reason} | gens: {kernel.generation} | LLM reqs: {kernel.llm_requests}")
    best = programs[0] if programs else None
    if best:
        print(f"\nBest: {best.id}  fitness={best.fitness:.6f} (island {best.island}, gen {best.generation})")
        print(f"\n{best.code}")
    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    print("[svdag_beauty] Available volcanic archetypes:")
    for name, (desc, seed, _over) in _INSTANCE_FACTORIES.items():
        print(f"  {name:<18} seed={seed}  {desc}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="svdag_beauty FunSearch kernel — volcanic SVDAG terrain")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="Run FunSearch evolution")
    rp.add_argument("--instances", default="stratovolcano,lava_field,caldera")
    rp.add_argument("--generations", type=int, default=30)
    rp.add_argument("--islands", type=int, default=4)
    rp.add_argument("--population", type=int, default=12)
    rp.add_argument("--migration-interval", type=int, default=10)
    rp.add_argument("--temperature", type=float, default=0.9)
    rp.add_argument("--candidates-per-island", type=int, default=1)
    rp.add_argument("--max-concurrent-llm", type=int, default=8)
    rp.add_argument("--target-fitness", type=float, default=None)
    rp.add_argument("--max-requests", type=int, default=None)
    rp.add_argument("--max-wall-seconds", type=float, default=None)
    rp.add_argument("--plateau-generations", type=int, default=None)
    rp.add_argument("--output-dir", default=None)
    rp.add_argument("--allow-unsandboxed", action="store_true",
                    help="DANGER: run untrusted candidate code without isolation")
    rp.add_argument("-v", "--verbose", action="store_true")

    bp = sub.add_parser("baselines", help="Evaluate seed + control programs (derisking gate)")
    bp.add_argument("--instance", default="stratovolcano")
    bp.add_argument("--allow-unsandboxed", action="store_true")
    bp.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("instances", help="List volcanic archetypes")

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
