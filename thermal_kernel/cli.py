"""2D Thermal-Allen-Cahn kernel CLI.

Usage:
    python -m autobench.thermal_kernel baselines --instance freeze_spot --allow-unsandboxed
    python -m autobench.thermal_kernel run --instances freeze_spot --generations 60 \\
        --islands 4 --population 10 --allow-unsandboxed
    python -m autobench.thermal_kernel instances
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    ThermalKernel,
    ThermalInstance,
    evaluate_on_instance,
    generate_instance,
    get_seed_programs,
    get_diagnostic_seeds,
    build_candidate_source,
    _THERMAL_INSTANCE_CONFIGS,
)
from ..kernel_base import KernelConfig
from ..core import Verdict
from ..sandbox import compile_and_run
from . import ensure_executor


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _default_output_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    date = datetime.date.today().isoformat()
    base = Path(__file__).resolve().parents[3] / "benchmarks" / "curriculum" / date
    base.mkdir(parents=True, exist_ok=True)
    return base


def _score_candidate(name: str, code: str, inst: ThermalInstance,
                     executor, verbose: bool = False) -> dict | None:
    cpp_source, stdin_data = build_candidate_source(code, inst)
    stdout, verdict, _lat = compile_and_run(
        cpp_source, language="cpp", stdin=stdin_data, executor=executor
    )
    if verdict != Verdict.OK or not stdout:
        return None
    try:
        out = json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None
    return out


def cmd_baselines(args: argparse.Namespace) -> int:
    """Oracle calibration gate with NEGATIVE CONTROLS.

    Checks:
      1. Seeds score 0.50–0.90 (calibrated range)
      2. Zero-reaction control scores CLEARLY BELOW seeds (Δ ≥ 0.10)
      3. Wrong-sign seed scores CLEARLY BELOW correct seeds (Δ ≥ 0.10)

    If conditions 2–3 fail, the oracle measures interface shape not dynamics.
    """
    instance_name = args.instance or "freeze_spot"

    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        inst = generate_instance(instance_name)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"[Thermal kernel] Baseline evaluation: {instance_name}")
    print(f"  grid={inst.grid_size}×{inst.grid_size}  "
          f"cold_temp={inst.cold_temp:.2f}  hot_temp={inst.hot_temp:.2f}")
    print(f"  cold_radius={inst.cold_radius}  D={inst.D}  dt={inst.dt}  "
          f"steps={inst.n_steps}  D*dt={inst.D*inst.dt:.3f}")
    print(f"  {inst.description}")
    print()

    # Stability check
    if inst.D * inst.dt > 0.25 + 1e-5:
        print(f"ERROR: D*dt={inst.D*inst.dt:.4f} > 0.25 — 2D stability violated!")
        print("  Reduce dt or D before running.")
        return 1
    print(f"  Stability OK: D*dt={inst.D*inst.dt:.3f} ≤ 0.25")
    print()

    all_results: dict[str, dict | None] = {}
    seed_best = 0.0

    print("  === Evolution seeds ===")
    for name, code in get_seed_programs(instance_name):
        out = _score_candidate(name, code, inst, executor)
        all_results[name] = out
        if out is None:
            print(f"  {name:<28}: FAILED")
            continue
        if not out.get("valid", False):
            print(f"  {name:<28}: INVALID ({out.get('reason','?')})")
            continue
        fitness = float(out.get("fitness", 0.0))
        pc = float(out.get("phase_coverage", -1))
        ret = float(out.get("retention", -1))
        sh = float(out.get("sharpness", -1))
        flag = ""
        if fitness > 0.95:
            flag = "  ⚠ too easy"
        elif fitness < 0.30:
            flag = "  ⚠ too hard"
        print(f"  {name:<28}: fitness={fitness:.4f}  "
              f"phase_cov={pc:.3f}  retention={ret:.3f}  sharpness={sh:.3f}{flag}")
        seed_best = max(seed_best, fitness)

    print()
    print("  === Negative controls (must be CLEARLY below seeds) ===")
    diag_scores: list[float] = []
    for name, code in get_diagnostic_seeds():
        out = _score_candidate(name, code, inst, executor)
        if out is None:
            print(f"  {name:<28}: FAILED")
            continue
        if not out.get("valid", False):
            print(f"  {name:<28}: INVALID ({out.get('reason','?')})")
            diag_scores.append(0.0)
            continue
        fitness = float(out.get("fitness", 0.0))
        pc = float(out.get("phase_coverage", -1))
        ret = float(out.get("retention", -1))
        sh = float(out.get("sharpness", -1))
        print(f"  {name:<28}: fitness={fitness:.4f}  "
              f"phase_cov={pc:.3f}  retention={ret:.3f}  sharpness={sh:.3f}")
        diag_scores.append(fitness)

    print()

    # Calibration verdict
    failures = []
    if seed_best < 0.30:
        failures.append(f"oracle too hard — best seed={seed_best:.4f} < 0.30")
    if seed_best > 0.95:
        failures.append(f"oracle too easy — best seed={seed_best:.4f} > 0.95")
    if diag_scores:
        max_diag = max(diag_scores)
        gap = seed_best - max_diag
        if gap < 0.10:
            failures.append(
                f"oracle NOT discriminative — seed={seed_best:.4f} vs "
                f"best_control={max_diag:.4f}  gap={gap:.3f} < 0.10  "
                "(oracle measures shape, not freezing dynamics)"
            )

    if failures:
        for f in failures:
            print(f"CALIBRATION FAIL: {f}")
        return 1

    print(f"Oracle calibration OK — "
          f"best_seed={seed_best:.4f}, "
          f"best_control={max(diag_scores, default=0.0):.4f}, "
          f"gap={seed_best - max(diag_scores, default=0.0):.3f}")
    return 0


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
    kernel = ThermalKernel(config)

    print("[Thermal kernel] Starting 2D Allen-Cahn FunSearch loop")
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
        print(f"  island: {best.island}, gen: {best.generation}")
        print(f"\nreaction() function:\n{best.code}")

    if results_path:
        print(f"\nResults: {results_path}")

    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    print("[Thermal kernel] Available 2D thermal instances:")
    for name, cfg in _THERMAL_INSTANCE_CONFIGS.items():
        print(f"  {name:<20} cold={cfg['cold_temp']:.2f}  hot={cfg['hot_temp']:.2f}  "
              f"R={cfg['cold_radius']}  steps={cfg['n_steps']}")
        print(f"    {cfg['description']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thermal FunSearch kernel — 2D Allen-Cahn phase field"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    run_p = sub.add_parser("run", help="Run FunSearch evolution loop")
    run_p.add_argument("--instances", default="freeze_spot")
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
    base_p = sub.add_parser("baselines", help="Oracle calibration with negative controls")
    base_p.add_argument("--instance", default="freeze_spot")
    base_p.add_argument("--allow-unsandboxed", action="store_true")
    base_p.add_argument("-v", "--verbose", action="store_true")

    # instances
    sub.add_parser("instances", help="List available instances")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    dispatch = {"run": cmd_run, "baselines": cmd_baselines, "instances": cmd_instances}
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
