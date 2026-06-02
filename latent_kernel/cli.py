"""Latent heat kernel CLI.

Usage:
    python -m autobench.latent_kernel baselines --instance freeze_latent --allow-unsandboxed
    python -m autobench.latent_kernel run --instances freeze_latent --generations 80 \\
        --islands 4 --population 12 --allow-unsandboxed
    python -m autobench.latent_kernel instances
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    LatentKernel,
    LatentInstance,
    generate_instance,
    get_seed_programs,
    get_diagnostic_seeds,
    build_candidate_source,
    ensure_sandboxed_executor as ensure_executor,
    _LATENT_INSTANCE_CONFIGS,
)
from ..kernels import KernelConfig
from ..core import Verdict
from ..engines.sandbox import compile_and_run


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


def _score(name: str, code: str, inst: LatentInstance, executor) -> dict | None:
    cpp_source, stdin_data = build_candidate_source(code, inst)
    stdout, verdict, _ = compile_and_run(
        cpp_source, language="cpp", stdin=stdin_data, executor=executor
    )
    if verdict != Verdict.OK or not stdout:
        return None
    try:
        return json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None


def cmd_baselines(args: argparse.Namespace) -> int:
    instance_name = args.instance or "freeze_latent"
    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        inst = generate_instance(instance_name)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"[Latent kernel] Baseline: {instance_name}")
    print(f"  grid={inst.grid_size}×{inst.grid_size}  cold={inst.cold_temp:.2f}  "
          f"hot={inst.hot_temp:.2f}  L={inst.L:.2f}  D_T={inst.D_T}")
    print(f"  φ_eq={inst.phi_eq:.3f}  steps={inst.n_steps}  D_phi*dt={inst.D_phi*inst.dt:.3f}")
    print(f"  {inst.description}")
    print()

    if inst.D_phi * inst.dt > 0.25 + 1e-5 or inst.D_T * inst.dt > 0.25 + 1e-5:
        print(f"ERROR: stability violated. D_phi*dt={inst.D_phi*inst.dt:.4f}, "
              f"D_T*dt={inst.D_T*inst.dt:.4f} — both must be ≤ 0.25")
        return 1
    print(f"  Stability OK: D_phi*dt={inst.D_phi*inst.dt:.3f}, "
          f"D_T*dt={inst.D_T*inst.dt:.3f} ≤ 0.25")
    print()

    seed_best = 0.0
    print("  === Evolution seeds ===")
    for name, code in get_seed_programs(instance_name):
        out = _score(name, code, inst, executor)
        if out is None:
            print(f"  {name:<28}: FAILED")
            continue
        if not out.get("valid", False):
            print(f"  {name:<28}: INVALID ({out.get('reason','?')})")
            continue
        f = float(out.get("fitness", 0))
        Tb   = float(out.get("T_balance",  -1))
        sh   = float(out.get("sharpness",  -1))
        ret  = float(out.get("retention",  -1))
        mp   = float(out.get("mean_phi_cold", -1))
        mT   = float(out.get("mean_T_cold",   -1))
        flag = "  ⚠ too easy" if f > 0.95 else ("  ⚠ too hard" if f < 0.30 else "")
        print(f"  {name:<28}: fit={f:.4f}  Tbal={Tb:.3f}  sh={sh:.3f}  "
              f"ret={ret:.3f}  φ={mp:.3f}  T={mT:.3f}{flag}")
        seed_best = max(seed_best, f)

    print()
    diag_scores: list[float] = []
    print("  === Negative controls ===")
    for name, code in get_diagnostic_seeds():
        out = _score(name, code, inst, executor)
        if out is None:
            print(f"  {name:<28}: FAILED")
            continue
        if not out.get("valid", False):
            print(f"  {name:<28}: INVALID ({out.get('reason','?')})")
            diag_scores.append(0.0)
            continue
        f = float(out.get("fitness", 0))
        Tb  = float(out.get("T_balance", -1))
        sh  = float(out.get("sharpness", -1))
        print(f"  {name:<28}: fit={f:.4f}  Tbal={Tb:.3f}  sh={sh:.3f}")
        diag_scores.append(f)

    print()
    failures = []
    if seed_best < 0.30:
        failures.append(f"oracle too hard — best_seed={seed_best:.4f} < 0.30")
    if seed_best > 0.95:
        failures.append(f"oracle too easy — best_seed={seed_best:.4f} > 0.95")
    max_diag = max(diag_scores, default=0.0)
    gap = seed_best - max_diag
    if gap < 0.10:
        failures.append(
            f"oracle NOT discriminative — seed={seed_best:.4f} vs "
            f"control={max_diag:.4f}  gap={gap:.3f} < 0.10"
        )
    if failures:
        for f in failures:
            print(f"CALIBRATION FAIL: {f}")
        return 1
    print(f"Oracle calibration OK — best_seed={seed_best:.4f}  "
          f"best_control={max_diag:.4f}  gap={gap:.3f}")
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
    kernel = LatentKernel(config)
    print("[Latent kernel] 2D coupled phase-thermal FunSearch")
    print(f"  instances: {instances}")
    print(f"  islands: {args.islands} × {args.population}")
    print(f"  sandbox: {kernel.executor.sandbox_type}")

    programs = kernel.run()
    results_path = kernel.save_results(programs)

    print(f"\nStopped: {kernel.stop_reason}")
    print(f"  gens={kernel.generation}  reqs={kernel.llm_requests}")

    best = programs[0] if programs else None
    if best:
        print(f"\nBest: {best.id}  fitness={best.fitness:.6f}")
        print(f"\n{best.code}")
    if results_path:
        print(f"\nResults: {results_path}")
    return 0


def cmd_instances(_: argparse.Namespace) -> int:
    print("[Latent kernel] Available instances:")
    for name, cfg in _LATENT_INSTANCE_CONFIGS.items():
        inst = generate_instance(name)
        print(f"  {name:<22} φ_eq={inst.phi_eq:.3f}  L={cfg['L']:.2f}  "
              f"T_cold={cfg['cold_temp']:.2f}  steps={cfg['n_steps']}")
        print(f"    {cfg['description'][:80]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Latent heat FunSearch kernel — coupled phase-thermal PDE"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--instances", default="freeze_latent")
    run_p.add_argument("--generations", type=int, default=80)
    run_p.add_argument("--islands", type=int, default=4)
    run_p.add_argument("--population", type=int, default=12)
    run_p.add_argument("--migration-interval", type=int, default=8)
    run_p.add_argument("--temperature", type=float, default=0.9)
    run_p.add_argument("--candidates-per-island", type=int, default=1)
    run_p.add_argument("--max-concurrent-llm", type=int, default=4)
    run_p.add_argument("--target-fitness", type=float, default=None)
    run_p.add_argument("--max-requests", type=int, default=None)
    run_p.add_argument("--max-wall-seconds", type=float, default=None)
    run_p.add_argument("--plateau-generations", type=int, default=15)
    run_p.add_argument("--output-dir", default=None)
    run_p.add_argument("--allow-unsandboxed", action="store_true")
    run_p.add_argument("-v", "--verbose", action="store_true")

    base_p = sub.add_parser("baselines")
    base_p.add_argument("--instance", default="freeze_latent")
    base_p.add_argument("--allow-unsandboxed", action="store_true")
    base_p.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("instances")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    dispatch = {"run": cmd_run, "baselines": cmd_baselines, "instances": cmd_instances}
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
