"""Oasis kernel CLI.

    python -m autobench.oasis_kernel run --instances clear_spring --allow-unsandboxed
    python -m autobench.oasis_kernel baselines --instance clear_spring --allow-unsandboxed
    python -m autobench.oasis_kernel instances
    python -m autobench.oasis_kernel render --instance clear_spring --allow-unsandboxed
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    OasisKernel,
    evaluate_on_instance,  # noqa: F401
    ensure_sandboxed_executor as ensure_executor,
    generate_instance,
    get_seed_programs,
    build_candidate_source,
    render_oasis,
    _OASIS_INSTANCE_CONFIGS,
)
from ..kernels import KernelConfig
from ..core import Verdict
from ..engines.sandbox import compile_and_run


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _default_output_dir(arg):
    if arg:
        return Path(arg)
    base = Path(__file__).resolve().parents[2] / "benchmarks" / "curriculum" / datetime.date.today().isoformat()
    base.mkdir(parents=True, exist_ok=True)
    return base


def cmd_run(args) -> int:
    instances = args.instances.split(",")
    config = KernelConfig(
        instances=instances, n_islands=args.islands,
        population_per_island=args.population, generations=args.generations,
        migration_interval=args.migration_interval, temperature=args.temperature,
        output_dir=_default_output_dir(args.output_dir),
        allow_unsandboxed=args.allow_unsandboxed,
        candidates_per_island=args.candidates_per_island,
        max_concurrent_llm=args.max_concurrent_llm, target_fitness=args.target_fitness,
        max_requests=args.max_requests, max_wall_seconds=args.max_wall_seconds,
        plateau_generations=args.plateau_generations,
    )
    kernel = OasisKernel(config)
    print("[Oasis kernel] Starting FunSearch loop")
    print(f"  instances: {instances}  islands: {args.islands}×{args.population}  sandbox: {kernel.executor.sandbox_type}")
    programs = kernel.run()
    results_path = kernel.save_results(programs)
    print(f"\nStopped: {kernel.stop_reason}  gens={kernel.generation}  llm_requests={kernel.llm_requests}")
    best = programs[0] if programs else None
    if best:
        print(f"\nBest flux(): {best.id}  fitness={best.fitness:.6f}  island={best.island} gen={best.generation}")
        print(f"\n{best.code}")
    if results_path:
        print(f"\nResults: {results_path}")
    return 0


def cmd_baselines(args) -> int:
    name = args.instance or "clear_spring"
    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        inst = generate_instance(name)
    except Exception as e:
        print(f"[ERROR] {e}"); return 1
    print(f"[Oasis kernel] Baseline evaluation: {name}")
    print(f"  visc={inst.viscosity:.3f}  spring={inst.spring:.1f}  artesian={inst.artesian:.2f}  steps={inst.n_steps}")
    print(f"  {inst.description}\n")
    max_score = 0.0
    for label, code in get_seed_programs(name):
        cpp, stdin_data = build_candidate_source(code, inst)
        stdout, verdict, _ = compile_and_run(cpp, language="cpp", stdin=stdin_data, executor=executor)
        if verdict != Verdict.OK or not stdout:
            print(f"  {label:<14}: FAILED (verdict={verdict})"); continue
        try:
            out = json.loads(stdout.strip().split("\n")[-1])
        except Exception:
            print(f"  {label:<14}: BAD OUTPUT {stdout[:80]!r}"); continue
        if not out.get("valid", False):
            print(f"  {label:<14}: INVALID ({out.get('reason','?')})"); continue
        f = float(out.get("fitness", 0))
        flag = "  ⚠ too easy" if f > 0.95 else ("  ⚠ too hard" if f < 0.30 else "")
        print(f"  {label:<14}: fitness={f:.4f}  cap={out['basin_capture']:.2f} "
              f"stab={out['stability']:.2f} pool={out['pool_fraction']:.2f} "
              f"breathe={out['breathing']:.2f}  (wet={out['wet']:.3f}){flag}")
        max_score = max(max_score, f)
    print()
    if max_score < 0.30:
        print("WARNING: oracle too hard."); return 1
    if max_score > 0.95:
        print("WARNING: oracle too easy."); return 1
    print(f"Oracle calibration OK (best seed={max_score:.4f}).")
    return 0


def cmd_instances(args) -> int:
    print("[Oasis kernel] Available oasis instances:")
    for name, cfg in _OASIS_INSTANCE_CONFIGS.items():
        print(f"  {name:<14} visc={cfg['viscosity']:.3f}  artesian={cfg['artesian']:.2f}")
        print(f"    {cfg['description']}")
    return 0


def cmd_render(args) -> int:
    name = args.instance or "clear_spring"
    executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
    inst = generate_instance(name)
    code = get_seed_programs(name)[0][1]
    out = Path(args.output or f"oasis_{name}.png")
    ok = render_oasis(code, inst, executor, out)
    print(f"render {'OK -> ' + str(out) if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Oasis FunSearch kernel — shallow-water flow evolution")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run FunSearch evolution loop")
    r.add_argument("--instances", default="clear_spring")
    r.add_argument("--generations", type=int, default=40)
    r.add_argument("--islands", type=int, default=4)
    r.add_argument("--population", type=int, default=8)
    r.add_argument("--migration-interval", type=int, default=8)
    r.add_argument("--temperature", type=float, default=0.9)
    r.add_argument("--candidates-per-island", type=int, default=1)
    r.add_argument("--max-concurrent-llm", type=int, default=4)
    r.add_argument("--target-fitness", type=float, default=None)
    r.add_argument("--max-requests", type=int, default=None)
    r.add_argument("--max-wall-seconds", type=float, default=None)
    r.add_argument("--plateau-generations", type=int, default=10)
    r.add_argument("--output-dir", default=None)
    r.add_argument("--allow-unsandboxed", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true")
    b = sub.add_parser("baselines", help="Evaluate seed flow laws (oracle calibration)")
    b.add_argument("--instance", default="clear_spring")
    b.add_argument("--allow-unsandboxed", action="store_true")
    b.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("instances", help="List oasis instances")
    rn = sub.add_parser("render", help="Render the seed oasis to a filmstrip PNG")
    rn.add_argument("--instance", default="clear_spring")
    rn.add_argument("--output", default=None)
    rn.add_argument("--allow-unsandboxed", action="store_true")
    rn.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args()
    setup_logging(getattr(args, "verbose", False))
    handler = {"run": cmd_run, "baselines": cmd_baselines,
               "instances": cmd_instances, "render": cmd_render}.get(args.cmd)
    if handler:
        return handler(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
