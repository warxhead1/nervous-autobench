"""SDF kernel CLI entry point.

Usage:
    python -m autobench.sdf_kernel run \\
        --instances gyroid,round_box,twisted_torus \\
        --generations 50 --islands 4 --population 20 \\
        --allow-unsandboxed

    python -m autobench.sdf_kernel compile \\
        --code-file my_sdf.cpp --instance gyroid --allow-unsandboxed

    python -m autobench.sdf_kernel eval \\
        --results-file sdf_results_gen50.json --instances sphere,round_box

    python -m autobench.sdf_kernel instances
        -- list available benchmark instances
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from . import (
    SDFKernel,
    SDFInstance,
    evaluate_on_instance,
    ensure_sandboxed_executor as ensure_executor,
    generate_instance,
    build_candidate_source,
    get_seed_programs,
    KNOWN_OPTIMALS,
    _INSTANCE_FACTORIES,
)
from ..kernels import KernelConfig, CandidateProgram


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
    """Run the full FunSearch loop."""
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

    kernel = SDFKernel(config)

    print("[SDF kernel] Starting FunSearch loop")
    print(f"  instances: {instances}")
    print(f"  islands: {args.islands} x {args.population} programs")
    print(f"  generations: {args.generations}")
    print(f"  sandbox: {kernel.executor.sandbox_type}")

    programs = kernel.run()
    kernel.save_results(programs)

    print(f"\nStopped: {kernel.stop_reason}")
    print(f"  generations run: {kernel.generation} | LLM requests: {kernel.llm_requests}")

    best = programs[0] if programs else None
    if best:
        print(f"\nBest sdf(): {best.id}")
        print(f"  fitness (1/(1+MSE)): {best.fitness:.6f}")
        print(f"  worst-case fitness:  {best.worst_fitness:.6f}")
        print(f"  island: {best.island}, gen: {best.generation}")
        print(f"\nsdf() function:\n{best.code}")

    if args.post_assess and config.output_dir:
        import glob
        result_files = sorted(
            glob.glob(str(config.output_dir / "sdf_results_gen*.json")),
            key=lambda p: Path(p).stat().st_mtime,
        )
        if result_files:
            results_path = Path(result_files[-1])
            try:
                from autobench.post_run_assess import assess_run
                report_path, png_paths = assess_run(
                    results_path, kernel="sdf", top_n=args.assess_top_n,
                    nervous_bin=kernel._nervous_bin,
                )
                print(f"\n[observer] Report: {report_path}")
                print(f"[observer] Renders ({len(png_paths)}): " +
                      ", ".join(p.name for p in png_paths))
            except Exception as e:
                print(f"\n[observer] Assessment failed (non-fatal): {e}")

    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Test-compile and evaluate an sdf() function against one instance (sandboxed)."""
    code_path = Path(args.code_file)
    if not code_path.exists():
        print(f"[ERROR] Code file not found: {code_path}")
        return 1
    code = code_path.read_text()
    instance_name = args.instance or "sphere"

    print(f"[SDF kernel] Compiling sdf() from {code_path}")

    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        print(f"[OK] Sandbox: {executor.sandbox_type}")

        inst = generate_instance(instance_name)
        result = evaluate_on_instance(code, inst, executor, run_timeout=30.0)

        if result is None:
            print("[ERROR] Candidate failed to compile/run inside the sandbox")
            return 1

        print(f"Instance: {inst.name} ({inst.n_samples} sample points)")
        print(f"Fitness (1/(1+MSE)): {result:.6f}")
        mse = 1.0 / result - 1.0
        print(f"MSE: {mse:.9f}")
        print(f"Known optimal MSE: {KNOWN_OPTIMALS.get(instance_name, '?')}")

        if args.output:
            Path(args.output).write_text(json.dumps({
                "instance": inst.name,
                "fitness": result,
                "mse": mse,
            }, indent=2))
            print(f"Result written to {args.output}")

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate top programs from a saved results file against benchmark instances."""
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return 1

    results = json.loads(results_path.read_text())
    instances_arg = args.instances.split(",") if args.instances else list(KNOWN_OPTIMALS.keys())

    print(f"[SDF kernel] Evaluating top programs from {results_path}")
    print(f"  instances: {instances_arg}")

    top_programs = results.get("top_programs", [])
    if not top_programs and "best_program" in results:
        top_programs = [results["best_program"]]

    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
    except Exception as e:
        print(f"[ERROR] Sandbox: {e}")
        return 1

    for prog in top_programs[:args.top_n]:
        pid = prog["id"]
        # Results JSON uses 'sdf_code' key; handle both for backward compat
        code = prog.get("sdf_code") or prog.get("code") or prog.get("priority_code", "")
        print(f"\n--- {pid} (stored_fitness={prog.get('fitness', '?'):.4f}) ---")

        for inst_name in instances_arg:
            try:
                inst = generate_instance(inst_name)
                fitness = evaluate_on_instance(code, inst, executor, run_timeout=30.0)
                if fitness is None:
                    print(f"  {inst_name}: FAILED")
                else:
                    mse = 1.0 / fitness - 1.0
                    print(f"  {inst_name}: fitness={fitness:.6f} MSE={mse:.9f}")
            except Exception as e:
                print(f"  {inst_name}: ERROR {e}")

    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Evaluate all baseline seed programs for a given instance."""
    instance_name = args.instance or "sphere"

    try:
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        inst = generate_instance(instance_name)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"[SDF kernel] Baseline evaluation for instance: {instance_name}")
    print(f"  {inst.n_samples} sample points, bbox=[{inst.bbox[0]}, {inst.bbox[1]}]")
    print(f"  Description: {inst.description}")
    print()

    seeds = get_seed_programs(instance_name)
    for variant_name, code in seeds:
        fitness = evaluate_on_instance(code, inst, executor, run_timeout=30.0)
        if fitness is None:
            print(f"  {variant_name}: FAILED TO COMPILE/RUN")
        else:
            mse = 1.0 / fitness - 1.0
            print(f"  {variant_name}: fitness={fitness:.6f} MSE={mse:.9f}")

    return 0


def cmd_instances(args: argparse.Namespace) -> int:
    """List available SDF benchmark instances."""
    print("[SDF kernel] Available benchmark instances:")
    for name, (fn, description, lo, hi, n) in _INSTANCE_FACTORIES.items():
        optimal = KNOWN_OPTIMALS.get(name, "?")
        print(f"  {name:<20} n_samples={n:<6} bbox=[{lo},{hi}]  optimal_MSE={optimal}")
        print(f"    {description}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SDF FunSearch kernel — shader math optimization")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run ---
    run_parser = sub.add_parser("run", help="Run FunSearch evolution loop")
    run_parser.add_argument(
        "--instances", default="sphere,round_box",
        help="Comma-separated instance names (default: sphere,round_box). "
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
                            help="Stop when best fitness >= this (1.0 = exact match)")
    run_parser.add_argument("--max-requests", type=int, default=None,
                            help="LLM request budget (billing unit)")
    run_parser.add_argument("--max-wall-seconds", type=float, default=None,
                            help="Wall-clock budget in seconds")
    run_parser.add_argument("--plateau-generations", type=int, default=None,
                            help="Stop after N gens with no improvement")
    run_parser.add_argument("--output-dir", default=None, help="Directory for results JSON")
    run_parser.add_argument(
        "--allow-unsandboxed", action="store_true",
        help="DANGER: run untrusted candidate code without isolation (trusted/attended only)"
    )
    run_parser.add_argument("-v", "--verbose", action="store_true")
    run_parser.add_argument(
        "--post-assess", action="store_true",
        help="Run MiniMax observer assessment after the run: render top programs, "
             "structural analysis, synthesis report."
    )
    run_parser.add_argument(
        "--assess-top-n", type=int, default=5,
        help="Number of top programs to render and assess (default: 5)"
    )

    # --- compile ---
    compile_parser = sub.add_parser("compile", help="Test-compile an sdf() function")
    compile_parser.add_argument("--code-file", required=True, help="Path to C++ sdf() function")
    compile_parser.add_argument("--instance", default="sphere",
                                help="SDF instance name (default: sphere)")
    compile_parser.add_argument("--output", default=None, help="Write result JSON to this path")
    compile_parser.add_argument("--allow-unsandboxed", action="store_true",
                                help="DANGER: run without isolation")
    compile_parser.add_argument("-v", "--verbose", action="store_true")

    # --- eval ---
    eval_parser = sub.add_parser("eval", help="Evaluate saved programs from results JSON")
    eval_parser.add_argument("--results-file", required=True, help="Path to results JSON")
    eval_parser.add_argument("--instances", default=None,
                             help="Comma-separated instances to evaluate against")
    eval_parser.add_argument("--top-n", type=int, default=3, help="How many top programs to eval")
    eval_parser.add_argument("--allow-unsandboxed", action="store_true",
                             help="DANGER: run without isolation")
    eval_parser.add_argument("-v", "--verbose", action="store_true")

    # --- baselines ---
    baselines_parser = sub.add_parser("baselines",
                                      help="Evaluate all seed baseline programs for an instance")
    baselines_parser.add_argument("--instance", default="sphere",
                                  help="SDF instance name (default: sphere)")
    baselines_parser.add_argument("--allow-unsandboxed", action="store_true",
                                  help="DANGER: run without isolation")
    baselines_parser.add_argument("-v", "--verbose", action="store_true")

    # --- instances ---
    sub.add_parser("instances", help="List available SDF benchmark instances")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    dispatch = {
        "run": cmd_run,
        "compile": cmd_compile,
        "eval": cmd_eval,
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
