"""TSP kernel CLI entry point.

Usage:
    python -m autobench.tsp_kernel run --instances berlin52,kroA100,eil101 \\
        --generations 50 --islands 5 --population 20

    python -m autobench.tsp_kernel compile --code-file heuristics/my_priority.cpp \\
        --instance berlin52

    python -m autobench.tsp_kernel eval --program-id island3_gen15_llm \\
        --instances berlin52,eil101,kroA100
"""

import argparse
import datetime
import logging
import sys
import tempfile
import json
from pathlib import Path

from . import (
    TSPKernel,
    KernelConfig,
    TSPInstance,
    fetch_tsplib_instance,
    ensure_sandboxed_executor,
    evaluate_on_instance,
    evaluate_fitness,
    CandidateProgram,
    KNOWN_OPTIMALS,
)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _default_output_dir(arg: str | None) -> Path:
    """Return a persistent output dir for this run.

    If the user passed --output-dir, honour it. Otherwise default to
    autobench/benchmarks/curriculum/<YYYY-MM-DD>/ (the existing convention
    for dated benchmark runs). Avoids the previous /tmp default that caused
    results to vanish on session restart.
    """
    if arg:
        return Path(arg)
    date = datetime.date.today().isoformat()
    base = Path(__file__).resolve().parents[1] / "benchmarks" / "curriculum" / date
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

    kernel = TSPKernel(config)

    print(f"[TSP kernel] Starting FunSearch loop")
    print(f"  instances: {instances}")
    print(f"  islands: {args.islands} x {args.population} programs")
    print(f"  generations: {args.generations}")

    programs = kernel.run()
    kernel.save_results(programs)

    print(f"\nStopped: {kernel.stop_reason}")
    print(f"  generations run: {kernel.generation} | MiniMax requests: {kernel.llm_requests}")

    best = programs[0] if programs else None
    if best:
        print(f"\nBest heuristic: {best.id}")
        print(f"  fitness (mean approx ratio): {best.fitness:.4f}")
        print(f"  worst-case ratio: {best.worst_fitness:.4f}")
        print(f"  island: {best.island}, gen: {best.generation}")
        print(f"\nPriority function:\n{best.priority_code}")

    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Test-compile and run a priority function against a single instance (sandboxed)."""
    code = Path(args.code_file).read_text()
    instance_name = args.instance or "berlin52"

    print(f"[TSP kernel] Compiling priority function from {args.code_file}")

    try:
        executor = ensure_sandboxed_executor(allow_unsandboxed=args.allow_unsandboxed)
        print(f"[OK] Sandbox: {executor.sandbox_type}")

        inst = fetch_tsplib_instance(instance_name)
        opt = KNOWN_OPTIMALS.get(instance_name.lower())
        if opt:
            inst.optimal_tour_length = opt

        result = evaluate_on_instance(code, inst, executor, run_timeout=30.0)
        if result is None:
            print("[ERROR] candidate failed to compile/run inside the sandbox")
            return 1

        print(f"Instance: {result.instance_name}")
        print(f"Tour length: {result.length:.2f}")
        if inst.optimal_tour_length:
            ratio = inst.optimal_tour_length / result.length
            print(f"Optimal: {inst.optimal_tour_length:.2f}")
            print(f"Approximation ratio: {ratio:.4f}")

        if args.output:
            Path(args.output).write_text(json.dumps({
                "tour": result.tour,
                "length": result.length,
                "instance": result.instance_name,
            }, indent=2))
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a saved program against benchmark instances."""
    results_path = Path(args.results_file)
    results = json.loads(results_path.read_text())

    instances = args.instances.split(",") if args.instances else list(KNOWN_OPTIMALS.keys())

    print(f"[TSP kernel] Evaluating top programs from {args.results_file}")
    print(f"  instances: {instances}")

    top_programs = results.get("top_programs", [])
    if not top_programs and "best_program" in results:
        top_programs = [results["best_program"]]

    for prog in top_programs[:args.top_n]:
        pid = prog["id"]
        code = prog["priority_code"]
        print(f"\n--- {pid} (fitness={prog.get('fitness', '?')}) ---")

        with tempfile.TemporaryDirectory() as work_dir:
            wd = Path(work_dir)
            candidate = CandidateProgram(
                id=pid,
                priority_code=code,
                island=prog.get("island", 0),
                generation=prog.get("generation", 0),
                source="llm",
            )

            inst_list = [fetch_tsplib_instance(n) for n in instances]
            for inst in inst_list:
                try:
                    mean_r, var_r, worst_r = evaluate_fitness(candidate, [inst], wd)
                    print(f"  {inst.name}: mean={mean_r:.4f} var={var_r:.4f} worst={worst_r:.4f}")
                except Exception as e:
                    print(f"  {inst.name}: ERROR {e}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TSP FunSearch kernel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="Run FunSearch loop")
    run_parser.add_argument("--instances", default="berlin52,eil101,kroA100", help="Comma-separated TSPLIB names")
    run_parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    run_parser.add_argument("--islands", type=int, default=5, help="Number of islands")
    run_parser.add_argument("--population", type=int, default=20, help="Programs per island")
    run_parser.add_argument("--migration-interval", type=int, default=10, help="Generations between migrations")
    run_parser.add_argument("--temperature", type=float, default=0.9, help="LLM sampling temperature")
    run_parser.add_argument("--candidates-per-island", type=int, default=1, help="LLM candidates each island generates per generation")
    run_parser.add_argument("--max-concurrent-llm", type=int, default=8, help="Max concurrent LLM calls")
    # Horizon governance (generations is the hard cap; these stop earlier).
    run_parser.add_argument("--target-fitness", type=float, default=None, help="Stop when best approx ratio >= this")
    run_parser.add_argument("--max-requests", type=int, default=None, help="MiniMax request budget (billing unit)")
    run_parser.add_argument("--max-wall-seconds", type=float, default=None, help="Wall-clock budget in seconds")
    run_parser.add_argument("--plateau-generations", type=int, default=None, help="Stop after N generations with no improvement")
    run_parser.add_argument("--output-dir", default=None, help="Directory for results")
    run_parser.add_argument("--allow-unsandboxed", action="store_true",
                            help="DANGER: run untrusted candidate code without isolation (trusted/attended only)")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    compile_parser = sub.add_parser("compile", help="Test-compile a priority function")
    compile_parser.add_argument("--code-file", required=True, help="Path to C++ priority function")
    compile_parser.add_argument("--instance", default="berlin52", help="TSPLIB instance name")
    compile_parser.add_argument("--output", default=None, help="Write result JSON to path")
    compile_parser.add_argument("--allow-unsandboxed", action="store_true",
                                help="DANGER: run untrusted candidate code without isolation (trusted/attended only)")
    compile_parser.add_argument("-v", "--verbose", action="store_true")

    eval_parser = sub.add_parser("eval", help="Evaluate saved programs")
    eval_parser.add_argument("--results-file", required=True, help="Path to results JSON")
    eval_parser.add_argument("--instances", default=None, help="Comma-separated instances to test")
    eval_parser.add_argument("--top-n", type=int, default=3, help="How many top programs to evaluate")

    args = parser.parse_args()
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    if args.cmd == "run":
        return cmd_run(args)
    elif args.cmd == "compile":
        return cmd_compile(args)
    elif args.cmd == "eval":
        return cmd_eval(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())