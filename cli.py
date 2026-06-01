"""Autobench CLI — recursive self-improvement coding agent harness."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

__version__ = "0.1.0"


def cmd_run(args):
    """Run harness on a benchmark."""
    harness_path = Path(args.harness)
    benchmark_path = Path(args.benchmark)

    if not harness_path.exists():
        print(f"Error: harness config not found: {harness_path}", file=sys.stderr)
        return 1

    if not benchmark_path.exists():
        print(f"Error: benchmark not found: {benchmark_path}", file=sys.stderr)
        return 1

    harness_config = json.load(open(harness_path))
    benchmark_config = json.load(open(benchmark_path))

    print(f"Running harness {args.harness} on benchmark {args.benchmark}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autobench.core", "run",
             "--harness", str(harness_path),
             "--benchmark", str(benchmark_path)],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        print("Error: autobench.core not found. Is autobench installed?", file=sys.stderr)
        return 1


def cmd_improve(args):
    """Run RSI loop on a benchmark."""
    benchmark_path = Path(args.benchmark)
    iterations = args.iterations

    if not benchmark_path.exists():
        print(f"Error: benchmark not found: {benchmark_path}", file=sys.stderr)
        return 1

    print(f"Running RSI loop on {args.benchmark} for {iterations} iterations")

    # nervous-bus-msqa: forward operator gates to the subprocess via env.
    # ContinuousModeDaemon.stage_promotion_from_population reads these.
    env = dict(os.environ)
    if getattr(args, "confirm_promotion", False):
        env["AUTOBENCH_CONFIRM_PROMOTION"] = "1"
    if getattr(args, "reject_promotion", False):
        env["AUTOBENCH_REJECT_PROMOTION"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autobench.rsi_loop", "run",
             "--benchmark", str(benchmark_path),
             "--iterations", str(iterations)],
            capture_output=True,
            text=True,
            env=env,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        print("Error: autobench.rsi_loop not found.", file=sys.stderr)
        return 1


def cmd_eval(args):
    """Analyze repo and classify change type."""
    repo_path = Path(args.repo)

    if not repo_path.exists():
        print(f"Error: repo not found: {repo_path}", file=sys.stderr)
        return 1

    print(f"Analyzing repository: {args.repo}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autobench.audit.repo_analyzer", "analyze",
             "--repo", str(repo_path)],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        output = json.loads(result.stdout)
        change_type = output.get("change_type", "unknown")
        print(f"\nClassified change type: {change_type}")
        return result.returncode
    except FileNotFoundError:
        from autobench.audit.repo_analyzer import analyze_repo
        analysis = analyze_repo(repo_path)
        print(json.dumps(analysis.to_dict(), indent=2))
        change_type = "unknown"
        print(f"\nClassified change type: {change_type}")
        return 0
    except json.JSONDecodeError:
        print("Error: failed to parse analyzer output", file=sys.stderr)
        return 1


def cmd_scaffold(args):
    """Generate test scaffolding for a repo."""
    repo_path = Path(args.repo)
    scaffold_type = args.type

    if not repo_path.exists():
        print(f"Error: repo not found: {repo_path}", file=sys.stderr)
        return 1

    print(f"Generating {scaffold_type} scaffolding for: {args.repo}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autobench.test_scaffolder", "scaffold",
             "--repo", str(repo_path),
             "--type", scaffold_type],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        from .test_scaffolder import TestScaffolder
        scaffolder = TestScaffolder(repo_path, scaffold_type)
        result = scaffolder.generate()
        print(json.dumps(result, indent=2))
        return 0


def cmd_sandbox(args):
    """Execute code in sandbox."""
    language = args.language
    code = args.code

    print(f"Executing {language} code in sandbox")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autobench.sandbox", "run",
             "--language", language,
             "--code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        from .engines.sandbox import SandboxExecutor
        executor = SandboxExecutor(timeout=30)
        result = executor.execute(code, language)
        print(json.dumps(result, indent=2))
        return 0
    except subprocess.TimeoutExpired:
        print("Error: sandbox execution timed out", file=sys.stderr)
        return 124


def cmd_trigger_daemon(args):
    """Start the producer-triggered cycle daemon (nervous-bus-1hlf).

    Subscribes to ``autobench.cycle.requested.v1`` via ``deer obs bus`` and
    runs one cycle per validated trigger. Exits after ``--max-runs`` if set.
    """
    from autobench.daemons.trigger_daemon import TriggerDaemon

    bead_id = getattr(args, "bead_id", None)
    max_runs = getattr(args, "max_runs", None)

    daemon = TriggerDaemon(bead_id_default=bead_id)
    handled = daemon.listen(max_runs=max_runs)
    print(f"[trigger-daemon] handled {handled} trigger(s); exiting")
    return 0


def cmd_version(args):
    """Show version."""
    print(f"autobench {__version__}")
    return 0


def cmd_replay(args):
    """Counterfactual replay — rerun a captured iteration with a forced override.

    Antagonist's 'I bet you would have lost here' weapon: reads the autobench
    JSONL capture, reconstructs the harness config at the start of the chosen
    iteration, applies ``--override`` mutations, replays the exact same
    benchmark cases, and prints (or writes) a comparison report.
    """
    from .rsi.replay import (
        CounterfactualRunner,
        ReplayLoader,
        filter_cases_by_id,
        harness_dict_to_config,
        load_cases_from_dir,
        merge_overrides,
    )

    debug_file = Path(args.debug_file).expanduser()
    if not debug_file.exists():
        print(f"Error: debug file not found: {debug_file}", file=sys.stderr)
        return 1

    loader = ReplayLoader(debug_file)
    if args.session_id not in loader.sessions():
        print(f"Error: session_id {args.session_id!r} not found in {debug_file}", file=sys.stderr)
        print(f"Known sessions: {loader.sessions()[:5]}{'...' if len(loader.sessions()) > 5 else ''}", file=sys.stderr)
        return 2

    iters = loader.iterations(args.session_id)
    if args.iteration not in iters:
        print(f"Error: iteration {args.iteration} not found for session {args.session_id}", file=sys.stderr)
        print(f"Available iterations: {iters}", file=sys.stderr)
        return 2

    try:
        override = merge_overrides(args.override or [])
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if not override:
        print("Error: at least one --override KEY=VAL is required", file=sys.stderr)
        return 2

    # Locate benchmark cases.
    if not args.benchmark_dir:
        print(
            "Error: --benchmark-dir is required (event payloads do not currently "
            "carry the benchmark name; supply the directory holding the case JSON files)",
            file=sys.stderr,
        )
        return 2
    try:
        all_cases = load_cases_from_dir(args.benchmark_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    captured_case_ids = loader.case_ids(args.session_id, args.iteration)
    cases = filter_cases_by_id(all_cases, captured_case_ids) if captured_case_ids else all_cases

    if not cases:
        print(
            f"Error: no benchmark cases matched. Captured case IDs: {captured_case_ids[:5]}; "
            f"directory: {args.benchmark_dir}",
            file=sys.stderr,
        )
        return 2

    harness_dict = loader.harness_at(args.session_id, args.iteration)
    original_harness = harness_dict_to_config(harness_dict)
    original_verdicts = loader.original_verdicts(args.session_id, args.iteration)
    original_score = loader.aggregate_score(args.session_id, args.iteration)

    from .evaluator import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator()
    runner = CounterfactualRunner(evaluator)
    comparison = runner.run(
        original_harness=original_harness,
        override=override,
        cases=cases,
        original_verdicts=original_verdicts,
        original_score=original_score,
    )
    # Carry unresolved-flip warnings through so render_text can show them.
    if harness_dict.get("_unresolved_flips"):
        comparison.original_harness["_unresolved_flips"] = harness_dict["_unresolved_flips"]
    comparison.session_id = args.session_id
    comparison.iteration = args.iteration

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(comparison.to_dict(), indent=2, default=str))
        print(f"Wrote replay comparison to {out_path}")
    else:
        print(comparison.render_text())
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="autobench",
        description="Recursive self-improvement coding agent harness system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run harness on benchmark")
    run_parser.add_argument("benchmark", help="Benchmark to run")
    run_parser.add_argument("--harness", required=True, help="Harness config file")

    improve_parser = subparsers.add_parser("improve", help="Run RSI loop")
    improve_parser.add_argument("benchmark", help="Benchmark to improve")
    improve_parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    # nervous-bus-msqa (wire-pop Phase 5): operator gates for cross-run
    # promotion. Default is stage-only; without one of these flags the
    # canonical harness is never swapped from a population cycle.
    improve_parser.add_argument(
        "--confirm-promotion",
        action="store_true",
        help="Approve the top confirmed-AHE advocate for canonical-harness swap.",
    )
    improve_parser.add_argument(
        "--reject-promotion",
        action="store_true",
        help="Explicitly reject the top confirmed-AHE advocate (logs to ledger, no swap).",
    )

    eval_parser = subparsers.add_parser("eval", help="Analyze repo and classify change type")
    eval_parser.add_argument("repo", help="Repository path to analyze")

    scaffold_parser = subparsers.add_parser("scaffold", help="Generate test scaffolding")
    scaffold_parser.add_argument("repo", help="Repository path")
    scaffold_parser.add_argument(
        "--type",
        choices=["feature", "bug_fix", "refactor", "redesign"],
        default="feature",
        help="Type of change",
    )

    sandbox_parser = subparsers.add_parser("sandbox", help="Execute code in sandbox")
    sandbox_parser.add_argument("code", help="Code to execute")
    sandbox_parser.add_argument("--language", default="python", help="Language")

    version_parser = subparsers.add_parser("version", help="Show version")

    # nervous-bus-1hlf — producer-triggered cycle daemon.
    trigger_parser = subparsers.add_parser(
        "trigger-daemon",
        help="Subscribe to autobench.cycle.requested.v1 and run a cycle per trigger.",
    )
    trigger_parser.add_argument(
        "--bead-id",
        default=None,
        help="Default tracker bead anchor applied when a trigger lacks its own bead_id.",
    )
    trigger_parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Exit after handling N triggers (test/staging knob). Default unlimited.",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Counterfactual replay — rerun a captured iteration with a forced override",
    )
    replay_parser.add_argument("--session-id", required=True, help="Session ID to replay")
    replay_parser.add_argument("--iteration", type=int, required=True, help="Iteration index to replay")
    replay_parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Override a harness field; dotted keys nest (e.g. budget.max_tokens=4096). May be repeated.",
    )
    replay_parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Directory of benchmark case JSON files (event payloads do not currently carry the benchmark name)",
    )
    replay_parser.add_argument(
        "--debug-file",
        default=str(Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"),
        help="Path to the autobench JSONL capture",
    )
    replay_parser.add_argument(
        "--out",
        default=None,
        help="Write JSON summary to this path (default: print human-readable to stdout)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "run": cmd_run,
        "improve": cmd_improve,
        "eval": cmd_eval,
        "scaffold": cmd_scaffold,
        "sandbox": cmd_sandbox,
        "version": cmd_version,
        "replay": cmd_replay,
        "trigger-daemon": cmd_trigger_daemon,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
