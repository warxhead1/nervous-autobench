"""greenhouse CLI — python -m greenhouse cycle|status|dry-run"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import export
from .cycle import CycleResult, run_cycle
from .goals import GoalsManifestError, load_manifest
from .ledger import Ledger


def _result_to_json(result: CycleResult) -> dict:
    d = asdict(result)
    d["drop_paths"] = [str(p) for p in result.drop_paths]
    return d


def cmd_cycle(args: argparse.Namespace) -> int:
    result = run_cycle(dry_run=False)
    print(json.dumps(_result_to_json(result), indent=2))
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    result = run_cycle(dry_run=True)
    print(json.dumps(_result_to_json(result), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        manifest = load_manifest()
    except GoalsManifestError as e:
        print(f"goals manifest error: {e}", file=sys.stderr)
        return 1

    ledger = Ledger()
    used = ledger.window_used(manifest.budget.window_seconds)
    remaining = ledger.remaining(manifest.budget.window_max_requests, manifest.budget.window_seconds)
    print(f"window: {used}/{manifest.budget.window_max_requests} requests used "
          f"({remaining} remaining, {manifest.budget.window_seconds:.0f}s window)")
    print(f"per-cycle cap: {manifest.budget.per_cycle_max_requests}")
    print()
    print(f"{'goal':<28} {'domain':<10} {'priority':>8} {'want':>5} {'dropped':>8}")
    for goal in sorted(manifest.goals, key=lambda g: -g.priority):
        dropped = export.dropped_count(goal.id)
        print(f"{goal.id:<28} {goal.domain:<10} {goal.priority:>8} {goal.want:>5} {dropped:>8}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="greenhouse", description="Goal-directed FunSearch evolution scheduler")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cycle = sub.add_parser("cycle", help="Run one budgeted evolution cycle")
    p_cycle.set_defaults(func=cmd_cycle)

    p_status = sub.add_parser("status", help="Show window usage and per-goal want-vs-dropped table")
    p_status.set_defaults(func=cmd_status)

    p_dry = sub.add_parser("dry-run", help="Exercise the full cycle pipeline with no LLM calls")
    p_dry.set_defaults(func=cmd_dry_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
