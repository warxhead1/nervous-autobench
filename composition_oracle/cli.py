"""Composition oracle CLI — test (terrain, reaction) kernel pair as a unit.

Usage:
    python -m autobench.composition_oracle test \\
        benchmarks/curriculum/2026-05-31/terrain_results_gen25.json \\
        benchmarks/curriculum/2026-05-31/phase_results_gen36.json

    python -m autobench.composition_oracle test \\
        --terrain-code "float terrain(vec2 p) { ... }" \\
        --reaction-code "float reaction(float phi, float temp) { ... }"
"""
import argparse
import sys

from . import evaluate_best_pair, evaluate_composition
from ..kernels import ensure_sandboxed_executor as ensure_executor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Composition oracle — test (terrain, reaction) pair"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("test", help="Evaluate best programs from result files")
    t.add_argument("terrain_json", help="terrain kernel results JSON")
    t.add_argument("phase_json",   help="phase kernel results JSON")
    t.add_argument("--allow-unsandboxed", action="store_true")

    args = parser.parse_args()

    if args.cmd == "test":
        executor = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
        try:
            result = evaluate_best_pair(args.terrain_json, args.phase_json, executor)
        except Exception as e:
            print(f"[ERROR] {e}")
            return 1

        print("[Composition oracle] Results:")
        print(f"  terrain fitness:      {result['terrain_fitness']:.4f}")
        print(f"  phase fitness:        {result['phase_fitness']:.4f}")
        cf = result['composition_fitness']
        if cf is None:
            print("  composition fitness:  FAILED  [oracle error — check kernel validity]")
            return 1
        status = "PASS (ready for TEngine)" if cf > 0.65 else "FAIL (kernels not composable yet)"
        print(f"  composition fitness:  {cf:.4f}  [{status}]")
        return 0 if (cf and cf > 0.65) else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
