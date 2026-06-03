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


GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


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


def _champion_from_results(path):
    import json
    r = json.load(open(path))
    b = r.get("best_program") or {}
    return b.get("density_code") or b.get("code", ""), b.get("fitness", 0.0)


def cmd_render(args) -> int:
    """Render one evolved compute_density to a PNG (champion of a results file or a code file)."""
    from .render import render_density_to_png
    if args.code_file:
        code = open(args.code_file).read(); fit = None
    elif args.results:
        code, fit = _champion_from_results(args.results)
    else:
        print(f"{RED}need --results or --code-file{RESET}"); return 1
    if not code.strip():
        print(f"{RED}no candidate code found{RESET}"); return 1
    ex = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
    out = args.out or "svdag_render.png"
    ok = render_density_to_png(code, ex, out, res=args.res, seed=args.seed, run_timeout=args.timeout)
    print(f"{GREEN if ok else RED}{'rendered '+out if ok else 'render FAILED'}{RESET}"
          + (f"  (champion fitness={fit:.4f})" if fit else ""))
    return 0 if ok else 1


def cmd_showcase(args) -> int:
    """Render the top-N programs of a results file into a labelled montage grid."""
    import json, math
    from .render import sample_occupancy, render_occupancy
    from PIL import Image, ImageDraw
    r = json.load(open(args.results))
    progs = r.get("top_programs") or ([r["best_program"]] if r.get("best_program") else [])
    progs = progs[: args.top_n]
    if not progs:
        print(f"{RED}no programs in {args.results}{RESET}"); return 1
    ex = ensure_executor(allow_unsandboxed=args.allow_unsandboxed)
    tiles = []
    for p in progs:
        code = p.get("density_code") or p.get("code", "")
        seed = float((p.get("diag") or {}).get("seed", 5151.0)) if isinstance(p.get("diag"), dict) else 5151.0
        O = sample_occupancy(code, ex, res=args.res, seed=seed, run_timeout=args.timeout)
        if O is None:
            continue
        img = render_occupancy(O, img_w=480, img_h=380)
        im = Image.fromarray(img, "RGB")
        d = ImageDraw.Draw(im)
        d.text((8, 8), f"isl{p.get('island','?')} gen{p.get('generation','?')}  fit={p.get('fitness',0):.4f}", fill=(255, 240, 220))
        tiles.append(im)
    if not tiles:
        print(f"{RED}all renders failed{RESET}"); return 1
    cols = min(args.cols, len(tiles)); rows = math.ceil(len(tiles) / cols)
    tw, th = tiles[0].size
    grid = Image.new("RGB", (cols * tw, rows * th), (12, 12, 20))
    for i, im in enumerate(tiles):
        grid.paste(im, ((i % cols) * tw, (i // cols) * th))
    out = args.out or "svdag_showcase.png"
    grid.save(out)
    print(f"{GREEN}showcase {out}{RESET}  ({len(tiles)} terrains, {cols}x{rows})")
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

    rdp = sub.add_parser("render", help="Render an evolved terrain to a PNG")
    rdp.add_argument("--results", default=None, help="results JSON (renders its champion)")
    rdp.add_argument("--code-file", default=None, help="a compute_density .c/.slang file")
    rdp.add_argument("--out", default=None)
    rdp.add_argument("--res", type=int, default=96)
    rdp.add_argument("--seed", type=float, default=5151.0)
    rdp.add_argument("--timeout", type=float, default=120.0)
    rdp.add_argument("--allow-unsandboxed", action="store_true")
    rdp.add_argument("-v", "--verbose", action="store_true")

    scp = sub.add_parser("showcase", help="Montage of the top-N evolved terrains")
    scp.add_argument("--results", required=True)
    scp.add_argument("--out", default=None)
    scp.add_argument("--top-n", type=int, default=6)
    scp.add_argument("--cols", type=int, default=3)
    scp.add_argument("--res", type=int, default=80)
    scp.add_argument("--seed", type=float, default=5151.0)
    scp.add_argument("--timeout", type=float, default=120.0)
    scp.add_argument("--allow-unsandboxed", action="store_true")
    scp.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    dispatch = {"run": cmd_run, "baselines": cmd_baselines, "instances": cmd_instances,
                "render": cmd_render, "showcase": cmd_showcase}
    handler = dispatch.get(args.cmd)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
