"""post_run_assess — post-run observer assessment pipeline.

After a kernel run completes, this module:
  1. Reads the top-N programs from the saved results JSON.
  2. Renders each to a PNG (SDF → sphere-trace, noise → GPU render).
  3. Asks MiniMax to assess each program's mathematical structure.
  4. Synthesizes a cross-program verdict and direction for next run.
  5. Saves a markdown report to the output dir.
  6. Publishes funsearch.assessment.v1 to the bus.

Usage (CLI):
    python3 -m autobench.post_run_assess \\
        --results-file benchmarks/curriculum/2026-05-30/sdf_results_gen21.json \\
        --kernel sdf --top-n 5

Usage (programmatic, from a kernel CLI after run):
    from autobench.post_run_assess import assess_run
    report_path, png_paths = assess_run(results_path, kernel="sdf", top_n=5)
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARTIFACTS_ROOT = Path(__file__).parent.parent / "benchmarks" / "artifacts"
ASSESSMENT_MODEL = "minimax-m2.7"
LLM_TIMEOUT = 90.0


# ---------------------------------------------------------------------------
# Per-program structural analysis
# ---------------------------------------------------------------------------

def _ask_minimax(prompt: str, timeout: float = LLM_TIMEOUT) -> str:
    """Call deer query --model minimax-m2.7 with the given prompt."""
    try:
        r = subprocess.run(
            ["deer", "query", "--model", ASSESSMENT_MODEL, "--terse", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception as e:
        logger.warning("MiniMax assessment call failed: %s", e)
        return ""


def _assess_sdf_program(prog: dict, png_path: Path | None) -> dict:
    """Ask MiniMax to assess one SDF program's mathematical structure."""
    code = prog.get("code", "") or prog.get("sdf_code", "")
    fitness = prog.get("fitness", 0.0)
    eikonal = prog.get("t_vector", {}).get("eikonal_score") or prog.get("_eikonal_score")
    topology = prog.get("t_vector", {}).get("topology_score") or prog.get("_topology_score")

    render_note = f"Rendered to {png_path.name}." if png_path and png_path.exists() else "No render available."

    prompt = f"""You are assessing an evolved SDF (signed-distance function) from a FunSearch run.

## Program (C++, sandboxed)
```cpp
{code[:1200]}
```

## Fitness metrics
- Combined fitness (0.6×eikonal + 0.4×topology): {fitness:.4f}
- Eikonal score exp(-½×mean_grad_err): {eikonal if eikonal is not None else "unknown"}
- Topology score (sign-change density vs target): {topology if topology is not None else "unknown"}
- {render_note}

## Your assessment (3-5 sentences)
1. Name the mathematical structure(s) present (e.g. gyroid, round box, Schwarz-P, torus knot, etc.)
2. Is this genuinely novel vs a simple rounded-box/sphere baseline? What specifically changed?
3. What does the eikonal score indicate about raymarching suitability?
4. What structural change would most improve this program toward a valid gyroid?"""

    answer = _ask_minimax(prompt)
    return {
        "program_id": prog.get("id", "?"),
        "fitness": fitness,
        "assessment": answer,
        "png": str(png_path) if png_path and png_path.exists() else None,
    }


def _assess_noise_program(prog: dict, png_path: Path | None) -> dict:
    """Ask MiniMax to assess one noise program's spectral structure."""
    code = prog.get("code", "")
    fitness = prog.get("fitness", 0.0)
    beta = prog.get("t_vector", {}).get("spectral_beta") or prog.get("_spectral_beta")
    hr = prog.get("t_vector", {}).get("harmonic_ratio") or prog.get("_harmonic_ratio")

    render_note = f"Rendered to {png_path.name}." if png_path and png_path.exists() else "No render available."

    prompt = f"""You are assessing an evolved GLSL noise function from a FunSearch run.

## Program (GLSL float noise(vec3 p))
```glsl
{code[:1200]}
```

## Fitness metrics
- RAPS spectral fitness: {fitness:.4f}
- Spectral beta (1/f slope): {beta:.3f if beta is not None else 'unknown'}
- Harmonic ratio: {hr:.3f if hr is not None else 'unknown'}
- {render_note}

## Your assessment (3-5 sentences)
1. Name the noise class (value noise, gradient noise, simplex-like, fBm, Worley, etc.)
2. Is this structurally novel vs the seed hash-value-noise baseline? What changed?
3. What does beta={beta:.2f if beta is not None else '?'} indicate about spectral character?
4. What single structural change would most expand this program's expressiveness?"""

    answer = _ask_minimax(prompt)
    return {
        "program_id": prog.get("id", "?"),
        "fitness": fitness,
        "assessment": answer,
        "png": str(png_path) if png_path and png_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_sdf(code: str, out_path: Path, instance_name: str = "") -> bool:
    # Try in-house CPU tracer first (headless, zero VRAM, takes C++ directly).
    try:
        from autobench.engines.sdf_tracer import render_sdf_cpp_to_png
        from autobench.artifact_store import _INSTANCE_CAMERA_DIST
        cam_dist = _INSTANCE_CAMERA_DIST.get(instance_name, 3.5)
        return render_sdf_cpp_to_png(code, out_path, viewport=(256, 256), camera_dist=cam_dist)
    except Exception:
        pass
    # GLSL fallback
    try:
        from autobench.artifact_store import render_sdf_to_png
        return render_sdf_to_png(code, out_path, viewport=(256, 256), i_time=1.2,
                                  instance_name=instance_name)
    except Exception as e:
        logger.warning("SDF render failed: %s", e)
        return False


def _render_noise(code: str, out_path: Path) -> bool:
    try:
        from autobench.noise_kernel import build_probe_shader
        from autobench.engines.shader_executor import ShaderExecutor
        shader = build_probe_shader(code)
        ex = ShaderExecutor()
        result = ex.render_only(shader, out_path=str(out_path), viewport=(512, 512), i_time=0.0)
        return result.frame_path != "" and Path(result.frame_path).exists()
    except Exception as e:
        logger.warning("Noise render failed: %s", e)
        return False


def _render_program(kernel: str, code: str, out_path: Path, instance_name: str = "") -> bool:
    if kernel == "sdf":
        return _render_sdf(code, out_path, instance_name=instance_name)
    elif kernel == "noise":
        return _render_noise(code, out_path)
    return False


# ---------------------------------------------------------------------------
# Synthesis — cross-program verdict
# ---------------------------------------------------------------------------

def _synthesize(kernel: str, assessments: list[dict], run_meta: dict) -> str:
    """Ask MiniMax for an overall synthesis across all assessed programs."""
    summaries = "\n\n".join(
        f"**Program {i+1}** (id={a['program_id']}, fitness={a['fitness']:.4f})\n{a['assessment']}"
        for i, a in enumerate(assessments)
        if a["assessment"]
    )

    best_fitness = run_meta.get("best_fitness", 0.0)
    stop_reason = run_meta.get("stop_reason", "")
    instances = run_meta.get("instances", [])
    gens = run_meta.get("generations_run", 0)

    prompt = f"""You are synthesizing the findings from a FunSearch {kernel.upper()} kernel run.

## Run summary
- Instances: {instances}
- Generations: {gens}
- Best fitness: {best_fitness:.4f}
- Stop reason: {stop_reason}

## Individual program assessments
{summaries}

## Your synthesis (4-6 sentences)
1. What mathematical structure(s) did this run converge on? Were they genuinely novel?
2. Did the oracle reward the right structure, or did it converge on a local optimum?
3. What did the run NOT discover that it should have (the dark forward — the missing structure)?
4. What is the single highest-leverage change for the next run to break past this plateau?
5. Rate the overall evolutionary progress: child's play / good baseline / genuine discovery / breakthrough."""

    return _ask_minimax(prompt, timeout=120.0)


# ---------------------------------------------------------------------------
# Main entry: assess_run()
# ---------------------------------------------------------------------------

def assess_run(
    results_path: Path,
    kernel: str,
    top_n: int = 5,
    nervous_bin: str | None = None,
) -> tuple[Path, list[Path]]:
    """Run the full post-run observer assessment.

    Returns (report_path, list_of_png_paths).
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    with open(results_path) as f:
        results = json.load(f)

    run_id = results.get("run_id", uuid.uuid4().hex[:12])
    top_programs = results.get("top_programs", [])
    best_program = results.get("best_program")
    if not top_programs and best_program:
        top_programs = [best_program]

    stop_reason = results.get("stop_reason", "")
    history = results.get("history", [])
    gens_run = len(history)
    instances = results.get("config", {}).get("instances", [])
    best_fitness = top_programs[0]["fitness"] if top_programs else 0.0

    output_dir = results_path.parent
    assess_dir = output_dir / f"assessment_{run_id[:8]}"
    assess_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[observer] Assessing {kernel} run {run_id[:8]} — top {min(top_n, len(top_programs))} programs")
    print(f"  results: {results_path.name}")
    print(f"  best fitness: {best_fitness:.4f}, gens: {gens_run}, stop: {stop_reason}")

    # --- Render and assess each program ---
    png_paths: list[Path] = []
    assessments: list[dict] = []

    for i, prog in enumerate(top_programs[:top_n]):
        code = prog.get("code", "") or prog.get("sdf_code", "")
        if not code:
            continue

        inst_name = prog.get("instance", "") or prog.get("t_vector", {}).get("instance", "")
        out_png = assess_dir / f"rank{i+1:02d}_{prog['id'][:30]}.png"
        rendered = _render_program(kernel, code, out_png, instance_name=inst_name)
        png_path = out_png if rendered else None
        if rendered:
            png_paths.append(out_png)
            print(f"  [render] rank {i+1}: {out_png.name}")
        else:
            print(f"  [render] rank {i+1}: no GPU or render failed")

        print(f"  [assess] rank {i+1} (fitness={prog['fitness']:.4f}) …", end=" ", flush=True)
        if kernel == "sdf":
            a = _assess_sdf_program(prog, png_path)
        else:
            a = _assess_noise_program(prog, png_path)
        assessments.append(a)
        print("done")

    # --- Synthesis ---
    print("  [synthesize] cross-program verdict …", end=" ", flush=True)
    run_meta = {
        "best_fitness": best_fitness,
        "stop_reason": stop_reason,
        "instances": instances,
        "generations_run": gens_run,
    }
    synthesis = _synthesize(kernel, assessments, run_meta)
    print("done")

    # --- Write markdown report ---
    report_path = output_dir / f"assessment_{run_id[:8]}.md"
    _write_report(report_path, kernel, run_meta, assessments, synthesis, run_id)
    print(f"  [report] {report_path}")

    # --- Publish bus event ---
    _publish_assessment(kernel, run_id, best_fitness, synthesis, assessments, nervous_bin)

    return report_path, png_paths


def _write_report(
    path: Path,
    kernel: str,
    run_meta: dict,
    assessments: list[dict],
    synthesis: str,
    run_id: str,
) -> None:
    lines = [
        f"# {kernel.upper()} Run Assessment — {run_id[:8]}",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  ",
        f"**Instances:** {', '.join(run_meta.get('instances', []))}  ",
        f"**Generations:** {run_meta.get('generations_run', 0)}  ",
        f"**Best fitness:** {run_meta.get('best_fitness', 0.0):.4f}  ",
        f"**Stop reason:** {run_meta.get('stop_reason', '')}",
        "",
        "---",
        "",
        "## Program Assessments",
        "",
    ]
    for i, a in enumerate(assessments):
        lines += [
            f"### Rank {i+1} — `{a['program_id']}` (fitness={a['fitness']:.4f})",
            "",
        ]
        if a.get("png"):
            lines.append(f"![render]({Path(a['png']).name})")
            lines.append("")
        lines += [a.get("assessment", "*(no assessment)*"), "", "---", ""]

    lines += [
        "## Synthesis",
        "",
        synthesis or "*(synthesis unavailable)*",
        "",
    ]
    path.write_text("\n".join(lines))


def _publish_assessment(
    kernel: str,
    run_id: str,
    best_fitness: float,
    synthesis: str,
    assessments: list[dict],
    nervous_bin: str | None,
) -> None:
    from autobench.artifact_store import _publish_artifact_event
    entry = {
        "id": uuid.uuid4().hex[:12],
        "kernel": kernel,
        "run_id": run_id,
        "generation": -1,
        "fitness": best_fitness,
        "instance": "multi",
        "artifact_path": "",
        "render_type": "assessment",
        "metadata": {
            "synthesis": synthesis[:500],
            "n_assessed": len(assessments),
            "top_fitness": [round(a["fitness"], 4) for a in assessments[:5]],
        },
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _publish_artifact_event(entry, nervous_bin)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s")

    p = argparse.ArgumentParser(description="Post-run observer assessment")
    p.add_argument("--results-file", required=True,
                   help="Path to sdf_results_gen*.json or noise_results_gen*.json")
    p.add_argument("--kernel", default="sdf", choices=["sdf", "noise", "tsp"],
                   help="Kernel type (default: sdf)")
    p.add_argument("--top-n", type=int, default=5,
                   help="How many programs to render and assess (default: 5)")
    args = p.parse_args()

    report, pngs = assess_run(Path(args.results_file), kernel=args.kernel, top_n=args.top_n)
    print(f"\nReport: {report}")
    print(f"PNGs ({len(pngs)}):")
    for p in pngs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
