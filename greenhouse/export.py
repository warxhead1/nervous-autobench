"""export — per-domain GLSL transpile + offline validation + drop-file writer.

Reuses the SAME conversion paths the rest of autobench uses to reach Shader
Garden / the shader-studio vault, rather than reimplementing them:

  - sdf:      ``autobench.glsl_validator.fix_and_validate`` — one call does
              cpp_to_glsl + GLSL-ES int/float fix + sphere-trace wrap +
              glslangValidator, and hands back push-ready GLSL.
  - terrain:  ``autobench.kernels.bridge.NervousKernelBridge._to_glsl`` (C
              with libc-suffixed math → GLSL) + ``_wrap_glsl_mainimage``
              (raymarch demo), then offline-validated the same way
              glsl_validator wraps arbitrary GLSL (studio preamble + main
              bridge + glslangValidator).
  - noise:    candidates are ALREADY GLSL ``float noise(vec3 p)`` — wrapped
              here in a small 2D noise-field demo ``mainImage``, validated
              the same way.

Domains without a wired export path (phase, latent, sph, thermal, oasis)
raise ``UnsupportedDomain`` — ``cycle.py`` catches it and skips export for
that goal's candidates (counted and reported, never silently dropped). Their
current publish paths (``publish_evolved_kernels.py``, hand-authored
per-domain ``mainImage`` wrappers) are involved enough that wiring them in
is left for follow-up; see the greenhouse README.

glslangValidator is optional at runtime: the shared ``_run_glslang`` helper
already warns and treats a missing binary as a pass (offline-friendly).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from autobench.glsl_validator import _fix_int_float, _run_glslang, _wrap_for_validation, fix_and_validate
from autobench.kernels.bridge import NervousKernelBridge

DEFAULT_DROPS_ROOT = Path.home() / ".cache" / "nervous-bus" / "greenhouse" / "drops"

_bridge = NervousKernelBridge()

_NOISE_MAINIMAGE = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 p = vec3(uv * 4.0, iTime * 0.15);
    float n = noise(p);
    vec3 col = vec3(0.5 + 0.5 * n);
    fragColor = vec4(col, 1.0);
}
"""


class UnsupportedDomain(ValueError):
    """Raised when ``export_candidate`` is asked to export a domain with no wired GLSL path."""


@dataclass
class ExportResult:
    validated: bool
    candidate_id: str
    errors: list[str]
    drop_path: Path | None = None
    glsl_bytes: int = 0


def _export_sdf(code: str, *, instance: str, fitness: float) -> tuple[bool, str | None, list[str]]:
    ok, errors, glsl = fix_and_validate(code, instance=instance)
    return ok, glsl, list(errors)


def _export_terrain(code: str, *, instance: str, fitness: float) -> tuple[bool, str | None, list[str]]:
    glsl = _bridge._to_glsl(code)
    glsl = _fix_int_float(glsl)
    user_glsl = _bridge._wrap_glsl_mainimage(glsl, instance or "terrain", fitness)
    ok, errors = _run_glslang(_wrap_for_validation(user_glsl))
    return ok, (user_glsl if ok else None), list(errors)


def _export_noise(code: str, *, instance: str, fitness: float) -> tuple[bool, str | None, list[str]]:
    glsl = _fix_int_float(code)
    user_glsl = glsl.strip() + "\n" + _NOISE_MAINIMAGE
    ok, errors = _run_glslang(_wrap_for_validation(user_glsl))
    return ok, (user_glsl if ok else None), list(errors)


_EXPORTERS: dict[str, Callable[..., tuple[bool, str | None, list[str]]]] = {
    "sdf": _export_sdf,
    "terrain": _export_terrain,
    "noise": _export_noise,
}


def supported_domains() -> frozenset[str]:
    return frozenset(_EXPORTERS)


def export_candidate(
    *,
    domain: str,
    goal_id: str,
    goal_notes: str,
    goal_tags: list[str],
    instance: str,
    program: dict,
    run_id: str,
    drops_root: Path | None = None,
) -> ExportResult:
    """Transpile, validate, and (if valid) write one candidate as a garden-shaped drop.

    ``program`` is a top_programs-shaped dict: id, code, fitness, generation,
    source (matches ``FunSearchKernel.save_results``' entry shape).
    """
    exporter = _EXPORTERS.get(domain)
    if exporter is None:
        raise UnsupportedDomain(domain)

    candidate_id = program["id"]
    fitness = float(program["fitness"])
    ok, glsl, errors = exporter(program["code"], instance=instance, fitness=fitness)

    if not ok or glsl is None:
        return ExportResult(validated=False, candidate_id=candidate_id, errors=errors)

    drop = {
        "id": f"greenhouse-{goal_id}-{candidate_id}",
        "title": f"[{goal_id}] {domain} gen{program.get('generation', 0)} fit={fitness:.4f}",
        "description": goal_notes or f"FunSearch-evolved {domain} candidate for goal '{goal_id}'.",
        "domain": domain,
        "fitness": fitness,
        "generation": program.get("generation", 0),
        "run_id": run_id,
        "author": "funsearch-greenhouse",
        "origin": "greenhouse",
        "language": "glsl",
        "glsl": glsl,
        "tags": list(goal_tags),
    }

    root = drops_root or DEFAULT_DROPS_ROOT
    goal_dir = root / goal_id
    goal_dir.mkdir(parents=True, exist_ok=True)
    drop_path = goal_dir / f"{run_id}-{candidate_id}.json"
    drop_path.write_text(json.dumps(drop, indent=2))

    return ExportResult(
        validated=True, candidate_id=candidate_id, errors=[],
        drop_path=drop_path, glsl_bytes=len(glsl.encode()),
    )


def dropped_count(goal_id: str, drops_root: Path | None = None) -> int:
    """How many validated candidates already sit on disk for this goal."""
    root = drops_root or DEFAULT_DROPS_ROOT
    goal_dir = root / goal_id
    if not goal_dir.is_dir():
        return 0
    return sum(1 for _ in goal_dir.glob("*.json"))
