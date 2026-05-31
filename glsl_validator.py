"""glsl_validator — end-to-end GLSL ES validator for evolved SDF programs.

Pipeline:
  C++ sdf() code
    → cpp_to_glsl()       (signature + math rename + f-suffix strip)
    → _fix_int_float()    (GLSL ES int×float strictness: float(i), float(N))
    → build_shadertoy_glsl() (wrap in sphere-tracing mainImage)
    → _wrap_for_validation() (add playground preamble + main bridge)
    → glslangValidator -S frag (pure GLSL syntax validation, no SPIR-V)
    → parse errors → return (ok, errors, fixed_glsl)

Usage:
    from autobench.glsl_validator import validate_sdf, fix_and_validate

    ok, errors, glsl = validate_sdf(cpp_code, instance="torus_knot")
    if not ok:
        print(errors)
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exact replica of the playground's wrapGlslFragment preamble and bridge
# ---------------------------------------------------------------------------

_GLSL_PREAMBLE = """\
#version 300 es
precision highp float;
layout(std140) uniform StudioUniforms {
  vec3 iResolution;
  float _pad0;
  float iTime;
  float iTimeDelta;
  uint iFrame;
  float iFrameRate;
  vec4 iMouse;
  vec4 iDate;
};
out vec4 fragColor;
vec2 resolution() { return iResolution.xy; }
float time() { return iTime; }
vec4 mouse() { return iMouse; }
vec2 fragCoordNormalized() { return gl_FragCoord.xy / iResolution.xy; }
"""

_MAIN_BRIDGE = """\

void main() {
    mainImage(fragColor, gl_FragCoord.xy);
}
"""


def _wrap_for_validation(sdf_glsl_with_main: str) -> str:
    """Wrap user GLSL the same way wrapGlslFragment does in the playground."""
    return _GLSL_PREAMBLE + sdf_glsl_with_main + _MAIN_BRIDGE


# ---------------------------------------------------------------------------
# GLSL ES int × float strictness fix
# ---------------------------------------------------------------------------

def _fix_int_float(code: str) -> str:
    """Fix GLSL ES 3.00 int × float type errors.

    GLSL ES (WebGL) rejects implicit int↔float promotions that C++ allows:
      pi2 * i / BASE_SAMPLES  →  pi2 * float(i) / float(BASE_SAMPLES)
      i * arc_dense           →  float(i) * arc_dense

    Strategy:
    - Collect all int variables (for-loop counters, const int names).
    - On a line-by-line basis, wrap those variables in float() when they
      appear as operands of * or / (arithmetic operators that require
      matching types in GLSL ES).
    - Skip for-loop header lines entirely to avoid corrupting
      `for (int i = 0; i < N; i++)` syntax.
    - Skip already-wrapped occurrences: float(VAR) is left alone.
    """
    # 1. Collect int variable names
    int_vars: set[str] = set()
    for m in re.finditer(r'\bfor\s*\(\s*int\s+(\w+)\s*=', code):
        int_vars.add(m.group(1))
    for m in re.finditer(r'\bconst\s+int\s+(\w+)\s*=', code):
        int_vars.add(m.group(1))
    # Plain int declarations: `int lo = 0;` (binary search variables)
    for m in re.finditer(r'(?<![a-zA-Z_])\bint\s+(\w+)\s*=', code):
        int_vars.add(m.group(1))

    if not int_vars:
        return code

    lines = code.split('\n')
    out = []
    for line in lines:
        # Protect for-loop headers — never touch `for (int i = ...; i < ...; i++)`
        if re.match(r'\s*for\s*\(', line):
            out.append(line)
            continue

        for var in int_vars:
            # Pattern A: VAR * expr  or  VAR / expr  — int as LEFT operand of * or /
            # Guard: not already float(VAR), not part of a longer identifier
            line = re.sub(
                r'(?<!float\()(?<![a-zA-Z_0-9])' + re.escape(var) +
                r'(?![a-zA-Z_0-9(])(?=\s*[*/])',
                f'float({var})',
                line,
            )
            # Pattern B: expr * VAR  or  expr / VAR  — int as RIGHT operand
            # Use a capture group for the operator since lookbehind needs fixed width.
            line = re.sub(
                r'([*/]\s*)(?<!float\()(?<![a-zA-Z_0-9])' + re.escape(var) +
                r'(?![a-zA-Z_0-9(])',
                r'\g<1>float(' + var + ')',
                line,
            )

        out.append(line)

    return '\n'.join(out)


# ---------------------------------------------------------------------------
# glslangValidator execution
# ---------------------------------------------------------------------------

_GLSLANG = 'glslangValidator'


def _run_glslang(glsl_source: str) -> tuple[bool, list[str]]:
    """Compile with glslangValidator -S frag (pure GLSL syntax, no SPIR-V).

    Returns (ok, error_lines).
    """
    with tempfile.NamedTemporaryFile(
        suffix='.frag', mode='w', delete=False, dir='/tmp'
    ) as f:
        f.write(glsl_source)
        fname = f.name
    try:
        r = subprocess.run(
            [_GLSLANG, '-S', 'frag', fname],
            capture_output=True, text=True, timeout=15,
        )
        all_output = r.stdout + r.stderr
        errors = [ln for ln in all_output.split('\n') if 'ERROR' in ln and ln.strip()]
        return r.returncode == 0 and not errors, errors
    except FileNotFoundError:
        logger.warning('glslangValidator not found — skipping compilation check')
        return True, []
    except subprocess.TimeoutExpired:
        return False, ['glslangValidator timed out']
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Error annotation: attach source context to each error
# ---------------------------------------------------------------------------

def _annotate_errors(errors: list[str], source: str) -> list[str]:
    """Add the offending source line to each error for context."""
    lines = source.split('\n')
    annotated = []
    for err in errors:
        m = re.search(r':(\d+):', err)
        if m:
            lineno = int(m.group(1))
            if 1 <= lineno <= len(lines):
                annotated.append(f'{err}')
                annotated.append(f'    → {lines[lineno - 1].strip()}')
            else:
                annotated.append(err)
        else:
            annotated.append(err)
    return annotated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_sdf(
    cpp_code: str,
    instance: str = '',
    camera_dist: Optional[float] = None,
) -> tuple[bool, list[str], Optional[str]]:
    """Full pipeline: C++ → fixed GLSL ES → compile check.

    Returns (ok, annotated_errors, full_glsl_or_None).
    ok=True means the shader compiled cleanly.
    full_glsl is the complete shader submitted to the validator (preamble included).
    """
    from autobench.playground_push import cpp_to_glsl, build_shadertoy_glsl, _INSTANCE_CAMERA_DIST

    # Step 1: C++ → GLSL (signature, math renames, f-suffix)
    glsl = cpp_to_glsl(cpp_code)
    if glsl is None:
        return False, ['GPU-incompatible: static arrays or large local arrays'], None

    # Step 2: GLSL ES int×float fix
    glsl = _fix_int_float(glsl)

    # Step 3: wrap with sphere-tracing mainImage
    cam = camera_dist or _INSTANCE_CAMERA_DIST.get(instance, 3.5)
    full_user = build_shadertoy_glsl(glsl, camera_dist=cam)

    # Step 4: add playground preamble + main bridge
    full = _wrap_for_validation(full_user)

    # Step 5: compile
    ok, errors = _run_glslang(full)
    annotated = _annotate_errors(errors, full)
    return ok, annotated, full


def fix_and_validate(
    cpp_code: str,
    instance: str = '',
    camera_dist: Optional[float] = None,
) -> tuple[bool, list[str], Optional[str]]:
    """Validate and return the fixed GLSL suitable for vault push.

    Same as validate_sdf but returns only the user-facing shader (sdf + mainImage),
    not the full preamble+bridge — ready for push_to_playground().
    """
    from autobench.playground_push import cpp_to_glsl, build_shadertoy_glsl, _INSTANCE_CAMERA_DIST

    glsl = cpp_to_glsl(cpp_code)
    if glsl is None:
        return False, ['GPU-incompatible: static arrays or large local arrays'], None

    glsl = _fix_int_float(glsl)
    cam = camera_dist or _INSTANCE_CAMERA_DIST.get(instance, 3.5)
    full_user = build_shadertoy_glsl(glsl, camera_dist=cam)

    full = _wrap_for_validation(full_user)
    ok, errors = _run_glslang(full)
    annotated = _annotate_errors(errors, full)
    return ok, annotated, full_user if ok else None


# ---------------------------------------------------------------------------
# Batch validation — run against a results JSON file
# ---------------------------------------------------------------------------

def validate_results_file(
    results_path: Path,
    top_n: int = 10,
) -> list[dict]:
    """Validate top-N programs from a kernel results JSON.

    Returns list of {rank, id, fitness, instance, ok, errors, glsl}.
    """
    import json
    with open(results_path) as f:
        data = json.load(f)

    top = data.get('top_programs', [])
    results = []
    for i, prog in enumerate(top[:top_n]):
        code = prog.get('sdf_code') or prog.get('code', '')
        if not code:
            continue
        instance = prog.get('instance', '')
        ok, errors, glsl = validate_sdf(code, instance=instance)
        results.append({
            'rank': i + 1,
            'id': prog.get('id', '?'),
            'fitness': prog.get('fitness', 0.0),
            'instance': instance,
            'ok': ok,
            'errors': errors,
            'glsl': glsl,
        })
    return results


# ---------------------------------------------------------------------------
# CLI — run standalone against a results file
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import json
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    p = argparse.ArgumentParser(description='Validate GLSL ES for evolved SDF programs')
    p.add_argument('results_file', help='sdf_results_gen*.json')
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--dump-glsl', action='store_true',
                   help='Print the full compiled GLSL for each program')
    args = p.parse_args()

    results = validate_results_file(Path(args.results_file), top_n=args.top_n)

    ok_count = sum(1 for r in results if r['ok'])
    print(f'\nValidated {len(results)} programs — {ok_count} OK, {len(results)-ok_count} FAIL\n')

    for r in results:
        status = 'OK  ' if r['ok'] else 'FAIL'
        print(f'  [{status}] rank{r["rank"]} fitness={r["fitness"]:.4f} '
              f'instance={r["instance"] or "?"} id={r["id"][:24]}')
        for err in r['errors']:
            print(f'         {err}')
        if args.dump_glsl and r['glsl']:
            print('  --- GLSL ---')
            for i, ln in enumerate(r['glsl'].split('\n'), 1):
                print(f'  {i:3d}: {ln}')
            print()

    return 0 if all(r['ok'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
