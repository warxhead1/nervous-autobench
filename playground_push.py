"""playground_push — push evolved SDF programs to the refactor-dashboard shader playground.

Converts C++ sdf() code to GLSL, wraps with a ShaderToy-compatible sphere-tracing
mainImage, and POSTs to the vault at /api/portal/vault/shaders.

From the playground the shader can be:
  - Viewed live (WebGL/WebGPU, auto-starts on load)
  - Cross-compiled to WGSL/Slang via /api/portal/shader/transpile
  - Dispatched to silo_tester for full TEngine GPU profiling
  - Exported as a standalone HTML file
  - Pushed to ShaderToy by copy-pasting the GLSL

GPU compatibility: programs using `static` local arrays can't be translated to GLSL
(no static storage in fragment shaders). Those are flagged and skipped. The caller
should provide the next-best GLSL-compatible program.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DASHBOARD_URL = "http://localhost:9104"
FUNSEARCH_AUTHOR = "funsearch-autobench"

# C++ math → GLSL built-ins
_MATH_RENAMES: dict[str, str] = {
    "sqrtf": "sqrt",
    "fabsf": "abs",
    "fmaxf": "max",
    "fminf": "min",
    "sinf": "sin",
    "cosf": "cos",
    "atan2f": "atan",
    "powf": "pow",
    "expf": "exp",
    "logf": "log",
    "floorf": "floor",
    "ceilf": "ceil",
    "roundf": "round",
}

# Per-instance camera orbit radius — keeps the shape in frame.
_INSTANCE_CAMERA_DIST: dict[str, float] = {
    "gyroid":        3.0,
    "round_box":     3.0,
    "sphere":        3.0,
    "warped_sphere": 3.0,
    "smooth_union":  3.5,
    "cloud_cluster": 4.5,
    "torus_knot":    2.0,
    "helix_tube":    2.5,
    "scherk_first":  2.2,
}

_SPHERE_TRACER_MAINIMAGE = """\

void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = (fragCoord - 0.5*iResolution.xy) / iResolution.y;
    float t = iTime * 0.4 + {i_time_offset};
    vec3 ro = vec3({cam_r}*cos(t), 0.8*{cam_r}, {cam_r}*sin(t));
    vec3 ww = normalize(-ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = cross(uu, ww);
    vec3 rd = normalize(uv.x*uu + uv.y*vv + 1.7*ww);

    float d = 0.001, max_d = {max_d};
    for (int i = 0; i < 96; i++) {{
        float h = sdf(ro + d*rd);
        if (abs(h) < 0.0015*d || d > max_d) break;
        d += h * 0.75;
    }}

    vec3 col = vec3(0.04, 0.02, 0.07);
    if (d < max_d) {{
        vec3 p = ro + d*rd;
        float e = 0.001;
        vec3 n = normalize(vec3(
            sdf(p+vec3(e,0,0)) - sdf(p-vec3(e,0,0)),
            sdf(p+vec3(0,e,0)) - sdf(p-vec3(0,e,0)),
            sdf(p+vec3(0,0,e)) - sdf(p-vec3(0,0,e))
        ));
        vec3 ld = normalize(vec3(1.5, 2.0, -0.5));
        float diff = max(dot(n, ld), 0.0);
        float spec = pow(max(dot(reflect(-ld, n), -rd), 0.0), 48.0) * 0.7;
        col = vec3(0.15, 0.55, 0.85)*diff + spec + vec3(0.08, 0.04, 0.12)*(1.0-diff);
        col *= exp(-d * 0.12);
    }}
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}}"""


_MAX_ARRAY_ELEMENTS = 64  # arrays larger than this per-pixel are unusable in realtime


def cpp_to_glsl(cpp_code: str) -> Optional[str]:
    """Convert evolved C++ sdf(float x,y,z) to GLSL float sdf(vec3 pos).

    Returns None if the code uses C++-only features that can't be translated
    to a WebGL fragment shader:
      - static local arrays (no static storage in GLSL fragment shaders)
      - large local arrays (>64 elements per pixel → GPU stall in realtime)
    """
    if "static " in cpp_code:
        return None

    # Reject programs with large local array allocations (1200-element arc tables etc.)
    # Pattern: `float arr[N]` or `float arr[CONST_NAME]` where resolved N > threshold.
    # First collect const int definitions to resolve symbolic sizes.
    const_ints: dict[str, int] = {}
    for m in re.finditer(r'\bconst\s+int\s+(\w+)\s*=\s*(\d+)', cpp_code):
        const_ints[m.group(1)] = int(m.group(2))
    for m in re.finditer(r'(?:float|int)\s+\w+\s*\[([^\]]+)\]', cpp_code):
        size_expr = m.group(1).strip()
        try:
            # Resolve symbolic names then eval the expression
            for name, val in const_ints.items():
                size_expr = re.sub(r'\b' + name + r'\b', str(val), size_expr)
            size = int(eval(size_expr, {"__builtins__": {}}))  # noqa: S307
            if size > _MAX_ARRAY_ELEMENTS:
                return None
        except Exception:
            pass  # can't resolve — allow through, compiler will decide

    code = cpp_code

    # Math function renames
    for cpp_fn, glsl_fn in _MATH_RENAMES.items():
        code = re.sub(r'\b' + cpp_fn + r'\b', glsl_fn, code)

    # Signature: extern "C" float sdf(float x, float y, float z) → float sdf(vec3 pos)
    # Use `pos` not `p` to avoid colliding with user variables named `p`.
    # Track whether replacement happened — only inject unpacking if it did.
    cpp_sig = re.compile(
        r'extern\s+"C"\s+float\s+sdf\s*\(\s*float\s+x\s*,\s*float\s+y\s*,\s*float\s+z\s*\)'
    )
    did_replace = bool(cpp_sig.search(code))
    code = cpp_sig.sub('float sdf(vec3 pos)', code)

    # Inject coordinate unpacking only when we performed the signature replacement
    # AND the unpacking isn't already present (idempotency guard).
    if did_replace and 'float x = pos.x' not in code:
        code = code.replace(
            'float sdf(vec3 pos) {\n',
            'float sdf(vec3 pos) {\n    float x = pos.x, y = pos.y, z = pos.z;\n',
            1,
        )

    # Strip f-suffix from float literals: 1.0f → 1.0, 0.5f → 0.5
    code = re.sub(r'(\d)f\b', r'\1', code)

    return code


def build_shadertoy_glsl(
    sdf_glsl: str,
    camera_dist: float = 3.5,
    i_time_offset: float = 1.2,
    max_dist: float = 12.0,
) -> str:
    """Wrap a GLSL sdf(vec3 pos) in a ShaderToy-compatible mainImage."""
    tracer = _SPHERE_TRACER_MAINIMAGE.format(
        cam_r=camera_dist,
        i_time_offset=i_time_offset,
        max_d=max_dist,
    )
    return sdf_glsl.strip() + "\n" + tracer


def push_to_playground(
    sdf_cpp_code: str,
    title: str,
    metadata: dict,
    dashboard_url: str = DASHBOARD_URL,
) -> Optional[str]:
    """Convert C++ SDF → validated GLSL → POST to vault.

    Runs the full pipeline through glsl_validator before pushing.
    Returns the shader ID string on success, None if GPU-incompatible or validation fails.
    """
    from autobench.glsl_validator import fix_and_validate
    instance = metadata.get("instance", "")
    ok, errors, full_shader = fix_and_validate(sdf_cpp_code, instance=instance)
    if not ok:
        logger.info("playground_push: '%s' failed validation — not pushed:", title)
        for err in errors:
            logger.info("  %s", err)
        return None

    body = json.dumps({
        "title": title,
        "author": FUNSEARCH_AUTHOR,
        "code": full_shader,
        "language": "glsl",
        "backend_config": {},  # vault loads this as mappedBuffers — keep empty
    }).encode()

    try:
        req = urllib.request.Request(
            f"{dashboard_url}/api/portal/vault/shaders",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            shader_id = result.get("id")
            logger.info("playground_push: '%s' → id=%s", title, shader_id)
            return shader_id
    except Exception as e:
        logger.warning("playground_push: failed to push '%s': %s", title, e)
        return None


def push_best_from_results(
    results_path: Path,
    dashboard_url: str = DASHBOARD_URL,
    top_n: int = 5,
) -> list[str]:
    """Push top-N GLSL-compatible programs from a kernel results JSON.

    Returns list of shader IDs that were successfully pushed.
    """
    results_path = Path(results_path)
    with open(results_path) as f:
        results = json.load(f)

    run_id = results.get("run_id", results_path.stem)[:12]
    top = results.get("top_programs", [])

    pushed: list[str] = []
    rank = 0
    for prog in top:
        code = prog.get("sdf_code") or prog.get("code", "")
        if not code:
            continue

        fitness = prog.get("fitness", 0.0)
        instance = prog.get("instance", "")
        gen = prog.get("generation", 0)

        title = f"[{instance or 'sdf'}] fitness={fitness:.4f} gen={gen} run={run_id}"
        metadata = {"instance": instance, "fitness": fitness, "generation": gen, "run_id": run_id}

        shader_id = push_to_playground(code, title, metadata, dashboard_url=dashboard_url)
        if shader_id:
            pushed.append(shader_id)
            rank += 1
            print(f"  pushed rank {rank}: {title}")
            if rank >= top_n:
                break
        else:
            print(f"  skipped (GPU-incompatible): {title}")

    return pushed


# ---------------------------------------------------------------------------
# CLI entry point — push results from a completed run
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(description="Push FunSearch programs to shader playground")
    p.add_argument("results_file", help="Path to sdf_results_gen*.json")
    p.add_argument("--top-n", type=int, default=5, help="Max programs to push (default: 5)")
    p.add_argument("--dashboard-url", default=DASHBOARD_URL)
    args = p.parse_args()

    ids = push_best_from_results(
        Path(args.results_file),
        dashboard_url=args.dashboard_url,
        top_n=args.top_n,
    )
    print(f"\nPushed {len(ids)} shader(s) to playground at {args.dashboard_url}")
    for sid in ids:
        print(f"  {args.dashboard_url} → vault id: {sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
