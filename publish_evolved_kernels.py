#!/usr/bin/env python3
"""publish_evolved_kernels.py — Publish the 3 evolved kernels to the TEngine shader vault.

Handles the non-terrain kernel types that NervousKernelBridge.publish_to_shader_vault
cannot wrap directly (phase/latent Allen-Cahn reaction kernels and SPH kernel).

Applies:
  - _to_glsl() from the bridge for f-suffix stripping and math-function substitution
  - Additional fixes: fmaxf( → max(, fminf( → min( (not in bridge table)
  - Strips extern "C" prefix from SPH kernel
  - Custom single-pass mainImage wrappers (heatmap/curve — no ping-pong buffers)
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_BENCH = Path("/home/eric/projects/nervous-bus/benchmarks/curriculum/2026-05-31")
PHASE_JSON  = _BENCH / "phase_results_gen16.json"
LATENT_JSON = _BENCH / "latent_results_gen21.json"
SPH_JSON    = _BENCH / "sph_results_gen31.json"

VAULT_URL = "http://127.0.0.1:9104/api/portal/vault/shaders"

# --------------------------------------------------------------------------- #
# Import the bridge's _to_glsl (reuse without subclassing)
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).parent))
from nervous_kernel_bridge import NervousKernelBridge  # noqa: E402

_bridge = NervousKernelBridge()


def _to_glsl_extended(c_code: str) -> str:
    """_to_glsl + extra subs not in the bridge table."""
    glsl = _bridge._to_glsl(c_code)
    # fmaxf / fminf are C extensions not in the bridge table
    glsl = re.sub(r'\bfmaxf\(', 'max(',  glsl)
    glsl = re.sub(r'\bfminf\(', 'min(',  glsl)
    # Strip extern "C" prefix that the SPH template emits
    glsl = re.sub(r'\bextern\s+"C"\s+', '', glsl)
    return glsl


def _assert_clean_glsl(glsl: str, label: str) -> None:
    """Raise if any known C-isms remain that would break WebGL compilation."""
    checks = [
        (r'\bfmaxf\(', 'fmaxf( still present'),
        (r'\bfminf\(', 'fminf( still present'),
        (r'\bextern\s+"C"', 'extern "C" still present'),
        (r'(\d+\.\d*|\d+)f\b', 'float f-suffix still present'),
        (r'\bfabsf\(', 'fabsf( still present'),
        (r'\btanhf\(', 'tanhf( still present'),
        (r'\bsinf\(',  'sinf( still present'),
        (r'\bpowf\(',  'powf( still present'),
    ]
    for pat, msg in checks:
        m = re.search(pat, glsl)
        if m:
            raise ValueError(f"[{label}] GLSL not clean — {msg}: {m.group()!r}")


# --------------------------------------------------------------------------- #
# Kernel 1 — Phase: Allen-Cahn heatmap (phi × temp domain)
# --------------------------------------------------------------------------- #

_PHASE_MAINIMAGE = """
// FunSearch Allen-Cahn phase kernel visualizer.
// Maps the (phi, temp) parameter space to a colour-coded heatmap.
// phi  = x-axis (0→1),  temp = y-axis (0→1).
// Positive reaction → red (growing phase), negative → blue (shrinking),
// near-zero → white.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = fragCoord / iResolution.xy;
    float phi  = uv.x;          // phase field  [0, 1]
    float temp = uv.y;          // temperature  [0, 1]

    float r = reaction(phi, temp);

    // Map signed reaction value to colour:  -3..+3 → blue..white..red
    float t = clamp(r / 3.0 + 0.5, 0.0, 1.0);
    vec3 col = mix(vec3(0.1, 0.3, 0.9), vec3(0.95, 0.95, 0.95), t * 2.0 - clamp(t * 2.0 - 1.0, 0.0, 1.0));
    col = mix(col, vec3(0.9, 0.15, 0.1), max(t * 2.0 - 1.0, 0.0));

    // Grid lines at phi=0.5, temp=0.5
    float grid = step(0.995, abs(sin(uv.x * 3.14159))) * 0.18
               + step(0.995, abs(sin(uv.y * 3.14159))) * 0.18;
    col = mix(col, vec3(0.0), grid);

    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}}
"""

# --------------------------------------------------------------------------- #
# Kernel 2 — Latent: coupled Allen-Cahn heatmap (phi × temp, lap_T animated)
# --------------------------------------------------------------------------- #

_LATENT_MAINIMAGE = """
// FunSearch latent-heat Allen-Cahn kernel visualizer.
// phi  = x-axis,  temp = y-axis,  lap_T = 0.3*sin(iTime).
// Same colour convention as the phase kernel.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = fragCoord / iResolution.xy;
    float phi   = uv.x;
    float temp  = uv.y;
    float lap_T = 0.3 * sin(iTime * 0.8);   // oscillating curvature term

    float r = reaction(phi, temp, lap_T);

    float t = clamp(r / 4.0 + 0.5, 0.0, 1.0);
    vec3 col = mix(vec3(0.05, 0.25, 0.85), vec3(0.95, 0.95, 0.95), t * 2.0 - clamp(t * 2.0 - 1.0, 0.0, 1.0));
    col = mix(col, vec3(0.85, 0.1, 0.05), max(t * 2.0 - 1.0, 0.0));

    // Subtle pulse showing lap_T variation
    float pulse = abs(lap_T) * 0.12;
    col += vec3(pulse * 0.4, pulse * 0.1, -pulse * 0.2);

    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}}
"""

# --------------------------------------------------------------------------- #
# Kernel 3 — SPH: kernel profile curve W(r/h)
# --------------------------------------------------------------------------- #

_SPH_MAINIMAGE = """
// FunSearch SPH kernel visualizer.
// Plots W(r, h=1) as a function of r/h.
// x-axis: r/h ∈ [0, 1.2],  y-axis: normalised W value.
// The compact support cutoff at r/h = 1 is marked with a dashed line.

float sampleW(float rh) {{
    float h = 1.0;
    return sph_kernel(rh * h, h);
}}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = vec3(0.08, 0.08, 0.12);

    float rh_range = 1.3;
    float rh = uv.x * rh_range;

    // Evaluate the curve value at this column
    float W_here = sampleW(rh);
    float W_max  = sampleW(0.0);
    float W_norm = (W_max > 0.0) ? W_here / W_max : 0.0;

    // Plot curve: draw band around the normalised value
    float curve_y = W_norm;
    float dist = abs(uv.y - curve_y);
    float line_w = 2.0 / iResolution.y;
    col = mix(vec3(0.2, 0.75, 0.95), col, smoothstep(0.0, line_w * 2.0, dist));

    // Fill under the curve
    if (uv.y < curve_y) {{
        col = mix(col, vec3(0.1, 0.35, 0.55), 0.35);
    }}

    // Dashed vertical at r/h = 1.0
    float cutoff_x = 1.0 / rh_range;
    float cx = abs(uv.x - cutoff_x);
    float dash = step(0.5, fract(uv.y * 14.0));
    col = mix(col, vec3(0.95, 0.65, 0.1), dash * (1.0 - smoothstep(0.0, 3.0 / iResolution.x, cx)));

    // Axis labels (grid lines)
    col = mix(col, vec3(0.35), step(uv.y, 0.012) + step(uv.x, 0.012 * iResolution.y / iResolution.x));

    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}}
"""

# --------------------------------------------------------------------------- #
# Post helper
# --------------------------------------------------------------------------- #

def post_shader(title: str, code: str, language: str, biome: str, fitness: float,
                generation: int) -> bool:
    payload = json.dumps({
        "title": title,
        "author": "funsearch-autobench",
        "code": code,
        "language": language,
        "backend_config": json.dumps({
            "biome": biome, "fitness": fitness, "generation": generation,
        }),
    }).encode()

    req = urllib.request.Request(
        VAULT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status == 200
            if not ok:
                print(f"  [ERROR] HTTP {resp.status}", file=sys.stderr)
            return ok
    except Exception as exc:
        print(f"  [ERROR] POST failed: {exc}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    results: list[tuple[str, bool]] = []

    # ---- Phase kernel ----
    print("--- Phase kernel (Allen-Cahn reaction heatmap) ---")
    pd = json.loads(PHASE_JSON.read_text())
    p = pd["top_programs"][0]
    phase_fitness    = float(p["fitness"])
    phase_generation = int(pd["generation"])
    phase_code_c     = p["reaction_code"]

    phase_glsl = _to_glsl_extended(phase_code_c)
    _assert_clean_glsl(phase_glsl, "phase")
    phase_full = phase_glsl + _PHASE_MAINIMAGE
    title = f"[phase] gen{phase_generation} — FunSearch fit={phase_fitness:.4f}"
    print(f"  Title: {title}")
    ok = post_shader(title, phase_full, "glsl", "phase", phase_fitness, phase_generation)
    print(f"  -> {'OK' if ok else 'FAILED'}")
    results.append((title, ok))

    # ---- Latent kernel ----
    print("--- Latent kernel (coupled Allen-Cahn heatmap) ---")
    ld = json.loads(LATENT_JSON.read_text())
    lp = ld["top_programs"][0]
    latent_fitness    = float(lp["fitness"])
    latent_generation = int(ld["generation"])
    latent_code_c     = lp["reaction_code"]

    latent_glsl = _to_glsl_extended(latent_code_c)
    _assert_clean_glsl(latent_glsl, "latent")
    latent_full = latent_glsl + _LATENT_MAINIMAGE
    title = f"[latent] gen{latent_generation} — FunSearch fit={latent_fitness:.4f}"
    print(f"  Title: {title}")
    ok = post_shader(title, latent_full, "glsl", "latent", latent_fitness, latent_generation)
    print(f"  -> {'OK' if ok else 'FAILED'}")
    results.append((title, ok))

    # ---- SPH kernel ----
    print("--- SPH kernel (W(r/h) profile curve) ---")
    sd = json.loads(SPH_JSON.read_text())
    sp = sd["top_programs"][0]
    sph_fitness    = float(sp["fitness"])
    sph_generation = int(sd["generation"])
    sph_code_c     = sp["sph_code"]

    sph_glsl = _to_glsl_extended(sph_code_c)
    _assert_clean_glsl(sph_glsl, "sph")
    sph_full = sph_glsl + _SPH_MAINIMAGE
    title = f"[sph] gen{sph_generation} — FunSearch fit={sph_fitness:.4f}"
    print(f"  Title: {title}")
    ok = post_shader(title, sph_full, "glsl", "sph", sph_fitness, sph_generation)
    print(f"  -> {'OK' if ok else 'FAILED'}")
    results.append((title, ok))

    # ---- Summary ----
    print()
    print("=== Publish summary ===")
    all_ok = True
    for t, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {t}")
        if not success:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
