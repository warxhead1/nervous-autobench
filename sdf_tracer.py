"""sdf_tracer — lightweight CPU sphere tracer for SDF visualization.

Compiles the evolved C++ sdf() function together with a minimal sphere-tracing
harness and produces a PPM frame, then converts to PNG via PIL.  Zero GPU memory,
zero ShaderExecutor dependency, works fully headless.

The math is SDF-faithful: same finite-difference normals (e=0.001), same march
step factor (0.75×), same hit epsilon (0.0015×d) as the GLSL probe in
artifact_store.py — so the geometry visible here is identical to what TEngine
would raymarch at the SDF evaluation level.  Camera and lighting are a preview
convention, not verified against TEngine's swapchain constants.

Render is gated to unsandboxed context only (mirrors the executor's trust level).
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# C++ sphere-tracing skeleton — matches GLSL probe in artifact_store.py exactly
# ---------------------------------------------------------------------------

_CPP_TRACER_SKELETON = r"""
// Lightweight SDF sphere tracer — SDF-faithful preview renderer.
// Matches the GLSL probe in artifact_store.py:
//   camera  : ro = (dist*cos(t), 0.8*dist, dist*sin(t)); looks at origin
//   march   : 96 steps, 0.75x step, |h|<0.0015*d || d>MAX_D terminates
//   normals : finite differences, e=0.001 (same as TEngine SDF evaluation)
//   lighting: ld=normalize(1.5,2.0,-0.5), Blinn-Phong k=48, gamma 0.4545
// Output: binary P6 PPM on stdout.
#include <bits/stdc++.h>
using namespace std;

extern "C" float sdf(float x, float y, float z);
// SDF_INJECT_BELOW

struct V3 { float x, y, z; };
static inline V3 v3(float x, float y, float z) { return {x,y,z}; }
static inline V3 vadd(V3 a, V3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline V3 vsub(V3 a, V3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline V3 vmul(V3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
static inline float vdot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline V3 vcross(V3 a, V3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
static inline V3 vnorm(V3 a) {
    float l = sqrtf(vdot(a,a));
    return l > 1e-9f ? vmul(a, 1.0f/l) : v3(0,1,0);
}
static inline float sdf3(V3 p) { return sdf(p.x, p.y, p.z); }
static inline float clamp01(float x) { return fmaxf(0.0f, fminf(1.0f, x)); }

int main(int argc, char** argv) {
    int W = WIDTH_PLACEHOLDER, H = HEIGHT_PLACEHOLDER;
    float cam_t  = TIME_PLACEHOLDER;
    float cam_r  = CAMDIST_PLACEHOLDER;
    float max_d  = MAXD_PLACEHOLDER;

    // Camera — orbit
    V3 ro = v3(cam_r * cosf(cam_t), 0.8f * cam_r, cam_r * sinf(cam_t));
    V3 ww = vnorm(vsub(v3(0,0,0), ro));
    V3 uu = vnorm(vcross(ww, v3(0,1,0)));
    V3 vv = vcross(uu, ww);
    float focal = 1.7f;

    // PPM header
    fprintf(stdout, "P6\n%d %d\n255\n", W, H);

    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            float u = ((float)px + 0.5f) / (float)H - 0.5f * (float)W / (float)H;
            float v = 0.5f - ((float)py + 0.5f) / (float)H;
            V3 rd = vnorm(vadd(vadd(vmul(uu, u), vmul(vv, v)), vmul(ww, focal)));

            // Sphere march
            float d = 0.001f;
            for (int i = 0; i < 96; i++) {
                float h = sdf3(vadd(ro, vmul(rd, d)));
                if (fabsf(h) < 0.0015f * d || d > max_d) break;
                d += h * 0.75f;
            }

            float cr, cg, cb;
            if (d < max_d) {
                // Surface hit
                V3 p = vadd(ro, vmul(rd, d));
                float e = 0.001f;
                V3 n = vnorm(v3(
                    sdf(p.x+e,p.y,p.z) - sdf(p.x-e,p.y,p.z),
                    sdf(p.x,p.y+e,p.z) - sdf(p.x,p.y-e,p.z),
                    sdf(p.x,p.y,p.z+e) - sdf(p.x,p.y,p.z-e)
                ));
                V3 ld = vnorm(v3(1.5f, 2.0f, -0.5f));
                float diff = fmaxf(vdot(n, ld), 0.0f);
                V3 refl = vsub(vmul(n, 2.0f * vdot(n, ld)), ld);
                float spec = powf(fmaxf(vdot(refl, vmul(rd, -1.0f)), 0.0f), 48.0f) * 0.7f;

                cr = 0.15f*diff + spec + 0.08f*(1.0f-diff);
                cg = 0.55f*diff + spec + 0.04f*(1.0f-diff);
                cb = 0.85f*diff + spec + 0.12f*(1.0f-diff);
                // Fog
                float fog = expf(-d * 0.12f);
                cr *= fog; cg *= fog; cb *= fog;
            } else {
                // Background — deep space
                cr = 0.04f; cg = 0.02f; cb = 0.07f;
            }

            // Gamma 1/2.2
            putchar((unsigned char)(powf(clamp01(cr), 0.4545f) * 255.0f + 0.5f));
            putchar((unsigned char)(powf(clamp01(cg), 0.4545f) * 255.0f + 0.5f));
            putchar((unsigned char)(powf(clamp01(cb), 0.4545f) * 255.0f + 0.5f));
        }
    }
    return 0;
}
"""


def _build_tracer_source(
    sdf_code: str,
    width: int,
    height: int,
    i_time: float,
    camera_dist: float,
    max_dist: float,
) -> str:
    src = _CPP_TRACER_SKELETON
    src = src.replace("WIDTH_PLACEHOLDER", str(width))
    src = src.replace("HEIGHT_PLACEHOLDER", str(height))
    src = src.replace("TIME_PLACEHOLDER", f"{i_time:.4f}f")
    src = src.replace("CAMDIST_PLACEHOLDER", f"{camera_dist:.4f}f")
    src = src.replace("MAXD_PLACEHOLDER", f"{max_dist:.4f}f")
    src = src.replace("// SDF_INJECT_BELOW", sdf_code)
    return src


def render_sdf_cpp_to_png(
    sdf_code: str,
    out_path: Path,
    viewport: tuple[int, int] = (256, 256),
    i_time: float = 1.2,
    camera_dist: float = 3.5,
    max_dist: float = 12.0,
    compile_timeout: float = 30.0,
    render_timeout: float = 60.0,
) -> bool:
    """Compile the C++ sdf() and render a sphere-traced PNG.

    Requires PIL (Pillow).  Only calls out to g++ and the compiled binary via
    subprocess — no GPU, no display, no ShaderExecutor.

    camera_dist: orbit radius. Rule of thumb: bbox_max * 2.0 for all instances
    to keep the shape in frame.  cloud_cluster (bbox 2.0) → dist 4.0,
    gyroid (bbox 1.5) → dist 3.0.  Default 3.5 is safe for bbox ≤ 1.5.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("sdf_tracer: PIL not available, cannot render")
        return False

    W, H = viewport
    source = _build_tracer_source(sdf_code, W, H, i_time, camera_dist, max_dist)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_p = Path(tmpdir) / "tracer.cpp"
            bin_p = Path(tmpdir) / "tracer"
            ppm_p = Path(tmpdir) / "out.ppm"

            src_p.write_text(source)

            # Compile
            r = subprocess.run(
                ["g++", "-O2", "-std=c++14", "-o", str(bin_p), str(src_p), "-lm"],
                capture_output=True, timeout=compile_timeout,
            )
            if r.returncode != 0:
                logger.warning(
                    "sdf_tracer compile failed:\n%s", r.stderr.decode(errors="replace")
                )
                return False

            # Run — write binary PPM to file to avoid null-byte stdout issues
            with open(ppm_p, "wb") as fout:
                r = subprocess.run(
                    [str(bin_p)],
                    stdout=fout,
                    stderr=subprocess.DEVNULL,
                    timeout=render_timeout,
                )
            if r.returncode != 0:
                logger.warning("sdf_tracer render exited %d", r.returncode)
                return False

            img = Image.open(ppm_p)
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            return True

    except subprocess.TimeoutExpired:
        logger.warning("sdf_tracer timed out (compile=%.0fs, render=%.0fs)", compile_timeout, render_timeout)
        return False
    except Exception as e:
        logger.warning("sdf_tracer failed: %s", e)
        return False
