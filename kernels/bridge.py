"""bridge — FunSearch → TEngine SVDAG hot-swap bridge (relocated).

Phase 1 of the kernel restructuring. Behaviour is byte-identical to
``autobench.nervous_kernel_bridge`` — pure relocation, no edits.

When FunSearch discovers a better terrain kernel, this bridge:
  1. Transpiles the evolved C function to Slang
  2. Splices it into nk_svdag_terrain_shape.slang (hot-swap target in TEngine)
  3. Waits 2 frames for vkUpdateIndirectExecutionSetShaderEXT to fire
  4. Reads back GPU cycle count from RES_WORK_TYPE_TELEMETRY for WT 653
  5. Returns nanoseconds

Usage:
    bridge = NervousKernelBridge()
    if bridge.is_available():
        ns = bridge.inject(terrain_c_code)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# C → Slang substitution table (applied in order)
# ---------------------------------------------------------------------------

TRANSPILE: list[tuple[str, str]] = [
    ("fract(",  "frac("),
    ("mix(",    "lerp("),
    ("fabsf(",  "abs("),
    ("sinf(",   "sin("),
    ("powf(",   "pow("),
    ("tanhf(",  "tanh("),
]

# vec2/vec3 are handled via word-boundary regex so that all forms are covered:
#   ` vec2 ` (declaration), `vec2(` (constructor), `(vec2 ` (parameter),
#   `, vec2 ` (second parameter), etc.
_VEC_SUBS: list[tuple[str, str]] = [
    (r"\bvec2\b", "float2"),
    (r"\bvec3\b", "float3"),
]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class BridgeError(Exception):
    """Raised by NervousKernelBridge.inject() on hard failure."""


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class NervousKernelBridge:
    """
    Hot-swap bridge: evolved C terrain function → Slang → TEngine SVDAG → ns timing.

    Usage:
        bridge = NervousKernelBridge()
        if bridge.is_available():
            ns = bridge.inject(terrain_c_code)
    """

    #: Hot-swap target shader — will be absent until the nervous-kernel silo
    #: branch is merged into TEngine.  is_available() returns False until then.
    SHADER_PATH: Path = (
        Path.home()
        / "projects"
        / "tengine"
        / "crates"
        / "tengine-dgc-hal"
        / "shaders"
        / "compute"
        / "svdag"
        / "nk_svdag_terrain_shape.slang"
    )

    #: Bus log written by all nervous-bus producers
    _DEBUG_JSONL: Path = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"

    def is_available(self) -> bool:
        """Returns True if the target shader file exists at the expected path."""
        return self.SHADER_PATH.is_file()

    def inject(self, c_code: str, world_id: int = 0) -> float:
        """Full round-trip: transpile → splice → wait → read timing.

        Returns GPU nanoseconds for WT 653, or raises BridgeError on failure.
        """
        if not self.is_available():
            raise BridgeError(
                f"Shader not found: {self.SHADER_PATH} — TEngine nervous-kernel "
                "branch not yet checked out or shader file not created."
            )

        slang = self._transpile(c_code)
        if not slang.strip():
            raise BridgeError("Transpilation produced empty output.")

        self._splice(slang)
        self._wait_reload()
        return self._read_wt_timing(wt_id=653)

    # ------------------------------------------------------------------
    # Transpilation
    # ------------------------------------------------------------------

    def _transpile(self, c_code: str) -> str:
        """Translate a C terrain function to Slang syntax.

        Applies the TRANSPILE substitution table (ordered, string replace) then
        handles all vec2/vec3 occurrences via word-boundary regex so that
        declarations, constructors, and parameter lists are all covered.
        Also renames the outermost terrain/nk_* function to nk_evolved_fn.
        """
        slang = c_code

        # Ordered string substitutions (function names only; no vec2/vec3 here)
        for c_sym, slang_sym in TRANSPILE:
            slang = slang.replace(c_sym, slang_sym)

        # Word-boundary vector-type substitutions
        for c_pat, slang_sym in _VEC_SUBS:
            slang = re.sub(c_pat, slang_sym, slang)

        # Rename the outermost public function to nk_evolved_fn.
        # Matches: `float <name>(` where <name> is not a private helper
        # (private helpers start with `_`). We rename the *last* non-underscore
        # function definition to preserve helpers like _rh_hash/_rh_vnoise.
        # Strategy: find all public `float <name>(` definitions, rename the last one.
        def _rename_last_public(src: str) -> str:
            pattern = re.compile(r'^(float\s+)([A-Za-z][A-Za-z0-9_]*)(\s*\()', re.MULTILINE)
            matches = list(pattern.finditer(src))
            if not matches:
                return src
            last = matches[-1]
            return src[:last.start()] + last.group(1) + "nk_evolved_fn" + last.group(3) + src[last.end():]

        slang = _rename_last_public(slang)
        return slang

    # ------------------------------------------------------------------
    # Shader splice
    # ------------------------------------------------------------------

    def _splice(self, slang_fn: str) -> None:
        """Write the new function body between the // [NK_EVOLVED_FUNCTION] markers."""
        text = self.SHADER_PATH.read_text()
        marker = "// [NK_EVOLVED_FUNCTION]"
        try:
            start = text.index(marker)
        except ValueError:
            raise BridgeError(
                f"Marker '{marker}' not found in {self.SHADER_PATH}. "
                "The shader may not be the expected hot-swap target."
            )

        end_marker = "public float compute_density"
        try:
            end = text.index(end_marker, start)
        except ValueError:
            raise BridgeError(
                f"End marker '{end_marker}' not found after '{marker}' in {self.SHADER_PATH}."
            )

        new_block = marker + " — hot-swap target; FunSearch kernel\n" + slang_fn + "\n\n"
        self.SHADER_PATH.write_text(text[:start] + new_block + text[end:])

    # ------------------------------------------------------------------
    # Frame wait
    # ------------------------------------------------------------------

    def _wait_reload(self) -> None:
        """Sleep 55ms (≈3 frames at 60fps, covers 2-frame vkUpdateIndirectExecutionSetShaderEXT cooldown)."""
        time.sleep(0.055)

    # ------------------------------------------------------------------
    # GPU timing readback
    # ------------------------------------------------------------------

    def _read_wt_timing(self, wt_id: int = 653) -> float:
        """Read the most recent tse.wt_timing.v1 event for wt_id from debug.jsonl.

        Returns nanoseconds as float.  Returns -1.0 if no matching event is found.

        The event data may contain either:
          - "ns": direct nanoseconds (preferred)
          - "cycles": raw cycle count — converted using the formula from wt_timeline.rs
        """
        if not self._DEBUG_JSONL.is_file():
            return -1.0

        warp_size = 32
        sm_count = int(os.environ.get("TENGINE_GPU_SM_COUNT", "48"))
        gpu_mhz = int(os.environ.get("TENGINE_GPU_CLOCK_MHZ", "2100"))
        divisor = warp_size * sm_count * gpu_mhz * 1000

        target_type = "tse.wt_timing.v1"
        last_event: dict | None = None

        try:
            with open(self._DEBUG_JSONL) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == target_type:
                        last_event = ev
        except OSError:
            return -1.0

        if last_event is None:
            return -1.0

        data = last_event.get("data", {})

        # Try per-WT array first (wt_timeline emits a list of wt entries)
        wt_entries = data.get("wts") or data.get("work_types") or []
        for entry in wt_entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("wt") == wt_id:
                # Direct ns field
                if "ns" in entry:
                    return float(entry["ns"])
                # ms field → convert to ns
                if "ms" in entry:
                    return float(entry["ms"]) * 1_000_000.0
                # Raw cycles
                cycles = entry.get("cycles")
                if cycles is not None and divisor > 0:
                    return (float(cycles) / divisor) * 1e9

        # Flat payload: data = {"wt": 653, "ns": ..., "cycles": ...}
        if data.get("wt") == wt_id:
            if "ns" in data:
                return float(data["ns"])
            if "ms" in data:
                return float(data["ms"]) * 1_000_000.0
            cycles = data.get("cycles")
            if cycles is not None and divisor > 0:
                return (float(cycles) / divisor) * 1e9

        # Fallback: any numeric "cycles" at the data root
        cycles = data.get("cycles")
        if cycles is not None and divisor > 0:
            return (float(cycles) / divisor) * 1e9

        return -1.0

    # ------------------------------------------------------------------
    # Shader vault publisher
    # ------------------------------------------------------------------

    DASHBOARD_URL: str = os.environ.get("TENGINE_DASHBOARD_URL", "http://127.0.0.1:9104")

    def _to_glsl(self, c_code: str) -> str:
        """C → GLSL: replace libc math functions first, then strip float literal f-suffixes.

        Order matters: fabsf( → abs( must happen before the float-suffix strip,
        otherwise the trailing f in fabsf would be eaten and abs( would never match.
        The float-suffix regex requires at least one digit before f (\d+ not \d*)
        so standalone variable names like `f`, `f.x`, `f.y` are never touched.
        """
        # 1. Named math functions (must come before the suffix strip)
        glsl = re.sub(r'\bfabsf\(', 'abs(',  c_code)
        glsl = re.sub(r'\bsinf\(',  'sin(',  glsl)
        glsl = re.sub(r'\bcosf\(',  'cos(',  glsl)
        glsl = re.sub(r'\bpowf\(',  'pow(',  glsl)
        glsl = re.sub(r'\btanhf\(', 'tanh(', glsl)
        glsl = re.sub(r'\bsqrtf\(', 'sqrt(', glsl)
        glsl = re.sub(r'\bexpf\(',  'exp(',  glsl)
        glsl = re.sub(r'\blogf\(',  'log(',  glsl)
        glsl = re.sub(r'\bfloorf\(', 'floor(', glsl)
        glsl = re.sub(r'\bceilf\(',  'ceil(',  glsl)

        # 2. Float literal f-suffix: 1.0f → 1.0, 0.58f → 0.58
        # \d+ requires at least one digit before the f — never matches standalone `f`
        glsl = re.sub(r'(\d+\.\d*|\d+)f\b', r'\1', glsl)
        return glsl

    def _wrap_glsl_mainimage(self, terrain_fn_glsl: str, biome: str, fitness: float) -> str:
        """Wrap evolved terrain function in a Shadertoy-compatible mainImage."""
        return terrain_fn_glsl + f"""
// FunSearch: biome={biome}, fitness={fitness:.4f}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(1024.0 + 300.0*sin(iTime*0.1), 450.0, iTime*10.0);
    vec3 ta = ro + vec3(0.0, -160.0, 500.0);
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0,1.0,0.0)));
    vec3 rd = normalize(uv.x*uu + uv.y*cross(uu,ww) + 1.4*ww);
    vec3 sun = normalize(vec3(0.6, 0.4, 0.3));
    vec3 col = mix(vec3(0.3,0.5,0.82), vec3(0.9,0.8,0.7), pow(max(dot(rd,sun),0.0),4.0));
    float t = 5.0;
    for (int i = 0; i < 80; i++) {{
        vec3 p = ro + rd*t;
        float h = terrain(p.xz * 0.0025) * 360.0 + 40.0;
        float diff = p.y - h;
        if (diff < 0.5 || t > 3500.0) break;
        t += max(diff * 0.6, 3.0);
    }}
    if (t < 3490.0) {{
        vec3 p = ro + rd*t;
        vec2 xz = p.xz * 0.0025;
        float e = 0.01;
        vec3 n = normalize(vec3(terrain(xz-vec2(e,0.0))-terrain(xz+vec2(e,0.0)),
                                2.0*e,
                                terrain(xz-vec2(0.0,e))-terrain(xz+vec2(0.0,e))));
        float h = terrain(xz)*360.0+40.0;
        float slope = 1.0 - n.y;
        vec3 mat = mix(vec3(0.28,0.46,0.18), vec3(0.38,0.30,0.22), clamp(slope*2.0,0.0,1.0));
        mat = mix(mat, vec3(0.88,0.94,0.98), smoothstep(305.0,375.0,h));
        mat = mix(mat, vec3(0.18,0.32,0.54), smoothstep(80.0,52.0,h));
        col = mat*(max(dot(n,sun),0.0)*vec3(1.35,1.1,0.85)+max(n.y,0.0)*vec3(0.1,0.16,0.26));
        col = mix(col, vec3(0.52,0.66,0.84), 1.0-exp(-t*0.00022));
    }}
    fragColor = vec4(pow(clamp(col,0.0,1.0), vec3(0.4545)), 1.0);
}}"""

    def publish_to_shader_vault(
        self,
        c_code: str,
        biome: str,
        fitness: float,
        generation: int,
        *,
        language: str = "glsl",
    ) -> bool:
        """POST evolved kernel to the shader studio vault.

        The shader appears immediately in the Cloud tab — no rebuild needed.
        Returns True on success, False on failure (never raises).
        """
        import urllib.request

        try:
            if language == "glsl":
                # Find the public terrain function: last float X(vec2 that doesn't
                # start with '_' (helpers are prefixed _; the main function is last).
                fn_names = re.findall(r'float\s+(\w+)\s*\(vec2', c_code)
                public = [n for n in fn_names if not n.startswith('_')]
                fn_name_str = public[-1] if public else (fn_names[-1] if fn_names else "terrain")
                glsl = self._to_glsl(c_code)
                # rename main terrain function to terrain() for the wrapper
                glsl = re.sub(
                    r'\bfloat\s+' + re.escape(fn_name_str) + r'\s*\(',
                    'float terrain(',
                    glsl,
                )
                code = self._wrap_glsl_mainimage(glsl, biome, fitness)
            else:
                code = self._transpile(c_code)

            payload = json.dumps({
                "title": f"[{biome}] gen{generation} — FunSearch fit={fitness:.4f}",
                "author": "funsearch-autobench",
                "code": code,
                "language": language,
                "backend_config": json.dumps({
                    "biome": biome, "fitness": fitness, "generation": generation,
                }),
            }).encode()

            req = urllib.request.Request(
                f"{self.DASHBOARD_URL}/api/portal/vault/shaders",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception as exc:
            print(f"[NK bridge] vault publish failed: {exc}", file=sys.stderr)
            return False


# ---------------------------------------------------------------------------
# Rolling hills test function (canonical FunSearch candidate)
# ---------------------------------------------------------------------------

_ROLLING_HILLS_C = """\
float _rh_hash(float n) { return fract(sinf(n) * 43758.5453f); }
float _rh_vnoise(vec2 x) {
    vec2 i = floor(x); vec2 f = fract(x); f = f*f*f*(10.0f - 15.0f*f + 6.0f*f*f);
    float n00 = _rh_hash(i.x + i.y * 57.0f); float n10 = _rh_hash(i.x + 1.0f + i.y * 57.0f);
    float n01 = _rh_hash(i.x + (i.y+1.0f) * 57.0f); float n11 = _rh_hash(i.x + 1.0f + (i.y+1.0f) * 57.0f);
    return mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y);
}
float nk_rolling_hills(vec2 p) {
    float v = 0.0f, a = 0.58f;
    for (int i = 0; i < 5; i++) {
        v += a * _rh_vnoise(p);
        p = vec2(p.x * 1.78f + p.y * 0.35f, p.x * 0.35f + p.y * 1.78f);
        a *= 0.53f;
    }
    return clamp(v, 0.0f, 1.0f);
}"""


# ---------------------------------------------------------------------------
# __main__ — standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bridge = NervousKernelBridge()
    available = bridge.is_available()
    print(f"NervousKernelBridge: available={available}"
          + ("" if available else " (shader not found yet)"))

    # Always show the transpiled Slang so the user can inspect it
    slang = bridge._transpile(_ROLLING_HILLS_C)
    print("--- Transpiled Slang ---")
    print(slang)

    if available:
        try:
            ns = bridge.inject(_ROLLING_HILLS_C)
            print(f"\nWT 653 timing: {ns:.1f} ns")
        except BridgeError as exc:
            print(f"\n[bridge] inject() failed: {exc}", file=sys.stderr)
