"""Shader executor for the autobench shader benchmark family.

Pipeline:
    1. Validate / compile the GLSL fragment shader with `glslangValidator -S frag`.
       (Compile errors → Verdict.CE.)
    2. Wrap it in a fullscreen-triangle vertex shader, render via moderngl headless
       (EGL backend) at the requested viewport size with `iResolution` and `iTime`
       uniforms set.
    3. Capture the rendered frame to PNG.
    4. Score against a reference image with SSIM (skimage.metrics.structural_similarity).
       Optionally compute LPIPS if `lpips` is importable.
    5. Map continuous score → verdict:
            SSIM ≥ 0.95           → OK   (p_score = SSIM)
            0.80 ≤ SSIM < 0.95    → VF   (p_score = SSIM)
            SSIM < 0.80           → WA   (p_score = SSIM)
            compile failure       → CE
            runtime crash         → RE
            render_time_ms > target → TLE

Rendering backend
-----------------
moderngl over EGL was picked because:
    - It works headless on this Linux host (NVIDIA proprietary driver +
      `EGL_PLATFORM=surfaceless` succeeds, verified at module-import time
      with a lightweight ctx probe).
    - It is pip-installable (`moderngl`, `glcontext`); no system-level X
      / Wayland session is required, so the same code path works on CI runners.
    - Falls back to a `NotImplementedError` (not a fake render) if EGL is
      unavailable on a given host, so test suites can mark cases
      `@pytest.mark.requires_gpu` and skip cleanly.

Pyopengl was explicitly rejected (too heavyweight, manual FBO juggling).
glslviewer was rejected as a last-resort fallback because the binary isn't
present on this host and adding a system package is out of scope for tier 1.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core import Verdict


# Default viewport
DEFAULT_VIEWPORT: tuple[int, int] = (512, 512)
DEFAULT_TARGET_MS: float = 5000.0

# SSIM thresholds — tier 1 benchmark
SSIM_OK_THRESHOLD: float = 0.95
SSIM_VF_THRESHOLD: float = 0.80


# Fullscreen-triangle vertex shader used to invoke any ShaderToy-style
# `mainImage(out vec4, in vec2)` fragment shader.
FULLSCREEN_VERTEX_SHADER: str = """\
#version 330 core
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Fragment-shader wrapper. The user-supplied shader is expected to define
# `void mainImage(out vec4 fragColor, in vec2 fragCoord)`. We forward
# `gl_FragCoord.xy` to it.
FRAGMENT_WRAPPER_HEAD: str = """\
#version 330 core
out vec4 fragColor;
uniform vec2 iResolution;
uniform float iTime;
"""

FRAGMENT_WRAPPER_TAIL: str = """\

void main() {
    mainImage(fragColor, gl_FragCoord.xy);
}
"""


@dataclass
class ShaderRunResult:
    """Result of running a single shader against a reference image."""

    verdict: Verdict = Verdict.OK
    ssim: float = 0.0
    lpips: float | None = None
    render_time_ms: float = 0.0
    compile_log: str = ""
    frame_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "render_time_ms": self.render_time_ms,
            "compile_log": self.compile_log,
            "frame_path": self.frame_path,
            "error": self.error,
        }


@dataclass
class ComputeRunResult:
    """Result of running a compute shader."""

    status: str = "OK"  # "OK" | "CE" | "RE"
    error: str = ""
    elapsed_ms: float = 0.0
    output_data: list = field(default_factory=list)  # readback from output SSBO

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "output_count": len(self.output_data),
        }


def wrap_fragment_shader(shader_src: str) -> str:
    """Wrap a `mainImage`-style fragment shader with our `#version 330 core`
    header (with iResolution/iTime uniforms) and a `void main()` entrypoint.

    If the shader already declares `#version`, we strip our header to avoid
    duplicates (only the first `#version` line is kept).
    """
    body = shader_src.strip()
    head = FRAGMENT_WRAPPER_HEAD
    if body.lstrip().startswith("#version"):
        # User supplied their own version directive — drop our head's #version line
        # but keep the uniform declarations.
        head = "\n".join(FRAGMENT_WRAPPER_HEAD.splitlines()[1:]) + "\n"
    return head + "\n" + body + "\n" + FRAGMENT_WRAPPER_TAIL


def validate_glsl(shader_src: str) -> tuple[bool, str]:
    """Validate GLSL fragment shader with `glslangValidator -S frag`.

    Returns (ok, log). On compile failure, ok=False and log carries stderr+stdout.
    The wrapped (final) shader is what we validate so the AC mirrors the actual
    pipeline.
    """
    final_src = wrap_fragment_shader(shader_src)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".frag", delete=False, encoding="utf-8"
    ) as f:
        f.write(final_src)
        path = f.name
    try:
        proc = subprocess.run(
            ["glslangValidator", "-S", "frag", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        log = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, log
    except FileNotFoundError:
        return False, "glslangValidator not installed"
    except subprocess.TimeoutExpired:
        return False, "glslangValidator timed out (>10s)"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class ShaderExecutor:
    """Headless shader executor.

    Lazy-initializes a moderngl context (EGL backend) on first render call so
    importing this module is cheap and so unit tests that only exercise the
    GLSL validator path don't need a GPU.
    """

    def __init__(
        self,
        viewport: tuple[int, int] = DEFAULT_VIEWPORT,
        target_ms: float = DEFAULT_TARGET_MS,
    ):
        self.viewport = viewport
        self.target_ms = target_ms
        self._ctx: Any = None
        self._fbo: Any = None
        self._vao_cache: dict[str, Any] = {}
        self._compute_available: bool = False

    # ------------------------------------------------------------------
    # Context lifecycle
    # ------------------------------------------------------------------

    def _ensure_ctx(self) -> Any:
        """Create an EGL standalone context on first use."""
        if self._ctx is not None:
            return self._ctx
        try:
            import moderngl  # type: ignore
        except ImportError as e:
            raise NotImplementedError(
                "moderngl is required for ShaderExecutor rendering. "
                "Install with `pip install moderngl`."
            ) from e

        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        try:
            try:
                ctx = moderngl.create_context(standalone=True, backend="egl", require=450)
            except Exception:
                ctx = moderngl.create_context(standalone=True, backend="egl")
        except Exception as e:
            raise NotImplementedError(
                f"Could not create headless EGL context: {e}. "
                "GPU rendering is not available on this host."
            ) from e

        self._ctx = ctx
        self._compute_available = ctx.version_code >= 430
        return ctx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        shader_src: str,
        out_path: str | Path,
        viewport: tuple[int, int] | None = None,
        i_time: float = 0.0,
    ) -> tuple[bool, str, float]:
        """Compile + render the shader into a PNG at out_path.

        Returns (success, log, render_time_ms). On compile failure success is
        False and log carries the compiler's stderr.
        """
        viewport = viewport or self.viewport

        # 1) Pre-validate
        ok, log = validate_glsl(shader_src)
        if not ok:
            return False, log, 0.0

        # 2) Render
        try:
            import moderngl  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise NotImplementedError(
                f"moderngl/Pillow not installed: {e}"
            ) from e

        ctx = self._ensure_ctx()

        wrapped = wrap_fragment_shader(shader_src)
        try:
            prog = ctx.program(
                vertex_shader=FULLSCREEN_VERTEX_SHADER,
                fragment_shader=wrapped,
            )
        except Exception as e:
            # moderngl raises on link/compile failure inside the GL driver,
            # even though glslangValidator passed (driver may be stricter).
            return False, f"GL program link error: {e}", 0.0

        try:
            prog["iResolution"].value = (float(viewport[0]), float(viewport[1]))
        except KeyError:
            pass
        try:
            prog["iTime"].value = float(i_time)
        except KeyError:
            pass

        verts = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        vbo = ctx.buffer(verts.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

        tex = ctx.texture(viewport, 4)
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)

        start = time.perf_counter()
        try:
            vao.render(moderngl.TRIANGLES)
            ctx.finish()
        except Exception as e:
            return False, f"GL render error: {e}", 0.0
        render_ms = (time.perf_counter() - start) * 1000.0

        raw = fbo.read(components=4)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(viewport[1], viewport[0], 4)
        arr = np.flipud(arr)  # GL origin is bottom-left; PNG is top-left.
        Image.fromarray(arr).save(str(out_path))

        # Release per-render GL objects (the context persists)
        try:
            fbo.release()
            tex.release()
            vao.release()
            vbo.release()
            prog.release()
        except Exception:
            pass

        return True, "", render_ms

    def score_ssim(self, frame_path: str | Path, reference_path: str | Path) -> float:
        """SSIM between two PNGs in RGB (alpha dropped). Returns a float in [-1, 1]."""
        from PIL import Image  # type: ignore
        from skimage.metrics import structural_similarity as ssim  # type: ignore

        a = np.array(Image.open(str(frame_path)).convert("RGB"))
        b = np.array(Image.open(str(reference_path)).convert("RGB"))
        if a.shape != b.shape:
            # Resize a to b
            a_img = Image.fromarray(a).resize(
                (b.shape[1], b.shape[0]), Image.LANCZOS
            )
            a = np.array(a_img)
        # skimage SSIM with channel_axis=-1 averages per-channel
        score = ssim(a, b, channel_axis=-1, data_range=255)
        return float(score)

    def score_lpips(self, frame_path: str | Path, reference_path: str | Path) -> float | None:
        """Compute LPIPS between two images, or None if lpips isn't available."""
        try:
            import lpips  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            return None

        model = lpips.LPIPS(net="alex")
        a = np.array(Image.open(str(frame_path)).convert("RGB")).astype("f4") / 127.5 - 1.0
        b = np.array(Image.open(str(reference_path)).convert("RGB")).astype("f4") / 127.5 - 1.0
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            d = model(ta, tb)
        return float(d.item())

    def render_only(
        self,
        shader_src: str,
        out_path: str | Path,
        viewport: tuple[int, int] | None = None,
        i_time: float = 0.0,
    ) -> ShaderRunResult:
        """Compile and render to out_path without any reference scoring.

        Returns ShaderRunResult with ssim=0.0. Use when you only want the PNG,
        not a quality comparison.
        """
        viewport = viewport or self.viewport
        out_path = str(out_path)
        ok, log = validate_glsl(shader_src)
        if not ok:
            return ShaderRunResult(verdict=Verdict.CE, compile_log=log[:2000],
                                   frame_path="", error="glsl compile failed")
        try:
            ok2, log2, render_ms = self.render(shader_src, out_path, viewport=viewport, i_time=i_time)
        except Exception as e:
            return ShaderRunResult(verdict=Verdict.RE, frame_path="",
                                   error=f"render error: {e}")
        if not ok2:
            return ShaderRunResult(verdict=Verdict.RE, render_time_ms=0.0,
                                   compile_log=log2[:2000], frame_path=out_path,
                                   error=log2[:500])
        return ShaderRunResult(verdict=Verdict.OK, ssim=0.0, render_time_ms=render_ms,
                               compile_log=log[:2000], frame_path=out_path)

    def run(
        self,
        shader_src: str,
        reference_path: str | Path,
        viewport: tuple[int, int] | None = None,
        out_path: str | Path | None = None,
        target_ms: float | None = None,
        i_time: float = 0.0,
        compute_lpips: bool = False,
    ) -> ShaderRunResult:
        """End-to-end: compile, render, score, return ShaderRunResult."""
        viewport = viewport or self.viewport
        target_ms = target_ms if target_ms is not None else self.target_ms

        if out_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            out_path = tmp.name
        out_path = str(out_path)

        # 1) validate
        ok, log = validate_glsl(shader_src)
        if not ok:
            return ShaderRunResult(
                verdict=Verdict.CE,
                compile_log=log[:2000],
                frame_path="",
                error="glsl compile failed",
            )

        # 2) render
        try:
            ok2, log2, render_ms = self.render(
                shader_src, out_path, viewport=viewport, i_time=i_time
            )
        except NotImplementedError as e:
            return ShaderRunResult(
                verdict=Verdict.RE,
                compile_log=log[:2000],
                frame_path="",
                error=f"render backend unavailable: {e}",
            )
        except Exception as e:
            return ShaderRunResult(
                verdict=Verdict.RE,
                compile_log=log[:2000],
                frame_path="",
                error=f"render crashed: {e}",
            )

        if not ok2:
            # GL-driver-level compile/link failure
            return ShaderRunResult(
                verdict=Verdict.CE,
                compile_log=(log + "\n---\n" + log2)[:2000],
                frame_path="",
                error="GL link failed after glslang pass",
            )

        # 3) TLE?
        if render_ms > target_ms:
            return ShaderRunResult(
                verdict=Verdict.TLE,
                ssim=0.0,
                render_time_ms=render_ms,
                compile_log=log[:2000],
                frame_path=out_path,
                error=f"render_time_ms {render_ms:.1f} > target {target_ms:.1f}",
            )

        # 4) score
        ref_path = Path(reference_path)
        if not ref_path.exists():
            return ShaderRunResult(
                verdict=Verdict.RE,
                render_time_ms=render_ms,
                compile_log=log[:2000],
                frame_path=out_path,
                error=f"reference image not found: {reference_path}",
            )

        try:
            ssim_score = self.score_ssim(out_path, ref_path)
        except Exception as e:
            return ShaderRunResult(
                verdict=Verdict.RE,
                render_time_ms=render_ms,
                compile_log=log[:2000],
                frame_path=out_path,
                error=f"ssim scoring failed: {e}",
            )

        lpips_score: float | None = None
        if compute_lpips:
            try:
                lpips_score = self.score_lpips(out_path, ref_path)
            except Exception:
                lpips_score = None

        # 5) verdict mapping
        verdict = verdict_from_ssim(ssim_score)

        return ShaderRunResult(
            verdict=verdict,
            ssim=ssim_score,
            lpips=lpips_score,
            render_time_ms=render_ms,
            compile_log=log[:2000],
            frame_path=out_path,
            error="",
        )

    @property
    def compute_available(self) -> bool:
        """True if OpenGL 4.3+ compute shaders are available."""
        try:
            self._ensure_ctx()
            return self._compute_available
        except Exception:
            return False

    def run_compute(
        self,
        compute_src: str,
        input_data: list[float],
        output_size: int,
        local_size_x: int = 64,
    ) -> ComputeRunResult:
        """Run a compute shader over input_data, return output_data.

        compute_src: GLSL compute shader source. Must declare:
            layout(local_size_x=N) in;
            layout(std430, binding=0) buffer Input { float data[]; };
            layout(std430, binding=1) buffer Output { float out_data[]; };
        input_data: list of floats written to binding=0
        output_size: number of floats to read back from binding=1
        """
        try:
            ctx = self._ensure_ctx()
        except NotImplementedError as e:
            return ComputeRunResult(status="RE", error=str(e))

        if not self._compute_available:
            return ComputeRunResult(
                status="CE", error="compute shaders require OpenGL 4.3+"
            )

        try:
            cs = ctx.compute_shader(compute_src)
        except Exception as e:
            return ComputeRunResult(status="CE", error=str(e))

        try:
            in_buf = ctx.buffer(struct.pack(f"{len(input_data)}f", *input_data))
            out_buf = ctx.buffer(b"\x00" * output_size * 4)
            in_buf.bind_to_storage_buffer(0)
            out_buf.bind_to_storage_buffer(1)
            start = time.perf_counter()
            cs.run(group_x=math.ceil(max(len(input_data), output_size) / local_size_x))
            ctx.finish()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            raw = out_buf.read()
            output = list(struct.unpack(f"{output_size}f", raw))
            try:
                in_buf.release()
                out_buf.release()
                cs.release()
            except Exception:
                pass
            return ComputeRunResult(status="OK", elapsed_ms=elapsed_ms, output_data=output)
        except Exception as e:
            return ComputeRunResult(status="RE", error=str(e))

    def run_compute_pingpong(
        self,
        compute_src: str,
        initial_state: list[float],
        n_steps: int,
        local_size_x: int = 64,
        readback_every: int = 0,
    ) -> tuple[ComputeRunResult, list[list[float]]]:
        """Run a compute shader N times with ping-pong SSBOs (physics timestep loop).

        The compute shader reads from binding=0 (current state) and writes to
        binding=1 (next state). After each step the buffers are swapped.

        readback_every: if > 0, read back state every N steps (0 = only final state).
        Returns: (final_result, list_of_snapshots)
        """
        try:
            ctx = self._ensure_ctx()
        except NotImplementedError as e:
            return ComputeRunResult(status="RE", error=str(e)), []

        if not self._compute_available:
            return ComputeRunResult(
                status="CE", error="compute shaders require OpenGL 4.3+"
            ), []

        try:
            cs = ctx.compute_shader(compute_src)
        except Exception as e:
            return ComputeRunResult(status="CE", error=str(e)), []

        n = len(initial_state)
        packed = struct.pack(f"{n}f", *initial_state)
        snapshots: list[list[float]] = []

        try:
            buf_a = ctx.buffer(packed)
            buf_b = ctx.buffer(packed)
            group_x = math.ceil(n / local_size_x)
            start = time.perf_counter()
            for step in range(n_steps):
                buf_a.bind_to_storage_buffer(0)
                buf_b.bind_to_storage_buffer(1)
                cs.run(group_x=group_x)
                ctx.finish()
                buf_a, buf_b = buf_b, buf_a
                if readback_every > 0 and (step + 1) % readback_every == 0:
                    raw = buf_a.read()
                    snapshots.append(list(struct.unpack(f"{n}f", raw)))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            raw = buf_a.read()
            output = list(struct.unpack(f"{n}f", raw))
            try:
                buf_a.release()
                buf_b.release()
                cs.release()
            except Exception:
                pass
            return (
                ComputeRunResult(status="OK", elapsed_ms=elapsed_ms, output_data=output),
                snapshots,
            )
        except Exception as e:
            return ComputeRunResult(status="RE", error=str(e)), snapshots

    def close(self) -> None:
        """Release the GL context if it was created."""
        if self._ctx is not None:
            try:
                self._ctx.release()
            except Exception:
                pass
            self._ctx = None


def verdict_from_ssim(ssim_score: float) -> Verdict:
    """Map a continuous SSIM in [-1, 1] onto a Verdict for tier-1 shader cases."""
    if ssim_score >= SSIM_OK_THRESHOLD:
        return Verdict.OK
    if ssim_score >= SSIM_VF_THRESHOLD:
        return Verdict.VF
    return Verdict.WA
