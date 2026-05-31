"""Tests for autobench.shader_executor.

Tests grouped:
    * unit / no GPU      — wrap_fragment_shader, verdict_from_ssim, validate_glsl
    * GPU (requires_gpu) — render, score_ssim, full run() round-trips

Tests that require a working EGL context are marked `requires_gpu` so they
skip cleanly on CI runners without a GPU. We detect availability by trying
to create a moderngl standalone EGL context at module import time.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from autobench.core import Verdict
from autobench.shader_executor import (
    SSIM_OK_THRESHOLD,
    SSIM_VF_THRESHOLD,
    ShaderExecutor,
    ShaderRunResult,
    validate_glsl,
    verdict_from_ssim,
    wrap_fragment_shader,
)


# ---------------------------------------------------------------------------
# GPU availability probe
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        import moderngl  # type: ignore
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        ctx = moderngl.create_context(standalone=True, backend="egl")
        ctx.release()
        return True
    except Exception:
        return False


_GPU = _gpu_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="no headless GL/EGL available")


# A trivial fragment shader that paints a solid color.
SOLID_RED = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

SOLID_BLUE = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(0.0, 0.0, 1.0, 1.0);
}
"""

# A gradient shader — used as the "reference" for the modified-shader test
# since SSIM on uniform solid-color images is degenerate (zero variance →
# luminance term collapses, score drops sharply for small mean shifts).
GRADIENT_SHADER = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;
    fragColor = vec4(uv.x, uv.y, 0.5, 1.0);
}
"""

# Same gradient with a small luminance shift — structurally identical, so
# SSIM should stay high but below 1.0.
GRADIENT_MODIFIED = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution;
    fragColor = vec4(uv.x * 0.95, uv.y * 0.95, 0.5, 1.0);
}
"""

# Syntactically broken GLSL.
BROKEN_SHADER = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // missing semicolon and unknown function
    fragColor = nonsenseFn(1.0, 0.0, 0.0, 1.0)
}
"""


# ---------------------------------------------------------------------------
# Unit tests (no GPU)
# ---------------------------------------------------------------------------


class TestVerdictMapping:
    def test_ok_threshold(self):
        assert verdict_from_ssim(1.0) == Verdict.OK
        assert verdict_from_ssim(SSIM_OK_THRESHOLD) == Verdict.OK
        assert verdict_from_ssim(0.99) == Verdict.OK

    def test_vf_band(self):
        assert verdict_from_ssim(SSIM_VF_THRESHOLD) == Verdict.VF
        assert verdict_from_ssim(0.87) == Verdict.VF
        assert verdict_from_ssim(SSIM_OK_THRESHOLD - 0.0001) == Verdict.VF

    def test_wa_floor(self):
        assert verdict_from_ssim(0.5) == Verdict.WA
        assert verdict_from_ssim(SSIM_VF_THRESHOLD - 0.0001) == Verdict.WA
        assert verdict_from_ssim(-1.0) == Verdict.WA


class TestWrapFragmentShader:
    def test_adds_version_and_main(self):
        wrapped = wrap_fragment_shader(SOLID_RED)
        assert "#version 330 core" in wrapped
        assert "uniform vec2 iResolution" in wrapped
        assert "uniform float iTime" in wrapped
        assert "void main()" in wrapped

    def test_preserves_user_version_directive(self):
        body = "#version 420 core\nvoid mainImage(out vec4 c, in vec2 f) { c = vec4(0); }"
        wrapped = wrap_fragment_shader(body)
        # User's directive kept, ours not duplicated
        assert wrapped.count("#version") == 1
        assert "#version 420 core" in wrapped


class TestValidateGlsl:
    def test_valid_shader(self):
        ok, log = validate_glsl(SOLID_RED)
        assert ok, f"expected valid, got log: {log}"

    def test_invalid_shader(self):
        ok, log = validate_glsl(BROKEN_SHADER)
        assert not ok
        assert "error" in log.lower() or "Error" in log


# ---------------------------------------------------------------------------
# GPU-dependent tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestShaderExecutorRender:
    def test_compile_error_returns_ce(self, tmp_path):
        ex = ShaderExecutor(viewport=(64, 64))
        # We need a reference to call .run(); create a dummy png.
        from PIL import Image
        ref = tmp_path / "ref.png"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(ref)

        result = ex.run(BROKEN_SHADER, reference_path=ref, viewport=(64, 64))
        assert result.verdict == Verdict.CE
        assert "error" in result.compile_log.lower()

    def test_solid_render_matches_itself(self, tmp_path):
        ex = ShaderExecutor(viewport=(64, 64))
        ref = tmp_path / "ref.png"
        out = tmp_path / "frame.png"

        ok, log, ms = ex.render(SOLID_RED, ref, viewport=(64, 64))
        assert ok, f"render failed: {log}"
        assert ms >= 0.0

        result = ex.run(SOLID_RED, reference_path=ref, viewport=(64, 64), out_path=out)
        assert result.verdict == Verdict.OK
        assert result.ssim >= 0.99

    def test_same_shader_twice_high_ssim(self, tmp_path):
        ex = ShaderExecutor(viewport=(64, 64))
        ref = tmp_path / "ref.png"
        out = tmp_path / "frame.png"

        ex.render(SOLID_RED, ref, viewport=(64, 64))
        # second invocation with same shader
        ok, _, _ = ex.render(SOLID_RED, out, viewport=(64, 64))
        assert ok
        s = ex.score_ssim(out, ref)
        assert s >= 0.99

    def test_modified_shader_lower_but_close(self, tmp_path):
        ex = ShaderExecutor(viewport=(128, 128))
        ref = tmp_path / "ref.png"
        out = tmp_path / "frame.png"

        ex.render(GRADIENT_SHADER, ref, viewport=(128, 128))
        ex.render(GRADIENT_MODIFIED, out, viewport=(128, 128))
        s = ex.score_ssim(out, ref)
        # Structurally identical gradient with small intensity shift —
        # SSIM should stay high but not perfect.
        assert s > 0.85, f"expected high SSIM, got {s}"
        assert s < 1.0

    def test_distinct_shaders_low_ssim(self, tmp_path):
        ex = ShaderExecutor(viewport=(64, 64))
        ref = tmp_path / "red.png"
        out = tmp_path / "blue.png"

        ex.render(SOLID_RED, ref, viewport=(64, 64))
        ex.render(SOLID_BLUE, out, viewport=(64, 64))
        s = ex.score_ssim(out, ref)
        # red vs blue: per-channel SSIM is dominated by mean shift,
        # so verdict should be WA
        v = verdict_from_ssim(s)
        assert v in (Verdict.WA, Verdict.VF), f"got verdict {v} with ssim {s}"


@requires_gpu
class TestShaderRunResultShape:
    def test_to_dict_has_expected_keys(self, tmp_path):
        ex = ShaderExecutor(viewport=(64, 64))
        ref = tmp_path / "ref.png"
        ex.render(SOLID_RED, ref, viewport=(64, 64))

        result = ex.run(SOLID_RED, reference_path=ref, viewport=(64, 64))
        d = result.to_dict()
        for k in ("verdict", "ssim", "lpips", "render_time_ms",
                  "compile_log", "frame_path", "error"):
            assert k in d
