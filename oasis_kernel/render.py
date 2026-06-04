"""Oasis render — run the evolved flux in render mode, draw a day-cycle strip.

Produces a horizontal filmstrip PNG (the gallery artifact) plus an animated GIF
beside it, showing the spring-fed pool filling and breathing over the run.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path

from ..engines.sandbox import SandboxedExecutor
from .instance import OasisInstance
from .oracle import build_candidate_source

logger = logging.getLogger(__name__)


def _compile_run(cpp_source: str, stdin_data: str) -> str | None:
    """Compile + run the harness directly (render needs more time/output than the
    eval sandbox allows). Returns stdout, or None on failure."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "oasis.cpp")
        binp = os.path.join(td, "oasis")
        Path(src).write_text(cpp_source)
        try:
            c = subprocess.run(["g++", "-O2", "-o", binp, src],
                               capture_output=True, text=True, timeout=90)
            if c.returncode != 0:
                logger.debug("oasis render compile failed: %s", c.stderr[:400])
                return None
            r = subprocess.run([binp], input=stdin_data,
                               capture_output=True, text=True, timeout=90)
            return r.stdout
        except Exception as e:
            logger.debug("oasis render subprocess failed: %s", e)
            return None


def _hillshade(H, np, az=315, alt=45):
    gy, gx = np.gradient(H * 12.0)
    azr, altr = math.radians(az), math.radians(alt)
    nx, ny, nz = -gx, -gy, np.ones_like(H)
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    sh = np.clip((nx * math.cos(altr) * math.cos(azr)
                  + ny * math.cos(altr) * math.sin(azr)
                  + nz * math.sin(altr)) / n, 0, 1)
    return 0.45 + 0.55 * sh


def _frame(H, w, np, Image, up=6):
    N = H.shape[0]
    sand_lo, sand_hi = np.array([0.42, 0.32, 0.20]), np.array([0.93, 0.84, 0.62])
    base = (sand_lo + (sand_hi - sand_lo) * H[..., None]) * _hillshade(H, np)[..., None]
    water = np.array([0.05, 0.30, 0.50])
    depth = np.clip(w / 0.10, 0, 1)[..., None]
    img = base * (1 - depth) + water * depth
    img += (0.16 * depth[..., 0] * (w > 1e-3))[..., None]
    im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    return im.resize((N * up, N * up), Image.BILINEAR)


def render_oasis(
    code: str,
    instance: OasisInstance,
    executor: SandboxedExecutor,
    out_path: Path,
) -> bool:
    """Render the evolved flow law to a filmstrip PNG at out_path (+ a .gif)."""
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:  # pragma: no cover
        logger.debug("oasis render needs numpy+PIL: %s", e)
        return False

    cpp_source, stdin_data = build_candidate_source(code, instance, render=True)
    stdout = _compile_run(cpp_source, stdin_data)
    if not stdout:
        return False

    lines = stdout.strip().split("\n")
    hdr_idx = next((i for i, ln in enumerate(lines) if ln.startswith("H ")), -1)
    if hdr_idx < 0 or hdr_idx + 1 >= len(lines):
        return False
    try:
        _, n_s, nf_s = lines[hdr_idx].split()
        N, nf = int(n_s), int(nf_s)
        H = np.array(lines[hdr_idx + 1].split(), dtype=np.float64).reshape(N, N)
        frames = [np.array(lines[hdr_idx + 2 + i].split(), dtype=np.float64).reshape(N, N)
                  for i in range(nf) if hdr_idx + 2 + i < len(lines)]
    except Exception as e:
        logger.debug("oasis render parse failed: %s", e)
        return False
    if not frames:
        return False

    imgs = [_frame(H, w, np, Image) for w in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imgs[0].save(out_path.with_suffix(".gif"), save_all=True,
                     append_images=imgs[1:], duration=80, loop=0)
    except Exception:
        pass
    pick = imgs[:: max(1, len(imgs) // 8)][:8]
    strip = Image.new("RGB", (sum(i.width for i in pick), pick[0].height))
    x = 0
    for im in pick:
        strip.paste(im, (x, 0)); x += im.width
    strip.save(out_path)
    return True
