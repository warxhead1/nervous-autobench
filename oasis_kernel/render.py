"""Oasis render — run the evolved flux, then light it like it's the 22nd century.

Two artifacts come out of one simulation:

* a **top-down PBR filmstrip** PNG (+ animated GIF) for the gallery thumbnail —
  Beer-Lambert water, Fresnel sky reflection, a sun glint that sparkles on the
  ripples, a foam shoreline, ray-marched soft shadows across the dunes, ambient
  occlusion in the basin, ACES tonemapping and a day-cycle colour grade; and
* a self-contained **Shadertoy GLSL** heightfield raymarcher (``.glsl`` sidecar)
  so the same oasis becomes a draggable, lit 3D card in the app + a Shadertoy export.

The simulation stays at the sim grid; only the *look* is upgraded.
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
from .shader import build_oasis_glsl

logger = logging.getLogger(__name__)

RES = 460          # render resolution (the sim grid is upsampled to this)
H_SCALE = 0.60     # terrain relief in world units
W_SCALE = 0.60     # water-depth -> world height
SHADOW_STEPS = 22  # ray-march steps for soft sun shadows
MAX_GIF_FRAMES = 14
ZOOM_LO, ZOOM_HI = 0.18, 0.82  # crop the sim window to the basin (pool is the hero)


# ──────────────────────────────────────────────────────────────────────────────
# Run the evolved harness in render mode
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# numpy lighting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _upsample(field, res, np, lo=0.0, hi=1.0):
    """Upsample a small grid to res×res, optionally cropping to a [lo,hi] window."""
    N = field.shape[0]
    ys = np.linspace(lo * (N - 1), hi * (N - 1), res)
    xs = np.linspace(lo * (N - 1), hi * (N - 1), res)
    gx, gy = np.meshgrid(xs, ys)
    return _bilinear(field, gx, gy, np)


def _bilinear(field, gx, gy, np):
    """Sample `field` (2D) at fractional pixel coords gx (col), gy (row)."""
    N = field.shape[0]
    x0 = np.clip(np.floor(gx).astype(np.int32), 0, N - 1)
    y0 = np.clip(np.floor(gy).astype(np.int32), 0, N - 1)
    x1 = np.clip(x0 + 1, 0, N - 1)
    y1 = np.clip(y0 + 1, 0, N - 1)
    fx = np.clip(gx - x0, 0, 1)
    fy = np.clip(gy - y0, 0, 1)
    a = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    b = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    return a * (1 - fy) + b * fy


def _fbm(res, np, seed=7):
    """Cheap multi-octave value noise for fine sand grain (the greebles)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((res, res), dtype=np.float64)
    amp, freq = 1.0, 6
    norm = 0.0
    for _ in range(5):
        g = rng.random((freq + 1, freq + 1))
        out += amp * _upsample(g, res, np)
        norm += amp
        amp *= 0.5
        freq *= 2
    out /= norm
    return out - out.mean()


def _box_blur(a, r, np):
    """Separable box blur (for ambient occlusion)."""
    if r < 1:
        return a
    k = 2 * r + 1
    c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="edge"), axis=0)
    a = (c[k:] - c[:-k]) / k
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="edge"), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def _normals(Z, np, strength=1.0):
    gy, gx = np.gradient(Z)
    nx, ny, nz = -gx * strength, -gy * strength, np.full_like(Z, 2.0 / RES)
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return nx / n, ny / n, nz / n


def _aces(x, np):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def _frame_pbr(H, w, day_phase, np, Image, grain):
    """Composite one fully-lit top-down frame. H, w are sim grids; day_phase∈[0,1]."""
    Hn = _upsample(H, RES, np, ZOOM_LO, ZOOM_HI)
    terr = Hn * H_SCALE + grain
    # smooth the diffusion stencil into an organic pool before lighting it
    depth = np.clip(_upsample(w, RES, np, ZOOM_LO, ZOOM_HI), 0, None)
    depth = _box_blur(_box_blur(depth, 6, np), 6, np)
    wet = depth > 1.5e-3
    surf = terr + depth * W_SCALE

    # ── sun: low + warm at dawn/dusk, high + white at noon ──
    alt = math.radians(11 + 58 * math.sin(math.pi * day_phase))
    az = math.radians(58 + 120 * day_phase)
    Lx, Ly, Lz = (math.cos(alt) * math.cos(az),
                  math.cos(alt) * math.sin(az), math.sin(alt))
    warm = np.array([1.0, 0.55, 0.26])
    noon = np.array([1.0, 0.97, 0.92])
    sun_col = warm + (noon - warm) * min(1.0, max(0.0, (math.sin(alt) - 0.12) / 0.5))
    sky_hi = np.array([0.42, 0.62, 0.92])
    sky_lo = np.array([0.78, 0.86, 0.98])

    # ── soft ray-marched sun shadows over the dune field ──
    iy, ix = np.mgrid[0:RES, 0:RES].astype(np.float64)
    px = -1 + 2 * ix / (RES - 1)
    py = -1 + 2 * iy / (RES - 1)
    shadow = np.ones((RES, RES))
    tan_alt = math.tan(alt)
    if tan_alt > 1e-3:
        for s in range(1, SHADOW_STEPS + 1):
            t = (s / SHADOW_STEPS) * 1.15
            sx = (px + Lx * t + 1) * 0.5 * (RES - 1)
            sy = (py + Ly * t + 1) * 0.5 * (RES - 1)
            zt = _bilinear(terr, sx, sy, np)
            ray = terr + tan_alt * t
            clearance = (ray - zt) / max(t, 1e-3)
            shadow = np.minimum(shadow, np.clip(8.0 * clearance, 0, 1))
    shadow = 0.25 + 0.75 * shadow

    # ── ambient occlusion: concavities (basin) get darker ──
    ao = np.clip(0.6 + 3.0 * (terr - _box_blur(terr, 12, np)), 0.25, 1.0)

    # ── terrain normals + sand albedo ──
    tnx, tny, tnz = _normals(terr, np, strength=2.2)
    diff = np.clip(tnx * Lx + tny * Ly + tnz * Lz, 0, 1)
    sand_lo, sand_hi = np.array([0.45, 0.31, 0.17]), np.array([0.95, 0.84, 0.62])
    albedo = sand_lo + (sand_hi - sand_lo) * Hn[..., None]
    # wet sand ring darkens + saturates near the shoreline
    shore = np.clip(1 - depth / 0.02, 0, 1) * wet
    albedo *= (1 - 0.45 * shore)[..., None]
    sky_amb = (sky_lo * (0.5 + 0.5 * tnz[..., None]))
    sand = albedo * (0.30 * sky_amb + sun_col * (diff * shadow)[..., None]) * ao[..., None]

    # ── water normals (surface + animated ripples) ──
    snx, sny, snz = _normals(surf, np, strength=2.2)
    ph = day_phase * 2 * math.pi
    rip = (np.sin(34 * px + 22 * py + 2.0 * ph) +
           0.6 * np.sin(26 * px - 31 * py + 1.3 * ph) +
           0.4 * np.sin(51 * py + 0.7 * ph))
    rgy, rgx = np.gradient(rip)
    rk = 0.06
    wnx, wny, wnz = snx + rk * rgx, sny + rk * rgy, snz
    wn = np.sqrt(wnx * wnx + wny * wny + wnz * wnz) + 1e-9
    wnx, wny, wnz = wnx / wn, wny / wn, wnz / wn

    Vx, Vy, Vz = 0.0, 0.0, 1.0  # top-down view
    cos_v = np.clip(wnx * Vx + wny * Vy + wnz * Vz, 0, 1)
    F = 0.02 + 0.98 * (1 - cos_v) ** 5

    # Beer-Lambert: shallow=cyan, deep=teal; bottom = attenuated sand
    absb = np.array([3.4, 1.4, 0.9])
    trans = np.exp(-absb * (depth * 6.0)[..., None])
    deep = np.array([0.02, 0.18, 0.24])
    bottom = sand * trans + deep * (1 - trans)
    caustic = (1 + 0.5 * np.clip(rip, 0, None) ** 2 * np.clip(1 - depth / 0.05, 0, 1))
    bottom *= caustic[..., None]

    refl = sky_hi  # top-down: reflect the high sky
    # sun specular glint (sharp), sparkling on ripple normals
    rdx = 2 * (wnx * Lx + wny * Ly + wnz * Lz) * wnx - Lx
    rdy = 2 * (wnx * Lx + wny * Ly + wnz * Lz) * wny - Ly
    rdz = 2 * (wnx * Lx + wny * Ly + wnz * Lz) * wnz - Lz
    spec = np.clip(rdx * Vx + rdy * Vy + rdz * Vz, 0, 1) ** 120
    water = bottom * (1 - F[..., None]) + refl * F[..., None]
    water += sun_col * (spec * shadow)[..., None] * 1.4

    # foam at the very edge of the pool
    foam = np.clip(1 - depth / 0.012, 0, 1) * wet * (0.5 + 0.5 * np.clip(rip, 0, 1))
    water += np.array([0.92, 0.96, 1.0]) * (foam * 0.6)[..., None]

    # ── composite (soft shoreline) + sky-tinted day grade ──
    blend = np.clip(depth / 0.004, 0, 1)[..., None] * wet[..., None]
    col = sand * (1 - blend) + water * blend

    grade = np.array([1.0, 1.0, 1.0]) * (0.85 + 0.15 * math.sin(math.pi * day_phase))
    grade = grade * (warm * 0.0 + 1.0) + (warm - 1.0) * (0.18 * (1 - math.sin(math.pi * day_phase)))
    col = col * grade

    # vignette + ACES + gamma
    vig = 1 - 0.28 * (px * px + py * py)
    col = col * vig[..., None]
    col = _aces(col, np)
    col = np.power(col, 1 / 2.2)
    im = Image.fromarray(np.clip(col * 255, 0, 255).astype(np.uint8))
    return im


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────

def render_oasis(
    code: str,
    instance: OasisInstance,
    executor: SandboxedExecutor,
    out_path: Path,
) -> bool:
    """Render the evolved flow law to a PBR filmstrip PNG at out_path.

    Also writes:
      * ``out_path.with_suffix('.gif')`` — animated day cycle
      * ``out_path.with_suffix('.glsl')`` — Shadertoy heightfield raymarcher
    """
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

    grain = _fbm(RES, np, seed=int(instance.dune_freq * 13) & 1023) * 0.022
    n_render = min(len(frames), MAX_GIF_FRAMES)
    pick_idx = [int(round(i * (len(frames) - 1) / max(1, n_render - 1)))
                for i in range(n_render)]
    imgs = [_frame_pbr(H, frames[k], k / max(1, len(frames) - 1), np, Image, grain)
            for k in pick_idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imgs[0].save(out_path.with_suffix(".gif"), save_all=True,
                     append_images=imgs[1:], duration=90, loop=0)
    except Exception:
        pass

    pick = imgs[:: max(1, len(imgs) // 8)][:8]
    strip = Image.new("RGB", (sum(i.width for i in pick), pick[0].height))
    x = 0
    for im in pick:
        strip.paste(im, (x, 0)); x += im.width
    strip.save(out_path)

    # ── Shadertoy heightfield raymarcher (live card + export) ──
    try:
        final = frames[-1]
        glsl = build_oasis_glsl(H, final, instance, np)
        out_path.with_suffix(".glsl").write_text(glsl)
    except Exception as e:
        logger.debug("oasis glsl build skipped: %s", e)

    return True
