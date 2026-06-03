"""CPU occupancy -> PNG renderer for svdag_beauty (Tier 1).

Turns an evolved compute_density into a visible volcanic-terrain image with NO
GPU and no tengine: sample the candidate over a world grid into an occupancy
field, then isometric-project the exposed voxels with a volcanic palette,
ambient-occlusion shading, and a z-buffer. This is what makes "fitness went up"
become "look at the volcano", feeds the assessment agent a real image, and
emits a phone artifact — all reusing the same sandbox the oracle uses.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from .oracle import build_candidate_source, _occupancy_from_stdout, WORLD_SIZE, WORLD_HEIGHT

logger = logging.getLogger(__name__)


def _render_stdin(res, ny, seed):
    return json.dumps({"nx": res, "nz": res, "ny": ny or res,
                       "world": WORLD_SIZE, "height": WORLD_HEIGHT, "seed": seed})


def sample_occupancy_native(code, res=128, ny=None, seed=1337.0, run_timeout=60.0):
    """Compile compute_density with g++ directly and sample it (no sandbox).

    Rendering targets chosen/best code, not arbitrary untrusted candidates, so a
    native compile is appropriate — and it dodges the gvisor overhead that makes
    high-res sampling time out. Returns occupancy bool array or None.
    """
    import subprocess, tempfile, os, shutil
    gpp = shutil.which("g++") or shutil.which("c++")
    if not gpp:
        return None
    d = tempfile.mkdtemp(prefix="svdag_render_")
    try:
        src = os.path.join(d, "s.cpp"); binp = os.path.join(d, "s")
        with open(src, "w") as f:
            f.write(build_candidate_source(code))
        c = subprocess.run([gpp, "-O2", "-o", binp, src], capture_output=True, text=True, timeout=60)
        if c.returncode != 0:
            logger.warning("native render compile failed: %s", c.stderr[:300])
            return None
        r = subprocess.run([binp], input=_render_stdin(res, ny, seed),
                           capture_output=True, text=True, timeout=run_timeout)
        if r.returncode != 0:
            return None
        return _occupancy_from_stdout(r.stdout)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("native render failed: %s", e)
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def sample_occupancy(code, executor, res=128, ny=None, seed=1337.0, run_timeout=30.0):
    """Compile + sample compute_density over a res^3 grid -> bool array (nx,nz,ny).

    Prefers a fast native compile; falls back to the sandbox executor if g++ is
    unavailable. Pass executor=None to force native.
    """
    O = sample_occupancy_native(code, res=res, ny=ny, seed=seed, run_timeout=max(run_timeout, 60.0))
    if O is not None or executor is None:
        return O
    from . import compile_and_run
    from ..core import Verdict
    out, verdict, _ = compile_and_run(
        build_candidate_source(code), "cpp",
        constraints={"max_time_seconds": run_timeout, "max_memory_mb": executor.max_memory_mb},
        stdin=_render_stdin(res, ny, seed), executor=executor,
    )
    if verdict != Verdict.OK:
        logger.warning("render sample non-OK verdict: %s", verdict)
        return None
    return _occupancy_from_stdout(out)


# volcanic altitude palette: basalt -> rust -> ash/snow cap
_PALETTE = np.array([
    [0.00, 38, 30, 30],
    [0.28, 74, 48, 42],
    [0.50, 120, 60, 44],
    [0.66, 158, 92, 60],
    [0.80, 150, 140, 132],
    [1.00, 232, 226, 220],
], dtype=float)


def _altitude_color(t):
    stops = _PALETTE[:, 0]
    out = np.empty((t.size, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.interp(t, stops, _PALETTE[:, c + 1])
    return out


def render_occupancy(O, img_w=960, img_h=760):
    """Isometric painter render of exposed voxels. Returns an (H,W,3) uint8 array."""
    NX, NZ, NY = O.shape
    empty = ~O

    # exposed = solid voxel with >=1 empty 6-neighbour (outside grid counts as empty)
    pad = np.pad(empty, 1, mode="constant", constant_values=True)
    nbr_empty = (
        pad[2:, 1:-1, 1:-1].astype(np.int16) + pad[:-2, 1:-1, 1:-1] +
        pad[1:-1, 2:, 1:-1] + pad[1:-1, :-2, 1:-1] +
        pad[1:-1, 1:-1, 2:] + pad[1:-1, 1:-1, :-2]
    )
    exposed = O & (nbr_empty > 0)
    top_face = O & np.pad(empty, ((0, 0), (0, 0), (0, 1)), constant_values=True)[:, :, 1:]

    ix, iz, iy = np.nonzero(exposed)
    if ix.size == 0:
        return np.full((img_h, img_w, 3), 18, np.uint8)

    # ambient occlusion: more open -> brighter; top faces get a sky-light boost
    ao = nbr_empty[ix, iz, iy] / 6.0
    shade = 0.40 + 0.50 * ao + 0.12 * top_face[ix, iz, iy]
    shade = np.clip(shade, 0.25, 1.15)

    # normalize altitude to the actually-occupied height band so the palette spans
    iy_lo, iy_hi = int(iy.min()), int(iy.max())
    t = (iy - iy_lo) / max(iy_hi - iy_lo, 1)
    rgb = _altitude_color(t) * shade[:, None]
    # subtle per-voxel grain so flat faces aren't dead-flat
    grain = ((np.sin(ix * 12.9898 + iz * 78.233 + iy * 37.719) * 43758.5) % 1.0 - 0.5) * 14.0
    rgb = np.clip(rgb + grain[:, None], 0, 255).astype(np.uint8)

    # isometric projection, camera at +x,+y,+z corner, with vertical exaggeration
    # so relief / overhangs read instead of looking top-down-flat.
    VEXAG = 1.25
    su = (ix - iz).astype(float)
    # screen-up (larger sv) = higher elevation AND farther back; py flips sv so the
    # summit and far ridges sit up-top, near-low ground at the front-bottom.
    sv = iy.astype(float) * VEXAG + (ix + iz) * 0.30
    su -= su.min(); sv -= sv.min()
    scale = min((img_w - 48) / (su.max() + 1), (img_h - 48) / (sv.max() + 1))
    px = (24 + su * scale).astype(np.int32)
    py = (img_h - 24 - sv * scale).astype(np.int32)
    key = ix + iz + iy                       # nearer to camera = larger
    order = np.argsort(key, kind="stable")   # draw far first; near overwrites

    # sky gradient background
    img = np.empty((img_h, img_w, 3), np.uint8)
    sky = np.linspace(34, 12, img_h)[:, None]
    img[:] = np.stack([sky + 14, sky + 10, sky + 22], axis=-1).astype(np.uint8).repeat(img_w, axis=1).reshape(img_h, img_w, 3)

    block = max(2, int(np.ceil(scale)) + 1)
    block = min(block, 7)
    for dy in range(block):
        for dx in range(block):
            yy = np.clip(py[order] + dy, 0, img_h - 1)
            xx = np.clip(px[order] + dx, 0, img_w - 1)
            img[yy, xx] = rgb[order]
    return img


def render_density_to_png(code, executor, out_path, *, res=128, seed=1337.0, run_timeout=30.0):
    """Full path: sample candidate -> render -> write PNG. Returns True on success."""
    O = sample_occupancy(code, executor, res=res, seed=seed, run_timeout=run_timeout)
    if O is None:
        return False
    frac = float(O.sum()) / O.size
    if frac <= 0.001 or frac >= 0.999:
        return False
    img = render_occupancy(O)
    try:
        from PIL import Image
        Image.fromarray(img, "RGB").save(str(out_path))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("PNG write failed: %s", e)
        return False
