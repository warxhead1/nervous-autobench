"""Oracle / LLM-facing helpers for the svdag_beauty kernel.

Evolves a single ``compute_density(world_x, world_z, cell_y, seed)`` function —
the build-time geometry of an SVDAG volcanic terrain. The candidate is compiled
in the sandbox, sampled over a fixed 48^3 world grid into a packed occupancy
bitset (solid where density > 0), and scored by a *generative membership*
oracle: a weighted geometric mean of structural signatures that distinguish
"rocky / porous / volcanic" terrain (solid-fraction band, surface roughness,
steep-slope presence, spectral slope, porosity/overhangs, vertical relief, and
connectivity to punish floating debris).

The oracle measures class membership, not pixel distance to one reference — so
its argmax is the whole family of volcanic terrains, not a single realization.
A real GPU render via tengine (the tengine.shadergen.eval contract) is the
*confirmation* step for the best candidate, never per-candidate fitness.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re

import numpy as np

from .instance import VolcanoInstance

logger = logging.getLogger(__name__)


# Fixed world scale — matches the absolute coordinate convention tengine's
# compute_density works in (no world/height params, like the real shader).
WORLD_SIZE = 256.0
WORLD_HEIGHT = 128.0
GRID_N = 48          # horizontal cells (x, z)
GRID_NY = 48         # vertical cells (y)


# ---------------------------------------------------------------------------
# C skeleton — fixed harness + shared noise helpers; LLM writes compute_density.
# ---------------------------------------------------------------------------

SVDAG_SKELETON = r"""// Auto-generated SVDAG density skeleton. LLM writes compute_density() only.
// Samples the candidate over a world grid and emits a packed occupancy bitset.
#include <bits/stdc++.h>
using namespace std;

// ---- shared noise helpers (available to compute_density) -------------------
static inline float nb_fract(float x){ return x - floorf(x); }
static inline float nb_hash2(float x, float z){
    float h = sinf(x*127.1f + z*311.7f) * 43758.5453f; return nb_fract(h);
}
static inline float nb_vnoise2(float x, float z){
    float xi=floorf(x), zi=floorf(z), xf=x-xi, zf=z-zi;
    float a=nb_hash2(xi,zi), b=nb_hash2(xi+1.f,zi);
    float c=nb_hash2(xi+1.f,zi+1.f), d=nb_hash2(xi,zi+1.f);
    float u=xf*xf*(3.f-2.f*xf), v=zf*zf*(3.f-2.f*zf);
    return a + (b-a)*u + (d-a)*v + (a-b-d+c)*u*v;   // [0,1]
}
static inline float nb_fbm2(float x, float z){
    float s=0.f, a=0.5f, f=1.f;
    for(int i=0;i<5;i++){ s+=a*nb_vnoise2(x*f,z*f); f*=2.f; a*=0.5f; }
    return s;   // ~[0,1]
}
static inline float nb_hash3(float x,float y,float z){
    float h = sinf(x*127.1f + y*311.7f + z*74.7f) * 43758.5453f; return nb_fract(h);
}
static inline float nb_vnoise3(float x,float y,float z){
    float xi=floorf(x),yi=floorf(y),zi=floorf(z),xf=x-xi,yf=y-yi,zf=z-zi;
    float u=xf*xf*(3.f-2.f*xf),v=yf*yf*(3.f-2.f*yf),w=zf*zf*(3.f-2.f*zf);
    float c000=nb_hash3(xi,yi,zi),       c100=nb_hash3(xi+1.f,yi,zi);
    float c010=nb_hash3(xi,yi+1.f,zi),   c110=nb_hash3(xi+1.f,yi+1.f,zi);
    float c001=nb_hash3(xi,yi,zi+1.f),   c101=nb_hash3(xi+1.f,yi,zi+1.f);
    float c011=nb_hash3(xi,yi+1.f,zi+1.f),c111=nb_hash3(xi+1.f,yi+1.f,zi+1.f);
    float x00=c000+(c100-c000)*u, x10=c010+(c110-c010)*u;
    float x01=c001+(c101-c001)*u, x11=c011+(c111-c011)*u;
    float y0=x00+(x10-x00)*v, y1=x01+(x11-x01)*v;
    return y0+(y1-y0)*w;   // [0,1]
}
static inline float nb_fbm3(float x,float y,float z){
    float s=0.f,a=0.5f,f=1.f;
    for(int i=0;i<4;i++){ s+=a*nb_vnoise3(x*f,y*f,z*f); f*=2.f; a*=0.5f; }
    return s;   // ~[0,1]
}

// === LLM-EVOLVED FUNCTION — do not modify the signature ===
extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed);
// ===========================================================

static double get_num(const string& s, const char* key, double dflt){
    string k = string("\"") + key + "\"";
    size_t p = s.find(k);
    if(p==string::npos) return dflt;
    p = s.find(':', p);
    if(p==string::npos) return dflt;
    p++;
    return strtod(s.c_str()+p, nullptr);
}

int main(){
    string s((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());
    int nx = (int)get_num(s,"nx",48);
    int nz = (int)get_num(s,"nz",48);
    int ny = (int)get_num(s,"ny",48);
    float world  = (float)get_num(s,"world",256.0);
    float height = (float)get_num(s,"height",128.0);
    float seed   = (float)get_num(s,"seed",1.0);

    long total = (long)nx*nz*ny;
    long nbytes = (total + 7) / 8;
    vector<unsigned char> bits(nbytes, 0);
    long solid = 0;
    long idx = 0;
    for(int ix=0; ix<nx; ix++){
        float wx = (ix + 0.5f) / nx * world;
        for(int iz=0; iz<nz; iz++){
            float wz = (iz + 0.5f) / nz * world;
            for(int iy=0; iy<ny; iy++){
                float wy = (iy + 0.5f) / ny * height;
                float d = compute_density(wx, wz, wy, seed);
                if(isfinite(d) && d > 0.0f){
                    bits[idx>>3] |= (unsigned char)(1u << (7 - (idx & 7)));
                    solid++;
                }
                idx++;
            }
        }
    }

    // base64 encode
    static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    string out; out.reserve(((nbytes+2)/3)*4);
    for(long i=0;i<nbytes;i+=3){
        unsigned v = bits[i] << 16;
        if(i+1<nbytes) v |= bits[i+1] << 8;
        if(i+2<nbytes) v |= bits[i+2];
        out += B64[(v>>18)&63];
        out += B64[(v>>12)&63];
        out += (i+1<nbytes) ? B64[(v>>6)&63] : '=';
        out += (i+2<nbytes) ? B64[v&63] : '=';
    }
    printf("{\"nx\":%d,\"nz\":%d,\"ny\":%d,\"solid\":%ld,\"occ\":\"%s\"}\n",
           nx, nz, ny, solid, out.c_str());
    return 0;
}
"""


COMPUTE_DENSITY_SIGNATURE = '''\
extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed);

// Goal: return the DENSITY of an SVDAG voxel at world cell (world_x, world_z, cell_y).
//   > 0  => SOLID rock (the voxel is filled)
//   <= 0 => EMPTY space (air / void / cave)
// The render fills every cell where compute_density > 0, so the shape of the
// d=0 isosurface IS the terrain. world_x, world_z span [0, 256]; cell_y spans
// [0, 128] (cell_y is the vertical/height axis). seed varies the realization.
//
// Pre-supplied helpers (no need to redefine — just call them):
//   nb_fbm2(x, z)        -> ~[0,1]  multi-octave 2D value noise (heightfields)
//   nb_fbm3(x, y, z)     -> ~[0,1]  multi-octave 3D value noise (caves/overhangs)
//   nb_vnoise2 / nb_vnoise3 / nb_hash2 / nb_hash3 — single-octave primitives
//
// You may define your own static helper functions above compute_density.
//
// What "rocky / porous / volcanic" terrain needs (the oracle rewards these):
//   - a coherent macro structure (a cone, ridge, caldera) — NOT uniform noise
//   - fractal surface roughness across scales (multi-octave fbm displacement)
//   - STEEP slopes (volcanic cones/cliffs), not gentle rolling hills
//   - POROSITY: overhangs and caves — the SVDAG's superpower over a heightmap.
//     A pure heightmap (surface_height - cell_y) has ZERO porosity. Carve voids
//     by subtracting 3D noise pockets: e.g. d -= cave_strength*step(nb_fbm3(...))
//   - the solid set must stay CONNECTED — avoid clouds of floating debris.
'''


# ---------------------------------------------------------------------------
# Seed programs — decent volcanic baselines (evolution starts here)
# ---------------------------------------------------------------------------

SEED_PROGRAMS: list[tuple[str, str]] = [
    ("fbm_cone", '''\
extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed) {
    float W = 256.0f, H = 128.0f;
    float nx = world_x / W, nz = world_z / W;
    float so = seed * 0.137f;
    float dx = nx - 0.5f, dz = nz - 0.5f;
    float r = sqrtf(dx*dx + dz*dz);
    float cone = 1.0f - r * 2.0f; if (cone < 0.0f) cone = 0.0f;
    cone = cone * cone;
    float ridges = nb_fbm2(nx*6.0f + so, nz*6.0f + so);
    float h = 0.22f + 0.55f*cone + 0.20f*(ridges - 0.5f);
    return h * H - cell_y;   // heightmap: solid below the surface
}'''),
    ("carved_cone", '''\
extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed) {
    float W = 256.0f, H = 128.0f;
    float nx = world_x / W, nz = world_z / W;
    float so = seed * 0.191f;
    float dx = nx - 0.5f, dz = nz - 0.5f;
    float r = sqrtf(dx*dx + dz*dz);
    float cone = 1.0f - r * 1.9f; if (cone < 0.0f) cone = 0.0f;
    float ridges = nb_fbm2(nx*7.0f + so, nz*7.0f + so);
    float h = 0.20f + 0.58f*cone*cone + 0.24f*(ridges - 0.5f);
    float surfY = h * H;
    float d = surfY - cell_y;
    // meandering lava tube: carve a void band that KEEPS a solid roof above and
    // floor below -> genuine enclosed overhangs/caves (porosity).
    float tube = nb_fbm2(nx*4.0f + so, nz*4.0f + so);     // [0,1]
    float tubeY = surfY * (0.35f + 0.30f*tube);
    float chan = nb_fbm2(nx*3.0f + so*1.7f, nz*3.0f + so*1.7f);   // pillar gate
    if (cell_y < surfY - 10.0f && fabsf(cell_y - tubeY) < 6.0f && chan > 0.46f)
        d = -1.0f;   // carve only along the channel -> pillars keep the roof attached
    return d;
}'''),
    ("ridged_plateau", '''\
extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed) {
    float W = 256.0f, H = 128.0f;
    float nx = world_x / W, nz = world_z / W, ny = cell_y / H;
    float so = seed * 0.077f;
    float base = nb_fbm2(nx*5.0f + so, nz*5.0f + so);
    float ridge = fabsf(nb_vnoise2(nx*11.0f + so, nz*11.0f + so) - 0.5f);
    float h = 0.16f + 0.26f*base + 0.22f*(1.0f - 2.0f*ridge);
    float d = h * H - cell_y;
    float cave = nb_fbm3(nx*8.0f, ny*7.0f + so, nz*8.0f);
    d -= 22.0f * fmaxf(cave - 0.66f, 0.0f);
    return d;
}'''),
]

# Negative controls — used only by the `baselines` derisking command to prove the
# oracle discriminates (these should score FAR below the seeds above).
CONTROL_PROGRAMS: list[tuple[str, str]] = [
    ("flat_slab", '''\
extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed) {
    return 64.0f - cell_y;   // featureless slab: no roughness, no porosity, no relief
}'''),
    ("noise_mush", '''\
extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed) {
    float n = nb_fbm3(world_x*0.08f, cell_y*0.08f, world_z*0.08f);
    return n - 0.5f;   // isotropic 3D noise: porous but no macro form, floaters everywhere
}'''),
]


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    return SEED_PROGRAMS


# ---------------------------------------------------------------------------
# Sandbox build + scoring
# ---------------------------------------------------------------------------

def build_candidate_source(density_code: str) -> str:
    return SVDAG_SKELETON + "\n" + density_code + "\n"


def _instance_stdin(instance: VolcanoInstance) -> str:
    return json.dumps({
        "nx": GRID_N, "nz": GRID_N, "ny": GRID_NY,
        "world": WORLD_SIZE, "height": WORLD_HEIGHT,
        "seed": instance.seed,
    })


# scalar sub-score helpers -> all map a raw statistic into [0,1]
def _band(x: float, lo: float, hi: float, soft: float) -> float:
    if x < lo:
        return math.exp(-((lo - x) / soft) ** 2)
    if x > hi:
        return math.exp(-((x - hi) / soft) ** 2)
    return 1.0


def _ramp(x: float, half: float) -> float:
    """0 at x=0, 0.5 at x=half, -> 1 as x grows. Rewards 'presence' of a trait."""
    x = max(x, 0.0)
    return x / (x + half) if half > 0 else 1.0


def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _occupancy_from_stdout(stdout: str) -> "np.ndarray | None":
    try:
        out = json.loads(stdout.strip())
        nx, nz, ny = int(out["nx"]), int(out["nz"]), int(out["ny"])
        raw = base64.b64decode(out["occ"])
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        need = nx * nz * ny
        if bits.size < need:
            return None
        return bits[:need].reshape(nx, nz, ny).astype(bool)
    except Exception as exc:  # noqa: BLE001
        logger.debug("occupancy decode failed: %s", exc)
        return None


def _largest_component_fraction(O: "np.ndarray") -> float:
    """Fraction of solid voxels in the largest 6-connected component.

    Prefers scipy.ndimage.label; falls back to a bounded numpy label-propagation
    that estimates the largest component (good enough as a fitness signal).
    """
    solid = int(O.sum())
    if solid == 0:
        return 0.0
    try:
        from scipy.ndimage import label  # type: ignore
        lab, n = label(O)
        if n == 0:
            return 0.0
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        return float(counts.max()) / solid
    except Exception:  # noqa: BLE001
        # numpy fallback: iterative min-propagation of unique ids over solids.
        ids = np.where(O, np.arange(O.size).reshape(O.shape) + 1, 0)
        for _ in range(64):
            prev = ids
            m = ids.copy()
            for ax in (0, 1, 2):
                a = np.roll(ids, 1, axis=ax); a[tuple(slice(0, 1) if d == ax else slice(None) for d in range(3))] = 0
                b = np.roll(ids, -1, axis=ax); b[tuple(slice(-1, None) if d == ax else slice(None) for d in range(3))] = 0
                for nb in (a, b):
                    take = O & (nb > 0) & ((m == 0) | (nb < m))
                    m = np.where(take, nb, m)
            ids = np.where(O, np.maximum(np.where(m > 0, m, ids), 1), 0)
            if np.array_equal(ids, prev):
                break
        labels = ids[O]
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        return float(counts.max()) / solid


def _spectral_beta(Hn: "np.ndarray") -> float:
    """Radial power-spectrum slope of the heightfield (fractal dimension proxy)."""
    h = Hn - Hn.mean()
    if not np.any(h):
        return 0.0
    F = np.fft.fftshift(np.fft.fft2(h))
    P = (F.real ** 2 + F.imag ** 2)
    n = Hn.shape[0]
    cy, cx = n // 2, n // 2
    yy, xx = np.indices((n, n))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    tbin = np.bincount(rr.ravel(), P.ravel())
    nr = np.bincount(rr.ravel())
    radial = tbin / np.maximum(nr, 1)
    k = np.arange(2, min(n // 2, len(radial)))
    pk = radial[k]
    good = pk > 0
    if good.sum() < 4:
        return 0.0
    slope = np.polyfit(np.log(k[good]), np.log(pk[good]), 1)[0]
    return float(-slope)   # beta = -slope of P(k)


def score_occupancy(O: "np.ndarray", instance: VolcanoInstance) -> tuple[float, dict]:
    """Volcanic-realism membership oracle. Returns (fitness, diagnostics)."""
    t = instance.targets
    total = O.size
    solid_frac = float(O.sum()) / total

    NY = O.shape[2]
    ys = np.arange(NY)
    masked = np.where(O, ys[None, None, :], -1)
    Htop = masked.max(axis=2)                      # topmost solid index, -1 if empty
    has_ground = Htop >= 0
    Hn = np.where(has_ground, Htop / (NY - 1), 0.0)

    relief = float(Hn[has_ground].std()) if has_ground.any() else 0.0
    gx = np.abs(np.diff(Hn, axis=0))
    gz = np.abs(np.diff(Hn, axis=1))
    rough = float((gx.mean() + gz.mean()) * 0.5)
    slopes = np.concatenate([gx.ravel(), gz.ravel()])
    steep_frac = float((slopes > t["steep_thresh"]).mean()) if slopes.size else 0.0

    # porosity: enclosed air (solid strictly above AND strictly below in column)
    below = np.maximum.accumulate(O, axis=2)
    above = np.maximum.accumulate(O[:, :, ::-1], axis=2)[:, :, ::-1]
    strict_below = np.zeros_like(O); strict_below[:, :, 1:] = below[:, :, :-1]
    strict_above = np.zeros_like(O); strict_above[:, :, :-1] = above[:, :, 1:]
    enclosed = (~O) & strict_below & strict_above
    pore_frac = float(enclosed.sum()) / total

    beta = _spectral_beta(Hn)
    lcc_frac = _largest_component_fraction(O)

    s_solid = _band(solid_frac, t["solid_lo"], t["solid_hi"], 0.08)
    s_rough = _ramp(rough, t["rough_half"])
    s_steep = _ramp(steep_frac, t["steep_half"])
    s_beta = _gauss(beta, t["beta_mu"], t["beta_sigma"])
    s_pore = _band(pore_frac, t["pore_lo"], t["pore_hi"], 0.02)
    s_relief = _ramp(relief, t["relief_half"])
    s_conn = max(0.0, min(1.0, (lcc_frac - 0.40) / 0.50))

    parts = {
        "solid": (s_solid, 1.0), "rough": (s_rough, 1.0), "steep": (s_steep, 0.8),
        "beta": (s_beta, 0.9), "pore": (s_pore, 1.2), "relief": (s_relief, 0.8),
        "conn": (s_conn, 1.0),
    }
    wsum = sum(w for _, w in parts.values())
    logsum = sum(w * math.log(max(s, 1e-3)) for s, w in parts.values())
    fitness = math.exp(logsum / wsum)

    diag = {
        "solid_frac": round(solid_frac, 4), "rough": round(rough, 4),
        "steep_frac": round(steep_frac, 4), "beta": round(beta, 3),
        "pore_frac": round(pore_frac, 4), "relief": round(relief, 4),
        "lcc_frac": round(lcc_frac, 4),
        "subscores": {k: round(s, 3) for k, (s, _) in parts.items()},
    }
    return fitness, diag


def evaluate_on_instance(
    density_code: str,
    instance: VolcanoInstance,
    executor: "SandboxedExecutor",
    run_timeout: float = 15.0,
) -> float | None:
    """Compile + sample compute_density, score volcanic realism. (0,1] or None."""
    from . import compile_and_run
    from ..core import Verdict

    source = build_candidate_source(density_code)
    stdout, verdict, _latency = compile_and_run(
        source,
        "cpp",
        constraints={"max_time_seconds": run_timeout, "max_memory_mb": executor.max_memory_mb},
        stdin=_instance_stdin(instance),
        executor=executor,
    )
    if verdict != Verdict.OK:
        logger.debug("svdag_beauty candidate non-OK verdict %s on %s", verdict, instance.name)
        return None
    O = _occupancy_from_stdout(stdout)
    if O is None:
        return None
    solid_frac = float(O.sum()) / O.size
    if solid_frac <= 0.001 or solid_frac >= 0.999:
        # degenerate (all air / all rock) — unrenderable, floor it.
        instance._last_diag = {"solid_frac": round(solid_frac, 4), "degenerate": True}
        return 0.01
    fitness, diag = score_occupancy(O, instance)
    instance._last_diag = diag
    return fitness


# ---------------------------------------------------------------------------
# Island personas + LLM prompt
# ---------------------------------------------------------------------------

SVDAG_ISLAND_PERSONAS = [
    ("volcanic geomorphologist",
     "Think in landforms: stratovolcano cones, calderas, lava deltas, spatter ramparts. "
     "Build a strong macro silhouette first (radial cone / ridge), then add fractal skin."),
    ("cave & overhang carver",
     "Obsess over POROSITY. Start from a solid mass and SUBTRACT 3D-noise voids to create "
     "caves, lava tubes, and overhangs — the structure a heightmap can never make."),
    ("fractal surface sculptor",
     "Stack multi-octave fbm at several frequencies for self-similar rocky roughness; "
     "use ridge noise (1 - 2*|n-0.5|) for sharp basalt crests and gullies."),
    ("erosion & drainage modeller",
     "Cut steep channels and gullies into the flanks with directional noise; think about "
     "how water and lava would incise the cone into badlands-like rugged relief."),
]

SVDAG_PROMPT_SKETCHES = [
    ("cone + fractal skin",
     "h = base + cone_weight*cone^2 + rough_weight*(fbm2 - 0.5); return h*H - cell_y; "
     "then carve voids with d -= k*max(fbm3 - thresh, 0)."),
    ("subtractive cave system",
     "Start solid below a rough surface, then punch lava tubes: for several 3D-noise "
     "bands, subtract a void where nb_fbm3(...) exceeds a threshold."),
    ("ridged badlands",
     "Use ridge noise r = 1 - 2*|nb_vnoise2 - 0.5| summed over octaves for sharp crests; "
     "add deep directional gullies; keep slopes steep."),
    ("caldera ring",
     "Build a cone, then subtract a central depression (a smooth bowl near r<0.15) and "
     "carve an inner crater wall; add overhangs on the rim."),
]


def build_llm_prompt(island, top_programs, generation, instance_names, hint=""):
    exemplars = ""
    for i, p in enumerate(sorted(top_programs, key=lambda x: -x.fitness)[:3]):
        exemplars += f"\n// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}\n"
    persona_name, persona_hint = SVDAG_ISLAND_PERSONAS[island.id % len(SVDAG_ISLAND_PERSONAS)]
    sketch_name, sketch_desc = SVDAG_PROMPT_SKETCHES[(island.id + generation) % len(SVDAG_PROMPT_SKETCHES)]
    target = instance_names[0] if instance_names else "stratovolcano"
    hint_section = f"\n## Strategic advice (plateau breaker)\n{hint}\n" if hint else ""
    return f"""You are a shader/terrain expert acting as a "{persona_name}".
{persona_hint}

Write a C `compute_density()` for an SVDAG VOLCANIC terrain — the function that
decides which voxels are solid rock. Target archetype: **{target}**.

{COMPUTE_DENSITY_SIGNATURE}

Island {island.id} — Generation {generation}
Approach hint ({sketch_name}): {sketch_desc}
{hint_section}
Top programs in this island:
{exemplars}

Your goal: produce terrain that reads as rocky, porous, volcanic — a coherent
macro form with steep fractal flanks AND genuine overhangs/caves (porosity),
while keeping the solid set connected (no floating debris).

Rules:
- Return ONLY the C compute_density() implementation in a single ```cpp code block
- Use the exact signature: extern "C" float compute_density(float world_x, float world_z, float cell_y, float seed)
- Call the provided nb_fbm2 / nb_fbm3 / nb_vnoise2 / nb_vnoise3 helpers (do not redefine them)
- You MAY add static helper functions above compute_density
- Do NOT redefine main() or the noise helpers
- Keep it under 40 lines
- A pure heightmap scores poorly — you MUST carve some porosity to win
"""


def parse_llm_response(response: str) -> str:
    if not response or not response.strip():
        return ""
    text = response.strip()
    fence = re.search(r"```[a-zA-Z0-9_+\-]*[ \t]*\n?(.*?)```", text, re.DOTALL)
    if fence and fence.group(1).strip():
        return fence.group(1).strip()
    sig = re.search(r'(?:extern\s+"C"\s+)?(?:inline\s+)?float\s+compute_density\s*\(', text)
    if sig:
        return text[sig.start():].strip()
    return text.replace("```cpp", "").replace("```c++", "").replace("```", "").strip()
