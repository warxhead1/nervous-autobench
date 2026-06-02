"""Topology oracle — sign-change-density scoring on a 3D grid.

Calibrated per-instance Gaussian targets (TOPO_TARGETS), the C++ topology
harness skeleton, and compute_topology_score which compiles + runs the harness
in the sandbox and applies the Gaussian oracle.
"""

from __future__ import annotations

import json
import logging
import math

from .instance import SDFInstance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topology oracle targets (empirically measured from analytical functions)
# ---------------------------------------------------------------------------

TOPO_TARGETS: dict[str, dict[str, float]] = {
    "gyroid":        {"target": 0.178, "sigma": 0.060},
    "round_box":     {"target": 0.009, "sigma": 0.005},
    "warped_sphere": {"target": 0.018, "sigma": 0.008},
    # New instances — measured on 24³ grid in their bboxes (2026-05-30)
    "cloud_cluster": {"target": 0.010, "sigma": 0.005},  # measured: 0.0095 (7-sphere min-union)
    "torus_knot":    {"target": 0.021, "sigma": 0.010},  # measured: 0.0212 (trefoil tube)
    "helix_tube":    {"target": 0.019, "sigma": 0.008},  # measured: 0.0186 (spring coil)
    "scherk_first":  {"target": 0.070, "sigma": 0.025},  # measured: 0.0696 (doubly-periodic)
    # fallback for unknown instances
    "_default":      {"target": 0.050, "sigma": 0.050},
}

# Topology harness skeleton — LO, HI, GRID_N replaced per-instance at call time.
TOPO_SKELETON = r"""// Topology harness — counts sign changes on a 3D grid.
// LLM-evolved sdf() is appended below this skeleton.
#include <bits/stdc++.h>
using namespace std;

extern "C" float sdf(float x, float y, float z);
// LLM_SDF_PLACEHOLDER

int main() {
    double lo = LO_PLACEHOLDER;
    double hi = HI_PLACEHOLDER;
    int n = GRID_N_PLACEHOLDER;
    double step = (hi - lo) / n;
    long sign_changes = 0;
    long total = (long)n * n * n;

    // Sign changes along x-axis
    for (int iz = 0; iz < n; iz++) {
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n - 1; ix++) {
                float a = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                float b = sdf((float)(lo + (ix + 1) * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                if (isfinite(a) && isfinite(b) && ((a >= 0.0f) != (b >= 0.0f)))
                    sign_changes++;
            }
        }
    }
    // Sign changes along y-axis
    for (int iz = 0; iz < n; iz++) {
        for (int iy = 0; iy < n - 1; iy++) {
            for (int ix = 0; ix < n; ix++) {
                float a = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                float b = sdf((float)(lo + ix * step),
                              (float)(lo + (iy + 1) * step),
                              (float)(lo + iz * step));
                if (isfinite(a) && isfinite(b) && ((a >= 0.0f) != (b >= 0.0f)))
                    sign_changes++;
            }
        }
    }
    // Sign changes along z-axis
    for (int iz = 0; iz < n - 1; iz++) {
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n; ix++) {
                float a = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + iz * step));
                float b = sdf((float)(lo + ix * step),
                              (float)(lo + iy * step),
                              (float)(lo + (iz + 1) * step));
                if (isfinite(a) && isfinite(b) && ((a >= 0.0f) != (b >= 0.0f)))
                    sign_changes++;
            }
        }
    }
    double density = (double)sign_changes / (3.0 * total);
    printf("{\"sign_changes\":%ld,\"total\":%ld,\"density\":%.6f}\n",
           sign_changes, 3 * total, density);
    return 0;
}
"""


def build_topology_source(sdf_code: str, instance: "SDFInstance", grid_n: int = 24) -> str:
    """Build C++ source for topology harness with bbox and sdf() baked in."""
    lo, hi = instance.bbox
    source = TOPO_SKELETON
    source = source.replace("LO_PLACEHOLDER", repr(float(lo)))
    source = source.replace("HI_PLACEHOLDER", repr(float(hi)))
    source = source.replace("GRID_N_PLACEHOLDER", str(grid_n))
    # Remove the placeholder comment and append the actual sdf code
    source = source.replace("// LLM_SDF_PLACEHOLDER", "")
    return source + "\n" + sdf_code + "\n"


def compute_topology_score(
    sdf_code: str,
    instance: "SDFInstance",
    executor: "SandboxedExecutor",
    run_timeout: float = 10.0,
    grid_n: int = 24,
) -> tuple[float, float]:
    """Compute topology score for sdf_code on instance.

    Compiles a C++ topology harness that evaluates sign_change_density on a
    grid_n^3 grid, then applies a Gaussian oracle against the calibrated target.

    Returns (topology_score, sign_change_density).
    On any failure, returns (0.0, 0.0) so the main oracle stays live.
    """
    # Resolve through the package namespace at call time so tests that
    # patch("autobench.sdf_kernel.compile_and_run") take effect here.
    from . import compile_and_run
    from ..core import Verdict

    source = build_topology_source(sdf_code, instance, grid_n=grid_n)
    try:
        stdout, verdict, _latency = compile_and_run(
            source,
            "cpp",
            constraints={"max_time_seconds": run_timeout, "max_memory_mb": executor.max_memory_mb},
            stdin="",
            executor=executor,
        )
        if verdict != Verdict.OK:
            logger.debug("Topology harness non-OK verdict %s on %s", verdict, instance.name)
            return 0.0, 0.0
        out = json.loads(stdout.strip())
        density = float(out["density"])
        if not math.isfinite(density) or density < 0:
            return 0.0, 0.0
    except Exception as exc:
        logger.debug("Topology score failed on %s: %s", instance.name, exc)
        return 0.0, 0.0

    params = TOPO_TARGETS.get(instance.name, TOPO_TARGETS["_default"])
    target = params["target"]
    sigma = params["sigma"]
    score = math.exp(-0.5 * ((density - target) / sigma) ** 2)
    return score, density
