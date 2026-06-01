"""eval — FunSearch evaluator backed by the NervousKernelBridge (relocated).

Phase 1 of the kernel restructuring. Behaviour is byte-identical to
``autobench.nervous_kernel_eval`` — pure relocation, no edits.

Evaluates a FunSearch terrain candidate via GPU (preferred) or CPU fallback.

GPU path: NervousKernelBridge.inject() → WT 653 timing → normalised fitness
CPU path: compile C via subprocess gcc, sample grid, compare to reference

Set NERVOUS_BUS_LIVE=1 to enable GPU evaluation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from .bridge import NervousKernelBridge, BridgeError


# ---------------------------------------------------------------------------
# CPU evaluator helpers
# ---------------------------------------------------------------------------

# Minimal C harness: compile the evolved function and sample a 16×16 grid.
# Returns JSON: {"valid": true/false, "mean": ..., "std": ..., "range": ...}
_CPU_HARNESS = r"""
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct { float x, y; } vec2;
static inline float fract(float x)     { return x - floorf(x); }
static inline float mix(float a, float b, float t) { return a + t*(b-a); }
static inline float fabsf_(float x)    { return x < 0 ? -x : x; }

// --- EVOLVED_CODE ---

#define GRID 16
int main(void) {
    float samples[GRID * GRID];
    int valid = 1;
    for (int iy = 0; iy < GRID && valid; iy++) {
        for (int ix = 0; ix < GRID && valid; ix++) {
            float px = -1.0f + 2.0f * ix / (float)(GRID - 1);
            float py = -1.0f + 2.0f * iy / (float)(GRID - 1);
            vec2 p = {px, py};
            float v = terrain(p);
            if (!isfinite(v) || v < -1e6f || v > 1e6f) { valid = 0; break; }
            samples[iy * GRID + ix] = v;
        }
    }
    if (!valid) { printf("{\"valid\":false}\n"); return 0; }

    double sum = 0.0, sum2 = 0.0;
    float vmin = samples[0], vmax = samples[0];
    for (int i = 0; i < GRID*GRID; i++) {
        sum  += samples[i];
        sum2 += (double)samples[i] * samples[i];
        if (samples[i] < vmin) vmin = samples[i];
        if (samples[i] > vmax) vmax = samples[i];
    }
    double mean = sum / (GRID*GRID);
    double var  = sum2/(GRID*GRID) - mean*mean;
    double std  = var > 0 ? sqrt(var) : 0.0;
    printf("{\"valid\":true,\"mean\":%.6f,\"std\":%.6f,\"range\":%.6f}\n",
           mean, std, vmax - vmin);
    return 0;
}
"""


class NervousKernelEvaluator:
    """
    Evaluates a FunSearch terrain candidate via GPU (preferred) or CPU fallback.

    GPU path: NervousKernelBridge.inject() → WT 653 timing
    CPU path: compile C via subprocess gcc, sample grid, compare to reference

    Set NERVOUS_BUS_LIVE=1 to enable GPU evaluation.
    """

    def __init__(self) -> None:
        self.bridge = NervousKernelBridge()

    def evaluate(self, terrain_c_code: str, instance: str) -> Optional[float]:
        """Returns fitness in [0, 1] or None on hard failure."""
        if os.environ.get("NERVOUS_BUS_LIVE") and self.bridge.is_available():
            return self._gpu_evaluate(terrain_c_code, instance)
        return self._cpu_evaluate(terrain_c_code, instance)

    # ------------------------------------------------------------------
    # GPU path
    # ------------------------------------------------------------------

    def _gpu_evaluate(self, c_code: str, instance: str) -> Optional[float]:
        """Inject kernel, wait, read WT 653 timing, convert to fitness ∈ [0,1]."""
        try:
            ns = self.bridge.inject(c_code)
            # Convert timing to fitness: faster = better, normalised to [0,1].
            # Baseline: 10ms (10_000_000 ns) = fitness 0.5
            # < 1ms = fitness near 1.0, > 100ms = fitness near 0.0
            if ns <= 0:
                return None
            fitness = 1.0 / (1.0 + ns / 2_000_000.0)  # 2ms midpoint
            if _NUMPY_AVAILABLE:
                import numpy as _np
                return float(_np.clip(fitness, 0.0, 1.0))
            return float(max(0.0, min(1.0, fitness)))
        except Exception as e:
            print(f"[NK bridge] GPU eval failed: {e}", file=sys.stderr)
            return None

    # ------------------------------------------------------------------
    # CPU path (fallback)
    # ------------------------------------------------------------------

    def _cpu_evaluate(self, c_code: str, instance: str) -> Optional[float]:
        """Compile via gcc and sample a small grid; return basic fitness."""
        # Build C source: inject evolved code into the harness
        harness = _CPU_HARNESS.replace("// --- EVOLVED_CODE ---", c_code, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "terrain_eval.c"
            exe = Path(tmpdir) / "terrain_eval"
            src.write_text(harness)

            # Compile
            try:
                result = subprocess.run(
                    ["gcc", "-O2", "-o", str(exe), str(src), "-lm"],
                    capture_output=True, text=True, timeout=15.0,
                )
                if result.returncode != 0:
                    return None
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return None

            # Run
            try:
                run_result = subprocess.run(
                    [str(exe)],
                    capture_output=True, text=True, timeout=5.0,
                )
                if run_result.returncode != 0:
                    return None
                stdout = run_result.stdout.strip()
                if not stdout:
                    return None
            except subprocess.TimeoutExpired:
                return None

        # Parse output
        try:
            out = json.loads(stdout.split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            return None

        if not out.get("valid", False):
            return None

        # Basic fitness: non-constant (range > 0.01), bounded std
        height_range = float(out.get("range", 0.0))
        std = float(out.get("std", 0.0))
        if height_range < 0.01:
            return 0.001  # constant function → reject

        # Reward non-trivial, bounded output in [0, 1]
        range_score = min(1.0, height_range / 0.5)
        std_score = min(1.0, std / 0.2)
        fitness = 0.5 * range_score + 0.5 * std_score
        return float(max(0.0, min(1.0, fitness)))


# ---------------------------------------------------------------------------
# __main__ — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .bridge import _ROLLING_HILLS_C

    evaluator = NervousKernelEvaluator()
    live = bool(os.environ.get("NERVOUS_BUS_LIVE"))
    gpu_avail = evaluator.bridge.is_available()
    print(f"NervousKernelEvaluator: NERVOUS_BUS_LIVE={live} bridge={gpu_avail}")

    fitness = evaluator.evaluate(_ROLLING_HILLS_C, "rolling_hills")
    path_used = "GPU" if (live and gpu_avail) else "CPU"
    print(f"Rolling hills fitness ({path_used}): {fitness}")
