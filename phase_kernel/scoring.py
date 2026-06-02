"""Phase field scoring — single-instance candidate evaluation.

Behavior-preserving split of phase_kernel/__init__.py: the sandbox
compile/run + oracle-phase fitness scheduling moved here verbatim.
"""
from __future__ import annotations

import json
import math
from typing import Optional

from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run
from .instance import PhaseInstance
from .oracle import build_candidate_source


# ---------------------------------------------------------------------------
# Single-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_on_instance(
    code: str,
    instance: PhaseInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 30.0,
    generation: int = 999,
) -> Optional[float]:
    """Evaluate one candidate on one instance.

    Oracle phase scheduling (Layer 0→1→2→3):
      gen <  3: equil_score only         — establish correct attractor basins first
      gen <  6: 0.5*tanh + 0.5*equil    — add interface shape
      gen >= 6: 0.5*tanh + 0.3*equil + 0.2*width  — full oracle
    """
    cpp_source, stdin_data = build_candidate_source(code, instance)

    stdout, verdict, _latency = compile_and_run(
        cpp_source,
        language="cpp",
        stdin=stdin_data,
        executor=executor,
    )
    if verdict != Verdict.OK or not stdout:
        return None

    try:
        out = json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None

    if not out.get("valid", False):
        return None

    tanh_s = float(out.get("tanh_score",  0.0))
    equil_s = float(out.get("equil_score", 0.0))
    width_s = float(out.get("width_score", 0.0))

    # Oracle phase scheduling: build up complexity over generations
    if generation < 3:
        fitness = equil_s                              # Layer 1: equilibrium basins only
    elif generation < 6:
        fitness = 0.5 * tanh_s + 0.5 * equil_s       # Layer 2: add interface shape
    else:
        fitness = 0.5 * tanh_s + 0.3 * equil_s + 0.2 * width_s  # Layer 3: full oracle

    fitness = min(fitness, 1.0)  # defensive: components should already be in [0,1]
    return fitness if math.isfinite(fitness) and fitness > 0 else None
