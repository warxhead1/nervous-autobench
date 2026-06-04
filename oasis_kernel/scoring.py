"""Oasis scoring — single-instance candidate evaluation via the sandboxed C++."""
from __future__ import annotations

import json
import math
from typing import Optional

from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run
from .instance import OasisInstance
from .oracle import build_candidate_source


def evaluate_on_instance(
    code: str,
    instance: OasisInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 30.0,
) -> Optional[float]:
    """Compile the evolved flux() + harness, run the shallow-water sim, score.

    Fitness = basin_capture · (0.4+0.6·stability) · pool_fraction ·
              (0.4+0.6·breathing), computed in the C++ harness. Returns None on
              compile/run failure or an invalid (NaN/flood/no-flow) result.
    """
    cpp_source, stdin_data = build_candidate_source(code, instance)
    stdout, verdict, _latency = compile_and_run(
        cpp_source, language="cpp", stdin=stdin_data, executor=executor,
    )
    if verdict != Verdict.OK or not stdout:
        return None
    try:
        out = json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None
    if not out.get("valid", False):
        return None
    fitness = float(out.get("fitness", 0.0))
    return fitness if math.isfinite(fitness) and fitness > 0 else None
