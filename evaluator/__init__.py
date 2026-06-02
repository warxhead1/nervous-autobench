"""Evaluation with verdict-level signals for autobench.

Classes:
    BenchmarkEvaluator — runs harness against benchmark suite
    emit_verdict(code_output, stderr, runtime, memory) — CE/RE/TLE/MLE/WA/OK
    score_harness(harness_results[], utility_weights) — U = 0.5·score + 0.25·cost + 0.25·time (SICA formula)

This package was split out of the former monolithic ``evaluator.py`` module
(behavior-preserving). The façade below re-exports the COMPLETE prior public
namespace so ``from autobench.evaluator import X`` keeps working unchanged.
"""

from __future__ import annotations

# Re-export the module-level names that were importable from the old flat
# module (these appeared in ``dir(autobench.evaluator)`` and code/tests may
# reach for them, e.g. monkeypatching ``autobench.evaluator.Path``).
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import HarnessConfig, HarnessResult, RolloutProtocol, Verdict
from ..observability import GENERATED_CODE_TRUNCATE_LEN, AutobenchObservability
from ..engines.sandbox import (
    ExecutionResult,
    SandboxedExecutor,
    compile_and_run,
    verify_output,
)

from .types import (
    DEFAULT_DISSENT_THRESHOLD,
    DEFAULT_JUDGES_PER_CASE,
    DEFAULT_MAX_COST_PER_CASE_USD,
    DEFAULT_WEIGHTS,
    ITERATIVE_MAX_ATTEMPTS,
    _VERDICT_PRECEDENCE,
    BenchmarkCase,
    BenchmarkResult,
    _build_revision_context,
    _find_last_usage,
    _normalize_p_cost,
    _worst_verdict,
)
from .judging import (
    JudgeVote,
    JudgingPool,
    emit_verdict,
    score_harness,
)
from .engine import BenchmarkEvaluator

__all__ = [
    # Constants
    "DEFAULT_JUDGES_PER_CASE",
    "DEFAULT_DISSENT_THRESHOLD",
    "ITERATIVE_MAX_ATTEMPTS",
    "DEFAULT_MAX_COST_PER_CASE_USD",
    "DEFAULT_WEIGHTS",
    "_VERDICT_PRECEDENCE",
    # Helpers
    "_build_revision_context",
    "_worst_verdict",
    "_find_last_usage",
    "_normalize_p_cost",
    # Dataclasses / classes
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkEvaluator",
    "JudgeVote",
    "JudgingPool",
    # Functions
    "emit_verdict",
    "score_harness",
    # Re-exported third-party / autobench symbols (back-compat namespace)
    "HarnessConfig",
    "HarnessResult",
    "RolloutProtocol",
    "Verdict",
    "GENERATED_CODE_TRUNCATE_LEN",
    "AutobenchObservability",
    "ExecutionResult",
    "SandboxedExecutor",
    "compile_and_run",
    "verify_output",
    "Path",
    "Any",
    "dataclass",
    "field",
    "os",
    "subprocess",
    "threading",
    "time",
]
