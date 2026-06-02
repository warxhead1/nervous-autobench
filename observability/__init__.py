"""Live observability layer for autobench.

Emits phase / iteration / sandbox / improver events on the nervous-bus so that a
single autobench run is visible end-to-end in real time. Designed to be
**non-blocking and never raise** — observability must not corrupt the
correctness of the harness.

Channels:
    autobench.phase.v1      — start/complete/error for a named phase
    autobench.iteration.v1  — RSI iteration start/complete + aggregate scores
    autobench.sandbox.v1    — per-case sandbox dispatch + completion
    autobench.improver.v1   — improver model call boundaries

Emission mechanism mirrors AutobenchResultPublisher: try `zellij pipe`, fall
back to ~/.cache/nervous-bus/debug.jsonl.

This package was split out of the former single-file ``observability.py``.
The split is behaviour-preserving: every name that was public at module level
before is re-exported here, so ``import autobench.observability as obs;
obs.X`` keeps working for every prior ``X``.
"""

from __future__ import annotations

# Re-export the stdlib / typing names that were visible at module level in the
# original single-file module so ``dir(autobench.observability)`` stays a
# superset of the pre-split namespace (consumers occasionally reach through
# the module for these).
import contextlib  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import random  # noqa: F401
import shutil  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
import time  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Iterator  # noqa: F401

# Channel constants + payload-size constants.
from .channels import (
    CHANNEL_AC_VERIFIED,
    CHANNEL_ADVERSARIAL_GENERATED,
    CHANNEL_ADVERSARIAL_ROUND,
    CHANNEL_BENCH_COMPLETED,
    CHANNEL_BUS_NOTIFY,
    CHANNEL_CASE_RESULT,
    CHANNEL_CHECKPOINT_REVERT,
    CHANNEL_CONTINUOUS_DIGEST,
    CHANNEL_CONTINUOUS_SESSION,
    CHANNEL_CROSS_DOMAIN_EVALUATION,
    CHANNEL_CURRICULUM_CYCLE,
    CHANNEL_CURRICULUM_PROBLEM,
    CHANNEL_CURRICULUM_PROBLEM_REJECTED,
    CHANNEL_CYCLE_REPORT,
    CHANNEL_CYCLE_REQUESTED,
    CHANNEL_DELTA_DIFF,
    CHANNEL_DIVERSITY,
    CHANNEL_FAILURE_CATEGORY,
    CHANNEL_FAILURE_PATTERN,
    CHANNEL_IMPROVER,
    CHANNEL_IMPROVER_DIVERGENCE,
    CHANNEL_IMPROVER_ENSEMBLE,
    CHANNEL_IMPROVER_REASONING,
    CHANNEL_ITERATION,
    CHANNEL_ITERATION_SUMMARY,
    CHANNEL_JUDGE_DISAGREEMENT,
    CHANNEL_JUDGE_POOL_VERDICT,
    CHANNEL_PHASE,
    CHANNEL_POPULATION_SUMMARY,
    CHANNEL_PREDICTION,
    CHANNEL_PREDICTION_CLIPPED,
    CHANNEL_PREDICTION_REFUTED_LIVE,
    CHANNEL_PREDICTION_VERIFIED,
    CHANNEL_PROMOTION_DECISION,
    CHANNEL_SANDBOX,
    CHANNEL_SANDBOX_STDERR,
    CHANNEL_SCORING_ADAPTED,
    CHANNEL_SYMBOL_LINEAGE,
    CHANNEL_THRESHOLD_ADAPTED,
    CHANNEL_WORKER,
    CHANNEL_WORKER_QUEUE_PRESSURE,
    GENERATED_CODE_TRUNCATE_LEN,
    SANDBOX_STDERR_EXCERPT_LEN,
    SOURCE,
)

# Stateless helpers (ULID/time, debug-dir, formatters, schema validation).
from ._util import (
    DEBUG_CACHE,
    DEBUG_FILE,
    _CROCKFORD,
    _DELTA_FIELDS,
    _TRUNCATE_MARKER,
    _dict_diff,
    _ensure_debug_dir,
    _fmt_value,
    _iso_now,
    _schemas_dir,
    _truncate,
    _ulid,
    _validate_data_payload,
)

# Core class + factory.
from .core import AutobenchObservability, make_observability

# Standalone convenience emitters. Importing this module BINDS them onto
# AutobenchObservability via assignment (the bindings run at import time), so
# this import must happen after ``core`` is imported above. We also re-export
# the function objects themselves to preserve the pre-split public namespace.
from .events import (
    _adversarial_curveball_generated,
    _adversarial_round_complete,
    _bench_completed_promotion,
    _bus_notify,
    _continuous_digest,
    _continuous_session_complete,
    _cross_domain_evaluation_complete,
    _cycle_report,
    _cycle_requested,
    _diversity_snapshot,
    _failure_pattern,
    _improver_delta_diff,
    _iteration_summary,
    _population_summary,
    _prediction_refuted_live,
    _promotion_decision,
    _sandbox_stderr,
    _worker_call,
    _worker_queue_pressure,
)


__all__ = [
    # Core
    "AutobenchObservability",
    "make_observability",
    # Constants
    "SOURCE",
    "GENERATED_CODE_TRUNCATE_LEN",
    "SANDBOX_STDERR_EXCERPT_LEN",
    "DEBUG_CACHE",
    "DEBUG_FILE",
    # Helpers
    "_ulid",
    "_iso_now",
    "_ensure_debug_dir",
    "_truncate",
    "_dict_diff",
    "_fmt_value",
    "_schemas_dir",
    "_validate_data_payload",
    # Channel constants
    "CHANNEL_PHASE",
    "CHANNEL_ITERATION",
    "CHANNEL_SANDBOX",
    "CHANNEL_IMPROVER",
    "CHANNEL_IMPROVER_REASONING",
    "CHANNEL_IMPROVER_DIVERGENCE",
    "CHANNEL_CASE_RESULT",
    "CHANNEL_THRESHOLD_ADAPTED",
    "CHANNEL_CURRICULUM_PROBLEM",
    "CHANNEL_CURRICULUM_PROBLEM_REJECTED",
    "CHANNEL_CURRICULUM_CYCLE",
    "CHANNEL_PREDICTION",
    "CHANNEL_PREDICTION_VERIFIED",
    "CHANNEL_PREDICTION_CLIPPED",
    "CHANNEL_CHECKPOINT_REVERT",
    "CHANNEL_JUDGE_POOL_VERDICT",
    "CHANNEL_JUDGE_DISAGREEMENT",
    "CHANNEL_IMPROVER_ENSEMBLE",
    "CHANNEL_AC_VERIFIED",
    "CHANNEL_SYMBOL_LINEAGE",
    "CHANNEL_DIVERSITY",
    "CHANNEL_ADVERSARIAL_GENERATED",
    "CHANNEL_ADVERSARIAL_ROUND",
    "CHANNEL_CONTINUOUS_SESSION",
    "CHANNEL_CONTINUOUS_DIGEST",
    "CHANNEL_PROMOTION_DECISION",
    "CHANNEL_WORKER",
    "CHANNEL_SANDBOX_STDERR",
    "CHANNEL_FAILURE_CATEGORY",
    "CHANNEL_WORKER_QUEUE_PRESSURE",
    "CHANNEL_ITERATION_SUMMARY",
    "CHANNEL_FAILURE_PATTERN",
    "CHANNEL_PREDICTION_REFUTED_LIVE",
    "CHANNEL_SCORING_ADAPTED",
    "CHANNEL_DELTA_DIFF",
    "CHANNEL_POPULATION_SUMMARY",
    "CHANNEL_CROSS_DOMAIN_EVALUATION",
    "CHANNEL_BENCH_COMPLETED",
    "CHANNEL_CYCLE_REQUESTED",
    "CHANNEL_CYCLE_REPORT",
    "CHANNEL_BUS_NOTIFY",
    # Standalone emitters (attached to the class via assignment in .events)
    "_diversity_snapshot",
    "_adversarial_curveball_generated",
    "_adversarial_round_complete",
    "_continuous_session_complete",
    "_continuous_digest",
    "_promotion_decision",
    "_worker_call",
    "_sandbox_stderr",
    "_worker_queue_pressure",
    "_iteration_summary",
    "_failure_pattern",
    "_prediction_refuted_live",
    "_improver_delta_diff",
    "_population_summary",
    "_cross_domain_evaluation_complete",
    "_bench_completed_promotion",
    "_cycle_requested",
    "_cycle_report",
    "_bus_notify",
]
