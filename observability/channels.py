"""Channel name constants + payload-size constants for autobench observability.

Canonical home for every ``CHANNEL_*`` string and the small numeric constants
that bound bus payloads. Submodules (``core``, ``events``) import the names they
need from here. This module imports nothing from the package to avoid cycles.

Channels:
    autobench.phase.v1      — start/complete/error for a named phase
    autobench.iteration.v1  — RSI iteration start/complete + aggregate scores
    autobench.sandbox.v1    — per-case sandbox dispatch + completion
    autobench.improver.v1   — improver model call boundaries
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Channel constants
# --------------------------------------------------------------------------- #

CHANNEL_PHASE = "autobench.phase.v1"
CHANNEL_ITERATION = "autobench.iteration.v1"
CHANNEL_SANDBOX = "autobench.sandbox.v1"
CHANNEL_IMPROVER = "autobench.improver.v1"
CHANNEL_IMPROVER_REASONING = "autobench.improver.reasoning.v1"
CHANNEL_IMPROVER_DIVERGENCE = "autobench.improver.divergence.v1"
CHANNEL_CASE_RESULT = "autobench.case.result.v1"
CHANNEL_THRESHOLD_ADAPTED = "autobench.improver.convergence.threshold_adapted.v1"
CHANNEL_CURRICULUM_PROBLEM = "autobench.curriculum.problem.v1"
CHANNEL_CURRICULUM_PROBLEM_REJECTED = "autobench.curriculum.problem.rejected.v1"
CHANNEL_CURRICULUM_CYCLE = "autobench.curriculum.cycle.v1"
CHANNEL_PREDICTION = "autobench.improver.prediction.v1"
CHANNEL_PREDICTION_VERIFIED = "autobench.improver.prediction.verified.v1"
# nervous-bus-8d1d: emitted when an improver-proposed prediction was
# clipped to feasible verdict-count deltas before persistence.
CHANNEL_PREDICTION_CLIPPED = "autobench.improver.prediction.clipped.v1"
# nervous-bus-sf0y: emitted when an iteration regressed > variance_floor
# (default 2σ ≈ 0.027) below the best-so-far iter and the RSI loop
# reverted its working harness to the best-iter checkpoint.
CHANNEL_CHECKPOINT_REVERT = "autobench.rsi.checkpoint_revert.v1"
# nervous-bus-c48: anonymous N-judge pool wired into the live evaluator loop.
# pool.verdict fires per case, disagreement fires when dissent_ratio > threshold.
CHANNEL_JUDGE_POOL_VERDICT = "autobench.judge.pool.verdict.v1"
CHANNEL_JUDGE_DISAGREEMENT = "autobench.judge.disagreement.v1"
# nervous-bus-9xd (wire-pop Phase 6): emitted once per RSI iteration when
# the active improver is the multi-improver ensemble. Records each
# anonymous instance's delta summary + the aggregator's vote outcome.
CHANNEL_IMPROVER_ENSEMBLE = "autobench.improver.ensemble.v1"
# nervous-bus-6kwv: emitted once per acceptance-criteria bullet executed during
# a hearth-loom loomie run. Foundation for closing the phantom-AC-passes gap
# (sibling epic hearth-loom-loom-o4cc5). The Python emitter lives here for SDK
# convenience and for autobench-side post-cycle AC verification; the primary
# producer is hearth-loom's internal/executor/verify_ac_gate.go.
CHANNEL_AC_VERIFIED = "hearth-loom.ac.verified.v1"
CHANNEL_SYMBOL_LINEAGE = "autobench.symbol.lineage.v1"

# Maximum bytes of generated code retained per event. Keeping this small (4 KiB)
# bounds the bus payload and the debug-file growth while still being long enough
# for AST feature extraction over typical competitive-programming solutions.
GENERATED_CODE_TRUNCATE_LEN = 4096

SOURCE = "/autobench"

# SACS diversity channel (autobench.diversity.v1).
CHANNEL_DIVERSITY = "autobench.diversity.v1"

# Adversarial dual co-evolution channels — Wave 5-X (nervous-bus-1rf).
CHANNEL_ADVERSARIAL_GENERATED = "autobench.adversarial.curveball_generated.v1"
CHANNEL_ADVERSARIAL_ROUND = "autobench.adversarial.round_complete.v1"

# Continuous-mode daemon channels (autobench/continuous.py).
CHANNEL_CONTINUOUS_SESSION = "autobench.continuous.session_complete.v1"
CHANNEL_CONTINUOUS_DIGEST = "autobench.continuous.digest.v1"

# Cross-run promotion decision (nervous-bus-msqa / wire-pop Phase 5).
CHANNEL_PROMOTION_DECISION = "autobench.continuous.promotion_decision.v1"

# Worker agent channel (autobench.worker.v1) — added with MiniMaxWorker.
CHANNEL_WORKER = "autobench.worker.v1"

# Sandbox stderr channel (autobench.sandbox.stderr.v1) — bead nervous-bus-bns.
CHANNEL_SANDBOX_STDERR = "autobench.sandbox.stderr.v1"
CHANNEL_FAILURE_CATEGORY = "autobench.failure.category.v1"

# Maximum characters of stderr retained per event. Kept tight (200) so the
# bus payload stays scannable and one event fits comfortably on a single
# pulse-dashboard row. The full stderr (up to 500 chars) remains available
# on HarnessResult.error.
SANDBOX_STDERR_EXCERPT_LEN = 200

# Worker queue-pressure channel (autobench.worker.queue_pressure.v1) — bead nervous-bus-8vn.
CHANNEL_WORKER_QUEUE_PRESSURE = "autobench.worker.queue_pressure.v1"

# Iteration-summary rollup channel (autobench.iteration.summary.v1).
CHANNEL_ITERATION_SUMMARY = "autobench.iteration.summary.v1"

# Failure-pattern channel (autobench.failure_pattern.v1) — nervous-bus-46v.
CHANNEL_FAILURE_PATTERN = "autobench.failure_pattern.v1"

# Live prediction refutation channel (nervous-bus-ykn).
CHANNEL_PREDICTION_REFUTED_LIVE = "autobench.improver.prediction.refuted_live.v1"
CHANNEL_SCORING_ADAPTED = "autobench.scoring.weights_adapted.v1"

# Harness before/after diff channel (autobench.improver.delta.diff.v1) — nervous-bus-utm.
CHANNEL_DELTA_DIFF = "autobench.improver.delta.diff.v1"

# Population summary channel (autobench.population.summary.v1) — bead nervous-bus-6yut.
CHANNEL_POPULATION_SUMMARY = "autobench.population.summary.v1"

# Cross-domain evaluation (autobench.cross_domain.evaluation.v1) — qp91.
CHANNEL_CROSS_DOMAIN_EVALUATION = "autobench.cross_domain.evaluation.v1"

# Forge handshake — bus.bead.bench_completed.v1 (nervous-bus-e8x9).
CHANNEL_BENCH_COMPLETED = "bus.bead.bench_completed.v1"

# Producer-triggered cycle channels — nervous-bus-1hlf.
CHANNEL_CYCLE_REQUESTED = "autobench.cycle.requested.v1"
CHANNEL_CYCLE_REPORT = "autobench.cycle.report.v1"

# bus.notify.v1 — nervous-bus-ibkg.
CHANNEL_BUS_NOTIFY = "bus.notify.v1"
