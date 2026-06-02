"""Continuous-mode daemon for autobench.

Long-running supervisor that runs :class:`SelfImprovingHarness` sessions
indefinitely, promotes any harness that beats the canonical baseline, and
emits per-session events on the nervous-bus. :class:`SurpriseDigest` mines
the last 24h of the debug log for an operator-facing "biggest surprise"
report (refutations, divergence wins, regressions, verdict-class flips).

CLI subcommands: ``run``, ``once``, ``digest``, ``status``.

Workspace layout: ``~/.autobench/continuous/{harness.json, stats.jsonl,
digests/, archive/}``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autobench.core import ContextManager, HarnessConfig, RolloutProtocol
from autobench.observability import (
    DEBUG_FILE,
    AutobenchObservability,
)
from autobench.audit.invalidation import (
    InvalidationEngine,
    promotion_scope_key,
    history_source_scope_key,
)


# Module-level invalidation engine (shared across all continuous runs)
_invalidation_engine: InvalidationEngine | None = None


def _get_invalidation_engine() -> InvalidationEngine:
    """Return the shared InvalidationEngine singleton for continuous mode."""
    global _invalidation_engine
    if _invalidation_engine is None:
        _invalidation_engine = InvalidationEngine()
    return _invalidation_engine


__all__ = [
    "ContinuousModeDaemon",
    "SurpriseDigest",
    "Surprise",
    "Digest",
    "PromotionDecision",
    "main",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_WORKSPACE = Path.home() / ".autobench" / "continuous"

# Recommended for MiniMax 15k/5h plan: 14000 = ~7% safety margin below the cap.
RECOMMENDED_RATE_MAX = 14000
RECOMMENDED_WINDOW_SECONDS = 18000  # 5h

# Daemon will run at most this many sessions per sliding window; each session
# burns 1 + n_iterations * (n_cases) requests, so 30 keeps headroom.
DEFAULT_SESSIONS_PER_WINDOW = 30

# Confidence threshold above which an AHE refutation is "confidently wrong".
CONFIDENT_THRESHOLD = 0.75

# Cross-run promotion ledger (nervous-bus-msqa / wire-pop Phase 5).
# Append-only JSONL at the repo's tools/ directory. One line per
# stage_promotion_from_population call (including the "no candidate" case).
# Resolved at call time so the test suite can point at a tmp_path via the
# ``ledger_path=`` arg or AUTOBENCH_PROMOTION_LEDGER env var.
PROMOTION_LEDGER_ENV = "AUTOBENCH_PROMOTION_LEDGER"
PROMOTION_CONFIRM_ENV = "AUTOBENCH_CONFIRM_PROMOTION"
PROMOTION_REJECT_ENV = "AUTOBENCH_REJECT_PROMOTION"

# Forge handshake (nervous-bus-vwc8 / e8x9): bead this daemon's canonical
# harness corresponds to. When set, _promote emits
# bus.bead.bench_completed.v1 so deer-flow's Forge auto-stamps the
# bench_delta seal. When None the bench_completed channel cannot fire
# (Forge needs a bead anchor).
BEAD_ID_ENV = "AUTOBENCH_BEAD_ID"


_LOG = logging.getLogger("autobench.continuous")


# --------------------------------------------------------------------------- #
# Workspace helpers
# --------------------------------------------------------------------------- #


def _serialise_harness(h: HarnessConfig) -> dict[str, Any]:
    return {
        "system_prompt": h.system_prompt,
        "rollout_protocol": h.rollout_protocol.value,
        "context_manager": h.context_manager.value,
        "tool_surface": h.tool_surface,
        "budget": dict(h.budget),
        # Verifiers are not currently round-trippable (they're callables); the
        # canonical store keeps them only by *name* so consumers see what was
        # configured. The runtime daemon recreates verifiers fresh each session.
        "verifier_names": [v.name for v in h.verifiers],
    }


def _deserialise_harness(raw: dict[str, Any]) -> HarnessConfig:
    rp = raw.get("rollout_protocol", RolloutProtocol.SINGLE.value)
    cm = raw.get("context_manager", ContextManager.FULL.value)
    try:
        rp_enum = RolloutProtocol(rp)
    except ValueError:
        rp_enum = RolloutProtocol.SINGLE
    try:
        cm_enum = ContextManager(cm)
    except ValueError:
        cm_enum = ContextManager.FULL
    return HarnessConfig(
        system_prompt=str(raw.get("system_prompt", "")),
        rollout_protocol=rp_enum,
        context_manager=cm_enum,
        tool_surface=str(raw.get("tool_surface", "")),
        budget=dict(raw.get("budget") or {}),
    )


def _empty_canonical() -> HarnessConfig:
    return HarnessConfig(
        system_prompt="You are a competitive programming assistant. Write correct code.",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="",
    )


# --------------------------------------------------------------------------- #
# Benchmark-case loading (duck-typed; depends on a sibling agent's curriculum)
# --------------------------------------------------------------------------- #


_BENCHMARKS_ROOT = Path(__file__).parent / "benchmarks"


def _load_cases_from_jsonl(path: Path) -> list[Any]:
    """Best-effort loader for a JSONL of benchmark cases.

    Returns a list of objects suitable to pass into ``BenchmarkEvaluator.run``.
    On failure or empty file returns ``[]`` — caller decides whether to skip.
    Uses ``autobench.evaluator.BenchmarkCase`` when importable; otherwise
    returns the raw dicts (the evaluator stub in tests can accept either).
    """
    if not path.is_file():
        return []
    try:
        from autobench.evaluator import BenchmarkCase  # local import keeps cold-start cheap
    except Exception:  # pragma: no cover — evaluator should always be importable
        BenchmarkCase = None  # type: ignore[assignment]

    cases: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            if BenchmarkCase is None:
                cases.append(raw)
            else:
                # Filter to known fields so we never explode on extras.
                kwargs = {
                    k: raw[k]
                    for k in (
                        "id",
                        "prompt",
                        "language",
                        "expected_output",
                        "expected_outputs",
                        "constraints",
                        "starter_code",
                        "test_inputs",
                        "metadata",
                    )
                    if k in raw
                }
                try:
                    cases.append(BenchmarkCase(**kwargs))
                except Exception:
                    cases.append(raw)
    return cases


def _pick_benchmark_source(workspace: Path) -> tuple[Path, list[Any]]:
    """Choose a benchmark source for this session.

    Priority:
      1. curriculum/today/cases.jsonl  (Wave 5-C output, if shipped)
      2. autobench/benchmarks/codeforces_tier1/cases.jsonl
      3. random older curriculum found under curriculum/
    Returns (path, cases). Path is informational (recorded with stats).
    """
    candidates: list[Path] = []
    curriculum = workspace.parent / "curriculum"
    today = curriculum / "today" / "cases.jsonl"
    if today.is_file():
        candidates.append(today)
    default = _BENCHMARKS_ROOT / "codeforces_tier1" / "cases.jsonl"
    if default.is_file():
        candidates.append(default)
    if curriculum.is_dir():
        for sub in sorted(curriculum.iterdir()):
            jsonl = sub / "cases.jsonl"
            if jsonl.is_file() and jsonl not in candidates:
                candidates.append(jsonl)

    for path in candidates:
        cases = _load_cases_from_jsonl(path)
        if cases:
            # Always cap to a reasonable session-sized slice. RSI per-session
            # cost scales linearly with case count; 8 is the sweet spot.
            if len(cases) > 8:
                cases = random.sample(cases, 8)
            return path, cases
    return Path("none"), []


# --------------------------------------------------------------------------- #
# ContinuousModeDaemon
# --------------------------------------------------------------------------- #


@dataclass
class _SessionRecord:
    session_id: str
    timestamp: str
    initial_score: float
    final_score: float
    n_iterations: int
    total_cost_usd: float
    promoted: bool
    benchmark_source: str
    duration_seconds: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    """One decision recorded by ``stage_promotion_from_population``.

    Persisted to the promotion ledger (``tools/promotion_ledger.jsonl`` by
    default) and emitted on the bus as
    ``autobench.continuous.promotion_decision.v1``. The dataclass is the
    authoritative shape; the schema mirrors it.
    """

    cycle_id: str
    candidate_advocate_id: str | None
    candidate_session_id: str | None
    candidate_score: float
    candidate_adjusted_score: float
    ahe_outcome: str           # "confirmed" | "partial" | "refuted" | "refuted_live" | "none"
    decision: str              # "staged" | "accepted" | "rejected"
    decided_by: str            # "cli" | "env" | "auto-skip" | "default"
    reason: str = ""

    def to_ledger_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = _iso_now_utc()
        return d

    def to_event_data(self) -> dict[str, Any]:
        return asdict(self)


def _iso_now_utc() -> str:
    """RFC3339-ish UTC timestamp with seconds precision."""
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _resolve_promotion_ledger_path(override: Path | None) -> Path:
    """Pick the ledger location. Precedence: explicit > env > default."""
    if override is not None:
        return Path(override)
    env_path = os.environ.get(PROMOTION_LEDGER_ENV, "").strip()
    if env_path:
        return Path(env_path)
    # Default: tools/promotion_ledger.jsonl relative to the repo root.
    # The repo root is the parent of the ``autobench`` package directory.
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "tools" / "promotion_ledger.jsonl"


class ContinuousModeDaemon:
    """Long-running supervisor that keeps autobench self-improving 24/7.

    Parameters:
        workspace: Directory under which canonical harness, stats, archive, and
            digests are stored. Created on demand.
        rate_budget: Optional :class:`autobench.budget_guard.RateBudgetGuard`.
            When provided, the daemon backs off (sleeps) when the budget guard
            reports the cap reached, rather than crashing.
        sessions_per_window: Soft cap on the number of sessions started per
            sliding window. Used only for pacing the inter-session sleep.
        evaluator: Optional pre-built ``BenchmarkEvaluator`` (for tests). When
            None, one is constructed lazily per session.
        improver: Optional improver-name string passed into ``SelfImprovingHarness``
            (``"minimax"`` | ``"anthropic"`` | ``"rule_based"``). Default ``"minimax"``.
        max_iterations: Per-session RSI iteration cap. Default 5 — kept low
            because the daemon expects many sessions over time.
        obs: Optional ``AutobenchObservability`` shared across sessions.
        bead_id: nervous-bus-vwc8 — Bead this daemon's canonical harness is
            attributed to. When None, falls back to ``AUTOBENCH_BEAD_ID`` env
            var; when both are None, the daemon runs unbound (legacy/test
            mode) and the bench_completed handshake stays silent. Forge
            consumers key the bench_delta seal on this id.
    """

    def __init__(
        self,
        workspace: Path | None = None,
        rate_budget: Any = None,
        sessions_per_window: int = DEFAULT_SESSIONS_PER_WINDOW,
        evaluator: Any = None,
        improver: str | None = "minimax",
        max_iterations: int = 5,
        obs: AutobenchObservability | None = None,
        bead_id: str | None = None,
    ) -> None:
        self.workspace = Path(workspace) if workspace else DEFAULT_WORKSPACE
        self.rate_budget = rate_budget
        self.sessions_per_window = max(1, int(sessions_per_window))
        self.evaluator = evaluator
        self.improver = improver
        self.max_iterations = max(1, int(max_iterations))
        self.obs = obs or AutobenchObservability(session_id=None)
        # nervous-bus-vwc8: explicit kwarg wins over env. Empty string in env
        # is treated as "unbound" — same as None — so operators can clear an
        # inherited env by passing "" without surprising precedence.
        if bead_id is not None:
            self.bead_id: str | None = str(bead_id) or None
        else:
            env_bead = os.environ.get(BEAD_ID_ENV, "").strip()
            self.bead_id = env_bead or None
        self._stop = False
        self._ensure_workspace()

    # ------------------------------------------------------------------ #
    # Workspace
    # ------------------------------------------------------------------ #

    def _ensure_workspace(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "archive").mkdir(exist_ok=True)
        (self.workspace / "digests").mkdir(exist_ok=True)
        h_path = self.workspace / "harness.json"
        if not h_path.exists():
            h_path.write_text(json.dumps(_serialise_harness(_empty_canonical()), indent=2))
        s_path = self.workspace / "stats.jsonl"
        if not s_path.exists():
            s_path.touch()

    @property
    def harness_path(self) -> Path:
        return self.workspace / "harness.json"

    @property
    def stats_path(self) -> Path:
        return self.workspace / "stats.jsonl"

    def current_canonical_harness(self) -> HarnessConfig:
        """Load the canonical harness from disk. Falls back to an empty config."""
        try:
            raw = json.loads(self.harness_path.read_text())
            return _deserialise_harness(raw)
        except Exception:
            _LOG.warning("could not load canonical harness; using fresh default")
            return _empty_canonical()

    def _promote(self, new_harness: HarnessConfig) -> None:
        """Archive current canonical, write the new one in its place."""
        try:
            ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            archive_target = self.workspace / "archive" / f"{ts}.json"
            if self.harness_path.exists():
                archive_target.write_text(self.harness_path.read_text())
            self.harness_path.write_text(
                json.dumps(_serialise_harness(new_harness), indent=2)
            )
        except Exception as e:
            _LOG.warning("promotion archive failed: %s", e)

    # ------------------------------------------------------------------ #
    # Cross-run promotion (nervous-bus-msqa / wire-pop Phase 5)
    # ------------------------------------------------------------------ #

    def stage_promotion_from_population(
        self,
        population_result: Any,  # autobench.population.PopulationResult
        *,
        confirm: bool | None = None,
        reject: bool | None = None,
        ledger_path: Path | None = None,
    ) -> "PromotionDecision":
        """Couple a PopulationRunner cycle to the canonical-harness swap.

        Selects the top confirmed-AHE advocate via
        :func:`autobench.population.select_promotion_candidate`. The actual
        swap only runs when ``confirm=True`` (or
        ``AUTOBENCH_CONFIRM_PROMOTION=1`` in env). The default behaviour is
        "stage only" — log the candidate to the ledger but never overwrite
        ``harness.json``. ``reject=True`` (or ``AUTOBENCH_REJECT_PROMOTION=1``)
        records an explicit rejection.

        Every call appends one line to the promotion ledger, including the
        "no promotable candidate" case (so the ledger is contiguous across
        cycles). The decision event is also published on the nervous-bus
        via ``autobench.continuous.promotion_decision.v1``.

        Args:
            population_result: A finished :class:`PopulationResult`.
            confirm: When True, fires :meth:`_promote` after staging.
                When None, reads ``AUTOBENCH_CONFIRM_PROMOTION``.
            reject: When True, records an explicit rejection without
                promoting. When None, reads ``AUTOBENCH_REJECT_PROMOTION``.
                ``reject`` wins over ``confirm`` when both are set.
            ledger_path: Override the default ledger location. Used by tests.

        Returns:
            A :class:`PromotionDecision` describing the final outcome.
        """
        from autobench.rsi.population import select_promotion_candidate

        # Resolve env-var gates when explicit args are None.
        if confirm is None:
            confirm = _env_truthy(PROMOTION_CONFIRM_ENV)
        if reject is None:
            reject = _env_truthy(PROMOTION_REJECT_ENV)

        cycle_id = str(getattr(population_result, "cycle_id", "") or "")

        # --- Bitloops-style source_scope_key temporal invalidation ----------
        # nervous-bus: check if this cycle was already distilled; if so,
        # prior events for this scope are auto-deactivated.
        scope_key = promotion_scope_key(cycle_id)
        inv_result = _get_invalidation_engine().check_and_invalidate(scope_key, {
            "cycle_id": cycle_id,
            "decision": "staged" if not confirm else "accepted",
        })
        if inv_result.was_invalidated:
            self.obs._publish("autobench.promotion.invalidation.v1", {
                "cycle_id": cycle_id,
                "count_deactivated": inv_result.count_deactivated,
                "reason": inv_result.reason,
            })
        # ---------------------------------------------------------------------

        advocates = list(getattr(population_result, "advocates", []) or [])
        candidate = select_promotion_candidate(advocates)

        # No promotable candidate → auto-skip. Always log.
        if candidate is None:
            decision = PromotionDecision(
                cycle_id=cycle_id,
                candidate_advocate_id=None,
                candidate_session_id=None,
                candidate_score=0.0,
                candidate_adjusted_score=0.0,
                ahe_outcome="none",
                decision="staged",
                decided_by="auto-skip",
                reason="no advocate had a confirmed or partial AHE outcome",
            )
            self._record_promotion_decision(decision, ledger_path=ledger_path)
            return decision

        # Explicit reject path wins over confirm.
        if reject:
            decided_by = "cli" if reject is True and not _env_truthy(PROMOTION_REJECT_ENV) else "env"
            # The above is awkward — simplify: if env says reject, attribute to env.
            decided_by = "env" if _env_truthy(PROMOTION_REJECT_ENV) else "cli"
            decision = PromotionDecision(
                cycle_id=cycle_id,
                candidate_advocate_id=candidate.advocate_id,
                candidate_session_id=candidate.session_id,
                candidate_score=float(candidate.best_score),
                candidate_adjusted_score=float(candidate.adjusted_score),
                ahe_outcome=str(candidate.ahe_outcome),
                decision="rejected",
                decided_by=decided_by,
                reason="operator rejected promotion",
            )
            self._record_promotion_decision(decision, ledger_path=ledger_path)
            return decision

        if confirm:
            # Accepted path — actually swap canonical.
            decided_by = "env" if _env_truthy(PROMOTION_CONFIRM_ENV) else "cli"
            try:
                self._promote(candidate.final_harness)
            except Exception as exc:  # noqa: BLE001 — never crash the daemon
                _LOG.warning("promotion swap failed for %s: %s", candidate.advocate_id, exc)
                decision = PromotionDecision(
                    cycle_id=cycle_id,
                    candidate_advocate_id=candidate.advocate_id,
                    candidate_session_id=candidate.session_id,
                    candidate_score=float(candidate.best_score),
                    candidate_adjusted_score=float(candidate.adjusted_score),
                    ahe_outcome=str(candidate.ahe_outcome),
                    decision="staged",  # fell back to staged on swap failure
                    decided_by=decided_by,
                    reason=f"promote() raised {type(exc).__name__}: {exc}",
                )
                self._record_promotion_decision(decision, ledger_path=ledger_path)
                return decision

            decision = PromotionDecision(
                cycle_id=cycle_id,
                candidate_advocate_id=candidate.advocate_id,
                candidate_session_id=candidate.session_id,
                candidate_score=float(candidate.best_score),
                candidate_adjusted_score=float(candidate.adjusted_score),
                ahe_outcome=str(candidate.ahe_outcome),
                decision="accepted",
                decided_by=decided_by,
                reason="confirmed by operator; canonical swapped",
            )
            self._record_promotion_decision(decision, ledger_path=ledger_path)
            # nervous-bus-e8x9: Forge handshake. Emit bench_completed only on
            # the accepted path (the swap actually happened) so the bead's
            # bench_delta seal is anchored to a real canonical change. The
            # call is best-effort and never raises — emission failure must
            # not poison the promotion path.
            try:
                self._emit_bench_completed(candidate)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("bench_completed emit failed: %s", e)
            return decision

        # Default — stage only.
        decision = PromotionDecision(
            cycle_id=cycle_id,
            candidate_advocate_id=candidate.advocate_id,
            candidate_session_id=candidate.session_id,
            candidate_score=float(candidate.best_score),
            candidate_adjusted_score=float(candidate.adjusted_score),
            ahe_outcome=str(candidate.ahe_outcome),
            decision="staged",
            decided_by="default",
            reason="awaiting explicit --confirm-promotion to swap canonical",
        )
        self._record_promotion_decision(decision, ledger_path=ledger_path)
        return decision

    def _record_promotion_decision(
        self,
        decision: "PromotionDecision",
        *,
        ledger_path: Path | None = None,
    ) -> None:
        """Append the decision to the ledger AND emit a bus event."""
        path = _resolve_promotion_ledger_path(ledger_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = decision.to_ledger_dict()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as e:  # noqa: BLE001 — ledger best-effort, never crash
            _LOG.warning("promotion ledger append failed: %s", e)

        # Best-effort bus emission. Falls back to debug.jsonl when no pipe.
        try:
            self.obs.promotion_decision(**decision.to_event_data())
        except Exception as e:  # noqa: BLE001
            _LOG.warning("promotion decision emit failed: %s", e)

    # ------------------------------------------------------------------ #
    # Forge handshake — nervous-bus-e8x9
    # ------------------------------------------------------------------ #

    def _last_canonical_score(self) -> float:
        """Return the most recent canonical aggregate_score known on disk.

        Reads the latest non-empty line of ``stats.jsonl`` and returns its
        ``final_score`` field (which is the score the canonical reached at
        the end of that session). Falls back to 0.0 when no history exists —
        the very first promotion of a daemon's lifetime has no measured
        baseline.
        """
        try:
            if not self.stats_path.is_file():
                return 0.0
            last: dict[str, Any] | None = None
            with self.stats_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = json.loads(line)
                    except Exception:
                        continue
            if not last:
                return 0.0
            return float(last.get("final_score", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 — best effort
            return 0.0

    def _emit_bench_completed(self, candidate: Any) -> None:
        """Emit bus.bead.bench_completed.v1 on the accepted-promotion path.

        ``candidate`` is the :class:`AdvocateResult` whose ``final_harness``
        replaced canonical. Skips emission and logs a warning when
        ``self.bead_id`` is None — the channel is bead-keyed and Forge has
        no anchor without it.
        """
        if not self.bead_id:
            _LOG.warning(
                "bench_completed skipped: daemon is unbound (no bead_id). "
                "Set AUTOBENCH_BEAD_ID or pass --bead-id to enable the "
                "Forge handshake."
            )
            return

        treatment = float(getattr(candidate, "aggregate_score", 0.0) or 0.0)
        # aggregate_score is the cross-domain figure of merit when available;
        # fall back to best_score for single-domain advocates that never set
        # aggregate.
        if treatment == 0.0:
            treatment = float(getattr(candidate, "best_score", 0.0) or 0.0)

        baseline = self._last_canonical_score()
        delta = treatment - baseline

        # n: number of cases the candidate was evaluated against. Derived
        # from final_result.case_results when present; schema requires
        # minimum 1, so floor accordingly.
        n = 1
        final_result = getattr(candidate, "final_result", None)
        if final_result is not None:
            cases = getattr(final_result, "case_results", None)
            if cases is not None:
                try:
                    n = max(1, len(cases))
                except Exception:  # noqa: BLE001
                    n = 1

        self.obs.bench_completed_promotion(
            bead_id=self.bead_id,
            baseline_metric=baseline,
            treatment_metric=treatment,
            delta=delta,
            n=n,
            passes_threshold=True,
        )

    def _append_stats(self, record: _SessionRecord) -> None:
        try:
            with self.stats_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            _LOG.warning("stats append failed: %s", e)

    # ------------------------------------------------------------------ #
    # Rate-budget pacing
    # ------------------------------------------------------------------ #

    def _slot_seconds(self) -> float:
        """Seconds between session starts to respect the per-window cap."""
        if self.rate_budget is None:
            # No guard configured — be conservative: spread 30 sessions across
            # 5 hours, i.e. one every ~10 minutes.
            return RECOMMENDED_WINDOW_SECONDS / max(1, self.sessions_per_window)
        window = float(getattr(self.rate_budget, "window_seconds", RECOMMENDED_WINDOW_SECONDS))
        return window / max(1, self.sessions_per_window)

    def _backoff_seconds(self) -> float:
        """How long to wait before retrying after a rate-budget refusal."""
        if self.rate_budget is None:
            return 60.0
        try:
            t = float(self.rate_budget.time_until_available())
            return max(60.0, t)
        except Exception:
            return 60.0

    def _budget_ok(self) -> tuple[bool, str | None]:
        if self.rate_budget is None:
            return True, None
        try:
            return self.rate_budget.check()
        except Exception:
            return True, None

    # ------------------------------------------------------------------ #
    # One session
    # ------------------------------------------------------------------ #

    def run_one_session(self) -> _SessionRecord:
        """Execute a single autobench session and update workspace state.

        Returns the :class:`_SessionRecord` for the session (also appended to
        ``stats.jsonl``). Even on hard failure a record is produced, with
        ``error`` populated and ``promoted=False`` — the daemon must never
        crash the loop.
        """
        session_id = self.obs.session_id
        started_iso = _iso_now()
        t0 = time.monotonic()
        canonical = self.current_canonical_harness()
        bench_path, cases = _pick_benchmark_source(self.workspace)
        initial_score = 0.0
        final_score = 0.0
        n_iterations = 0
        total_cost = 0.0
        promoted = False
        error = ""

        # Avoid late imports during tests that swap the evaluator: the heavy
        # imports only happen when we actually have cases to run.
        try:
            if not cases:
                error = f"no benchmark cases available (tried {bench_path})"
            else:
                # Allow tests to inject a fake evaluator that needs no
                # generate_fn — just call .run(harness, cases).
                if self.evaluator is None:
                    from autobench.evaluator import BenchmarkEvaluator
                    evaluator = BenchmarkEvaluator(obs=self.obs)
                else:
                    evaluator = self.evaluator

                # Baseline score *before* improvement
                pre = evaluator.run(canonical, cases, obs=self.obs) if _accepts_obs(evaluator.run) else evaluator.run(canonical, cases)
                initial_score = float(getattr(pre, "aggregate_score", 0.0))

                # RSI loop
                from autobench.rsi.loop import SelfImprovingHarness
                sih = SelfImprovingHarness(
                    current_harness=canonical,
                    evaluator=evaluator,
                    max_iterations=self.max_iterations,
                    default_improver=self.improver,
                    obs=self.obs,
                )
                final_h, final_result, history = sih.improve(cases)
                n_iterations = len(history)
                final_score = float(getattr(final_result, "aggregate_score", 0.0))
                total_cost = float(
                    sum(getattr(r, "cost_dollars", 0.0) for r in (
                        getattr(final_result, "case_results", []) or []
                    ))
                )

                # Promotion: strict > so noise doesn't churn canonical
                if final_score > initial_score:
                    self._promote(final_h)
                    promoted = True
        except Exception as exc:  # noqa: BLE001 — daemon never crashes
            error = f"{type(exc).__name__}: {exc}"
            _LOG.error("session failed: %s\n%s", error, traceback.format_exc())

        duration = time.monotonic() - t0
        record = _SessionRecord(
            session_id=session_id,
            timestamp=started_iso,
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=n_iterations,
            total_cost_usd=total_cost,
            promoted=promoted,
            benchmark_source=str(bench_path),
            duration_seconds=duration,
            error=error,
        )
        self._append_stats(record)
        try:
            self.obs.continuous_session_complete(
                initial_score=initial_score,
                final_score=final_score,
                n_iterations=n_iterations,
                total_cost_usd=total_cost,
                promoted=promoted,
                benchmark_source=str(bench_path),
                duration_seconds=duration,
            )
        except Exception as e:
            _LOG.warning("session-complete emit failed: %s", e)
        return record

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def request_stop(self) -> None:
        self._stop = True

    def run_forever(self, max_sessions: int | None = None) -> None:
        """Main loop. Blocks until SIGINT/SIGTERM or ``max_sessions`` reached.

        ``max_sessions`` is primarily for tests; production use leaves it None.
        """
        # Install signal handlers so the daemon shuts down cleanly.
        def _on_signal(signum, frame):  # noqa: ANN001
            _LOG.info("received signal %s; stopping after current session", signum)
            self.request_stop()

        try:
            signal.signal(signal.SIGINT, _on_signal)
            signal.signal(signal.SIGTERM, _on_signal)
        except (ValueError, AttributeError):
            # Not on main thread / not supported — fine; tests use max_sessions.
            pass

        n_run = 0
        slot = self._slot_seconds()
        while not self._stop:
            ok, reason = self._budget_ok()
            if not ok:
                wait = self._backoff_seconds()
                _LOG.info("rate budget blocked (%s); sleeping %.0fs", reason, wait)
                _sleep_interruptible(wait, self)
                continue
            try:
                self.run_one_session()
            except Exception as e:  # safety net — _run_one_session already catches
                _LOG.error("unexpected fall-through: %s", e)
            n_run += 1
            if max_sessions is not None and n_run >= max_sessions:
                break
            _sleep_interruptible(slot, self)


def _sleep_interruptible(seconds: float, daemon: ContinuousModeDaemon) -> None:
    """Sleep in 1s slices so the daemon can react to ``request_stop``."""
    end = time.monotonic() + max(0.0, seconds)
    while not daemon._stop and time.monotonic() < end:
        time.sleep(min(1.0, end - time.monotonic()))


def _accepts_obs(callable_: Any) -> bool:
    """True if the callable's signature accepts an ``obs`` kwarg."""
    try:
        import inspect
        return "obs" in inspect.signature(callable_).parameters
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------- #
# SurpriseDigest
# --------------------------------------------------------------------------- #


@dataclass
class Surprise:
    """One notable event mined from the debug log."""

    kind: str  # "confident_wrong" | "divergence_win" | "regression" | "verdict_flip"
    score: float  # how interesting — higher is more
    summary: str
    session_id: str = ""
    iteration: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Digest:
    """Result of mining a 24h window."""

    date: str
    n_sessions: int
    n_promotions: int
    surprises: list[Surprise] = field(default_factory=list)

    @property
    def n_surprises(self) -> int:
        return len(self.surprises)

    @property
    def biggest_surprise(self) -> Surprise | None:
        if not self.surprises:
            return None
        return max(self.surprises, key=lambda s: s.score)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# autobench continuous-mode digest — {self.date}")
        lines.append("")
        lines.append(
            f"- Sessions: **{self.n_sessions}** "
            f"(promoted: **{self.n_promotions}**)"
        )
        lines.append(f"- Surprises flagged: **{self.n_surprises}**")
        big = self.biggest_surprise
        if big is not None:
            lines.append("")
            lines.append(f"## Biggest surprise — `{big.kind}` (score {big.score:.2f})")
            lines.append("")
            lines.append(big.summary)
        if self.surprises:
            lines.append("")
            lines.append("## All surprises")
            lines.append("")
            for s in sorted(self.surprises, key=lambda s: -s.score):
                lines.append(f"- **{s.kind}** [{s.score:.2f}]: {s.summary}")
        else:
            lines.append("")
            lines.append("_No surprises in this window. The world is calm._")
        return "\n".join(lines) + "\n"


class SurpriseDigest:
    """Mines the last 24h of ``debug.jsonl`` for notable events.

    Parameters:
        workspace: Continuous-mode workspace path (for output digest writing).
        debug_file: Override the source JSONL for tests. Defaults to
            ``~/.cache/nervous-bus/debug.jsonl``.
        window_seconds: How far back to look. Default 24h.
        now: Override the reference 'now' for tests.
    """

    def __init__(
        self,
        workspace: Path | None = None,
        debug_file: Path | None = None,
        window_seconds: float = 86400.0,
        now: dt.datetime | None = None,
    ) -> None:
        self.workspace = Path(workspace) if workspace else DEFAULT_WORKSPACE
        self.debug_file = Path(debug_file) if debug_file else DEBUG_FILE
        self.window_seconds = float(window_seconds)
        self._now = now or dt.datetime.utcnow()

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def _iter_events(self) -> list[dict[str, Any]]:
        if not self.debug_file.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            for line in self.debug_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if not isinstance(e, dict):
                    continue
                # Cutoff filter
                t = e.get("time", "")
                if t:
                    try:
                        et = dt.datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")
                        if (self._now - et).total_seconds() > self.window_seconds:
                            continue
                    except Exception:
                        pass
                events.append(e)
        except Exception as e:
            _LOG.warning("digest read failed: %s", e)
        return events

    # ------------------------------------------------------------------ #
    # Mining
    # ------------------------------------------------------------------ #

    def generate_digest(self, date: str | None = None) -> Digest:
        events = self._iter_events()
        if date is None:
            date = self._now.strftime("%Y-%m-%d")

        surprises: list[Surprise] = []

        # --- 1. Confidently-wrong predictions (AHE refutations) ----------- #
        for e in events:
            if e.get("type") != "autobench.improver.prediction.verified.v1":
                continue
            d = e.get("data") or {}
            confidence = float(d.get("confidence", 0.0) or 0.0)
            outcome = str(d.get("outcome_label", ""))
            if outcome != "refuted":
                continue
            if confidence < CONFIDENT_THRESHOLD:
                continue
            # Score: more confidence ⇒ more interesting. Floor at the threshold.
            score = 0.5 + (confidence - CONFIDENT_THRESHOLD) * 2.0
            summary = (
                f"AHE: improver predicted with confidence {confidence:.2f} "
                f"and was REFUTED (iter {d.get('iteration', '?')}, "
                f"model={d.get('model', '?')}). "
                f"score_match_ratio={d.get('score_match_ratio', 0.0):.2f}."
            )
            surprises.append(Surprise(
                kind="confident_wrong",
                score=score,
                summary=summary,
                session_id=str(d.get("session_id", "")),
                iteration=int(d.get("iteration", 0) or 0),
                raw=d,
            ))

        # --- 2. Divergence wins (LLM diverged from heuristic) ------------- #
        divergent_events: list[tuple[str, int, dict[str, Any]]] = []
        for e in events:
            if e.get("type") != "autobench.improver.divergence.v1":
                continue
            d = e.get("data") or {}
            if not d.get("divergent"):
                continue
            divergent_events.append((
                str(d.get("session_id", "")),
                int(d.get("iteration", 0) or 0),
                d,
            ))

        # Cross-reference with iteration events to detect score deltas.
        iter_scores: dict[tuple[str, int], float] = {}
        for e in events:
            if e.get("type") != "autobench.iteration.v1":
                continue
            d = e.get("data") or {}
            if d.get("status") != "complete":
                continue
            sid = str(d.get("session_id", ""))
            it = int(d.get("iteration", 0) or 0)
            iter_scores[(sid, it)] = float(d.get("aggregate_score", 0.0) or 0.0)

        for sid, it, d in divergent_events:
            score_now = iter_scores.get((sid, it))
            score_prev = iter_scores.get((sid, it - 1))
            if score_now is None or score_prev is None:
                continue
            delta = score_now - score_prev
            if delta <= 0.01:
                # Either no improvement or degradation handled below.
                continue
            interest = 0.5 + min(delta * 5.0, 1.5)
            summary = (
                f"LLM diverged from rule-based heuristic and **won** "
                f"(+{delta:.3f} score, iter {it}). "
                f"Divergence: {d.get('divergence_summary', '?')}"
            )
            surprises.append(Surprise(
                kind="divergence_win",
                score=interest,
                summary=summary,
                session_id=sid,
                iteration=it,
                raw=d,
            ))

        # --- 3. Score regressions (within a single session) --------------- #
        # Per session, find the max initial score and compare to final score.
        per_session_scores: dict[str, list[tuple[int, float]]] = {}
        for (sid, it), s in iter_scores.items():
            per_session_scores.setdefault(sid, []).append((it, s))
        for sid, scores in per_session_scores.items():
            scores.sort()
            if len(scores) < 2:
                continue
            initial = scores[0][1]
            final = scores[-1][1]
            if final < initial - 0.01:
                interest = 0.4 + min((initial - final) * 5.0, 1.5)
                summary = (
                    f"Score regression in session {sid[:8]}…: "
                    f"started at {initial:.3f}, ended at {final:.3f} "
                    f"(Δ={final - initial:+.3f} over {len(scores)} iters)."
                )
                surprises.append(Surprise(
                    kind="regression",
                    score=interest,
                    summary=summary,
                    session_id=sid,
                    iteration=scores[-1][0],
                    raw={"initial": initial, "final": final, "n": len(scores)},
                ))

        # --- 4. Verdict-class flips between consecutive iterations -------- #
        # Group iteration events by session and look at verdict_counts shape.
        per_session_verdicts: dict[str, list[tuple[int, dict[str, int]]]] = {}
        for e in events:
            if e.get("type") != "autobench.iteration.v1":
                continue
            d = e.get("data") or {}
            if d.get("status") != "complete":
                continue
            sid = str(d.get("session_id", ""))
            it = int(d.get("iteration", 0) or 0)
            vc = d.get("verdict_counts") or {}
            if isinstance(vc, dict):
                per_session_verdicts.setdefault(sid, []).append((it, dict(vc)))
        for sid, rows in per_session_verdicts.items():
            rows.sort()
            for i in range(1, len(rows)):
                prev_v = set(rows[i - 1][1].keys())
                curr_v = set(rows[i][1].keys())
                introduced = curr_v - prev_v
                dropped = prev_v - curr_v
                if not introduced and not dropped:
                    continue
                # New failure verdicts (anything not in {OK, RV}) are most
                # interesting; new OK is also interesting (a fix).
                interesting_changes = (
                    {v for v in introduced if v not in {"OK", "RV"}}
                    | {v for v in dropped if v not in {"OK", "RV"}}
                )
                interest = 0.3 + 0.2 * len(interesting_changes)
                summary = (
                    f"Verdict class flip in session {sid[:8]}… iter "
                    f"{rows[i - 1][0]}→{rows[i][0]}: "
                    f"+{sorted(introduced) or '∅'} / -{sorted(dropped) or '∅'}"
                )
                surprises.append(Surprise(
                    kind="verdict_flip",
                    score=interest,
                    summary=summary,
                    session_id=sid,
                    iteration=rows[i][0],
                    raw={"introduced": list(introduced), "dropped": list(dropped)},
                ))

        # --- Session count + promotion count from stats.jsonl ------------- #
        n_sessions = 0
        n_promotions = 0
        stats_path = self.workspace / "stats.jsonl"
        if stats_path.is_file():
            try:
                for line in stats_path.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("timestamp", "")
                    try:
                        et = dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                        if (self._now - et).total_seconds() > self.window_seconds:
                            continue
                    except Exception:
                        pass
                    n_sessions += 1
                    if rec.get("promoted"):
                        n_promotions += 1
            except Exception as e:
                _LOG.warning("digest stats read failed: %s", e)

        return Digest(
            date=date,
            n_sessions=n_sessions,
            n_promotions=n_promotions,
            surprises=surprises,
        )

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def write_digest(self, digest: Digest, obs: AutobenchObservability | None = None) -> Path:
        digests_dir = self.workspace / "digests"
        digests_dir.mkdir(parents=True, exist_ok=True)
        out_path = digests_dir / f"{digest.date}.md"
        try:
            out_path.write_text(digest.to_markdown(), encoding="utf-8")
        except Exception as e:
            _LOG.warning("digest write failed: %s", e)
        # Emit on the bus.
        try:
            big = digest.biggest_surprise
            summary = big.summary if big else "no surprises"
            (obs or AutobenchObservability()).continuous_digest(
                date=digest.date,
                n_sessions=digest.n_sessions,
                n_promotions=digest.n_promotions,
                n_surprises=digest.n_surprises,
                biggest_surprise_summary=summary,
                digest_path=str(out_path),
            )
        except Exception as e:
            _LOG.warning("digest emit failed: %s", e)
        return out_path


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #


def _iso_now() -> str:
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_rate_budget(args: argparse.Namespace) -> Any:
    """Construct a RateBudgetGuard, with import fallback for sibling-wave skew."""
    try:
        from autobench.audit.budget_guard import RateBudgetGuard
    except Exception:
        return None
    return RateBudgetGuard(
        max_requests=int(args.max_requests),
        window_seconds=float(args.window_seconds),
        safety_margin=float(args.safety_margin),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    daemon = ContinuousModeDaemon(
        workspace=Path(args.workspace) if args.workspace else None,
        rate_budget=_build_rate_budget(args),
        sessions_per_window=args.sessions_per_window,
        improver=args.improver,
        max_iterations=args.max_iterations,
        bead_id=getattr(args, "bead_id", None),
    )
    print(f"[continuous] daemon starting (workspace={daemon.workspace})", flush=True)
    daemon.run_forever()
    print("[continuous] daemon stopped.", flush=True)
    return 0


def _cmd_once(args: argparse.Namespace) -> int:
    daemon = ContinuousModeDaemon(
        workspace=Path(args.workspace) if args.workspace else None,
        rate_budget=_build_rate_budget(args),
        sessions_per_window=args.sessions_per_window,
        improver=args.improver,
        max_iterations=args.max_iterations,
        bead_id=getattr(args, "bead_id", None),
    )
    rec = daemon.run_one_session()
    print(json.dumps(rec.to_dict(), indent=2))
    return 0


def _cmd_digest(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace) if args.workspace else DEFAULT_WORKSPACE
    workspace.mkdir(parents=True, exist_ok=True)
    sd = SurpriseDigest(workspace=workspace)
    digest = sd.generate_digest()
    out = sd.write_digest(digest)
    print(digest.to_markdown())
    print(f"(wrote {out})", file=sys.stderr)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace) if args.workspace else DEFAULT_WORKSPACE
    daemon = ContinuousModeDaemon(workspace=workspace)
    canonical = daemon.current_canonical_harness()
    print("# autobench continuous-mode status")
    print(f"workspace: {workspace}")
    print()
    print("## canonical harness")
    print(json.dumps(_serialise_harness(canonical), indent=2))
    print()

    # Recent stats (last 10)
    print("## recent sessions (last 10)")
    if daemon.stats_path.is_file():
        rows = [line for line in daemon.stats_path.read_text().splitlines() if line.strip()]
        for line in rows[-10:]:
            try:
                rec = json.loads(line)
                print(
                    f"  {rec.get('timestamp', '?')}  "
                    f"score {rec.get('initial_score', 0):.3f} → {rec.get('final_score', 0):.3f}  "
                    f"{'[PROMOTED]' if rec.get('promoted') else ''} "
                    f"{rec.get('error', '')}"
                )
            except Exception:
                continue
        if not rows:
            print("  (no sessions recorded yet)")
    else:
        print("  (no stats file)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autobench.continuous",
        description="autobench continuous-mode daemon: autonomous 24/7 self-improvement.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=f"workspace directory (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=RECOMMENDED_RATE_MAX,
        help=(
            "rate-budget cap (sliding-window). Default 14000 — safe for "
            "MiniMax 15k/5h plan with ~7%% margin."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=RECOMMENDED_WINDOW_SECONDS,
        help="sliding-window length in seconds (default 18000 = 5h).",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.0,
        help="extra safety margin on top of --max-requests (default 0; max already includes margin).",
    )
    parser.add_argument(
        "--sessions-per-window",
        type=int,
        default=DEFAULT_SESSIONS_PER_WINDOW,
        help="target sessions per window (used for inter-session pacing).",
    )
    parser.add_argument(
        "--improver",
        default="minimax",
        choices=["minimax", "anthropic", "rule_based"],
        help="improver model to use (default minimax).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="RSI iteration cap per session (default 5).",
    )
    # nervous-bus-vwc8: Forge handshake bead binding. Falls back to
    # AUTOBENCH_BEAD_ID env when unset. When neither is supplied the daemon
    # runs unbound and the bench_completed channel stays silent.
    parser.add_argument(
        "--bead-id",
        dest="bead_id",
        default=None,
        help=(
            "Bead this daemon's canonical harness is attributed to. Required "
            "for the Forge handshake (bus.bead.bench_completed.v1). "
            f"Falls back to ${BEAD_ID_ENV}."
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=False)
    sub.add_parser("run", help="run the daemon (foreground, blocks).")
    sub.add_parser("once", help="run one session and exit (cron-friendly).")
    sub.add_parser("digest", help="generate and print today's digest.")
    sub.add_parser("status", help="print canonical harness + recent stats.")

    args = parser.parse_args(argv)
    cmd = args.cmd or "status"  # default to status for `python -m autobench.continuous`

    dispatch = {
        "run": _cmd_run,
        "once": _cmd_once,
        "digest": _cmd_digest,
        "status": _cmd_status,
    }
    return dispatch[cmd](args)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("AUTOBENCH_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    raise SystemExit(main())
