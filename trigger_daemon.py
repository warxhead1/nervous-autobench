"""Event-driven autobench trigger daemon (nervous-bus-1hlf).

Long-running listener that subscribes to ``autobench.cycle.requested.v1`` via
``deer obs bus`` and spawns one in-process cycle per validated trigger.

Architecture::

    deer obs bus --channel=autobench.cycle.requested.v1   (stdout JSONL)
        |
        v
    TriggerDaemon.listen() — read line-by-line
        |
        v
    handle_trigger(event)   — pure dict -> dict
        - validate event against schema
        - build cycle config (overrides + defaults)
        - run PopulationRunner in-process
        - distill report
        - emit autobench.cycle.report.v1
        - return report dict

Tests use ``handle_trigger`` directly and pass a ``mock_subprocess`` to
``listen`` so no real ``deer`` binary is needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .observability import (
    CHANNEL_CYCLE_REPORT,
    CHANNEL_CYCLE_REQUESTED,
    AutobenchObservability,
    _iso_now,
    _ulid,
)


__all__ = ["TriggerDaemon", "build_cycle_config", "CycleConfig"]

# --------------------------------------------------------------------------- #
# autobench.command.v1 — control-plane channel consumed by TriggerDaemon.
# --------------------------------------------------------------------------- #

CHANNEL_COMMAND = "autobench.command.v1"
CHANNEL_COMMAND_ACK = "autobench.command.acknowledged.v1"

# Runtime-config file written atomically when set_budget / set_generations
# commands land. Active kernel runs can poll this file periodically to pick
# up overrides without restart. Fields:
#   {"paused": bool, "max_requests": int|null, "max_cost_usd": float|null,
#    "target_generations": int|null}
_RUNTIME_CONFIG_PATH = (
    Path.home() / ".config" / "nervous-bus" / "autobench-runtime.json"
)

# Module-level pause flag. Kernel enforcement is out of scope here; the flag
# is set/cleared so callers can check ``is_autobench_paused()``.
_PAUSED: bool = False


def is_autobench_paused() -> bool:
    """Return True when a 'pause' command has been received and not yet resumed."""
    return _PAUSED


# --------------------------------------------------------------------------- #
# Defaults (mirror env-var defaults used by run_first.py)
# --------------------------------------------------------------------------- #

DEFAULT_N_ADVOCATES = 3
DEFAULT_MAX_ITER = 5
DEFAULT_BUDGET_SECONDS = 1800.0
DEFAULT_ADVERSARIAL_RATIO = 0.0
DEFAULT_TARGET_SKILL = 0.5


@dataclass
class CycleConfig:
    """Resolved cycle configuration built from a trigger event."""

    correlation_id: str
    requested_by: str
    domain: str
    bead_id: str | None = None
    n_advocates: int = DEFAULT_N_ADVOCATES
    max_iter: int = DEFAULT_MAX_ITER
    budget_seconds: float = DEFAULT_BUDGET_SECONDS
    target_skill: float | None = None
    adversarial_ratio: float = DEFAULT_ADVERSARIAL_RATIO
    judges_per_case: int | None = None
    improver_strategy: str | None = None
    notes: str = ""


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    """Return the ``data`` block of an envelope (or the raw event)."""
    d = event.get("data")
    if isinstance(d, dict):
        return d
    return event


def _validate_trigger(event: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """Validate one trigger event against the requested schema.

    Returns ``(ok, error_message, data_block)``. ``ok=False`` callers should
    emit a failure report rather than running the cycle. When the
    ``jsonschema`` package is unavailable the function returns
    ``(True, "", data)`` since the bus contract is permissive on
    optional dependency presence; the cycle-config builder still defends
    against missing required fields.
    """
    data = _event_data(event)

    # Defend against the obvious "required missing" cases before reaching
    # jsonschema, so the daemon still works without jsonschema installed.
    required = ("correlation_id", "requested_by", "domain")
    missing = [k for k in required if not str(data.get(k, "") or "").strip()]
    if missing:
        return False, f"missing required field(s): {','.join(missing)}", data
    if not isinstance(data.get("correlation_id"), str) or len(data["correlation_id"]) != 26:
        return False, "correlation_id must be a 26-char ULID", data

    try:
        import jsonschema  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return True, "", data
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / f"{CHANNEL_CYCLE_REQUESTED}.json"
    )
    if not schema_path.is_file():
        return True, "", data
    try:
        schema = json.loads(schema_path.read_text())
        data_schema = schema.get("properties", {}).get("data", {})
        if not data_schema:
            return True, "", data
        validator = jsonschema.Draft202012Validator(data_schema)
        errs = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if errs:
            msg = "; ".join(f"{list(e.path)}: {e.message}" for e in errs[:3])
            return False, msg, data
        return True, "", data
    except Exception as e:  # noqa: BLE001
        return True, f"validator setup failed: {e}", data


def build_cycle_config(event_data: dict[str, Any]) -> CycleConfig:
    """Build a CycleConfig from a (validated) trigger data dict.

    Missing overrides default to autobench's env-var defaults. Caller is
    responsible for prior validation; this helper coerces types defensively
    so a partially-malformed dict doesn't crash the builder.
    """
    def _int(key: str, default: int) -> int:
        v = event_data.get(key, default)
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def _float(key: str, default: float) -> float:
        v = event_data.get(key, default)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def _opt_str(key: str) -> str | None:
        v = event_data.get(key)
        if v is None:
            return None
        return str(v)

    return CycleConfig(
        correlation_id=str(event_data.get("correlation_id", "") or ""),
        requested_by=str(event_data.get("requested_by", "") or "operator"),
        domain=str(event_data.get("domain", "") or ""),
        bead_id=_opt_str("bead_id"),
        n_advocates=_int("n_advocates", DEFAULT_N_ADVOCATES),
        max_iter=_int("max_iter", DEFAULT_MAX_ITER),
        budget_seconds=_float("budget_seconds", DEFAULT_BUDGET_SECONDS),
        target_skill=(
            _float("target_skill", DEFAULT_TARGET_SKILL)
            if event_data.get("target_skill") is not None
            else None
        ),
        adversarial_ratio=_float("adversarial_ratio", DEFAULT_ADVERSARIAL_RATIO),
        judges_per_case=(
            _int("judges_per_case", 0)
            if event_data.get("judges_per_case") is not None
            else None
        ),
        improver_strategy=_opt_str("improver_strategy"),
        notes=str(event_data.get("notes", "") or "")[:500],
    )


class TriggerDaemon:
    """Subscribe to cycle.requested events and run one cycle per trigger.

    Args:
        bead_id_default: Tracker bead anchor applied when a trigger lacks
            its own ``bead_id`` (so unattributed operator triggers still
            close the Forge handshake when promotion accepts).
        runner_factory: Optional callable returning an object exposing
            ``run(config: CycleConfig) -> tuple[dict, dict, str, str]``
            where the return is ``(events, summary_overrides,
            started_at, completed_at)``. Used by tests to inject a
            deterministic stand-in for PopulationRunner. When None the
            real :func:`_run_cycle_with_population_runner` is used.
        obs: Optional observability instance for emission (cycle.report,
            cycle.requested). When None one is constructed.
        debug_file: Optional override for the obs debug-file path.
    """

    def __init__(
        self,
        bead_id_default: str | None = None,
        runner_factory: Callable[..., Any] | None = None,
        obs: AutobenchObservability | None = None,
        debug_file: Path | None = None,
    ) -> None:
        self.bead_id_default = bead_id_default or os.environ.get(
            "AUTOBENCH_BEAD_ID"
        )
        self.runner_factory = runner_factory
        self.obs = obs or AutobenchObservability(debug_file=debug_file)
        self.runs_handled = 0

    # ------------------------------------------------------------------ #
    # Pure handler (test surface)
    # ------------------------------------------------------------------ #

    def handle_trigger(self, event: dict[str, Any]) -> dict[str, Any]:
        """Validate, run the cycle, distill, emit, return the report.

        Never raises. On validation failure emits a degenerate report
        with ``summary.promoted=False`` and a failure note in
        ``patterns.notes`` rather than running the cycle.
        """
        from .distillation import CycleDistiller

        ok, err, data = _validate_trigger(event)
        if not ok:
            return self._emit_failure_report(
                data=data,
                error_msg=err,
            )

        config = build_cycle_config(data)
        # Apply the daemon's default bead_id when the trigger omitted one.
        if not config.bead_id and self.bead_id_default:
            config.bead_id = self.bead_id_default

        started_at = _iso_now()
        try:
            events, summary_overrides, started_at_run, completed_at_run = self._run_cycle(
                config=config,
            )
            started_at = started_at_run or started_at
            completed_at = completed_at_run or _iso_now()
        except Exception as exc:  # noqa: BLE001 — handle_trigger never raises
            return self._emit_failure_report(
                data=data,
                error_msg=f"cycle execution failed: {type(exc).__name__}: {exc}",
                started_at=started_at,
                completed_at=_iso_now(),
            )

        distiller = CycleDistiller()
        cycle_id = summary_overrides.get("cycle_id") or _ulid()
        report = distiller.distill_from_events(
            events=events,
            cycle_id=str(cycle_id),
            domain=config.domain,
            requested_by=config.requested_by,
            correlation_id=config.correlation_id,
            started_at=started_at,
            completed_at=completed_at,
            bead_id=config.bead_id,
            n_advocates_hint=summary_overrides.get(
                "n_advocates_hint", config.n_advocates
            ),
            n_cases_hint=summary_overrides.get("n_cases_hint"),
            baseline_score=summary_overrides.get("baseline_score"),
        )
        try:
            self.obs.cycle_report(report)
        except Exception as e:  # noqa: BLE001
            print(f"[trigger_daemon] cycle_report emit failed: {e}", file=sys.stderr)

        # Emit bus.bead.bench_completed.v1 when this trigger is bead-attributed.
        # Deer-flow's Forge bus_consumer subscribes to this channel to stamp the
        # `bench_delta` seal — cycle.report.v1 alone is not enough because the
        # Forge doesn't subscribe to it.
        if config.bead_id:
            try:
                summary = report.get("summary", {})
                baseline_metric = float(summary.get("aggregate_score_baseline", 0.0))
                treatment_metric = float(summary.get("aggregate_score_best", 0.0))
                delta = treatment_metric - baseline_metric
                n = int(summary.get("n_cases", 0))
                passes_threshold = bool(summary.get("promoted", False))
                self.obs.bench_completed_promotion(
                    bead_id=config.bead_id,
                    baseline_metric=baseline_metric,
                    treatment_metric=treatment_metric,
                    delta=delta,
                    n=max(1, n),
                    passes_threshold=passes_threshold,
                    ci_lower=None,
                    ci_upper=None,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[trigger_daemon] bench_completed_promotion emit failed: {e}",
                    file=sys.stderr,
                )

        self.runs_handled += 1
        return report

    # ------------------------------------------------------------------ #
    # Command handler — autobench.command.v1
    # ------------------------------------------------------------------ #

    def handle_command(self, event: dict[str, Any]) -> None:
        """Process one autobench.command.v1 event.

        Handles: pause, resume, set_budget, set_generations.
        Always emits autobench.command.acknowledged.v1 after handling.
        Never raises — command handling must not crash the daemon.
        """
        global _PAUSED  # noqa: PLW0603 — module-level pause flag

        data = _event_data(event)
        command_id = str(
            data.get("command_id") or event.get("id") or _ulid()
        )
        action = str(data.get("action") or "")
        params = data.get("params") or {}
        status = "acknowledged"
        error_message: str | None = None

        try:
            if action == "pause":
                _PAUSED = True
                print(
                    f"[trigger_daemon] pause requested (command_id={command_id})",
                    file=sys.stderr,
                )
                self._update_runtime_config()
            elif action == "resume":
                _PAUSED = False
                print(
                    f"[trigger_daemon] resume requested (command_id={command_id})",
                    file=sys.stderr,
                )
                self._update_runtime_config()
            elif action == "set_budget":
                max_requests = params.get("max_requests")
                max_cost_usd = params.get("max_cost_usd")
                print(
                    f"[trigger_daemon] set_budget — max_requests={max_requests}, "
                    f"max_cost_usd={max_cost_usd} (command_id={command_id})",
                    file=sys.stderr,
                )
                self._update_runtime_config(
                    max_requests=(
                        int(max_requests) if max_requests is not None else None
                    ),
                    max_cost_usd=(
                        float(max_cost_usd) if max_cost_usd is not None else None
                    ),
                )
            elif action == "set_generations":
                target_gen = params.get("target_generations")
                print(
                    f"[trigger_daemon] set_generations — target_generations={target_gen}"
                    f" (command_id={command_id})",
                    file=sys.stderr,
                )
                self._update_runtime_config(
                    target_generations=(
                        int(target_gen) if target_gen is not None else None
                    ),
                )
            else:
                # Unknown / unimplemented action — log and ack with error.
                error_message = f"unhandled action: {action!r}"
                status = "error"
                print(
                    f"[trigger_daemon] unhandled command action {action!r} "
                    f"(command_id={command_id})",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001 — never propagate from command handler
            status = "error"
            error_message = f"{type(exc).__name__}: {exc}"
            print(
                f"[trigger_daemon] command handler error: {error_message}",
                file=sys.stderr,
            )

        # Emit acknowledgement back to the bus.
        try:
            ack_data: dict[str, Any] = {
                "command_id": command_id,
                "action": action,
                "status": status,
            }
            if error_message is not None:
                ack_data["error_message"] = error_message
            self.obs._publish(CHANNEL_COMMAND_ACK, ack_data)
        except Exception as e:  # noqa: BLE001
            print(
                f"[trigger_daemon] command ack emit failed: {e}", file=sys.stderr
            )

    @staticmethod
    def _update_runtime_config(
        max_requests: int | None = None,
        max_cost_usd: float | None = None,
        target_generations: int | None = None,
    ) -> None:
        """Atomically merge updates into the autobench runtime-config JSON.

        Reads the current config (if any), applies the non-None overrides,
        and writes back atomically via rename. Any running kernel can poll
        this file to pick up new budget/generation limits without restart.
        """
        path = _RUNTIME_CONFIG_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Read-modify-write: preserve any fields we're not updating.
            cfg: dict[str, Any] = {
                "paused": _PAUSED,
                "max_requests": None,
                "max_cost_usd": None,
                "target_generations": None,
            }
            if path.is_file():
                try:
                    cfg.update(json.loads(path.read_text()))
                except Exception:  # noqa: BLE001 — stale/corrupt file: ignore
                    pass
            # Sync paused flag in case it drifted.
            cfg["paused"] = _PAUSED
            if max_requests is not None:
                cfg["max_requests"] = max_requests
            if max_cost_usd is not None:
                cfg["max_cost_usd"] = max_cost_usd
            if target_generations is not None:
                cfg["target_generations"] = target_generations
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cfg))
            os.replace(tmp, path)  # atomic on POSIX
        except Exception as e:  # noqa: BLE001 — config write failure is non-fatal
            print(
                f"[trigger_daemon] runtime-config write failed: {e}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------ #
    # Cycle runner — pluggable so tests can inject a stub
    # ------------------------------------------------------------------ #

    def _run_cycle(
        self,
        config: CycleConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
        """Run one cycle. Returns (events, overrides, started_at, completed_at).

        Calls ``runner_factory`` when provided; falls back to the real
        PopulationRunner-backed runner otherwise.
        """
        if self.runner_factory is not None:
            return self.runner_factory(config)
        return _run_cycle_with_population_runner(config, self.obs)

    # ------------------------------------------------------------------ #
    # Failure-report emitter
    # ------------------------------------------------------------------ #

    def _emit_failure_report(
        self,
        data: dict[str, Any],
        error_msg: str,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Emit a degenerate cycle.report with promoted=False + notes."""
        now = _iso_now()
        started = started_at or now
        completed = completed_at or now
        correlation_id = str(data.get("correlation_id", "") or "unknown")
        domain = str(data.get("domain", "") or "unknown")
        requested_by = str(data.get("requested_by", "") or "operator")
        report: dict[str, Any] = {
            "correlation_id": correlation_id,
            "cycle_id": _ulid(),
            "domain": domain,
            "requested_by": requested_by,
            "started_at": started,
            "completed_at": completed,
            "ts": completed,
            "summary": {
                "n_advocates": 0,
                "n_iterations": 0,
                "n_cases": 0,
                "promoted": False,
                "promoted_advocate_id": "",
                "aggregate_score_best": 0.0,
                "aggregate_score_baseline": 0.0,
                "ahe_outcomes": {
                    "confirmed": 0,
                    "partial": 0,
                    "refuted": 0,
                    "refuted_live": 0,
                    "none": 0,
                },
            },
            "patterns": {
                "top_failure_modes": [],
                "successful_deltas": [],
                "dissent_hotspots": [],
                "lineage_diversity": 0.0,
                "cross_domain_score": {},
                "notes": f"trigger validation failed: {error_msg}",
            },
            "cost": {
                "worker_calls": 0,
                "improver_calls": 0,
                "judge_calls": 0,
                "total_requests": 0,
            },
        }
        bead = str(data.get("bead_id", "") or "") or self.bead_id_default
        if bead:
            report["bead_id"] = str(bead)
        try:
            self.obs.cycle_report(report)
        except Exception as e:  # noqa: BLE001
            print(f"[trigger_daemon] failure cycle_report emit failed: {e}", file=sys.stderr)
        return report

    # ------------------------------------------------------------------ #
    # Listen loop
    # ------------------------------------------------------------------ #

    def listen(
        self,
        max_runs: int | None = None,
        mock_subprocess: Iterable[str] | None = None,
    ) -> int:
        """Blocking listen loop. Returns the number of triggers handled.

        Args:
            max_runs: When set, exit after handling N triggers. Tests use 1.
            mock_subprocess: When set, an iterable of JSONL strings to
                consume in place of spawning ``deer obs bus``. Each item
                is one event payload (one line, no trailing newline
                required). Used by tests to avoid spawning a real CLI.
        """
        if mock_subprocess is not None:
            return self._listen_iterable(mock_subprocess, max_runs=max_runs)

        # Subscription strategy: prefer tailing the canonical bus log
        # (~/.cache/nervous-bus/debug.jsonl) because that's filesystem state
        # we control and emits full CloudEvents JSONL — what `loomie-milestones`
        # already does. Falls back to `deer obs bus` only when AUTOBENCH_USE_DEER=1
        # is explicitly set (kept for future deer parity once `deer obs bus --json`
        # is fixed upstream — currently it emits "<ts> <type>" summary lines
        # rather than the full event payload).
        if os.environ.get("AUTOBENCH_USE_DEER") == "1":
            return self._listen_via_deer(max_runs=max_runs)
        return self._listen_via_debug_tail(max_runs=max_runs)

    def _listen_via_debug_tail(self, max_runs: int | None) -> int:
        """Tail the nervous-bus debug.jsonl and filter by type. Reliable path."""
        debug_path = Path(
            os.environ.get(
                "NERVOUS_BUS_DEBUG_FILE",
                str(Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"),
            )
        )
        # Skip any historical events — only react to triggers fired AFTER
        # the daemon started. Using `tail -F -n 0` matches the loomie-milestones
        # convention.
        cmd = ["tail", "-F", "-n", "0", str(debug_path)]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[trigger_daemon] failed to spawn tail on {debug_path}: {exc}",
                file=sys.stderr,
            )
            return 0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                # Quick pre-filter — skip lines that don't carry a handled type.
                if CHANNEL_CYCLE_REQUESTED not in line and CHANNEL_COMMAND not in line:
                    continue
                handled = self._consume_line(line)
                if handled and max_runs is not None and self.runs_handled >= max_runs:
                    break
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        return self.runs_handled

    def _listen_via_deer(self, max_runs: int | None) -> int:
        """Subscribe via `deer obs bus`. Currently broken (see _listen)."""
        import shutil

        if shutil.which("deer") is None:
            print(
                "[trigger_daemon] 'deer' CLI not found on PATH — "
                "set AUTOBENCH_USE_DEER=0 (default) to use the debug.jsonl tail "
                "fallback; exiting cleanly.",
                file=sys.stderr,
            )
            return 0
        cmd = ["deer", "obs", "bus", f"--type={CHANNEL_CYCLE_REQUESTED}", "--json"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[trigger_daemon] failed to spawn deer obs bus: {exc}",
                  file=sys.stderr)
            return 0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                handled = self._consume_line(line)
                if handled and max_runs is not None and self.runs_handled >= max_runs:
                    break
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        return self.runs_handled

    def _listen_iterable(
        self,
        lines: Iterable[str],
        max_runs: int | None = None,
    ) -> int:
        for line in lines:
            handled = self._consume_line(line)
            if handled and max_runs is not None and self.runs_handled >= max_runs:
                break
        return self.runs_handled

    def _consume_line(self, line: str) -> bool:
        """Parse one JSONL line and dispatch by event type. Returns True on dispatch."""
        line = line.strip()
        if not line:
            return False
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(
                f"[trigger_daemon] dropping malformed JSON line: {line[:120]!r}",
                file=sys.stderr,
            )
            return False
        if not isinstance(event, dict):
            return False
        event_type = event.get("type", "")
        if event_type == CHANNEL_COMMAND:
            self.handle_command(event)
            return True
        # Default: treat all other matching events as cycle triggers.
        self.handle_trigger(event)
        return True


# --------------------------------------------------------------------------- #
# Real cycle runner (PopulationRunner-backed)
# --------------------------------------------------------------------------- #

def _run_cycle_with_population_runner(
    config: CycleConfig,
    obs: AutobenchObservability,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    """Run a cycle using the production PopulationRunner.

    This path is intentionally minimal — it constructs a small fixed
    benchmark from the requested domain via the BenchmarkRegistry, spins
    up a PopulationRunner, and returns ``(events, overrides, started, ended)``.
    Heavy production runs typically still launch via
    ``autobench.benchmarks.<domain>.run_first`` — this helper exists so the
    trigger daemon can run lightweight cycles in-process without depending
    on any specific benchmark's main(). Events are read from the obs's
    debug-file fallback after the cycle completes.

    On error returns an empty event list and zero-iteration overrides so
    the distiller produces a benign report.
    """
    from .benchmark_registry import BenchmarkRegistry
    from .core import ContextManager, HarnessConfig, RolloutProtocol
    from .evaluator import BenchmarkEvaluator
    from .population import PopulationRunner

    started_at = _iso_now()
    try:
        registry = BenchmarkRegistry.default()
        cases_by_domain = registry.load_all_cases()
    except Exception:  # noqa: BLE001
        cases_by_domain = {}

    cases = cases_by_domain.get(config.domain) or []
    if not cases:
        # No cases available for the requested domain — return an empty
        # event set with zero-iteration overrides so the distiller emits
        # a clean "no-op" report.
        return (
            [],
            {
                "cycle_id": _ulid(),
                "n_advocates_hint": 0,
                "n_cases_hint": 0,
            },
            started_at,
            _iso_now(),
        )

    # Construct a minimal harness factory + evaluator factory. Real runs
    # would wire a worker; the daemon's in-process cycle is intentionally
    # spartan so it stays test-runnable without external API keys.
    def _harness_factory() -> HarnessConfig:
        return HarnessConfig(
            system_prompt="You are an autobench worker.",
            rollout_protocol=RolloutProtocol.SINGLE,
            context_manager=ContextManager.FULL,
            tool_surface="",
            verifiers=[],
            budget={
                "max_tokens": 2048,
                "max_time_seconds": 10,
                "max_cost_dollars": 0.0,
                "max_memory_mb": 256,
            },
        )

    def _evaluator_factory() -> BenchmarkEvaluator:
        return BenchmarkEvaluator(obs=obs)

    improver = getattr(config, "improver_strategy", None) or os.environ.get(
        "AUTOBENCH_DEFAULT_IMPROVER", "minimax"
    )
    runner = PopulationRunner(
        n_advocates=max(1, int(config.n_advocates)),
        initial_harness_factory=_harness_factory,
        evaluator_factory=_evaluator_factory,
        observability_factory=lambda: obs,
        improver=improver,
        max_iterations_per_advocate=max(1, int(config.max_iter)),
        budget_per_advocate_seconds=float(config.budget_seconds),
        improvement_threshold=0.02,
        adversarial_ratio=float(config.adversarial_ratio),
        registry=registry,
    )

    try:
        result = runner.run(cases)
        cycle_id = getattr(result, "cycle_id", "") or _ulid()
        n_advocates = len(getattr(result, "advocates", []) or [])
    except Exception:  # noqa: BLE001 — surface as empty events to handle_trigger
        cycle_id = _ulid()
        n_advocates = config.n_advocates
    completed_at = _iso_now()

    events = _read_debug_events(obs)
    return (
        events,
        {
            "cycle_id": cycle_id,
            "n_advocates_hint": n_advocates,
            "n_cases_hint": len(cases),
        },
        started_at,
        completed_at,
    )


def _read_debug_events(obs: AutobenchObservability) -> list[dict[str, Any]]:
    """Read all events from the obs debug-file fallback. Best-effort."""
    path = getattr(obs, "_debug_file", None)
    events: list[dict[str, Any]] = []
    if not path:
        return events
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    except Exception:  # noqa: BLE001
        return []
    return events


# --------------------------------------------------------------------------- #
# CLI entrypoint — runs the daemon as a subscriber.
#
# Usage:
#   python -m autobench.trigger_daemon              # listen forever
#   python -m autobench.trigger_daemon --max-runs 1 # exit after first trigger
#
# Pairs with the substrate-proof demo:
#   nervous publish autobench.cycle.requested.v1 '{"domain":"codeforces_tier1",...}'
#
# The daemon spawns ``deer obs bus --type=autobench.cycle.requested.v1 --json``
# under the hood and emits autobench.cycle.report.v1 after each handled
# trigger so downstream consumers (Forge bead-attachment, hearth-loom AC
# enrichment) get the full report on the bus.
# --------------------------------------------------------------------------- #

def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="autobench.trigger_daemon",
        description="Listen for autobench.cycle.requested.v1 events and run a "
                    "cycle per trigger, emitting autobench.cycle.report.v1 to "
                    "the nervous-bus.",
    )
    p.add_argument("--max-runs", type=int, default=None,
                   help="Exit after handling N triggers. Default: unlimited.")
    p.add_argument("--bead-id", default=None,
                   help="Anchor bead for triggers that don't carry their own. "
                        "Reads AUTOBENCH_BEAD_ID env if unset.")
    p.add_argument("--debug-file", default=None,
                   help="Override the obs debug-file path. Default: "
                        "~/.cache/nervous-bus/debug.jsonl (set by observability).")
    args = p.parse_args()

    daemon = TriggerDaemon(
        bead_id_default=args.bead_id,
        debug_file=Path(args.debug_file) if args.debug_file else None,
    )
    print(
        f"[trigger_daemon] starting — listening on {CHANNEL_CYCLE_REQUESTED}"
        + (f", will exit after {args.max_runs} run(s)" if args.max_runs else "")
        + ".",
        file=sys.stderr,
    )
    handled = daemon.listen(max_runs=args.max_runs)
    print(f"[trigger_daemon] exiting — handled {handled} trigger(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
