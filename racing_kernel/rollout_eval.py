"""GPU-queue rollout evaluator for the racing FunSearch kernel.

Admission-gated GPU rollout path for scoring candidate racing controllers via
the svdag_racing silo.  The CPU oracle (oracle.evaluate_on_instance) remains the
DEFAULT; GPU rollout is opt-in via ``enabled=True``.

Flow:
  1. Call GPUAdmissionGate.should_submit(island) — on DEFER, fall back to the
     CPU oracle immediately.
  2. Build a _RacingGPUJob with a pre-minted job_id / correlation_id so
     we can correlate the result without re-reading the publisher's internal state.
  3. Publish via GPUJobPublisher (and record_submitted on success).
  4. Poll nbus:autobench.gpu_result.v1 via redis-cli XREVRANGE until the matching
     result arrives or timeout expires.
  5. Derive fitness from the result fields:
       - status == "timed_out" OR verdict in {TLE, RE, CE, VF} → 0.0
       - Otherwise extract lap_time_ms from silo_tester_report if present,
         then compute speed_score against ref_lap_time_s.
       - collision_free (silo_tester_report.collisions == 0) → track bonus.
  6. On any failure / timeout → record_completed + CPU oracle fallback.

Controller-in-engine coupling:
  The racing kernel evolves a PYTHON policy function that cannot yet execute
  inside the svdag_racing silo — the engine-side distill/deploy path is tracked
  in nervous-bus-71cn.6 (distill to artifact) and 71cn.7 (deploy to silo).
  Until those land, the shader_artifact_path field is a PLACEHOLDER (the policy
  source is written to a temp file so the job is schema-valid and enqueueable,
  but the engine does not execute it in a racing loop).  A WARNING is logged on
  every GPU rollout attempt to make this coupling visible.

  FULL GPU ROLLOUT will be available once:
    - nervous-bus-71cn.6: controller distillation → shader artifact
    - nervous-bus-71cn.7: artifact deploy to svdag_racing silo

ref_lap_time calibration from tengine.race.episode.v1:
  calibrate_ref_lap_time(instance, redis_bin) reads nbus:tengine.race.episode.v1
  and returns the minimum completed lap_time_ms (converted to seconds) whose
  track_id matches instance.name or track_seed matches instance.seed.  Falls
  back to instance.ref_lap_time_s if no qualifying episodes are found.

Public surface:
  evaluate_via_rollout(code, instance, *, island, enabled, publisher, gate, ...)
      → float | None
  calibrate_ref_lap_time(instance, redis_bin=None) → float
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from autobench.bus.gpu_admission import GPUAdmissionGate, SubmitDecision
from autobench.bus.gpu_publisher import GPUJobPublisher
from autobench.bus.gpu_types import GPUJob
from autobench.bus.idgen import ulid, iso_now
from autobench.racing_kernel.instance import RacingInstance
from autobench.racing_kernel.oracle import evaluate_on_instance, _speed_score

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SILO_ID = "svdag_racing"
RESULT_STREAM = "nbus:autobench.gpu_result.v1"
EPISODE_STREAM = "nbus:tengine.race.episode.v1"

# Maximum number of XREVRANGE entries to scan per poll tick
_XREVRANGE_SCAN_COUNT = 50

# Polling interval (seconds) between redis-cli XREVRANGE polls
_POLL_INTERVAL_S = 0.5

# Fallback fitness on timed-out / failed / unrecoverable GPU result
_FAILURE_FITNESS = 0.0

# Fitness weight for track (collision) term when scoring from GPU result
# (v1 simplification: only lap_time + collision_free available from silo report)
_W_SPEED_GPU = 0.70
_W_TRACK_GPU = 0.30

# ── Coupling warning (emit once per process) ──────────────────────────────────

_COUPLING_WARNING_EMITTED = False


def _emit_coupling_warning() -> None:
    global _COUPLING_WARNING_EMITTED
    if not _COUPLING_WARNING_EMITTED:
        logger.warning(
            "GPU rollout v1: in-engine controller execution requires "
            "nervous-bus-71cn.6 (distill controller → shader artifact) and "
            "nervous-bus-71cn.7 (deploy artifact to svdag_racing silo). "
            "This rollout path enqueues a PLACEHOLDER controller reference; "
            "the engine does NOT yet execute the Python policy in a racing loop. "
            "CPU oracle remains the authoritative scorer until 71cn.6/71cn.7 land."
        )
        _COUPLING_WARNING_EMITTED = True


# ── Extended GPUJob dataclass (local, not modifying gpu_types.py) ─────────────

@dataclass
class _RacingGPUJob(GPUJob):
    """GPUJob extended with pre-minted queue-correlation fields.

    We pre-mint job_id / correlation_id so the caller retains them for result
    matching.  GPUJobPublisher.publish calls ``data.setdefault("job_id", ulid())``
    which no-ops when job_id is already set — so the IDs we mint here are the
    ones that appear in the Redis stream.
    """
    job_id: str = field(default_factory=ulid)
    correlation_id: str = field(default_factory=ulid)
    source_island: str = ""
    track_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "job_id": self.job_id,
            "correlation_id": self.correlation_id,
            "source_island": self.source_island,
            "track_seed": self.track_seed,
        })
        return base


# ── Redis XREVRANGE helpers (same pattern as gpu_admission._parse_xrevrange_lines)

def _redis_bin() -> str | None:
    """Locate redis-cli on PATH."""
    return shutil.which("redis-cli")


def _xrevrange_entries(
    stream: str,
    count: int,
    redis_bin: str | None,
) -> list[dict[str, Any]]:
    """Fetch up to *count* most-recent entries from *stream* via XREVRANGE.

    Returns a list of parsed entry dicts (key "data" → parsed JSON dict).
    Returns [] on any failure.
    """
    if redis_bin is None:
        return []
    try:
        proc = subprocess.run(
            [redis_bin, "XREVRANGE", stream, "+", "-", "COUNT", str(count)],
            capture_output=True,
            timeout=3,
        )
    except Exception as exc:
        logger.debug("rollout_eval: XREVRANGE %s failed: %s", stream, exc)
        return []

    if proc.returncode != 0:
        logger.debug(
            "rollout_eval: XREVRANGE %s exit %d: %s",
            stream, proc.returncode,
            proc.stderr.decode(errors="replace").strip()[:200],
        )
        return []

    return _parse_xrevrange_output(proc.stdout.decode(errors="replace"))


def _parse_xrevrange_output(raw: str) -> list[dict[str, Any]]:
    """Parse flat redis-cli XREVRANGE output into a list of data dicts.

    Each Redis entry is: <id>\\n[field_count]\\nkey\\nvalue\\n...
    We split entries by detecting lines that look like Redis stream entry IDs
    (contain '-') and collect key/value pairs after each ID line.
    """
    lines = [l for l in raw.splitlines() if l.strip()]
    if not lines:
        return []

    entries: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        # An entry ID line contains exactly one '-' and looks like "NNN-NNN"
        line = lines[i].strip()
        if "-" in line and not line.startswith(" "):
            # Start of new entry
            i += 1
            # Skip optional numeric field-count line
            if i < len(lines) and lines[i].strip().isdigit():
                i += 1
            # Collect k/v pairs
            kv: dict[str, str] = {}
            while i + 1 < len(lines):
                # Stop at next entry ID (contains '-' and not indented)
                nxt = lines[i].strip()
                if "-" in nxt and not lines[i].startswith(" "):
                    break
                key = lines[i].strip()
                val = lines[i + 1].strip()
                kv[key] = val
                i += 2
            raw_data = kv.get("data")
            if raw_data:
                try:
                    entries.append(json.loads(raw_data))
                except json.JSONDecodeError:
                    pass
        else:
            i += 1

    return entries


# ── ref_lap_time calibration from tengine.race.episode.v1 ─────────────────────

def calibrate_ref_lap_time(
    instance: RacingInstance,
    redis_bin: str | None = None,
) -> float:
    """Read recent tengine.race.episode.v1 events and return the best lap time (s).

    Matches entries where:
      - ``outcome == "completed"``
      - ``track_id == instance.name`` OR ``track_seed == instance.seed``
      - ``lap_time_ms`` is a positive number

    Returns the minimum qualifying lap_time_ms / 1000.0, or
    ``instance.ref_lap_time_s`` if no qualifying episodes are found.

    Args:
        instance:  The RacingInstance to calibrate for.
        redis_bin: Path to redis-cli; None → locate via PATH.

    Returns:
        Calibrated reference lap time in seconds.
    """
    if redis_bin is None:
        redis_bin = _redis_bin()

    entries = _xrevrange_entries(EPISODE_STREAM, _XREVRANGE_SCAN_COUNT, redis_bin)
    if not entries:
        logger.debug(
            "rollout_eval: no tengine.race.episode entries found; "
            "using synthetic ref_lap_time_s=%.2f for track '%s'",
            instance.ref_lap_time_s, instance.name,
        )
        return instance.ref_lap_time_s

    best_ms: float | None = None
    for data in entries:
        if data.get("outcome") != "completed":
            continue
        # Match by track_id (name) OR track_seed
        if (
            data.get("track_id") != instance.name
            and data.get("track_seed") != instance.seed
        ):
            continue
        try:
            ms = float(data["lap_time_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if ms > 0:
            if best_ms is None or ms < best_ms:
                best_ms = ms

    if best_ms is not None:
        calibrated = best_ms / 1000.0
        logger.info(
            "rollout_eval: calibrated ref_lap_time for '%s' from episode bus: "
            "%.3f s (was synthetic %.3f s)",
            instance.name, calibrated, instance.ref_lap_time_s,
        )
        return calibrated

    logger.debug(
        "rollout_eval: no qualifying episodes for track '%s'; "
        "keeping synthetic ref_lap_time_s=%.2f",
        instance.name, instance.ref_lap_time_s,
    )
    return instance.ref_lap_time_s


# ── GPU result polling ────────────────────────────────────────────────────────

def _poll_for_result(
    job_id: str,
    correlation_id: str,
    timeout_s: float,
    redis_bin: str | None,
) -> dict[str, Any] | None:
    """Poll nbus:autobench.gpu_result.v1 until the matching result arrives.

    Matches by job_id first, then correlation_id.  Polls every _POLL_INTERVAL_S
    seconds until the result is found or *timeout_s* elapses.

    Returns:
        The parsed ``data`` dict from the matching result event, or None on
        timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        entries = _xrevrange_entries(RESULT_STREAM, _XREVRANGE_SCAN_COUNT, redis_bin)
        for data in entries:
            if data.get("job_id") == job_id:
                logger.debug("rollout_eval: result matched by job_id=%s", job_id)
                return data
            if data.get("correlation_id") == correlation_id:
                logger.debug(
                    "rollout_eval: result matched by correlation_id=%s", correlation_id
                )
                return data
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_POLL_INTERVAL_S, remaining))

    logger.debug(
        "rollout_eval: no result for job_id=%s within %.1f s", job_id, timeout_s
    )
    return None


# ── Fitness derivation from gpu_result.v1 ────────────────────────────────────

def _fitness_from_result(
    result: dict[str, Any],
    instance: RacingInstance,
    ref_lap_time_s: float,
) -> float:
    """Derive a fitness value in [0, 1] from a gpu_result.v1 data dict.

    v1 simplification:
      - Only ``lap_time_ms`` (from silo_tester_report) and collision counts are
        used; smoothness is not available from the engine result in v1.
      - Speed term (weight 0.70) is computed from lap_time_s vs ref_lap_time_s
        using the same _speed_score function as the CPU oracle.
      - Track term (weight 0.30) is binary: 1.0 if no collisions are reported,
        0.0 otherwise.

    Low-fitness conditions (→ 0.0):
      - status == "timed_out" or "failed"
      - verdict in {TLE, RE, CE, VF}
    """
    status = result.get("status", "")
    verdict = result.get("verdict", "")

    if status in ("timed_out", "failed"):
        logger.debug(
            "rollout_eval: low fitness: status=%s verdict=%s", status, verdict
        )
        return _FAILURE_FITNESS

    if verdict in ("TLE", "RE", "CE", "VF"):
        logger.debug(
            "rollout_eval: low fitness: verdict=%s", verdict
        )
        return _FAILURE_FITNESS

    # Extract lap_time_ms from silo_tester_report or top-level if available
    silo_report: dict[str, Any] = result.get("silo_tester_report") or {}
    lap_time_ms: float | None = None

    for candidate_key in ("lap_time_ms", "best_lap_time_ms", "lap_ms"):
        raw = silo_report.get(candidate_key)
        if raw is not None:
            try:
                lap_time_ms = float(raw)
                break
            except (TypeError, ValueError):
                pass

    # Also try top-level latency_ms as a coarse proxy if silo_report has nothing
    if lap_time_ms is None:
        latency = result.get("latency_ms")
        if latency is not None:
            try:
                lap_time_ms = float(latency)
                logger.debug(
                    "rollout_eval: using latency_ms=%.1f as lap_time proxy "
                    "(no lap_time_ms in silo_tester_report)",
                    lap_time_ms,
                )
            except (TypeError, ValueError):
                pass

    if lap_time_ms is None or lap_time_ms <= 0:
        logger.debug("rollout_eval: no lap_time available; returning failure fitness")
        return _FAILURE_FITNESS

    lap_time_s = lap_time_ms / 1000.0
    speed = _speed_score(lap_time_s, ref_lap_time_s)

    # Collision-free bonus
    collisions = silo_report.get("collisions", 0)
    try:
        collision_free = 1.0 if int(collisions) == 0 else 0.0
    except (TypeError, ValueError):
        collision_free = 1.0  # unknown → give benefit of doubt

    fitness = _W_SPEED_GPU * speed + _W_TRACK_GPU * collision_free
    fitness = max(0.0, min(1.0, fitness))

    logger.info(
        "rollout_eval: GPU result → lap=%.3f s (ref=%.3f s) speed_score=%.3f "
        "collision_free=%.1f → fitness=%.4f",
        lap_time_s, ref_lap_time_s, speed, collision_free, fitness,
    )
    return fitness


# ── Placeholder controller reference ─────────────────────────────────────────

def _write_placeholder_artifact(code: str, job_id: str) -> str:
    """Write the policy source to a temp file and return its path.

    This is the PLACEHOLDER shader_artifact_path used until 71cn.6 (distill)
    produces a real engine-loadable artifact.  The path is schema-valid (a
    string) but the svdag_racing silo runner does not yet consume it.
    """
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, f"racing_controller_{job_id}.py")
    try:
        with open(path, "w") as fh:
            fh.write(code)
    except OSError as exc:
        logger.warning("rollout_eval: could not write placeholder artifact: %s", exc)
        path = f"/tmp/racing_controller_placeholder_{job_id}.py"
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate_via_rollout(
    code: str,
    instance: RacingInstance,
    *,
    island: int | str | None = None,
    enabled: bool = True,
    publisher: GPUJobPublisher | None = None,
    gate: GPUAdmissionGate | None = None,
    redis_bin: str | None = None,
    frames: int = 180,
    timeout_s: float = 120.0,
    ref_lap_time_s: float | None = None,
    # Injection hook for tests: replaces the real result-poll with a callable
    # result_fetcher(job_id, correlation_id, timeout_s, redis_bin) → dict|None
    _result_fetcher: Callable[..., dict[str, Any] | None] | None = None,
) -> float | None:
    """Score a racing controller via an admission-gated GPU rollout.

    This is an OPT-IN alternative to the CPU oracle.  Pass ``enabled=False``
    (the default in most callers) to route directly to the CPU oracle.

    Args:
        code:            The ``racing_line`` policy source string.
        instance:        RacingInstance describing the track.
        island:          Source island identifier for admission-gate quota.
        enabled:         If False, delegates immediately to the CPU oracle.
        publisher:       GPUJobPublisher to use (None → create a new one).
        gate:            GPUAdmissionGate to use (None → create a new one).
        redis_bin:       Path to redis-cli (None → locate via PATH).
        frames:          Number of frames to render in the GPU rollout.
        timeout_s:       Seconds to wait for a GPU result before falling back.
        ref_lap_time_s:  Reference lap time override (None → use
                         calibrate_ref_lap_time → instance.ref_lap_time_s).
        _result_fetcher: Test-injection hook replacing redis polling.

    Returns:
        Fitness in (0, 1], or None if the CPU oracle also fails to score.
    """
    if not enabled:
        logger.debug("rollout_eval: GPU rollout disabled; using CPU oracle")
        return evaluate_on_instance(code, instance)

    # Emit the coupling warning once so operators see it in the logs.
    _emit_coupling_warning()

    # Resolve optional collaborators
    if gate is None:
        gate = GPUAdmissionGate()
    if publisher is None:
        publisher = GPUJobPublisher()
    if redis_bin is None:
        redis_bin = _redis_bin()

    # ── Admission gate ────────────────────────────────────────────────────── #
    decision = gate.should_submit(island)
    if decision == SubmitDecision.DEFER:
        logger.debug(
            "rollout_eval: DEFER by admission gate (island=%s); CPU oracle fallback",
            island,
        )
        return evaluate_on_instance(code, instance)

    # ── Resolve reference lap time (calibrate from bus or use synthetic) ──── #
    if ref_lap_time_s is None:
        ref_lap_time_s = calibrate_ref_lap_time(instance, redis_bin)

    # ── Build job ─────────────────────────────────────────────────────────── #
    job_id = ulid()
    correlation_id = ulid()
    artifact_path = _write_placeholder_artifact(code, job_id)

    job = _RacingGPUJob(
        silo_id=SILO_ID,
        shader_artifact_path=artifact_path,
        case_id=f"{instance.name}_{instance.seed}",
        frames=frames,
        timeout_s=timeout_s,
        job_id=job_id,
        correlation_id=correlation_id,
        source_island=str(island) if island is not None else "",
        track_seed=instance.seed,
    )

    # ── Publish ───────────────────────────────────────────────────────────── #
    ok = publisher.publish(job)  # type: ignore[arg-type]
    if not ok:
        logger.warning(
            "rollout_eval: GPUJobPublisher.publish failed for job_id=%s; "
            "CPU oracle fallback",
            job_id,
        )
        return evaluate_on_instance(code, instance)

    gate.record_submitted(island)
    logger.info(
        "rollout_eval: enqueued GPU job job_id=%s correlation_id=%s "
        "track=%s island=%s frames=%d timeout=%.1f",
        job_id, correlation_id, instance.name, island, frames, timeout_s,
    )

    # ── Poll for result ───────────────────────────────────────────────────── #
    # Use injected fetcher in tests; real redis poll in production.
    fetcher = _result_fetcher or _poll_for_result
    result: dict[str, Any] | None
    try:
        result = fetcher(job_id, correlation_id, timeout_s, redis_bin)
    finally:
        # Always free the in-flight slot, success or not.
        gate.record_completed(island)

    if result is None:
        logger.warning(
            "rollout_eval: no GPU result for job_id=%s within %.1f s; "
            "CPU oracle fallback",
            job_id, timeout_s,
        )
        return evaluate_on_instance(code, instance)

    # ── Derive fitness ────────────────────────────────────────────────────── #
    fitness = _fitness_from_result(result, instance, ref_lap_time_s)
    return fitness
