"""racing_kernel/loop_feedback.py — close the ai_brain episode loop.

LOOP DESCRIPTION
================
1. **Deploy** (nervous-bus-71cn.7): a distilled brain is deployed as an enemy
   pod controller in svdag_racing_silo.  It drives laps at real physics speed.

2. **Episode bus**: each completed run emits ``tengine.race.episode.v1`` on
   ``nbus:tengine.race.episode.v1`` with ``pilot_kind="ai_brain"``.  The
   episode carries ``lap_time_ms``, ``outcome``, ``track_id``/``track_seed``,
   ``pilot_id`` (=``brain_id``), and an aggregate ``summary`` dict.

3. **This module** (nervous-bus-71cn.8): reads those episodes from Redis,
   filters to ``ai_brain`` + ``completed`` + positive ``lap_time_ms``, and
   produces two outputs for the *next* FunSearch evolution round:
     a. ``updated_ref_lap_time`` — per-track minimum ai_brain lap time (s).
        This tightens the oracle's physics ceiling so that evolution is always
        chasing the best *deployed* brain, not just the synthetic estimate.
     b. ``seed_candidates`` — list of ``brain_id`` strings whose episodes had
        the best lap times.  v1: empty list — episode events do not carry the
        source program string.  The downstream hook documents this gap; seeding
        from brain_id requires a registry lookup not yet implemented (tracked
        as a v2 extension).

4. **Hook in loop.py** calls :func:`recalibrate_instances_from_episodes` after
   ``load_instances()`` at kernel startup.  The oracle uses
   ``instance.ref_lap_time_s`` directly (oracle.py line 403), so mutating that
   field on the live instance objects immediately influences the next
   evaluation round — no changes to oracle.py or rollout_eval.py are needed.

Cycle:
  deploy(71cn.7) → ai_brain runs laps → tengine.race.episode.v1 on bus →
  consume_ai_brain_episodes() → recalibrate ref_lap_time + log best pilots →
  next FunSearch generation scores against a tighter target → better brain →
  repeat.

Design notes
------------
- Reuses ``_xrevrange_entries``, ``_parse_xrevrange_output``, ``_redis_bin``,
  and ``EPISODE_STREAM`` from ``rollout_eval`` — no second Redis parser.
- Accepts an optional ``_entries_fetcher`` injection hook so unit tests can
  run without Redis (same pattern as rollout_eval's ``_result_fetcher``).
- Does NOT edit oracle.py, rollout_eval.py, or distill.py.

Public surface
--------------
  consume_ai_brain_episodes(instances, track_id=None, count=200, redis_bin=None,
                             _entries_fetcher=None)
      -> EpisodeFeedback

  recalibrate_instances_from_episodes(instances, ...) -> EpisodeFeedback
      Convenience wrapper: calls consume_ai_brain_episodes and mutates each
      instance's ref_lap_time_s if a better (faster) ai_brain lap was found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from autobench.racing_kernel.rollout_eval import (
    EPISODE_STREAM,
    _redis_bin,
    _xrevrange_entries,
)
from autobench.racing_kernel.instance import RacingInstance

logger = logging.getLogger(__name__)

# Number of recent episode stream entries to scan
_DEFAULT_SCAN_COUNT = 200


@dataclass
class EpisodeFeedback:
    """Result of consuming ai_brain episodes from the bus.

    Attributes:
        updated_ref_lap_time:
            Mapping of track name → minimum ai_brain lap time (seconds) found
            in the scanned episodes.  Only populated for tracks that had at
            least one qualifying ai_brain episode.

        seed_candidates:
            List of ``pilot_id`` (brain_id) values from the best-lap episode
            per track.  v1: always empty — source program strings are not
            carried in episode events and require a brain registry lookup
            (planned for v2).

        episodes_scanned:
            Total episode entries inspected (includes all pilot_kinds).

        episodes_used:
            Number of ai_brain+completed+positive-lap entries that contributed
            to updated_ref_lap_time.
    """
    updated_ref_lap_time: dict[str, float] = field(default_factory=dict)
    seed_candidates: list[str] = field(default_factory=list)
    episodes_scanned: int = 0
    episodes_used: int = 0


def consume_ai_brain_episodes(
    instances: list[RacingInstance],
    track_id: str | None = None,
    count: int = _DEFAULT_SCAN_COUNT,
    redis_bin: str | None = None,
    *,
    _entries_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> EpisodeFeedback:
    """Read recent ai_brain episodes and derive feedback signals.

    Reads up to *count* most-recent entries from
    ``nbus:tengine.race.episode.v1`` and filters to:
      - ``pilot_kind == "ai_brain"``
      - ``outcome == "completed"``
      - ``lap_time_ms > 0``
      - track match: ``track_id`` field equals ``instance.name`` OR
        ``track_seed`` equals ``instance.seed`` for at least one instance
        in *instances* (unless ``track_id`` is specified, in which case only
        episodes for that track are collected).

    For each qualifying track, returns the minimum lap time in seconds in
    ``updated_ref_lap_time`` and the corresponding ``pilot_id`` in
    ``seed_candidates`` (v1: empty — see module docstring).

    Args:
        instances:        List of RacingInstance objects currently loaded by
                          the kernel.  Used for track-name / seed matching.
        track_id:         If provided, only episodes matching this track_id
                          (or a track whose name matches) are considered.
                          None → all tracks in *instances*.
        count:            Max XREVRANGE entries to scan.
        redis_bin:        Path to redis-cli binary.  None → locate via PATH.
        _entries_fetcher: Test-injection hook.  If provided, called as
                          ``fetcher(stream, count, redis_bin)`` instead of
                          the real XREVRANGE call.  Lets unit tests pass
                          synthetic episode lists without touching Redis.

    Returns:
        EpisodeFeedback with per-track minimum ai_brain lap times (s) and
        (v1) an empty seed_candidates list.
    """
    if redis_bin is None:
        redis_bin = _redis_bin()

    fetcher = _entries_fetcher or _xrevrange_entries
    entries: list[dict[str, Any]] = fetcher(EPISODE_STREAM, count, redis_bin)

    feedback = EpisodeFeedback(episodes_scanned=len(entries))

    # Build lookup structures for track matching.
    # Map: name → instance, seed → instance
    name_to_inst: dict[str, RacingInstance] = {i.name: i for i in instances}
    seed_to_inst: dict[int, RacingInstance] = {i.seed: i for i in instances}

    # Per-track: best (minimum) lap_time_ms from ai_brain episodes
    best_ms: dict[str, float] = {}   # track_name → min lap_time_ms
    best_pilot: dict[str, str] = {}  # track_name → pilot_id of best lap

    for data in entries:
        # Only ai_brain + completed episodes
        if data.get("pilot_kind") != "ai_brain":
            continue
        if data.get("outcome") != "completed":
            continue

        # Extract lap time
        try:
            ms = float(data["lap_time_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if ms <= 0:
            continue

        # Resolve track name from episode fields
        ep_track_id: str | None = data.get("track_id")
        ep_track_seed_raw = data.get("track_seed")

        resolved_name: str | None = None

        if ep_track_id and ep_track_id in name_to_inst:
            resolved_name = ep_track_id
        elif ep_track_seed_raw is not None:
            try:
                ep_seed = int(ep_track_seed_raw)
            except (TypeError, ValueError):
                ep_seed = None
            if ep_seed is not None and ep_seed in seed_to_inst:
                resolved_name = seed_to_inst[ep_seed].name

        if resolved_name is None:
            logger.debug(
                "loop_feedback: episode track_id=%r seed=%r not in loaded instances; skipping",
                ep_track_id, ep_track_seed_raw,
            )
            continue

        # Apply optional track_id filter
        if track_id is not None and resolved_name != track_id:
            continue

        feedback.episodes_used += 1

        if resolved_name not in best_ms or ms < best_ms[resolved_name]:
            best_ms[resolved_name] = ms
            pilot = data.get("pilot_id", "")
            best_pilot[resolved_name] = pilot

    # Populate feedback
    for track_name, ms in best_ms.items():
        calibrated_s = ms / 1000.0
        feedback.updated_ref_lap_time[track_name] = calibrated_s
        logger.info(
            "loop_feedback: track '%s' — best ai_brain lap %.3f s "
            "(pilot_id=%r); %d episode(s) used",
            track_name, calibrated_s, best_pilot.get(track_name, ""), feedback.episodes_used,
        )

    # v1: seed_candidates is always empty — source programs are not carried in
    # episode events.  A future v2 will cross-reference brain_id against a
    # brain registry to retrieve the source program for seeding.
    feedback.seed_candidates = []

    if not feedback.updated_ref_lap_time:
        logger.debug(
            "loop_feedback: no qualifying ai_brain episodes found "
            "(%d total entries scanned); ref_lap_time unchanged",
            feedback.episodes_scanned,
        )

    return feedback


def recalibrate_instances_from_episodes(
    instances: list[RacingInstance],
    track_id: str | None = None,
    count: int = _DEFAULT_SCAN_COUNT,
    redis_bin: str | None = None,
    *,
    _entries_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> EpisodeFeedback:
    """Consume ai_brain episodes and mutate instance.ref_lap_time_s in-place.

    This is the primary hook called from ``RacingKernel.__init__`` after
    ``load_instances()``.  It combines episode consumption with instance
    mutation so the caller only needs one call.

    For each instance whose track appears in the returned
    ``EpisodeFeedback.updated_ref_lap_time``, the instance's
    ``ref_lap_time_s`` is updated IF the ai_brain lap is faster (smaller)
    than the current value.  This is a monotone-tighten policy: deployed
    brains can only raise the bar, never lower it back to synthetic defaults.

    Args:
        instances:        Live RacingInstance objects to potentially update.
        track_id:         Optional filter (see consume_ai_brain_episodes).
        count:            Max XREVRANGE entries to scan.
        redis_bin:        Path to redis-cli (None → PATH lookup).
        _entries_fetcher: Test-injection hook (see consume_ai_brain_episodes).

    Returns:
        EpisodeFeedback — the same object returned by consume_ai_brain_episodes,
        for inspection / logging by the caller.
    """
    feedback = consume_ai_brain_episodes(
        instances,
        track_id=track_id,
        count=count,
        redis_bin=redis_bin,
        _entries_fetcher=_entries_fetcher,
    )

    for inst in instances:
        new_ref = feedback.updated_ref_lap_time.get(inst.name)
        if new_ref is None:
            continue
        if new_ref < inst.ref_lap_time_s:
            logger.info(
                "loop_feedback: tightening ref_lap_time for '%s': "
                "%.3f s → %.3f s (ai_brain improvement %.1f%%)",
                inst.name, inst.ref_lap_time_s, new_ref,
                (inst.ref_lap_time_s - new_ref) / inst.ref_lap_time_s * 100,
            )
            inst.ref_lap_time_s = new_ref
        else:
            logger.debug(
                "loop_feedback: ai_brain lap for '%s' (%.3f s) is not faster "
                "than current ref (%.3f s); no update",
                inst.name, new_ref, inst.ref_lap_time_s,
            )

    return feedback
