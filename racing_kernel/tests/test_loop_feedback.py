"""Tests for racing_kernel/loop_feedback.py — ai_brain episode feedback.

All tests run without LLM, network, Redis, or real GPU.
Episode entries are injected via the _entries_fetcher hook.

Run:
    NBUS_ROOT=/home/eric/projects/nervous-bus \
    PYTHONPATH=/home/eric/projects/nervous-autobench/.claude/worktrees/race-loop \
    python -m pytest -m "not live" racing_kernel/tests/test_loop_feedback.py -q
"""

from __future__ import annotations

from typing import Any

import pytest

from autobench.racing_kernel.instance import generate_instance
from autobench.racing_kernel.loop_feedback import (
    EpisodeFeedback,
    consume_ai_brain_episodes,
    recalibrate_instances_from_episodes,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetcher(episodes: list[dict[str, Any]]):
    """Return a stub _entries_fetcher that always returns the given episodes."""
    def _f(stream, count, redis_bin):  # noqa: ARG001
        return episodes
    return _f


def _episode(
    pilot_kind: str = "ai_brain",
    outcome: str = "completed",
    track_id: str = "oval",
    lap_time_ms: float = 5000.0,
    pilot_id: str = "brain_001",
    track_seed: int | None = None,
) -> dict[str, Any]:
    """Build a minimal mock episode dict."""
    ep: dict[str, Any] = {
        "pilot_kind": pilot_kind,
        "outcome": outcome,
        "track_id": track_id,
        "lap_time_ms": lap_time_ms,
        "pilot_id": pilot_id,
    }
    if track_seed is not None:
        ep["track_seed"] = track_seed
    return ep


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestConsumeAiBrainEpisodes:

    def _instances(self, *names):
        return [generate_instance(n) for n in names]

    # Case 1: empty stream → no updates, no seeds
    def test_empty_stream_returns_empty_feedback(self):
        """Empty episode list → EpisodeFeedback with no updates."""
        instances = self._instances("oval")
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher([]))
        assert isinstance(fb, EpisodeFeedback)
        assert fb.updated_ref_lap_time == {}
        assert fb.seed_candidates == []
        assert fb.episodes_scanned == 0
        assert fb.episodes_used == 0

    # Case 2: only non-ai_brain episodes → no updates
    def test_non_ai_brain_episodes_are_ignored(self):
        """human and scripted episodes are not used."""
        instances = self._instances("oval")
        episodes = [
            _episode(pilot_kind="human", lap_time_ms=4000.0),
            _episode(pilot_kind="scripted", lap_time_ms=4500.0),
        ]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert fb.updated_ref_lap_time == {}
        assert fb.episodes_used == 0

    # Case 3: ai_brain + non-completed outcome → not used
    def test_non_completed_ai_brain_episodes_are_ignored(self):
        """crashed/dnf/aborted ai_brain episodes are excluded."""
        instances = self._instances("oval")
        episodes = [
            _episode(outcome="crashed", lap_time_ms=3000.0),
            _episode(outcome="dnf", lap_time_ms=3200.0),
            _episode(outcome="aborted", lap_time_ms=3100.0),
        ]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert fb.updated_ref_lap_time == {}
        assert fb.episodes_used == 0

    # Case 4: zero or negative lap_time_ms → not used
    def test_zero_or_negative_lap_time_excluded(self):
        """Episodes with lap_time_ms <= 0 are skipped."""
        instances = self._instances("oval")
        episodes = [
            _episode(lap_time_ms=0.0),
            _episode(lap_time_ms=-100.0),
        ]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert fb.updated_ref_lap_time == {}
        assert fb.episodes_used == 0

    # Case 5: single qualifying episode → correct calibration + brain_id in seeds
    def test_single_qualifying_episode_sets_ref_lap_time(self):
        """One qualifying ai_brain episode → ref_lap_time updated, pilot_id in seeds."""
        instances = self._instances("oval")
        lap_ms = 4200.0
        episodes = [_episode(track_id="oval", lap_time_ms=lap_ms, pilot_id="brain_x")]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert "oval" in fb.updated_ref_lap_time
        assert abs(fb.updated_ref_lap_time["oval"] - lap_ms / 1000.0) < 1e-6
        assert fb.episodes_used == 1
        assert "brain_x" in fb.seed_candidates

    # Case 6: multiple ai_brain laps for same track → minimum wins; fastest pilot in seeds
    def test_minimum_lap_time_wins_per_track(self):
        """Multiple ai_brain laps for one track → fastest lap selected; brain_B in seeds."""
        instances = self._instances("oval")
        episodes = [
            _episode(track_id="oval", lap_time_ms=5000.0, pilot_id="brain_A"),
            _episode(track_id="oval", lap_time_ms=3800.0, pilot_id="brain_B"),  # fastest
            _episode(track_id="oval", lap_time_ms=4200.0, pilot_id="brain_C"),
        ]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert fb.updated_ref_lap_time["oval"] == pytest.approx(3.8)
        assert fb.episodes_used == 3
        # seed_candidates carries the pilot_id of the fastest lap (brain_B)
        assert "brain_B" in fb.seed_candidates

    # Case 7: multiple tracks, each gets its own minimum
    def test_per_track_minimum_independent(self):
        """Each track accumulates its own minimum ai_brain lap time."""
        instances = self._instances("oval", "chicane")
        episodes = [
            _episode(track_id="oval", lap_time_ms=5000.0),
            _episode(track_id="oval", lap_time_ms=4100.0),
            _episode(track_id="chicane", lap_time_ms=6000.0),
            _episode(track_id="chicane", lap_time_ms=5500.0),
        ]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert "oval" in fb.updated_ref_lap_time
        assert "chicane" in fb.updated_ref_lap_time
        assert fb.updated_ref_lap_time["oval"] == pytest.approx(4.1)
        assert fb.updated_ref_lap_time["chicane"] == pytest.approx(5.5)

    # Case 8: track_id filter restricts to one track
    def test_track_id_filter(self):
        """track_id kwarg restricts results to matching track only."""
        instances = self._instances("oval", "chicane")
        episodes = [
            _episode(track_id="oval", lap_time_ms=4000.0),
            _episode(track_id="chicane", lap_time_ms=5000.0),
        ]
        fb = consume_ai_brain_episodes(
            instances, track_id="oval", _entries_fetcher=_fetcher(episodes)
        )
        assert "oval" in fb.updated_ref_lap_time
        assert "chicane" not in fb.updated_ref_lap_time

    # Case 9: track matched by track_seed (no track_id field)
    def test_track_seed_fallback_matching(self):
        """Episodes without track_id but with matching track_seed are resolved."""
        inst = generate_instance("oval")
        episodes = [{
            "pilot_kind": "ai_brain",
            "outcome": "completed",
            "track_seed": inst.seed,   # match via seed
            "lap_time_ms": 3900.0,
            "pilot_id": "brain_seed_test",
            # no track_id key
        }]
        fb = consume_ai_brain_episodes([inst], _entries_fetcher=_fetcher(episodes))
        assert "oval" in fb.updated_ref_lap_time
        assert fb.updated_ref_lap_time["oval"] == pytest.approx(3.9)

    # Case 10: episode for unknown track → silently ignored
    def test_unknown_track_episodes_are_skipped(self):
        """Episodes for tracks not in loaded instances are skipped silently."""
        instances = self._instances("oval")
        episodes = [
            _episode(track_id="nonexistent_track_xyz", lap_time_ms=1000.0),
        ]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert fb.updated_ref_lap_time == {}
        assert fb.episodes_used == 0

    # Case 11: mixed valid + invalid → only valid ones counted
    def test_mixed_episodes_only_valid_counted(self):
        """Mix of valid and invalid episodes: only qualifying rows used."""
        instances = self._instances("oval")
        episodes = [
            _episode(pilot_kind="human", lap_time_ms=3000.0),        # wrong kind
            _episode(outcome="crashed", lap_time_ms=3100.0),          # wrong outcome
            _episode(lap_time_ms=0.0),                                 # zero lap_time
            _episode(track_id="nonexistent", lap_time_ms=1000.0),     # unknown track
            _episode(track_id="oval", lap_time_ms=4500.0),            # valid
        ]
        fb = consume_ai_brain_episodes(instances, _entries_fetcher=_fetcher(episodes))
        assert "oval" in fb.updated_ref_lap_time
        assert fb.updated_ref_lap_time["oval"] == pytest.approx(4.5)
        assert fb.episodes_used == 1


class TestRecalibrateInstancesFromEpisodes:

    # Case 12: faster ai_brain lap tightens instance.ref_lap_time_s
    def test_faster_ai_brain_lap_tightens_ref_time(self):
        """When ai_brain lap is faster than synthetic ref, instance is updated."""
        inst = generate_instance("oval")
        original_ref = inst.ref_lap_time_s

        # Inject a lap that is genuinely faster than the synthetic reference
        faster_ms = (original_ref - 1.0) * 1000.0  # 1 s faster
        assert faster_ms > 0, "synthetic ref must be > 1 s for this test"

        episodes = [_episode(track_id="oval", lap_time_ms=faster_ms)]
        fb = recalibrate_instances_from_episodes([inst], _entries_fetcher=_fetcher(episodes))

        assert "oval" in fb.updated_ref_lap_time
        assert inst.ref_lap_time_s == pytest.approx(faster_ms / 1000.0)
        assert inst.ref_lap_time_s < original_ref

    # Case 13: slower ai_brain lap does NOT change the ref time (monotone-tighten)
    def test_slower_ai_brain_lap_does_not_change_ref_time(self):
        """When ai_brain lap is slower than current ref, instance is NOT updated."""
        inst = generate_instance("oval")
        original_ref = inst.ref_lap_time_s

        # Inject a lap that is SLOWER than the synthetic reference
        slower_ms = (original_ref + 5.0) * 1000.0

        episodes = [_episode(track_id="oval", lap_time_ms=slower_ms)]
        recalibrate_instances_from_episodes([inst], _entries_fetcher=_fetcher(episodes))

        # ref_lap_time_s must be unchanged
        assert inst.ref_lap_time_s == pytest.approx(original_ref)

    # Case 14: empty episodes → instance unchanged
    def test_empty_episodes_leaves_instance_unchanged(self):
        """Empty stream → instance.ref_lap_time_s is unmodified."""
        inst = generate_instance("oval")
        original_ref = inst.ref_lap_time_s
        recalibrate_instances_from_episodes([inst], _entries_fetcher=_fetcher([]))
        assert inst.ref_lap_time_s == pytest.approx(original_ref)

    # Case 15: End-to-end influence proof — oracle score changes after recalibration
    def test_oracle_score_influenced_after_recalibration(self):
        """
        After recalibration, the oracle gives a different fitness score for the
        SAME program — proving that ai_brain episodes measurably influence the
        next evolution round.

        Approach:
          1. Score a seed program against the default (synthetic) ref_lap_time.
          2. Inject a faster ai_brain lap → recalibrate → ref_lap_time tightens.
          3. Score the same program again; fitness must be lower (harder target).
        """
        from autobench.racing_kernel.oracle import evaluate_on_instance, _SEED_PURE_PURSUIT

        inst = generate_instance("oval")
        original_ref = inst.ref_lap_time_s

        # Score before calibration
        score_before = evaluate_on_instance(_SEED_PURE_PURSUIT, inst)
        assert score_before is not None

        # Inject a faster ai_brain lap (1 second faster than synthetic)
        faster_ms = (original_ref - 1.0) * 1000.0
        assert faster_ms > 0
        episodes = [_episode(track_id="oval", lap_time_ms=faster_ms)]
        recalibrate_instances_from_episodes([inst], _entries_fetcher=_fetcher(episodes))

        # Tighter target → same program scores lower (or at most equal)
        score_after = evaluate_on_instance(_SEED_PURE_PURSUIT, inst)
        assert score_after is not None
        # Speed component is scored against a tighter ref → overall fitness drops
        assert score_after < score_before, (
            f"Expected fitness to decrease after tightening ref_lap_time "
            f"({original_ref:.3f}s → {inst.ref_lap_time_s:.3f}s), "
            f"but got {score_before:.4f} → {score_after:.4f}"
        )
