"""Tests for the live-loop JudgingPool wiring (nervous-bus-c48, wire-pop Phase 7).

Covers:
- 5-judge consensus when all agree (dissent=0).
- Mixed verdicts produce correct consensus + dissent ratio.
- Dissent exceeding threshold emits the disagreement event.
- n=1 fallback path is bit-for-bit identical to the pre-wire single-judge path.
- Anonymous = each judge wrapper is a fresh instance (no shared state).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from autobench.core import HarnessConfig, Verdict
from autobench.evaluator import (
    DEFAULT_DISSENT_THRESHOLD,
    DEFAULT_JUDGES_PER_CASE,
    BenchmarkCase,
    BenchmarkEvaluator,
)
from autobench.observability import (
    CHANNEL_JUDGE_DISAGREEMENT,
    CHANNEL_JUDGE_POOL_VERDICT,
    AutobenchObservability,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_obs(tmp_path: Path) -> AutobenchObservability:
    """An observability instance whose pipe is force-disabled — JSONL only."""
    obs = AutobenchObservability(
        session_id="01TESTSESSIONULIDXXXXXXXXX",
        debug_file=tmp_path / "debug.jsonl",
    )
    obs._pipe_disabled = True  # force the cheap append-to-disk path
    return obs


def _events_for_channel(tmp_path: Path, channel: str) -> list[dict]:
    """Read every JSONL event for one channel out of the debug fallback file."""
    debug_file = tmp_path / "debug.jsonl"
    if not debug_file.exists():
        return []
    out: list[dict] = []
    for line in debug_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == channel:
            out.append(ev)
    return out


def _make_case(case_id: str = "ok-case") -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        prompt="print('hi')",
        language="python",
        expected_output="hi\n",
        test_inputs=[],
    )


def _make_harness() -> HarnessConfig:
    return HarnessConfig(
        budget={"max_cost_dollars": 0.0, "max_time_seconds": 30, "max_memory_mb": 512},
    )


def _ok_generate_fn(prompt: str, cfg: HarnessConfig) -> str:
    return "print('hi')\n"


# --------------------------------------------------------------------------- #
# Judge factories
# --------------------------------------------------------------------------- #


def _unanimous_ok_factory(prompt: str, context: dict) -> dict:
    return {
        "judge_id": "shared-id",  # the anonymous wrapper overrides this
        "verdict": "OK",
        "p_score": 0.9,
        "p_cost": 0.1,
        "p_time": 0.8,
        "reasoning": "looks fine",
    }


# --------------------------------------------------------------------------- #
# 1) 5-judge consensus when all 5 agree → dissent=0.
# --------------------------------------------------------------------------- #


def test_pool_unanimous_consensus_emits_zero_dissent(tmp_path):
    obs = _make_obs(tmp_path)
    ev = BenchmarkEvaluator(
        generate_fn=_ok_generate_fn,
        obs=obs,
        judge_factory=_unanimous_ok_factory,
        judges_per_case=5,
    )
    res = ev.run(_make_harness(), [_make_case("u-1")])

    assert len(res.case_results) == 1
    r = res.case_results[0]
    jp = (r.metadata or {}).get("judge_pool")
    assert jp is not None, "expected judge_pool metadata when judges_per_case > 1"
    assert jp["n_judges"] == 5
    assert jp["n_votes"] == 5
    assert jp["consensus_verdict"] == "OK"
    assert jp["dissent_ratio"] == 0.0
    assert jp["verdict_distribution"] == {"OK": 5}
    # mean of p_score=0.9 × 5 = 0.9
    assert abs(jp["consensus_p_score"] - 0.9) < 1e-6

    # autobench.judge.pool.verdict.v1 fired exactly once for the one case.
    pool_evs = _events_for_channel(tmp_path, CHANNEL_JUDGE_POOL_VERDICT)
    assert len(pool_evs) == 1
    payload = pool_evs[0]["data"]
    assert payload["case_id"] == "u-1"
    assert payload["dissent_ratio"] == 0.0
    assert payload["n_judges"] == 5
    assert payload["n_votes"] == 5
    assert payload["consensus_verdict"] == "OK"
    assert payload["verdict_distribution"] == {"OK": 5}

    # No disagreement event when unanimous.
    disagreement_evs = _events_for_channel(tmp_path, CHANNEL_JUDGE_DISAGREEMENT)
    assert disagreement_evs == []

    # BenchmarkResult envelope surfaces the dissent metric (Phase 7 contract).
    md = res.metadata
    assert md["judge_pool_contested_count"] == 0
    assert md["judges_per_case"] == 5
    assert md["judge_pool_dissent_threshold"] == DEFAULT_DISSENT_THRESHOLD
    assert len(md["judge_pool_dissent_ratios"]) == 1
    assert md["judge_pool_dissent_ratios"][0]["dissent_ratio"] == 0.0


# --------------------------------------------------------------------------- #
# 2) Mixed verdicts → correct consensus + dissent ratio.
# --------------------------------------------------------------------------- #


def test_pool_mixed_verdicts_compute_dissent_ratio(tmp_path):
    """3 judges vote OK, 2 vote WA → consensus=OK, dissent=2/5=0.4 (NOT > 0.4)."""
    obs = _make_obs(tmp_path)

    # Stateful factory: yields a deterministic sequence across calls so
    # whichever 5 anonymous wrappers fire, the 5 votes come back as
    # [OK, OK, OK, WA, WA] in some order. The pool uses the multiset, so
    # ordering doesn't matter.
    verdict_seq = iter(["OK", "OK", "OK", "WA", "WA"])

    import threading
    seq_lock = threading.Lock()

    def _mixed_factory(prompt: str, context: dict) -> dict:
        with seq_lock:
            v = next(verdict_seq)
        return {
            "judge_id": "raw-id",
            "verdict": v,
            "p_score": 1.0 if v == "OK" else 0.0,
            "p_cost": 0.5,
            "p_time": 0.5,
            "reasoning": v,
        }

    ev = BenchmarkEvaluator(
        generate_fn=_ok_generate_fn,
        obs=obs,
        judge_factory=_mixed_factory,
        judges_per_case=5,
        dissent_threshold=0.4,  # 0.4 is NOT > 0.4, so no disagreement event
    )
    res = ev.run(_make_harness(), [_make_case("m-1")])

    jp = (res.case_results[0].metadata or {}).get("judge_pool")
    assert jp["n_votes"] == 5
    assert jp["consensus_verdict"] == "OK"
    assert jp["verdict_distribution"] == {"OK": 3, "WA": 2}
    # dissent = 2 / 5
    assert abs(jp["dissent_ratio"] - 0.4) < 1e-9

    # Threshold is strictly-greater-than: 0.4 == 0.4 must NOT trip the
    # escalation. This is the boundary case AHE relies on.
    disagreement_evs = _events_for_channel(tmp_path, CHANNEL_JUDGE_DISAGREEMENT)
    assert disagreement_evs == [], (
        "dissent_ratio == threshold should NOT emit disagreement (strict >)"
    )

    # Envelope shows the case as NOT contested (matches event behavior).
    assert res.metadata["judge_pool_contested_count"] == 0


# --------------------------------------------------------------------------- #
# 3) Dissent exceeding threshold emits the disagreement event.
# --------------------------------------------------------------------------- #


def test_pool_high_dissent_emits_disagreement_event(tmp_path):
    """3 OK / 2 WA at threshold=0.3 → 0.4 > 0.3, disagreement fires."""
    obs = _make_obs(tmp_path)

    verdict_seq = iter(["OK", "OK", "OK", "WA", "WA"])
    import threading
    seq_lock = threading.Lock()

    def _mixed_factory(prompt: str, context: dict) -> dict:
        with seq_lock:
            v = next(verdict_seq)
        return {
            "judge_id": "raw-id",
            "verdict": v,
            "p_score": 1.0 if v == "OK" else 0.0,
            "p_cost": 0.5,
            "p_time": 0.5,
        }

    ev = BenchmarkEvaluator(
        generate_fn=_ok_generate_fn,
        obs=obs,
        judge_factory=_mixed_factory,
        judges_per_case=5,
        dissent_threshold=0.3,
    )
    res = ev.run(_make_harness(), [_make_case("d-1")])

    disagreement_evs = _events_for_channel(tmp_path, CHANNEL_JUDGE_DISAGREEMENT)
    assert len(disagreement_evs) == 1
    payload = disagreement_evs[0]["data"]
    assert payload["case_id"] == "d-1"
    assert payload["consensus_verdict"] == "OK"
    assert payload["dissent_threshold"] == 0.3
    assert abs(payload["dissent_ratio"] - 0.4) < 1e-9
    assert "WA" in payload["minority_verdicts"]
    assert "OK" not in payload["minority_verdicts"]
    assert payload["verdict_distribution"] == {"OK": 3, "WA": 2}

    # Envelope marks the case as contested (count == 1).
    assert res.metadata["judge_pool_contested_count"] == 1


# --------------------------------------------------------------------------- #
# 4) n=1 fallback — bit-for-bit identical to the pre-wire path.
# --------------------------------------------------------------------------- #


def test_pool_n1_fallback_does_not_invoke_pool(tmp_path):
    """When judges_per_case=1, the pool is disabled — no events, no metadata,
    no judge_factory calls. The HarnessResult must match the legacy path."""
    obs = _make_obs(tmp_path)
    call_count = {"n": 0}

    def _counting_factory(prompt: str, context: dict) -> dict:
        call_count["n"] += 1
        return {"verdict": "OK", "p_score": 1.0, "p_cost": 0.0, "p_time": 1.0}

    ev = BenchmarkEvaluator(
        generate_fn=_ok_generate_fn,
        obs=obs,
        judge_factory=_counting_factory,
        judges_per_case=1,
    )
    res = ev.run(_make_harness(), [_make_case("n1-1")])

    # Factory must NEVER be called when judges_per_case=1 — pool path is
    # gated out entirely.
    assert call_count["n"] == 0
    # No judge_pool metadata leaked onto the result.
    assert "judge_pool" not in (res.case_results[0].metadata or {})
    # No bus events from either new channel.
    assert _events_for_channel(tmp_path, CHANNEL_JUDGE_POOL_VERDICT) == []
    assert _events_for_channel(tmp_path, CHANNEL_JUDGE_DISAGREEMENT) == []
    # Envelope still surfaces the configured judges_per_case (=1) and an
    # empty dissent_ratios list so downstream consumers don't KeyError.
    assert res.metadata["judges_per_case"] == 1
    assert res.metadata["judge_pool_dissent_ratios"] == []
    assert res.metadata["judge_pool_contested_count"] == 0


def test_pool_no_factory_disables_pool_even_at_high_n(tmp_path, monkeypatch):
    """judges_per_case=5 and judge_factory=None → pool stays off ONLY when
    the MiniMax default-factory gate is also closed (no API key OR the
    opt-out env is set). When the key is present, the default fires —
    that's the Move #2 behavior change, not a regression."""
    # Force the default-factory gate closed so the legacy "no factory, no
    # pool" path is exercised.
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("AUTOBENCH_DISABLE_DEFAULT_JUDGE", "1")

    obs = _make_obs(tmp_path)
    ev = BenchmarkEvaluator(
        generate_fn=_ok_generate_fn,
        obs=obs,
        judge_factory=None,
        judges_per_case=5,
    )
    res = ev.run(_make_harness(), [_make_case("nf-1")])

    assert "judge_pool" not in (res.case_results[0].metadata or {})
    assert _events_for_channel(tmp_path, CHANNEL_JUDGE_POOL_VERDICT) == []


# --------------------------------------------------------------------------- #
# 5) Anonymous = each judge wrapper is a fresh instance, no shared state.
# --------------------------------------------------------------------------- #


def test_pool_anonymous_judges_have_unique_slots(tmp_path):
    """Verify that the N wrappers are independent function objects, each
    sees a unique slot, and the emitted judge_id rewrites whatever the
    underlying factory returned. No cross-call state can leak."""
    obs = _make_obs(tmp_path)
    seen_ids: list[str] = []
    seen_wrapper_objects: list[int] = []

    def _id_capturing_factory(prompt: str, context: dict) -> dict:
        # The factory returns the SAME judge_id every time — the anonymous
        # wrapper around it must override this with anon-<slot>.
        return {
            "judge_id": "raw-shared-id",
            "verdict": "OK",
            "p_score": 0.5,
            "p_cost": 0.5,
            "p_time": 0.5,
        }

    ev = BenchmarkEvaluator(
        generate_fn=_ok_generate_fn,
        obs=obs,
        judge_factory=_id_capturing_factory,
        judges_per_case=5,
    )
    res = ev.run(_make_harness(), [_make_case("a-1")])

    jp = (res.case_results[0].metadata or {}).get("judge_pool")
    assert jp is not None
    votes_summary = jp["votes_summary"]
    assert len(votes_summary) == 5
    slots = [v["slot"] for v in votes_summary]
    # Unique slot indices, one per anonymous wrapper.
    assert sorted(slots) == [0, 1, 2, 3, 4]


def test_pool_default_judges_per_case_is_five(monkeypatch):
    """Default ensemble size is 5 per the c48 cost model (1500 req/cycle)."""
    monkeypatch.delenv("AUTOBENCH_JUDGES_PER_CASE", raising=False)
    ev = BenchmarkEvaluator(judge_factory=_unanimous_ok_factory)
    assert ev.judges_per_case == DEFAULT_JUDGES_PER_CASE == 5


def test_pool_env_overrides_default(monkeypatch):
    """AUTOBENCH_JUDGES_PER_CASE env knob controls N."""
    monkeypatch.setenv("AUTOBENCH_JUDGES_PER_CASE", "3")
    ev = BenchmarkEvaluator(judge_factory=_unanimous_ok_factory)
    assert ev.judges_per_case == 3


def test_pool_env_clamps_to_min_one(monkeypatch):
    """AUTOBENCH_JUDGES_PER_CASE=0 is illegal → clamped to 1."""
    monkeypatch.setenv("AUTOBENCH_JUDGES_PER_CASE", "0")
    ev = BenchmarkEvaluator(judge_factory=_unanimous_ok_factory)
    assert ev.judges_per_case == 1


def test_pool_env_garbage_falls_back_to_default(monkeypatch):
    """Unparseable env value → falls back to DEFAULT_JUDGES_PER_CASE."""
    monkeypatch.setenv("AUTOBENCH_JUDGES_PER_CASE", "not-a-number")
    ev = BenchmarkEvaluator(judge_factory=_unanimous_ok_factory)
    assert ev.judges_per_case == DEFAULT_JUDGES_PER_CASE


def test_pool_explicit_kwarg_beats_env(monkeypatch):
    """Explicit judges_per_case= kwarg overrides the env var."""
    monkeypatch.setenv("AUTOBENCH_JUDGES_PER_CASE", "7")
    ev = BenchmarkEvaluator(judge_factory=_unanimous_ok_factory, judges_per_case=2)
    assert ev.judges_per_case == 2
