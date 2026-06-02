"""Tests for the multi-improver ensemble (nervous-bus-9xd, wire-pop Phase 6).

Covers:
  * Strategy A (vote): majority vote produces one merged delta, dissent is
    recorded in ``vote_outcome["selected_mask"]``.
  * Strategy A tie-break: when no single value has a strict majority, the
    aggregator picks the first non-noop contender in fan-out order.
  * Strategy B (parallel): best-arm-by-score wins; the chosen ``(harness, delta)``
    pair matches the highest-scoring arm.
  * Three anonymous arms produce three independent wrapper calls (verified via
    the mock wrapper factory's call count).
  * Single-instance fallback (n=1) is behaviourally identical to the legacy
    single-improver path AND emits no ensemble event.
  * Observability — when an ensemble runs (n>=2) with ``obs`` present, exactly
    one ``autobench.improver.ensemble.v1`` event lands on the bus.
  * RSI loop integration — ``default_improver='minimax_ensemble'`` routes
    through the ensemble path; ``AUTOBENCH_ADVOCATES=1`` with no ensemble env
    set leaves the legacy ``default_improver='minimax'`` path bit-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autobench.core import (
    ContextManager,
    HarnessConfig,
    HarnessResult,
    RolloutProtocol,
    Verdict,
)
from autobench.evaluator import BenchmarkResult
from autobench.llm.ensemble import (
    DEFAULT_N_INSTANCES,
    MultiImproverEnsemble,
    aggregate_deltas,
)
from autobench.observability import AutobenchObservability, CHANNEL_IMPROVER_ENSEMBLE
from autobench.rsi.loop import ImprovementDelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_harness(
    *,
    system_prompt: str = "solve coding problems",
    rollout: RolloutProtocol = RolloutProtocol.SINGLE,
    ctx: ContextManager = ContextManager.FULL,
) -> HarnessConfig:
    return HarnessConfig(
        system_prompt=system_prompt,
        rollout_protocol=rollout,
        context_manager=ctx,
        tool_surface="bash, python",
        verifiers=[],
        budget={"max_tokens": 8192, "max_time_seconds": 30, "max_cost_dollars": 1.0},
    )


def _make_bench_result() -> BenchmarkResult:
    cases = [
        HarnessResult(verdict=Verdict.CE, p_score=0.0, latency_ms=10),
        HarnessResult(verdict=Verdict.OK, p_score=1.0, latency_ms=11),
    ]
    return BenchmarkResult(
        case_results=cases,
        aggregate_score=0.5,
        total_latency_ms=21.0,
        verdict_counts={"CE": 1, "OK": 1},
    )


def _delta_a() -> tuple[HarnessConfig, ImprovementDelta]:
    """Arm A: proposes a system_prompt change + iterative protocol."""
    h = _make_harness(rollout=RolloutProtocol.ITERATIVE)
    h.system_prompt = "solve coding problems\nPrefer short solutions."
    d = ImprovementDelta(
        system_prompt_delta="Prefer short solutions.",
        rollout_protocol_changed=True,
        improvement_summary="A: shorter prompts",
        budget_delta={"max_tokens": 6000},
    )
    h.budget["max_tokens"] = 6000
    return h, d


def _delta_b() -> tuple[HarnessConfig, ImprovementDelta]:
    """Arm B: same system_prompt change + iterative protocol (agrees with A)."""
    h = _make_harness(rollout=RolloutProtocol.ITERATIVE)
    h.system_prompt = "solve coding problems\nPrefer short solutions."
    d = ImprovementDelta(
        system_prompt_delta="Prefer short solutions.",
        rollout_protocol_changed=True,
        improvement_summary="B: also shorter",
        budget_delta={"max_tokens": 6000},
    )
    h.budget["max_tokens"] = 6000
    return h, d


def _delta_c() -> tuple[HarnessConfig, ImprovementDelta]:
    """Arm C: dissents — proposes a hierarchical context manager instead."""
    h = _make_harness(ctx=ContextManager.HIERARCHICAL)
    d = ImprovementDelta(
        context_manager_changed=True,
        improvement_summary="C: switch to hierarchical context",
    )
    return h, d


# ---------------------------------------------------------------------------
# (a) aggregate_deltas — vote majority
# ---------------------------------------------------------------------------


def test_vote_majority_produces_merged_delta():
    """Strategy A: when 2/3 arms agree on a field, the majority wins."""
    pairs = [_delta_a(), _delta_b(), _delta_c()]
    chosen_harness, chosen_delta, vote_outcome = aggregate_deltas(
        pairs, strategy="vote", baseline_harness=_make_harness(),
    )

    # A and B agree on system_prompt_delta + rollout_protocol_changed +
    # budget_delta; C dissents with context_manager.
    assert chosen_delta.system_prompt_delta == "Prefer short solutions."
    assert chosen_delta.rollout_protocol_changed is True
    assert chosen_delta.budget_delta == {"max_tokens": 6000}
    # Context manager change is a 1-vote minority — should NOT carry over.
    assert chosen_delta.context_manager_changed is False

    # The chosen harness should reflect the winning fields. A or B's
    # harness embodies the most winners; either is acceptable.
    assert chosen_harness.rollout_protocol == RolloutProtocol.ITERATIVE
    assert chosen_harness.budget["max_tokens"] == 6000

    # vote_outcome must record dissent: only A and B "selected"; C not.
    selected_mask = vote_outcome["selected_mask"]
    assert selected_mask[0] is True
    assert selected_mask[1] is True
    assert selected_mask[2] is False
    assert vote_outcome["selected_instance_idx"] is None
    # Field votes must contain all five votable fields.
    assert set(vote_outcome["field_votes"].keys()) >= {
        "system_prompt_delta", "rollout_protocol_changed",
        "context_manager_changed", "tool_surface_delta", "budget_delta",
    }


# ---------------------------------------------------------------------------
# (b) aggregate_deltas — vote tie-break
# ---------------------------------------------------------------------------


def test_vote_tie_breaks_to_first_non_noop_in_fan_out_order():
    """When two values each get N/2 votes, first non-noop in order wins.

    With 2 arms (one proposes change, one proposes nothing) every field has a
    1-1 tie. The aggregator must pick the non-noop value, since otherwise
    50% dissent would silently default to no-op.
    """
    h_change, d_change = _delta_a()
    # Arm with all-no-op delta.
    h_noop = _make_harness()
    d_noop = ImprovementDelta()

    pairs = [(h_change, d_change), (h_noop, d_noop)]
    chosen_harness, chosen_delta, vote_outcome = aggregate_deltas(
        pairs, strategy="vote", baseline_harness=_make_harness(),
    )

    # Tie-break: the change wins on every contested field.
    assert chosen_delta.system_prompt_delta == "Prefer short solutions."
    assert chosen_delta.rollout_protocol_changed is True
    # Three fields actually tied (the change arm proposes a value; the noop
    # arm proposes the empty/False value). The other two fields (tool_surface,
    # context_manager) are unanimous noops — no tie there.
    assert set(vote_outcome["ties_broken"]) == {
        "system_prompt_delta", "rollout_protocol_changed", "budget_delta",
    }


def test_vote_tie_falls_back_to_first_value_when_all_noops():
    """All-noop tie: the aggregator must still return a (no-op) delta."""
    h_noop = _make_harness()
    d_noop_1 = ImprovementDelta(improvement_summary="A says nothing")
    d_noop_2 = ImprovementDelta(improvement_summary="B says nothing")
    pairs = [(h_noop, d_noop_1), (h_noop, d_noop_2)]
    chosen_harness, chosen_delta, vote_outcome = aggregate_deltas(
        pairs, strategy="vote", baseline_harness=_make_harness(),
    )

    assert chosen_delta.system_prompt_delta == ""
    assert chosen_delta.rollout_protocol_changed is False
    # Harness should match the (deep-copy of) baseline.
    assert chosen_harness.system_prompt == "solve coding problems"


# ---------------------------------------------------------------------------
# (c) aggregate_deltas — parallel best-arm
# ---------------------------------------------------------------------------


def test_parallel_strategy_keeps_best_by_score_arm():
    """Strategy B: highest-scoring arm wins outright."""
    pairs = [_delta_a(), _delta_b(), _delta_c()]
    arm_scores = [0.6, 0.7, 0.9]  # C wins despite being the lone dissenter
    chosen_harness, chosen_delta, vote_outcome = aggregate_deltas(
        pairs, strategy="parallel", arm_scores=arm_scores,
    )

    assert chosen_delta.improvement_summary == "C: switch to hierarchical context"
    assert chosen_delta.context_manager_changed is True
    assert chosen_harness.context_manager == ContextManager.HIERARCHICAL
    assert vote_outcome["selected_instance_idx"] == 2
    assert vote_outcome["selected_score"] == pytest.approx(0.9)
    assert vote_outcome["arm_scores"] == [0.6, 0.7, 0.9]


def test_parallel_strategy_requires_arm_scores():
    pairs = [_delta_a(), _delta_b()]
    with pytest.raises(ValueError, match="arm_scores"):
        aggregate_deltas(pairs, strategy="parallel")
    with pytest.raises(ValueError, match="arm_scores"):
        aggregate_deltas(pairs, strategy="parallel", arm_scores=[0.5])


def test_invalid_strategy_raises():
    pairs = [_delta_a()]
    with pytest.raises(ValueError, match="strategy"):
        aggregate_deltas(pairs, strategy="bogus")


def test_empty_pairs_raises():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_deltas([], strategy="vote")


# ---------------------------------------------------------------------------
# (d) MultiImproverEnsemble — three independent wrapper calls
# ---------------------------------------------------------------------------


class _StubWrapper:
    """Records its own ``improve`` calls and returns a configurable pair."""

    def __init__(self, harness: HarnessConfig, delta: ImprovementDelta) -> None:
        self._h = harness
        self._d = delta
        self.calls: list[dict] = []

    def improve(self, h, r, *, obs=None, iteration=0, revert_history=None,
                cross_advocate_context=None):
        self.calls.append({
            "h": h, "r": r, "obs": obs, "iteration": iteration,
            "revert_history": revert_history,
            "cross_advocate_context": cross_advocate_context,
        })
        return self._h, self._d


def _make_factory_with_distinct_arms(arms):
    """Build a factory that hands out a fresh stub wrapper per call.

    Each call returns the next stub from ``arms`` — verifying the ensemble
    instantiates N anonymous instances (no shared state).
    """
    iterator = iter(arms)
    instantiated: list[_StubWrapper] = []

    def factory():
        try:
            wrapper = next(iterator)
        except StopIteration:
            pytest.fail("factory called more times than arms provided")
        instantiated.append(wrapper)
        return wrapper

    return factory, instantiated


def test_three_anonymous_instances_each_get_one_call():
    a_h, a_d = _delta_a()
    b_h, b_d = _delta_b()
    c_h, c_d = _delta_c()
    arms = [_StubWrapper(a_h, a_d), _StubWrapper(b_h, b_d), _StubWrapper(c_h, c_d)]
    factory, instantiated = _make_factory_with_distinct_arms(arms)

    ensemble = MultiImproverEnsemble(
        n_instances=3, strategy="vote", wrapper_factory=factory,
    )
    chosen_h, chosen_d = ensemble.improve(
        _make_harness(), _make_bench_result(), iteration=2,
    )

    assert len(instantiated) == 3, "factory must be called once per arm"
    # Each arm's wrapper sees exactly one improve() call.
    for w in instantiated:
        assert len(w.calls) == 1
        assert w.calls[0]["iteration"] == 2
    # Vote-merged delta has A+B's system_prompt change (2/3 majority).
    assert chosen_d.system_prompt_delta == "Prefer short solutions."


# ---------------------------------------------------------------------------
# (e) Single-instance fallback (n=1) — no ensemble event, identical to legacy
# ---------------------------------------------------------------------------


def test_n_eq_1_short_circuits_and_emits_no_ensemble_event(tmp_path: Path,
                                                            monkeypatch):
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    debug_file = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)

    a_h, a_d = _delta_a()
    arms = [_StubWrapper(a_h, a_d)]
    factory, _ = _make_factory_with_distinct_arms(arms)

    ensemble = MultiImproverEnsemble(
        n_instances=1, strategy="vote", wrapper_factory=factory,
    )
    chosen_h, chosen_d = ensemble.improve(
        _make_harness(), _make_bench_result(), obs=obs, iteration=0,
    )

    # Returned pair is bit-identical to what the single arm produced.
    assert chosen_h is a_h
    assert chosen_d is a_d

    # No ensemble event should have been emitted.
    if debug_file.exists():
        lines = debug_file.read_text().splitlines()
        ensemble_events = [
            json.loads(line) for line in lines
            if CHANNEL_IMPROVER_ENSEMBLE in line
        ]
        assert ensemble_events == []


# ---------------------------------------------------------------------------
# (f) Ensemble event fires on the bus when n>=2
# ---------------------------------------------------------------------------


def test_ensemble_event_fires_on_bus(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    debug_file = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)

    a_h, a_d = _delta_a()
    b_h, b_d = _delta_b()
    c_h, c_d = _delta_c()
    arms = [_StubWrapper(a_h, a_d), _StubWrapper(b_h, b_d), _StubWrapper(c_h, c_d)]
    factory, _ = _make_factory_with_distinct_arms(arms)

    ensemble = MultiImproverEnsemble(
        n_instances=3, strategy="vote", wrapper_factory=factory,
    )
    ensemble.improve(
        _make_harness(), _make_bench_result(), obs=obs, iteration=4,
    )

    assert debug_file.exists(), "debug file must exist after publish"
    lines = debug_file.read_text().splitlines()
    ensemble_events = [
        json.loads(line) for line in lines
        if f'"{CHANNEL_IMPROVER_ENSEMBLE}"' in line
    ]
    assert len(ensemble_events) == 1
    event = ensemble_events[0]
    assert event["type"] == CHANNEL_IMPROVER_ENSEMBLE
    data = event["data"]
    assert data["iteration"] == 4
    assert data["strategy"] == "vote"
    assert data["n_instances"] == 3
    assert len(data["instances"]) == 3
    for idx, instance in enumerate(data["instances"]):
        assert instance["instance_idx"] == idx
        assert "delta_summary" in instance
        assert "selected" in instance
    assert "field_votes" in data["vote_outcome"]


# ---------------------------------------------------------------------------
# (g) Parallel-strategy ensemble: best arm wins, arm scores emitted
# ---------------------------------------------------------------------------


def test_parallel_ensemble_emits_arm_scores(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    debug_file = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)

    a_h, a_d = _delta_a()
    b_h, b_d = _delta_b()
    c_h, c_d = _delta_c()
    arms = [_StubWrapper(a_h, a_d), _StubWrapper(b_h, b_d), _StubWrapper(c_h, c_d)]
    factory, _ = _make_factory_with_distinct_arms(arms)

    # arm_evaluator: arm-2 (C) scores 0.9, arm-1 (B) scores 0.7, arm-0 (A) 0.6.
    score_map: dict[int, float] = {}

    def arm_evaluator(cand_harness):
        # Look up by which arm produced this harness (the stubs return their
        # canned harness verbatim, so identity matters).
        if cand_harness is c_h:
            return 0.9
        if cand_harness is b_h:
            return 0.7
        return 0.6

    ensemble = MultiImproverEnsemble(
        n_instances=3, strategy="parallel", wrapper_factory=factory,
        arm_evaluator=arm_evaluator,
    )
    chosen_h, chosen_d = ensemble.improve(
        _make_harness(), _make_bench_result(), obs=obs, iteration=7,
    )

    # C wins by best-arm.
    assert chosen_h is c_h
    assert chosen_d is c_d

    lines = debug_file.read_text().splitlines()
    ensemble_events = [
        json.loads(line) for line in lines
        if f'"{CHANNEL_IMPROVER_ENSEMBLE}"' in line
    ]
    assert len(ensemble_events) == 1
    data = ensemble_events[0]["data"]
    assert data["strategy"] == "parallel"
    assert data["vote_outcome"]["selected_instance_idx"] == 2
    assert data["vote_outcome"]["selected_score"] == pytest.approx(0.9)
    # Each instance carries its forward-eval score.
    scores = [inst["score"] for inst in data["instances"]]
    assert scores == pytest.approx([0.6, 0.7, 0.9])


# ---------------------------------------------------------------------------
# (h) RSI loop integration — minimax_ensemble resolves to the ensemble path
# ---------------------------------------------------------------------------


def test_self_improving_harness_resolves_minimax_ensemble(monkeypatch):
    """``default_improver='minimax_ensemble'`` returns a callable that fans out."""
    from autobench.llm import ensemble as mim_mod
    from autobench.rsi.loop import SelfImprovingHarness

    a_h, a_d = _delta_a()
    b_h, b_d = _delta_b()
    c_h, c_d = _delta_c()
    arms = [_StubWrapper(a_h, a_d), _StubWrapper(b_h, b_d), _StubWrapper(c_h, c_d)]
    factory, _ = _make_factory_with_distinct_arms(arms)

    # Patch the module-level default factory to inject our stubs while still
    # exercising the real `_resolve_improver_fn` code path.
    monkeypatch.setattr(mim_mod, "_default_wrapper_factory", factory)

    sih = SelfImprovingHarness(
        current_harness=_make_harness(),
        evaluator=MagicMock(),
        default_improver="minimax_ensemble",
    )
    resolved = sih._resolve_improver_fn(None)
    new_h, new_d = resolved(_make_harness(), _make_bench_result(), iteration=1)

    # Vote-merged delta from A+B majority.
    assert new_d.system_prompt_delta == "Prefer short solutions."


def test_self_improving_harness_legacy_minimax_path_unchanged(monkeypatch):
    """``default_improver='minimax'`` does NOT touch the ensemble path.

    Regression guard for the bead's "zero behavioral regression on the
    single-improver path" acceptance criterion.
    """
    from autobench.llm import minimax as mm_mod
    from autobench.rsi.loop import SelfImprovingHarness

    a_h, a_d = _delta_a()
    captured: list[_StubWrapper] = []

    class _SingletonFactory:
        def __init__(self):
            self.wrapper = _StubWrapper(a_h, a_d)

        def __call__(self, *args, **kwargs):
            captured.append(self.wrapper)
            return self.wrapper

    fake_factory = _SingletonFactory()
    monkeypatch.setattr(mm_mod, "MiniMaxLLMWrapper", fake_factory)

    sih = SelfImprovingHarness(
        current_harness=_make_harness(),
        evaluator=MagicMock(),
        default_improver="minimax",
    )
    resolved = sih._resolve_improver_fn(None)
    new_h, new_d = resolved(_make_harness(), _make_bench_result(), iteration=0)

    # Exactly ONE wrapper instantiated — the legacy single-improver path.
    assert len(captured) == 1
    assert new_d is a_d


# ---------------------------------------------------------------------------
# (i) Env-driven defaults
# ---------------------------------------------------------------------------


def test_env_drives_strategy_default(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_IMPROVER_STRATEGY", "parallel")
    ens = MultiImproverEnsemble(
        n_instances=2,
        wrapper_factory=lambda: _StubWrapper(*_delta_a()),
        arm_evaluator=lambda _h: 0.5,
    )
    assert ens.strategy == "parallel"


def test_env_drives_n_instances_default(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_IMPROVER_ENSEMBLE_N", "5")
    ens = MultiImproverEnsemble(
        wrapper_factory=lambda: _StubWrapper(*_delta_a()),
    )
    assert ens.n_instances == 5


def test_env_unset_uses_default_n_instances(monkeypatch):
    monkeypatch.delenv("AUTOBENCH_IMPROVER_ENSEMBLE_N", raising=False)
    monkeypatch.delenv("AUTOBENCH_IMPROVER_STRATEGY", raising=False)
    ens = MultiImproverEnsemble(
        wrapper_factory=lambda: _StubWrapper(*_delta_a()),
    )
    assert ens.n_instances == DEFAULT_N_INSTANCES
    assert ens.strategy == "vote"


# ---------------------------------------------------------------------------
# (j) Schema sanity — the emitted event validates against the v1 schema
# ---------------------------------------------------------------------------


def test_emitted_event_matches_schema(tmp_path: Path, monkeypatch):
    """Per the schema-first rule, the emitted event must match the v1 schema."""
    pytest.importorskip("jsonschema")
    import jsonschema

    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    debug_file = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug_file)

    a_h, a_d = _delta_a()
    b_h, b_d = _delta_b()
    arms = [_StubWrapper(a_h, a_d), _StubWrapper(b_h, b_d)]
    factory, _ = _make_factory_with_distinct_arms(arms)

    ensemble = MultiImproverEnsemble(
        n_instances=2, strategy="vote", wrapper_factory=factory,
    )
    ensemble.improve(
        _make_harness(), _make_bench_result(), obs=obs, iteration=0,
    )

    from tests._paths import SCHEMA_DIR
    schema_path = SCHEMA_DIR / "autobench.improver.ensemble.v1.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    lines = debug_file.read_text().splitlines()
    events = [
        json.loads(line) for line in lines
        if f'"{CHANNEL_IMPROVER_ENSEMBLE}"' in line
    ]
    assert len(events) == 1
    # ``time`` may have ms precision (date-time format accepts it). Validate.
    validator.validate(events[0])


# ---------------------------------------------------------------------------
# (k) Partial-failure resilience — one arm raises, ensemble proceeds
# ---------------------------------------------------------------------------


class _RaisingWrapper:
    def improve(self, h, r, **kwargs):
        raise RuntimeError("simulated MiniMax 500")


def test_one_failing_arm_does_not_kill_the_ensemble():
    a_h, a_d = _delta_a()
    b_h, b_d = _delta_b()
    arms_seq = iter([_StubWrapper(a_h, a_d), _RaisingWrapper(), _StubWrapper(b_h, b_d)])
    ensemble = MultiImproverEnsemble(
        n_instances=3, strategy="vote",
        wrapper_factory=lambda: next(arms_seq),
    )
    chosen_h, chosen_d = ensemble.improve(_make_harness(), _make_bench_result())
    # A and B survive, both propose the same change — majority of survivors.
    assert chosen_d.system_prompt_delta == "Prefer short solutions."


def test_all_arms_failing_raises():
    raising = iter([_RaisingWrapper(), _RaisingWrapper()])
    ensemble = MultiImproverEnsemble(
        n_instances=2, strategy="vote",
        wrapper_factory=lambda: next(raising),
    )
    with pytest.raises(RuntimeError, match="simulated"):
        ensemble.improve(_make_harness(), _make_bench_result())
