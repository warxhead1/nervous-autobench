"""Phase 2 wiring tests — cross-advocate diversity in PopulationRunner.

Covers nervous-bus-bo86:
    * Lineage-signature distance correlates with structural delta distance.
    * adjusted_score promotes an exploratory lineage over a slightly-better
      but redundant one when diversity_weight is high enough.
    * diversity_weight=0 reproduces Phase-1 winner_id bit-for-bit.
    * The SIBLINGS prompt block contains each completed sibling's recent
      delta when the runner threads cross_advocate_context through.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from autobench.core import HarnessConfig, HarnessResult, Verdict
from autobench.diversity import (
    lineage_signature,
    pairwise_lineage_distance,
)
from autobench.evaluator import BenchmarkResult
from autobench.minimax_improver import _format_siblings_block
from autobench.observability import (
    CHANNEL_POPULATION_SUMMARY,
    AutobenchObservability,
)
from autobench.population import (
    PopulationResult,
    PopulationRunner,
    _read_diversity_weight_env,
)
from autobench.rsi_loop import ImprovementDelta


from tests._paths import SCHEMA_DIR
SCHEMA_PATH = SCHEMA_DIR / "autobench.population.summary.v1.json"


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #


def _dummy_case_result(verdict: Verdict = Verdict.OK) -> HarnessResult:
    return HarnessResult(p_score=1.0 if verdict == Verdict.OK else 0.0, verdict=verdict)


class _ScriptedEvaluator:
    """Returns scripted aggregate_score per call; advances index per ``run``."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.run_count = 0

    def run(
        self,
        harness: HarnessConfig,
        cases: list[Any],
        obs: Any = None,
    ) -> BenchmarkResult:
        idx = min(self.run_count, len(self._scores) - 1)
        self.run_count += 1
        return BenchmarkResult(
            case_results=[_dummy_case_result()],
            aggregate_score=self._scores[idx],
            total_latency_ms=0.0,
            verdict_counts={"OK": 1},
            metadata={},
        )


def _make_scripted_improver(per_advocate_deltas: dict[int, list[ImprovementDelta]]):
    """Build an improver_fn that emits a per-advocate, per-iteration delta.

    The advocate index is recovered from the harness's system_prompt prefix
    (we tag the initial harness with ``A{n}:`` so the same improver_fn can
    route per-advocate without state).
    """
    # call_counter scoped per (advocate_idx, iter_idx) via closure
    call_counts: dict[int, int] = {}

    def _improver(
        harness: HarnessConfig,
        result: BenchmarkResult,
        iteration: int = 0,
        revert_history: list[Any] | None = None,
        cross_advocate_context: list[Any] | None = None,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        # Parse the advocate marker from the system_prompt prefix.
        prefix = (harness.system_prompt or "")[:4]
        try:
            adv_idx = int(prefix.split(":")[0].lstrip("A"))
        except (ValueError, IndexError):
            adv_idx = 0
        deltas = per_advocate_deltas.get(adv_idx, [])
        idx = call_counts.get(adv_idx, 0)
        call_counts[adv_idx] = idx + 1
        if not deltas:
            d = ImprovementDelta(improvement_summary=f"a{adv_idx}-i{iteration}")
        else:
            d = deltas[min(idx, len(deltas) - 1)]
        # Apply the delta to the harness in the most minimal way we can —
        # the test only cares about the delta history, not the harness
        # contents themselves.
        new = replace(
            harness,
            system_prompt=(harness.system_prompt or "") + f"|i{iteration}",
        )
        return new, d

    return _improver, call_counts


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force debug-file emission (no zellij in PATH)."""
    path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Env parsing
# --------------------------------------------------------------------------- #


def test_diversity_weight_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOBENCH_DIVERSITY_WEIGHT", raising=False)
    assert _read_diversity_weight_env() == pytest.approx(0.10)


def test_diversity_weight_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_DIVERSITY_WEIGHT", "0.25")
    assert _read_diversity_weight_env() == pytest.approx(0.25)


def test_diversity_weight_env_negative_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_DIVERSITY_WEIGHT", "-0.5")
    assert _read_diversity_weight_env() == 0.0


def test_diversity_weight_env_bad_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_DIVERSITY_WEIGHT", "not_a_number")
    assert _read_diversity_weight_env(default=0.10) == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# Lineage-signature distance
# --------------------------------------------------------------------------- #


def test_lineage_distance_close_for_similar_deltas() -> None:
    """Three lineages with near-identical structural deltas → mutual distance ~0."""
    d_sysprompt_only = ImprovementDelta(
        system_prompt_delta="STRENGTHEN code-only directive",
        improvement_summary="emphasize code-only output",
    )
    sig_a = lineage_signature([d_sysprompt_only])
    sig_b = lineage_signature([d_sysprompt_only])
    # Identical → distance must be 0 (cosine sim = 1, overlap = 1).
    assert pairwise_lineage_distance(sig_a, sig_b) == pytest.approx(0.0, abs=1e-9)


def test_lineage_distance_far_for_disjoint_field_deltas() -> None:
    """Two lineages touching disjoint fields → maximal pairwise distance.

    The SACS overlap_ratio is 0 when no fields are shared, so similarity is
    0 and distance is 1.0.
    """
    d_sysprompt = ImprovementDelta(
        system_prompt_delta="emphasize code-only",
        improvement_summary="sysprompt only",
    )
    d_budget = ImprovementDelta(
        budget_delta={"max_tokens": 6000},
        improvement_summary="budget only",
    )
    sig_a = lineage_signature([d_sysprompt])
    sig_b = lineage_signature([d_budget])
    assert pairwise_lineage_distance(sig_a, sig_b) == pytest.approx(1.0, abs=1e-9)


def test_fingerprint_distance_correlates_with_delta_distance() -> None:
    """3 advocates: two nearly-identical, one structurally different.

    The "odd-one-out" lineage must score higher mean pairwise distance than
    either of the two convergent siblings.
    """
    d_same_a = ImprovementDelta(
        system_prompt_delta="add code-only directive",
        improvement_summary="code-only",
    )
    d_same_b = ImprovementDelta(
        system_prompt_delta="STRENGTHEN code-only",
        improvement_summary="reinforce code-only",
    )
    d_diff = ImprovementDelta(
        budget_delta={"max_tokens": 4000, "max_time_seconds": 30},
        improvement_summary="tighten budget",
    )

    sigs = [
        lineage_signature([d_same_a]),
        lineage_signature([d_same_b]),
        lineage_signature([d_diff]),
    ]

    def mean_dist(i: int) -> float:
        ds = [pairwise_lineage_distance(sigs[i], sigs[j]) for j in range(3) if j != i]
        return sum(ds) / len(ds)

    d0 = mean_dist(0)
    d1 = mean_dist(1)
    d2 = mean_dist(2)
    # The odd-one-out lineage (index 2) must be more diverse than either
    # of the two convergent ones.
    assert d2 > d0
    assert d2 > d1


# --------------------------------------------------------------------------- #
# Adjusted-score winner promotion
# --------------------------------------------------------------------------- #


def _build_runner_for_three_advocates(
    debug_file: Path,
    scripts: list[_ScriptedEvaluator],
    improver_fn,
    diversity_weight: float,
) -> PopulationRunner:
    obs_list = [
        AutobenchObservability(session_id=f"ADV-{i}" + "A" * 21, debug_file=debug_file)
        for i in range(3)
    ]
    pending_obs = iter(obs_list)
    pending_eval = iter(scripts)
    harness_idx = iter(range(3))

    # Per Phase-1 fixture pattern: supply a run_summary_obs so the summary
    # event lands in debug_file (default constructor doesn't see the path).
    summary_obs = AutobenchObservability(
        session_id="POP-SUM-XXXXXXXXXXXXXXXX", debug_file=debug_file,
    )

    runner = PopulationRunner(
        n_advocates=3,
        initial_harness_factory=lambda: HarnessConfig(
            system_prompt=f"A{next(harness_idx)}:BASE",
        ),
        evaluator_factory=lambda: next(pending_eval),
        observability_factory=lambda: next(pending_obs),
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
        diversity_penalty_weight=diversity_weight,
        run_summary_obs=summary_obs,
    )
    # Inject our improver via the SelfImprovingHarness's improve(improver_fn=...).
    # The runner calls sih.improve(benchmark_cases) with no improver_fn, so we
    # need to monkeypatch its _resolve_improver_fn behavior. The cleanest hook
    # is to override _run_one_advocate to thread our test improver through.
    orig_run_one = runner._run_one_advocate

    def _patched_run_one(index: int, cases: list[Any], **_kw: Any):
        from autobench.rsi_loop import SelfImprovingHarness
        from autobench.budget_guard import BudgetGuard  # noqa: F401 — kept for parity

        adv_id = runner._advocate_id_for(index)
        obs = runner._make_observability()
        harness = runner.initial_harness_factory()
        evaluator = runner.evaluator_factory()
        sih = SelfImprovingHarness(
            current_harness=harness,
            evaluator=evaluator,
            max_iterations=runner.max_iterations_per_advocate,
            default_improver=None,
            obs=obs,
            budget_guard=None,
            cross_advocate_context=runner._cross_advocate_context,
            improvement_threshold=0.001,
        )
        final_h, final_r, history = sih.improve(cases, improver_fn=improver_fn)
        best_score = max(
            (float(r.aggregate_score) for (_h, r, _d) in history), default=0.0
        )
        best_iter = max(
            range(len(history)),
            key=lambda i: history[i][1].aggregate_score,
            default=-1,
        ) if history else -1
        from autobench.population import AdvocateResult
        return AdvocateResult(
            advocate_id=adv_id,
            session_id=obs.session_id,
            final_harness=final_h,
            final_result=final_r,
            history=history,
            best_score=best_score,
            best_iter=best_iter,
        )

    runner._run_one_advocate = _patched_run_one  # type: ignore[method-assign]
    return runner


def test_adjusted_score_favors_exploratory_lineage(debug_file: Path) -> None:
    """advocate-2 has the highest best_score but is convergent with sibling-1.

    advocate-0 has slightly lower best_score but maximal divergence from
    both siblings. With diversity_weight=0.5 (intentionally aggressive),
    advocate-0's adjusted_score must surpass advocate-2's.
    """
    # Advocate 0: edits BUDGET (disjoint field set from siblings 1 & 2).
    # Advocates 1 & 2: both edit SYSTEM_PROMPT (convergent).
    d_budget = ImprovementDelta(
        budget_delta={"max_tokens": 4000, "max_time_seconds": 30},
        improvement_summary="tighten budget",
    )
    d_sysprompt_a = ImprovementDelta(
        system_prompt_delta="add code-only directive",
        improvement_summary="code-only sysprompt",
    )
    d_sysprompt_b = ImprovementDelta(
        system_prompt_delta="STRENGTHEN code-only",
        improvement_summary="reinforce code-only sysprompt",
    )
    per_advocate = {0: [d_budget], 1: [d_sysprompt_a], 2: [d_sysprompt_b]}
    improver_fn, _ = _make_scripted_improver(per_advocate)

    # Scores: advocate-2 wins the raw race, advocate-0 a hair behind.
    scripts = [
        _ScriptedEvaluator(scores=[0.62]),   # advocate-0 — exploratory
        _ScriptedEvaluator(scores=[0.60]),   # advocate-1
        _ScriptedEvaluator(scores=[0.65]),   # advocate-2 — best raw, convergent
    ]
    runner = _build_runner_for_three_advocates(
        debug_file, scripts, improver_fn, diversity_weight=0.5,
    )
    result = runner.run(benchmark_cases=[])

    by_id = {a.advocate_id: a for a in result.advocates}
    # Sanity: raw winner is advocate-2.
    assert result.winner_id == "advocate-2"
    # Adjusted winner should be advocate-0 — its disjoint field set drives a
    # mean pairwise distance close to 1.0, while advocate-2's signature is
    # close to advocate-1's so its distance is much smaller.
    assert by_id["advocate-0"].diversity_score > by_id["advocate-2"].diversity_score
    assert result.adjusted_winner_id == "advocate-0"


def test_diversity_weight_zero_preserves_phase1_winner(debug_file: Path) -> None:
    """weight=0 → adjusted_winner_id == winner_id and equals raw best."""
    d_budget = ImprovementDelta(
        budget_delta={"max_tokens": 4000}, improvement_summary="tighten budget",
    )
    d_sysprompt_a = ImprovementDelta(
        system_prompt_delta="code-only", improvement_summary="code-only sysprompt",
    )
    d_sysprompt_b = ImprovementDelta(
        system_prompt_delta="STRENGTHEN code-only",
        improvement_summary="reinforce code-only",
    )
    per_advocate = {0: [d_budget], 1: [d_sysprompt_a], 2: [d_sysprompt_b]}
    improver_fn, _ = _make_scripted_improver(per_advocate)

    scripts = [
        _ScriptedEvaluator(scores=[0.62]),
        _ScriptedEvaluator(scores=[0.60]),
        _ScriptedEvaluator(scores=[0.65]),
    ]
    runner = _build_runner_for_three_advocates(
        debug_file, scripts, improver_fn, diversity_weight=0.0,
    )
    result = runner.run(benchmark_cases=[])

    # With weight=0, adjusted_score == best_score, so the adjusted winner
    # MUST match the raw winner (Phase-1 bit-for-bit compat guarantee).
    assert result.adjusted_winner_id == result.winner_id == "advocate-2"
    # And the diversity_weight on the result must be exactly 0.0.
    assert result.diversity_weight == 0.0


# --------------------------------------------------------------------------- #
# Siblings prompt block
# --------------------------------------------------------------------------- #


def test_format_siblings_block_empty_input_is_empty_string() -> None:
    assert _format_siblings_block(None) == ""
    assert _format_siblings_block([]) == ""


def test_format_siblings_block_renders_each_sibling() -> None:
    """Each provided sibling delta appears in the rendered block."""
    s1 = ImprovementDelta(
        improvement_summary="added STRENGTHEN code-only directive",
        system_prompt_delta="STRENGTHEN code-only",
    )
    s2 = ImprovementDelta(
        improvement_summary="switched rollout to iterative",
        rollout_protocol_changed=True,
    )
    block = _format_siblings_block([s1, s2])
    assert "SIBLINGS" in block
    assert "your siblings" not in block.lower() or "siblings" in block.lower()
    assert "sibling-0" in block
    assert "sibling-1" in block
    assert "STRENGTHEN code-only" in block
    assert "rollout_protocol_changed" in block
    # The summary lines must be quoted so the LLM sees them as quoted text.
    assert 'summary="added STRENGTHEN code-only directive"' in block


def test_siblings_context_threaded_to_improver(debug_file: Path) -> None:
    """The 3rd advocate's improver sees a SIBLINGS block referencing the first two.

    We assert this at the call layer: the improver_fn captures its
    cross_advocate_context kwarg per advocate, and we verify the third
    advocate received deltas matching the first two.
    """
    seen: list[tuple[int, list[Any] | None]] = []

    d0 = ImprovementDelta(
        improvement_summary="advocate-0 picked budget",
        budget_delta={"max_tokens": 4000},
    )
    d1 = ImprovementDelta(
        improvement_summary="advocate-1 picked sysprompt",
        system_prompt_delta="code-only",
    )
    d2 = ImprovementDelta(
        improvement_summary="advocate-2 picked rollout",
        rollout_protocol_changed=True,
    )
    per_advocate = {0: [d0], 1: [d1], 2: [d2]}

    def _improver(
        harness: HarnessConfig,
        result: BenchmarkResult,
        iteration: int = 0,
        revert_history: list[Any] | None = None,
        cross_advocate_context: list[Any] | None = None,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        # Recover advocate idx from the harness prefix.
        prefix = (harness.system_prompt or "")[:4]
        try:
            adv_idx = int(prefix.split(":")[0].lstrip("A"))
        except (ValueError, IndexError):
            adv_idx = -1
        # Capture a snapshot (the runner reuses the same list reference).
        snapshot = (
            list(cross_advocate_context) if cross_advocate_context is not None else None
        )
        seen.append((adv_idx, snapshot))
        d = per_advocate.get(adv_idx, [ImprovementDelta()])[0]
        new = replace(
            harness,
            system_prompt=(harness.system_prompt or "") + "|x",
        )
        return new, d

    scripts = [
        _ScriptedEvaluator(scores=[0.50]),
        _ScriptedEvaluator(scores=[0.55]),
        _ScriptedEvaluator(scores=[0.60]),
    ]
    runner = _build_runner_for_three_advocates(
        debug_file, scripts, _improver, diversity_weight=0.10,
    )
    runner.run(benchmark_cases=[])

    # Advocate-0 sees no siblings yet (it ran first).
    assert seen[0][0] == 0
    assert not seen[0][1]
    # Advocate-1 sees exactly one sibling (advocate-0's delta).
    assert seen[1][0] == 1
    assert seen[1][1] is not None and len(seen[1][1]) == 1
    assert seen[1][1][0].improvement_summary == "advocate-0 picked budget"
    # Advocate-2 sees both prior advocates' deltas.
    assert seen[2][0] == 2
    assert seen[2][1] is not None and len(seen[2][1]) == 2
    summaries = {d.improvement_summary for d in seen[2][1]}
    assert summaries == {"advocate-0 picked budget", "advocate-1 picked sysprompt"}

    # Rendered block for advocate-2 should mention both predecessors.
    rendered = _format_siblings_block(seen[2][1])
    assert "advocate-0 picked budget" in rendered
    assert "advocate-1 picked sysprompt" in rendered


# --------------------------------------------------------------------------- #
# Population summary emission carries new fields
# --------------------------------------------------------------------------- #


def test_population_summary_emits_adjusted_winner(debug_file: Path) -> None:
    """The bus envelope's data block carries adjusted_winner_id + diversity_weight."""
    d_budget = ImprovementDelta(
        budget_delta={"max_tokens": 4000}, improvement_summary="budget",
    )
    d_sys = ImprovementDelta(
        system_prompt_delta="code-only", improvement_summary="sysprompt",
    )
    per_advocate = {0: [d_budget], 1: [d_sys], 2: [d_sys]}
    improver_fn, _ = _make_scripted_improver(per_advocate)
    scripts = [
        _ScriptedEvaluator(scores=[0.50]),
        _ScriptedEvaluator(scores=[0.55]),
        _ScriptedEvaluator(scores=[0.60]),
    ]
    runner = _build_runner_for_three_advocates(
        debug_file, scripts, improver_fn, diversity_weight=0.10,
    )
    runner.run(benchmark_cases=[])

    events = _read_events(debug_file)
    summaries = [e for e in events if e.get("type") == CHANNEL_POPULATION_SUMMARY]
    assert len(summaries) == 1
    data = summaries[0]["data"]
    assert "adjusted_winner_id" in data
    assert data["diversity_weight"] == pytest.approx(0.10)
    # Per-advocate entries must carry the diversity fields.
    for a in data["advocates"]:
        assert "diversity_score" in a
        assert "adjusted_score" in a


def test_population_summary_validates_against_schema(debug_file: Path) -> None:
    """Emitted event must still validate against the bumped v1 schema."""
    pytest.importorskip("jsonschema")
    import jsonschema

    d_budget = ImprovementDelta(
        budget_delta={"max_tokens": 4000}, improvement_summary="budget",
    )
    d_sys = ImprovementDelta(
        system_prompt_delta="code-only", improvement_summary="sysprompt",
    )
    per_advocate = {0: [d_budget], 1: [d_sys], 2: [d_sys]}
    improver_fn, _ = _make_scripted_improver(per_advocate)
    scripts = [
        _ScriptedEvaluator(scores=[0.50]),
        _ScriptedEvaluator(scores=[0.55]),
        _ScriptedEvaluator(scores=[0.60]),
    ]
    runner = _build_runner_for_three_advocates(
        debug_file, scripts, improver_fn, diversity_weight=0.10,
    )
    runner.run(benchmark_cases=[])

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    events = _read_events(debug_file)
    summaries = [e for e in events if e.get("type") == CHANNEL_POPULATION_SUMMARY]
    assert len(summaries) == 1
    # We may need to fill in fields the schema requires that the debug
    # emitter doesn't synthesise (specversion/source/datacontenttype/etc.).
    env = summaries[0]
    env.setdefault("specversion", "1.0")
    env.setdefault("source", "/autobench")
    env.setdefault("datacontenttype", "application/json")
    validator.validate(env)
