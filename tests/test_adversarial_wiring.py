"""Phase 3 wiring tests — adversarial gotcha mix in benchmark assembly.

Covers nervous-bus-gdzo (wire-pop Phase 3):
    * ``generate_adversarial_case_mix`` returns N curveballs and emits the
      ``curveball_generated`` + ``round_complete`` event types.
    * ``assemble_benchmark_cases`` swaps ~20% of cases for curveballs while
      preserving total length, and emits both event types during a synthetic
      population run.
    * ``mine_failure_modes_from_result`` extracts modes from a prior
      :class:`PopulationResult` and falls back to defaults on empty input.
    * Mix ratio is correct (rounded up) for various corpus sizes.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from autobench.adversarial import (
    DEFAULT_FAILURE_MODES,
    AdversarialCase,
    AdversarialGenerator,
    _round_up_ratio,
    generate_adversarial_case_mix,
    mine_failure_modes_from_result,
)
from autobench.benchmark_assembly import (
    DEFAULT_ADVERSARIAL_RATIO,
    assemble_benchmark_cases,
)
from autobench.evaluator import BenchmarkCase
from autobench.observability import (
    CHANNEL_ADVERSARIAL_GENERATED,
    CHANNEL_ADVERSARIAL_ROUND,
    AutobenchObservability,
)




# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force zellij missing so observability falls back to a temp debug file."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    # Don't leak a real API key into tests — generator must use static fallback.
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_obs(debug_path: Path) -> AutobenchObservability:
    return AutobenchObservability(debug_file=debug_path)


def _base_case(i: int) -> BenchmarkCase:
    return BenchmarkCase(
        id=f"base-{i:03d}",
        prompt=f"problem #{i}",
        language="python",
        expected_output="42\n",
        test_inputs=[""],
    )


# --------------------------------------------------------------------------- #
# _round_up_ratio
# --------------------------------------------------------------------------- #


def test_round_up_ratio_one_in_five() -> None:
    """5 cases, 0.20 ratio → 1 curveball (ceil)."""
    assert _round_up_ratio(5, 0.20) == 1


def test_round_up_ratio_two_in_ten() -> None:
    """10 cases, 0.20 ratio → 2 curveballs."""
    assert _round_up_ratio(10, 0.20) == 2


def test_round_up_ratio_rounds_up() -> None:
    """6 cases, 0.20 ratio → 6*0.2=1.2 → ceil = 2."""
    assert _round_up_ratio(6, 0.20) == 2


def test_round_up_ratio_clamps_to_total() -> None:
    """Asking for more than 100% returns total, not more."""
    assert _round_up_ratio(5, 2.0) == 5


def test_round_up_ratio_zero_inputs() -> None:
    assert _round_up_ratio(0, 0.5) == 0
    assert _round_up_ratio(10, 0.0) == 0


# --------------------------------------------------------------------------- #
# mine_failure_modes_from_result
# --------------------------------------------------------------------------- #


@dataclass
class _StubResult:
    """Minimal duck-typed BenchmarkResult for mining tests."""
    verdict_counts: dict[str, int] = field(default_factory=dict)
    aggregate_score: float = 0.0


@dataclass
class _StubAdvocate:
    history: list[tuple[Any, Any, Any]] = field(default_factory=list)


@dataclass
class _StubPopulationResult:
    advocates: list[_StubAdvocate] = field(default_factory=list)


def test_mine_failure_modes_none_returns_empty() -> None:
    assert mine_failure_modes_from_result(None) == []


def test_mine_failure_modes_no_history_returns_empty() -> None:
    pr = _StubPopulationResult(advocates=[_StubAdvocate(history=[])])
    assert mine_failure_modes_from_result(pr) == []


def test_mine_failure_modes_maps_verdicts() -> None:
    """WA in last iter should surface as 'off_by_one'."""
    last_result = _StubResult(verdict_counts={"OK": 2, "WA": 3, "TLE": 1})
    pr = _StubPopulationResult(advocates=[
        _StubAdvocate(history=[(object(), last_result, object())]),
    ])
    modes = mine_failure_modes_from_result(pr)
    # off_by_one (from WA=3) should outrank integer_overflow (from TLE=1).
    assert "off_by_one" in modes
    assert "integer_overflow" in modes
    assert modes.index("off_by_one") < modes.index("integer_overflow")


def test_mine_failure_modes_ignores_ok() -> None:
    """OK verdicts must not pollute the mode list."""
    last_result = _StubResult(verdict_counts={"OK": 10})
    pr = _StubPopulationResult(advocates=[
        _StubAdvocate(history=[(object(), last_result, object())]),
    ])
    assert mine_failure_modes_from_result(pr) == []


# --------------------------------------------------------------------------- #
# generate_adversarial_case_mix
# --------------------------------------------------------------------------- #


def test_generate_adversarial_case_mix_returns_n_cases(debug_file: Path) -> None:
    """Asking for 4 cases yields 4 BenchmarkCases, all adversarial-marked."""
    obs = _make_obs(debug_file)
    cases = generate_adversarial_case_mix(
        n_cases=4,
        failure_modes=list(DEFAULT_FAILURE_MODES),
        obs=obs,
    )
    assert len(cases) == 4
    assert all(isinstance(c, BenchmarkCase) for c in cases)
    assert all(c.metadata.get("adversarial") is True for c in cases)


def test_generate_adversarial_case_mix_emits_both_event_types(debug_file: Path) -> None:
    """One curveball_generated per case + one round_complete summary."""
    obs = _make_obs(debug_file)
    cases = generate_adversarial_case_mix(
        n_cases=3,
        failure_modes=["off_by_one"],
        obs=obs,
    )
    assert len(cases) == 3

    events = _read_events(debug_file)
    types = [e.get("type") for e in events]
    assert types.count(CHANNEL_ADVERSARIAL_GENERATED) == 3, (
        f"expected 3 curveball_generated events, got {types}"
    )
    assert types.count(CHANNEL_ADVERSARIAL_ROUND) == 1, (
        f"expected 1 round_complete event, got {types}"
    )


def test_generate_adversarial_case_mix_zero_returns_empty(debug_file: Path) -> None:
    obs = _make_obs(debug_file)
    assert generate_adversarial_case_mix(n_cases=0, obs=obs) == []


def test_generate_adversarial_case_mix_default_modes_when_empty(debug_file: Path) -> None:
    """Empty failure_modes argument falls back to DEFAULT_FAILURE_MODES."""
    obs = _make_obs(debug_file)
    cases = generate_adversarial_case_mix(n_cases=2, failure_modes=[], obs=obs)
    assert len(cases) == 2
    # Each case's target_failure_mode in metadata should be one of the defaults.
    for c in cases:
        assert c.metadata.get("target_failure_mode") in DEFAULT_FAILURE_MODES


# --------------------------------------------------------------------------- #
# assemble_benchmark_cases
# --------------------------------------------------------------------------- #


def test_assemble_preserves_total_length(debug_file: Path) -> None:
    """Replacing 20% of 10 cases yields 10 cases out — not 12."""
    obs = _make_obs(debug_file)
    base = [_base_case(i) for i in range(10)]
    out = assemble_benchmark_cases(
        base_cases=base,
        prior_result=None,
        adversarial_ratio=0.20,
        obs=obs,
        rng=random.Random(42),
    )
    assert len(out) == 10


def test_assemble_mix_ratio_two_of_ten(debug_file: Path) -> None:
    """At 0.20 with 10 cases → exactly 2 are adversarial."""
    obs = _make_obs(debug_file)
    base = [_base_case(i) for i in range(10)]
    out = assemble_benchmark_cases(
        base_cases=base,
        prior_result=None,
        adversarial_ratio=0.20,
        obs=obs,
        rng=random.Random(7),
    )
    adv_count = sum(1 for c in out if c.metadata.get("adversarial") is True)
    assert adv_count == 2, f"expected 2 adversarial cases, got {adv_count}"


def test_assemble_mix_ratio_one_of_five(debug_file: Path) -> None:
    """At 0.20 with 5 cases → ceil = 1 adversarial."""
    obs = _make_obs(debug_file)
    base = [_base_case(i) for i in range(5)]
    out = assemble_benchmark_cases(
        base_cases=base,
        prior_result=None,
        adversarial_ratio=0.20,
        obs=obs,
        rng=random.Random(0),
    )
    adv_count = sum(1 for c in out if c.metadata.get("adversarial") is True)
    assert adv_count == 1, f"expected 1 adversarial case, got {adv_count}"


def test_assemble_emits_both_event_types(debug_file: Path) -> None:
    """A synthetic assembly call emits curveball_generated AND round_complete."""
    obs = _make_obs(debug_file)
    base = [_base_case(i) for i in range(10)]
    _ = assemble_benchmark_cases(
        base_cases=base,
        prior_result=None,
        adversarial_ratio=0.20,
        obs=obs,
        rng=random.Random(1),
    )

    events = _read_events(debug_file)
    types = [e.get("type") for e in events]
    # 2 curveballs → 2 curveball_generated; 1 assembly → 1 round_complete.
    assert types.count(CHANNEL_ADVERSARIAL_GENERATED) == 2
    assert types.count(CHANNEL_ADVERSARIAL_ROUND) == 1


def test_assemble_zero_ratio_is_passthrough(debug_file: Path) -> None:
    """adversarial_ratio=0 returns base_cases unmodified — no emissions."""
    obs = _make_obs(debug_file)
    base = [_base_case(i) for i in range(10)]
    out = assemble_benchmark_cases(
        base_cases=base,
        prior_result=None,
        adversarial_ratio=0.0,
        obs=obs,
    )
    assert [c.id for c in out] == [c.id for c in base]
    # No events should have fired.
    events = _read_events(debug_file)
    types = [e.get("type") for e in events]
    assert CHANNEL_ADVERSARIAL_GENERATED not in types
    assert CHANNEL_ADVERSARIAL_ROUND not in types


def test_assemble_empty_base_returns_empty(debug_file: Path) -> None:
    obs = _make_obs(debug_file)
    assert assemble_benchmark_cases([], obs=obs) == []


def test_assemble_uses_mined_failure_modes(debug_file: Path) -> None:
    """When a prior_result is provided, mined modes drive the curveballs."""
    obs = _make_obs(debug_file)
    # Prior result has heavy WA → off_by_one should dominate.
    last_result = _StubResult(verdict_counts={"OK": 1, "WA": 8})
    pr = _StubPopulationResult(advocates=[
        _StubAdvocate(history=[(object(), last_result, object())]),
    ])
    base = [_base_case(i) for i in range(10)]
    out = assemble_benchmark_cases(
        base_cases=base,
        prior_result=pr,
        adversarial_ratio=0.20,
        obs=obs,
        rng=random.Random(3),
    )
    adv = [c for c in out if c.metadata.get("adversarial") is True]
    assert len(adv) == 2
    # All curveballs should target off_by_one (only mode mined).
    assert all(c.metadata.get("target_failure_mode") == "off_by_one" for c in adv), (
        f"expected all adv cases to target off_by_one, got "
        f"{[c.metadata.get('target_failure_mode') for c in adv]}"
    )


def test_assemble_default_ratio_constant() -> None:
    """The module-level default ratio matches the bead spec (20%)."""
    assert DEFAULT_ADVERSARIAL_RATIO == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# PopulationRunner integration — adversarial_ratio threading
# --------------------------------------------------------------------------- #


def test_population_runner_threads_adversarial_ratio(debug_file: Path) -> None:
    """PopulationRunner(adversarial_ratio=0.2).run() emits both event types.

    The runner should apply the mix step at the start of ``run()`` and the
    resulting events should land on the adversarial_obs we pass in.
    """
    from autobench.core import HarnessConfig, HarnessResult, Verdict
    from autobench.evaluator import BenchmarkResult
    from autobench.population import PopulationRunner

    class _ScriptedEvaluator:
        def run(
            self,
            harness: HarnessConfig,
            cases: list[Any],
            obs: Any = None,
        ) -> BenchmarkResult:
            return BenchmarkResult(
                case_results=[
                    HarnessResult(p_score=1.0, verdict=Verdict.OK) for _ in cases
                ],
                aggregate_score=1.0,
                total_latency_ms=0.0,
                verdict_counts={"OK": len(cases)},
                metadata={},
            )

    base = [_base_case(i) for i in range(5)]
    adv_obs = _make_obs(debug_file)

    runner = PopulationRunner(
        n_advocates=1,
        initial_harness_factory=lambda: HarnessConfig(),
        evaluator_factory=lambda: _ScriptedEvaluator(),
        observability_factory=lambda: AutobenchObservability(),
        improver="minimax",
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=None,
        adversarial_ratio=0.20,
        prior_result=None,
        adversarial_obs=adv_obs,
    )

    _ = runner.run(base)

    events = _read_events(debug_file)
    types = [e.get("type") for e in events]
    # 5 cases at 0.20 → 1 curveball → 1 curveball_generated + 1 round_complete.
    assert types.count(CHANNEL_ADVERSARIAL_GENERATED) >= 1, (
        f"expected >=1 curveball_generated, got types={types}"
    )
    assert types.count(CHANNEL_ADVERSARIAL_ROUND) >= 1, (
        f"expected >=1 round_complete, got types={types}"
    )
