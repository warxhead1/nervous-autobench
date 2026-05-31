"""Tests for the cross-domain benchmark registry (nervous-bus-qp91).

Verifies:
    * ``BenchmarkRegistry.default()`` builds the expected three-domain set
      from defaults and honors ``AUTOBENCH_DOMAINS`` / ``AUTOBENCH_DOMAIN_WEIGHTS``.
    * ``load_all_cases`` returns ``dict[domain_name, list[BenchmarkCase]]``,
      preserves shape across calls, and skips optional domains on load
      failure rather than raising.
    * ``aggregate_score`` returns a weighted average using the renormalized
      weights actually present in the score map.
    * PopulationRunner backwards-compat: passing ``list[BenchmarkCase]`` to
      ``run()`` still produces a single-domain AdvocateResult with
      ``per_domain_scores == {DEFAULT_DOMAIN: best_score}``.
    * PopulationRunner cross-domain shape: passing
      ``dict[str, list[BenchmarkCase]]`` evaluates each advocate against
      every domain and produces a weighted aggregate.
    * Disabled domains via ``AUTOBENCH_DOMAINS`` drop out of the registry
      and the aggregate renormalizes.
    * Empty cases for a secondary domain contribute 0.0 without raising.
    * The cross_domain_evaluation_complete event fires once per advocate
      when cross-domain mode is active and validates against the schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from autobench.benchmark_registry import (
    DEFAULT_DOMAIN,
    DEFAULT_WEIGHTS,
    BenchmarkDomain,
    BenchmarkRegistry,
    _read_enabled_domains_env,
    _read_weight_overrides_env,
)
from autobench.core import HarnessConfig, HarnessResult, Verdict
from autobench.evaluator import BenchmarkCase, BenchmarkResult
from autobench.observability import (
    CHANNEL_CROSS_DOMAIN_EVALUATION,
    AutobenchObservability,
)
from autobench.population import (
    AdvocateResult,
    PopulationRunner,
    select_promotion_candidate,
)
from autobench.rsi_loop import ImprovementDelta


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _dummy_case_result(verdict: Verdict = Verdict.OK) -> HarnessResult:
    return HarnessResult(p_score=1.0 if verdict == Verdict.OK else 0.0, verdict=verdict)


class _ScriptedEvaluator:
    """Evaluator returning scripted scores. Tracks how many cases it saw."""

    def __init__(self, scores_by_call: list[float]) -> None:
        self._scores = list(scores_by_call)
        self.run_count = 0
        self.harnesses_seen: list[HarnessConfig] = []
        self.case_counts_seen: list[int] = []

    def run(
        self,
        harness: HarnessConfig,
        cases: list[BenchmarkCase],
        obs: Any = None,
        iteration: int = 0,
    ) -> BenchmarkResult:
        self.harnesses_seen.append(harness)
        self.case_counts_seen.append(len(cases))
        idx = min(self.run_count, len(self._scores) - 1)
        self.run_count += 1
        return BenchmarkResult(
            case_results=[_dummy_case_result()],
            aggregate_score=self._scores[idx],
            total_latency_ms=0.0,
            verdict_counts={"OK": 1},
            metadata={},
        )


def _make_case(case_id: str) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        prompt=f"prompt for {case_id}",
        language="python",
        expected_output="",
    )


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force debug-file emission so we can read events back."""
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


def test_read_enabled_domains_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOBENCH_DOMAINS", raising=False)
    assert _read_enabled_domains_env() is None


def test_read_enabled_domains_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_DOMAINS", "codeforces_tier1, multifile_refactor")
    assert _read_enabled_domains_env() == ["codeforces_tier1", "multifile_refactor"]


def test_read_weight_overrides_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AUTOBENCH_DOMAIN_WEIGHTS",
        "codeforces_tier1=0.7, multifile_refactor=0.3",
    )
    out = _read_weight_overrides_env()
    assert out == {"codeforces_tier1": 0.7, "multifile_refactor": 0.3}


def test_read_weight_overrides_ignores_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AUTOBENCH_DOMAIN_WEIGHTS",
        "broken, codeforces_tier1=abc, multifile_refactor=0.5",
    )
    assert _read_weight_overrides_env() == {"multifile_refactor": 0.5}


# --------------------------------------------------------------------------- #
# Registry construction
# --------------------------------------------------------------------------- #


def test_default_registry_lists_three_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOBENCH_DOMAINS", raising=False)
    monkeypatch.delenv("AUTOBENCH_DOMAIN_WEIGHTS", raising=False)
    reg = BenchmarkRegistry.default()
    names = reg.enabled_domains()
    assert names == ["codeforces_tier1", "multifile_refactor", "shader_tier1"]


def test_default_registry_default_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOBENCH_DOMAINS", raising=False)
    monkeypatch.delenv("AUTOBENCH_DOMAIN_WEIGHTS", raising=False)
    reg = BenchmarkRegistry.default()
    assert reg.weights() == DEFAULT_WEIGHTS


def test_autobench_domains_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBENCH_DOMAINS", "codeforces_tier1,multifile_refactor")
    monkeypatch.delenv("AUTOBENCH_DOMAIN_WEIGHTS", raising=False)
    reg = BenchmarkRegistry.default()
    assert reg.enabled_domains() == ["codeforces_tier1", "multifile_refactor"]


def test_autobench_domain_weights_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOBENCH_DOMAINS", raising=False)
    monkeypatch.setenv(
        "AUTOBENCH_DOMAIN_WEIGHTS",
        "codeforces_tier1=0.9,multifile_refactor=0.05",
    )
    reg = BenchmarkRegistry.default()
    w = reg.weights()
    assert w["codeforces_tier1"] == pytest.approx(0.9)
    assert w["multifile_refactor"] == pytest.approx(0.05)
    # shader_tier1 keeps its default (no override)
    assert w["shader_tier1"] == pytest.approx(DEFAULT_WEIGHTS["shader_tier1"])


# --------------------------------------------------------------------------- #
# Built-in loaders (smoke — file shape, not full validation)
# --------------------------------------------------------------------------- #


def test_load_all_cases_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """At least cf-tier-1 + multifile_refactor must load cleanly."""
    monkeypatch.delenv("AUTOBENCH_DOMAINS", raising=False)
    monkeypatch.delenv("AUTOBENCH_DOMAIN_WEIGHTS", raising=False)
    reg = BenchmarkRegistry.default()
    by_domain = reg.load_all_cases()
    assert isinstance(by_domain, dict)
    assert "codeforces_tier1" in by_domain
    assert "multifile_refactor" in by_domain
    # cf-tier-1 ships 20 curated cases.
    assert len(by_domain["codeforces_tier1"]) == 20
    # multifile_refactor ships 5 cases.
    assert len(by_domain["multifile_refactor"]) == 5


def test_optional_domain_skips_on_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optional domain whose loader raises is dropped — required propagates."""
    def _boom() -> list[BenchmarkCase]:
        raise RuntimeError("boom")

    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="ok", case_loader=lambda: [_make_case("a")], weight=0.5),
            BenchmarkDomain(name="optional_bad", case_loader=_boom, weight=0.5, optional=True),
        ]
    )
    out = reg.load_all_cases()
    assert "ok" in out and "optional_bad" not in out
    assert len(out["ok"]) == 1


def test_required_domain_failure_propagates() -> None:
    def _boom() -> list[BenchmarkCase]:
        raise RuntimeError("kaboom")

    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="required_bad", case_loader=_boom, weight=1.0, optional=False),
        ]
    )
    with pytest.raises(RuntimeError, match="kaboom"):
        reg.load_all_cases()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_score_weighted_average() -> None:
    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="a", case_loader=lambda: [], weight=0.6),
            BenchmarkDomain(name="b", case_loader=lambda: [], weight=0.4),
        ]
    )
    score = reg.aggregate_score({"a": 1.0, "b": 0.0})
    assert score == pytest.approx(0.6)


def test_aggregate_score_renormalizes_missing_domain() -> None:
    """Domain present in registry but absent in the score map is dropped
    from the denominator (renormalization)."""
    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="a", case_loader=lambda: [], weight=0.5),
            BenchmarkDomain(name="b", case_loader=lambda: [], weight=0.3),
            BenchmarkDomain(name="c", case_loader=lambda: [], weight=0.2),
        ]
    )
    # 'c' is missing from the input — weights renormalize across a + b only.
    # Expected: (0.5 * 0.8 + 0.3 * 0.2) / (0.5 + 0.3) = (0.4 + 0.06) / 0.8 = 0.575
    score = reg.aggregate_score({"a": 0.8, "b": 0.2})
    assert score == pytest.approx(0.575)


def test_aggregate_empty_returns_zero() -> None:
    reg = BenchmarkRegistry.default()
    assert reg.aggregate_score({}) == 0.0


def test_aggregate_all_zero_weights_falls_back_to_uniform() -> None:
    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="a", case_loader=lambda: [], weight=0.0),
            BenchmarkDomain(name="b", case_loader=lambda: [], weight=0.0),
        ]
    )
    # Defensive: uniform mean when total weight is 0.
    assert reg.aggregate_score({"a": 0.6, "b": 0.2}) == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# PopulationRunner: backwards-compat list shape
# --------------------------------------------------------------------------- #


def test_runner_list_shape_back_compat(debug_file: Path) -> None:
    """Passing ``list[BenchmarkCase]`` runs single-domain just like Phase 5."""
    evaluator = _ScriptedEvaluator(scores_by_call=[0.30, 0.40])
    obs = AutobenchObservability(debug_file=debug_file)

    runner = PopulationRunner(
        n_advocates=1,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: evaluator,  # type: ignore[arg-type]
        observability_factory=lambda: obs,
        improver=None,
        max_iterations_per_advocate=2,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
    )

    result = runner.run([_make_case("cf-0")])

    assert len(result.advocates) == 1
    ar = result.advocates[0]
    # Legacy semantics — aggregate_score == best_score, per_domain has the
    # single default key.
    assert ar.per_domain_scores == {DEFAULT_DOMAIN: ar.best_score}
    assert ar.aggregate_score == ar.best_score

    # No cross-domain event in single-domain mode.
    events = _read_events(debug_file)
    cd_events = [e for e in events if e.get("type") == CHANNEL_CROSS_DOMAIN_EVALUATION]
    assert cd_events == []


# --------------------------------------------------------------------------- #
# PopulationRunner: cross-domain dict shape
# --------------------------------------------------------------------------- #


def test_runner_dict_shape_evaluates_both_domains(debug_file: Path) -> None:
    """Two-domain run evaluates the FINAL harness against each domain."""
    # The scripted evaluator returns the SAME score per call regardless of
    # which cases we hand it — the important thing is that it gets CALLED
    # for each domain (RSI for the primary + 1 post-RSI eval for the
    # secondary).
    evaluator = _ScriptedEvaluator(scores_by_call=[0.6, 0.6, 0.6, 0.6])
    obs = AutobenchObservability(debug_file=debug_file)

    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="d1", case_loader=lambda: [], weight=0.5),
            BenchmarkDomain(name="d2", case_loader=lambda: [], weight=0.5),
        ]
    )

    runner = PopulationRunner(
        n_advocates=1,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: evaluator,  # type: ignore[arg-type]
        observability_factory=lambda: obs,
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
        registry=reg,
    )

    cases_by_domain = {
        "d1": [_make_case("d1-0")],
        "d2": [_make_case("d2-0"), _make_case("d2-1")],
    }
    result = runner.run(cases_by_domain)

    assert len(result.advocates) == 1
    ar = result.advocates[0]
    # Both domains are in per_domain_scores.
    assert set(ar.per_domain_scores.keys()) == {"d1", "d2"}
    # Aggregate is the weighted mean (here, equal weights so it's the
    # simple average).
    assert ar.aggregate_score == pytest.approx(
        (ar.per_domain_scores["d1"] + ar.per_domain_scores["d2"]) / 2
    )

    # Cross-domain event fired exactly once (one advocate).
    events = _read_events(debug_file)
    cd_events = [e for e in events if e.get("type") == CHANNEL_CROSS_DOMAIN_EVALUATION]
    assert len(cd_events) == 1
    payload = cd_events[0]["data"]
    assert payload["advocate_id"] == "advocate-0"
    assert payload["primary_domain"] == "d1"
    assert set(payload["per_domain_scores"].keys()) == {"d1", "d2"}
    assert payload["aggregate_score"] == pytest.approx(ar.aggregate_score)
    assert "cycle_id" in payload


def test_runner_empty_secondary_domain_contributes_zero(debug_file: Path) -> None:
    """A secondary domain with no cases contributes 0.0 without raising."""
    evaluator = _ScriptedEvaluator(scores_by_call=[0.8, 0.8, 0.8])
    obs = AutobenchObservability(debug_file=debug_file)

    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="primary", case_loader=lambda: [], weight=0.7),
            BenchmarkDomain(name="empty", case_loader=lambda: [], weight=0.3),
        ]
    )

    runner = PopulationRunner(
        n_advocates=1,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: evaluator,  # type: ignore[arg-type]
        observability_factory=lambda: obs,
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
        registry=reg,
    )

    cases_by_domain = {"primary": [_make_case("p-0")], "empty": []}
    result = runner.run(cases_by_domain)
    ar = result.advocates[0]
    assert ar.per_domain_scores["empty"] == 0.0
    # Aggregate is weighted mean of primary's score and the 0.0 from "empty".
    # weights (renormalized): {primary: 0.7, empty: 0.3} → already sum to 1.0
    expected = 0.7 * ar.per_domain_scores["primary"] + 0.3 * 0.0
    assert ar.aggregate_score == pytest.approx(expected)


def test_runner_disabled_domain_via_env(
    monkeypatch: pytest.MonkeyPatch, debug_file: Path
) -> None:
    """AUTOBENCH_DOMAINS filters which domains the default registry exposes.

    We don't run the full PopulationRunner here — we verify that the env
    var actually changes the set of domains the default registry yields,
    AND that aggregate_score renormalizes over only the enabled ones.
    """
    monkeypatch.setenv("AUTOBENCH_DOMAINS", "codeforces_tier1")
    monkeypatch.delenv("AUTOBENCH_DOMAIN_WEIGHTS", raising=False)

    reg = BenchmarkRegistry.default()
    assert reg.enabled_domains() == ["codeforces_tier1"]

    # Aggregate over just the enabled key — weight normalizes to 1.0.
    score = reg.aggregate_score({"codeforces_tier1": 0.42})
    assert score == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def test_cross_domain_schema_exists() -> None:
    schema_path = SCHEMA_DIR / "autobench.cross_domain.evaluation.v1.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text())
    assert schema["title"] == "autobench.cross_domain.evaluation v1"
    required = set(schema["properties"]["data"]["required"])
    assert {"advocate_id", "per_domain_scores", "aggregate_score", "weights"} <= required


def test_cross_domain_event_matches_schema(debug_file: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema_path = SCHEMA_DIR / "autobench.cross_domain.evaluation.v1.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    evaluator = _ScriptedEvaluator(scores_by_call=[0.5, 0.5, 0.5])
    obs = AutobenchObservability(debug_file=debug_file)
    reg = BenchmarkRegistry(
        domains=[
            BenchmarkDomain(name="d1", case_loader=lambda: [], weight=0.5),
            BenchmarkDomain(name="d2", case_loader=lambda: [], weight=0.5),
        ]
    )
    runner = PopulationRunner(
        n_advocates=1,
        initial_harness_factory=lambda: HarnessConfig(system_prompt="BASE"),
        evaluator_factory=lambda: evaluator,  # type: ignore[arg-type]
        observability_factory=lambda: obs,
        improver=None,
        max_iterations_per_advocate=1,
        budget_per_advocate_seconds=None,
        improvement_threshold=0.001,
        registry=reg,
    )
    runner.run({"d1": [_make_case("d1-0")], "d2": [_make_case("d2-0")]})

    events = _read_events(debug_file)
    cd_events = [e for e in events if e.get("type") == CHANNEL_CROSS_DOMAIN_EVALUATION]
    assert cd_events
    for e in cd_events:
        validator.validate(e)
