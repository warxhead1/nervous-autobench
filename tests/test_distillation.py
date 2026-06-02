"""Tests for autobench.distillation (nervous-bus-1hlf)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.evaluation.distillation import CycleDistiller


from tests._paths import SCHEMA_DIR
REPORT_SCHEMA_PATH = SCHEMA_DIR / "autobench.cycle.report.v1.json"


def _envelope(channel: str, data: dict) -> dict:
    """Wrap a data dict in a CloudEvents-lite envelope (matches obs emission)."""
    return {
        "specversion": "1.0",
        "id": "01ABCDEFGHJKMNPQRSTVWXYZ00",
        "source": "/autobench",
        "type": channel,
        "datacontenttype": "application/json",
        "time": "2026-05-16T00:00:00Z",
        "data": dict(data),
    }


# --------------------------------------------------------------------------- #
# Synthetic event factories
# --------------------------------------------------------------------------- #

def _case_results(verdicts: list[str]) -> list[dict]:
    events = []
    for i, v in enumerate(verdicts):
        events.append(
            _envelope(
                "autobench.case.result.v1",
                {
                    "case_id": f"case-{i:03d}",
                    "iteration": 0,
                    "language": "python",
                    "verdict": v,
                    "p_score": 1.0 if v == "OK" else 0.0,
                    "latency_ms": 100.0,
                    "generated_code": "print('hi')",
                    "generated_code_length": 12,
                    "attempt": 1,
                },
            )
        )
    return events


def _verified(it: int, label: str, score_delta: float = 0.05) -> dict:
    return _envelope(
        "autobench.improver.prediction.verified.v1",
        {
            "iteration": it,
            "predicted": {
                "predicted_score_delta": 0.05,
                "predicted_verdict_class_changes": {},
                "confidence": 0.8,
                "rationale": "test rationale",
            },
            "actual_score_delta": score_delta,
            "actual_verdict_class_changes": {},
            "score_delta_error": 0.0,
            "verdict_match_ratio": 1.0,
            "outcome_label": label,
            "confidence_calibration": 0.0,
        },
    )


def _diff(it: int, prompt_diff: str = "+added line") -> dict:
    return _envelope(
        "autobench.improver.delta.diff.v1",
        {
            "iteration": it,
            "system_prompt_diff": prompt_diff,
            "tool_surface_diff": "",
            "rollout_protocol_change": None,
            "context_manager_change": None,
            "budget_changes": {},
            "no_change": False,
        },
    )


def _disagreement(case_id: str, ratio: float) -> dict:
    return _envelope(
        "autobench.judge.disagreement.v1",
        {
            "case_id": case_id,
            "iteration": 0,
            "n_votes": 5,
            "consensus_verdict": "OK",
            "dissent_ratio": ratio,
            "dissent_threshold": 0.4,
            "verdict_distribution": {"OK": 3, "WA": 2},
            "minority_verdicts": ["WA"],
        },
    )


def _worker(n: int) -> list[dict]:
    return [
        _envelope(
            "autobench.worker.v1",
            {
                "case_id": f"case-{i}",
                "model": "minimax",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost_usd": 0.01,
                "latency_ms": 500.0,
                "code_preview": "print('hi')",
            },
        )
        for i in range(n)
    ]


def _improver_complete(n: int) -> list[dict]:
    return [
        _envelope(
            "autobench.improver.v1",
            {
                "model": "minimax",
                "status": "complete",
                "completion_tokens": 50,
                "delta_summary": "tweaked prompt",
            },
        )
        for _ in range(n)
    ]


def _judge_verdict(n_votes: int) -> dict:
    return _envelope(
        "autobench.judge.pool.verdict.v1",
        {
            "case_id": "c1",
            "iteration": 0,
            "n_judges": n_votes,
            "n_votes": n_votes,
            "consensus_verdict": "OK",
            "dissent_ratio": 0.0,
            "verdict_distribution": {"OK": n_votes},
            "consensus_p_score": 1.0,
        },
    )


def _population_summary(n_advocates: int, best_iter: int = 2) -> dict:
    advocates = [
        {
            "advocate_id": f"advocate-{i}",
            "session_id": f"sess-{i}",
            "final_score": 0.6 + 0.05 * i,
            "best_iter": best_iter,
            "diversity_score": 0.5,
            "adjusted_score": 0.6 + 0.05 * i,
        }
        for i in range(n_advocates)
    ]
    return _envelope(
        "autobench.population.summary.v1",
        {
            "cycle_id": "cyc-1",
            "advocates": advocates,
            "winner_id": "advocate-0",
            "winner_score": max(a["final_score"] for a in advocates),
            "cycle_started_at": "2026-05-16T00:00:00Z",
            "cycle_ended_at": "2026-05-16T00:30:00Z",
        },
    )


def _promotion(accepted: bool) -> dict:
    return _envelope(
        "autobench.continuous.promotion_decision.v1",
        {
            "cycle_id": "cyc-1",
            "candidate_advocate_id": "advocate-0",
            "candidate_session_id": "sess-0",
            "candidate_score": 0.7,
            "candidate_adjusted_score": 0.7,
            "ahe_outcome": "confirmed",
            "decision": "accepted" if accepted else "staged",
            "decided_by": "cli",
            "reason": "test",
        },
    )


def _cross_domain(advocate_id: str, scores: dict[str, float]) -> dict:
    return _envelope(
        "autobench.cross_domain.evaluation.v1",
        {
            "advocate_id": advocate_id,
            "per_domain_scores": scores,
            "aggregate_score": sum(scores.values()) / max(len(scores), 1),
            "weights": {k: 1.0 / len(scores) for k in scores},
        },
    )


# --------------------------------------------------------------------------- #
# Schema-validation helper
# --------------------------------------------------------------------------- #

def _validate_report(data: dict) -> None:
    """Validate distilled data against the report v1 data block schema."""
    import jsonschema
    schema = json.loads(REPORT_SCHEMA_PATH.read_text())
    data_schema = schema["properties"]["data"]
    jsonschema.Draft202012Validator(data_schema).validate(data)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_distill_full_event_set_produces_valid_report():
    """Every event type present → full distilled report passes schema."""
    events = []
    events.extend(_case_results(["OK"] * 10 + ["WA"] * 5 + ["TLE"] * 3 + ["CE"] * 2))
    events.append(_verified(0, "confirmed", 0.08))
    events.append(_verified(1, "refuted", -0.02))
    events.append(_diff(0, "+improved prompt"))
    events.append(_diff(1, "+regressed"))
    events.append(_disagreement("case-005", 0.6))
    events.append(_disagreement("case-006", 0.5))
    events.append(_disagreement("case-007", 0.3))  # below threshold, must skip
    events.append(_judge_verdict(5))
    events.append(_judge_verdict(5))
    events.append(_cross_domain("advocate-0", {"codeforces_tier1": 0.7, "shader_tier1": 0.4}))
    events.append(_cross_domain("advocate-1", {"codeforces_tier1": 0.6, "shader_tier1": 0.5}))
    events.append(_population_summary(2, best_iter=4))
    events.append(_promotion(True))
    events.extend(_worker(15))
    events.extend(_improver_complete(4))

    d = CycleDistiller()
    report = d.distill_from_events(
        events=events,
        cycle_id="CYC0000000000000000000000A",
        domain="codeforces_tier1",
        requested_by="operator",
        correlation_id="CORR000000000000000000000A",
        started_at="2026-05-16T00:00:00Z",
        completed_at="2026-05-16T00:30:00Z",
    )

    _validate_report(report)
    assert report["summary"]["n_advocates"] == 2
    assert report["summary"]["promoted"] is True
    assert report["summary"]["promoted_advocate_id"] == "advocate-0"
    assert report["summary"]["ahe_outcomes"]["confirmed"] == 1
    assert report["summary"]["ahe_outcomes"]["refuted"] == 1
    assert len(report["patterns"]["top_failure_modes"]) <= 5
    # WA was the most common non-OK verdict (5 of them).
    assert report["patterns"]["top_failure_modes"][0]["failure_mode"] == "WA"
    assert report["patterns"]["top_failure_modes"][0]["count"] == 5
    # Two disagreements exceeded threshold (0.6 and 0.5); 0.3 must skip.
    assert len(report["patterns"]["dissent_hotspots"]) == 2
    assert report["patterns"]["dissent_hotspots"][0]["dissent_ratio"] == 0.6
    # Cross-domain averaged across two advocates.
    assert pytest.approx(report["patterns"]["cross_domain_score"]["codeforces_tier1"], abs=1e-9) == 0.65
    # Cost: 15 worker + 4 improver-complete + 10 judge votes (5+5) = 29
    assert report["cost"]["worker_calls"] == 15
    assert report["cost"]["improver_calls"] == 4
    assert report["cost"]["judge_calls"] == 10
    assert report["cost"]["total_requests"] == 29
    # Successful delta from the confirmed verification.
    assert len(report["patterns"]["successful_deltas"]) == 1
    assert report["patterns"]["successful_deltas"][0]["score_delta"] == pytest.approx(0.08)


def test_distill_with_no_events_returns_empty_but_valid():
    """Empty event list → all-zero / empty distilled report still passes schema."""
    d = CycleDistiller()
    report = d.distill_from_events(
        events=[],
        cycle_id="C" * 26,
        domain="codeforces_tier1",
        requested_by="operator",
        correlation_id="X" * 26,
        started_at="2026-05-16T00:00:00Z",
        completed_at="2026-05-16T00:30:00Z",
        n_advocates_hint=0,
    )
    _validate_report(report)
    assert report["summary"]["n_advocates"] == 0
    assert report["summary"]["promoted"] is False
    assert report["patterns"]["top_failure_modes"] == []
    assert report["patterns"]["successful_deltas"] == []
    assert report["patterns"]["dissent_hotspots"] == []
    assert report["patterns"]["lineage_diversity"] == 0.0
    assert report["patterns"]["cross_domain_score"] == {}
    assert report["cost"]["total_requests"] == 0


def test_distill_missing_judge_events_returns_empty_dissent():
    """No judge.disagreement events → dissent_hotspots is empty."""
    events = _case_results(["OK", "WA"])
    d = CycleDistiller()
    report = d.distill_from_events(
        events=events,
        cycle_id="C" * 26,
        domain="codeforces_tier1",
        requested_by="operator",
        correlation_id="X" * 26,
        started_at="2026-05-16T00:00:00Z",
        completed_at="2026-05-16T00:30:00Z",
    )
    _validate_report(report)
    assert report["patterns"]["dissent_hotspots"] == []
    # case.result-derived failure mode still shows up.
    assert any(m["failure_mode"] == "WA" for m in report["patterns"]["top_failure_modes"])


def test_top_failure_modes_truncated_to_five():
    """More than 5 distinct verdicts → only top 5 returned, by count desc."""
    verdicts = (
        ["WA"] * 10
        + ["TLE"] * 9
        + ["RE"] * 8
        + ["CE"] * 7
        + ["MLE"] * 6
        + ["IDLE"] * 5  # 6th distinct verdict — must be excluded
        + ["OK"] * 100
    )
    events = _case_results(verdicts)
    d = CycleDistiller()
    report = d.distill_from_events(
        events=events,
        cycle_id="C" * 26,
        domain="codeforces_tier1",
        requested_by="operator",
        correlation_id="X" * 26,
        started_at="2026-05-16T00:00:00Z",
        completed_at="2026-05-16T00:30:00Z",
    )
    _validate_report(report)
    modes = report["patterns"]["top_failure_modes"]
    assert len(modes) == 5
    assert [m["failure_mode"] for m in modes] == ["WA", "TLE", "RE", "CE", "MLE"]
    # Fractions sum to less than 1.0 (OK and IDLE excluded; OK from numerator,
    # IDLE truncated — total denominator is the sum of all non-OK counts seen).
    assert all(0.0 <= m["fraction"] <= 1.0 for m in modes)


def test_distill_from_observability_reads_debug_file(tmp_path):
    """distill_from_observability folds the obs's debug-file fallback."""
    from autobench.observability import AutobenchObservability

    debug = tmp_path / "debug.jsonl"
    obs = AutobenchObservability(debug_file=debug)
    # Force pipe-disabled so every emit goes to the debug file.
    obs._pipe_disabled = True

    obs.case_result(
        case_id="c1",
        iteration=0,
        language="python",
        verdict="WA",
        p_score=0.0,
        latency_ms=10.0,
        generated_code="print()",
        generated_code_length=7,
    )
    obs.case_result(
        case_id="c2",
        iteration=0,
        language="python",
        verdict="OK",
        p_score=1.0,
        latency_ms=10.0,
        generated_code="print()",
        generated_code_length=7,
    )

    d = CycleDistiller()
    report = d.distill_from_observability(
        obs=obs,
        cycle_id="C" * 26,
        domain="codeforces_tier1",
        requested_by="operator",
        correlation_id="X" * 26,
        started_at="2026-05-16T00:00:00Z",
        completed_at="2026-05-16T00:30:00Z",
    )
    _validate_report(report)
    # The one WA verdict shows as a failure mode.
    assert any(m["failure_mode"] == "WA" for m in report["patterns"]["top_failure_modes"])
    assert report["summary"]["n_cases"] == 2
