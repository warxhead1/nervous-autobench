"""Tests for the failure-pattern detector + obs channel (nervous-bus-46v).

Covers:
    * Pure-detector behaviour (threshold, normalisation, caps, multi-class).
    * The obs layer emits a schema-valid envelope on
      ``autobench.failure_pattern.v1``.
    * End-to-end: 5 CE'd cases all starting with ``<think>`` produces a
      single FailurePattern with verdict=CE, sample_count=5, prefix
      containing ``<think>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from autobench.core import HarnessResult, Verdict
from autobench.failure_pattern import FailurePattern, detect_failure_patterns
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_FAILURE_PATTERN,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "autobench.failure_pattern.v1.json"


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #

def _make_result(case_id: str, verdict: Verdict, code: str) -> HarnessResult:
    return HarnessResult(
        verdict=verdict,
        metadata={
            "case_id": case_id,
            "language": "python",
            "generated_code": code,
            "generated_code_length": len(code),
        },
    )


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force zellij-pipe failure so all emissions land in the debug file."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Pure detector
# --------------------------------------------------------------------------- #

def test_empty_input_returns_no_patterns() -> None:
    assert detect_failure_patterns([]) == []


def test_threshold_not_met_returns_no_patterns() -> None:
    """Two cases sharing a prefix do not cross the default threshold of 3."""
    cases = [
        _make_result("c1", Verdict.CE, "<think>foo"),
        _make_result("c2", Verdict.CE, "<think>bar"),
    ]
    assert detect_failure_patterns(cases) == []


def test_threshold_met_returns_one_pattern() -> None:
    # First 20 chars of "<think>I should consider the trailing variation" are
    # identical across all 3 cases, so the bucket holds 3 → ≥ threshold=3.
    cases = [
        _make_result(f"c{i}", Verdict.CE, "<think>I should consider — variant " + str(i))
        for i in range(3)
    ]
    out = detect_failure_patterns(cases)
    assert len(out) == 1
    p = out[0]
    assert p.verdict == "CE"
    assert p.sample_count == 3
    assert p.total_in_class == 3
    assert p.sample_case_ids == ["c0", "c1", "c2"]
    assert p.prefix.startswith("<think>")
    assert len(p.prefix) == 20


def test_ok_verdicts_excluded() -> None:
    """A 5-strong OK cluster must not appear in the output."""
    cases = [
        _make_result(f"good{i}", Verdict.OK, "print('hello')")
        for i in range(5)
    ]
    assert detect_failure_patterns(cases) == []


def test_prefix_normalisation_strips_leading_whitespace_and_collapses_newlines() -> None:
    """Cases with different leading whitespace + newlines bucket together."""
    cases = [
        _make_result("c0", Verdict.WA, "  \n  hello\nworld\n"),
        _make_result("c1", Verdict.WA, "\n\nhello\nworld\n"),
        _make_result("c2", Verdict.WA, "hello\nworld\n"),
    ]
    out = detect_failure_patterns(cases, prefix_len=11, threshold=3)
    assert len(out) == 1
    # "hello\nworld" → "hello|world"
    assert out[0].prefix == "hello|world"
    assert out[0].sample_count == 3


def test_multiple_distinct_patterns_coexist() -> None:
    """Two different verdict classes each surface their own pattern."""
    cases = (
        [_make_result(f"ce{i}", Verdict.CE, "<think>boom") for i in range(3)]
        + [_make_result(f"wa{i}", Verdict.WA, "def solve():\n  pass") for i in range(3)]
    )
    out = detect_failure_patterns(cases)
    assert len(out) == 2
    verdicts = {p.verdict for p in out}
    assert verdicts == {"CE", "WA"}
    by_verdict = {p.verdict: p for p in out}
    assert by_verdict["CE"].sample_count == 3
    assert by_verdict["WA"].sample_count == 3


def test_max_prefixes_per_verdict_cap_honoured() -> None:
    """When 4 distinct prefixes each cross threshold, cap=2 keeps top-2."""
    cases = []
    # Four prefixes each appear exactly 3 times — but only the two with the
    # alphabetically-earliest prefix should survive the cap when sample_counts
    # tie.
    for prefix in ("aaa", "bbb", "ccc", "ddd"):
        for i in range(3):
            cases.append(_make_result(f"{prefix}_{i}", Verdict.RE, prefix + "_body"))
    out = detect_failure_patterns(cases, prefix_len=3, threshold=3, max_prefixes_per_verdict=2)
    assert len(out) == 2
    # All have sample_count=3; deterministic tie-break by prefix ASC means
    # "aaa" + "bbb" win.
    prefixes = {p.prefix for p in out}
    assert prefixes == {"aaa", "bbb"}
    for p in out:
        assert p.total_in_class == 12  # all 12 RE cases


def test_sample_case_ids_capped_at_five() -> None:
    cases = [_make_result(f"c{i}", Verdict.CE, "<think>x") for i in range(8)]
    out = detect_failure_patterns(cases)
    assert len(out) == 1
    assert len(out[0].sample_case_ids) == 5
    assert out[0].sample_count == 8


def test_string_verdict_handled() -> None:
    """A simple object exposing verdict as a plain string still buckets."""

    @dataclass
    class _Stub:
        verdict: str
        metadata: dict[str, Any] = field(default_factory=dict)

    cases = [
        _Stub(verdict="TLE", metadata={"case_id": f"c{i}", "generated_code": "while True: pass"})
        for i in range(3)
    ]
    out = detect_failure_patterns(cases)
    assert len(out) == 1
    assert out[0].verdict == "TLE"


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

def test_schema_loads() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    assert SCHEMA_PATH.exists()
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)


def test_emitted_event_validates_against_schema(debug_file: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    obs = AutobenchObservability(debug_file=debug_file)
    pattern = FailurePattern(
        verdict="CE",
        prefix="<think>I should",
        sample_count=5,
        total_in_class=5,
        sample_case_ids=["c0", "c1", "c2", "c3", "c4"],
    )
    obs.failure_pattern(iteration=2, pattern=pattern)

    events = _read_events(debug_file)
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == CHANNEL_FAILURE_PATTERN

    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    errors = sorted(validator.iter_errors(ev), key=lambda e: list(e.path))
    assert not errors, [e.message for e in errors]

    assert ev["data"]["iteration"] == 2
    assert ev["data"]["verdict"] == "CE"
    assert ev["data"]["sample_count"] == 5
    assert ev["data"]["total_in_class"] == 5
    assert ev["data"]["sample_case_ids"] == ["c0", "c1", "c2", "c3", "c4"]
    assert ev["data"]["prefix_len_chars"] == 20
    assert ev["data"]["session_id"] == obs.session_id


def test_emit_explicit_prefix_len_chars(debug_file: Path) -> None:
    """The obs method accepts a non-default prefix_len_chars override."""
    obs = AutobenchObservability(debug_file=debug_file)
    pattern = FailurePattern(
        verdict="WA",
        prefix="hello|world",
        sample_count=3,
        total_in_class=4,
        sample_case_ids=["c0", "c1", "c2"],
    )
    obs.failure_pattern(iteration=0, pattern=pattern, prefix_len_chars=11)
    events = _read_events(debug_file)
    assert len(events) == 1
    assert events[0]["data"]["prefix_len_chars"] == 11


# --------------------------------------------------------------------------- #
# End-to-end: <think> CE cluster
# --------------------------------------------------------------------------- #

def test_end_to_end_think_prefix_ce_cluster(debug_file: Path) -> None:
    """5 CE'd cases all starting with ``<think>`` → 1 pattern emitted, schema-valid."""
    jsonschema = pytest.importorskip("jsonschema")

    cases = [
        _make_result(f"case-{i}", Verdict.CE, "<think>\nThe solution requires sorting...\n```py\nprint(1)\n```")
        for i in range(5)
    ]
    patterns = detect_failure_patterns(cases)
    assert len(patterns) == 1
    p = patterns[0]
    assert p.verdict == "CE"
    assert p.sample_count == 5
    assert p.total_in_class == 5
    assert "<think>" in p.prefix

    # Drive the obs layer with the detected pattern.
    obs = AutobenchObservability(debug_file=debug_file)
    obs.failure_pattern(iteration=7, pattern=p)
    events = _read_events(debug_file)
    assert len(events) == 1
    ev = events[0]
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    errors = sorted(validator.iter_errors(ev), key=lambda e: list(e.path))
    assert not errors, [e.message for e in errors]
    assert ev["data"]["iteration"] == 7
    assert "<think>" in ev["data"]["prefix"]
    assert ev["data"]["sample_count"] == 5


# --------------------------------------------------------------------------- #
# Defensive paths
# --------------------------------------------------------------------------- #

def test_zero_threshold_returns_empty() -> None:
    cases = [_make_result(f"c{i}", Verdict.CE, "x") for i in range(3)]
    assert detect_failure_patterns(cases, threshold=0) == []


def test_empty_generated_code_buckets_to_empty_prefix() -> None:
    """Cases with no code still bucket; useful 'agent produced nothing' signal."""
    cases = [
        HarnessResult(
            verdict=Verdict.CE,
            metadata={"case_id": f"c{i}", "generated_code": ""},
        )
        for i in range(3)
    ]
    out = detect_failure_patterns(cases)
    assert len(out) == 1
    assert out[0].prefix == ""
    assert out[0].sample_count == 3
