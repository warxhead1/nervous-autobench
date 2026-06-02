"""Tests for the evidence-driven diagnosis prompt (nervous-bus-6ed).

These tests pin two invariants:
  1. ``_build_evidence_section`` extracts per-verdict-class generated_code
     samples from a BenchmarkResult-like input.
  2. The diagnosis prompt built by ``AnthropicLLMWrapper._build_diagnosis_prompt``
     (and the MiniMax mirror) contains the structural evidence and does NOT
     contain the prior canned "Guidance: reduce max_tokens..." sentences.

The motivating session was 01KRQTNMM8RFKC477DRRRVMVS4 (2026-05-16 10-min
cycle), where the canned guidance caused the improver to misdiagnose a
77% CE rate as "code too complex" when the actual cause was <think> prose
leaking before the code.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from autobench.core import ContextManager, HarnessConfig, RolloutProtocol, Verdict
from autobench.llm.anthropic import _build_evidence_section


def _mock_result(case_id: str, verdict: Verdict, code: str):
    r = MagicMock()
    r.verdict = verdict
    r.metadata = {"case_id": case_id, "generated_code": code}
    return r


def test_evidence_section_groups_by_verdict():
    results = [
        _mock_result("cf-1", Verdict.CE, "<think>\nReasoning here\n</think>\nimport sys"),
        _mock_result("cf-2", Verdict.CE, "<think>more reasoning</think>"),
        _mock_result("cf-3", Verdict.OK, "import sys\ndef main(): pass"),
    ]
    out = _build_evidence_section(results)

    # Each verdict shows up with its own bucket label
    assert "[CE] 2 case(s)" in out
    assert "[OK] 1 case(s)" in out

    # The actual generated_code surfaces, including the structural pattern
    assert "cf-1" in out
    assert "cf-2" in out
    assert "cf-3" in out
    assert "<think>" in out  # the diagnostic signal that the canned prompt was blind to
    assert "import sys" in out


def test_evidence_section_truncates_long_code():
    long_code = "x" * 500
    results = [_mock_result("cf-long", Verdict.WA, long_code)]
    out = _build_evidence_section(results, code_preview_chars=140)
    # Truncated to <= preview chars + repr overhead; full 500 chars must not appear
    assert "x" * 500 not in out


def test_evidence_section_caps_samples_per_verdict():
    """Many CE cases — only max_per_verdict samples should appear."""
    many = [
        _mock_result(f"cf-{i}", Verdict.CE, f"<think>case {i}</think>")
        for i in range(10)
    ]
    out = _build_evidence_section(many, max_per_verdict=3)
    assert "[CE] 10 case(s)" in out  # full count is reported
    # Only first 3 case_ids actually appear in samples
    sample_ids = [f"cf-{i}" for i in range(10) if f"cf-{i}" in out]
    assert len(sample_ids) == 3


def test_evidence_section_handles_empty_results():
    assert "no case results" in _build_evidence_section([]).lower()


def test_evidence_section_handles_empty_code():
    results = [_mock_result("cf-x", Verdict.RE, "")]
    out = _build_evidence_section(results)
    assert "cf-x" in out
    assert "(empty)" in out


def test_diagnosis_prompt_contains_evidence_not_canned_guidance(monkeypatch):
    """The end-to-end prompt the improver receives must contain raw evidence
    and must NOT contain the deprecated canned guidance sentences."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr("anthropic.Anthropic", MagicMock())  # don't need real client
    from autobench.llm.anthropic import AnthropicLLMWrapper

    wrapper = AnthropicLLMWrapper()

    bench = MagicMock()
    bench.verdict_counts = {"CE": 3, "OK": 1}
    bench.case_results = [
        _mock_result("cf-4A", Verdict.CE, "<think>We need to produce</think>"),
        _mock_result("cf-10A", Verdict.CE, "<think>We need to parse</think>"),
        _mock_result("cf-1A", Verdict.CE, "<think>classical theatre square</think>"),
        _mock_result("cf-9A", Verdict.OK, "import sys, math\ndef main():"),
    ]
    bench.aggregate_score = 0.25
    bench.pass_rate = lambda: 0.25
    bench.total_latency_ms = 9000

    harness = HarnessConfig(
        system_prompt="You write Python code.",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="",
        budget={"max_tokens": 4096, "max_time_seconds": 10, "max_cost_dollars": 0.5},
        verifiers=["stdout_diff"],
    )

    prompt = wrapper._build_diagnosis_prompt(harness, bench)

    # Evidence section is present with the structural signal
    assert "EVIDENCE" in prompt
    assert "<think>" in prompt  # the prose-leak pattern that the canned prompt was blind to
    assert "cf-4A" in prompt and "cf-9A" in prompt

    # Canned guidance phrases from the removed VERDICT_STRATEGIES MUST NOT appear
    forbidden_phrases = [
        "DOMINANT VERDICT ANALYSIS",
        "Compilation errors dominate",
        "reduce max_tokens to cut over-generation",
        "Guidance: reduce",
        "Prefer shorter solutions over comprehensive ones",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in prompt, f"deprecated canned guidance leaked into prompt: {phrase!r}"


def test_minimax_diagnosis_prompt_uses_same_evidence_pattern(monkeypatch):
    """MiniMax improver's prompt must also be evidence-driven (mirror of llm_improver)."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    from autobench.llm.minimax import MiniMaxLLMWrapper

    wrapper = MiniMaxLLMWrapper()

    bench = MagicMock()
    bench.verdict_counts = {"CE": 2, "OK": 1}
    bench.case_results = [
        _mock_result("cf-A", Verdict.CE, "<think>reasoning</think>"),
        _mock_result("cf-B", Verdict.CE, "<think>more reasoning</think>"),
        _mock_result("cf-C", Verdict.OK, "import sys"),
    ]
    bench.aggregate_score = 0.33
    bench.pass_rate = lambda: 0.33
    bench.total_latency_ms = 1000

    harness = HarnessConfig(
        system_prompt="...",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="",
        budget={"max_tokens": 4096},
        verifiers=["stdout_diff"],
    )

    prompt = wrapper._build_diagnosis_prompt(harness, bench)

    assert "EVIDENCE" in prompt
    assert "<think>" in prompt
    assert "cf-A" in prompt
    # Same forbidden canned phrases
    assert "DOMINANT VERDICT ANALYSIS" not in prompt
    assert "Compilation errors dominate" not in prompt
