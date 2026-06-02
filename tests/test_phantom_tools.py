"""Tests for nervous-bus-ldd1: improver-proposed phantom tools must be ignored.

Motivating incident: cycle session 01KRSDHKD7M0JQ44AKFY8PR7FN iter 1's
improver appended this to tool_surface::

    "Add a pre-execution validation tool: validate_code_start(s) → returns
    {valid: bool, reason: string}. ... Also add a syntax_check(s) tool ..."

Neither tool exists. The harness has no machinery to materialize them —
``tool_surface`` is just a string the worker LLM reads. The improver was
writing fiction. We reject any non-no-op tool_surface_changes at parse
time and log a warning.

The "keep" / "" / null no-op tokens (nervous-bus-9h5y) must still pass
cleanly without contaminating the tool_surface with literal sentinel text.
"""

from __future__ import annotations

import json
import logging

from autobench.core import HarnessConfig
from autobench.llm.minimax import MiniMaxLLMWrapper, _is_no_op_value


def _wrapper() -> MiniMaxLLMWrapper:
    """Construct a wrapper without hitting the network."""
    return MiniMaxLLMWrapper(api_key="test-key-not-used")


def _base_harness() -> HarnessConfig:
    return HarnessConfig(tool_surface="original tool surface string")


def test_phantom_tool_proposal_is_rejected(caplog) -> None:
    """A non-empty tool_surface_changes proposing new tools must be dropped."""
    response = json.dumps({
        "system_prompt_changes": "",
        "rollout_protocol": "keep",
        "context_manager": "keep",
        "tool_surface_changes": (
            "Add a pre-execution validation tool: validate_code_start(s) → "
            "returns {valid: bool, reason: string}. Also add a syntax_check(s) "
            "tool that does a basic Python syntax check on the code."
        ),
        "budget_changes": {},
        "rationale": "fictional tool to reduce CE",
    })

    base = _base_harness()
    with caplog.at_level(logging.WARNING, logger="autobench.llm.minimax"):
        new_harness, delta = _wrapper()._parse_llm_response(base, response)

    # tool_surface must be UNCHANGED — phantom tool proposal was ignored.
    assert new_harness.tool_surface == base.tool_surface, (
        "tool_surface should be unchanged after phantom-tool proposal; "
        f"got {new_harness.tool_surface!r}"
    )
    # delta.tool_surface_delta must be empty (no signal of an applied change).
    assert delta.tool_surface_delta == "", (
        f"delta.tool_surface_delta should be empty, got {delta.tool_surface_delta!r}"
    )
    # A warning should have been emitted with the [ldd1] tag.
    assert any("[ldd1]" in rec.message for rec in caplog.records), (
        "expected an [ldd1] warning log line; "
        f"got {[rec.message for rec in caplog.records]!r}"
    )


def test_keep_passes_cleanly() -> None:
    """tool_surface_changes='keep' is a no-op token and must not append text."""
    response = json.dumps({
        "system_prompt_changes": "",
        "rollout_protocol": "keep",
        "context_manager": "keep",
        "tool_surface_changes": "keep",
        "budget_changes": {},
        "rationale": "nothing to change",
    })
    base = _base_harness()
    new_harness, delta = _wrapper()._parse_llm_response(base, response)
    assert new_harness.tool_surface == base.tool_surface
    assert delta.tool_surface_delta == ""


def test_empty_string_passes_cleanly() -> None:
    """tool_surface_changes='' must not append a newline-only delta."""
    response = json.dumps({
        "system_prompt_changes": "",
        "rollout_protocol": "keep",
        "context_manager": "keep",
        "tool_surface_changes": "",
        "budget_changes": {},
        "rationale": "no change",
    })
    base = _base_harness()
    new_harness, delta = _wrapper()._parse_llm_response(base, response)
    assert new_harness.tool_surface == base.tool_surface
    assert delta.tool_surface_delta == ""


def test_null_value_passes_cleanly() -> None:
    """tool_surface_changes: null must not crash; treated as no-op."""
    response = json.dumps({
        "system_prompt_changes": "",
        "rollout_protocol": "keep",
        "context_manager": "keep",
        "tool_surface_changes": None,
        "budget_changes": {},
        "rationale": "no change",
    })
    base = _base_harness()
    new_harness, delta = _wrapper()._parse_llm_response(base, response)
    assert new_harness.tool_surface == base.tool_surface
    assert delta.tool_surface_delta == ""


def test_rule_based_parser_also_rejects_phantom_tools(caplog) -> None:
    """The legacy in-module ``_parse_llm_improvement`` must reject phantom tools too."""
    from autobench.rsi.loop import _parse_llm_improvement
    base = _base_harness()
    text = json.dumps({
        "system_prompt_changes": "",
        "rollout_protocol": "keep",
        "context_manager": "keep",
        "tool_surface_changes": "Add validate_code_start(s) tool that ...",
        "rationale": "phantom",
    })
    with caplog.at_level(logging.WARNING, logger="autobench.rsi.loop"):
        new_harness, delta = _parse_llm_improvement(base, text)
    assert new_harness.tool_surface == base.tool_surface
    assert delta.tool_surface_delta == ""
    assert any("[ldd1]" in rec.message for rec in caplog.records)


def test_constraint_text_present_in_diagnosis_prompts() -> None:
    """Diagnosis prompts must explicitly state the tool_surface_changes constraint."""
    from autobench.evaluator import BenchmarkResult
    from autobench.llm.minimax import MiniMaxLLMWrapper

    bench = BenchmarkResult(
        case_results=[],
        aggregate_score=0.5,
        total_latency_ms=0.0,
        verdict_counts={"OK": 1},
        metadata={},
    )
    prompt = _wrapper()._build_diagnosis_prompt(_base_harness(), bench)
    assert "ldd1" in prompt or "does not support" in prompt
    assert "tool_surface_changes" in prompt
    # Reaffirm "no inventing APIs" intent appears.
    assert "ignored" in prompt or "will be ignored" in prompt
