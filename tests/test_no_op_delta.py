"""Tests for the no-op delta detector — guards against the 19ur-style
unbounded-append bug surfacing through model-emitted 'no change' sentinels.

Motivating incident: cycle 5 (session 01KRS6CWS6JDE946S2RJMT57WX) iter 1, the
improver emitted `"tool_surface_changes": "keep"` to mean "don't change the
tool surface." The delta applier treated the string literally and appended
the word "keep" to the tool_surface. Same bug shape would let "none", empty
strings, or null bleed into system_prompt across iterations.
"""

from __future__ import annotations

import pytest

from autobench.core import ContextManager, HarnessConfig, RolloutProtocol
from autobench.llm.minimax import MiniMaxLLMWrapper, _is_no_op_value


@pytest.mark.parametrize(
    "value",
    ["keep", "Keep", "KEEP", "no change", "no_change", "nochange",
     "none", "None", "null", "", "   ", None],
)
def test_no_op_values_are_treated_as_no_op(value):
    assert _is_no_op_value(value) is True


@pytest.mark.parametrize(
    "value",
    ["Add explicit structure enforcement",
     "single", "iterative", "full", "compact",
     "  some real change  ", "0"],
)
def test_real_values_are_not_no_op(value):
    assert _is_no_op_value(value) is False


def test_non_string_non_none_returns_false():
    """Numbers, dicts, lists should not be treated as no-op — they're real values."""
    assert _is_no_op_value(42) is False
    assert _is_no_op_value({"k": "v"}) is False
    assert _is_no_op_value(["x"]) is False
    assert _is_no_op_value(0) is False


def _make_wrapper() -> MiniMaxLLMWrapper:
    return MiniMaxLLMWrapper(api_key="fake", model="MiniMax-M2.7")


def _baseline_config() -> HarnessConfig:
    return HarnessConfig(
        system_prompt="ORIG_PROMPT",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="ORIG_TOOL_SURFACE",
        verifiers=[],
        budget={"max_tokens": 4096, "max_time_seconds": 10, "max_cost_dollars": 0.5},
    )


def test_tool_surface_keep_does_not_append():
    """The exact cycle 5 iter 1 bug — `tool_surface_changes: 'keep'`
    must NOT append the word 'keep' to the tool surface."""
    wrapper = _make_wrapper()
    response = (
        '{"system_prompt_changes": "",'
        '"rollout_protocol": "keep",'
        '"context_manager": "keep",'
        '"tool_surface_changes": "keep",'
        '"budget_changes": {},'
        '"rationale": "no change needed"}'
    )
    new_config, delta = wrapper._parse_llm_response(_baseline_config(), response)
    assert new_config.tool_surface == "ORIG_TOOL_SURFACE", \
        "tool_surface must be unchanged when LLM says 'keep'"
    assert delta.tool_surface_delta == "" or delta.tool_surface_delta is None
    assert new_config.system_prompt == "ORIG_PROMPT"


def test_system_prompt_no_change_does_not_append():
    wrapper = _make_wrapper()
    response = (
        '{"system_prompt_changes": "no change",'
        '"rollout_protocol": "keep",'
        '"tool_surface_changes": "",'
        '"budget_changes": {}}'
    )
    new_config, delta = wrapper._parse_llm_response(_baseline_config(), response)
    assert new_config.system_prompt == "ORIG_PROMPT"
    assert not delta.system_prompt_delta


def test_real_delta_still_applies():
    """Sanity: when the LLM does propose a real change, non-tool deltas land.

    Note: ``tool_surface_changes`` is rejected per nervous-bus-ldd1 — the
    harness has no machinery to materialize improver-proposed tools — so
    this test now asserts system_prompt/protocol/budget deltas still apply
    while tool_surface stays at its baseline.
    """
    wrapper = _make_wrapper()
    response = (
        '{"system_prompt_changes": "ADDITION",'
        '"rollout_protocol": "iterative",'
        '"tool_surface_changes": "+verify_code",'
        '"budget_changes": {"max_tokens": 8192}}'
    )
    new_config, delta = wrapper._parse_llm_response(_baseline_config(), response)
    assert "ADDITION" in new_config.system_prompt
    # nervous-bus-ldd1: tool_surface_changes is ignored.
    assert new_config.tool_surface == "ORIG_TOOL_SURFACE"
    assert "+verify_code" not in new_config.tool_surface
    assert new_config.rollout_protocol == RolloutProtocol.ITERATIVE
    assert new_config.budget["max_tokens"] == 8192
    assert delta.system_prompt_delta == "ADDITION"
    assert delta.tool_surface_delta == ""
