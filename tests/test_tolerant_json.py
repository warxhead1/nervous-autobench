"""Tests for _tolerant_json_loads — handles near-miss JSON emitted by LLMs.

Real-world failure that motivated this: session 01KRRP71DVQS52XMM858H5W7QN
(2026-05-16) — improver emitted a sensible delta with
`"OK":+2,"CE":-2` in predicted_verdict_class_changes. One `+` blocked the
entire RSI loop because json.loads is strict.
"""

from __future__ import annotations

import json

import pytest

from autobench.llm_improver import _tolerant_json_loads


def test_strict_json_passes_through():
    obj = _tolerant_json_loads('{"a": 1, "b": "hi"}')
    assert obj == {"a": 1, "b": "hi"}


def test_positive_sign_prefix_on_number():
    """The bug that motivated this — `+2` is invalid JSON."""
    obj = _tolerant_json_loads('{"OK": +2, "CE": -2, "WA": 0}')
    assert obj == {"OK": 2, "CE": -2, "WA": 0}


def test_positive_sign_inside_nested_object():
    """The exact pattern from session 01KRRP71DVQS52XMM858H5W7QN."""
    raw = (
        '{"system_prompt_changes": "foo", '
        '"prediction": {"predicted_verdict_class_changes": '
        '{"OK": +2, "CE": -2, "WA": 0}, "confidence": 0.75}}'
    )
    obj = _tolerant_json_loads(raw)
    assert obj is not None
    assert obj["prediction"]["predicted_verdict_class_changes"]["OK"] == 2
    assert obj["prediction"]["predicted_verdict_class_changes"]["CE"] == -2
    assert obj["prediction"]["confidence"] == 0.75


def test_trailing_comma_before_close_brace():
    obj = _tolerant_json_loads('{"a": 1, "b": 2,}')
    assert obj == {"a": 1, "b": 2}


def test_trailing_comma_in_array():
    obj = _tolerant_json_loads('{"xs": [1, 2, 3,]}')
    assert obj == {"xs": [1, 2, 3]}


def test_single_quote_escape_python_style():
    """Some LLMs emit `\\'` (Python-style escape) inside JSON strings."""
    raw = r"""{"msg": "it\'s working"}"""
    obj = _tolerant_json_loads(raw)
    assert obj == {"msg": "it's working"}


def test_invalid_json_returns_none():
    """Total garbage should return None, not raise."""
    assert _tolerant_json_loads("not json at all") is None
    assert _tolerant_json_loads("{") is None
    assert _tolerant_json_loads('{"a":}') is None


def test_does_not_corrupt_valid_negative_numbers():
    """A leading `-` is valid JSON; we must not touch it."""
    obj = _tolerant_json_loads('{"delta": -0.15, "neg_int": -42}')
    assert obj == {"delta": -0.15, "neg_int": -42}


def test_missing_opening_quote_on_keys():
    """Cycle 5 (session 01KRS6CWS6JDE946S2RJMT57WX) iter 0 dropped the
    opening quote from every key after the first. Repair must recover."""
    raw = '{"first_key": 1, second_key": 2, third_key": "hello"}'
    obj = _tolerant_json_loads(raw)
    assert obj == {"first_key": 1, "second_key": 2, "third_key": "hello"}


def test_real_session_01KRS6_payload():
    """The exact malformed JSON that defeated cycle 5 iter 0's improver."""
    raw = (
        '{"system_prompt_changes":"Add explicit structure enforcement",'
        'rollout_protocol":"single",context_manager":"keep",'
        'tool_surface_changes":"No changes needed",'
        'budget_changes":{"max_tokens":6144,"max_time_seconds":10,'
        '"max_cost_dollars":0.5},rationale":"15% CE rate"}'
    )
    obj = _tolerant_json_loads(raw)
    assert obj is not None, "repair rule must recover the cycle 5 payload"
    assert obj["rollout_protocol"] == "single"
    assert obj["context_manager"] == "keep"
    assert obj["budget_changes"]["max_tokens"] == 6144
    assert obj["rationale"] == "15% CE rate"


def test_real_session_01KRRP_payload():
    """The exact 1443-char response that broke session 01KRRP71DVQS52XMM858H5W7QN."""
    raw = (
        '{"system_prompt_changes":"Add instruction: \'IMPORTANT: Your output MUST be '
        "a complete, runnable Python program. Do not output code with syntax errors "
        "or incomplete statements. The code must start with 'import' and end with a "
        "complete function definition or main() call. Verify your code is complete "
        "before finishing.' Also add: 'If you output empty content or incomplete "
        'code, the submission will fail.\'", "rollout_protocol":"iterative", '
        '"context_manager":"full", "tool_surface_changes":"Add a \'verify_code()\' '
        "tool that checks: 1) output contains 'import', 2) output ends with "
        "complete code, 3) no truncated lines. Also add 'retry()' tool to regenerate "
        'on verification failure.", "budget_changes":{"max_tokens":8192,'
        '"max_time_seconds":15,"max_cost_dollars":0.8},"rationale":"CE samples show '
        'truncated/incomplete code...","prediction":{"predicted_score_delta":0.08,'
        '"predicted_verdict_class_changes":{"OK":+2,"CE":-2,"WA":0},'
        '"confidence":0.75,"rationale":"..."}}'
    )
    obj = _tolerant_json_loads(raw)
    assert obj is not None, "tolerant parser must handle the real payload"
    assert obj["system_prompt_changes"].startswith("Add instruction")
    assert obj["rollout_protocol"] == "iterative"
    assert obj["budget_changes"]["max_tokens"] == 8192
    assert obj["prediction"]["confidence"] == 0.75
    assert obj["prediction"]["predicted_verdict_class_changes"] == {
        "OK": 2,
        "CE": -2,
        "WA": 0,
    }
