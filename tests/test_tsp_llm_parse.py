"""Tests for parse_llm_response — the LLM-output -> C++ extraction.

Based on the real MiniMax response shape (a ```cpp fenced block; deer-flow
status lines go to stderr, so stdout is just the fence). A junk or empty
response must yield "" so the candidate scores 0.0 rather than crashing the loop.
"""

from __future__ import annotations

from autobench.tsp_kernel import parse_llm_response

FENCE = "```"


def test_fenced_cpp_block():
    raw = f'{FENCE}cpp\nextern "C" double priority(int node){{ return 0.0; }}\n{FENCE}'
    out = parse_llm_response(raw)
    assert out.startswith('extern "C" double priority')
    assert FENCE not in out


def test_fenced_plain_block():
    raw = f"{FENCE}\ndouble priority(int n){{return 1;}}\n{FENCE}"
    out = parse_llm_response(raw)
    assert out == "double priority(int n){return 1;}"


def test_priority_function_markers():
    raw = "PRIORITY_FUNCTION\ndouble priority(int n){return 2;}\nPRIORITY_FUNCTION"
    out = parse_llm_response(raw)
    assert "double priority" in out
    assert "PRIORITY_FUNCTION" not in out


def test_bare_definition_with_surrounding_prose():
    raw = 'Sure!\nextern "C" double priority(int node){ return -1.0; }'
    out = parse_llm_response(raw)
    assert out.startswith('extern "C" double priority')
    assert "Sure!" not in out


def test_empty_and_whitespace_return_empty():
    assert parse_llm_response("") == ""
    assert parse_llm_response("   \n\t ") == ""


def test_prose_without_code_has_no_signature():
    # A refusal / non-code reply yields no priority() signature, so the candidate
    # will fail to compile and score 0.0 — we only assert it doesn't raise.
    out = parse_llm_response("I cannot help with that request.")
    assert "double priority" not in out
