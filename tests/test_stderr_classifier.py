"""Tests for stderr_classifier — pure regex / no LLM."""

from __future__ import annotations

import pytest

from autobench.stderr_classifier import classify, category_summary


# --- CE cases ---------------------------------------------------------------

def test_ce_prose_prefix_wins_over_generic_syntax() -> None:
    """The dominant failure mode — worker emits prose before code."""
    stderr = (
        '  File "/tmp/autobench_sandbox_X/main.py", line 1\n'
        '    We need to solve a problem: "Train and Peter (CF 8A)". '
        "Let's recall the known approach\n"
        '          ^\n'
        'SyntaxError: invalid syntax'
    )
    out = classify(stderr, "CE")
    assert out["category"] == "ce_python_prose_prefix"
    assert out["confidence"] >= 0.9
    assert "prose" in out["hint"].lower()


def test_ce_markdown_fence() -> None:
    stderr = (
        '  File "/tmp/main.py", line 1\n'
        '    ```python\n'
        '    ^\n'
        'SyntaxError: invalid syntax'
    )
    out = classify(stderr, "CE")
    assert out["category"] == "ce_markdown_fence"
    assert "fence" in out["hint"].lower() or "markdown" in out["hint"].lower()


def test_ce_indentation() -> None:
    stderr = 'IndentationError: unexpected indent'
    out = classify(stderr, "CE")
    assert out["category"] == "ce_indentation_error"


def test_ce_unmatched_bracket() -> None:
    stderr = "SyntaxError: '(' was never closed"
    out = classify(stderr, "CE")
    # First-match-wins ordering: prose pattern doesn't match here, fence
    # doesn't match, indentation doesn't match, then unmatched_bracket fires.
    assert out["category"] == "ce_unmatched_bracket"


def test_ce_generic_syntax_fallback() -> None:
    stderr = "SyntaxError: invalid token at line 5"
    out = classify(stderr, "CE")
    assert out["category"] == "ce_syntax_error_other"
    assert out["confidence"] < 0.9  # lower than the specific patterns


# --- RE cases ---------------------------------------------------------------

def test_re_name_error_extracts_identifier() -> None:
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/tmp/main.py", line 5, in <module>\n'
        "    print(undefned_var)\n"
        "NameError: name 'undefned_var' is not defined"
    )
    out = classify(stderr, "RE")
    assert out["category"] == "re_name_error"
    assert "undefned_var" in out["hint"]


def test_re_type_error() -> None:
    stderr = "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
    out = classify(stderr, "RE")
    assert out["category"] == "re_type_error"
    assert "TypeError" in out["hint"] or "type" in out["hint"].lower()


def test_re_value_error() -> None:
    stderr = "ValueError: invalid literal for int() with base 10: ''"
    out = classify(stderr, "RE")
    assert out["category"] == "re_value_error"


def test_re_index_error() -> None:
    stderr = "IndexError: list index out of range"
    out = classify(stderr, "RE")
    assert out["category"] == "re_index_error"


def test_re_recursion_limit() -> None:
    stderr = "RecursionError: maximum recursion depth exceeded"
    out = classify(stderr, "RE")
    assert out["category"] == "re_recursion_limit"


def test_re_eof_on_stdin() -> None:
    stderr = "EOFError: EOF when reading a line"
    out = classify(stderr, "RE")
    assert out["category"] == "re_eof_on_stdin"


def test_re_zero_division() -> None:
    stderr = "ZeroDivisionError: integer division or modulo by zero"
    out = classify(stderr, "RE")
    assert out["category"] == "re_zero_division"


# --- TLE / MLE --------------------------------------------------------------

def test_tle_runtime() -> None:
    out = classify("subprocess.TimeoutExpired: killed by signal SIGTERM", "TLE")
    assert out["category"] == "tle_runtime"


def test_mle_memory() -> None:
    out = classify("MemoryError: cannot allocate 2.0 GiB for an array", "MLE")
    assert out["category"] == "mle_memory"


# --- Edge cases -------------------------------------------------------------

def test_no_stderr_returns_no_stderr_bucket() -> None:
    out = classify("", "CE")
    assert out["category"] == "ce_no_stderr"
    assert out["confidence"] < 0.5


def test_non_error_verdict_skips_classification() -> None:
    out = classify("anything", "OK")
    assert out["category"] == "no_classification"
    assert out["confidence"] == 0.0


def test_wa_is_also_skipped() -> None:
    """WA is intentionally not classified — same gate as sandbox.stderr.v1."""
    out = classify("expected 42 got 41", "WA")
    assert out["category"] == "no_classification"


def test_unknown_re_pattern_returns_unknown_bucket() -> None:
    stderr = "some weird error we have never seen before"
    out = classify(stderr, "RE")
    assert out["category"] == "re_unknown"
    assert out["confidence"] == 0.0


def test_classifier_never_raises() -> None:
    """No matter the inputs, never raise."""
    for stderr in [None, "", "\x00\xff", "X" * 100000]:
        for verdict in [None, "", "OK", "??", "CE", "RE", "WA", "TLE", "MLE"]:
            out = classify(stderr or "", verdict or "")
            assert isinstance(out, dict)
            assert "category" in out


# --- category_summary -------------------------------------------------------

def test_category_summary_counts_and_sorts() -> None:
    events = [
        {"category": "ce_python_prose_prefix"},
        {"category": "ce_python_prose_prefix"},
        {"category": "ce_python_prose_prefix"},
        {"category": "re_value_error"},
        {"category": "re_value_error"},
        {"category": "tle_runtime"},
    ]
    summary = category_summary(events)
    keys = list(summary.keys())
    assert keys[0] == "ce_python_prose_prefix"  # most common first
    assert summary["ce_python_prose_prefix"] == 3
    assert summary["re_value_error"] == 2
    assert summary["tle_runtime"] == 1


def test_category_summary_handles_missing_keys() -> None:
    events = [{}, {"foo": "bar"}, {"category": "ce_python_prose_prefix"}]
    summary = category_summary(events)
    assert summary["unknown"] == 2  # two events with no 'category' key
    assert summary["ce_python_prose_prefix"] == 1
