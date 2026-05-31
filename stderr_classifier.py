"""Classify sandbox stderr text into a named failure category.

Pure, deterministic, regex-driven — no LLM call. Built around the failure modes
actually observed in autobench cycles (recovered_cycles/2026-05-17_*.json
showed ~70% CE failures, dominated by the "prose-prefix" pattern where the
worker emits explanatory text before the Python code).

The output schema mirrors ``schemas/autobench.failure.category.v1.json``:

    {
        "category":      "<snake_case_id>",
        "subcategory":   "<optional finer-grained tag>",
        "hint":          "<short imperative for the improver>",
        "confidence":    <float 0..1>,
        "matched_pattern": "<diagnostic snippet, optional>",
    }

When nothing matches with high confidence we return ``unknown`` with
``confidence=0.0``. Callers always get a dict; never raises.

Adding a new category: append a tuple to ``_PATTERNS`` (or extend
``_classify_python_compile_error`` for nuanced CE handling). Tests in
``autobench/tests/test_stderr_classifier.py`` should pin the new case.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# Pattern table. Order matters: first match wins. Patterns are evaluated only
# when the parent verdict is in the listed ``verdicts`` set, which keeps the
# CE-specific patterns from misclassifying e.g. an RE traceback that happens
# to contain the word "syntax".
# --------------------------------------------------------------------------- #

_PATTERNS: list[dict[str, Any]] = [
    # CE: prose prefix — the dominant failure mode the improver discovered.
    # MiniMax sometimes emits "We need to solve...", "Let's think...",
    # "Here's the solution:", before any code. The prose line appears IN
    # the SyntaxError context block (the offending source line), so we
    # require BOTH a SyntaxError marker AND a prose phrase — order-free.
    {
        "category": "ce_python_prose_prefix",
        "subcategory": "explanatory_text_before_code",
        "verdicts": ("CE",),
        "regex_all": [
            re.compile(r"SyntaxError|invalid syntax", re.IGNORECASE),
            re.compile(
                r"\b(we need|let's|we'll|here'?s|first,?|the problem|to solve|"
                r"i'?ll|consider|note that|approach is|let me|let us)\b",
                re.IGNORECASE,
            ),
        ],
        "hint": "Worker emitted prose before code. Reinforce 'first character must be valid Python'.",
        "confidence": 0.95,
    },
    # CE: markdown code fence leaked into source.
    {
        "category": "ce_markdown_fence",
        "subcategory": "code_fence_leak",
        "verdicts": ("CE",),
        "regex_all": [
            re.compile(r"SyntaxError|invalid syntax", re.IGNORECASE),
            re.compile(r"```", re.IGNORECASE),
        ],
        "hint": "Worker wrapped code in ```python fences. Reinforce 'no markdown fences'.",
        "confidence": 0.95,
    },
    # CE: indentation.
    {
        "category": "ce_indentation_error",
        "verdicts": ("CE",),
        "regex": re.compile(r"IndentationError|unexpected indent|expected an indented block", re.IGNORECASE),
        "hint": "Indentation off — likely mixed tabs/spaces or missing block body.",
        "confidence": 0.9,
    },
    # CE: unmatched bracket / EOL while scanning.
    {
        "category": "ce_unmatched_bracket",
        "verdicts": ("CE",),
        "regex": re.compile(r"unexpected EOF|unmatched|EOL while scanning|was never closed", re.IGNORECASE),
        "hint": "Unbalanced parens/brackets/quotes — usually truncated output.",
        "confidence": 0.85,
    },
    # CE: generic syntax error (lower priority than the prose-prefix one).
    {
        "category": "ce_syntax_error_other",
        "verdicts": ("CE",),
        "regex": re.compile(r"SyntaxError|invalid syntax", re.IGNORECASE),
        "hint": "Generic syntax error. Inspect first 200 chars of generated code.",
        "confidence": 0.6,
    },
    # RE: NameError — referenced an unbound name (often a forgotten import).
    {
        "category": "re_name_error",
        "verdicts": ("RE",),
        "regex": re.compile(r"NameError: name '([^']+)' is not defined", re.IGNORECASE),
        "hint": "Undefined name '{group1}'. Add an import or define before use.",
        "confidence": 0.95,
    },
    # RE: TypeError.
    {
        "category": "re_type_error",
        "verdicts": ("RE",),
        "regex": re.compile(r"TypeError: ([^\n]+)", re.IGNORECASE),
        "hint": "TypeError: {group1}. Check argument types / signatures.",
        "confidence": 0.9,
    },
    # RE: ValueError (often int-parse failures on EOF / blank input).
    {
        "category": "re_value_error",
        "verdicts": ("RE",),
        "regex": re.compile(r"ValueError: ([^\n]+)", re.IGNORECASE),
        "hint": "ValueError: {group1}. Validate input shape before parsing.",
        "confidence": 0.9,
    },
    # RE: IndexError — off-by-one or empty-list access.
    {
        "category": "re_index_error",
        "verdicts": ("RE",),
        "regex": re.compile(r"IndexError: ([^\n]+)", re.IGNORECASE),
        "hint": "IndexError: {group1}. Likely off-by-one or empty-input edge case.",
        "confidence": 0.9,
    },
    # RE: KeyError.
    {
        "category": "re_key_error",
        "verdicts": ("RE",),
        "regex": re.compile(r"KeyError: ([^\n]+)", re.IGNORECASE),
        "hint": "KeyError: {group1}. Missing dict key — handle absent case.",
        "confidence": 0.9,
    },
    # RE: ZeroDivisionError.
    {
        "category": "re_zero_division",
        "verdicts": ("RE",),
        "regex": re.compile(r"ZeroDivisionError", re.IGNORECASE),
        "hint": "Division by zero. Guard denominator with an explicit check.",
        "confidence": 0.95,
    },
    # RE: recursion limit.
    {
        "category": "re_recursion_limit",
        "verdicts": ("RE",),
        "regex": re.compile(r"RecursionError|maximum recursion depth", re.IGNORECASE),
        "hint": "Recursion too deep. Convert to iteration or raise sys.setrecursionlimit.",
        "confidence": 0.95,
    },
    # RE: stdin EOF (common when problem reads N inputs but worker reads more).
    {
        "category": "re_eof_on_stdin",
        "verdicts": ("RE",),
        "regex": re.compile(r"EOFError|EOF when reading a line", re.IGNORECASE),
        "hint": "Read past end-of-input. Count tokens or use try/except EOFError.",
        "confidence": 0.95,
    },
    # TLE: TimeoutExpired / process killed by sandbox.
    {
        "category": "tle_runtime",
        "verdicts": ("TLE",),
        "regex": re.compile(r"TimeoutExpired|killed by signal|SIGTERM|wall.?time", re.IGNORECASE),
        "hint": "Wall-time exceeded. Asymptotic complexity too high for input size.",
        "confidence": 0.9,
    },
    # MLE: OOMkill or MemoryError.
    {
        "category": "mle_memory",
        "verdicts": ("MLE",),
        "regex": re.compile(r"MemoryError|OOMKill|Out of memory|cannot allocate", re.IGNORECASE),
        "hint": "Out of memory. Switch to streaming or in-place algorithm.",
        "confidence": 0.95,
    },
]


def classify(
    stderr: str,
    verdict: str,
    language: str = "python",
) -> dict[str, Any]:
    """Return a dict describing the failure category for one error-class verdict.

    Args:
        stderr: The raw stderr text (or excerpt) from the sandbox.
        verdict: One of "CE", "RE", "TLE", "MLE". Other verdicts return a
                 ``no_classification`` sentinel since this channel only fires
                 for error-class outcomes.
        language: Reserved for future language-specific pattern dispatch.

    Always returns a dict — never raises.
    """
    v = (verdict or "").upper()
    if v not in {"CE", "RE", "TLE", "MLE"}:
        return {
            "category": "no_classification",
            "confidence": 0.0,
            "hint": "Verdict is not an error class; classifier intentionally skipped.",
        }

    text = stderr or ""
    if not text.strip():
        # Common: sandbox killed the process before stderr was flushed.
        return {
            "category": f"{v.lower()}_no_stderr",
            "confidence": 0.4,
            "hint": "No stderr captured — sandbox killed process or stderr unflushed.",
        }

    for entry in _PATTERNS:
        if v not in entry["verdicts"]:
            continue

        # Two entry shapes: ``regex`` (single match, captures available for
        # hint formatting) or ``regex_all`` (list — all must match for the
        # category to fire, no group capture). The latter is for cases where
        # we need to AND multiple unordered signals together (e.g. SyntaxError
        # marker AND a prose phrase appearing in either order).
        if "regex_all" in entry:
            matches = [r.search(text) for r in entry["regex_all"]]
            if not all(matches):
                continue
            match = matches[0]  # use first for excerpt
        else:
            match = entry["regex"].search(text)
            if not match:
                continue

        # Format the hint with any captured groups (best-effort).
        hint = entry["hint"]
        try:
            groups = match.groupdict()
            for i, g in enumerate(match.groups() or [], start=1):
                groups[f"group{i}"] = g[:80] if isinstance(g, str) else g
            hint = hint.format(**{k: vv for k, vv in groups.items() if vv is not None})
        except (KeyError, IndexError):
            pass  # Hint stays templated; better than crashing.

        # Provide a short matched-pattern excerpt for diagnostic JSON.
        excerpt = text[max(0, match.start() - 20):match.end() + 20]
        excerpt = excerpt[:120].replace("\n", " ⏎ ")

        out: dict[str, Any] = {
            "category": entry["category"],
            "confidence": float(entry["confidence"]),
            "hint": hint[:240],
            "matched_pattern": excerpt,
        }
        if "subcategory" in entry:
            out["subcategory"] = entry["subcategory"]
        return out

    # Fall-through: verdict-typed unknown bucket.
    return {
        "category": f"{v.lower()}_unknown",
        "confidence": 0.0,
        "hint": "No pattern matched — extend autobench/stderr_classifier.py._PATTERNS.",
    }


def category_summary(events: list[dict[str, Any]]) -> dict[str, int]:
    """Return a category → count histogram from a list of classification events.

    Convenience for distillation / report code: feed in the list of
    autobench.failure.category.v1 ``data`` payloads and get a ranked summary.
    """
    counts: dict[str, int] = {}
    for e in events:
        cat = e.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    # Sort by count desc for stable display in reports.
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
