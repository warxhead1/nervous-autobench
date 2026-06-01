"""Shared helpers and base types for the LLM improver wrappers.

Consolidates JSON-parse and prompt-fragment helpers that were previously
duplicated or scattered across ``llm_improver`` / ``minimax_improver``.
Also defines the :class:`LLMImprovementResult` dataclass and a stub
:class:`BaseLLMImprover` ABC that future providers can subclass.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..core import HarnessConfig
from ..rsi.loop import ImprovementDelta

logger = logging.getLogger(__name__)


# Pre-clean rules for near-miss JSON emitted by LLMs. Each is a tiny
# transformation that fixes a specific recurring failure mode WITHOUT
# changing semantics. Final json.loads is strict so any rule that would
# corrupt valid JSON is also rejected downstream.
_JSON_CLEAN_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r":\s*\+(\d)"), r": \1"),
    (re.compile(r"(\[|\{|\s|,)\s*\+(\d)"), r"\1\2"),
    (re.compile(r"\\'"), "'"),
    (re.compile(r",(\s*[\}\]])"), r"\1"),
    (re.compile(r"([,\{])(\s*)([A-Za-z_]\w*)\":"), r'\1\2"\3":'),
)


def tolerant_json_loads(blob: str) -> Any:
    """Parse JSON with a strict pass, then a clean-and-retry fallback.

    Returns ``None`` on total failure (caller treats this as "fall back
    to rule-based improver"). Targeted regex fixes cover the near-misses
    observed in the wild — no permissive json5 dependency.
    """
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        first_err = exc
    cleaned = blob
    for pattern, replacement in _JSON_CLEAN_RULES:
        cleaned = pattern.sub(replacement, cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.debug(
            "tolerant_json_loads: both passes failed. strict=%s | cleaned=%s",
            first_err, exc,
        )
        return None


def build_evidence_section(
    case_results: list,
    max_per_verdict: int = 3,
    code_preview_chars: int = 140,
) -> str:
    """Build a per-verdict-class evidence section for the diagnosis prompt.

    Returns up to ``max_per_verdict`` sample generated_code previews per
    distinct verdict class, with case_id and a single-line code excerpt.
    Empty string if no results.
    """
    if not case_results:
        return "(no case results available — first iteration or empty benchmark)"

    by_verdict: dict[str, list] = {}
    for r in case_results:
        verdict = r.verdict.value if hasattr(r.verdict, "value") else str(r.verdict)
        by_verdict.setdefault(verdict, []).append(r)

    lines: list[str] = []
    for verdict in sorted(by_verdict):
        bucket = by_verdict[verdict]
        lines.append(f"  [{verdict}] {len(bucket)} case(s) — samples:")
        for r in bucket[:max_per_verdict]:
            case_id = (r.metadata or {}).get("case_id", "?")
            code = ((r.metadata or {}).get("generated_code") or "")[:code_preview_chars]
            preview = code.replace("\n", " | ").rstrip()
            if not preview:
                preview = "(empty)"
            lines.append(f"    - {case_id}: {preview!r}")
    return "\n".join(lines)


# Back-compat aliases: a few legacy call sites (and many tests) still
# import the private underscored names. Keep them.
_tolerant_json_loads = tolerant_json_loads
_build_evidence_section = build_evidence_section


@dataclass
class LLMImprovementResult:
    """Structured result from an LLM-driven harness improvement."""

    suggested_harness: HarnessConfig
    delta: ImprovementDelta
    raw_response: str
    model_used: str
    tokens_used: int
    cost_dollars: float
    latency_ms: float


class BaseLLMImprover(ABC):
    """Common interface for LLM improver wrappers.

    Subclasses expose a single :meth:`improve` entry point used by
    :class:`autobench.rsi.SelfImprovingHarness` (after rsi_loop) and
    :class:`autobench.llm.MultiImproverEnsemble`.
    """

    @abstractmethod
    def improve(
        self,
        current_config: HarnessConfig,
        benchmark_results: Any,
        **kwargs: Any,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        """Propose a new harness + delta; must not raise on API failure."""
