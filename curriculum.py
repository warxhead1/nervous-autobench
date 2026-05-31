"""Curriculum agent — daily synthesis of fresh CodeForces-style problems.

The curriculum agent closes the *static-corpus memorization* risk: with only the
20-problem `codeforces_tier1` set, an improver eventually overfits its prompt
template to those specific problems. SIMA-2 and Agent0 independently converged
on the same pattern — **curriculum agent + executor + judge** — as the path past
this ceiling.

Pipeline
========

    debug.jsonl (24h of session events)
            │
            ▼
    CurriculumAgent.analyze_history(...)
            │            returns CurriculumGoals
            ▼
    CurriculumAgent.synthesize_problems(goals, n=10)
            │            asks MiniMax (or any LLM) for N calibrated problems
            ▼
    CurriculumAgent.save_problems(...)
            │            writes autobench/benchmarks/curriculum/YYYY-MM-DD/
            │              ├─ cases.jsonl  (BenchmarkCase JSONL)
            │              └─ manifest.json
            ▼
    CurriculumScheduler  runs the cycle once per day at a fixed hour.

The agent identifies failure patterns from session history — e.g. "5/5 TLEs
happened on N>10⁵" → goal **large-array timeouts** — and asks the generator to
produce problems that drill exactly that weakness.

Bias / drift risks
==================

The most subtle bias: **the curriculum agent generates problems that IT finds
hard, not what the WORKER finds hard.** A naive single-model setup will drift
the benchmark toward the generator's own blind spots. Mitigations baked in:

1. **Different model than the worker.** Pass a ``model`` distinct from whatever
   the harness uses. Eg. M2.7 as curriculum, M2.5 as worker — or vice versa.
   The class accepts an arbitrary model string; the bias mitigation is
   compositional, not enforced in code.

2. **Mix with a fixed-corpus baseline.** Don't run a benchmark on the
   curriculum-only set. The recommended ratio is roughly 70% curriculum / 30%
   fixed CF tier 1 (or similar gold-standard set). The curriculum dir is laid
   out so a higher-level driver can interleave manifests trivially.

3. **Human review queue.** Each generated problem ships with
   ``target_skills`` + ``rationale`` + ``difficulty_rating``. A periodic audit
   should flag rows where (a) the prompt content doesn't match the declared
   target_skills, (b) the expected_output looks unverified, or (c) the
   difficulty_rating is wildly off the prompt's actual complexity. The
   manifest emits enough metadata for a reviewer to triage in minutes.

CLI
===

::

    python3 -m autobench.curriculum daemon            # run forever, daily at 06:00
    python3 -m autobench.curriculum once              # run one cycle now
    python3 -m autobench.curriculum once --n 5        # specify problem count
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from autobench.observability import AutobenchObservability

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cache root for synthesized problem sets (per-date)
# --------------------------------------------------------------------------- #

DEFAULT_CACHE_DIR = Path("autobench/data")


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #

@dataclass
class CurriculumGoals:
    """Output of session-history analysis.

    Attributes:
        failure_categories: skills/conditions where the agent is failing —
            the agent should be drilled with MORE problems like these.
        mastery_categories: skills/conditions where the agent is succeeding —
            raise the difficulty here.
        n_sessions_analyzed: how many BenchmarkResult-shaped records were read.
        evidence: a compact dict mapping each category to the metric that
            triggered it (e.g. ``{"large-array-timeouts": "5/5 TLEs at N>=1e5"}``).
    """

    failure_categories: list[str] = field(default_factory=list)
    mastery_categories: list[str] = field(default_factory=list)
    n_sessions_analyzed: int = 0
    evidence: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "failure_categories": list(self.failure_categories),
            "mastery_categories": list(self.mastery_categories),
            "n_sessions_analyzed": int(self.n_sessions_analyzed),
        }


@dataclass
class GeneratedProblem:
    """One curriculum-generated problem. Maps cleanly onto BenchmarkCase.

    The ``target_skills`` / ``difficulty_rating`` / ``rationale`` fields are
    curriculum-specific metadata; the rest are BenchmarkCase-shaped so the
    problem can be loaded by the existing harness without translation.
    """

    id: str
    prompt: str
    expected_output: str
    language: str = "python"
    test_inputs: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(
        default_factory=lambda: {"max_time_seconds": 10, "max_memory_mb": 256}
    )
    starter_code: str = ""
    target_skills: list[str] = field(default_factory=list)
    difficulty_rating: int = 1000
    rationale: str = ""
    generator_model: str = ""

    def to_case_dict(self) -> dict[str, Any]:
        """Render as a dict that the BenchmarkCase loader will accept verbatim."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "language": self.language,
            "expected_output": self.expected_output,
            "test_inputs": list(self.test_inputs),
            "constraints": dict(self.constraints),
            "starter_code": self.starter_code,
            "metadata": {
                "source": "curriculum",
                "target_skills": list(self.target_skills),
                "difficulty_rating": int(self.difficulty_rating),
                "rationale": self.rationale,
                "generator_model": self.generator_model,
            },
        }


# --------------------------------------------------------------------------- #
# Heuristics for analyze_history
# --------------------------------------------------------------------------- #

_LARGE_N_RX = re.compile(r"(?:N|n)\s*(?:=|<=|>=|>|<)\s*([0-9eE^*+\-,_]+)")


def _looks_like_large_input(prompt: str) -> bool:
    """Crude heuristic: does the prompt mention N >= ~10^5?"""
    for m in _LARGE_N_RX.finditer(prompt or ""):
        raw = m.group(1).replace(",", "").replace("_", "")
        # Common scientific shapes
        try:
            if "e" in raw or "E" in raw:
                if float(raw) >= 1e5:
                    return True
            elif "^" in raw or "**" in raw:
                # e.g. 10^5
                parts = re.split(r"\^|\*\*", raw)
                if len(parts) == 2:
                    if float(parts[0]) ** float(parts[1]) >= 1e5:
                        return True
            else:
                if float(raw) >= 1e5:
                    return True
        except (ValueError, OverflowError):
            continue
    return False


def _tags_of(case: dict[str, Any]) -> list[str]:
    """Pull tags out of a case-result-ish dict — tolerant of multiple shapes."""
    md = case.get("metadata") or {}
    tags = md.get("tags") or md.get("target_skills") or []
    if isinstance(tags, str):
        tags = [tags]
    return [str(t).lower() for t in tags]


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = (
    "You are a CURRICULUM AGENT. You design new CodeForces-style coding problems "
    "that target SPECIFIC weaknesses in a coding-agent harness. Return ONLY a "
    "JSON array — no prose, no markdown fences, no preamble. Each element is an "
    "object with EXACTLY these keys:\n"
    '  "id":               string, short stable id like "curr-001"\n'
    '  "prompt":           string, full problem statement (CF-style)\n'
    '  "expected_output":  string, the canonical answer for sample_input\n'
    '  "sample_input":     string, a single sample input that produces expected_output\n'
    '  "difficulty_rating": integer, CF-style rating 800..3500\n'
    '  "target_skills":    array of short skill tags (e.g. ["dp","arrays"])\n'
    '  "rationale":        string, one sentence — why this problem drills the goal\n'
)


def _build_synthesis_prompt(goals: CurriculumGoals, n: int) -> str:
    bullets_fail = "\n".join(
        f"- {cat}  ({goals.evidence.get(cat, 'no evidence')})"
        for cat in goals.failure_categories
    ) or "- (none identified — generate balanced general problems)"
    bullets_mastery = "\n".join(f"- {cat}" for cat in goals.mastery_categories) or "- (none)"

    return (
        f"Generate exactly {n} new coding problems calibrated to the goals below.\n\n"
        "FAILURE CATEGORIES (drill harder — most problems should target these):\n"
        f"{bullets_fail}\n\n"
        "MASTERY CATEGORIES (raise difficulty here):\n"
        f"{bullets_mastery}\n\n"
        f"Return a JSON array of {n} problem objects following the schema in the "
        "system instruction. Make sure each problem's expected_output is the "
        "deterministic answer for the given sample_input."
    )


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

_REQUIRED_KEYS = {
    "id",
    "prompt",
    "expected_output",
    "sample_input",
    "difficulty_rating",
    "target_skills",
    "rationale",
}


def _strip_codefence(text: str) -> str:
    """Strip reasoning preambles and code fences before JSON parsing.

    Reasoning models (MiniMax-M2.7, DeepSeek-R1 family, etc.) emit
    ``<think>...</think>`` blocks before the actual answer. The closing tag
    may be absent on truncated responses — in that case we treat the whole
    thing as preamble and return empty so the caller logs an honest
    'no JSON emitted' rejection instead of 'parse failed at char 0'.

    Code fences (``` or ```json) are stripped after the reasoning preamble.
    """
    t = text.strip()
    if t.startswith("<think>"):
        end = t.find("</think>")
        if end == -1:
            return ""
        t = t[end + len("</think>"):].strip()
    if t.startswith("```"):
        # remove opening fence
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        # remove trailing fence
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def _parse_problems(
    raw_text: str,
    generator_model: str,
) -> tuple[list[GeneratedProblem], list[dict[str, Any]]]:
    """Parse the LLM response into a list of GeneratedProblem.

    Skips malformed entries (logs a warning) rather than raising — so a single
    bad row doesn't poison the whole batch.

    Returns ``(problems, rejections)``. Each rejection is a dict carrying the
    fields for ``autobench.curriculum.problem.rejected.v1`` (row_index, reason,
    detail, missing_keys, raw_excerpt) — emission is the caller's job so that
    cycle_id/obs threading stays out of this pure parser.
    """
    text = _strip_codefence(raw_text or "")
    rejections: list[dict[str, Any]] = []
    if not text:
        return ([], rejections)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("curriculum: JSON decode failed: %s", e)
        rejections.append({
            "row_index": None,
            "reason": "json_decode_error",
            "detail": str(e),
            "raw_excerpt": text[:1024],
        })
        return ([], rejections)

    if not isinstance(parsed, list):
        logger.warning("curriculum: top-level JSON is not an array (got %s)", type(parsed).__name__)
        rejections.append({
            "row_index": None,
            "reason": "not_an_array",
            "detail": f"top-level JSON is {type(parsed).__name__}, expected array",
            "raw_excerpt": text[:1024],
        })
        return ([], rejections)

    out: list[GeneratedProblem] = []
    for i, obj in enumerate(parsed):
        if not isinstance(obj, dict):
            logger.warning("curriculum: row %d is not an object — skipping", i)
            rejections.append({
                "row_index": i,
                "reason": "not_an_object",
                "detail": f"row is {type(obj).__name__}, expected object",
                "raw_excerpt": json.dumps(obj)[:1024],
            })
            continue
        missing = _REQUIRED_KEYS - set(obj.keys())
        if missing:
            sorted_missing = sorted(missing)
            logger.warning("curriculum: row %d missing keys %s — skipping", i, sorted_missing)
            rejections.append({
                "row_index": i,
                "reason": "missing_required_keys",
                "detail": f"missing {sorted_missing}",
                "missing_keys": sorted_missing,
                "raw_excerpt": json.dumps(obj)[:1024],
            })
            continue
        try:
            gp = GeneratedProblem(
                id=str(obj["id"]),
                prompt=str(obj["prompt"]),
                expected_output=str(obj["expected_output"]),
                test_inputs=[str(obj["sample_input"])],
                target_skills=[str(s) for s in (obj.get("target_skills") or [])],
                difficulty_rating=int(obj.get("difficulty_rating") or 1000),
                rationale=str(obj.get("rationale") or ""),
                generator_model=generator_model,
            )
            out.append(gp)
        except (TypeError, ValueError) as e:
            logger.warning("curriculum: row %d malformed (%s) — skipping", i, e)
            rejections.append({
                "row_index": i,
                "reason": "field_validation_error",
                "detail": f"{type(e).__name__}: {e}",
                "raw_excerpt": json.dumps(obj)[:1024],
            })
            continue

    return (out, rejections)


# --------------------------------------------------------------------------- #
# MiniMax-as-judge (semantic validation)
# --------------------------------------------------------------------------- #
#
# Structural _parse_problems answers "is the row well-formed?" Judge answers
# the next-tier questions: is it solvable, does the claimed expected_output
# match an independent solution, do the claimed target_skills accurately
# describe the problem? One LLM call per problem produces all three signals.
#
# Budget cost: ~1 extra request per problem. With RateBudgetGuard wired in,
# overnight burns stay inside the 14250-req/5h MiniMax coding-plan cap.

_JUDGE_SYSTEM_PROMPT = (
    "You are a JUDGE for coding-curriculum problems. Given a problem statement, "
    "its claimed expected_output, and the claimed required-skills, you must:\n"
    "  1. Solve the problem yourself for the provided sample input — do NOT "
    "trust the claimed expected_output.\n"
    "  2. Compare your computed output (whitespace-insensitive) against the "
    "claimed one.\n"
    "  3. Identify the 2-4 skills the problem TRULY exercises (don't just echo "
    "the claim).\n"
    "  4. Decide whether the generator-claimed skills are an accurate "
    "description.\n\n"
    "Return ONLY a JSON object — no prose, no markdown fences, no preamble — "
    "with EXACTLY these keys:\n"
    '  "solvable":             boolean — can you produce a deterministic answer?\n'
    '  "judge_output":         string  — your computed output for the sample input\n'
    '  "output_matches":       boolean — does your output equal the claimed one '
    "(whitespace/case-insensitive)?\n"
    '  "actual_skills":        array of strings — the 2-4 skills the problem '
    "truly exercises\n"
    '  "claimed_skills_valid": boolean — do the generator-claimed skills match '
    "actual_skills (overlap is sufficient)?\n"
    '  "notes":                string — one sentence, especially when rejecting\n'
)


@dataclass
class JudgeVerdict:
    """Outcome of MiniMax-as-judge LLM-side validation on one problem.

    ``accepted=True`` means the problem clears all three checks (solvable +
    output match + at least one skill overlap). ``reason`` is "ok" for accepts
    and one of the judge-stage rejection enums otherwise: judge_unsolvable,
    output_mismatch, skill_mismatch, judge_error. ``judge_output`` and
    ``actual_skills`` carry the judge's independent solution for downstream
    telemetry — they're useful even on accepts because they reveal whether
    the judge silently disagreed about skill labels.
    """

    accepted: bool
    reason: str
    detail: str = ""
    judge_output: str = ""
    actual_skills: list[str] = field(default_factory=list)
    raw_response: str = ""


def _build_judge_prompt(p: "GeneratedProblem") -> str:
    sample = p.test_inputs[0] if p.test_inputs else ""
    return (
        "PROBLEM STATEMENT:\n"
        f"{p.prompt}\n\n"
        f"SAMPLE INPUT:\n{sample}\n\n"
        f"CLAIMED expected_output:\n{p.expected_output}\n\n"
        f"CLAIMED target_skills: {json.dumps(p.target_skills)}\n\n"
        "Produce the JSON judge verdict described in the system instruction."
    )


def _normalize_output(s: str) -> str:
    """Whitespace/case-insensitive normalization for output comparison.

    The judge LLM and the generator may disagree on trailing newlines, multiple
    spaces, or capitalization of "YES"/"yes". Pre-normalize both sides before
    treating them as a mismatch — only structurally different answers should
    register as output_mismatch.
    """
    return " ".join(str(s).strip().split()).lower()


def _parse_judge_response(raw: str) -> dict[str, Any] | None:
    """Strip code fences / <think> blocks, parse JSON, return dict or None."""
    cleaned = _strip_codefence(raw)
    if not cleaned:
        return None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------- #
# CurriculumAgent
# --------------------------------------------------------------------------- #


# Type alias for an LLM caller — takes (system_prompt, user_prompt) -> raw text.
LLMCaller = Callable[[str, str], str]


class CurriculumAgent:
    """Synthesizes fresh problems calibrated to recent harness performance.

    Args:
        model: model id (e.g. ``"MiniMax-M2.7"``). Stored in metadata; the
            actual HTTP client is constructed lazily so the class is importable
            without an API key.
        api_key: optional override for ``MINIMAX_API_KEY``. ``None`` defers
            to env var.
        output_dir: where ``save_problems`` writes day-stamped subdirs.
        llm_caller: optional callable for testing — bypasses MiniMax entirely
            when supplied. Receives (system_prompt, user_prompt) and returns
            the raw response text (which should be a JSON array).
        obs: optional observability instance for event emission.
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        api_key: str | None = None,
        output_dir: Path | str = Path("autobench/benchmarks/curriculum"),
        llm_caller: LLMCaller | None = None,
        obs: AutobenchObservability | None = None,
        read_timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.output_dir = Path(output_dir)
        self._llm_caller = llm_caller
        self.obs = obs
        self.read_timeout = read_timeout

    # ------------------------------------------------------------------ #
    # analyze_history
    # ------------------------------------------------------------------ #

    def analyze_history(self, session_results: list[dict]) -> CurriculumGoals:
        """Inspect session histories and identify failure / mastery patterns.

        Accepts a flexible shape: each element can be either
          - a ``BenchmarkResult``-shaped dict (has ``case_results``), or
          - a single ``case`` event dict (has ``verdict`` + ``prompt``), or
          - a CloudEvents-lite envelope (has ``type`` + ``data``).

        Heuristics (intentionally cheap and explicit — easy to extend):

        * **large-array-timeouts** — >= 60% of TLEs occur on prompts mentioning
          N >= 10^5. Strongly correlated with quadratic algorithm choices.
        * **dynamic-programming-deep-state** — >= 40% of WA/RE on cases tagged
          ``dp`` (and difficulty ≥ 1500) → emit ``dp-deep-state`` failure.
        * **negative-number-edge** — >= 40% of WAs on cases mentioning
          "negative" or "signed" in prompt.
        * **mastery: implementation-easy** — >= 80% pass rate on cases tagged
          ``implementation`` with rating ≤ 1000.
        """
        case_recs = _flatten_to_cases(session_results)
        n_sessions = _count_sessions(session_results)

        if not case_recs:
            return CurriculumGoals(n_sessions_analyzed=n_sessions)

        verdicts = [str(c.get("verdict", "")).upper() for c in case_recs]
        n_total = len(case_recs)
        n_tle = sum(1 for v in verdicts if v == "TLE")
        n_wa = sum(1 for v in verdicts if v == "WA")
        n_ok = sum(1 for v in verdicts if v == "OK")

        failure: list[str] = []
        mastery: list[str] = []
        evidence: dict[str, str] = {}

        # 1. Large-array timeouts
        if n_tle >= 2:
            tle_large = sum(
                1 for c, v in zip(case_recs, verdicts)
                if v == "TLE" and _looks_like_large_input(c.get("prompt", ""))
            )
            if tle_large / max(n_tle, 1) >= 0.6:
                failure.append("large-array-timeouts")
                evidence["large-array-timeouts"] = (
                    f"{tle_large}/{n_tle} TLEs happened on N>=1e5"
                )

        # 2. DP deep-state
        dp_fails = [
            c for c, v in zip(case_recs, verdicts)
            if v in {"WA", "RE"} and "dp" in _tags_of(c)
        ]
        dp_total = [c for c in case_recs if "dp" in _tags_of(c)]
        if dp_total and len(dp_fails) / len(dp_total) >= 0.4:
            failure.append("dp-deep-state")
            evidence["dp-deep-state"] = (
                f"{len(dp_fails)}/{len(dp_total)} DP cases failed (WA/RE)"
            )

        # 3. Negative-number edge cases
        neg_wa = sum(
            1 for c, v in zip(case_recs, verdicts)
            if v == "WA" and re.search(r"\bnegative\b|\bsigned\b|<\s*0", c.get("prompt", ""), re.I)
        )
        if n_wa >= 2 and neg_wa / max(n_wa, 1) >= 0.4:
            failure.append("negative-number-edge")
            evidence["negative-number-edge"] = (
                f"{neg_wa}/{n_wa} WAs involved negative numbers"
            )

        # 4. Mastery: implementation-easy
        impl_easy = [
            c for c in case_recs
            if "implementation" in _tags_of(c)
            and int((c.get("metadata") or {}).get("rating", 0) or 0) <= 1000
            and int((c.get("metadata") or {}).get("rating", 1) or 1) > 0
        ]
        if len(impl_easy) >= 3:
            ok_easy = sum(1 for c in impl_easy if str(c.get("verdict", "")).upper() == "OK")
            if ok_easy / len(impl_easy) >= 0.8:
                mastery.append("implementation-easy")
                evidence["implementation-easy"] = (
                    f"{ok_easy}/{len(impl_easy)} implementation@<=1000 passed"
                )

        # 5. Mastery: math-easy
        math_cases = [c for c in case_recs if "math" in _tags_of(c)]
        if len(math_cases) >= 3:
            ok_math = sum(1 for c in math_cases if str(c.get("verdict", "")).upper() == "OK")
            if ok_math / len(math_cases) >= 0.8:
                mastery.append("math-easy")
                evidence["math-easy"] = f"{ok_math}/{len(math_cases)} math passed"

        return CurriculumGoals(
            failure_categories=failure,
            mastery_categories=mastery,
            n_sessions_analyzed=n_sessions,
            evidence=evidence,
        )

    # ------------------------------------------------------------------ #
    # synthesize_problems
    # ------------------------------------------------------------------ #

    def synthesize_problems(
        self,
        goals: CurriculumGoals,
        n: int = 10,
        max_attempts: int = 3,
        cycle_id: str = "",
    ) -> list[GeneratedProblem]:
        """Ask the LLM to generate ``n`` calibrated problems.

        Retries up to ``max_attempts`` times if the response is unparseable;
        on persistent failure returns whatever partial set was parsed
        (possibly empty). Never raises.

        ``cycle_id`` is stamped onto any rejection events emitted while
        parsing. Callers that have already minted the cycle id should pass it
        through; an empty string is acceptable — consumers can join via
        session_id + time.
        """
        if n <= 0:
            return []

        user_prompt = _build_synthesis_prompt(goals, n)
        problems: list[GeneratedProblem] = []
        rejections: list[dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):
            try:
                raw = self._call_llm(_SYSTEM_PROMPT, user_prompt)
            except Exception as exc:  # noqa: BLE001 — never raise from this layer
                logger.warning(
                    "curriculum: LLM call attempt %d/%d failed: %s",
                    attempt, max_attempts, exc,
                )
                continue

            parsed, attempt_rejections = _parse_problems(raw, generator_model=self.model)
            rejections.extend(attempt_rejections)
            if parsed:
                problems = parsed
                break
            logger.info(
                "curriculum: attempt %d/%d returned 0 valid problems; retrying",
                attempt, max_attempts,
            )

        # Emit per-problem rejection telemetry so a blown cycle has diagnostic
        # detail beyond the aggregate count in autobench.curriculum.cycle.v1.
        if self.obs is not None and rejections:
            for r in rejections:
                self.obs.curriculum_problem_rejected(
                    reason=r["reason"],
                    detail=r.get("detail", ""),
                    generator_model=self.model,
                    cycle_id=cycle_id,
                    row_index=r.get("row_index"),
                    missing_keys=r.get("missing_keys"),
                    raw_excerpt=r.get("raw_excerpt", ""),
                )

        # Light validation — drop entries with empty prompts/outputs.
        return [p for p in problems if p.prompt.strip() and p.expected_output.strip()]

    # ------------------------------------------------------------------ #
    # judge_problem (LLM-side semantic validation)
    # ------------------------------------------------------------------ #

    def judge_problem(
        self,
        problem: GeneratedProblem,
        budget_guard: Any = None,
        skip_on_budget_exhaustion: bool = True,
    ) -> JudgeVerdict:
        """Run a single judge LLM call and return a verdict.

        ``budget_guard`` is an optional :class:`RateBudgetGuard` (duck-typed —
        we only call ``.check()`` and ``.record_request()``). When the guard
        reports cap-reached and ``skip_on_budget_exhaustion`` is True, the
        problem is accepted without judging (better to ship unvalidated than
        drop good problems mid-burn). Set False to fail-closed on budget
        exhaustion instead.

        Never raises — LLM errors come back as ``reason="judge_error"``.
        """
        if budget_guard is not None:
            ok, reason = budget_guard.check()
            if not ok:
                if skip_on_budget_exhaustion:
                    return JudgeVerdict(
                        accepted=True,
                        reason="judge_skipped_budget",
                        detail=reason or "rate budget exhausted",
                    )
                return JudgeVerdict(
                    accepted=False,
                    reason="judge_error",
                    detail=f"budget exhausted: {reason or 'unknown'}",
                )

        try:
            raw = self._call_llm(_JUDGE_SYSTEM_PROMPT, _build_judge_prompt(problem))
            if budget_guard is not None and hasattr(budget_guard, "record_request"):
                budget_guard.record_request()
        except Exception as exc:  # noqa: BLE001 — never raise from this layer
            return JudgeVerdict(
                accepted=False,
                reason="judge_error",
                detail=f"judge LLM call raised: {exc}"[:1024],
            )

        obj = _parse_judge_response(raw)
        if obj is None:
            return JudgeVerdict(
                accepted=False,
                reason="judge_error",
                detail="judge returned unparseable response",
                raw_response=raw[:1024],
            )

        solvable = bool(obj.get("solvable", False))
        output_matches_claim = bool(obj.get("output_matches", False))
        claimed_skills_valid = bool(obj.get("claimed_skills_valid", False))
        judge_output = str(obj.get("judge_output", ""))
        actual_skills_raw = obj.get("actual_skills", [])
        if not isinstance(actual_skills_raw, list):
            actual_skills_raw = []
        actual_skills = [str(s) for s in actual_skills_raw if str(s).strip()]
        notes = str(obj.get("notes", ""))[:512]

        # 1. solvability
        if not solvable:
            return JudgeVerdict(
                accepted=False, reason="judge_unsolvable",
                detail=f"judge unsolvable: {notes}"[:1024],
                judge_output=judge_output, actual_skills=actual_skills,
                raw_response=raw[:1024],
            )

        # 2. output match. Trust judge's own boolean first; backstop with
        # normalized comparison so a judge over-rejecting on whitespace doesn't
        # silently drop a correct problem.
        if not output_matches_claim:
            norm_judge = _normalize_output(judge_output)
            norm_claim = _normalize_output(problem.expected_output)
            if norm_judge != norm_claim:
                return JudgeVerdict(
                    accepted=False, reason="output_mismatch",
                    detail=(
                        f"judge={norm_judge[:200]!r} != "
                        f"claimed={norm_claim[:200]!r}: {notes}"
                    )[:1024],
                    judge_output=judge_output, actual_skills=actual_skills,
                    raw_response=raw[:1024],
                )

        # 3. skill match. Soft criterion: at least one claimed skill must
        # appear in actual_skills (case-insensitive). Rejects only when claim
        # and reality are completely disjoint — the most defensible bar at
        # this label-fidelity stage.
        if not claimed_skills_valid:
            claimed = {s.lower() for s in problem.target_skills}
            actual = {s.lower() for s in actual_skills}
            if claimed and actual and not (claimed & actual):
                return JudgeVerdict(
                    accepted=False, reason="skill_mismatch",
                    detail=(
                        f"claimed={sorted(claimed)} ∩ actual={sorted(actual)} "
                        f"= ∅: {notes}"
                    )[:1024],
                    judge_output=judge_output, actual_skills=actual_skills,
                    raw_response=raw[:1024],
                )

        return JudgeVerdict(
            accepted=True, reason="ok",
            detail="passed judge",
            judge_output=judge_output, actual_skills=actual_skills,
        )

    # ------------------------------------------------------------------ #
    # save_problems
    # ------------------------------------------------------------------ #

    def save_problems(
        self,
        problems: list[GeneratedProblem],
        date: str | None = None,
        goals: CurriculumGoals | None = None,
        cycle_id: str | None = None,
    ) -> Path:
        """Write problems as ``BenchmarkCase`` JSONL + a manifest.

        Multi-cycle-safe layout (so N cycles on the same date all preserve
        their data + are attributable to their cycle_id):

            <output_dir>/<date>/
              cycles/<cycle_id>/
                cases.jsonl       # this cycle's truth
                manifest.json     # this cycle's manifest
              cases.jsonl         # daily roll-up (all cycles concatenated)
              manifest.json       # daily manifest with cycles: [...]

        Each row in the rolled-up ``cases.jsonl`` carries
        ``metadata.cycle_id`` + ``metadata.cycle_date`` so a reader can
        attribute any problem back to its origin cycle.

        ``cycle_id`` is optional for back-compat. When omitted a synthetic
        ``solo-<epoch>`` id is minted so single-cycle callers still get a
        sharded directory (and never overwrite a sibling cycle).

        Returns the day-stamped directory path (the daily root, not the
        per-cycle shard) — preserved for back-compat with existing
        callers that inspect ``day_dir / "cases.jsonl"``.
        """
        ds = date or _dt.date.today().isoformat()
        cid = cycle_id or f"solo-{int(time.time() * 1000)}"
        day_dir = self.output_dir / ds
        cycle_dir = day_dir / "cycles" / cid
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Stamp each row with cycle attribution metadata.
        rows: list[dict[str, Any]] = []
        for p in problems:
            row = p.to_case_dict()
            md = dict(row.get("metadata") or {})
            md.setdefault("cycle_id", cid)
            md.setdefault("cycle_date", ds)
            row["metadata"] = md
            rows.append(row)

        # 1. Per-cycle shard (the truth)
        with open(cycle_dir / "cases.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        cycle_manifest = {
            "name": f"curriculum-{ds}-{cid}",
            "description": (
                f"Curriculum cycle {cid} for {ds}. "
                f"Synthesized by {self.model} from session-history analysis."
            ),
            "version": 1,
            "source": "curriculum",
            "generator_model": self.model,
            "date": ds,
            "cycle_id": cid,
            "n_problems": len(problems),
            "goals": goals.summary() if goals else {},
            "problems": [
                {
                    "id": p.id,
                    "difficulty_rating": p.difficulty_rating,
                    "target_skills": p.target_skills,
                }
                for p in problems
            ],
        }
        with open(cycle_dir / "manifest.json", "w") as fh:
            json.dump(cycle_manifest, fh, indent=2)

        # 2. Daily roll-up — regenerated from every shard under cycles/.
        # Idempotent: running the same cycle twice updates its shard then
        # rebuilds the roll-up, which still contains every cycle.
        _rebuild_daily_rollup(day_dir, current_cycle=cycle_manifest)

        # Emit per-problem events
        if self.obs is not None:
            for p in problems:
                self.obs.curriculum_problem_generated(
                    case_id=p.id,
                    prompt=p.prompt,
                    target_skills=p.target_skills,
                    difficulty_rating=p.difficulty_rating,
                    generator_model=p.generator_model or self.model,
                    rationale=p.rationale,
                    date=ds,
                    cycle_id=cid,
                )

        return day_dir

    # ------------------------------------------------------------------ #
    # LLM dispatch
    # ------------------------------------------------------------------ #

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Dispatch to the injected ``llm_caller`` if present, else MiniMax HTTP."""
        if self._llm_caller is not None:
            return self._llm_caller(system_prompt, user_prompt)
        return self._call_minimax(system_prompt, user_prompt)

    def _call_minimax(self, system_prompt: str, user_prompt: str) -> str:
        """Real HTTP call to MiniMax /chat/completions. Imports lazily."""
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY not set; pass api_key= or use llm_caller= for testing"
            )
        import httpx  # local import — only required for the real path

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 8192,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = "https://api.minimax.io/v1/chat/completions"
        timeout = httpx.Timeout(connect=10.0, read=self.read_timeout, write=30.0, pool=5.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"MiniMax response shape unexpected: {e}")


# --------------------------------------------------------------------------- #
# Multi-cycle rollup
# --------------------------------------------------------------------------- #


def _rebuild_daily_rollup(day_dir: Path, current_cycle: dict[str, Any] | None = None) -> None:
    """Regenerate ``<day_dir>/cases.jsonl`` + ``manifest.json`` from shards.

    The roll-up is the authoritative back-compat view that legacy readers
    (``tools/curriculum_diversity_drift.py``, ``autobench/continuous.py``)
    consume. Shards under ``<day_dir>/cycles/<cycle_id>/`` are the source of
    truth. We concatenate every shard's ``cases.jsonl`` in cycle-id-sorted
    order (cycle ids embed an epoch, so sort = chronological) and union the
    per-cycle manifests into a daily index.

    The daily manifest preserves the historical top-level shape (``name``,
    ``n_problems``, ``goals``, ``problems``) by mirroring the *most recent*
    cycle there, then adds a ``cycles: [...]`` list with every cycle's
    summary. ``current_cycle`` is the freshly-written cycle manifest — when
    supplied we use it for the top-level mirror; otherwise we derive it
    from the newest shard on disk.
    """
    cycles_root = day_dir / "cycles"
    if not cycles_root.exists():
        return

    shard_dirs = sorted(
        d for d in cycles_root.iterdir()
        if d.is_dir() and (d / "cases.jsonl").exists()
    )
    if not shard_dirs:
        return

    # 1. Roll up cases.jsonl
    rollup_path = day_dir / "cases.jsonl"
    with open(rollup_path, "w") as out:
        for sd in shard_dirs:
            with open(sd / "cases.jsonl") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")

    # 2. Roll up manifest
    cycle_summaries: list[dict[str, Any]] = []
    for sd in shard_dirs:
        mf_path = sd / "manifest.json"
        if not mf_path.exists():
            continue
        try:
            mf = json.loads(mf_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cycle_summaries.append({
            "cycle_id": mf.get("cycle_id", sd.name),
            "n_problems": int(mf.get("n_problems", 0)),
            "generator_model": mf.get("generator_model", ""),
            "goals": mf.get("goals", {}),
            "problems": mf.get("problems", []),
        })

    # Top-level mirror = current cycle if supplied, else newest shard.
    if current_cycle is not None:
        mirror = current_cycle
    else:
        try:
            mirror = json.loads((shard_dirs[-1] / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            mirror = {}

    ds = day_dir.name
    daily_manifest = {
        "name": f"curriculum-{ds}",
        "description": (
            f"Curriculum-generated problems for {ds}. Rolled-up across "
            f"{len(cycle_summaries)} cycle(s); per-cycle shards live under "
            f"cycles/<cycle_id>/."
        ),
        "version": 1,
        "source": "curriculum",
        "date": ds,
        # Back-compat top-level fields (mirror of newest cycle)
        "generator_model": mirror.get("generator_model", ""),
        "n_problems": sum(c["n_problems"] for c in cycle_summaries),
        "goals": mirror.get("goals", {}),
        "problems": [
            p
            for c in cycle_summaries
            for p in c.get("problems", [])
        ],
        "current_cycle_id": mirror.get("cycle_id", ""),
        "cycles": cycle_summaries,
    }
    with open(day_dir / "manifest.json", "w") as fh:
        json.dump(daily_manifest, fh, indent=2)


# --------------------------------------------------------------------------- #
# Session-history flattening helpers
# --------------------------------------------------------------------------- #


def _flatten_to_cases(session_results: list[dict]) -> list[dict[str, Any]]:
    """Normalise diverse input shapes into a flat list of per-case dicts."""
    out: list[dict[str, Any]] = []
    for sr in session_results or []:
        if not isinstance(sr, dict):
            continue
        # CloudEvents envelope
        if "type" in sr and "data" in sr and isinstance(sr["data"], dict):
            sr = sr["data"]
        # BenchmarkResult-shaped
        if "case_results" in sr and isinstance(sr["case_results"], list):
            for c in sr["case_results"]:
                if isinstance(c, dict):
                    out.append(c)
            continue
        # Per-case dict
        if "verdict" in sr or "prompt" in sr:
            out.append(sr)
    return out


def _count_sessions(session_results: list[dict]) -> int:
    """Count distinct sessions in the input (best-effort)."""
    seen = set()
    count = 0
    for sr in session_results or []:
        if not isinstance(sr, dict):
            continue
        sid = None
        if "session_id" in sr:
            sid = sr["session_id"]
        elif isinstance(sr.get("data"), dict):
            sid = sr["data"].get("session_id")
        if sid is not None:
            if sid in seen:
                continue
            seen.add(sid)
            count += 1
        else:
            count += 1
    return count


def _read_debug_jsonl(
    path: Path,
    since: _dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Read events from a JSONL debug file (best-effort)."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    cutoff_iso = since.strftime("%Y-%m-%dT%H:%M:%S") if since else None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cutoff_iso and isinstance(ev.get("time"), str) and ev["time"] < cutoff_iso:
                continue
            out.append(ev)
    return out


# --------------------------------------------------------------------------- #
# CurriculumScheduler
# --------------------------------------------------------------------------- #


class CurriculumScheduler:
    """Runs the curriculum cycle on a daily cadence.

    Args:
        curriculum_agent: a ready-to-use ``CurriculumAgent``.
        daily_at_hour: integer 0..23 — local hour to run each day.
        debug_jsonl: path to the session-event JSONL (default ``~/.cache/nervous-bus/debug.jsonl``).
        n_problems: how many problems per cycle.
        obs: optional observability instance.
    """

    def __init__(
        self,
        curriculum_agent: CurriculumAgent,
        daily_at_hour: int = 6,
        debug_jsonl: Path | str | None = None,
        n_problems: int = 10,
        obs: AutobenchObservability | None = None,
        validate: bool = False,
        budget_guard: Any = None,
    ) -> None:
        if not (0 <= daily_at_hour <= 23):
            raise ValueError(f"daily_at_hour must be in 0..23, got {daily_at_hour}")
        self.agent = curriculum_agent
        self.daily_at_hour = int(daily_at_hour)
        self.debug_jsonl = (
            Path(debug_jsonl)
            if debug_jsonl is not None
            else Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"
        )
        self.n_problems = int(n_problems)
        self.obs = obs or curriculum_agent.obs
        self.validate = bool(validate)
        self.budget_guard = budget_guard

    def run_once(self) -> dict[str, Any]:
        """Read last 24h of events, analyze, synthesize, save. Emit cycle event."""
        since = _dt.datetime.utcnow() - _dt.timedelta(hours=24)
        events = _read_debug_jsonl(self.debug_jsonl, since=since)

        goals = self.agent.analyze_history(events)

        # Mint cycle_id before synthesis so rejection events can carry it.
        ds = _dt.date.today().isoformat()
        cycle_id = f"curr-{ds}-{int(time.time())}"

        problems = self.agent.synthesize_problems(
            goals, n=self.n_problems, cycle_id=cycle_id,
        )

        n_generated = len(problems)
        n_judge_rejected = 0
        if self.validate and problems:
            kept: list[GeneratedProblem] = []
            for row_idx, p in enumerate(problems):
                verdict = self.agent.judge_problem(p, budget_guard=self.budget_guard)
                if verdict.accepted:
                    kept.append(p)
                    continue
                n_judge_rejected += 1
                if self.obs is not None:
                    self.obs.curriculum_problem_rejected(
                        reason=verdict.reason,
                        detail=verdict.detail,
                        generator_model=self.agent.model,
                        cycle_id=cycle_id,
                        row_index=row_idx,
                        raw_excerpt=verdict.raw_response[:1024],
                        stage="judge",
                        judge_detail={
                            "judge_output": verdict.judge_output[:512],
                            "claimed_output": p.expected_output[:512],
                            "actual_skills": verdict.actual_skills,
                            "claimed_skills": p.target_skills,
                            "notes": verdict.detail[:512],
                        },
                    )
            logger.info(
                "curriculum: judge kept %d/%d (rejected=%d) cycle=%s",
                len(kept), n_generated, n_judge_rejected, cycle_id,
            )
            problems = kept

        day_dir = self.agent.save_problems(
            problems, date=ds, goals=goals, cycle_id=cycle_id,
        )
        if self.obs is not None:
            self.obs.curriculum_cycle_complete(
                cycle_id=cycle_id,
                n_problems_generated=self.n_problems,
                n_problems_validated=len(problems),
                n_problems_rejected=max(self.n_problems - len(problems), 0),
                goals_summary=goals.summary(),
                generator_model=self.agent.model,
                date=ds,
            )

        return {
            "cycle_id": cycle_id,
            "date": ds,
            "day_dir": str(day_dir),
            "n_validated": len(problems),
            "n_requested": self.n_problems,
            "n_judge_rejected": n_judge_rejected,
            "validate": self.validate,
            "goals": goals.summary(),
        }

    def _seconds_until_next_run(self, now: _dt.datetime | None = None) -> float:
        now = now or _dt.datetime.now()
        target = now.replace(hour=self.daily_at_hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target = target + _dt.timedelta(days=1)
        return (target - now).total_seconds()

    def run_daemon(self) -> None:  # pragma: no cover — long-running loop
        """Sleep until next scheduled hour, run, repeat forever."""
        logger.info("curriculum scheduler started; daily at %02d:00", self.daily_at_hour)
        while True:
            secs = self._seconds_until_next_run()
            logger.info("curriculum: sleeping %.0fs until next cycle", secs)
            time.sleep(secs)
            try:
                summary = self.run_once()
                logger.info("curriculum: cycle complete — %s", summary)
            except Exception as exc:  # noqa: BLE001 — daemon must not die
                logger.exception("curriculum: cycle failed: %s", exc)
                time.sleep(60)  # brief backoff


# --------------------------------------------------------------------------- #
# daily_synthesis — convenience entry point for the RSI cycle
# --------------------------------------------------------------------------- #


def _difficulty_for_target_skill(target_skill: float) -> int:
    """Map a target population skill score (0..1) → CF-style difficulty rating.

    Lower skill score → easier problems (lower rating). Higher skill score
    means we want the population to find the benchmark roughly that hard,
    so we generate harder problems. The mapping is intentionally coarse —
    auto-calibration based on historical scores is a follow-up (Phase 5+).

    Anchors:
      * skill 0.0 → 800 (CF Newbie floor)
      * skill 0.5 → 1400 (Specialist)
      * skill 1.0 → 2400 (Master+)
    """
    t = max(0.0, min(1.0, float(target_skill)))
    return int(round(800 + t * 1600))


def daily_synthesis(
    n: int = 20,
    target_skill: float = 0.5,
    *,
    date: str | None = None,
    cache_dir: Path | str | None = None,
    obs: AutobenchObservability | None = None,
    llm_caller: LLMCaller | None = None,
    model: str = "MiniMax-M2.7",
    api_key: str | None = None,
    validate: bool = False,
    history_events: list[dict] | None = None,
) -> list[Any]:
    """Synthesize (or cache-load) ``n`` fresh BenchmarkCase problems.

    This is the public entry point the RSI cycle calls into. It:

    1. Computes a per-date cache key (``cache_dir / curriculum_<date>.jsonl``).
       If the file already exists, it's loaded and returned — no LLM hop.
    2. Otherwise builds a ``CurriculumAgent``, calls ``synthesize_problems``,
       writes the cache file, and emits per-problem + cycle events on ``obs``.
    3. Returns ``list[BenchmarkCase]`` — already adapted to the harness shape.
       Returns an empty list if the LLM produces nothing (caller is
       expected to fall back to a fixed corpus).

    Args:
        n: requested problem count.
        target_skill: 0..1, where the operator wants this benchmark to score
            on the current population. Mapped to a CF difficulty rating.
        date: ISO date string for the cache key. Defaults to today UTC.
        cache_dir: cache root. Defaults to ``autobench/data/``.
        obs: observability instance for event emission. Optional.
        llm_caller: testing seam — when set, bypasses MiniMax.
        model: model id (stored in metadata; ignored when ``llm_caller`` is set).
        api_key: optional override for ``MINIMAX_API_KEY``.
        validate: if True, run an LLM-side validation pass on each problem
            (NOT IMPLEMENTED — currently a no-op flag for forward compat).
            Disabled by default for cost discipline.
        history_events: optional pre-loaded session history for
            analyze_history. When omitted, goals default to empty (balanced
            general problems at the target difficulty).

    Returns:
        list of ``BenchmarkCase`` objects ready to feed the evaluator.
    """
    # Local import — avoid circular import between curriculum & evaluator at
    # module load time (evaluator imports nothing from curriculum, but the
    # reverse is rare enough we keep the load lazy).
    from autobench.evaluator import BenchmarkCase

    ds = date or _dt.date.today().isoformat()
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"curriculum_{ds}.jsonl"

    # --- Cache hit --------------------------------------------------------
    if cache_path.exists():
        cases: list[BenchmarkCase] = []
        try:
            with open(cache_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    cases.append(
                        BenchmarkCase(
                            id=d["id"],
                            prompt=d["prompt"],
                            language=d.get("language", "python"),
                            expected_output=d.get("expected_output", ""),
                            expected_outputs=d.get("expected_outputs", []),
                            constraints=d.get("constraints", {}),
                            starter_code=d.get("starter_code", ""),
                            test_inputs=d.get("test_inputs", []),
                            metadata=d.get("metadata", {}),
                        )
                    )
            logger.info(
                "curriculum: cache hit %s (%d cases, skill=%.2f)",
                cache_path, len(cases), target_skill,
            )
            return cases
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "curriculum: cache read failed for %s (%s) — regenerating",
                cache_path, exc,
            )

    # --- Cache miss → synthesize -----------------------------------------
    agent = CurriculumAgent(
        model=model,
        api_key=api_key,
        llm_caller=llm_caller,
        obs=obs,
    )
    # Build goals — if history is supplied, mine it; otherwise emit empty
    # goals tagged with the target_skill so the prompt still nudges
    # difficulty. The synthesis prompt currently bakes only failure/mastery
    # categories; difficulty calibration rides on the rating hint below.
    if history_events:
        goals = agent.analyze_history(history_events)
    else:
        goals = CurriculumGoals(n_sessions_analyzed=0)

    # Inject a difficulty nudge into the goals' evidence dict — the prompt
    # template surfaces evidence verbatim, so the generator sees the target.
    difficulty_rating = _difficulty_for_target_skill(target_skill)
    goals.evidence["target_difficulty"] = (
        f"target CF rating ~{difficulty_rating} "
        f"(operator target_skill={target_skill:.2f})"
    )
    if "target-difficulty-band" not in goals.failure_categories:
        goals.failure_categories.append("target-difficulty-band")

    # Mint cycle_id before synthesis so rejection events can carry it.
    cycle_id = f"curr-{ds}-{int(time.time())}"
    problems = agent.synthesize_problems(goals, n=n, cycle_id=cycle_id)

    # Defensive: tag each problem with a date-stable unique id so multi-day
    # caches never collide on bare "curr-001" ids from the generator.
    id_prefix = ds.replace("-", "")
    for i, p in enumerate(problems):
        if not p.id.startswith(id_prefix):
            p.id = f"curr-{id_prefix}-{i:03d}-{p.id}"

    if validate:
        # LLM-side semantic validation: one judge call per problem checks
        # solvability + output match + skill-label fidelity. Rejected problems
        # are dropped from the cache and emit a judge-stage curriculum.
        # problem.rejected.v1 event so downstream analyzers can cluster
        # generator failure modes by reason.
        kept: list[GeneratedProblem] = []
        for row_idx, p in enumerate(problems):
            verdict = agent.judge_problem(p)
            if verdict.accepted:
                kept.append(p)
                continue
            if obs is not None:
                obs.curriculum_problem_rejected(
                    reason=verdict.reason,
                    detail=verdict.detail,
                    generator_model=model,
                    cycle_id=cycle_id,
                    row_index=row_idx,
                    raw_excerpt=verdict.raw_response[:1024],
                    stage="judge",
                    judge_detail={
                        "judge_output": verdict.judge_output[:512],
                        "claimed_output": p.expected_output[:512],
                        "actual_skills": verdict.actual_skills,
                        "claimed_skills": p.target_skills,
                        "notes": verdict.detail[:512],
                    },
                )
        logger.info(
            "curriculum: judge kept %d/%d problems (rejected=%d)",
            len(kept), len(problems), len(problems) - len(kept),
        )
        problems = kept

    n_requested = n
    n_validated = len(problems)
    n_rejected = max(n_requested - n_validated, 0)

    if not problems:
        # Emit a cycle event even on empty so observers can track failures,
        # but skip cache write (caller falls back to fixed corpus).
        if obs is not None:
            obs.curriculum_cycle_complete(
                cycle_id=cycle_id,
                n_problems_generated=n_requested,
                n_problems_validated=0,
                n_problems_rejected=n_requested,
                goals_summary=goals.summary(),
                generator_model=model,
                date=ds,
            )
        return []

    # --- Write the cache (multi-cycle-safe: shard + roll-up) -------------
    # Per-cycle shard preserves attribution when daily_synthesis is invoked
    # more than once a day (which happens when the cache file is removed or
    # corrupted mid-day). Roll-up file at the legacy path remains the
    # back-compat single-source-of-truth that the RSI cycle reads.
    shard_dir = cache_root / ds / "cycles" / cycle_id
    try:
        shard_dir.mkdir(parents=True, exist_ok=True)
        with open(shard_dir / "cases.jsonl", "w") as fh:
            for p in problems:
                row = p.to_case_dict()
                md = dict(row.get("metadata") or {})
                md.setdefault("cycle_id", cycle_id)
                md.setdefault("cycle_date", ds)
                row["metadata"] = md
                fh.write(json.dumps(row) + "\n")
        # Roll up every shard for this date into the legacy single-file
        # cache. Concatenates in cycle-id-sorted order (epoch-embedded).
        shards_root = cache_root / ds / "cycles"
        all_shards = sorted(
            d for d in shards_root.iterdir()
            if d.is_dir() and (d / "cases.jsonl").exists()
        )
        with open(cache_path, "w") as out:
            for sd in all_shards:
                with open(sd / "cases.jsonl") as fh:
                    for line in fh:
                        if line.strip():
                            out.write(line if line.endswith("\n") else line + "\n")
        logger.info(
            "curriculum: wrote %d cases to %s (shard=%s, skill=%.2f, rating≈%d)",
            len(problems), cache_path, shard_dir, target_skill, difficulty_rating,
        )
    except OSError as exc:
        logger.warning(
            "curriculum: cache write failed (%s) — proceeding without cache",
            exc,
        )

    # --- Emit events ------------------------------------------------------
    if obs is not None:
        for p in problems:
            obs.curriculum_problem_generated(
                case_id=p.id,
                prompt=p.prompt,
                target_skills=p.target_skills,
                difficulty_rating=p.difficulty_rating,
                generator_model=p.generator_model or model,
                rationale=p.rationale,
                date=ds,
                cycle_id=cycle_id,
            )
        obs.curriculum_cycle_complete(
            cycle_id=cycle_id,
            n_problems_generated=n_requested,
            n_problems_validated=n_validated,
            n_problems_rejected=n_rejected,
            goals_summary=goals.summary(),
            generator_model=model,
            date=ds,
        )

    # --- Adapt GeneratedProblem → BenchmarkCase --------------------------
    cases: list[BenchmarkCase] = []
    for p in problems:
        d = p.to_case_dict()
        cases.append(
            BenchmarkCase(
                id=d["id"],
                prompt=d["prompt"],
                language=d.get("language", "python"),
                expected_output=d.get("expected_output", ""),
                expected_outputs=d.get("expected_outputs", []),
                constraints=d.get("constraints", {}),
                starter_code=d.get("starter_code", ""),
                test_inputs=d.get("test_inputs", []),
                metadata=d.get("metadata", {}),
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--output-dir", default="autobench/benchmarks/curriculum")
    parser.add_argument(
        "--validate", action="store_true",
        help="run MiniMax-as-judge semantic validation on each generated problem "
             "(~1 extra request/problem; rejections emit on the bus with "
             "stage=judge)",
    )
    parser.add_argument(
        "--max-requests", type=int, default=15000,
        help="RateBudgetGuard cap when --validate is set (MiniMax coding plan "
             "default 15000/5h with 5%% safety margin)",
    )
    parser.add_argument(
        "--read-timeout", type=float, default=300.0,
        help="HTTP read timeout in seconds for MiniMax API calls (default 300). "
             "Raise if you see 'read operation timed out' on long-generation burns.",
    )


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("autobench.curriculum")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_once = sub.add_parser("once", help="run one curriculum cycle now")
    _add_common_args(p_once)

    p_daemon = sub.add_parser("daemon", help="run forever (daily at --hour)")
    _add_common_args(p_daemon)
    p_daemon.add_argument("--hour", type=int, default=6)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    # Wire observability so curriculum.cycle.v1 AND the per-problem
    # curriculum.problem.rejected.v1 events actually land on the bus.
    # Without obs the parser collects rejections but emits none.
    obs = AutobenchObservability()

    # Optional budget guard for the judge pass. When --validate is off we
    # don't pay the import cost. The guard tracks judge LLM calls; generator
    # calls aren't routed through it here (separate bead).
    budget_guard = None
    if args.validate:
        from autobench.budget_guard import RateBudgetGuard
        budget_guard = RateBudgetGuard(max_requests=args.max_requests)
        logger.info(
            "curriculum: validate=True, RateBudgetGuard cap=%d (effective %d after margin)",
            budget_guard.max_requests, budget_guard._effective_max,
        )

    agent = CurriculumAgent(
        model=args.model, output_dir=args.output_dir, obs=obs,
        read_timeout=args.read_timeout,
    )
    if args.cmd == "once":
        sched = CurriculumScheduler(
            agent, n_problems=args.n, obs=obs,
            validate=args.validate, budget_guard=budget_guard,
        )
        summary = sched.run_once()
        print(json.dumps(summary, indent=2))
        return 0
    if args.cmd == "daemon":
        sched = CurriculumScheduler(
            agent, daily_at_hour=args.hour, n_problems=args.n, obs=obs,
            validate=args.validate, budget_guard=budget_guard,
        )
        sched.run_daemon()
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
