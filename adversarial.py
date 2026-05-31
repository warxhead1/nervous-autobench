"""Adversarial dual co-evolution — Code-A1 / SAGE pattern.

One LLM (the *generator*) is prompted to invent hard problems with tricky
edge cases that target a specified failure mode. Another LLM (the *worker*)
tries to solve them. The :class:`BenchmarkEvaluator` adjudicates each
solution. The result is a curriculum-closing loop where the system
generates its own evaluation data, drift-resistant against memorisation
of fixed benchmarks.

Research context (see ``autobench/research/frontier_harness_2026.md``):
SIMA-2 and Agent0 independently converged on a generator+executor+judge
triple as the path to real creative emergence. arXiv:2603.15611 (Code-A1)
and arXiv:2603.15255 (SAGE) describe the pattern in detail.

Relationship to :class:`autobench.test_scaffolder.CurveballGenerator`
====================================================================
``test_scaffolder.CurveballGenerator`` is a *rule-based*, deterministic
synthesizer — boundary / adversarial-input / race-condition / resource-
exhaustion templates fired against a list of baseline inputs. It runs
offline and produces predictable static curveballs.

:class:`AdversarialGenerator` (this module) is the *LLM-driven* version —
MiniMax (or any chat-completions-compatible model) writes a brand-new
competitive-programming problem with a hidden gotcha each call. Higher
temperature is recommended (default 0.7 vs the improver's 0.3) because
the goal here is creativity, not faithfulness.

The two coexist: :meth:`AdversarialGenerator.generate_curveball` accepts
an optional ``seed_from: CurveballCase | None`` so an LLM can riff on a
static template when desired.

Public surface
==============
- :class:`AdversarialCase` — one generated curveball.
- :class:`AdversarialRoundResult` — one full generate→solve→judge round.
- :class:`AdversarialGenerator` — wraps the MiniMax chat-completions API.
- :class:`AdversarialDual` — orchestrates generator + worker + evaluator.

All emission methods are non-blocking and never raise; they degrade to
the debug-file fallback via :class:`AutobenchObservability`.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .core import HarnessConfig
from .evaluator import BenchmarkCase, BenchmarkEvaluator
from .observability import (
    CHANNEL_ADVERSARIAL_GENERATED,
    CHANNEL_ADVERSARIAL_ROUND,
    AutobenchObservability,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure-mode catalog — these steer the generator prompt.
# ---------------------------------------------------------------------------

#: Canonical target failure modes the generator can be steered toward.
#: The string is interpolated directly into the prompt; the prose around
#: each describes the kind of trap to set.
TARGET_FAILURE_MODES: tuple[str, ...] = (
    "integer_overflow",
    "empty_input",
    "negative_numbers",
    "unicode_handling",
    "off_by_one",
    "floating_point_precision",
    "recursive_depth",
    "case_sensitivity",
    "trailing_whitespace",
    "single_element_collection",
)


# ---------------------------------------------------------------------------
# Static fallback curveballs — used when the generator response is malformed
# so the dual loop never deadlocks on a bad LLM call.
# ---------------------------------------------------------------------------

_STATIC_FALLBACK: dict[str, dict[str, str]] = {
    "integer_overflow": {
        "prompt": (
            "Read two integers a and b from stdin (space-separated) and "
            "print their product. The inputs may be up to 10^18 in absolute "
            "value."
        ),
        "sample_input": "1000000000000000000 1000000000000000000\n",
        "expected_output": "1000000000000000000000000000000000000\n",
        "gotcha": "Languages with fixed-width 64-bit ints silently overflow.",
    },
    "empty_input": {
        "prompt": (
            "Read a list of integers from stdin (one per line) and print the "
            "minimum value, or the literal string 'EMPTY' if no integers were "
            "supplied."
        ),
        "sample_input": "",
        "expected_output": "EMPTY\n",
        "gotcha": "Solvers often crash on empty stdin or print 0 / nothing.",
    },
    "negative_numbers": {
        "prompt": (
            "Read n on the first line, then n integers on the second line "
            "(space-separated). Print the count of integers strictly less "
            "than zero."
        ),
        "sample_input": "5\n-3 -1 0 1 -7\n",
        "expected_output": "3\n",
        "gotcha": "Solutions using unsigned types or abs() mis-handle negatives.",
    },
    "off_by_one": {
        "prompt": (
            "Read n on the first line. Print the sum 1+2+...+n (inclusive on "
            "both ends)."
        ),
        "sample_input": "10\n",
        "expected_output": "55\n",
        "gotcha": "Many solvers loop range(n) and produce 45 instead of 55.",
    },
}


def _default_static_case(target_failure_mode: str | None) -> dict[str, str]:
    """Return a static fallback case for the requested failure mode.

    Falls back to ``integer_overflow`` when no mode (or an unknown mode) is
    supplied. The static catalog is intentionally small — its job is to keep
    the dual loop alive when the LLM misbehaves, not to be a curriculum.
    """
    if target_failure_mode and target_failure_mode in _STATIC_FALLBACK:
        return dict(_STATIC_FALLBACK[target_failure_mode])
    # Otherwise pick the integer_overflow default — the most common gotcha.
    return dict(_STATIC_FALLBACK["integer_overflow"])


# ---------------------------------------------------------------------------
# ULID helper (kept tiny, duplicates observability.py's helper rather than
# importing — that helper is private to that module's namespace.)
# ---------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    """Return a 26-char Crockford-base32 ULID-like identifier."""
    ts_ms = int(time.time() * 1000)
    time_part = ""
    n = ts_ms
    for _ in range(10):
        time_part = _CROCKFORD[n & 0x1F] + time_part
        n >>= 5
    rand_part = "".join(random.choice(_CROCKFORD) for _ in range(16))
    return time_part + rand_part


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdversarialCase:
    """One LLM-generated curveball.

    Attributes:
        case_id: Unique identifier for this case.
        prompt: The problem statement in natural language.
        sample_input: Concrete stdin string the worker will receive.
        expected_output: The correct stdout (used by the evaluator).
        gotcha: One-sentence description of the trap the generator set.
        target_failure_mode: The failure mode the generator was steered toward
            (one of :data:`TARGET_FAILURE_MODES`), or ``None`` for free-form.
        generator_model: The LLM identifier (e.g. ``MiniMax-M2.7``) that
            produced this case. ``"fallback_static"`` for the rule-based path.
    """

    case_id: str
    prompt: str
    sample_input: str
    expected_output: str
    gotcha: str
    target_failure_mode: str | None = None
    generator_model: str = ""

    def to_benchmark_case(self, language: str = "python") -> BenchmarkCase:
        """Convert to a :class:`BenchmarkCase` for the evaluator.

        The metadata block carries the gotcha and target failure mode so that
        downstream consumers (failure clustering, diversity scoring) can group
        results by curveball category.
        """
        return BenchmarkCase(
            id=self.case_id,
            prompt=self.prompt,
            language=language,
            expected_output=self.expected_output,
            test_inputs=[self.sample_input] if self.sample_input else [],
            metadata={
                "adversarial": True,
                "gotcha": self.gotcha,
                "target_failure_mode": self.target_failure_mode or "",
                "generator_model": self.generator_model,
            },
        )


@dataclass
class AdversarialRoundResult:
    """Aggregated outcome of one generate→solve→judge round.

    Attributes:
        round_id: ULID for the round (also the correlation id on the bus).
        cases: The generated :class:`AdversarialCase` list.
        verdicts: Per-case verdict strings, aligned with ``cases``.
        worker_codes: Per-case worker solutions, aligned with ``cases``.
        scores: Per-case p_score in [0.0, 1.0], aligned with ``cases``.
        verdict_counts: Aggregate counts (``{"OK": 3, "WA": 2}``).
        failure_categories: Counts by ``target_failure_mode`` for failing
            cases — surfaces which traps the worker fell into.
        mean_score: Arithmetic mean of ``scores``.
    """

    round_id: str
    cases: list[AdversarialCase] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    worker_codes: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    verdict_counts: dict[str, int] = field(default_factory=dict)
    failure_categories: dict[str, int] = field(default_factory=dict)
    mean_score: float = 0.0

    def to_benchmark_cases(self, language: str = "python") -> list[BenchmarkCase]:
        """Return the underlying cases as a feed-able list for a SelfImprovingHarness."""
        return [c.to_benchmark_case(language=language) for c in self.cases]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class AdversarialGenerator:
    """LLM-driven curveball generator.

    Calls a chat-completions endpoint (default MiniMax) with a
    problem-generation prompt and parses the result into an
    :class:`AdversarialCase`. On any parse / HTTP / schema failure, falls
    back to a static template from :data:`_STATIC_FALLBACK` so the dual
    loop never deadlocks.

    Args:
        model: Model identifier. Default ``MiniMax-M2.7``.
        api_key: API bearer token. Falls back to ``$MINIMAX_API_KEY``.
        temperature: Sampling temperature. Higher than the improver default
            because creative curveballs are the goal. MiniMax requires
            ``(0.0, 1.0]``.
        max_tokens: Output cap.
        timeout_seconds: HTTP timeout per call.
        base_url: Override for testing (default MiniMax public endpoint).
        obs: Optional :class:`AutobenchObservability` for bus emissions.
    """

    BASE_URL = "https://api.minimax.io/v1"
    CHAT_ENDPOINT = "/chat/completions"

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
        base_url: str | None = None,
        obs: AutobenchObservability | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        # Don't raise on missing key — the static fallback handles it. We
        # only fail if a real HTTP call is attempted without a key, and
        # even then we degrade rather than crash.
        if not (0.0 < temperature <= 1.0):
            raise ValueError(
                f"AdversarialGenerator temperature must be in (0.0, 1.0], "
                f"got {temperature}"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.obs = obs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_curveball(
        self,
        domain: str = "competitive_programming",
        target_difficulty: str = "medium",
        target_failure_mode: str | None = None,
        seed_from: Any = None,
    ) -> AdversarialCase:
        """Generate one adversarial case.

        Args:
            domain: Free-form domain tag. ``competitive_programming`` is the
                default and the only one whose prompt is fully tuned;
                anything else still works but the generator may produce
                less-targeted curveballs.
            target_difficulty: One of ``easy|medium|hard``. Free-form;
                interpolated verbatim into the prompt.
            target_failure_mode: A specific gotcha to aim for (see
                :data:`TARGET_FAILURE_MODES`). When ``None`` the generator
                picks freely.
            seed_from: Optional
                :class:`autobench.test_scaffolder.CurveballCase` to riff on.
                When provided, the generator is asked to elaborate the
                static case into a full natural-language problem rather
                than starting from scratch. The dataclass is imported
                lazily inside this method so this module stays free of
                a hard dependency on :mod:`test_scaffolder`.

        Returns:
            A populated :class:`AdversarialCase`. On any failure the static
            fallback is returned with ``generator_model="fallback_static"``.
        """
        case_id = "adv-" + _ulid()
        prompt = self._build_prompt(
            domain=domain,
            target_difficulty=target_difficulty,
            target_failure_mode=target_failure_mode,
            seed_from=seed_from,
        )

        # No API key → straight to fallback. We still emit so the bus
        # records that a curveball was produced; the consumer sees
        # generator_model="fallback_static" and can filter.
        if not self.api_key:
            case = self._fallback_case(
                case_id, target_failure_mode, reason="no_api_key"
            )
            self._emit_generated(case)
            return case

        try:
            response = self._call_llm(prompt)
        except Exception as exc:  # noqa: BLE001 — degrade rather than raise
            logger.warning(
                "AdversarialGenerator HTTP call failed (%s); using static fallback",
                exc,
            )
            case = self._fallback_case(
                case_id,
                target_failure_mode,
                reason=f"http_error: {type(exc).__name__}",
            )
            self._emit_generated(case)
            return case

        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            case = self._fallback_case(
                case_id, target_failure_mode, reason="malformed_response"
            )
            self._emit_generated(case)
            return case

        parsed = self._parse_response(text)
        if parsed is None:
            case = self._fallback_case(
                case_id, target_failure_mode, reason="parse_failed"
            )
            self._emit_generated(case)
            return case

        case = AdversarialCase(
            case_id=case_id,
            prompt=parsed.get("prompt", ""),
            sample_input=parsed.get("sample_input", ""),
            expected_output=parsed.get("expected_output", ""),
            gotcha=parsed.get("gotcha", ""),
            target_failure_mode=target_failure_mode,
            generator_model=self.model,
        )
        # Belt-and-braces: if the LLM returned a structurally-empty case,
        # fall back so the worker has something concrete to chew on.
        if not case.prompt or not case.expected_output:
            case = self._fallback_case(
                case_id, target_failure_mode, reason="empty_fields"
            )
        self._emit_generated(case)
        return case

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        domain: str,
        target_difficulty: str,
        target_failure_mode: str | None,
        seed_from: Any,
    ) -> str:
        """Compose the generator prompt.

        The prompt asks for a structured JSON object so :meth:`_parse_response`
        can extract fields deterministically. We also explicitly name the
        target failure mode so the LLM can't ignore the steer.
        """
        failure_clause = (
            f"The hidden trap must specifically target the failure mode: "
            f"**{target_failure_mode}**. "
            f"Solvers who forget about {target_failure_mode} should produce "
            f"the wrong answer."
        ) if target_failure_mode else (
            "Choose any single gotcha (overflow, off-by-one, empty input, "
            "unicode, negative numbers, etc.) and target it precisely."
        )

        seed_clause = ""
        if seed_from is not None:
            # Import lazily so test_scaffolder doesn't become a hard import
            # cycle if anyone trims this module in the future.
            try:
                seed_desc = (
                    f"\nRIFF ON THIS STATIC TEMPLATE:\n"
                    f"  name: {getattr(seed_from, 'name', '?')}\n"
                    f"  category: {getattr(seed_from, 'category', '?')}\n"
                    f"  description: {getattr(seed_from, 'description', '?')}\n"
                    f"Elaborate it into a full natural-language problem.\n"
                )
                seed_clause = seed_desc
            except Exception:  # noqa: BLE001
                seed_clause = ""

        return f"""You are an adversarial problem generator for a {domain} benchmark.

Generate ONE problem at **{target_difficulty}** difficulty with a hidden
edge case designed to trip up solvers who handle the obvious cases.

{failure_clause}
{seed_clause}
OUTPUT REQUIREMENTS:
Return ONLY a JSON object with these exact keys (no markdown, no preamble):
{{
  "prompt": "<natural-language problem statement — at least 2 sentences>",
  "sample_input": "<the exact stdin string the solver will receive — may be empty>",
  "expected_output": "<the exact correct stdout the solver should produce>",
  "gotcha": "<one sentence describing the trap you set>"
}}

The sample_input and expected_output must be CONCRETE strings — not
placeholders, not 'see example'. The expected_output must match what a
correct solver would print verbatim (including any trailing newline).

Return ONLY the JSON. No surrounding prose.
"""

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Single HTTP POST to the chat-completions endpoint."""
        system_instruction = (
            "You are a creative adversarial problem-setter. "
            "Return ONLY valid JSON. No markdown, no preamble, no commentary."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}{self.CHAT_ENDPOINT}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def _parse_response(self, text: str) -> dict[str, str] | None:
        """Extract the JSON object from a model response.

        Tolerant of markdown fences and extra prose — pulls the first
        balanced ``{...}`` block and json-loads it. Returns ``None`` on any
        parse failure (caller falls back to static).
        """
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if len(lines) >= 3:
                t = "\n".join(lines[1:-1])

        match = re.search(r"\{[\s\S]*\}", t)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        # Coerce all values to strings so downstream BenchmarkCase doesn't
        # choke on a model that returned a number for expected_output.
        return {
            "prompt": str(parsed.get("prompt", "")),
            "sample_input": str(parsed.get("sample_input", "")),
            "expected_output": str(parsed.get("expected_output", "")),
            "gotcha": str(parsed.get("gotcha", "")),
        }

    def _fallback_case(
        self,
        case_id: str,
        target_failure_mode: str | None,
        reason: str,
    ) -> AdversarialCase:
        """Build a static-template case when the LLM path failed."""
        tmpl = _default_static_case(target_failure_mode)
        logger.info(
            "AdversarialGenerator using static fallback (reason=%s, mode=%s)",
            reason, target_failure_mode,
        )
        return AdversarialCase(
            case_id=case_id,
            prompt=tmpl["prompt"],
            sample_input=tmpl["sample_input"],
            expected_output=tmpl["expected_output"],
            gotcha=tmpl["gotcha"],
            target_failure_mode=target_failure_mode,
            generator_model="fallback_static",
        )

    def _emit_generated(self, case: AdversarialCase) -> None:
        """Emit a curveball_generated event. Never raises."""
        if self.obs is None:
            return
        try:
            self.obs.adversarial_curveball_generated(
                case_id=case.case_id,
                gotcha=case.gotcha,
                target_failure_mode=case.target_failure_mode,
                generator_model=case.generator_model,
                prompt_preview=case.prompt[:200],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("emit curveball_generated failed: %s", exc)


# ---------------------------------------------------------------------------
# Dual
# ---------------------------------------------------------------------------


WorkerFn = Callable[[str, HarnessConfig], str]


class AdversarialDual:
    """Generator + worker + judge co-evolution orchestrator.

    Each round:
      1. The generator produces ``n_cases`` curveballs.
      2. The worker (any callable matching the
         :class:`~autobench.evaluator.BenchmarkEvaluator` generate_fn shape)
         tries to solve each.
      3. The evaluator runs the worker's solution against
         ``sample_input``/``expected_output`` and emits a verdict.
      4. Per-case results are aggregated into
         :class:`AdversarialRoundResult`.

    The output is a list of :class:`BenchmarkCase`-shaped objects ready to
    feed into a :class:`autobench.rsi_loop.SelfImprovingHarness` as
    drift-resistant evaluation data — every iteration sees a *new*
    curriculum, so the harness cannot memorise the test set.

    Args:
        generator: The :class:`AdversarialGenerator` that produces cases.
        worker: A callable ``(prompt, HarnessConfig) -> code`` that solves
            problems. Typically ``MiniMaxWorker.generate`` from
            :mod:`autobench.worker_agent` once that lands.
        evaluator: A :class:`BenchmarkEvaluator` with its own
            ``generate_fn`` already wired to ``worker``. The dual passes the
            evaluator a one-case batch per generated curveball.
        harness: HarnessConfig the worker should run under. Defaults to a
            minimal ``HarnessConfig()`` so a smoke test can construct a
            Dual without thinking about it.
        obs: Optional observability for ``round_complete`` emissions.
    """

    def __init__(
        self,
        generator: AdversarialGenerator,
        worker: WorkerFn,
        evaluator: BenchmarkEvaluator,
        harness: HarnessConfig | None = None,
        obs: AutobenchObservability | None = None,
        language: str = "python",
    ) -> None:
        self.generator = generator
        self.worker = worker
        self.evaluator = evaluator
        self.harness = harness or HarnessConfig()
        self.obs = obs
        self.language = language

    def run_round(
        self,
        n_cases: int = 5,
        target_failure_modes: list[str | None] | None = None,
        target_difficulty: str = "medium",
        domain: str = "competitive_programming",
    ) -> AdversarialRoundResult:
        """Run one full generate→solve→judge round.

        Args:
            n_cases: How many curveballs to generate.
            target_failure_modes: Optional list of failure modes to steer the
                generator toward (one per case). When shorter than
                ``n_cases``, remaining cases use ``None`` (free choice).
                When ``None``, all cases are free-choice.
            target_difficulty: Difficulty passed to the generator.
            domain: Domain tag passed to the generator.

        Returns:
            :class:`AdversarialRoundResult` with per-case verdicts and
            aggregate stats.
        """
        round_id = "advr-" + _ulid()
        modes: list[str | None] = list(target_failure_modes or [])
        while len(modes) < n_cases:
            modes.append(None)

        cases: list[AdversarialCase] = []
        verdicts: list[str] = []
        worker_codes: list[str] = []
        scores: list[float] = []
        verdict_counts: dict[str, int] = {}
        failure_categories: dict[str, int] = {}

        # Wire the evaluator's generate_fn to our worker for this round,
        # capturing the produced code so we can report it back. We restore
        # the original generate_fn afterwards so callers who reuse the
        # evaluator outside the dual aren't surprised.
        original_generate = self.evaluator.generate_fn
        last_code: dict[str, str] = {}

        def _capturing_worker(prompt: str, cfg: HarnessConfig) -> str:
            try:
                code = self.worker(prompt, cfg) or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("worker raised on adversarial case: %s", exc)
                code = ""
            last_code["code"] = code
            return code

        self.evaluator.generate_fn = _capturing_worker

        try:
            for i in range(n_cases):
                case = self.generator.generate_curveball(
                    domain=domain,
                    target_difficulty=target_difficulty,
                    target_failure_mode=modes[i],
                )
                cases.append(case)

                last_code["code"] = ""
                bench_case = case.to_benchmark_case(language=self.language)
                bench_result = self.evaluator.run(self.harness, [bench_case])

                if not bench_result.case_results:
                    # Shouldn't happen, but defend against an empty list.
                    verdicts.append("RE")
                    worker_codes.append(last_code["code"])
                    scores.append(0.0)
                    verdict_counts["RE"] = verdict_counts.get("RE", 0) + 1
                    if case.target_failure_mode:
                        failure_categories[case.target_failure_mode] = (
                            failure_categories.get(case.target_failure_mode, 0) + 1
                        )
                    continue

                hr = bench_result.case_results[0]
                v = hr.verdict.value if hasattr(hr.verdict, "value") else str(hr.verdict)
                verdicts.append(v)
                worker_codes.append(last_code["code"])
                scores.append(float(hr.p_score))
                verdict_counts[v] = verdict_counts.get(v, 0) + 1

                # Failure-category tally: any non-OK verdict on a steered
                # case counts as a successful trap.
                if v != "OK" and case.target_failure_mode:
                    failure_categories[case.target_failure_mode] = (
                        failure_categories.get(case.target_failure_mode, 0) + 1
                    )
        finally:
            self.evaluator.generate_fn = original_generate

        mean_score = (sum(scores) / len(scores)) if scores else 0.0
        result = AdversarialRoundResult(
            round_id=round_id,
            cases=cases,
            verdicts=verdicts,
            worker_codes=worker_codes,
            scores=scores,
            verdict_counts=verdict_counts,
            failure_categories=failure_categories,
            mean_score=mean_score,
        )

        self._emit_round_complete(result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_round_complete(self, result: AdversarialRoundResult) -> None:
        if self.obs is None:
            return
        try:
            self.obs.adversarial_round_complete(
                round_id=result.round_id,
                n_cases=len(result.cases),
                verdict_counts=result.verdict_counts,
                failure_categories=result.failure_categories,
                mean_score=result.mean_score,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("emit adversarial_round_complete failed: %s", exc)


# ---------------------------------------------------------------------------
# Benchmark-assembly helpers (wire-pop Phase 3 — nervous-bus-gdzo)
# ---------------------------------------------------------------------------
#
# The PopulationRunner builds its per-cycle benchmark case set from a base
# corpus (cf-tier-1 or daily curriculum). Phase 3 closes the co-evolution loop
# by replacing ~20% of cases with adversarially-generated gotchas keyed to the
# *previous* cycle's failure modes. These helpers live here (alongside the
# generator) so the benchmark-assembly module stays thin.

#: First-cycle fallback when no prior PopulationResult is available. The bead
#: (nervous-bus-gdzo) names these four modes explicitly. They include modes
#: that are NOT in :data:`TARGET_FAILURE_MODES` (e.g. ``missing_modulo``,
#: ``edge_case_empty_input``) — the generator interpolates them as free-form
#: strings, which is fine: the failure-mode label only steers the prompt.
DEFAULT_FAILURE_MODES: tuple[str, ...] = (
    "off_by_one",
    "edge_case_empty_input",
    "missing_modulo",
    "integer_overflow",
)


def mine_failure_modes_from_result(prior_result: Any) -> list[str]:
    """Extract a failure-mode list from the previous cycle's PopulationResult.

    Strategy:
      1. Look at each advocate's iteration history. For non-OK verdicts in the
         last iteration's :class:`BenchmarkResult.verdict_counts`, map the
         verdict letter to a heuristic failure-mode label.
      2. Tally and return the most common labels (deduped, ordered by
         frequency).
      3. If no signal can be mined, return an empty list (caller falls back
         to :data:`DEFAULT_FAILURE_MODES`).

    Verdict → failure-mode mapping is intentionally small and heuristic — the
    cleaner long-term path is to mine refuted AHE predictions, but per the
    bead this is the "ship with a TODO" fallback.

    TODO(nervous-bus-gdzo-followup): Mine refuted AHE prediction deltas
    (``verdict_pattern`` in ``PredictionVerification``) for sharper steering.
    """
    if prior_result is None:
        return []

    verdict_to_mode = {
        "WA": "off_by_one",
        "TLE": "integer_overflow",
        "RE": "edge_case_empty_input",
        "MLE": "integer_overflow",
        "CE": "edge_case_empty_input",
    }

    counts: dict[str, int] = {}
    advocates = getattr(prior_result, "advocates", None) or []
    for advocate in advocates:
        history = getattr(advocate, "history", None) or []
        if not history:
            continue
        # Inspect last iteration's verdict_counts.
        _h, last_result, _d = history[-1]
        verdict_counts = getattr(last_result, "verdict_counts", None) or {}
        for verdict, count in verdict_counts.items():
            if verdict == "OK":
                continue
            mode = verdict_to_mode.get(str(verdict))
            if mode is None:
                continue
            counts[mode] = counts.get(mode, 0) + int(count)

    if not counts:
        return []

    # Sort by frequency descending, then alphabetically for stable ordering.
    return sorted(counts.keys(), key=lambda m: (-counts[m], m))


def _round_up_ratio(total: int, ratio: float) -> int:
    """Return ``ceil(total * ratio)``, clamped to ``[0, total]``.

    Used to size the adversarial slice of an assembled benchmark — the bead
    spec says "round up": 1-of-5, 2-of-10, etc.
    """
    if total <= 0:
        return 0
    if ratio <= 0.0:
        return 0
    raw = total * ratio
    n = int(raw)
    if raw > n:
        n += 1
    return min(max(0, n), total)


def generate_adversarial_case_mix(
    n_cases: int,
    failure_modes: list[str] | None = None,
    generator: AdversarialGenerator | None = None,
    obs: AutobenchObservability | None = None,
    language: str = "python",
    target_difficulty: str = "medium",
    domain: str = "competitive_programming",
) -> list[BenchmarkCase]:
    """Produce ``n_cases`` adversarial :class:`BenchmarkCase` instances.

    Each curveball is steered toward a failure mode from ``failure_modes``,
    cycling through the list if shorter than ``n_cases``. Falls back to
    :data:`DEFAULT_FAILURE_MODES` when ``failure_modes`` is empty.

    Side effects:
      * Emits one ``autobench.adversarial.curveball_generated.v1`` event per
        case (via the generator's own emit path — never raises).
      * Emits one ``autobench.adversarial.round_complete.v1`` event summarising
        the assembled set when ``obs`` is provided. ``verdict_counts`` /
        ``failure_categories`` are placeholders here because no worker has
        run yet — they get a single ``"PENDING": n_cases`` marker so
        downstream consumers can identify these as assembly-time emissions.

    Args:
        n_cases: How many curveballs to generate. ``0`` returns ``[]``.
        failure_modes: Failure modes to cycle through. Empty / None →
            :data:`DEFAULT_FAILURE_MODES`.
        generator: Optional :class:`AdversarialGenerator`. When ``None`` a
            default one is constructed (which will degrade to the static
            fallback when no MINIMAX_API_KEY is set).
        obs: Optional observability for the ``round_complete`` summary.
            ``curveball_generated`` events are emitted via ``generator.obs``,
            so pass ``obs`` to the generator if you want both event types on
            the same session id.
        language: Target language tag stamped on the BenchmarkCase.
        target_difficulty: Generator difficulty hint.
        domain: Generator domain tag.

    Returns:
        List of :class:`BenchmarkCase` objects ready to interleave into a
        benchmark set. Length equals ``n_cases`` (or ``0`` when ``n_cases <= 0``).
    """
    if n_cases <= 0:
        return []

    modes = list(failure_modes or []) or list(DEFAULT_FAILURE_MODES)
    if generator is None:
        generator = AdversarialGenerator(obs=obs)

    round_id = "advr-" + _ulid()
    cases: list[BenchmarkCase] = []
    adversarial_cases: list[AdversarialCase] = []
    failure_categories: dict[str, int] = {}

    for i in range(n_cases):
        mode = modes[i % len(modes)]
        adv_case = generator.generate_curveball(
            domain=domain,
            target_difficulty=target_difficulty,
            target_failure_mode=mode,
        )
        adversarial_cases.append(adv_case)
        cases.append(adv_case.to_benchmark_case(language=language))
        if adv_case.target_failure_mode:
            failure_categories[adv_case.target_failure_mode] = (
                failure_categories.get(adv_case.target_failure_mode, 0) + 1
            )

    # Emit a round_complete summary so downstream consumers see the assembly
    # event even when no worker run follows. verdict_counts uses a single
    # "PENDING" marker because no verdict has been adjudicated yet.
    if obs is not None:
        try:
            obs.adversarial_round_complete(
                round_id=round_id,
                n_cases=len(adversarial_cases),
                verdict_counts={"PENDING": len(adversarial_cases)},
                failure_categories=failure_categories,
                mean_score=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("emit adversarial_round_complete failed: %s", exc)

    return cases


__all__ = [
    "AdversarialCase",
    "AdversarialDual",
    "AdversarialGenerator",
    "AdversarialRoundResult",
    "DEFAULT_FAILURE_MODES",
    "TARGET_FAILURE_MODES",
    "generate_adversarial_case_mix",
    "mine_failure_modes_from_result",
]
