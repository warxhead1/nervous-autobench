"""Anthropic LLM improver for autobench's RSI loop.

Calls the Anthropic SDK directly (no LangGraph / deer-flow gateway).
Implements the SICA-style step-(4) "improve" call for a self-improving
coding-agent harness.

Public surface: :class:`AnthropicLLMWrapper`.
Architecture history (deer-flow REST API endpoints, SICA pattern notes,
nervous-bus-6ed verdict-strategy removal) is preserved in
``_checkpoints/architecture-history.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
from typing import Any

from ..core import ContextManager, HarnessConfig, RolloutProtocol
from ..observability import AutobenchObservability
from ..rsi.loop import ImprovementDelta
from .base import LLMImprovementResult, _build_evidence_section, _tolerant_json_loads

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hypothetical REST API types (not yet implemented in deer-flow gateway)
# ---------------------------------------------------------------------------

# These would live in deer-flow if a proper /api/harness/improve endpoint
# were added. Documenting the shape here so the integration path is clear.
#
# class HarnessImprovementRequest(BaseModel):
#     harness: dict
#     benchmark_result: dict
#     question: str | None = None
#
# class HarnessImprovementResponse(BaseModel):
#     suggested_harness: dict
#     delta: dict
#     rationale: str


# ---------------------------------------------------------------------------
# Verdict → improvement strategy (SICA-style dominant-verdict diagnosis)
# ---------------------------------------------------------------------------

# These rules mirror the heuristic logic in rsi_loop.py improve_harness(),
# but expressed as a prompt-fragment library so the LLM gets consistent guidance.
# The SICA insight is: dominant verdict => specific fix, not general guidance.

# NOTE on removed VERDICT_STRATEGIES (2026-05-16, nervous-bus-6ed):
#
# This module previously exposed a `VERDICT_STRATEGIES` dict that mapped each
# dominant verdict class to a hardcoded "Guidance: do X" string, which was
# injected verbatim into the diagnosis prompt as a "DOMINANT VERDICT ANALYSIS"
# section. The improver dutifully followed the guidance and proposed the
# prescribed delta — it never actually diagnosed.
#
# Session 01KRQTNMM8RFKC477DRRRVMVS4 made this concrete: 77% CE rate driven
# by `<think>` reasoning prose leaking into code submissions. The canned
# guidance said "reduce max_tokens, simplify code" — wrong fix entirely. The
# improver could not see the `<think>` prefix in generated_code because the
# prompt didn't include generated_code at all.
#
# The current diagnosis prompt now includes per-verdict-class evidence
# (sample generated_code previews) and asks the improver to diagnose the
# structural pattern from that evidence directly.


# Pre-clean rules for near-miss JSON emitted by LLMs. Each one is a tiny
# transformation that fixes a specific recurring failure mode WITHOUT changing
# semantics. Applied in order; final json.loads is strict so any rule that
# would corrupt valid JSON is also rejected by the strict pass below.
#
# Why we need this: in session 01KRRP71DVQS52XMM858H5W7QN (2026-05-16) the
# improver emitted a sensible delta with `"OK":+2,"CE":-2` in the
# predicted_verdict_class_changes block. Python json rejects the leading `+`,
# falls back to rule-based, and the entire RSI iter→iter signal is lost.
# One character was blocking the whole loop.
#
# The rules and the ``_tolerant_json_loads`` / ``_build_evidence_section``
# helpers themselves now live in :mod:`autobench.llm.base` so the
# minimax wrapper can share them.


class AnthropicLLMWrapper:
    """LLM improver using the Anthropic SDK directly.

    This class wraps the Anthropic SDK to provide a clean interface for
    harness improvement suggestions. It is NOT a deer-flow client — it calls
    the Anthropic API directly using the same pattern as deer-flow's
    `ClaudeChatModel` but without the LangGraph wrapper.

    Usage::

        improver = AnthropicLLMWrapper(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-sonnet-4-6",
        )
        result = improver.suggest_harness_improvements(
            current_config=harness_config,
            benchmark_results=benchmark_result,
        )
        # result is an LLMImprovementResult with suggested_harness + delta

    Integration path with deer-flow:
    - For a full deer-flow integration, a new endpoint `POST /api/harness/improve`
      could be added to the gateway that wraps this class with auth + rate limiting.
    - The `deer act` command could also invoke this improver if queue inspection
      reveals RSI patterns worth acting on (hypothetical — not yet implemented).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        thinking_enabled: bool = False,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the LLM improver.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Model name (default: claude-sonnet-4-6).
            thinking_enabled: Whether to enable extended thinking (default False
                since this is a structured output task, not open-ended reasoning).
            max_tokens: Max output tokens (default 4096 — sufficient for JSON).
            timeout_seconds: Per-request timeout (default 60s).
            max_retries: Number of retries on rate limit / server error (default 3).
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set or passed as api_key argument"
            )

        self.model = model
        self.thinking_enabled = thinking_enabled
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Import here so this module works even if the SDK isn't installed
        # (autobench can still fall back to the rule-based improver).
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for AnthropicLLMWrapper. "
                "Install with: pip install anthropic"
            ) from exc

        self._anthropic = anthropic

    def suggest_harness_improvements(
        self,
        current_config: HarnessConfig,
        benchmark_results: "BenchmarkResult",  # BenchmarkResult from evaluator.py
        obs: AutobenchObservability | None = None,
        iteration: int = 0,
        revert_history: list[dict[str, Any]] | None = None,
    ) -> LLMImprovementResult:
        """Generate improved harness configuration from benchmark results.

        Uses the SICA-style dominant-verdict diagnosis: find the most common
        non-OK verdict, apply the corresponding improvement strategy via LLM.

        Args:
            current_config: The current harness configuration.
            benchmark_results: Results from the last benchmark run.
            revert_history: Best-iter-keep revert records from the RSI loop
                (nervous-bus-sf0y). When non-empty, surfaced to the
                improver as a REVERT HISTORY block so it can reconsider
                instead of doubling down on a regressed hypothesis.

        Returns:
            LLMImprovementResult with suggested_harness, delta, and metadata.

        Raises:
            ValueError: If benchmark_results has no cases.
            RuntimeError: If all retries are exhausted.
        """
        start_time = time.monotonic()

        # Build the diagnosis prompt
        prompt = self._build_diagnosis_prompt(
            current_config, benchmark_results, revert_history=revert_history,
        )

        # Call the LLM with retry logic
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._call_anthropic(prompt)
                break
            except self._anthropic.RateLimitError as e:
                last_error = e
                backoff_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(
                    "Rate limited on attempt %d/%d, retrying after %dms",
                    attempt, self.max_retries, backoff_ms
                )
                time.sleep(backoff_ms / 1000)
            except self._anthropic.InternalServerError as e:
                last_error = e
                backoff_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(
                    "Server error on attempt %d/%d, retrying after %dms",
                    attempt, self.max_retries, backoff_ms
                )
                time.sleep(backoff_ms / 1000)

        if "response" not in locals():
            raise RuntimeError(
                f"AnthropicLLMWrapper exhausted {self.max_retries} retries"
            ) from last_error

        latency_ms = (time.monotonic() - start_time) * 1000

        # Parse and build result
        suggested_harness, delta = self._parse_llm_response(
            current_config, response.content[0].text
        )

        # Estimate cost (Anthropic pricing as of May 2026)
        input_tokens = response.usage.input_tokens if hasattr(response, "usage") else 0
        output_tokens = response.usage.output_tokens if hasattr(response, "usage") else 0
        cost = self._estimate_cost(input_tokens, output_tokens)

        raw_text = response.content[0].text

        if obs is not None:
            is_empty_delta = (
                not delta.system_prompt_delta
                and not delta.rollout_protocol_changed
                and not delta.context_manager_changed
                and not delta.tool_surface_delta
                and not delta.budget_delta
            )
            if is_empty_delta and delta.improvement_summary in ("", "LLM-generated improvement"):
                parse_status = "fell_back_to_rule_based"
                fallback_reason = "json_extract_failed_or_no_keys"
            elif is_empty_delta:
                parse_status = "no_change"
                fallback_reason = None
            else:
                parse_status = "ok"
                fallback_reason = None
            obs.improver_reasoning(
                model=self.model,
                iteration=iteration,
                prompt=prompt,
                raw_response=raw_text,
                parsed_delta=asdict(delta),
                parse_status=parse_status,
                fallback_reason=fallback_reason,
                latency_ms=latency_ms,
                cost_dollars=cost,
            )

        return LLMImprovementResult(
            suggested_harness=suggested_harness,
            delta=delta,
            raw_response=raw_text,
            model_used=self.model,
            tokens_used=output_tokens,
            cost_dollars=cost,
            latency_ms=latency_ms,
        )

    def _call_anthropic(self, prompt: str) -> Any:
        """Make a single call to the Anthropic API."""
        client = self._anthropic.Anthropic(api_key=self.api_key)

        system_instruction = (
            "You are improving a coding agent harness configuration. "
            "Return ONLY valid JSON — no markdown, no explanation, no preamble. "
            "The JSON must have these exact keys:\n"
            "  system_prompt_changes: string (what to add/change in system_prompt)\n"
            "  rollout_protocol: string (one of: single, iterative, self_revision, monte_carlo, keep)\n"
            "  context_manager: string (one of: full, budgeted, semantic, hierarchical, keep)\n"
            "  tool_surface_changes: string (what to add/change in tool_surface)\n"
            "  budget_changes: object with keys max_tokens, max_time_seconds, max_cost_dollars (all optional)\n"
            "  rationale: string (brief explanation of the changes)\n"
        )

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout_seconds,
        )
        return message

    def _build_diagnosis_prompt(
        self,
        current_config: HarnessConfig,
        benchmark_results: "BenchmarkResult",
        revert_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build an evidence-driven diagnosis prompt.

        The prompt includes:
        1. Optional REVERT HISTORY block (nervous-bus-sf0y) when the RSI
           loop has rolled back from a >variance_floor regression.
        2. Current harness config (abbreviated)
        3. Benchmark results summary with verdict distribution
        4. Per-verdict-class evidence: sample generated_code previews so the
           improver can diagnose structural failure patterns directly from
           the data rather than from a canned verdict→fix lookup.
        5. Instruction to return JSON with specific change fields.
        """
        verdict_counts = benchmark_results.verdict_counts
        total_cases = max(len(benchmark_results.case_results), 1)

        verdict_pct = {
            k: f"{v / total_cases * 100:.0f}%" for k, v in verdict_counts.items()
        }

        evidence = _build_evidence_section(benchmark_results.case_results)

        # Format budget for prompt
        budget_lines = "\n".join(
            f"    {k}: {v}" for k, v in current_config.budget.items()
        )

        # Lazy import — keep anthropic free of minimax import unless
        # actually called. ``_format_revert_history_block`` is defined in
        # minimax so both wrappers share the rendering.
        from .minimax import _format_revert_history_block
        revert_block = _format_revert_history_block(revert_history)

        # nervous-bus-19ur: full-feed (no 200/100-char truncation). The
        # delta is appended to system_prompt each iteration, so the improver
        # needs to see the accumulated text to propose coherent edits.
        return f"""Analyze benchmark results and generate an improved harness configuration.
{revert_block}
CURRENT HARNESS CONFIG:
- system_prompt: {current_config.system_prompt}
- rollout_protocol: {current_config.rollout_protocol.value}
- context_manager: {current_config.context_manager.value}
- tool_surface: {current_config.tool_surface}
- budget:
{budget_lines}

BENCHMARK RESULTS:
- aggregate_score: {benchmark_results.aggregate_score:.3f}
- pass_rate: {benchmark_results.pass_rate():.1%}
- total_latency_ms: {benchmark_results.total_latency_ms:.0f}
- verdict_distribution: {verdict_pct}

EVIDENCE — sample generated code per verdict class:
{evidence}

DIAGNOSIS INSTRUCTIONS:
Examine the EVIDENCE block above and identify the specific STRUCTURAL or
SEMANTIC failure pattern shared by cases with the same non-OK verdict.
Look for: prose prefixes before code (e.g. "<think>" tags, "Here's the
solution:" preambles, explanatory paragraphs), missing imports, wrong
language emitted, malformed function bodies, lines not actually code, etc.
Compare to passing (OK) cases to spot what they have that the failures
lack. Your delta should target the specific pattern you observed, NOT
generic guidance. The aggregate score will only improve if your delta
actually addresses the failure cause visible in the evidence.

CONSTRAINT (nervous-bus-ldd1):
tool_surface_changes must be empty string, "keep", or "no change". The
harness does not support dynamic tool registration; proposing new tools
(e.g. "add a validate_code_start tool", "add a syntax_check function")
will be ignored at parse time and logged as a warning. Improvements come
from prompt + protocol + budget changes, NOT from inventing APIs the
worker cannot call.

OUTPUT REQUIREMENTS:
Return a JSON object with these exact keys:
{{
  "system_prompt_changes": "what to add/change in system_prompt (string)",
  "rollout_protocol": "single|iterative|self_revision|monte_carlo|keep",
  "context_manager": "full|budgeted|semantic|hierarchical|keep",
  "tool_surface_changes": "must be \"\" or \"keep\" — see CONSTRAINT above",
  "budget_changes": {{"max_tokens": N, "max_time_seconds": N, "max_cost_dollars": N}},
  "rationale": "brief explanation of the changes",
  "prediction": {{
    "predicted_score_delta": <float, expected change in aggregate_score next iteration>,
    "predicted_verdict_class_changes": {{"OK": +N, "TLE": -N, "WA": -N, "...": ...}},
    "confidence": <0.0 to 1.0>,
    "rationale": "<one sentence on why you expect this outcome>"
  }}
}}

The "prediction" field implements the AHE falsifiability contract — your edit
is verified against the next iteration's actuals. Be honest about confidence;
overconfident wrong predictions are surfaced as warnings.

Return ONLY valid JSON, no markdown or surrounding text.
"""

    def _parse_llm_response(
        self,
        current_config: HarnessConfig,
        response_text: str,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        """Parse LLM JSON response into HarnessConfig + ImprovementDelta.

        This is the same logic as rsi_loop._parse_llm_improvement() but
        returns a full LLMImprovementResult instead of a tuple.
        """
        delta = ImprovementDelta()

        # Try to extract JSON from response (handle markdown code fences)
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1])
        elif text.startswith("```json"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1])

        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            logger.warning("Could not extract JSON from LLM response")
            return current_config, delta

        parsed = _tolerant_json_loads(json_match.group())
        if parsed is None:
            logger.warning("JSON parse error (incl. tolerant fallback)")
            return current_config, delta

        # Build new harness config
        new_harness = HarnessConfig(
            system_prompt=current_config.system_prompt,
            rollout_protocol=current_config.rollout_protocol,
            context_manager=current_config.context_manager,
            tool_surface=current_config.tool_surface,
            verifiers=current_config.verifiers,
            budget=current_config.budget.copy(),
        )

        delta.system_prompt_delta = parsed.get("system_prompt_changes", "")
        if delta.system_prompt_delta:
            new_harness.system_prompt = (
                current_config.system_prompt + "\n" + delta.system_prompt_delta
            )

        rp = parsed.get("rollout_protocol", "keep")
        if rp != "keep":
            try:
                new_harness.rollout_protocol = RolloutProtocol(rp)
                delta.rollout_protocol_changed = True
            except ValueError:
                pass

        cm = parsed.get("context_manager", "keep")
        if cm != "keep":
            try:
                new_harness.context_manager = ContextManager(cm)
                delta.context_manager_changed = True
            except ValueError:
                pass

        ts = parsed.get("tool_surface_changes", "")
        if ts:
            delta.tool_surface_delta = ts
            new_harness.tool_surface = current_config.tool_surface + "\n" + ts

        bc = parsed.get("budget_changes", {})
        if bc:
            delta.budget_delta = bc
            new_harness.budget.update(bc)

        delta.improvement_summary = parsed.get("rationale", "LLM-generated improvement")
        delta.delta_score = 0.0  # Calculated by caller after benchmark

        # AHE prediction contract: attach a Prediction if the model supplied one.
        try:
            from ..ahe import parse_prediction_from_llm_response
            delta.prediction = parse_prediction_from_llm_response(response_text)
        except Exception:  # noqa: BLE001 — prediction is best-effort
            pass

        return new_harness, delta

    @staticmethod
    def _calc_backoff_ms(attempt: int, error: Exception) -> int:
        """Exponential backoff with 20% jitter, respects Retry-After header."""
        base_ms = 2000 * (1 << (attempt - 1))
        jitter_ms = int(base_ms * 0.2)
        total_ms = base_ms + jitter_ms

        if hasattr(error, "response") and error.response is not None:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    total_ms = int(retry_after) * 1000
                except (ValueError, TypeError):
                    pass

        return total_ms

    @staticmethod
    def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in dollars based on Anthropic pricing (May 2026).

        Pricing for claude-sonnet-4-6:
        - Input: $3.50/MTok
        - Output: $17.50/MTok

        These rates are approximate; actual billing depends on model used.
        """
        input_cost = (input_tokens / 1_000_000) * 3.50
        output_cost = (output_tokens / 1_000_000) * 17.50
        return input_cost + output_cost

