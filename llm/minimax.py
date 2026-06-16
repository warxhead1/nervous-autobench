"""MiniMax LLM improver — default for the autobench RSI loop.

Drop-in replacement for :class:`AnthropicLLMWrapper`. Same public methods
and return type, Anthropic-compatible ``/v1/messages`` endpoint (also
supports ``/v1/chat/completions`` for A/B). Failures fall back to the
rule-based improver so the RSI loop never wedges.

Wire shape, billing rationale, and the "why not OpenAI" notes (reasoning
goes into a separate ``thinking`` block, eliminating JSON-collision
risk) are preserved in ``_checkpoints/architecture-history.md``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from dataclasses import asdict

from ..core import ContextManager, HarnessConfig, RolloutProtocol, Verdict
from .anthropic import LLMImprovementResult, _build_evidence_section, _tolerant_json_loads
from ..observability import AutobenchObservability
from ..rsi.loop import ImprovementDelta, improve_harness

logger = logging.getLogger(__name__)


_NO_OP_DELTA_TOKENS = frozenset({"", "keep", "no change", "no_change", "nochange", "none", "null"})


def _format_siblings_block(
    cross_advocate_context: list[Any] | None,
) -> str:
    """Render a SIBLINGS block for the diagnosis prompt (nervous-bus-bo86).

    ``cross_advocate_context`` is an iterable of :class:`ImprovementDelta`
    instances (or dict-equivalents) representing the most recent hypothesis
    each already-completed sibling advocate produced THIS cycle. Empty /
    None → returns ``""`` so callers interpolate unconditionally without
    stray whitespace.

    The block is soft encouragement only — the improver may still propose
    a similar hypothesis; the hard signal is the post-cycle adjusted_score
    bonus that biases winner-selection toward exploratory lineages.
    """
    if not cross_advocate_context:
        return ""
    lines = [
        "",
        "SIBLINGS (Phase 2 of wire-pop, nervous-bus-bo86):",
        "Your sibling advocates this cycle recently proposed the following",
        "hypotheses. Produce a hypothesis that is STRUCTURALLY DISTINCT from",
        "these — touch different fields, or push the same field in a different",
        "direction. Population diversity is a tracked metric.",
    ]
    for i, d in enumerate(cross_advocate_context):
        # Accept ImprovementDelta-shaped objects OR plain dicts.
        summary = getattr(d, "improvement_summary", None)
        sysprompt = getattr(d, "system_prompt_delta", None)
        proto_changed = getattr(d, "rollout_protocol_changed", None)
        ctx_changed = getattr(d, "context_manager_changed", None)
        budget = getattr(d, "budget_delta", None)
        if summary is None and isinstance(d, dict):
            summary = d.get("improvement_summary")
            sysprompt = d.get("system_prompt_delta")
            proto_changed = d.get("rollout_protocol_changed")
            ctx_changed = d.get("context_manager_changed")
            budget = d.get("budget_delta")
        parts: list[str] = []
        if summary:
            parts.append(f'summary="{str(summary)[:120]}"')
        if sysprompt:
            parts.append(f'system_prompt+="{str(sysprompt)[:80]}"')
        if proto_changed:
            parts.append("rollout_protocol_changed")
        if ctx_changed:
            parts.append("context_manager_changed")
        if budget:
            keys = sorted(list(budget.keys())) if isinstance(budget, dict) else []
            if keys:
                parts.append(f"budget_keys={keys}")
        body = "; ".join(parts) if parts else "(no-op delta)"
        lines.append(f"  - sibling-{i}: {body}")
    lines.append("")
    return "\n".join(lines)


def _format_revert_history_block(
    revert_history: list[dict[str, Any]] | None,
) -> str:
    """Render a REVERT HISTORY block for the diagnosis prompt (nervous-bus-sf0y).

    Empty / None → returns ``""`` so callers can interpolate unconditionally
    without producing stray whitespace. Otherwise renders one line per revert::

        REVERT HISTORY (best-iter-keep, nervous-bus-sf0y):
        - iter 1 regressed by -0.075 (score 0.628 vs best 0.703 at iter 0); reverted to iter 0's harness. Reconsider hypothesis.

    The block is wrapped with leading + trailing blank lines so it slots
    cleanly into a multi-section prompt.
    """
    if not revert_history:
        return ""
    lines = ["", "REVERT HISTORY (best-iter-keep, nervous-bus-sf0y):"]
    for entry in revert_history:
        try:
            iter_n = int(entry.get("iter_regressed", -1))
            iter_score = float(entry.get("iter_score", 0.0))
            best_score = float(entry.get("best_score", 0.0))
            best_iter = int(entry.get("best_iter", -1))
            delta = float(entry.get("regression_delta", iter_score - best_score))
        except (TypeError, ValueError):
            continue
        lines.append(
            f"- iter {iter_n} regressed by {delta:+.3f} "
            f"(score {iter_score:.3f} vs best {best_score:.3f} at iter "
            f"{best_iter}); reverted to iter {best_iter}'s harness. "
            f"Reconsider hypothesis — your previous edit caused this regression."
        )
    lines.append("")
    return "\n".join(lines)


def _is_no_op_value(value: Any) -> bool:
    """Return True if ``value`` is a model-emitted 'no change' sentinel.

    LLMs express 'leave this field alone' in many surface forms — empty
    string, the literal word "keep", "no change", a JSON null parsed as
    None. Treating all of these as no-ops prevents the delta applier from
    appending those tokens as literal text to system_prompt / tool_surface.
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _NO_OP_DELTA_TOKENS


# ---------------------------------------------------------------------------
# Billing unit — REQUESTS, not dollars (nervous-bus-dq7l).
# ---------------------------------------------------------------------------
# Pricing table deliberately removed. The MiniMax coding plan bills by
# requests-per-5h (14250 cap). Any in-tree $/token rate is a fiction the
# moment list pricing shifts. cost_dollars on observability events is now
# hardcoded to 0.0 — schema field retained for back-compat only.


class MiniMaxLLMWrapper:
    """LLM improver using the MiniMax chat-completions API directly.

    Drop-in replacement for :class:`AnthropicLLMWrapper`. Same public
    methods, same return type (:class:`LLMImprovementResult`), same
    JSON-extraction logic. The only differences are the HTTP shape
    (OpenAI-compatible) and the fact that any failure path here yields
    a rule-based fallback rather than a raised exception, because this
    wrapper is intended to be the *default* improver in the RSI loop.

    Usage::

        improver = MiniMaxLLMWrapper(api_key=os.environ["MINIMAX_API_KEY"])
        result = improver.suggest_harness_improvements(
            current_config=harness,
            benchmark_results=bench_result,
        )
        # or, drop-in for SelfImprovingHarness.improver_fn:
        new_harness, delta = improver.improve(harness, bench_result)
    """

    BASE_URL = "https://api.minimax.io"
    OPENAI_PATH = "/v1/chat/completions"
    ANTHROPIC_PATH = "/anthropic/v1/messages"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-M2.7",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        base_url: str | None = None,
        endpoint_mode: str = "anthropic",
        thinking_budget: int | None = 2048,
    ) -> None:
        """Construct a MiniMax improver wrapper.

        About ``thinking_budget`` (verified honoured against
        ``/anthropic/v1/messages`` 2026-05-16 via
        ``tools/check_minimax_thinking.py``): when set in ``"anthropic"``
        mode, attaches ``thinking={"type": "enabled", "budget_tokens": N}``
        so the reasoning trace is capped without truncating the JSON-delta
        answer. Default 2048 (higher than the worker's 1024) because
        diagnosing harness deltas from per-case evidence is more
        reasoning-heavy than per-case code generation — we want the
        improver to actually reason about failure patterns, not just spit
        out a templated delta. Pass ``None`` to omit the field.

        In ``"openai"`` mode the field is omitted regardless.
        """
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "MINIMAX_API_KEY must be set or passed as api_key argument"
            )

        # MiniMax requires temperature in (0.0, 1.0] — strictly > 0.
        if not (0.0 < temperature <= 1.0):
            raise ValueError(
                f"MiniMax temperature must be in (0.0, 1.0], got {temperature}"
            )

        if endpoint_mode not in ("openai", "anthropic"):
            raise ValueError(
                f"endpoint_mode must be 'openai' or 'anthropic', got {endpoint_mode!r}"
            )

        if thinking_budget is not None:
            if not isinstance(thinking_budget, int) or thinking_budget <= 0:
                raise ValueError(
                    "thinking_budget must be None or a positive int, "
                    f"got {thinking_budget!r}"
                )
            if thinking_budget > max_tokens:
                raise ValueError(
                    f"thinking_budget ({thinking_budget}) must be <= "
                    f"max_tokens ({max_tokens})"
                )

        self.model = os.environ.get("MINIMAX_MODEL", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.endpoint_mode = endpoint_mode
        self.thinking_budget = thinking_budget

        # Persistent HTTP client — reused across all suggest_harness_improvements()
        # calls so we skip the TLS handshake on every request. ~200-500ms
        # savings per call against api.minimax.io.
        self._http_client: httpx.Client | None = httpx.Client(
            timeout=self.timeout_seconds
        )

    # ------------------------------------------------------------------
    # Public API — mirrors AnthropicLLMWrapper
    # ------------------------------------------------------------------

    def suggest_harness_improvements(
        self,
        current_config: HarnessConfig,
        benchmark_results: Any,  # BenchmarkResult — typed loose to avoid cycle
        obs: AutobenchObservability | None = None,
        iteration: int = 0,
        revert_history: list[dict[str, Any]] | None = None,
        cross_advocate_context: list[Any] | None = None,
    ) -> LLMImprovementResult:
        """Generate improved harness config from benchmark results.

        Falls back to a rule-based improvement (wrapped in an
        ``LLMImprovementResult``) if the API call or JSON parse fails.

        When ``obs`` is supplied, emits an ``autobench.improver.reasoning.v1``
        event capturing the prompt, raw response, parsed delta, parse status,
        latency, and (if available) cost.

        ``revert_history`` (nervous-bus-sf0y) carries best-iter-keep revert
        records from the RSI loop. When non-empty, a "REVERT HISTORY" block
        is prepended to the diagnosis prompt so the improver can see "iter
        N regressed by X; reverted to iter M" and reconsider instead of
        doubling down on the same wrong-direction hypothesis.
        """
        start_time = time.monotonic()
        prompt = self._build_diagnosis_prompt(
            current_config, benchmark_results,
            revert_history=revert_history,
            cross_advocate_context=cross_advocate_context,
        )

        try:
            response_json = self._call_with_retries(prompt)
        except Exception as exc:  # noqa: BLE001 — fall back on any error
            logger.warning(
                "MiniMax call failed (%s); falling back to rule-based improver",
                exc,
            )
            result = self._rule_based_fallback(
                current_config, benchmark_results, start_time
            )
            if obs is not None:
                obs.improver_reasoning(
                    model=self.model,
                    iteration=iteration,
                    prompt=prompt,
                    raw_response="",
                    parsed_delta=asdict(result.delta),
                    parse_status="fell_back_to_rule_based",
                    fallback_reason=f"http_or_transport_error: {type(exc).__name__}: {exc}",
                    latency_ms=result.latency_ms,
                    cost_dollars=0.0,
                )
            return result

        latency_ms = (time.monotonic() - start_time) * 1000

        try:
            text, input_tokens, output_tokens = _parse_response(
                response_json, self.endpoint_mode
            )
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "MiniMax response malformed for endpoint_mode=%s (%s); falling back",
                self.endpoint_mode, exc,
            )
            result = self._rule_based_fallback(
                current_config, benchmark_results, start_time
            )
            if obs is not None:
                obs.improver_reasoning(
                    model=self.model,
                    iteration=iteration,
                    prompt=prompt,
                    raw_response=json.dumps(response_json)[:2000],
                    parsed_delta=asdict(result.delta),
                    parse_status="fell_back_to_rule_based",
                    fallback_reason=f"malformed_response: {type(exc).__name__}: {exc}",
                    latency_ms=result.latency_ms,
                    cost_dollars=0.0,
                )
            return result

        suggested_harness, delta = self._parse_llm_response(current_config, text)

        cost = self._estimate_cost(input_tokens, output_tokens, self.model)

        # Decide parse_status: was the response actually parseable?
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

        if obs is not None:
            obs.improver_reasoning(
                model=self.model,
                iteration=iteration,
                prompt=prompt,
                raw_response=text,
                parsed_delta=asdict(delta),
                parse_status=parse_status,
                fallback_reason=fallback_reason,
                latency_ms=latency_ms,
                cost_dollars=cost,
            )

        return LLMImprovementResult(
            suggested_harness=suggested_harness,
            delta=delta,
            raw_response=text,
            model_used=self.model,
            tokens_used=output_tokens,
            cost_dollars=cost,
            latency_ms=latency_ms,
        )

    def improve(
        self,
        current_config: HarnessConfig,
        benchmark_results: Any,
        obs: AutobenchObservability | None = None,
        iteration: int = 0,
        revert_history: list[dict[str, Any]] | None = None,
        cross_advocate_context: list[Any] | None = None,
    ) -> tuple[HarnessConfig, ImprovementDelta]:
        """Drop-in improver_fn for :class:`SelfImprovingHarness`.

        Optional ``obs`` and ``iteration`` propagate reasoning capture; both
        default to ``None``/0 to preserve the existing call surface.
        ``revert_history`` carries best-iter-keep revert context
        (nervous-bus-sf0y) when the RSI loop has rolled back from a
        regressed iteration.
        """
        result = self.suggest_harness_improvements(
            current_config, benchmark_results, obs=obs, iteration=iteration,
            revert_history=revert_history,
            cross_advocate_context=cross_advocate_context,
        )
        return result.suggested_harness, result.delta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_with_retries(self, prompt: str) -> dict[str, Any]:
        """POST to MiniMax with exponential backoff (3 retries, base 1s)."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_minimax(prompt)
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                # Retry on 429 and 5xx; bail immediately on 4xx auth / bad request.
                if status != 429 and not (500 <= status < 600):
                    raise
                backoff_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(
                    "MiniMax HTTP %d on attempt %d/%d; retrying after %dms",
                    status, attempt, self.max_retries, backoff_ms,
                )
                time.sleep(backoff_ms / 1000)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                backoff_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(
                    "MiniMax transport error on attempt %d/%d (%s); retrying after %dms",
                    attempt, self.max_retries, e, backoff_ms,
                )
                time.sleep(backoff_ms / 1000)

        raise RuntimeError(
            f"MiniMaxLLMWrapper exhausted {self.max_retries} retries"
        ) from last_error

    def _call_minimax(self, prompt: str) -> dict[str, Any]:
        """Single HTTP POST to MiniMax. Branches on ``endpoint_mode``.

        - ``"anthropic"`` (default): POST /anthropic/v1/messages with the
          system instruction as the top-level ``system`` field. MiniMax-M2.7
          serialises reasoning into a separate ``"thinking"`` content block
          which ``_parse_response`` drops at parse time.
        - ``"openai"``: POST /v1/chat/completions with the system instruction
          as the first message. Reasoning (if present) gets inlined into
          ``choices[0].message.content`` as ``<think>...</think>``, which
          ``_parse_llm_response`` does NOT strip — a latent JSON-collision
          risk that motivated the Anthropic-default switch.
        """
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

        if self.endpoint_mode == "anthropic":
            payload: dict[str, Any] = {
                "model": self.model,
                "system": system_instruction,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.thinking_budget is not None:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            url = f"{self.base_url}{self.ANTHROPIC_PATH}"
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            url = f"{self.base_url}{self.OPENAI_PATH}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        client = self._http_client
        if client is None:
            with httpx.Client(timeout=self.timeout_seconds) as fallback_client:
                resp = fallback_client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Release the persistent HTTP client. Safe to call multiple times."""
        client = self._http_client
        self._http_client = None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 — close is best-effort
                pass

    def __del__(self) -> None:  # pragma: no cover — GC timing dependent
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _build_diagnosis_prompt(
        self,
        current_config: HarnessConfig,
        benchmark_results: Any,
        revert_history: list[dict[str, Any]] | None = None,
        cross_advocate_context: list[Any] | None = None,
    ) -> str:
        """Evidence-driven diagnosis prompt (mirrors AnthropicLLMWrapper path).

        See llm_improver._build_diagnosis_prompt + nervous-bus-6ed for the
        rationale: the prior canned VERDICT_STRATEGIES lookup short-circuited
        actual diagnosis. The improver now sees sample generated_code per
        verdict class and diagnoses the structural failure pattern directly.

        ``revert_history`` (nervous-bus-sf0y): when the RSI loop has rolled
        back from a >variance_floor regression, this list carries
        per-revert metadata that's surfaced to the improver as a REVERT
        HISTORY block. Empty list / None → block omitted.
        """
        verdict_counts = benchmark_results.verdict_counts or {}
        total_cases = max(len(benchmark_results.case_results), 1)

        verdict_pct = {
            (k.value if isinstance(k, Verdict) else k):
                f"{(v / total_cases) * 100:.0f}%"
            for k, v in verdict_counts.items()
        }

        evidence = _build_evidence_section(benchmark_results.case_results)

        budget_lines = "\n".join(
            f"    {k}: {v}" for k, v in current_config.budget.items()
        )

        revert_block = _format_revert_history_block(revert_history)
        siblings_block = _format_siblings_block(cross_advocate_context)

        # nervous-bus-19ur: feed the FULL system_prompt (and tool_surface) to
        # the improver. Previously this was truncated to 200/100 chars, but
        # _apply_delta_to_config appends each new delta with a newline, so the
        # prompt grows unbounded across iterations. The improver was proposing
        # edits to a prompt it could not see — append-semantics are preserved
        # (schema-stable), but the visibility blind spot is fixed.
        return f"""Analyze benchmark results and generate an improved harness configuration.
{revert_block}{siblings_block}
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
        """Extract JSON from MiniMax content and turn it into delta + harness."""
        delta = ImprovementDelta()

        text = response_text.strip()
        # Strip ```json or ``` fences if the model added them despite instruction.
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1])

        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            logger.warning("Could not extract JSON from MiniMax response")
            return current_config, delta

        parsed = _tolerant_json_loads(json_match.group())
        if parsed is None:
            logger.warning("JSON parse error (incl. tolerant fallback)")
            return current_config, delta

        new_harness = HarnessConfig(
            system_prompt=current_config.system_prompt,
            rollout_protocol=current_config.rollout_protocol,
            context_manager=current_config.context_manager,
            tool_surface=current_config.tool_surface,
            verifiers=current_config.verifiers,
            budget=current_config.budget.copy(),
        )

        # 9h5y: cycle 5 iter 1 emitted `"tool_surface_changes": "keep"` as the
        # model's way of saying "no change," and we appended the literal word
        # 'keep' to the tool_surface. Treat these tokens as semantic no-ops
        # across every string-delta field, not just rollout_protocol /
        # context_manager.
        sysprompt_change = parsed.get("system_prompt_changes", "") or ""
        if sysprompt_change and not _is_no_op_value(sysprompt_change):
            delta.system_prompt_delta = sysprompt_change
            new_harness.system_prompt = (
                current_config.system_prompt + "\n" + sysprompt_change
            )

        rp = parsed.get("rollout_protocol", "keep")
        if rp and not _is_no_op_value(rp):
            try:
                new_harness.rollout_protocol = RolloutProtocol(rp)
                delta.rollout_protocol_changed = True
            except ValueError:
                pass

        cm = parsed.get("context_manager", "keep")
        if cm and not _is_no_op_value(cm):
            try:
                new_harness.context_manager = ContextManager(cm)
                delta.context_manager_changed = True
            except ValueError:
                pass

        ts = parsed.get("tool_surface_changes", "") or ""
        if ts and not _is_no_op_value(ts):
            # nervous-bus-ldd1: the harness has no machinery to materialize
            # improver-proposed tools — tool_surface is a string the worker LLM
            # reads, not a registry of callable functions. The improver was
            # writing fiction ("Add a validate_code_start tool...") that the
            # worker either tried to call (and failed), believed existed (false
            # safety), or ignored. Reject any non-no-op tool_surface_changes
            # outright; the diagnosis prompt now states this constraint
            # explicitly. Improvement comes from prompt + protocol + budget,
            # not from inventing APIs.
            logger.warning(
                "[ldd1] improver attempted to propose tool_surface change; "
                "ignored: %r",
                ts[:100],
            )

        bc = parsed.get("budget_changes", {}) or {}
        if bc and isinstance(bc, dict):
            delta.budget_delta = bc
            new_harness.budget.update(bc)

        delta.improvement_summary = parsed.get("rationale", "LLM-generated improvement")
        delta.delta_score = 0.0

        # AHE prediction contract: attach a Prediction if the model supplied one.
        try:
            from ..audit.ahe import parse_prediction_from_llm_response
            delta.prediction = parse_prediction_from_llm_response(response_text)
        except Exception:  # noqa: BLE001 — prediction is best-effort
            pass

        return new_harness, delta

    def _rule_based_fallback(
        self,
        current_config: HarnessConfig,
        benchmark_results: Any,
        start_time: float,
    ) -> LLMImprovementResult:
        """Wrap the rule-based improver in an LLMImprovementResult."""
        new_harness, delta = improve_harness(current_config, benchmark_results)
        latency_ms = (time.monotonic() - start_time) * 1000
        return LLMImprovementResult(
            suggested_harness=new_harness,
            delta=delta,
            raw_response="",
            model_used=f"{self.model} (fallback: rule_based)",
            tokens_used=0,
            cost_dollars=0.0,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _calc_backoff_ms(attempt: int, error: Exception) -> int:
        """Exponential backoff starting at 1s, with ~20% jitter.

        Honours Retry-After on httpx.HTTPStatusError when present.
        """
        base_ms = 1000 * (1 << (attempt - 1))  # 1s, 2s, 4s, ...
        jitter_ms = int(base_ms * 0.2 * random.random())
        total_ms = base_ms + jitter_ms

        resp = getattr(error, "response", None)
        if resp is not None:
            retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
            if retry_after:
                try:
                    total_ms = int(float(retry_after)) * 1000
                except (ValueError, TypeError):
                    pass

        return total_ms

    @staticmethod
    def _estimate_cost(
        input_tokens: int,
        output_tokens: int,
        model: str = "MiniMax-M2.7",
    ) -> float:
        """REMOVED — see nervous-bus-dq7l.

        Returns 0.0 unconditionally. MiniMax coding plan bills by
        requests-per-5h; in-tree $/token tables are fictions. Operators
        wanting a real $ figure must fetch live pricing themselves.
        """
        return 0.0


# ---------------------------------------------------------------------------
# Module-level response parser (mirrors worker_agent._parse_response)
# ---------------------------------------------------------------------------

def _parse_response(
    response_json: dict[str, Any], endpoint_mode: str
) -> tuple[str, int, int]:
    """Parse a MiniMax response into ``(raw_text, input_tokens, output_tokens)``.

    Branches on ``endpoint_mode``:
      - ``"openai"``: ``choices[0].message.content`` is a single string;
        usage carries ``prompt_tokens`` / ``completion_tokens``.
      - ``"anthropic"``: ``content`` is a list of blocks like
        ``[{"type": "thinking", "thinking": "..."},
          {"type": "text", "text": "...JSON..."}]``. ONLY ``"text"`` blocks
        are concatenated — thinking blocks are deliberately dropped here so
        the downstream JSON-extraction regex in ``_parse_llm_response``
        never sees stray ``{...}`` from reasoning prose. Usage carries
        ``input_tokens`` / ``output_tokens``.

    Raises:
        KeyError / IndexError / TypeError on malformed payloads — caller
        wraps these into a rule-based fallback path.
    """
    if endpoint_mode == "anthropic":
        blocks = response_json["content"]  # KeyError if missing
        text_parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        raw_text = "".join(text_parts)
        usage = response_json.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return raw_text, input_tokens, output_tokens

    # openai
    raw_text = response_json["choices"][0]["message"]["content"]
    usage = response_json.get("usage", {}) or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    return raw_text, input_tokens, output_tokens
