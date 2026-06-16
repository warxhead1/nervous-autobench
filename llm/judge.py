"""Default MiniMax-backed judge factory for the JudgingPool.

When ``BenchmarkEvaluator`` is constructed with ``judges_per_case > 1`` but no
explicit ``judge_factory`` is passed, run_first.py and friends have historically
fallen back to the legacy single-verdict path — meaning ``judge.pool.verdict.v1``
and ``judge.disagreement.v1`` events were 0 across every cycle.

This module wires a sensible default factory so judges fire as soon as
``MINIMAX_API_KEY`` is present and ``judges_per_case > 1``. Tests still pass
explicit factories, so the default does not affect them.

Factory contract per ``Evaluator._score_with_pool``:

    factory(prompt: str, context: dict[str, Any]) -> dict[str, Any]
        keys: judge_id (filled by anon wrapper), verdict (Verdict-string),
              p_score (float 0..1), p_cost (float 0..1), p_time (float 0..1),
              reasoning (str)

The factory is intentionally cheap (max_tokens=256, temperature=0.0) and
defensive — every failure mode degrades to "ratify the sandbox verdict"
which is a no-op signal for the dissent calculator.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

_VERDICTS = ("OK", "WA", "RE", "CE", "TLE", "MLE")
_DEFAULT_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
_DEFAULT_TIMEOUT = 45.0
_DEFAULT_MAX_TOKENS = 256

_SYSTEM_PROMPT = """You are an anonymous code judge in a 5-judge ensemble. You score one
worker solution at a time. Respond with a SINGLE JSON object — no prose,
no markdown fences — matching exactly this schema:

{"verdict": "OK"|"WA"|"RE"|"CE"|"TLE"|"MLE",
 "p_score": <float in [0,1]>,
 "p_cost":  <float in [0,1]>,
 "p_time":  <float in [0,1]>,
 "reasoning": "<one short sentence>"}

Verdict semantics:
  OK  — output matches expected and code looks correct
  WA  — runs but wrong answer
  RE  — runtime error / exception during execution
  CE  — compile/syntax error or won't start
  TLE — would time out on intended inputs
  MLE — would exceed memory on intended inputs

p_score is your independent confidence the solution is correct (1.0 = certain
correct, 0.0 = certain wrong). p_cost and p_time are 0.5 unless you have a
specific reason to score them differently — they exist for future cost/perf
judges and you should not invent values for them."""


def _judge_user_prompt(prompt: str, context: dict[str, Any]) -> str:
    """Build the user-facing judge prompt from evaluator-supplied pieces."""
    pieces = [prompt.rstrip()]
    expected = (context.get("expected_output") or "").strip()
    stdout = (context.get("worker_output") or "").strip()
    stderr = (context.get("worker_stderr") or "").strip()
    sb_verdict = context.get("sandbox_verdict") or "?"
    sb_score = context.get("sandbox_p_score")
    if expected:
        pieces.append(f"\n--- Expected output ---\n{expected[:1200]}")
    if stdout:
        pieces.append(f"\n--- Worker stdout ---\n{stdout[:800]}")
    if stderr:
        pieces.append(f"\n--- Worker stderr ---\n{stderr[:800]}")
    pieces.append(
        f"\n--- Sandbox observation ---\nverdict={sb_verdict}"
        + (f" p_score={sb_score:.3f}" if isinstance(sb_score, (int, float)) else "")
    )
    pieces.append("\nRespond with the JSON object only.")
    return "\n".join(pieces)


def _parse_judge_response(raw: str, sandbox_fallback: dict[str, Any]) -> dict[str, Any]:
    """Extract the first JSON object from ``raw``; fall back to ratifying sandbox."""
    if not raw:
        return sandbox_fallback
    # Defang markdown fences if the model ignored the directive.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    # First {...} block — works even when the model prepends prose despite
    # the system prompt.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return sandbox_fallback
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return sandbox_fallback
    if not isinstance(parsed, dict):
        return sandbox_fallback

    verdict = str(parsed.get("verdict", "")).upper()
    if verdict not in _VERDICTS:
        verdict = sandbox_fallback.get("verdict", "OK")

    def _clamp(key: str, default: float) -> float:
        try:
            v = float(parsed.get(key, default))
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, v))

    return {
        "verdict": verdict,
        "p_score": _clamp("p_score", float(sandbox_fallback.get("p_score", 0.5))),
        "p_cost": _clamp("p_cost", 0.5),
        "p_time": _clamp("p_time", 0.5),
        "reasoning": str(parsed.get("reasoning", ""))[:240],
    }


def make_minimax_judge_factory(
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    http_client_factory: Callable[..., Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Return a factory the JudgingPool can fan-out to.

    ``http_client_factory`` exists for tests — production calls use httpx.
    Returns a factory callable; the actual factory captures these settings
    in its closure so the JudgingPool sees a simple ``(prompt, context) -> dict``.
    """
    resolved_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
    if not resolved_key:
        raise RuntimeError("make_minimax_judge_factory: MINIMAX_API_KEY not set")

    def _call_minimax(client, prompt: str, context: dict[str, Any]) -> tuple[str, str]:
        """Make one MiniMax API call; returns (raw_content, error_reason)."""
        resp = client.post(
            "https://api.minimax.io/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _judge_user_prompt(prompt, context)},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"], ""

    def _factory(prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        # Build a sandbox-ratifying fallback first so any failure below still
        # produces a valid JudgingPool vote (just one that agrees with the
        # sandbox — dissent will collapse to 0 in that worst case).
        sb_verdict = str(context.get("sandbox_verdict", "OK")).upper()
        if sb_verdict not in _VERDICTS:
            sb_verdict = "OK"
        sb_score = context.get("sandbox_p_score")
        sb_fallback = {
            "verdict": sb_verdict,
            "p_score": float(sb_score) if isinstance(sb_score, (int, float)) else 0.5,
            "p_cost": 0.5,
            "p_time": 0.5,
            "reasoning": "sandbox-ratified (judge unavailable)",
        }

        # Retry with exponential backoff.  Three attempts: 45s → 90s → 180s.
        # Any surviving response is enough for a valid vote; only fall back
        # to sandbox ratification when all retries are exhausted.
        sb_verdict = str(context.get("sandbox_verdict", "OK")).upper()
        if sb_verdict not in _VERDICTS:
            sb_verdict = "OK"
        sb_score = context.get("sandbox_p_score")
        sb_fallback = {
            "verdict": sb_verdict,
            "p_score": float(sb_score) if isinstance(sb_score, (int, float)) else 0.5,
            "p_cost": 0.5,
            "p_time": 0.5,
            "reasoning": "sandbox-ratified (judge unavailable)",
        }

        raw = ""
        last_error = ""
        for attempt in range(3):
            try:
                if http_client_factory is not None:
                    client = http_client_factory(timeout=timeout * (2**attempt))
                else:
                    import httpx  # noqa: WPS433 — lazy import keeps tests cheap
                    client = httpx.Client(timeout=timeout * (2**attempt))
                with client:
                    raw, last_error = _call_minimax(client, prompt, context)
                    break
            except Exception as exc:  # noqa: BLE001 — pool must never raise
                last_error = str(exc)
                logger.warning(
                    "default_judge_factory: MiniMax attempt %d/3 failed: %s",
                    attempt + 1, exc,
                )
                if attempt < 2:
                    import time
                    time.sleep(1.5 ** attempt)

        if not raw:
            logger.warning("default_judge_factory: all 3 retries exhausted, using sandbox fallback (last error: %s)", last_error)
            return sb_fallback

        return _parse_judge_response(raw, sb_fallback)

    return _factory
