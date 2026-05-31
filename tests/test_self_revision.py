"""Tests for RolloutProtocol.SELF_REVISION (nervous-bus-dils).

Self-revision is a Reflexion-style feedback loop within a single case:
the worker's first attempt is shown its own stderr / observed vs expected
stdout, and gets exactly one revision pass before the case is scored.

Distinct from ITERATIVE (x3os), which re-runs the SAME prompt up to
ITERATIVE_MAX_ATTEMPTS times. SELF_REVISION caps at 2 attempts and the
second attempt receives an augmented prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.core import HarnessConfig, RolloutProtocol
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator
from autobench.observability import AutobenchObservability, CHANNEL_CASE_RESULT


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    return tmp_path / "debug.jsonl"


def _events_on(path: Path, channel: str) -> list[dict]:
    if not path.exists():
        return []
    return [
        e
        for e in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        if e.get("type") == channel
    ]


def test_self_revision_fails_then_succeeds_with_feedback(debug_file: Path) -> None:
    """SELF_REVISION: first call fails, second call (which sees execution
    feedback in the prompt) succeeds — final HarnessResult is OK at attempt=2."""
    obs = AutobenchObservability(debug_file=debug_file)

    seen_prompts: list[str] = []

    def revising(prompt: str, cfg: HarnessConfig) -> str:
        seen_prompts.append(prompt)
        # First call: emit wrong stdout (WA verdict). Second call: emit "ok".
        # The second prompt MUST contain the revision_context marker so we
        # know the harness actually fed feedback back to generate_fn.
        if "Prior attempt feedback" in prompt:
            return "print('ok')\n"
        return "print('wrong_answer_marker')\n"

    ev = BenchmarkEvaluator(generate_fn=revising, obs=obs)
    case = BenchmarkCase(
        id="revision_probe",
        prompt="print ok",
        expected_output="ok\n",
        language="python",
    )
    harness = HarnessConfig(rollout_protocol=RolloutProtocol.SELF_REVISION)

    result = ev.run(harness, [case])

    # Two attempts emitted (1 = WA, 2 = OK)
    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    attempts = [e["data"]["attempt"] for e in events]
    assert attempts == [1, 2], f"expected [1,2], got {attempts}"

    # Final result passes
    assert len(result.case_results) == 1
    assert result.case_results[0].is_pass()

    # Worker was called twice, and the second prompt carried prior-attempt feedback
    assert len(seen_prompts) == 2
    assert "Prior attempt feedback" not in seen_prompts[0]
    assert "Prior attempt feedback" in seen_prompts[1]
    # Feedback string carries verdict + observed-vs-expected
    assert "WA" in seen_prompts[1]
    assert "wrong_answer_marker" in seen_prompts[1]


def test_self_revision_first_pass_skips_revision(debug_file: Path) -> None:
    """SELF_REVISION: when the first attempt is OK, no revision is triggered —
    exactly one event with attempt=1, generate_fn called once."""
    obs = AutobenchObservability(debug_file=debug_file)

    call_count = {"n": 0}

    def first_pass(prompt: str, cfg: HarnessConfig) -> str:
        call_count["n"] += 1
        return "print('ok')\n"

    ev = BenchmarkEvaluator(generate_fn=first_pass, obs=obs)
    case = BenchmarkCase(
        id="happy_path",
        prompt="print ok",
        expected_output="ok\n",
        language="python",
    )
    ev.run(HarnessConfig(rollout_protocol=RolloutProtocol.SELF_REVISION), [case])

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    assert len(events) == 1
    assert events[0]["data"]["attempt"] == 1
    assert call_count["n"] == 1


def test_self_revision_caps_at_two_attempts(debug_file: Path) -> None:
    """SELF_REVISION: even if the revision pass also fails, no third attempt —
    cap at 2 (distinguishes us from ITERATIVE which allows 3)."""
    obs = AutobenchObservability(debug_file=debug_file)

    def always_wrong(prompt: str, cfg: HarnessConfig) -> str:
        return "print('still_wrong')\n"

    ev = BenchmarkEvaluator(generate_fn=always_wrong, obs=obs)
    case = BenchmarkCase(
        id="stubborn",
        prompt="x",
        expected_output="ok\n",
        language="python",
    )
    ev.run(HarnessConfig(rollout_protocol=RolloutProtocol.SELF_REVISION), [case])

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    attempts = [e["data"]["attempt"] for e in events]
    assert attempts == [1, 2], f"SELF_REVISION must stop at 2, got {attempts}"


def test_iterative_behavior_unchanged_by_self_revision_wiring(debug_file: Path) -> None:
    """Regression: SELF_REVISION wiring must NOT bleed into ITERATIVE — the
    iterative protocol still does up to 3 attempts with the SAME prompt
    (no Prior-attempt-feedback marker injected)."""
    obs = AutobenchObservability(debug_file=debug_file)

    seen_prompts: list[str] = []

    def flaky(prompt: str, cfg: HarnessConfig) -> str:
        seen_prompts.append(prompt)
        return "print('wrong_answer_marker')\n"

    ev = BenchmarkEvaluator(generate_fn=flaky, obs=obs)
    case = BenchmarkCase(
        id="iter_probe",
        prompt="iterative prompt body",
        expected_output="ok\n",
        language="python",
    )
    ev.run(HarnessConfig(rollout_protocol=RolloutProtocol.ITERATIVE), [case])

    # ITERATIVE_MAX_ATTEMPTS = 3 retries, none of which should have feedback injected
    assert len(seen_prompts) == 3
    for p in seen_prompts:
        assert "Prior attempt feedback" not in p
        assert p == "iterative prompt body"
