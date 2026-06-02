"""Tests that AutobenchObservability is wired through the live RSI loop.

Verifies that:
    * Running `SelfImprovingHarness.improve()` with `obs=...` emits at least
      one event on each of the four autobench channels (phase, iteration,
      sandbox, improver) with the same session_id.
    * Running without `obs` (`obs=None`) is fully backwards compatible — no
      exceptions, no emissions.
    * `sandbox_dispatch` events carry the `case_id` passed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.core import HarnessConfig
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_IMPROVER,
    CHANNEL_ITERATION,
    CHANNEL_PHASE,
    CHANNEL_SANDBOX,
)
from autobench.rsi.loop import SelfImprovingHarness


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force fallback-to-debug-file path by stubbing the zellij-pipe call."""
    from autobench import observability as obs_mod
    monkeypatch.setattr(
        obs_mod.AutobenchObservability,
        "_try_zellij_pipe",
        lambda self, channel, payload: False,
    )
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            id="probe-1",
            prompt="print(1)",
            language="python",
            expected_output="1\n",
        ),
        BenchmarkCase(
            id="probe-2",
            prompt="print(2)",
            language="python",
            expected_output="2\n",
        ),
    ]


def _generate(prompt: str, _cfg: HarnessConfig) -> str:
    # Trivial generator: literal echo of the prompt as a python program.
    return prompt


def test_wiring_emits_all_four_channels(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    session_id = obs.session_id

    evaluator = BenchmarkEvaluator(generate_fn=_generate, obs=obs)
    harness = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=evaluator,
        max_iterations=2,
        default_improver="rule_based",
        obs=obs,
    )
    harness.improve(_make_cases())

    events = _read_events(debug_file)
    assert events, "expected at least one event written to debug file"

    by_channel: dict[str, list[dict]] = {}
    for event in events:
        by_channel.setdefault(event["type"], []).append(event)

    for ch in (CHANNEL_PHASE, CHANNEL_ITERATION, CHANNEL_SANDBOX, CHANNEL_IMPROVER):
        assert by_channel.get(ch), f"no events emitted on channel {ch}"

    # All session_ids should match
    for event in events:
        assert event["data"]["session_id"] == session_id


def test_wiring_obs_none_is_noop(debug_file: Path) -> None:
    """obs=None must run cleanly with no emissions and no exceptions."""
    evaluator = BenchmarkEvaluator(generate_fn=_generate)  # no obs
    harness = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=evaluator,
        max_iterations=1,
        default_improver="rule_based",
        # obs=None (default)
    )
    final_harness, result, history = harness.improve(_make_cases())
    assert final_harness is not None
    assert result is not None
    assert len(history) >= 1

    # No emissions: debug file should not exist (or be empty).
    assert not debug_file.exists() or debug_file.read_text() == ""


def test_sandbox_dispatch_carries_case_id(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)

    evaluator = BenchmarkEvaluator(generate_fn=_generate, obs=obs)
    harness = SelfImprovingHarness(
        current_harness=HarnessConfig(),
        evaluator=evaluator,
        max_iterations=1,
        default_improver="rule_based",
        obs=obs,
    )
    harness.improve(_make_cases())

    events = _read_events(debug_file)
    dispatches = [
        e for e in events
        if e["type"] == CHANNEL_SANDBOX and e["data"].get("status") == "dispatch"
    ]
    assert dispatches, "expected at least one sandbox_dispatch event"
    case_ids = {e["data"]["case_id"] for e in dispatches}
    assert "probe-1" in case_ids
    assert "probe-2" in case_ids
