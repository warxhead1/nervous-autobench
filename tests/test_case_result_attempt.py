"""Tests for the case.result.v1 `attempt` field (nervous-bus-x3os).

The iterative rollout protocol re-runs cases that fail until a pass is reached
or ITERATIVE_MAX_ATTEMPTS is exhausted. Each retry MUST emit its own
case.result.v1 event carrying a monotonically increasing 1-indexed attempt
counter so downstream consumers can distinguish first-attempt-OK from
retry-OK when scoring self-correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.core import HarnessConfig, RolloutProtocol
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator, ITERATIVE_MAX_ATTEMPTS
from autobench.observability import AutobenchObservability, CHANNEL_CASE_RESULT


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "autobench.case.result.v1.json"


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


def test_iterative_attempt_field_increments_until_pass(debug_file: Path) -> None:
    """Iterative protocol: fail twice, pass on the 3rd → emits 3 events
    with attempt=1, 2, 3 in order, and the final HarnessResult is the pass."""
    obs = AutobenchObservability(debug_file=debug_file)

    # generate_fn returns broken code for the first two calls, then a correct
    # program. The case expects "ok\n" so the verdict is OK only on the 3rd.
    call_count = {"n": 0}

    def flaky_generate(prompt: str, cfg: HarnessConfig) -> str:
        call_count["n"] += 1
        if call_count["n"] < 3:
            # Wrong stdout (WA) — exits 0 but doesn't match expected_output
            return "print('wrong_answer_marker')\n"
        return "print('ok')\n"

    ev = BenchmarkEvaluator(generate_fn=flaky_generate, obs=obs)
    case = BenchmarkCase(
        id="retry_probe",
        prompt="print ok",
        expected_output="ok\n",
        language="python",
    )
    harness = HarnessConfig(rollout_protocol=RolloutProtocol.ITERATIVE)

    result = ev.run(harness, [case])

    # 3 attempts emitted, in order
    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    attempts = [e["data"]["attempt"] for e in events]
    assert attempts == [1, 2, 3], f"expected [1,2,3], got {attempts}"

    # All three carry the same case_id
    assert {e["data"]["case_id"] for e in events} == {"retry_probe"}

    # Final HarnessResult is the pass (attempt 3 was OK)
    assert len(result.case_results) == 1
    assert result.case_results[0].is_pass()
    assert call_count["n"] == 3


def test_single_protocol_does_not_retry(debug_file: Path) -> None:
    """SINGLE protocol: even on failure, attempt=1 and no retry."""
    obs = AutobenchObservability(debug_file=debug_file)

    def broken(prompt: str, cfg: HarnessConfig) -> str:
        return "print('wrong_answer_marker')\n"

    ev = BenchmarkEvaluator(generate_fn=broken, obs=obs)
    case = BenchmarkCase(
        id="single_probe",
        prompt="x",
        expected_output="ok\n",
        language="python",
    )
    ev.run(HarnessConfig(rollout_protocol=RolloutProtocol.SINGLE), [case])

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    assert len(events) == 1, f"expected exactly 1 event, got {len(events)}"
    assert events[0]["data"]["attempt"] == 1


def test_iterative_stops_at_max_attempts(debug_file: Path) -> None:
    """Iterative protocol: always-failing case caps at ITERATIVE_MAX_ATTEMPTS."""
    obs = AutobenchObservability(debug_file=debug_file)

    def always_broken(prompt: str, cfg: HarnessConfig) -> str:
        return "print('wrong_answer_marker')\n"

    ev = BenchmarkEvaluator(generate_fn=always_broken, obs=obs)
    case = BenchmarkCase(
        id="stubborn",
        prompt="x",
        expected_output="ok\n",
        language="python",
    )
    ev.run(HarnessConfig(rollout_protocol=RolloutProtocol.ITERATIVE), [case])

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    attempts = [e["data"]["attempt"] for e in events]
    assert attempts == list(range(1, ITERATIVE_MAX_ATTEMPTS + 1))


def test_attempt_field_validates_against_schema(debug_file: Path) -> None:
    """Events with the new `attempt` field pass v1 schema validation."""
    pytest.importorskip("jsonschema")
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    obs = AutobenchObservability(debug_file=debug_file)

    def short_code(prompt: str, cfg: HarnessConfig) -> str:
        return "print('ok')\n"

    ev = BenchmarkEvaluator(generate_fn=short_code, obs=obs)
    case = BenchmarkCase(
        id="schema_probe",
        prompt="x",
        expected_output="ok\n",
        language="python",
    )
    ev.run(HarnessConfig(rollout_protocol=RolloutProtocol.ITERATIVE), [case])

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    assert events
    for e in events:
        validator.validate(e)
        assert e["data"]["attempt"] >= 1
