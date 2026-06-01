"""Tests for the generated-code capture path (nervous-bus-hvz).

Closes the single biggest autobench observability gap — previously the agent's
generated code was discarded after measuring its length. These tests verify:

* HarnessResult.metadata now carries the captured (possibly truncated) code AND
  the original length, plus the case language.
* For short code, generated_code_length == len(metadata["generated_code"]).
* For code longer than the truncation budget, the captured string is exactly
  GENERATED_CODE_TRUNCATE_LEN chars while length records the original size.
* An ``autobench.case.result.v1`` event is emitted per case on the bus
  (debug-file fallback), one per case, and each event validates against the
  new JSON schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autobench.core import HarnessConfig
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_CASE_RESULT,
    GENERATED_CODE_TRUNCATE_LEN,
)


from tests._paths import SCHEMA_DIR
SCHEMA_PATH = SCHEMA_DIR / "autobench.case.result.v1.json"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force pipe-disabled mode so emissions always land in the debug file."""
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

SHORT_CODE = "print('hello')\n"
# Build a string that clearly exceeds the 4 KiB truncation budget.
LONG_CODE = ("# pad\n" * 2000) + "print('end')\n"


def _build_evaluator(code_to_return: str, obs: AutobenchObservability) -> BenchmarkEvaluator:
    return BenchmarkEvaluator(
        generate_fn=lambda prompt, cfg: code_to_return,
        obs=obs,
    )


def _build_case(case_id: str = "probe") -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        prompt="print hello",
        expected_output="hello\n",
        language="python",
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_metadata_captures_short_code(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    ev = _build_evaluator(SHORT_CODE, obs)
    res = ev.run(HarnessConfig(), [_build_case()])

    assert len(res.case_results) == 1
    meta = res.case_results[0].metadata
    assert "generated_code" in meta
    assert isinstance(meta["generated_code"], str)
    assert meta["generated_code"] == SHORT_CODE
    assert meta["generated_code_length"] == len(SHORT_CODE)
    assert meta["generated_code_length"] == len(meta["generated_code"])
    assert meta["language"] == "python"


def test_metadata_truncates_long_code(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    ev = _build_evaluator(LONG_CODE, obs)
    res = ev.run(HarnessConfig(), [_build_case()])

    meta = res.case_results[0].metadata
    assert len(meta["generated_code"]) == GENERATED_CODE_TRUNCATE_LEN
    assert meta["generated_code_length"] == len(LONG_CODE)
    assert meta["generated_code_length"] > GENERATED_CODE_TRUNCATE_LEN
    # The captured prefix should be a real prefix of the original code.
    assert LONG_CODE.startswith(meta["generated_code"])


def test_case_result_event_emitted(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    ev = _build_evaluator(SHORT_CODE, obs)
    cases = [_build_case("c1"), _build_case("c2"), _build_case("c3")]
    ev.run(HarnessConfig(), cases)

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    assert len(events) == len(cases), f"expected {len(cases)} events, got {len(events)}"

    seen_ids = {e["data"]["case_id"] for e in events}
    assert seen_ids == {"c1", "c2", "c3"}

    for e in events:
        d = e["data"]
        assert d["language"] == "python"
        assert d["generated_code"] == SHORT_CODE
        assert d["generated_code_length"] == len(SHORT_CODE)
        # Stable session id across all events.
        assert d["session_id"] == obs.session_id


def test_case_result_event_truncated_field(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    ev = _build_evaluator(LONG_CODE, obs)
    ev.run(HarnessConfig(), [_build_case()])

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    assert events
    d = events[0]["data"]
    assert len(d["generated_code"]) == GENERATED_CODE_TRUNCATE_LEN
    assert d["generated_code_length"] == len(LONG_CODE)


def test_case_result_event_validates_against_schema(debug_file: Path) -> None:
    pytest.importorskip("jsonschema")
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    obs = AutobenchObservability(debug_file=debug_file)
    ev = _build_evaluator(SHORT_CODE, obs)
    ev.run(HarnessConfig(), [_build_case(), _build_case("c2")])

    events = _events_on(debug_file, CHANNEL_CASE_RESULT)
    assert events
    for e in events:
        # Will raise if invalid; pytest reports the path on failure.
        validator.validate(e)


def test_no_emission_when_obs_none(debug_file: Path) -> None:
    """Without an obs instance, the metadata is still captured but nothing is
    written to the debug file."""
    ev = BenchmarkEvaluator(generate_fn=lambda prompt, cfg: SHORT_CODE)
    res = ev.run(HarnessConfig(), [_build_case()])

    meta = res.case_results[0].metadata
    assert meta["generated_code"] == SHORT_CODE
    assert meta["generated_code_length"] == len(SHORT_CODE)
    assert not debug_file.exists(), "should not have written to debug file without obs"
