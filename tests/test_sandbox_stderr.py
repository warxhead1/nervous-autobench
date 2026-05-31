"""Tests for the sandbox-stderr observability channel (bead nervous-bus-bns).

Verifies that:

* ``AutobenchObservability.sandbox_stderr`` writes a well-formed
  CloudEvents-lite envelope with ``type == "autobench.sandbox.stderr.v1"``.
* The emitter truncates an over-length stderr defensively to 200 chars.
* The wiring in ``BenchmarkEvaluator._run_case`` emits exactly one
  ``sandbox.stderr.v1`` event when the verdict is in {CE, RE, TLE, MLE},
  and zero events when the verdict is OK or WA.
* Emitted events validate against ``schemas/autobench.sandbox.stderr.v1.json``.

The integration tests fake the sandbox executor so we can drive a synthetic
CE verdict with a ``<think>``-prefix SyntaxError as the stderr text — the
exact pattern that motivated this bead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from autobench.core import HarnessConfig, Verdict
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_SANDBOX_STDERR,
    SANDBOX_STDERR_EXCERPT_LEN,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "autobench.sandbox.stderr.v1.json"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force the pipe-disabled path so every emission lands in the debug file."""
    monkeypatch.setenv("AUTOBENCH_OBS_DISABLE_PIPE", "1")
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stderr_events(path: Path) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == CHANNEL_SANDBOX_STDERR]


@dataclass
class _FakeExecResult:
    """Minimal stand-in for SandboxedExecutor.execute() return value."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    latency_ms: float = 1.0
    memory_mb: float = 1.0


class _FakeExecutor:
    """Executor stub that returns a pre-baked result for every call."""
    def __init__(self, result: _FakeExecResult) -> None:
        self._result = result
        self.obs: AutobenchObservability | None = None

    def execute(self, **_kwargs) -> _FakeExecResult:  # noqa: D401
        return self._result


def _build_evaluator(
    debug_file: Path,
    exec_result: _FakeExecResult,
    code: str = "print('x')\n",
) -> tuple[BenchmarkEvaluator, AutobenchObservability]:
    obs = AutobenchObservability(debug_file=debug_file)
    ev = BenchmarkEvaluator(
        generate_fn=lambda prompt, cfg: code,
        obs=obs,
    )
    ev.executor = _FakeExecutor(exec_result)  # type: ignore[assignment]
    return ev, obs


def _case(case_id: str = "probe", expected: str = "hello\n") -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        prompt="say hello",
        expected_output=expected,
        language="python",
    )


# --------------------------------------------------------------------------- #
# Unit: emitter behaviour
# --------------------------------------------------------------------------- #

def test_emitter_truncates_long_stderr(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    big = "X" * (SANDBOX_STDERR_EXCERPT_LEN + 500)
    obs.sandbox_stderr(  # type: ignore[attr-defined]
        case_id="c1",
        iteration=3,
        verdict="CE",
        stderr_excerpt=big,
        exit_code=1,
        language="python",
    )

    events = _stderr_events(debug_file)
    assert len(events) == 1
    d = events[0]["data"]
    assert len(d["stderr_excerpt"]) == SANDBOX_STDERR_EXCERPT_LEN
    assert d["stderr_excerpt"] == "X" * SANDBOX_STDERR_EXCERPT_LEN
    assert d["case_id"] == "c1"
    assert d["iteration"] == 3
    assert d["verdict"] == "CE"
    assert d["exit_code"] == 1
    assert d["language"] == "python"
    assert d["session_id"] == obs.session_id


def test_emitter_handles_short_stderr_unchanged(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    msg = "boom: short message"
    obs.sandbox_stderr(  # type: ignore[attr-defined]
        case_id="c2",
        iteration=0,
        verdict="RE",
        stderr_excerpt=msg,
        exit_code=2,
        language="python",
    )
    events = _stderr_events(debug_file)
    assert len(events) == 1
    assert events[0]["data"]["stderr_excerpt"] == msg


def test_emitter_accepts_null_exit_code(debug_file: Path) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.sandbox_stderr(  # type: ignore[attr-defined]
        case_id="c3",
        iteration=0,
        verdict="RE",
        stderr_excerpt="oops",
        exit_code=None,
        language="glsl",
    )
    events = _stderr_events(debug_file)
    assert len(events) == 1
    assert events[0]["data"]["exit_code"] is None


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

def test_event_validates_against_schema(debug_file: Path) -> None:
    pytest.importorskip("jsonschema")
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    obs = AutobenchObservability(debug_file=debug_file)
    for verdict in ("CE", "RE", "TLE", "MLE"):
        obs.sandbox_stderr(  # type: ignore[attr-defined]
            case_id=f"c-{verdict}",
            iteration=1,
            verdict=verdict,
            stderr_excerpt=f"{verdict} stderr line",
            exit_code=1,
            language="python",
        )

    events = _stderr_events(debug_file)
    assert len(events) == 4
    for e in events:
        validator.validate(e)


# --------------------------------------------------------------------------- #
# Integration: wiring through BenchmarkEvaluator._run_case
# --------------------------------------------------------------------------- #

# A '<think>'-prefix Python source — the exact pattern that motivated this
# bead. Running it under Python yields a SyntaxError on line 1 because the
# stray '<' token isn't a valid statement.
THINK_LEAK_CODE = "<think>\nfoo\n</think>\nprint('hello')\n"
THINK_LEAK_STDERR = (
    'File "/tmp/sol.py", line 1\n'
    '    <think>\n'
    '    ^\n'
    'SyntaxError: invalid syntax\n'
)


def test_integration_ce_emits_one_stderr_event(debug_file: Path) -> None:
    """A CE verdict from the sandbox produces exactly one sandbox.stderr event
    whose excerpt matches the captured stderr from the fake executor."""
    exec_result = _FakeExecResult(
        stdout="",
        stderr=THINK_LEAK_STDERR,
        exit_code=1,
        latency_ms=12.5,
    )
    ev, _obs = _build_evaluator(debug_file, exec_result, code=THINK_LEAK_CODE)

    # Force the verdict path: by stubbing emit_verdict we can deterministically
    # exercise the CE branch regardless of how the real sandbox would route
    # this stderr text.
    ev.emit_verdict = lambda **_kwargs: Verdict.CE  # type: ignore[assignment]

    ev.run(HarnessConfig(), [_case("think-leak")])

    events = _stderr_events(debug_file)
    assert len(events) == 1, f"expected 1 sandbox.stderr event, got {len(events)}"
    d = events[0]["data"]
    assert d["case_id"] == "think-leak"
    assert d["verdict"] == "CE"
    assert d["exit_code"] == 1
    assert d["language"] == "python"
    # The first 200 chars of the stderr text should be carried verbatim
    # because the source is shorter than 200 chars.
    assert d["stderr_excerpt"] == THINK_LEAK_STDERR
    # And the same SyntaxError signature would be obvious to a consumer:
    assert "SyntaxError" in d["stderr_excerpt"]
    assert "<think>" in d["stderr_excerpt"]


@pytest.mark.parametrize("verdict_value", ["CE", "RE", "TLE", "MLE"])
def test_integration_each_error_class_emits_event(
    debug_file: Path, verdict_value: str
) -> None:
    exec_result = _FakeExecResult(
        stdout="",
        stderr=f"{verdict_value}-stderr-text",
        exit_code=1,
        latency_ms=1.0,
    )
    ev, _obs = _build_evaluator(debug_file, exec_result)
    ev.emit_verdict = lambda **_kwargs: Verdict(verdict_value)  # type: ignore[assignment]

    ev.run(HarnessConfig(), [_case(f"case-{verdict_value}")])

    events = _stderr_events(debug_file)
    assert len(events) == 1
    assert events[0]["data"]["verdict"] == verdict_value
    assert events[0]["data"]["stderr_excerpt"] == f"{verdict_value}-stderr-text"


@pytest.mark.parametrize("verdict_value", ["OK", "WA"])
def test_integration_ok_and_wa_skip_stderr_event(
    debug_file: Path, verdict_value: str
) -> None:
    """OK is the success case (no error signal). WA is wrong-answer, not an
    error class — its stderr is not actionable diagnostic data. Neither
    verdict should produce a sandbox.stderr event."""
    exec_result = _FakeExecResult(
        stdout="hello\n",
        stderr="should-not-be-emitted",
        exit_code=0,
        latency_ms=1.0,
    )
    ev, _obs = _build_evaluator(debug_file, exec_result)
    ev.emit_verdict = lambda **_kwargs: Verdict(verdict_value)  # type: ignore[assignment]

    ev.run(HarnessConfig(), [_case()])

    events = _stderr_events(debug_file)
    assert events == [], f"{verdict_value} should not emit; got {events!r}"
