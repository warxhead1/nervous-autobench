"""Tests for the adversarial dual co-evolution module (nervous-bus-1rf).

Covers:
    * AdversarialGenerator prompt template includes the steered failure mode.
    * Generator HTTP payload shape (model, temperature, JSON-only system prompt).
    * Successful JSON-response parsing into an AdversarialCase.
    * Malformed response → fallback to a static template.
    * AdversarialDual.run_round aggregates verdicts, scores, and failure
      categories correctly with a mocked worker + mocked evaluator path.
    * Schema validation for both new channels.
    * Observability monkey-patch is wired (callable on instance).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tests._paths import SCHEMA_DIR, NBUS_ROOT as REPO_ROOT

from autobench.rsi.adversarial import (
    _STATIC_FALLBACK,
    AdversarialCase,
    AdversarialDual,
    AdversarialGenerator,
    AdversarialRoundResult,
)
from autobench.core import HarnessConfig, HarnessResult, Verdict
from autobench.evaluator import BenchmarkCase, BenchmarkEvaluator, BenchmarkResult
from autobench.observability import (
    CHANNEL_ADVERSARIAL_GENERATED,
    CHANNEL_ADVERSARIAL_ROUND,
    AutobenchObservability,
)




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_generator_payload(
    *,
    prompt: str = "Read n and print 1+2+...+n.",
    sample_input: str = "10\n",
    expected_output: str = "55\n",
    gotcha: str = "Off-by-one: range(n) gives 45 instead of 55.",
) -> str:
    return json.dumps({
        "prompt": prompt,
        "sample_input": sample_input,
        "expected_output": expected_output,
        "gotcha": gotcha,
    })


def _chat_response(content: str) -> dict:
    return {
        "id": "test-id",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 90, "completion_tokens": 60, "total_tokens": 150},
    }


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force zellij missing so observability falls back to a temp debug file."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return tmp_path / "debug.jsonl"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Generator construction
# ---------------------------------------------------------------------------

def test_generator_instantiates_without_api_key(monkeypatch):
    """No key → constructor must NOT raise (degrades to static fallback)."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    gen = AdversarialGenerator()
    assert gen.api_key == ""
    assert gen.model == "MiniMax-M2.7"
    assert gen.temperature == 0.7


def test_generator_rejects_zero_temperature(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    with pytest.raises(ValueError, match="temperature"):
        AdversarialGenerator(temperature=0.0)


def test_generator_uses_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key-xyz")
    gen = AdversarialGenerator()
    assert gen.api_key == "env-key-xyz"


# ---------------------------------------------------------------------------
# Prompt template: target_failure_mode actually steers the prompt
# ---------------------------------------------------------------------------

def test_target_failure_mode_appears_in_request_body(monkeypatch):
    """The steer must reach the LLM verbatim in the user message body."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    gen = AdversarialGenerator()

    captured: dict = {}

    class _StubClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=_chat_response(_valid_generator_payload()))
            return resp

    with patch("autobench.rsi.adversarial.httpx.Client", _StubClient):
        gen.generate_curveball(target_failure_mode="integer_overflow")

    user_msg = captured["json"]["messages"][1]["content"]
    assert "integer_overflow" in user_msg
    # And the failure-clause phrasing actually steers, not just mentions:
    assert "must specifically target" in user_msg or "specifically target the failure mode" in user_msg


def test_difficulty_and_domain_appear_in_prompt(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    gen = AdversarialGenerator()
    captured: dict = {}

    class _StubClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            captured["json"] = json
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=_chat_response(_valid_generator_payload()))
            return resp

    with patch("autobench.rsi.adversarial.httpx.Client", _StubClient):
        gen.generate_curveball(
            domain="shader_programming",
            target_difficulty="hard",
            target_failure_mode="off_by_one",
        )

    user_msg = captured["json"]["messages"][1]["content"]
    assert "shader_programming" in user_msg
    assert "hard" in user_msg
    assert "off_by_one" in user_msg


# ---------------------------------------------------------------------------
# HTTP payload shape
# ---------------------------------------------------------------------------

def test_request_payload_shape(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sek")
    gen = AdversarialGenerator(model="MiniMax-M2.7", temperature=0.8, max_tokens=512)
    captured: dict = {}

    class _StubClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=_chat_response(_valid_generator_payload()))
            return resp

    with patch("autobench.rsi.adversarial.httpx.Client", _StubClient):
        gen.generate_curveball()

    assert captured["url"] == "https://api.minimax.io/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sek"
    payload = captured["json"]
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["temperature"] == 0.8
    assert payload["max_tokens"] == 512
    assert len(payload["messages"]) == 2
    # System message must demand JSON-only output (the parser depends on it).
    assert "JSON" in payload["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Successful response parsing
# ---------------------------------------------------------------------------

def test_parses_valid_response_into_adversarial_case(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    gen = AdversarialGenerator()

    class _StubClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=_chat_response(
                _valid_generator_payload(
                    prompt="Read two ints and print their product.",
                    sample_input="10 20\n",
                    expected_output="200\n",
                    gotcha="Overflow on 64-bit ints.",
                )
            ))
            return resp

    with patch("autobench.rsi.adversarial.httpx.Client", _StubClient):
        case = gen.generate_curveball(target_failure_mode="integer_overflow")

    assert isinstance(case, AdversarialCase)
    assert case.case_id.startswith("adv-")
    assert "product" in case.prompt
    assert case.sample_input == "10 20\n"
    assert case.expected_output == "200\n"
    assert case.gotcha == "Overflow on 64-bit ints."
    assert case.target_failure_mode == "integer_overflow"
    assert case.generator_model == "MiniMax-M2.7"


def test_parses_markdown_fenced_response(monkeypatch):
    """Tolerant of ```json fences even though system prompt forbids them."""
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    gen = AdversarialGenerator()

    fenced = "```json\n" + _valid_generator_payload() + "\n```"

    class _StubClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=_chat_response(fenced))
            return resp

    with patch("autobench.rsi.adversarial.httpx.Client", _StubClient):
        case = gen.generate_curveball()

    assert case.generator_model == "MiniMax-M2.7"  # not the fallback
    assert case.expected_output == "55\n"


# ---------------------------------------------------------------------------
# Malformed-response fallback
# ---------------------------------------------------------------------------

def test_malformed_json_falls_back_to_static(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    gen = AdversarialGenerator()

    class _StubClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            # Junk that contains no JSON object at all.
            resp.json = MagicMock(return_value=_chat_response("just words, no json here"))
            return resp

    with patch("autobench.rsi.adversarial.httpx.Client", _STAtic := _StubClient):
        case = gen.generate_curveball(target_failure_mode="empty_input")

    assert case.generator_model == "fallback_static"
    # Matches the static template for empty_input:
    assert case.expected_output == _STATIC_FALLBACK["empty_input"]["expected_output"]


def test_http_error_falls_back_to_static(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    gen = AdversarialGenerator()

    class _BoomClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            raise RuntimeError("network down")

    with patch("autobench.rsi.adversarial.httpx.Client", _BoomClient):
        case = gen.generate_curveball(target_failure_mode="off_by_one")

    assert case.generator_model == "fallback_static"
    assert case.gotcha == _STATIC_FALLBACK["off_by_one"]["gotcha"]


def test_no_api_key_uses_static_fallback(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    gen = AdversarialGenerator()
    case = gen.generate_curveball(target_failure_mode="negative_numbers")
    assert case.generator_model == "fallback_static"
    assert case.target_failure_mode == "negative_numbers"
    assert case.expected_output  # non-empty


def test_unknown_failure_mode_falls_back_to_overflow_template(monkeypatch):
    """Unknown mode should not crash — picks the overflow default."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    gen = AdversarialGenerator()
    case = gen.generate_curveball(target_failure_mode="quantum_entanglement_bug")
    assert case.generator_model == "fallback_static"
    assert case.expected_output == _STATIC_FALLBACK["integer_overflow"]["expected_output"]


# ---------------------------------------------------------------------------
# AdversarialDual.run_round
# ---------------------------------------------------------------------------

class _FakeGenerator:
    """Deterministic non-LLM stand-in for AdversarialGenerator."""

    def __init__(self, cases: list[AdversarialCase]):
        self._cases = list(cases)
        self.calls: list[dict] = []

    def generate_curveball(
        self,
        domain="competitive_programming",
        target_difficulty="medium",
        target_failure_mode=None,
        seed_from=None,
    ):
        self.calls.append({
            "domain": domain,
            "difficulty": target_difficulty,
            "mode": target_failure_mode,
        })
        return self._cases.pop(0)


def _make_static_case(case_id: str, mode: str | None) -> AdversarialCase:
    return AdversarialCase(
        case_id=case_id,
        prompt=f"problem-{case_id}",
        sample_input="1\n",
        expected_output="1\n",
        gotcha=f"trap for {mode}",
        target_failure_mode=mode,
        generator_model="test",
    )


class _StubEvaluator:
    """Behaves like BenchmarkEvaluator for the dual's purposes.

    Returns a BenchmarkResult containing one HarnessResult whose verdict
    is dictated by the case's metadata['force_verdict'] field, defaulting
    to OK. This lets us write deterministic dual-loop assertions without
    standing up the sandbox.
    """

    def __init__(self, verdict_map: dict[str, Verdict]):
        self.generate_fn = lambda prompt, cfg: ""
        self.verdict_map = verdict_map
        self.runs: list[BenchmarkCase] = []

    def run(self, harness, cases):
        self.runs.extend(cases)
        # Call generate_fn so the dual's capture wrapper records the code.
        out_results = []
        for case in cases:
            code = self.generate_fn(case.prompt, harness)
            v = self.verdict_map.get(case.id, Verdict.OK)
            out_results.append(HarnessResult(
                p_score=1.0 if v == Verdict.OK else 0.0,
                verdict=v,
                latency_ms=10.0,
                metadata={"code": code},
            ))
        verdict_counts: dict[str, int] = {}
        for hr in out_results:
            verdict_counts[hr.verdict.value] = verdict_counts.get(hr.verdict.value, 0) + 1
        return BenchmarkResult(
            case_results=out_results,
            aggregate_score=sum(r.p_score for r in out_results) / max(1, len(out_results)),
            total_latency_ms=sum(r.latency_ms for r in out_results),
            verdict_counts=verdict_counts,
        )


def test_run_round_aggregates_verdicts_and_failures():
    cases = [
        _make_static_case("c1", "integer_overflow"),
        _make_static_case("c2", "empty_input"),
        _make_static_case("c3", "off_by_one"),
    ]
    gen = _FakeGenerator(cases)
    # c1 → WA (worker fell into the trap), c2 → OK, c3 → RE
    evaluator = _StubEvaluator({
        "c1": Verdict.WA,
        "c2": Verdict.OK,
        "c3": Verdict.RE,
    })

    worker_calls: list[str] = []

    def worker(prompt, cfg):
        worker_calls.append(prompt)
        return f"print('answer for {prompt[:5]}')"

    dual = AdversarialDual(
        generator=gen,  # type: ignore[arg-type]
        worker=worker,
        evaluator=evaluator,  # type: ignore[arg-type]
    )

    result = dual.run_round(
        n_cases=3,
        target_failure_modes=["integer_overflow", "empty_input", "off_by_one"],
    )

    assert isinstance(result, AdversarialRoundResult)
    assert result.round_id.startswith("advr-")
    assert len(result.cases) == 3
    assert result.verdicts == ["WA", "OK", "RE"]
    assert result.verdict_counts == {"WA": 1, "OK": 1, "RE": 1}
    assert result.scores == [0.0, 1.0, 0.0]
    assert result.mean_score == pytest.approx(1.0 / 3)
    # Worker was called per case:
    assert len(worker_calls) == 3
    # failure_categories counts only non-OK with a target_failure_mode:
    assert result.failure_categories == {
        "integer_overflow": 1,
        "off_by_one": 1,
    }
    # Worker code is captured:
    assert all(code.startswith("print(") for code in result.worker_codes)


def test_run_round_pads_failure_modes_with_none():
    cases = [_make_static_case(f"c{i}", None) for i in range(3)]
    gen = _FakeGenerator(cases)
    evaluator = _StubEvaluator({"c0": Verdict.OK, "c1": Verdict.OK, "c2": Verdict.OK})

    dual = AdversarialDual(
        generator=gen,  # type: ignore[arg-type]
        worker=lambda p, c: "ok",
        evaluator=evaluator,  # type: ignore[arg-type]
    )

    # Pass only one mode; rest are padded to None.
    result = dual.run_round(n_cases=3, target_failure_modes=["integer_overflow"])

    # Generator should have been called once with the steer, twice with None.
    assert gen.calls[0]["mode"] == "integer_overflow"
    assert gen.calls[1]["mode"] is None
    assert gen.calls[2]["mode"] is None
    assert result.mean_score == 1.0


def test_run_round_handles_worker_exceptions():
    """A worker that raises must not poison the round — recorded as empty code."""
    cases = [_make_static_case("c1", "integer_overflow")]
    gen = _FakeGenerator(cases)
    evaluator = _StubEvaluator({"c1": Verdict.RE})

    def boom_worker(prompt, cfg):
        raise RuntimeError("LLM API down")

    dual = AdversarialDual(
        generator=gen,  # type: ignore[arg-type]
        worker=boom_worker,
        evaluator=evaluator,  # type: ignore[arg-type]
    )
    result = dual.run_round(n_cases=1, target_failure_modes=["integer_overflow"])
    assert result.verdicts == ["RE"]
    assert result.worker_codes == [""]  # capture saw empty after the raise


def test_run_round_restores_evaluator_generate_fn():
    """Dual must not permanently mutate the evaluator's generate_fn."""
    cases = [_make_static_case("c1", None)]
    gen = _FakeGenerator(cases)
    evaluator = _StubEvaluator({"c1": Verdict.OK})
    original = evaluator.generate_fn

    dual = AdversarialDual(
        generator=gen,  # type: ignore[arg-type]
        worker=lambda p, c: "x",
        evaluator=evaluator,  # type: ignore[arg-type]
    )
    dual.run_round(n_cases=1)
    assert evaluator.generate_fn is original


def test_round_result_converts_to_benchmark_cases():
    """The output should plug straight back into a SelfImprovingHarness."""
    cases = [
        _make_static_case("c1", "integer_overflow"),
        _make_static_case("c2", "empty_input"),
    ]
    gen = _FakeGenerator(cases)
    evaluator = _StubEvaluator({"c1": Verdict.OK, "c2": Verdict.WA})
    dual = AdversarialDual(
        generator=gen,  # type: ignore[arg-type]
        worker=lambda p, c: "x",
        evaluator=evaluator,  # type: ignore[arg-type]
    )
    result = dual.run_round(n_cases=2)
    bench_cases = result.to_benchmark_cases()
    assert len(bench_cases) == 2
    assert all(isinstance(b, BenchmarkCase) for b in bench_cases)
    assert bench_cases[0].metadata["adversarial"] is True
    assert bench_cases[0].metadata["gotcha"]


# ---------------------------------------------------------------------------
# Observability emissions
# ---------------------------------------------------------------------------

def test_curveball_generated_emission(debug_file: Path, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    obs = AutobenchObservability(debug_file=debug_file)
    gen = AdversarialGenerator(obs=obs)
    case = gen.generate_curveball(target_failure_mode="integer_overflow")

    events = [e for e in _read_events(debug_file)
              if e.get("type") == CHANNEL_ADVERSARIAL_GENERATED]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["case_id"] == case.case_id
    assert data["target_failure_mode"] == "integer_overflow"
    assert data["generator_model"] == "fallback_static"
    assert data["gotcha"]
    assert len(data["prompt_preview"]) <= 200


def test_round_complete_emission(debug_file: Path):
    obs = AutobenchObservability(debug_file=debug_file)
    cases = [_make_static_case("c1", "integer_overflow")]
    gen = _FakeGenerator(cases)
    evaluator = _StubEvaluator({"c1": Verdict.WA})

    dual = AdversarialDual(
        generator=gen,  # type: ignore[arg-type]
        worker=lambda p, c: "x",
        evaluator=evaluator,  # type: ignore[arg-type]
        obs=obs,
    )
    result = dual.run_round(n_cases=1, target_failure_modes=["integer_overflow"])

    events = [e for e in _read_events(debug_file)
              if e.get("type") == CHANNEL_ADVERSARIAL_ROUND]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["round_id"] == result.round_id
    assert data["n_cases"] == 1
    assert data["verdict_counts"] == {"WA": 1}
    assert data["failure_categories"] == {"integer_overflow": 1}
    assert 0.0 <= data["mean_score"] <= 1.0


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_curveball_generated_schema_loads():
    jsonschema = pytest.importorskip("jsonschema")
    path = SCHEMA_DIR / "autobench.adversarial.curveball_generated.v1.json"
    assert path.exists()
    schema = json.loads(path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)


def test_round_complete_schema_loads():
    jsonschema = pytest.importorskip("jsonschema")
    path = SCHEMA_DIR / "autobench.adversarial.round_complete.v1.json"
    assert path.exists()
    schema = json.loads(path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)


def test_emitted_events_validate_against_schemas(debug_file: Path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    obs = AutobenchObservability(debug_file=debug_file)
    gen = AdversarialGenerator(obs=obs)
    gen.generate_curveball(target_failure_mode="integer_overflow")

    cases = [_make_static_case("c1", "integer_overflow")]
    dual = AdversarialDual(
        generator=_FakeGenerator(cases),  # type: ignore[arg-type]
        worker=lambda p, c: "x",
        evaluator=_StubEvaluator({"c1": Verdict.WA}),  # type: ignore[arg-type]
        obs=obs,
    )
    dual.run_round(n_cases=1, target_failure_modes=["integer_overflow"])

    schemas = {
        CHANNEL_ADVERSARIAL_GENERATED: jsonschema.Draft202012Validator(
            json.loads((SCHEMA_DIR / "autobench.adversarial.curveball_generated.v1.json").read_text())
        ),
        CHANNEL_ADVERSARIAL_ROUND: jsonschema.Draft202012Validator(
            json.loads((SCHEMA_DIR / "autobench.adversarial.round_complete.v1.json").read_text())
        ),
    }

    events = _read_events(debug_file)
    relevant = [e for e in events if e.get("type") in schemas]
    assert relevant, "expected at least one adversarial event in the debug file"
    for ev in relevant:
        validator = schemas[ev["type"]]
        errors = sorted(validator.iter_errors(ev), key=lambda e: list(e.path))
        assert not errors, (
            f"event failed schema {ev['type']}: "
            f"{[e.message for e in errors]}"
        )


# ---------------------------------------------------------------------------
# Sanity: observability methods exist on the class
# ---------------------------------------------------------------------------

def test_observability_has_adversarial_methods():
    obs = AutobenchObservability()
    assert callable(getattr(obs, "adversarial_curveball_generated", None))
    assert callable(getattr(obs, "adversarial_round_complete", None))
