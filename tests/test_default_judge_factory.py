"""Tests for default_judge_factory + evaluator wiring."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from autobench.llm.judge import (
    _parse_judge_response,
    make_minimax_judge_factory,
)
from autobench.evaluator import BenchmarkEvaluator


class ParseJudgeResponseTests(unittest.TestCase):
    def _fallback(self) -> dict:
        return {"verdict": "OK", "p_score": 0.5, "p_cost": 0.5, "p_time": 0.5,
                "reasoning": "sandbox"}

    def test_well_formed_json(self) -> None:
        raw = '{"verdict": "WA", "p_score": 0.2, "p_cost": 0.5, "p_time": 0.5, "reasoning": "wrong"}'
        out = _parse_judge_response(raw, self._fallback())
        self.assertEqual(out["verdict"], "WA")
        self.assertAlmostEqual(out["p_score"], 0.2)
        self.assertEqual(out["reasoning"], "wrong")

    def test_markdown_fenced_json(self) -> None:
        raw = '```json\n{"verdict": "OK", "p_score": 1.0}\n```'
        out = _parse_judge_response(raw, self._fallback())
        self.assertEqual(out["verdict"], "OK")
        self.assertAlmostEqual(out["p_score"], 1.0)

    def test_prose_then_json(self) -> None:
        raw = 'Here is my answer:\n{"verdict": "TLE", "p_score": 0.0}\nThanks.'
        out = _parse_judge_response(raw, self._fallback())
        self.assertEqual(out["verdict"], "TLE")
        self.assertAlmostEqual(out["p_score"], 0.0)

    def test_unknown_verdict_falls_back(self) -> None:
        raw = '{"verdict": "FOO", "p_score": 0.5}'
        out = _parse_judge_response(raw, self._fallback())
        self.assertEqual(out["verdict"], "OK")  # sandbox fallback

    def test_no_json_falls_back(self) -> None:
        out = _parse_judge_response("not json at all", self._fallback())
        self.assertEqual(out, self._fallback())

    def test_empty_raw_falls_back(self) -> None:
        out = _parse_judge_response("", self._fallback())
        self.assertEqual(out, self._fallback())

    def test_score_clamping(self) -> None:
        raw = '{"verdict": "OK", "p_score": 2.5, "p_cost": -0.3}'
        out = _parse_judge_response(raw, self._fallback())
        self.assertEqual(out["p_score"], 1.0)
        self.assertEqual(out["p_cost"], 0.0)


class FactoryCallTests(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                make_minimax_judge_factory()

    def test_factory_uses_http_client(self) -> None:
        # Stand-in client that returns a canned MiniMax response.
        class FakeResp:
            def raise_for_status(self) -> None: pass
            def json(self) -> dict:
                return {"choices": [{"message": {"content":
                    '{"verdict": "WA", "p_score": 0.1, "reasoning": "diff"}'}}]}

        class FakeClient:
            def __init__(self, **kw) -> None: self.posts: list = []
            def __enter__(self) -> "FakeClient": return self
            def __exit__(self, *a) -> None: pass
            def post(self, url, **kw):
                self.posts.append((url, kw))
                return FakeResp()

        client = FakeClient()
        factory = make_minimax_judge_factory(
            api_key="sk-fake", http_client_factory=lambda **kw: client,
        )
        out = factory("prompt", {"sandbox_verdict": "OK", "sandbox_p_score": 0.7})
        self.assertEqual(out["verdict"], "WA")
        self.assertAlmostEqual(out["p_score"], 0.1)
        # confirm we hit MiniMax
        self.assertEqual(len(client.posts), 1)
        self.assertIn("/v1/chat/completions", client.posts[0][0])

    def test_factory_falls_back_on_http_error(self) -> None:
        def bad_client(**kw):
            raise RuntimeError("connection refused")
        factory = make_minimax_judge_factory(
            api_key="sk-fake", http_client_factory=bad_client,
        )
        out = factory("p", {"sandbox_verdict": "RE", "sandbox_p_score": 0.0})
        self.assertEqual(out["verdict"], "RE")
        self.assertEqual(out["p_score"], 0.0)
        self.assertIn("unavailable", out["reasoning"])


class EvaluatorWiringTests(unittest.TestCase):
    """Confirm the default factory is installed at the right gate."""

    def test_default_fires_when_key_present_and_judges_gt_1(self) -> None:
        with mock.patch.dict(os.environ,
                             {"MINIMAX_API_KEY": "sk-fake",
                              "AUTOBENCH_JUDGES_PER_CASE": "5"},
                             clear=False):
            os.environ.pop("AUTOBENCH_DISABLE_DEFAULT_JUDGE", None)
            ev = BenchmarkEvaluator()
            self.assertIsNotNone(ev.judge_factory)
            self.assertEqual(ev.judges_per_case, 5)

    def test_default_silent_when_no_key(self) -> None:
        with mock.patch.dict(os.environ, {"AUTOBENCH_JUDGES_PER_CASE": "5"},
                             clear=True):
            ev = BenchmarkEvaluator()
            self.assertIsNone(ev.judge_factory)

    def test_default_silent_when_judges_eq_1(self) -> None:
        with mock.patch.dict(os.environ,
                             {"MINIMAX_API_KEY": "sk-fake",
                              "AUTOBENCH_JUDGES_PER_CASE": "1"},
                             clear=False):
            ev = BenchmarkEvaluator()
            self.assertIsNone(ev.judge_factory)

    def test_disable_env_opts_out(self) -> None:
        with mock.patch.dict(os.environ,
                             {"MINIMAX_API_KEY": "sk-fake",
                              "AUTOBENCH_JUDGES_PER_CASE": "5",
                              "AUTOBENCH_DISABLE_DEFAULT_JUDGE": "1"},
                             clear=False):
            ev = BenchmarkEvaluator()
            self.assertIsNone(ev.judge_factory)

    def test_explicit_factory_not_overridden(self) -> None:
        sentinel = lambda prompt, context: {"verdict": "OK", "p_score": 1.0}
        with mock.patch.dict(os.environ,
                             {"MINIMAX_API_KEY": "sk-fake",
                              "AUTOBENCH_JUDGES_PER_CASE": "5"},
                             clear=False):
            ev = BenchmarkEvaluator(judge_factory=sentinel)
            self.assertIs(ev.judge_factory, sentinel)


if __name__ == "__main__":
    unittest.main()
