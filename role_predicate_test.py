"""Tests for session_state and role_predicate modules."""

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autobench.session_state import (
    SessionState,
    finish_session,
    generate_ulid,
    is_session_complete,
    is_valid_ulid,
    parse_rfc3339,
    rfc3339_now,
    start_session,
)
from autobench.core import HarnessResult, Verdict
from autobench.role_predicate import (
    ActivationPredicate,
    RoleSpecActivationBuilder,
    evaluate_predicate,
    build_predicate_from_autobench_result,
    build_verdict_routing_predicates,
)


class TestUlid(unittest.TestCase):
    def test_is_valid_ulid_valid(self):
        valid_ulid = generate_ulid()
        self.assertTrue(is_valid_ulid(valid_ulid))

    def test_is_valid_ulid_invalid(self):
        self.assertFalse(is_valid_ulid(""))
        self.assertFalse(is_valid_ulid("not-a-ulid"))
        self.assertFalse(is_valid_ulid("123"))
        # "I", "L", "O", "U" are excluded from Crockford base32
        self.assertFalse(is_valid_ulid("I" * 26))

    def test_generate_ulid_length(self):
        ulid = generate_ulid()
        self.assertEqual(len(ulid), 26)  # Crockford base32 encoded


class TestRfc3339(unittest.TestCase):
    def test_rfc3339_now_format(self):
        now = rfc3339_now()
        # Should be parseable as ISO format
        parsed = parse_rfc3339(now)
        self.assertIsInstance(parsed, datetime)
        self.assertEqual(parsed.tzinfo, timezone.utc)

    def test_parse_rfc3339_with_z(self):
        parsed = parse_rfc3339("2026-05-15T10:30:00Z")
        self.assertEqual(parsed.tzinfo, timezone.utc)


class TestSessionState(unittest.TestCase):
    def test_create_session(self):
        session = start_session()
        self.assertEqual(session.status, "active")
        self.assertIsNone(session.terminated_at)
        self.assertIsNone(session.termination_reason)
        self.assertTrue(is_valid_ulid(session.session_id))

    def test_session_to_dict(self):
        session = start_session()
        data = session.to_dict()
        self.assertIn("session_id", data)
        self.assertIn("started_at", data)
        self.assertEqual(data["status"], "active")

    def test_session_json_roundtrip(self):
        session = start_session()
        json_str = session.to_json()
        restored = SessionState.from_json(json_str)
        self.assertEqual(session.session_id, restored.session_id)
        self.assertEqual(session.status, restored.status)

    def test_finish_session_completed(self):
        session = start_session()
        finished = finish_session(session, "All tests passed")
        self.assertEqual(finished.status, "completed")
        self.assertIsNotNone(finished.terminated_at)
        self.assertEqual(finished.termination_reason, "All tests passed")
        self.assertTrue(is_session_complete(finished))

    def test_finish_session_failed(self):
        session = start_session()
        finished = finish_session(session, "Test failed: assertion error")
        self.assertEqual(finished.status, "failed")

    def test_finish_session_timed_out(self):
        session = start_session()
        finished = finish_session(session, "Session timed out after 300s")
        self.assertEqual(finished.status, "timed_out")

    def test_duration_seconds_active(self):
        session = start_session()
        self.assertIsNone(session.duration_seconds())

    def test_duration_seconds_finished(self):
        session = start_session()
        finished = finish_session(session, "done")
        self.assertIsNotNone(finished.duration_seconds())
        self.assertGreaterEqual(finished.duration_seconds(), 0)

    def test_invalid_ulid_raises(self):
        with self.assertRaises(ValueError):
            SessionState(
                session_id="not-valid",
                started_at=rfc3339_now(),
            )


class TestPredicateOperators(unittest.TestCase):
    def test_eq_operator(self):
        pred = ActivationPredicate(field="verdict", operator="eq", value="CE")
        ctx = {"verdict": "CE"}
        self.assertTrue(evaluate_predicate(pred, ctx))

        ctx2 = {"verdict": "OK"}
        self.assertFalse(evaluate_predicate(pred, ctx2))

    def test_ne_operator(self):
        pred = ActivationPredicate(field="verdict", operator="ne", value="CE")
        ctx = {"verdict": "OK"}
        self.assertTrue(evaluate_predicate(pred, ctx))
        ctx2 = {"verdict": "CE"}
        self.assertFalse(evaluate_predicate(pred, ctx2))

    def test_gt_operator(self):
        pred = ActivationPredicate(field="quality", operator="gt", value=0.5)
        self.assertTrue(evaluate_predicate(pred, {"quality": 0.8}))
        self.assertFalse(evaluate_predicate(pred, {"quality": 0.3}))
        self.assertFalse(evaluate_predicate(pred, {"quality": 0.5}))

    def test_lt_operator(self):
        pred = ActivationPredicate(field="cost", operator="lt", value=0.1)
        self.assertTrue(evaluate_predicate(pred, {"cost": 0.05}))
        self.assertFalse(evaluate_predicate(pred, {"cost": 0.15}))

    def test_contains_operator_string(self):
        pred = ActivationPredicate(field="error", operator="contains", value="SyntaxError")
        self.assertTrue(evaluate_predicate(pred, {"error": "SyntaxError in line 42"}))
        self.assertFalse(evaluate_predicate(pred, {"error": "RuntimeError"}))

    def test_contains_operator_list(self):
        pred = ActivationPredicate(field="verdicts", operator="contains", value="CE")
        self.assertTrue(evaluate_predicate(pred, {"verdicts": ["CE", "RE", "OK"]}))
        self.assertFalse(evaluate_predicate(pred, {"verdicts": ["OK", "WA"]}))

    def test_regex_operator(self):
        pred = ActivationPredicate(field="error", operator="regex", value=r"Error:\s*\d+")
        self.assertTrue(evaluate_predicate(pred, {"error": "Error: 42 occurred"}))
        self.assertFalse(evaluate_predicate(pred, {"error": "Success"}))

    def test_missing_field(self):
        pred = ActivationPredicate(field="verdict", operator="eq", value="CE")
        ctx = {}
        self.assertFalse(evaluate_predicate(pred, ctx))

    def test_nested_field(self):
        pred = ActivationPredicate(field="metadata.error_type", operator="eq", value="CE")
        ctx = {"metadata": {"error_type": "CE"}}
        self.assertTrue(evaluate_predicate(pred, ctx))


class TestBuildPredicateFromResult(unittest.TestCase):
    def test_build_from_ok_result(self):
        result = HarnessResult(verdict=Verdict.OK)
        pred = build_predicate_from_autobench_result(result)
        self.assertEqual(pred.field, "verdict")
        self.assertEqual(pred.operator, "eq")
        self.assertEqual(pred.value, "OK")

    def test_build_from_ce_result(self):
        result = HarnessResult(verdict=Verdict.CE)
        pred = build_predicate_from_autobench_result(result)
        self.assertEqual(pred.value, "CE")

    def test_build_from_tle_result(self):
        result = HarnessResult(verdict=Verdict.TLE)
        pred = build_predicate_from_autobench_result(result)
        self.assertEqual(pred.value, "TLE")


class TestVerdictRoutingPredicates(unittest.TestCase):
    def test_all_verdicts_mapped(self):
        predicates = build_verdict_routing_predicates()
        for verdict in Verdict:
            self.assertIn(verdict, predicates)
            self.assertEqual(predicates[verdict].operator, "eq")
            self.assertEqual(predicates[verdict].field, "verdict")


class TestRoleSpecActivationBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = RoleSpecActivationBuilder()

    def test_build_basic_role(self):
        pred = ActivationPredicate(field="verdict", operator="eq", value="CE")
        spec = self.builder.build(
            name="error-handler",
            predicate=pred,
            capability="error_recovery",
        )
        self.assertEqual(spec["name"], "error-handler")
        self.assertEqual(spec["capability"], "error_recovery")
        self.assertEqual(spec["activation_predicate"]["field"], "verdict")
        self.assertEqual(spec["activation_predicate"]["operator"], "eq")

    def test_build_from_result(self):
        result = HarnessResult(verdict=Verdict.RE)
        spec = self.builder.build_from_result(result, name="handler")
        self.assertEqual(spec["name"], "handler")
        self.assertEqual(spec["activation_predicate"]["value"], "RE")

    def test_build_error_handler_role(self):
        spec = self.builder.build_error_handler_role()
        self.assertEqual(spec["name"], "error-handler")
        self.assertEqual(spec["activation_predicate"]["value"], "CE")
        self.assertIn("handles", spec["metadata"])

    def test_build_timeout_handler_role(self):
        spec = self.builder.build_timeout_handler_role()
        self.assertEqual(spec["name"], "timeout-handler")
        self.assertEqual(spec["activation_predicate"]["value"], "TLE")

    def test_build_debug_agent_role(self):
        spec = self.builder.build_debug_agent_role()
        self.assertEqual(spec["name"], "debug-agent")
        self.assertEqual(spec["activation_predicate"]["value"], "WA")

    def test_build_worker_role(self):
        spec = self.builder.build_worker_role()
        self.assertEqual(spec["name"], "worker")
        self.assertEqual(spec["activation_predicate"]["value"], "OK")
        self.assertTrue(spec.get("metadata", {}).get("fallback"))

    def test_build_all_handler_roles(self):
        roles = self.builder.build_all_handler_roles()
        self.assertEqual(len(roles), 4)
        names = {r["name"] for r in roles}
        self.assertEqual(
            names,
            {"error-handler", "timeout-handler", "debug-agent", "worker"},
        )

    def test_add_and_retrieve_predicate(self):
        pred = ActivationPredicate(field="verdict", operator="eq", value="CE")
        self.builder.add_predicate("ce_route", pred)
        retrieved = self.builder.predicates["ce_route"]
        self.assertEqual(retrieved.value, "CE")


if __name__ == "__main__":
    unittest.main()