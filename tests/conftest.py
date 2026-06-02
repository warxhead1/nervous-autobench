"""Pytest configuration for autobench tests."""

from __future__ import annotations

import pytest

# Tests that exercise live external services — a real LLM endpoint (MiniMax /
# Anthropic), the running nervous-bus / Redis, or a Firecracker VM. They pass
# locally with those services available but cannot run in plain CI, so they are
# marked `live` here (centralised, rather than scattered decorators) and the CI
# workflow runs `-m "not live"`. Run them locally with `pytest -m live`.
LIVE_TESTS = frozenset({
    "tests/test_bus_notify.py::test_emitter_drops_empty_channels",
    "tests/test_bus_notify.py::test_emitter_drops_malformed_priority",
    "tests/test_bus_notify.py::test_emitter_drops_oversize_summary",
    "tests/test_case_result_attempt.py::test_iterative_stops_at_max_attempts",
    "tests/test_case_result_capture.py::test_case_result_event_emitted",
    "tests/test_firecracker.py::test_integration_real_vm_boots",
    "tests/test_multi_improver.py::test_self_improving_harness_legacy_minimax_path_unchanged",
    "tests/test_multi_improver.py::test_self_improving_harness_resolves_minimax_ensemble",
    "tests/test_reasoning_capture.py::test_all_reasoning_and_divergence_events_validate",
    "tests/test_reasoning_capture.py::test_divergence_true_when_llm_disagrees_with_heuristic",
    "tests/test_self_revision.py::test_self_revision_caps_at_two_attempts",
    "tests/test_tsp_horizons.py::test_plateau_stops_when_no_improvement",
    "tests/test_wiring.py::test_wiring_emits_all_four_channels",
})


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_tools: integration test that needs external CLI tools "
        "(ast-grep, difft); auto-skipped when binaries are absent.",
    )
    config.addinivalue_line(
        "markers",
        "live: needs a live external service (LLM endpoint, nervous-bus/Redis, "
        "or Firecracker VM); excluded from CI via `-m 'not live'`.",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in LIVE_TESTS:
            item.add_marker(pytest.mark.live)
