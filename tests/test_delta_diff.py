"""Tests for the harness before/after diff helper + observability channel.

Covers:
    * Unit: identical harnesses → no_change=true, empty diffs.
    * Unit: system_prompt change → unified diff present, others empty/null.
    * Unit: budget max_tokens 4096 → 2048 → budget_changes populated.
    * Unit: rollout_protocol change → rollout_protocol_change populated.
    * Unit: multiple fields changing simultaneously.
    * Schema validation against emitted payload (no_change=false case AND
      no_change=true case).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from autobench.core import ContextManager, HarnessConfig, RolloutProtocol
from autobench.harness_diff import diff_harnesses
from autobench.observability import (
    AutobenchObservability,
    CHANNEL_DELTA_DIFF,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "autobench.improver.delta.diff.v1.json"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def base_harness() -> HarnessConfig:
    return HarnessConfig(
        system_prompt="Solve the problem.\nReturn code only.\n",
        rollout_protocol=RolloutProtocol.SINGLE,
        context_manager=ContextManager.FULL,
        tool_surface="run_python(code: str) -> str\n",
        budget={
            "max_tokens": 4096,
            "max_time_seconds": 30,
            "max_cost_dollars": 0.10,
            "max_memory_mb": 512,
        },
    )


@pytest.fixture
def debug_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Clean debug file + empty PATH so zellij pipe falls back to JSONL."""
    path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Unit: diff_harnesses
# --------------------------------------------------------------------------- #

def test_identical_harnesses_no_change(base_harness: HarnessConfig) -> None:
    other = copy.deepcopy(base_harness)
    out = diff_harnesses(base_harness, other)
    assert out["no_change"] is True
    assert out["system_prompt_diff"] == ""
    assert out["tool_surface_diff"] == ""
    assert out["rollout_protocol_change"] is None
    assert out["context_manager_change"] is None
    assert out["budget_changes"] == {}


def test_system_prompt_change_yields_unified_diff(base_harness: HarnessConfig) -> None:
    after = copy.deepcopy(base_harness)
    after.system_prompt = "Solve the problem.\nReturn code only.\nThink step by step.\n"
    out = diff_harnesses(base_harness, after)
    assert out["no_change"] is False
    sp = out["system_prompt_diff"]
    # Unified-diff hallmarks: file headers + "+ " line for the added text.
    assert "--- a/system_prompt" in sp
    assert "+++ b/system_prompt" in sp
    assert "Think step by step." in sp
    assert sp.startswith("--- ")
    # Other fields untouched.
    assert out["tool_surface_diff"] == ""
    assert out["rollout_protocol_change"] is None
    assert out["context_manager_change"] is None
    assert out["budget_changes"] == {}


def test_budget_max_tokens_change(base_harness: HarnessConfig) -> None:
    after = copy.deepcopy(base_harness)
    after.budget["max_tokens"] = 2048
    out = diff_harnesses(base_harness, after)
    assert out["no_change"] is False
    assert out["budget_changes"] == {
        "max_tokens": {"before": 4096, "after": 2048},
    }
    # All other tracked fields unchanged.
    assert out["system_prompt_diff"] == ""
    assert out["tool_surface_diff"] == ""
    assert out["rollout_protocol_change"] is None
    assert out["context_manager_change"] is None


def test_rollout_protocol_change(base_harness: HarnessConfig) -> None:
    after = copy.deepcopy(base_harness)
    after.rollout_protocol = RolloutProtocol.SELF_REVISION
    out = diff_harnesses(base_harness, after)
    assert out["no_change"] is False
    rp = out["rollout_protocol_change"]
    assert rp is not None
    assert rp["before"] == RolloutProtocol.SINGLE.value
    assert rp["after"] == RolloutProtocol.SELF_REVISION.value
    assert out["context_manager_change"] is None
    assert out["budget_changes"] == {}


def test_multiple_fields_change_simultaneously(base_harness: HarnessConfig) -> None:
    after = copy.deepcopy(base_harness)
    after.system_prompt = base_harness.system_prompt + "Be concise.\n"
    after.tool_surface = "run_python(code: str) -> str\nweb_search(q: str) -> str\n"
    after.context_manager = ContextManager.BUDGETED
    after.budget["max_cost_dollars"] = 0.25
    after.budget["max_memory_mb"] = 1024

    out = diff_harnesses(base_harness, after)
    assert out["no_change"] is False
    assert "Be concise." in out["system_prompt_diff"]
    assert "web_search" in out["tool_surface_diff"]
    assert out["context_manager_change"] == {
        "before": ContextManager.FULL.value,
        "after": ContextManager.BUDGETED.value,
    }
    assert out["budget_changes"] == {
        "max_cost_dollars": {"before": 0.10, "after": 0.25},
        "max_memory_mb": {"before": 512, "after": 1024},
    }
    # rollout_protocol still unchanged.
    assert out["rollout_protocol_change"] is None


# --------------------------------------------------------------------------- #
# Emission + schema validation
# --------------------------------------------------------------------------- #

def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _events_on(path: Path, channel: str) -> list[dict]:
    return [e for e in _read_events(path) if e.get("type") == channel]


def test_emit_no_change_validates_against_schema(
    debug_file: Path, base_harness: HarnessConfig,
) -> None:
    obs = AutobenchObservability(debug_file=debug_file)
    obs.improver_delta_diff(iteration=1, before=base_harness, after=base_harness)
    events = _events_on(debug_file, CHANNEL_DELTA_DIFF)
    assert len(events) == 1
    ev = events[0]
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)
    assert ev["data"]["no_change"] is True
    assert ev["data"]["iteration"] == 1


def test_emit_with_changes_validates_against_schema(
    debug_file: Path, base_harness: HarnessConfig,
) -> None:
    after = copy.deepcopy(base_harness)
    after.system_prompt = base_harness.system_prompt + "Think step by step.\n"
    after.budget["max_tokens"] = 8192
    after.rollout_protocol = RolloutProtocol.ITERATIVE

    obs = AutobenchObservability(debug_file=debug_file)
    obs.improver_delta_diff(iteration=2, before=base_harness, after=after)
    events = _events_on(debug_file, CHANNEL_DELTA_DIFF)
    assert len(events) == 1
    ev = events[0]
    jsonschema.Draft202012Validator(_load_schema()).validate(ev)
    data = ev["data"]
    assert data["no_change"] is False
    assert data["iteration"] == 2
    assert "Think step by step." in data["system_prompt_diff"]
    assert data["budget_changes"]["max_tokens"] == {"before": 4096, "after": 8192}
    assert data["rollout_protocol_change"]["after"] == RolloutProtocol.ITERATIVE.value
    # Session id stamped on every event.
    assert data["session_id"] == obs.session_id
