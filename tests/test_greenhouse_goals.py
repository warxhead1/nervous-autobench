"""Tests for greenhouse.goals — manifest loading and validation."""
from __future__ import annotations

import json

import pytest

from greenhouse.goals import GoalsManifestError, goals_path, load_manifest, parse_manifest

_VALID = {
    "version": 1,
    "budget": {"window_seconds": 18000, "window_max_requests": 1500, "per_cycle_max_requests": 300},
    "goals": [
        {"id": "g1", "domain": "sdf", "instances": ["gyroid"], "priority": 2, "want": 3},
    ],
}


def _manifest(**overrides):
    d = json.loads(json.dumps(_VALID))
    d.update(overrides)
    return d


def test_parse_valid_manifest_roundtrip():
    m = parse_manifest(_VALID)
    assert m.version == 1
    assert m.budget.window_seconds == 18000
    assert m.budget.window_max_requests == 1500
    assert m.budget.per_cycle_max_requests == 300
    assert len(m.goals) == 1
    g = m.goals[0]
    assert g.id == "g1" and g.domain == "sdf" and g.instances == ["gyroid"]
    assert g.priority == 2 and g.want == 3
    assert g.target_fitness is None and g.tags == [] and g.notes == ""


def test_goal_defaults_priority_and_want():
    d = _manifest()
    del d["goals"][0]["priority"]
    del d["goals"][0]["want"]
    m = parse_manifest(d)
    assert m.goals[0].priority == 1
    assert m.goals[0].want == 1


def test_optional_fields_parsed():
    d = _manifest()
    d["goals"][0]["target_fitness"] = 0.85
    d["goals"][0]["tags"] = ["showcase"]
    d["goals"][0]["notes"] = "some notes"
    m = parse_manifest(d)
    g = m.goals[0]
    assert g.target_fitness == pytest.approx(0.85)
    assert g.tags == ["showcase"]
    assert g.notes == "some notes"


@pytest.mark.parametrize("key", ["window_seconds", "window_max_requests", "per_cycle_max_requests"])
def test_budget_missing_field_rejected(key):
    d = _manifest()
    del d["budget"][key]
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_budget_per_cycle_exceeding_window_rejected():
    d = _manifest()
    d["budget"]["per_cycle_max_requests"] = 2000
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_budget_non_positive_rejected():
    d = _manifest()
    d["budget"]["window_max_requests"] = 0
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_unknown_domain_rejected():
    d = _manifest()
    d["goals"][0]["domain"] = "not_a_domain"
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_duplicate_goal_id_rejected():
    d = _manifest()
    d["goals"].append(json.loads(json.dumps(d["goals"][0])))
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_empty_instances_rejected():
    d = _manifest()
    d["goals"][0]["instances"] = []
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_negative_want_rejected():
    d = _manifest()
    d["goals"][0]["want"] = -1
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_priority_below_one_rejected():
    d = _manifest()
    d["goals"][0]["priority"] = 0
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_empty_goals_list_rejected():
    d = _manifest()
    d["goals"] = []
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_unsupported_version_rejected():
    d = _manifest()
    d["version"] = 2
    with pytest.raises(GoalsManifestError):
        parse_manifest(d)


def test_load_manifest_missing_file_raises(tmp_path):
    with pytest.raises(GoalsManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")


def test_load_manifest_invalid_json_raises(tmp_path):
    p = tmp_path / "goals.json"
    p.write_text("{not json")
    with pytest.raises(GoalsManifestError, match="not valid JSON"):
        load_manifest(p)


def test_load_manifest_reads_valid_file(tmp_path):
    p = tmp_path / "goals.json"
    p.write_text(json.dumps(_VALID))
    m = load_manifest(p)
    assert m.goals[0].id == "g1"


def test_example_manifest_is_valid():
    example = json.loads((_pkg_root() / "greenhouse" / "goals.example.json").read_text())
    m = parse_manifest(example)
    assert len(m.goals) == 3
    assert {g.domain for g in m.goals} == {"terrain", "noise", "sdf"}


def test_goals_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom-goals.json"
    monkeypatch.setenv("GREENHOUSE_GOALS", str(custom))
    assert goals_path() == custom


def _pkg_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent
