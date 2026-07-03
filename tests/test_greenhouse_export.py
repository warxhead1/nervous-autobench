"""Tests for greenhouse.export — per-domain GLSL transpile/validate/drop-write."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenhouse import export

_FIXTURES = Path(__file__).resolve().parent.parent / "greenhouse" / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / f"dry_run_{name}.json").read_text())


def test_sdf_valid_candidate_exports(tmp_path):
    program = _fixture("sdf")["top_programs"][0]
    result = export.export_candidate(
        domain="sdf", goal_id="g1", goal_notes="notes", goal_tags=["a"],
        instance="gyroid", program=program, run_id="run1", drops_root=tmp_path,
    )
    assert result.validated is True
    assert result.drop_path is not None and result.drop_path.is_file()
    assert result.glsl_bytes > 0
    drop = json.loads(result.drop_path.read_text())
    assert drop["domain"] == "sdf"
    assert drop["origin"] == "greenhouse"
    assert drop["author"] == "funsearch-greenhouse"
    assert drop["language"] == "glsl"
    assert drop["tags"] == ["a"]
    assert "float sdf(vec3 pos)" in drop["glsl"]


def test_sdf_gpu_incompatible_candidate_not_exported(tmp_path):
    program = _fixture("sdf")["top_programs"][1]  # static array -> GPU-incompatible
    result = export.export_candidate(
        domain="sdf", goal_id="g1", goal_notes="", goal_tags=[],
        instance="gyroid", program=program, run_id="run1", drops_root=tmp_path,
    )
    assert result.validated is False
    assert result.drop_path is None
    assert result.errors


def test_terrain_valid_candidate_exports(tmp_path):
    program = _fixture("terrain")["top_programs"][0]
    result = export.export_candidate(
        domain="terrain", goal_id="g2", goal_notes="", goal_tags=[],
        instance="rolling_hills", program=program, run_id="run2", drops_root=tmp_path,
    )
    assert result.validated is True
    drop = json.loads(result.drop_path.read_text())
    assert "float terrain(vec2 p)" in drop["glsl"]
    assert "fabsf" not in drop["glsl"] and "sinf(" not in drop["glsl"]


def test_noise_valid_candidate_exports(tmp_path):
    program = _fixture("noise")["top_programs"][0]
    result = export.export_candidate(
        domain="noise", goal_id="g3", goal_notes="", goal_tags=[],
        instance="perlin_like", program=program, run_id="run3", drops_root=tmp_path,
    )
    assert result.validated is True
    drop = json.loads(result.drop_path.read_text())
    assert "float noise(vec3 p)" in drop["glsl"]
    assert "mainImage" in drop["glsl"]


def test_unsupported_domain_raises(tmp_path):
    program = {"id": "x", "code": "", "fitness": 0.5, "generation": 0}
    with pytest.raises(export.UnsupportedDomain):
        export.export_candidate(
            domain="phase", goal_id="g4", goal_notes="", goal_tags=[],
            instance="", program=program, run_id="run4", drops_root=tmp_path,
        )


def test_dropped_count_reflects_written_drops(tmp_path):
    assert export.dropped_count("g1", tmp_path) == 0
    program = _fixture("sdf")["top_programs"][0]
    export.export_candidate(
        domain="sdf", goal_id="g1", goal_notes="", goal_tags=[],
        instance="gyroid", program=program, run_id="run1", drops_root=tmp_path,
    )
    assert export.dropped_count("g1", tmp_path) == 1
    assert export.dropped_count("g-nonexistent", tmp_path) == 0
