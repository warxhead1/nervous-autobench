"""Tests for curriculum.py wiring into the cf-tier-1 run_first benchmark loader.

Covers Phase 4 of the wire-pop epic (nervous-bus-pmrg). The wiring lives in
``autobench.benchmarks.codeforces_tier1.run_first._load_cases`` and the new
``autobench.curriculum.daily_synthesis`` entry point.

What's exercised here:

* Default behavior (no env flag) returns the 20 fixed cf-tier-1 cases.
* ``AUTOBENCH_CURRICULUM=1`` routes through ``daily_synthesis``.
* Empty curriculum output triggers fallback to the fixed set — never raises.
* ``daily_synthesis`` caches per-date: two calls on the same date hit disk
  on the second call (LLM caller invoked exactly once).
* ``target_skill`` is plumbed through to a CF-rating hint so the operator
  can dial difficulty without a full auto-calibration loop.

The tests use a fake ``llm_caller`` to bypass MiniMax entirely — no API key
required, no network. Cache writes go to a tmp dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from autobench.benchmarks.codeforces_tier1.run_first import (
    _load_cases,
    _load_fixed_cases,
    CASES_FILE,
)
from autobench.curriculum import (
    DEFAULT_CACHE_DIR,
    _difficulty_for_target_skill,
    daily_synthesis,
)
from autobench.evaluator import BenchmarkCase


# --------------------------------------------------------------------------- #
# Fake LLM response — one synthesized problem JSON
# --------------------------------------------------------------------------- #


def _fake_llm_response(n: int) -> str:
    rows = []
    for i in range(n):
        rows.append({
            "id": f"curr-{i:03d}",
            "prompt": (
                f"Given an integer N (1 <= N <= 100), print the sum 1+2+...+N "
                f"(problem #{i})."
            ),
            "expected_output": "15",
            "sample_input": "5",
            "difficulty_rating": 1200,
            "target_skills": ["math", "implementation"],
            "rationale": "drills simple loop arithmetic",
        })
    return json.dumps(rows)


def _make_fake_caller(n: int, calls: list[int]) -> Any:
    """Return an ``llm_caller`` that records each invocation in ``calls``."""

    def _caller(system_prompt: str, user_prompt: str) -> str:
        calls.append(1)
        return _fake_llm_response(n)

    return _caller


# --------------------------------------------------------------------------- #
# Disabled-by-default path
# --------------------------------------------------------------------------- #


def test_curriculum_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without AUTOBENCH_CURRICULUM env, _load_cases returns the fixed set."""
    monkeypatch.delenv("AUTOBENCH_CURRICULUM", raising=False)
    monkeypatch.delenv("AUTOBENCH_CURRICULUM_SKILL", raising=False)

    cases = _load_cases()  # default use_curriculum=False
    fixed = _load_fixed_cases()

    assert isinstance(cases, list)
    assert len(cases) == len(fixed)
    assert len(cases) > 0  # cf-tier-1 must have content (20 cases on disk)
    assert all(isinstance(c, BenchmarkCase) for c in cases)
    # Same ids — same source file
    assert {c.id for c in cases} == {c.id for c in fixed}


def test_load_cases_use_curriculum_false_ignores_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with env set, use_curriculum=False argument forces fixed path."""
    monkeypatch.setenv("AUTOBENCH_CURRICULUM", "1")
    cases = _load_cases(use_curriculum=False)
    fixed = _load_fixed_cases()
    assert {c.id for c in cases} == {c.id for c in fixed}


# --------------------------------------------------------------------------- #
# Enabled path
# --------------------------------------------------------------------------- #


def test_curriculum_enabled_calls_daily_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With curriculum enabled, _load_cases routes through daily_synthesis."""
    # Redirect cache to a tmp dir so we don't pollute autobench/data.
    monkeypatch.setattr(
        "autobench.curriculum.DEFAULT_CACHE_DIR",
        tmp_path / "cache",
    )

    calls: list[int] = []
    fake = _make_fake_caller(n=5, calls=calls)

    sentinel: list[int] = []

    def _wrapped_synthesis(*args: Any, **kwargs: Any) -> list[BenchmarkCase]:
        sentinel.append(1)
        # Inject the fake caller so the wrapper doesn't need a key.
        kwargs.setdefault("llm_caller", fake)
        kwargs.setdefault("cache_dir", tmp_path / "cache")
        return daily_synthesis(*args, **kwargs)

    with patch(
        "autobench.curriculum.daily_synthesis",
        side_effect=_wrapped_synthesis,
    ):
        cases = _load_cases(use_curriculum=True, target_skill=0.5)

    assert sentinel == [1], "daily_synthesis must be invoked exactly once per cycle"
    assert len(cases) == 5
    assert all(isinstance(c, BenchmarkCase) for c in cases)
    # Curriculum ids carry the date prefix
    assert all(c.id.startswith("curr-") for c in cases)


def test_curriculum_empty_falls_back_to_fixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """daily_synthesis returns [] → _load_cases falls through, never raises."""
    monkeypatch.setattr(
        "autobench.curriculum.DEFAULT_CACHE_DIR",
        tmp_path / "cache",
    )

    def _empty_synthesis(*args: Any, **kwargs: Any) -> list[BenchmarkCase]:
        return []

    with patch(
        "autobench.curriculum.daily_synthesis",
        side_effect=_empty_synthesis,
    ):
        cases = _load_cases(use_curriculum=True, target_skill=0.5)

    fixed = _load_fixed_cases()
    assert len(cases) == len(fixed)
    assert {c.id for c in cases} == {c.id for c in fixed}


def test_curriculum_exception_falls_back_to_fixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If daily_synthesis raises, _load_cases still produces cf-tier-1 cases."""
    monkeypatch.setattr(
        "autobench.curriculum.DEFAULT_CACHE_DIR",
        tmp_path / "cache",
    )

    def _boom(*args: Any, **kwargs: Any) -> list[BenchmarkCase]:
        raise RuntimeError("simulated curriculum failure")

    with patch(
        "autobench.curriculum.daily_synthesis",
        side_effect=_boom,
    ):
        cases = _load_cases(use_curriculum=True, target_skill=0.5)

    assert len(cases) == len(_load_fixed_cases())


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_curriculum_caches_per_date(tmp_path: Path) -> None:
    """Two calls on the same date — LLM invoked exactly once (cache hit on 2nd)."""
    calls: list[int] = []
    fake = _make_fake_caller(n=3, calls=calls)

    cases_a = daily_synthesis(
        n=3,
        target_skill=0.5,
        date="2026-05-16",
        cache_dir=tmp_path,
        llm_caller=fake,
    )
    cases_b = daily_synthesis(
        n=3,
        target_skill=0.5,
        date="2026-05-16",
        cache_dir=tmp_path,
        llm_caller=fake,
    )

    assert len(cases_a) == 3
    assert len(cases_b) == 3
    assert len(calls) == 1, "LLM caller should only fire on cache miss"
    # Ids match across calls — same cached payload
    assert [c.id for c in cases_a] == [c.id for c in cases_b]


def test_curriculum_different_dates_resynthesize(tmp_path: Path) -> None:
    """Different date → different cache file → LLM called once per date."""
    calls: list[int] = []
    fake = _make_fake_caller(n=2, calls=calls)

    daily_synthesis(
        n=2, target_skill=0.5, date="2026-05-16",
        cache_dir=tmp_path, llm_caller=fake,
    )
    daily_synthesis(
        n=2, target_skill=0.5, date="2026-05-17",
        cache_dir=tmp_path, llm_caller=fake,
    )

    assert len(calls) == 2
    assert (tmp_path / "curriculum_2026-05-16.jsonl").exists()
    assert (tmp_path / "curriculum_2026-05-17.jsonl").exists()


def test_curriculum_empty_does_not_cache(tmp_path: Path) -> None:
    """If the LLM returns nothing, no cache file is written."""
    def _empty(system: str, user: str) -> str:
        return "[]"

    cases = daily_synthesis(
        n=3, target_skill=0.5, date="2026-05-16",
        cache_dir=tmp_path, llm_caller=_empty,
    )
    assert cases == []
    assert not (tmp_path / "curriculum_2026-05-16.jsonl").exists()


# --------------------------------------------------------------------------- #
# target_skill → difficulty mapping
# --------------------------------------------------------------------------- #


def test_difficulty_mapping_monotonic() -> None:
    """Higher target_skill → higher CF difficulty rating, clamped to [800, 2400]."""
    assert _difficulty_for_target_skill(0.0) == 800
    assert _difficulty_for_target_skill(0.5) == 1600
    assert _difficulty_for_target_skill(1.0) == 2400
    # Out-of-range gets clamped
    assert _difficulty_for_target_skill(-0.5) == 800
    assert _difficulty_for_target_skill(2.0) == 2400
    # Monotonic
    assert (
        _difficulty_for_target_skill(0.2)
        < _difficulty_for_target_skill(0.6)
        < _difficulty_for_target_skill(0.9)
    )


def test_difficulty_hint_lands_in_synthesis_prompt(tmp_path: Path) -> None:
    """target_skill must flow into the LLM prompt as a difficulty nudge."""
    captured: dict[str, str] = {}

    def _capture(system: str, user: str) -> str:
        captured["user"] = user
        return _fake_llm_response(n=2)

    daily_synthesis(
        n=2, target_skill=0.8, date="2026-05-16",
        cache_dir=tmp_path, llm_caller=_capture,
    )

    assert "user" in captured
    # difficulty band shows up in the synthesis prompt via evidence dict
    assert "target_difficulty" in captured["user"] or "target-difficulty-band" in captured["user"]


# --------------------------------------------------------------------------- #
# Unique ids
# --------------------------------------------------------------------------- #


def test_curriculum_ids_get_date_prefix(tmp_path: Path) -> None:
    """Generated problems carry a date-stable id prefix to avoid collisions."""
    fake = _make_fake_caller(n=3, calls=[])
    cases = daily_synthesis(
        n=3, target_skill=0.5, date="2026-05-16",
        cache_dir=tmp_path, llm_caller=fake,
    )
    # date prefix without dashes
    prefix = "curr-20260516-"
    assert all(c.id.startswith(prefix) for c in cases), (
        f"expected ids prefixed with {prefix!r}, got {[c.id for c in cases]}"
    )
