"""Tests for greenhouse.ledger — persistent sliding-window request ledger."""
from __future__ import annotations

import json

from greenhouse.ledger import Ledger


def test_record_and_window_used(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    now = 1_000_000.0
    ledger.record(run_id="r1", goal_id="g1", requests=10, ts=now - 100)
    ledger.record(run_id="r2", goal_id="g1", requests=5, ts=now - 50)
    assert ledger.window_used(200, now=now) == 15


def test_record_zero_or_negative_is_noop(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    ledger.record(run_id="r1", goal_id="g1", requests=0, ts=1000.0)
    ledger.record(run_id="r2", goal_id="g1", requests=-5, ts=1000.0)
    assert not ledger.path.exists()
    assert ledger.window_used(1000, now=2000.0) == 0


def test_window_used_excludes_entries_outside_window(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    now = 1_000_000.0
    ledger.record(run_id="old", goal_id="g1", requests=99, ts=now - 500)  # outside a 100s window
    ledger.record(run_id="recent", goal_id="g1", requests=7, ts=now - 10)
    assert ledger.window_used(100, now=now) == 7


def test_window_boundary_is_exclusive_at_cutoff(tmp_path):
    """An entry timestamped exactly at the cutoff (now - window_seconds) is outside the window."""
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    now = 1_000_000.0
    window = 100.0
    ledger.record(run_id="at_cutoff", goal_id="g1", requests=3, ts=now - window)
    ledger.record(run_id="just_inside", goal_id="g1", requests=4, ts=now - window + 0.001)
    assert ledger.window_used(window, now=now) == 4


def test_remaining_floors_at_zero(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    now = 1_000_000.0
    ledger.record(run_id="r1", goal_id="g1", requests=500, ts=now - 10)
    assert ledger.remaining(300, 1000, now=now) == 0


def test_remaining_reflects_usage(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    now = 1_000_000.0
    ledger.record(run_id="r1", goal_id="g1", requests=100, ts=now - 10)
    assert ledger.remaining(300, 1000, now=now) == 200


def test_ledger_persists_across_instances(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path=path).record(run_id="r1", goal_id="g1", requests=10, ts=1000.0)
    second = Ledger(path=path)
    assert second.window_used(10_000, now=2000.0) == 10


def test_corrupted_line_is_tolerated(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"ts": 1000.0, "run_id": "r1", "goal_id": "g1", "requests": 5}) + "\n")
        f.write("not json at all\n")
        f.write(json.dumps({"ts": 1001.0, "run_id": "r2", "goal_id": "g1", "requests": 3}) + "\n")
    ledger = Ledger(path=path)
    assert ledger.window_used(10_000, now=2000.0) == 8


def test_record_writes_jsonl_shape(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path=path).record(run_id="r1", goal_id="g1", requests=10, ts=1234.5)
    line = path.read_text().strip()
    entry = json.loads(line)
    assert entry == {"ts": 1234.5, "run_id": "r1", "goal_id": "g1", "requests": 10}
