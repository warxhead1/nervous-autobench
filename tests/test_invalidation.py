"""
Tests for autobench.invalidation — Bitloops-style source_scope_key temporal invalidation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from autobench.audit.invalidation import (
    InvalidationEngine,
    InvalidationResult,
    InvalidationStore,
    ahe_scope_key,
    bead_scope_key,
    history_source_scope_key,
    promotion_scope_key,
    schema_scope_key,
)


# --------------------------------------------------------------------------- #
# Scope key construction tests
# --------------------------------------------------------------------------- #

class TestScopeKeyConstruction:
    """Scope keys must be stable and unambiguous across all event types."""

    def test_history_source_scope_key_full(self) -> None:
        sk = history_source_scope_key("sess-abc", turn_id="turn-1", checkpoint_id="cp-3")
        assert sk == "history_source:sess-abc:turn-1:cp-3"

    def test_history_source_scope_key_optionals(self) -> None:
        sk = history_source_scope_key("sess-abc")
        assert sk == "history_source:sess-abc:_:_"

        sk2 = history_source_scope_key("sess-abc", turn_id="turn-1")
        assert sk2 == "history_source:sess-abc:turn-1:_"

    def test_ahe_scope_key(self) -> None:
        sk = ahe_scope_key("sess-abc", "prob-xyz", 5)
        assert sk == "ahe:sess-abc:prob-xyz:5"

    def test_bead_scope_key(self) -> None:
        sk = bead_scope_key("bead-123", "exec-456")
        assert sk == "ac_verify:bead-123:exec-456"

    def test_schema_scope_key(self) -> None:
        sk = schema_scope_key("promotion_decision", "v2.1.0")
        assert sk == "schema_scope:promotion_decision:v2.1.0"

    def test_promotion_scope_key(self) -> None:
        sk = promotion_scope_key("cycle-01KRT3GG2W0EEG86GK9FW7KBA0")
        assert sk == "promotion:cycle-01KRT3GG2W0EEG86GK9FW7KBA0"

    def test_scope_keys_are_unique_per_type(self) -> None:
        """Different event types must produce different key prefixes."""
        assert history_source_scope_key("x", "y", "z").startswith("history_source:")
        assert ahe_scope_key("x", "y", 1).startswith("ahe:")
        assert bead_scope_key("x", "y").startswith("ac_verify:")
        assert schema_scope_key("x", "y").startswith("schema_scope:")
        assert promotion_scope_key("x").startswith("promotion:")

        # Cross-type collision check
        keys = [
            history_source_scope_key("a:b:c"),
            ahe_scope_key("a", "b", 1),
            bead_scope_key("a:b", "c"),
            schema_scope_key("a:b", "c"),
            promotion_scope_key("a:b:c"),
        ]
        assert len(keys) == len(set(keys)), "scope keys must not collide across types"


# --------------------------------------------------------------------------- #
# InvalidationStore tests
# --------------------------------------------------------------------------- #

class TestInvalidationStore:
    """InvalidationStore persists and retrieves invalidation state."""

    def test_record_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = InvalidationStore(path=Path(tmpdir) / "store.jsonl")

            assert store.is_invalidated("promotion:cycle-1") is False
            assert store.last_invalidation("promotion:cycle-1") is None

            store.record_invalidation("promotion:cycle-1", "re-distillation", 5)

            assert store.is_invalidated("promotion:cycle-1") is True
            ts = store.last_invalidation("promotion:cycle-1")
            assert ts is not None
            assert "T" in ts  # ISO format contains 'T'

    def test_persistence_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.jsonl"

            store_a = InvalidationStore(path=path)
            store_a.record_invalidation("ahe:s1:p1:1", "re-distillation", 3)
            del store_a

            store_b = InvalidationStore(path=path)
            assert store_b.is_invalidated("ahe:s1:p1:1") is True
            assert store_b.last_invalidation("ahe:s1:p1:1") is not None

    def test_unknown_scope_not_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = InvalidationStore(path=Path(tmpdir) / "store.jsonl")
            assert store.is_invalidated("never-seen-scope") is False
            assert store.last_invalidation("never-seen-scope") is None

    def test_empty_store_file_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            path.touch()
            store = InvalidationStore(path=path)
            assert store.is_invalidated("any") is False


# --------------------------------------------------------------------------- #
# InvalidationEngine tests
# --------------------------------------------------------------------------- #

class TestInvalidationEngine:
    """InvalidationEngine.check_and_invalidate marks re-distilled scopes."""

    def test_first_seen_scope_not_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = InvalidationEngine()
            engine.store = InvalidationStore(path=Path(tmpdir) / "store.jsonl")

            result = engine.check_and_invalidate("promotion:cycle-1", {"cycle_id": "cycle-1"})

            assert result.was_invalidated is False
            assert result.count_deactivated == 0
            assert result.reason is None

    def test_re_distillation_triggers_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "store.jsonl"
            engine = InvalidationEngine()
            engine.store = InvalidationStore(path=store_path)

            # First call — not invalidated
            result1 = engine.check_and_invalidate("ahe:s1:p1:1", {"session_id": "s1"})
            assert result1.was_invalidated is False

            # Simulate a prior entry by manually recording
            engine.store.record_invalidation("ahe:s1:p1:1", "re-distillation", 0)

            # Second call — should be invalidated
            result2 = engine.check_and_invalidate("ahe:s1:p1:1", {"session_id": "s1"})
            assert result2.was_invalidated is True
            assert result2.reason == "re-distillation"

    def test_count_deactivated_accumulated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "store.jsonl"
            engine = InvalidationEngine()
            engine.store = InvalidationStore(path=store_path)

            deactivated_total = 0

            def count_fn(scope_key: str) -> int:
                return 3  # stub returns 3 per prefix match

            engine.register("ahe:", count_fn, "autobench.improver.prediction.v1")

            # Pre-populate so check_and_invalidate sees it as re-distillation
            engine.store.record_invalidation("ahe:s1:p1:1", "test", 0)

            result = engine.check_and_invalidate("ahe:s1:p1:1", {"session_id": "s1"})

            assert result.was_invalidated is True
            assert result.count_deactivated == 3  # 1 prefix match × 3

    def test_register_and_fire(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "store.jsonl"
            engine = InvalidationEngine()
            engine.store = InvalidationStore(path=store_path)

            calls: list[str] = []

            def collect_fn(sk: str) -> int:
                calls.append(sk)
                return 1

            engine.register("promotion:", collect_fn, "autobench.continuous.promotion.v1")

            # Pre-populate
            engine.store.record_invalidation("promotion:cycle-1", "test", 0)

            result = engine.check_and_invalidate("promotion:cycle-1", {})

            assert result.was_invalidated is True
            assert "promotion:cycle-1" in calls

    def test_check_and_invalidate_records_new_invalidation(self) -> None:
        """After a re-distillation, the scope key's invalidation timestamp advances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "store.jsonl"
            engine = InvalidationEngine()
            engine.store = InvalidationStore(path=store_path)

            engine.store.record_invalidation("promotion:c1", "first", 0)
            first_ts = engine.store.last_invalidation("promotion:c1")

            # Re-distillation
            engine.check_and_invalidate("promotion:c1", {})

            second_ts = engine.store.last_invalidation("promotion:c1")
            assert second_ts is not None
            assert second_ts != first_ts  # timestamp advanced

            # Count still 0 (stub no-op)
            assert engine.store.is_invalidated("promotion:c1") is True


# --------------------------------------------------------------------------- #
# InvalidationResult dataclass
# --------------------------------------------------------------------------- #

class TestInvalidationResult:
    """InvalidationResult carries deactivation metadata."""

    def test_fields_present(self) -> None:
        r = InvalidationResult(was_invalidated=True, count_deactivated=7, reason="re-distillation")
        assert r.was_invalidated is True
        assert r.count_deactivated == 7
        assert r.reason == "re-distillation"

    def test_none_reason_when_not_invalidated(self) -> None:
        r = InvalidationResult(was_invalidated=False, count_deactivated=0, reason=None)
        assert r.was_invalidated is False
        assert r.reason is None