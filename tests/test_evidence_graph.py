"""Tests for autobench.evidence_graph — Bitloops-style evidence graph schema.

Verifies:
    * Symbol lineage insert + recursive query
    * Schema version recording
    * Event invalidation on schema change
    * DevQL-style query interfaces
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from autobench.evidence_graph import EvidenceGraphDB


@pytest.fixture
def db(tmp_path: Path) -> EvidenceGraphDB:
    """Provide a fresh EvidenceGraphDB backed by a temporary file."""
    path = str(tmp_path / "evidence_graph.db")
    return EvidenceGraphDB(path=path)


# --------------------------------------------------------------------------- #
# Symbol lineage — insert + recursive query
# --------------------------------------------------------------------------- #

def test_insert_symbol_lineage(db: EvidenceGraphDB) -> None:
    row = {
        "relation_id": "rel-001",
        "repo_id": "repo-alpha",
        "session_id": "sess-abc",
        "checkpoint_id": "ckpt-001",
        "event_time": "2026-05-18T12:00:00Z",
        "agent": "claude-sonnet",
        "lineage_kind": "refactor",
        "source_symbol_id": "sym-foo",
        "source_artefact_id": "art-foo-py",
        "source_blob_sha": "abc123",
        "dest_symbol_id": "sym-bar",
        "dest_artefact_id": "art-bar-py",
        "dest_blob_sha": "def456",
        "commit_sha": "a1b2c3d",
    }
    db.insert_symbol_lineage(row)

    results = db.query_symbol_lineage("repo-alpha", "sym-foo")
    assert len(results) == 1
    assert results[0]["relation_id"] == "rel-001"
    assert results[0]["lineage_kind"] == "refactor"


def test_recursive_lineage_query(db: EvidenceGraphDB) -> None:
    """Insert a chain A -> B -> C and verify depth-limited traversal."""
    chain = [
        {
            "relation_id": "rel-chain-1",
            "repo_id": "repo-beta",
            "session_id": "sess-1",
            "checkpoint_id": "ckpt-1",
            "event_time": "2026-05-18T10:00:00Z",
            "lineage_kind": "extract",
            "source_symbol_id": "sym-a",
            "source_artefact_id": "art-a-py",
            "dest_symbol_id": "sym-b",
            "dest_artefact_id": "art-b-py",
        },
        {
            "relation_id": "rel-chain-2",
            "repo_id": "repo-beta",
            "session_id": "sess-1",
            "checkpoint_id": "ckpt-2",
            "event_time": "2026-05-18T11:00:00Z",
            "lineage_kind": "inline",
            "source_symbol_id": "sym-b",
            "source_artefact_id": "art-b-py",
            "dest_symbol_id": "sym-c",
            "dest_artefact_id": "art-c-py",
        },
        {
            "relation_id": "rel-chain-3",
            "repo_id": "repo-beta",
            "session_id": "sess-1",
            "checkpoint_id": "ckpt-3",
            "event_time": "2026-05-18T12:00:00Z",
            "lineage_kind": "rename_derived",
            "source_symbol_id": "sym-c",
            "source_artefact_id": "art-c-py",
            "dest_symbol_id": "sym-d",
            "dest_artefact_id": "art-d-py",
        },
    ]
    for row in chain:
        db.insert_symbol_lineage(row)

    # Query from A — should traverse the full chain
    results = db.query_symbol_lineage("repo-beta", "sym-a", depth=5)
    assert len(results) == 3
    assert [r["relation_id"] for r in results] == [
        "rel-chain-1",
        "rel-chain-2",
        "rel-chain-3",
    ]
    # Depth limit respected
    results_limited = db.query_symbol_lineage("repo-beta", "sym-a", depth=1)
    assert len(results_limited) == 1


def test_symbol_lineage_with_minimal_fields(db: EvidenceGraphDB) -> None:
    """Required fields only — source_blob_sha/dest_blob_sha/commit_sha are optional."""
    row = {
        "relation_id": "rel-min",
        "repo_id": "repo-gamma",
        "session_id": "sess-g",
        "checkpoint_id": "ckpt-g",
        "event_time": "2026-05-18T12:00:00Z",
        "lineage_kind": "copy",
        "source_symbol_id": "sym-x",
        "source_artefact_id": "art-x-py",
        "dest_symbol_id": "sym-y",
        "dest_artefact_id": "art-y-py",
    }
    db.insert_symbol_lineage(row)
    results = db.query_symbol_lineage("repo-gamma", "sym-x")
    assert len(results) == 1
    assert results[0]["commit_sha"] == ""


# --------------------------------------------------------------------------- #
# Schema version recording
# --------------------------------------------------------------------------- #

def test_record_schema_version(db: EvidenceGraphDB) -> None:
    schema_text = '{"type": "object", "properties": {"foo": {"type": "string"}}}'
    db.record_schema_version("autobench.result.v1", "v1", schema_text)

    import sqlite3
    with sqlite3.connect(db.path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM schema_versions WHERE channel_type = ?",
            ["autobench.result.v1"],
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["version_id"] == "v1"
    assert rows[0]["schema_hash"] is not None
    assert rows[0]["生效_at"] != ""


def test_record_schema_version_idempotent(db: EvidenceGraphDB) -> None:
    """Same (channel_type, version_id) replaces the old row."""
    schema_v1 = '{"type": "object"}'
    schema_v2 = '{"type": "object", "properties": {"bar": {"type": "integer"}}}'

    db.record_schema_version("autobench.foo.v1", "v1", schema_v1)
    db.record_schema_version("autobench.foo.v1", "v1", schema_v2)

    import sqlite3
    with sqlite3.connect(db.path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT schema_hash FROM schema_versions WHERE channel_type = ? AND version_id = ?",
            ["autobench.foo.v1", "v1"],
        ).fetchall()
    assert len(rows) == 1
    # The newer schema text should have replaced the old one
    import hashlib
    assert rows[0]["schema_hash"] == hashlib.sha256(schema_v2.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Event invalidation on schema change
# --------------------------------------------------------------------------- #

def test_invalidate_events_for_schema_change(db: EvidenceGraphDB) -> None:
    import sqlite3

    # Insert some prior events under the old version
    with sqlite3.connect(db.path) as conn:
        conn.execute("""
            INSERT INTO schema_events
                (event_id, channel_type, schema_version_id, valid, invalidated_reason)
            VALUES ('evt-old-1', 'autobench.test.v1', 'v1', 1, NULL)
        """)
        conn.execute("""
            INSERT INTO schema_events
                (event_id, channel_type, schema_version_id, valid, invalidated_reason)
            VALUES ('evt-old-2', 'autobench.test.v1', 'v1', 1, NULL)
        """)
        conn.execute("""
            INSERT INTO schema_events
                (event_id, channel_type, schema_version_id, valid, invalidated_reason)
            VALUES ('evt-new', 'autobench.test.v1', 'v2', 1, NULL)
        """)

    # Simulate a schema upgrade from v1 -> v2
    count = db.invalidate_events_for_schema_change("autobench.test.v1", "v2")
    assert count == 2

    with sqlite3.connect(db.path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT event_id, valid, invalidated_reason FROM schema_events ORDER BY event_id"
        ).fetchall()

    assert rows[0]["event_id"] == "evt-new"
    assert rows[0]["valid"] == 1
    assert rows[1]["event_id"] == "evt-old-1"
    assert rows[1]["valid"] == 0
    assert "schema_changed_to_v2" in rows[1]["invalidated_reason"]


def test_invalidate_events_only_affects_target_channel(db: EvidenceGraphDB) -> None:
    """Schema change for channel A must not invalidate events for channel B."""
    import sqlite3

    with sqlite3.connect(db.path) as conn:
        conn.execute("""
            INSERT INTO schema_events
                (event_id, channel_type, schema_version_id, valid, invalidated_reason)
            VALUES ('evt-a', 'autobench.channel-a.v1', 'v1', 1, NULL)
        """)
        conn.execute("""
            INSERT INTO schema_events
                (event_id, channel_type, schema_version_id, valid, invalidated_reason)
            VALUES ('evt-b', 'autobench.channel-b.v1', 'v1', 1, NULL)
        """)

    db.invalidate_events_for_schema_change("autobench.channel-a.v1", "v2")

    with sqlite3.connect(db.path) as conn:
        conn.row_factory = sqlite3.Row
        rows = dict(conn.execute(
            "SELECT event_id, valid FROM schema_events"
        ).fetchall())

    assert rows["evt-b"] == 1, "channel-b events should not be invalidated"


# --------------------------------------------------------------------------- #
# DevQL-style query interfaces
# --------------------------------------------------------------------------- #

def test_devql_select_artefacts_by_path(db: EvidenceGraphDB) -> None:
    rows = [
        {
            "relation_id": f"rel-f-{i}",
            "repo_id": "repo-dev",
            "checkpoint_id": f"ckpt-{i}",
            "session_id": "sess-d",
            "event_time": f"2026-05-18T{i:02d}:00:00Z",
            "change_kind": "modify",
            "path_before": f"src/foo{i}.py",
            "path_after": f"src/bar{i}.py",
            "blob_sha_before": f"sha-before-{i}",
            "blob_sha_after": f"sha-after-{i}",
        }
        for i in range(1, 4)
    ]
    for row in rows:
        db.insert_checkpoint_file(row)

    results = db.devql_select_artefacts_by_path("repo-dev", "src/bar2.py")
    assert len(results) == 1
    assert results[0]["checkpoint_id"] == "ckpt-2"


def test_devql_query_lineage_for_symbol(db: EvidenceGraphDB) -> None:
    chain = [
        {
            "relation_id": "rel-d1",
            "repo_id": "repo-devql",
            "session_id": "sess-d1",
            "checkpoint_id": "ckpt-d1",
            "event_time": "2026-05-18T10:00:00Z",
            "lineage_kind": "extract",
            "source_symbol_id": "sym-orig",
            "source_artefact_id": "art-orig-py",
            "dest_symbol_id": "sym-extracted",
            "dest_artefact_id": "art-extracted-py",
        },
        {
            "relation_id": "rel-d2",
            "repo_id": "repo-devql",
            "session_id": "sess-d1",
            "checkpoint_id": "ckpt-d2",
            "event_time": "2026-05-18T11:00:00Z",
            "lineage_kind": "inline",
            "source_symbol_id": "sym-extracted",
            "source_artefact_id": "art-extracted-py",
            "dest_symbol_id": "sym-final",
            "dest_artefact_id": "art-final-py",
        },
    ]
    for row in chain:
        db.insert_symbol_lineage(row)

    results = db.devql_query_lineage_for_symbol("repo-devql", "sym-orig")
    assert len(results) == 2


# --------------------------------------------------------------------------- #
# Checkpoint files insert
# --------------------------------------------------------------------------- #

def test_insert_checkpoint_file_rename(db: EvidenceGraphDB) -> None:
    row = {
        "relation_id": "rel-rename-1",
        "repo_id": "repo-r",
        "checkpoint_id": "ckpt-r",
        "session_id": "sess-r",
        "event_time": "2026-05-18T12:00:00Z",
        "agent": "copilot",
        "change_kind": "rename",
        "path_before": "old/utils.py",
        "path_after": "new/utils.py",
        "blob_sha_before": "sha-old",
        "blob_sha_after": "sha-new",
        "copy_source_path": None,
        "copy_source_blob_sha": None,
        "commit_sha": "deadbeef",
    }
    db.insert_checkpoint_file(row)

    results = db.devql_select_artefacts_by_path("repo-r", "new/utils.py")
    assert len(results) == 1
    assert results[0]["change_kind"] == "rename"
    assert results[0]["blob_sha_after"] == "sha-new"


def test_insert_checkpoint_file_copy(db: EvidenceGraphDB) -> None:
    row = {
        "relation_id": "rel-copy-1",
        "repo_id": "repo-c",
        "checkpoint_id": "ckpt-c",
        "session_id": "sess-c",
        "event_time": "2026-05-18T12:00:00Z",
        "change_kind": "copy",
        "path_before": None,
        "path_after": "lib/widget.py",
        "blob_sha_before": None,
        "blob_sha_after": "sha-copied",
        "copy_source_path": "templates/widget.py",
        "copy_source_blob_sha": "sha-template",
        "commit_sha": "f00d",
    }
    db.insert_checkpoint_file(row)

    with __import__("sqlite3").connect(db.path) as conn:
        conn.row_factory = __import__("sqlite3").Row
        row_back = conn.execute(
            "SELECT * FROM checkpoint_files WHERE relation_id = ?", ["rel-copy-1"]
        ).fetchone()
    assert row_back is not None
    assert row_back["change_kind"] == "copy"
    assert row_back["copy_source_path"] == "templates/widget.py"