"""
Bitloops-style evidence graph schema for nervous-bus.
Mirrors: checkpoint_sqlite_schema.rs → checkpoint_artefact_lineage table.

Tables:
  - symbol_evidence_lineage: tracks symbol evolution across checkpoints
  - checkpoint_files: tracks file-level changes with blob_sha
  - schema_versions: tracks schema version history
  - schema_events: marks events as valid/invalid based on schema version
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
-- Symbol lineage: mirrors Bitloops checkpoint_artefact_lineage
CREATE TABLE IF NOT EXISTS symbol_evidence_lineage (
    relation_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    event_time TEXT NOT NULL,
    agent TEXT DEFAULT '',
    lineage_kind TEXT NOT NULL,        -- 'refactor' | 'extract' | 'inline' | 'rename_derived' | 'copy'
    source_symbol_id TEXT NOT NULL,
    source_artefact_id TEXT NOT NULL,
    source_blob_sha TEXT,
    dest_symbol_id TEXT NOT NULL,
    dest_artefact_id TEXT NOT NULL,
    dest_blob_sha TEXT,
    commit_sha TEXT
);

CREATE INDEX IF NOT EXISTS symbol_lineage_source_idx ON symbol_evidence_lineage(repo_id, source_artefact_id);
CREATE INDEX IF NOT EXISTS symbol_lineage_dest_idx ON symbol_evidence_lineage(repo_id, dest_artefact_id);
CREATE INDEX IF NOT EXISTS symbol_lineage_checkpoint_idx ON symbol_evidence_lineage(repo_id, checkpoint_id);
CREATE INDEX IF NOT EXISTS symbol_lineage_event_time_idx ON symbol_evidence_lineage(repo_id, event_time);

-- File changes: mirrors Bitloops checkpoint_files
CREATE TABLE IF NOT EXISTS checkpoint_files (
    relation_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_time TEXT NOT NULL,
    agent TEXT DEFAULT '',
    change_kind TEXT NOT NULL,          -- 'add' | 'modify' | 'delete' | 'rename' | 'copy'
    path_before TEXT,
    path_after TEXT,
    blob_sha_before TEXT,
    blob_sha_after TEXT,
    copy_source_path TEXT,
    copy_source_blob_sha TEXT,
    commit_sha TEXT
);

CREATE INDEX IF NOT EXISTS checkpoint_files_lookup_idx ON checkpoint_files(repo_id, path_after, blob_sha_after);
CREATE INDEX IF NOT EXISTS checkpoint_files_commit_idx ON checkpoint_files(repo_id, commit_sha);

-- Schema version tracking (Bitloops-style source_scope_key for schemas)
CREATE TABLE IF NOT EXISTS schema_versions (
    channel_type TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    version_id TEXT NOT NULL,
    生效_at TEXT NOT NULL,
    PRIMARY KEY (channel_type, version_id)
);

-- Event validity tracking
CREATE TABLE IF NOT EXISTS schema_events (
    event_id TEXT NOT NULL,
    channel_type TEXT NOT NULL,
    schema_version_id TEXT NOT NULL,
    valid INTEGER NOT NULL DEFAULT 1,
    invalidated_reason TEXT,
    PRIMARY KEY (event_id)
);
"""


class EvidenceGraphDB:
    """Bitloops-style evidence graph store."""

    def __init__(self, path: str = "/home/eric/.cache/nervous-bus/evidence_graph.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA_SQL)

    def insert_symbol_lineage(self, row: dict[str, Any]) -> None:
        """Insert a symbol_evidence_lineage row."""
        sql = """
        INSERT INTO symbol_evidence_lineage
        (relation_id, repo_id, session_id, checkpoint_id, event_time, agent,
         lineage_kind, source_symbol_id, source_artefact_id, source_blob_sha,
         dest_symbol_id, dest_artefact_id, dest_blob_sha, commit_sha)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with sqlite3.connect(self.path) as conn:
            conn.execute(sql, [
                row["relation_id"], row["repo_id"], row["session_id"],
                row["checkpoint_id"], row["event_time"], row.get("agent", ""),
                row["lineage_kind"], row["source_symbol_id"], row["source_artefact_id"],
                row.get("source_blob_sha", ""), row["dest_symbol_id"], row["dest_artefact_id"],
                row.get("dest_blob_sha", ""), row.get("commit_sha", ""),
            ])

    def insert_checkpoint_file(self, row: dict[str, Any]) -> None:
        """Insert a checkpoint_files row."""
        sql = """
        INSERT INTO checkpoint_files
        (relation_id, repo_id, checkpoint_id, session_id, event_time, agent,
         change_kind, path_before, path_after, blob_sha_before, blob_sha_after,
         copy_source_path, copy_source_blob_sha, commit_sha)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with sqlite3.connect(self.path) as conn:
            conn.execute(sql, [
                row["relation_id"], row["repo_id"], row["checkpoint_id"],
                row["session_id"], row["event_time"], row.get("agent", ""),
                row["change_kind"], row.get("path_before"), row.get("path_after"),
                row.get("blob_sha_before", ""), row.get("blob_sha_after", ""),
                row.get("copy_source_path"), row.get("copy_source_blob_sha", ""),
                row.get("commit_sha", ""),
            ])

    def query_symbol_lineage(
        self, repo_id: str, symbol_id: str, depth: int = 3
    ) -> list[dict[str, Any]]:
        """Query symbol lineage: find all evolutionary ancestors/descendants of symbol_id."""
        sql = """
        WITH RECURSIVE lineage_trace AS (
            SELECT relation_id, source_symbol_id, dest_symbol_id,
                   source_artefact_id, dest_artefact_id, checkpoint_id,
                   event_time, lineage_kind, commit_sha, 0 as depth
            FROM symbol_evidence_lineage
            WHERE repo_id = ? AND (source_symbol_id = ? OR dest_symbol_id = ?)

            UNION ALL

            SELECT e.relation_id, e.source_symbol_id, e.dest_symbol_id,
                   e.source_artefact_id, e.dest_artefact_id, e.checkpoint_id,
                   e.event_time, e.lineage_kind, e.commit_sha, lt.depth + 1
            FROM symbol_evidence_lineage e
            JOIN lineage_trace lt ON lt.dest_symbol_id = e.source_symbol_id
                                 AND lt.dest_artefact_id = e.source_artefact_id
            WHERE e.repo_id = ? AND lt.depth + 1 < ?
        )
        SELECT * FROM lineage_trace ORDER BY depth, event_time;
        """
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, [repo_id, symbol_id, symbol_id, repo_id, depth]).fetchall()
            return [dict(r) for r in rows]

    def record_schema_version(self, channel_type: str, version_id: str, schema_text: str) -> None:
        """Record a schema version and compute its hash."""
        schema_hash = hashlib.sha256(schema_text.encode()).hexdigest()
        生效_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO schema_versions (channel_type, schema_hash, version_id, 生效_at)
                VALUES (?, ?, ?, ?)
            """, [channel_type, schema_hash, version_id, 生效_at])

    def invalidate_events_for_schema_change(self, channel_type: str, new_version_id: str) -> int:
        """When a schema changes, invalidate all prior events for this channel."""
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute("""
                UPDATE schema_events
                SET valid = 0, invalidated_reason = ?
                WHERE channel_type = ? AND schema_version_id != ? AND valid = 1
            """, [f"schema_changed_to_{new_version_id}", channel_type, new_version_id])
            return cursor.rowcount

    def devql_select_artefacts_by_path(self, repo_id: str, path: str) -> list[dict[str, Any]]:
        """DevQL equivalent: repo('...') -> selectArtefacts(path: '...') -> checkpoints()"""
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT DISTINCT checkpoint_id, session_id, event_time, change_kind,
                       path_before, path_after, blob_sha_before, blob_sha_after
                FROM checkpoint_files
                WHERE repo_id = ? AND (path_before = ? OR path_after = ?)
                ORDER BY event_time DESC
            """, [repo_id, path, path]).fetchall()
            return [dict(r) for r in rows]

    def devql_query_lineage_for_symbol(
        self, repo_id: str, symbol_fqn: str, depth: int = 5
    ) -> list[dict[str, Any]]:
        """DevQL equivalent: repo('...') -> selectArtefacts(symbol_fqn: '...') -> artefact_lineage"""
        return self.query_symbol_lineage(repo_id, symbol_fqn, depth=depth)