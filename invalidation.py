"""
Bitloops-style source_scope_key temporal invalidation for nervous-bus events.

Every event producer registers its scope key format. When a new event arrives
for the same scope, all prior events for that scope are auto-deactivated.

Mirrors: Bitloops storage.rs:237-247 (temporal invalidation) and lifecycle.rs:72-83 (scope key).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Scope key constructors (mirrors Bitloops lifecycle.rs:72-83)
# --------------------------------------------------------------------------- #

def history_source_scope_key(
    session_id: str,
    turn_id: str | None = None,
    checkpoint_id: str | None = None,
) -> str:
    """History (agent session) scope key. Format: history_source:{session}:{turn}:{checkpoint}"""
    return f"history_source:{session_id}:{turn_id or '_'}:{checkpoint_id or '_'}"


def ahe_scope_key(session_id: str, problem_id: str, iteration: int) -> str:
    """AHE prediction scope key. Format: ahe:{session_id}:{problem_id}:{iteration}"""
    return f"ahe:{session_id}:{problem_id}:{iteration}"


def bead_scope_key(bead_id: str, exec_id: str) -> str:
    """AC verification scope key. Format: ac_verify:{bead_id}:{exec_id}"""
    return f"ac_verify:{bead_id}:{exec_id}"


def schema_scope_key(channel_type: str, schema_version_id: str) -> str:
    """Schema version scope key. Format: schema_scope:{channel_type}:{version}"""
    return f"schema_scope:{channel_type}:{schema_version_id}"


def promotion_scope_key(cycle_id: str) -> str:
    """Promotion ledger scope key. Format: promotion:{cycle_id}"""
    return f"promotion:{cycle_id}"


# --------------------------------------------------------------------------- #
# Invalidation store
# --------------------------------------------------------------------------- #

class InvalidationStore:
    """
    Tracks which scope keys have been invalidated and by what.
    Mirrors Bitloops' context_guidance_distillation_runs table pattern.

    File: ~/.cache/nervous-bus/invalidation_store.jsonl
    One line per invalidation event: {scope_key, invalidated_at, reason, count}
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or Path.home() / ".cache" / "nervous-bus" / "invalidation_store.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._invalidated: dict[str, str] = {}  # scope_key -> invalidated_at
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            self._invalidated[entry["scope_key"]] = entry["invalidated_at"]
                        except Exception:
                            continue

    def record_invalidation(self, scope_key: str, reason: str, count: int) -> None:
        """Record that scope_key was invalidated (prior events deactivated)."""
        invalidated_at = datetime.now(timezone.utc).isoformat()
        self._invalidated[scope_key] = invalidated_at
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "scope_key": scope_key,
                "invalidated_at": invalidated_at,
                "reason": reason,
                "count": count,
            }) + "\n")

    def is_invalidated(self, scope_key: str) -> bool:
        """Check if a scope has been invalidated (prior events are stale)."""
        return scope_key in self._invalidated

    def last_invalidation(self, scope_key: str) -> str | None:
        """Return ISO timestamp of last invalidation for scope_key."""
        return self._invalidated.get(scope_key)


# --------------------------------------------------------------------------- #
# Invalidation result
# --------------------------------------------------------------------------- #

@dataclass
class InvalidationResult:
    """Result of a check_and_invalidate call."""
    was_invalidated: bool
    count_deactivated: int
    reason: str | None


# --------------------------------------------------------------------------- #
# Invalidation engine
# --------------------------------------------------------------------------- #

class InvalidationEngine:
    """
    Bitloops-style temporal invalidation engine.

    Consumers call ``check_and_invalidate(scope_key, event_producer)`` before
    inserting new events. If scope_key was already seen, prior events are
    marked stale and a new scope epoch begins.
    """

    def __init__(self) -> None:
        self.store = InvalidationStore()
        # Registry: scope_key_prefix -> (invalidate_fn, event_channel)
        self._registrations: dict[str, tuple[Callable[[str], int], str]] = {}

    def register(
        self,
        scope_key_prefix: str,
        invalidate_fn: Callable[[str], int],
        channel: str,
    ) -> None:
        """Register a scope key prefix and its invalidation function."""
        self._registrations[scope_key_prefix] = (invalidate_fn, channel)

    def check_and_invalidate(
        self,
        scope_key: str,
        event_data: dict[str, object],
    ) -> InvalidationResult:
        """
        Check if scope_key is already known. If so, run invalidation before
        inserting new event. Returns InvalidationResult with count deactivated.
        """
        if self.store.is_invalidated(scope_key):
            # Re-distillation for same scope — invalidate prior events
            count = 0
            for prefix, (invalidate_fn, _channel) in self._registrations.items():
                if scope_key.startswith(prefix):
                    count += invalidate_fn(scope_key)

            self.store.record_invalidation(scope_key, "re-distillation", count)

            return InvalidationResult(
                was_invalidated=True,
                count_deactivated=count,
                reason="re-distillation",
            )

        return InvalidationResult(
            was_invalidated=False,
            count_deactivated=0,
            reason=None,
        )

    def emit_invalidation_event(
        self,
        scope_key: str,
        count: int,
        reason: str,
    ) -> None:
        """Emit bus event for invalidation (for observability)."""
        try:
            from autobench.observability import AutobenchObservability
            obs = AutobenchObservability()
            obs._publish("autobench.invalidation.v1", {
                "scope_key": scope_key,
                "count_deactivated": count,
                "reason": reason,
            })
        except Exception:
            # Observability must never raise
            pass


# --------------------------------------------------------------------------- #
# Module-level shared engine (lazy init)
# --------------------------------------------------------------------------- #

_invalidation_engine: InvalidationEngine | None = None


def get_invalidation_engine() -> InvalidationEngine:
    """Return the shared InvalidationEngine singleton."""
    global _invalidation_engine
    if _invalidation_engine is None:
        _invalidation_engine = InvalidationEngine()
    return _invalidation_engine


# --------------------------------------------------------------------------- #
# Stub invalidation callbacks for external registration
# --------------------------------------------------------------------------- #

def _stub_invalidate_fn(scope_key: str) -> int:
    """Placeholder invalidate function. Replace via register()."""
    return 0


def _register_stub(scope_key_prefix: str, channel: str) -> None:
    """Register a stub invalidation function for a prefix."""
    engine = get_invalidation_engine()
    engine.register(scope_key_prefix, _stub_invalidate_fn, channel)