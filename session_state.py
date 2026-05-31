"""Session state tracking for autobench harness runs.

Provides session lifecycle management with ULID-based identification,
RFC3339 timestamps, and status tracking for deer-flow integration.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional


# ULID pattern: 26 alphanumeric characters
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def is_valid_ulid(value: str) -> bool:
    """Check if a string is a valid ULID."""
    return bool(_ULID_RE.match(value))


def _base32_char(c: int) -> str:
    """Return Crockford base32 character for a value 0-31."""
    return "0123456789ABCDEFGHJKMNPQRSTVWXYZ"[c & 31]


def generate_ulid() -> str:
    """Generate a ULID-like identifier (26 characters).

    Uses timestamp + random to produce a sortable, monotonic ID.
    Uses Crockford base32 encoding (no I, L, O, U).
    """
    import random
    import time

    timestamp_ms = int(time.time() * 1000)
    random_bits = random.getrandbits(80)

    # Encode timestamp (48 bits) + random (80 bits) = 128 bits total
    # 26 characters at 5 bits each = 130 bits capacity
    result = []
    value = (timestamp_ms << 80) | random_bits
    for _ in range(26):
        result.append(_base32_char(value & 31))
        value >>= 5

    return "".join(reversed(result))


def rfc3339_now() -> str:
    """Return current time in RFC3339 format (UTC)."""
    return datetime.now(timezone.utc).isoformat()


def parse_rfc3339(value: str) -> datetime:
    """Parse an RFC3339 timestamp string to datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


SessionStatus = Literal["active", "completed", "failed", "timed_out"]


@dataclass
class SessionState:
    """Represents the state of an autobench harness session.

    Attributes:
        session_id: Unique identifier (ULID format, sortable and monotonic).
        started_at: RFC3339 timestamp when session started.
        terminated_at: RFC3339 timestamp when session ended (None if active).
        termination_reason: Human-readable reason for termination.
        status: Current session status.
    """

    session_id: str
    started_at: str
    terminated_at: Optional[str] = None
    termination_reason: Optional[str] = None
    status: SessionStatus = "active"

    def __post_init__(self) -> None:
        if not is_valid_ulid(self.session_id):
            raise ValueError(f"Invalid ULID: {self.session_id!r}")
        if self.status not in ("active", "completed", "failed", "timed_out"):
            raise ValueError(f"Invalid status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON encoding."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "terminated_at": self.terminated_at,
            "termination_reason": self.termination_reason,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Deserialize from a dictionary (e.g., after JSON decoding)."""
        return cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            terminated_at=data.get("terminated_at"),
            termination_reason=data.get("termination_reason"),
            status=data.get("status", "active"),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> SessionState:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))

    def is_terminal(self) -> bool:
        """Return True if session has reached a terminal state."""
        return self.status in ("completed", "failed", "timed_out")

    def duration_seconds(self) -> Optional[float]:
        """Return session duration in seconds, or None if still active."""
        if self.terminated_at is None:
            return None
        start = parse_rfc3339(self.started_at)
        end = parse_rfc3339(self.terminated_at)
        return (end - start).total_seconds()


def start_session() -> SessionState:
    """Create a new session in the 'active' state."""
    return SessionState(
        session_id=generate_ulid(),
        started_at=rfc3339_now(),
        status="active",
    )


def finish_session(state: SessionState, reason: str) -> SessionState:
    """Mark a session as finished with a termination reason.

    Args:
        state: The current session state.
        reason: Human-readable termination reason.

    Returns:
        A new SessionState with terminated_at and status populated.
        The status is set based on common patterns in the reason string.
    """
    terminal_status: SessionStatus
    reason_lower = reason.lower()

    if "timeout" in reason_lower or "timed out" in reason_lower:
        terminal_status = "timed_out"
    elif "failed" in reason_lower or "error" in reason_lower:
        terminal_status = "failed"
    else:
        terminal_status = "completed"

    return SessionState(
        session_id=state.session_id,
        started_at=state.started_at,
        terminated_at=rfc3339_now(),
        termination_reason=reason,
        status=terminal_status,
    )


def is_session_complete(state: SessionState) -> bool:
    """Return True if the session has reached a terminal state."""
    return state.is_terminal()