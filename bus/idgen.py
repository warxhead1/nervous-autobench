"""Unified ULID + RFC3339 helpers for autobench events.

Phase 2A of the autobench restructuring consolidates the three near-duplicate
ULID helpers (signal_bus._ulid, gpu_types._ulid, integration._generate_ulid)
and the two RFC3339 helpers (signal_bus._iso_now, gpu_types._iso_now,
integration._rfc3339_now) into a single canonical implementation.

Two ULID shapes coexist in this codebase — by design, not by accident:

1. The 10-char seconds-since-epoch timestamp + 16-char hex random shape used
   by signal_bus.py and gpu_types.py. This matches the upstream nervous shell
   SDK's "ulid" format. *All* new bus events should use this format.

2. The 13-char millis-since-epoch timestamp + 15-char random shape used by
   integration.NervousBusPublisher._generate_ulid. This predates the
   consolidation; downstream consumers (deer-flow, the nervous-bus redis
   mirror) may parse the field by *position*, so changing its format would
   silently break those parsers. The integration helper is intentionally
   left as-is in its private module — do NOT replace it with `ulid()` here.

If you need a fresh bus event, import `ulid` and `iso_now` from this module.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone


def ulid() -> str:
    """Generate a 26-char ULID-shaped identifier.

    10-char s timestamp + 16-char hex random. Matches the format previously
    produced by ``signal_bus._ulid()`` and ``gpu_types._ulid()`` and the
    upstream nervous shell SDK.
    """
    ts = int(time.time())
    hex_chars = "0123456789ABCDEF"
    rand_part = "".join(random.choice(hex_chars) for _ in range(16))
    return f"{ts:010d}{rand_part}"


def iso_now() -> str:
    """Return current UTC time as RFC3339 with explicit timezone offset.

    Replaces the two prior implementations:
    - ``signal_bus._iso_now()`` and ``gpu_types._iso_now()`` used
      ``time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())``
      (UTC trailing "Z" suffix, no sub-second precision).
    - ``integration._rfc3339_now()`` used
      ``datetime.now(timezone.utc).isoformat()`` (timezone-aware, sub-second
      precision if available).

    The datetime-based form is the canonical choice here: it's RFC3339
    compliant in either case, it carries timezone information explicitly
    (avoiding the ambiguous "Z" suffix), and it gives sub-second precision
    for free. Both old call sites in signal_bus and gpu_types are updated
    to use this helper.
    """
    return datetime.now(timezone.utc).isoformat()
