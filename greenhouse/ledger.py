"""ledger — persistent sliding-window request ledger for the greenhouse.

Cycles run as separate oneshot processes (systemd timer), so budget state
cannot live in memory between runs — every request spend is appended to a
durable JSONL ledger and the sliding window is recomputed by filtering on
timestamp each time.

Deliberately NOT ``autobench.audit.budget_guard.RateBudgetGuard``: that guard
keeps its window in an in-memory ``deque`` keyed off ``time.monotonic()``,
which does not survive process exit and therefore cannot bound spend across
a multi-invocation background scheduler. The window math here mirrors its
semantics (strict ``ts > cutoff`` retention) so the two stay conceptually
compatible.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LEDGER_PATH = Path.home() / ".cache" / "nervous-bus" / "greenhouse" / "ledger.jsonl"


@dataclass
class Ledger:
    path: Path = field(default_factory=lambda: DEFAULT_LEDGER_PATH)

    def record(self, *, run_id: str, goal_id: str, requests: int, ts: float | None = None) -> None:
        """Append one spend entry. No-op for non-positive request counts."""
        if requests <= 0:
            return
        entry = {
            "ts": ts if ts is not None else time.time(),
            "run_id": run_id,
            "goal_id": goal_id,
            "requests": int(requests),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _entries_since(self, cutoff: float) -> list[dict]:
        if not self.path.is_file():
            return []
        out: list[dict] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a corrupted trailing line, don't crash the cycle
            if entry.get("ts", 0) > cutoff:
                out.append(entry)
        return out

    def window_used(self, window_seconds: float, *, now: float | None = None) -> int:
        """Requests spent strictly after (now - window_seconds)."""
        now = now if now is not None else time.time()
        cutoff = now - window_seconds
        return sum(int(e.get("requests", 0)) for e in self._entries_since(cutoff))

    def remaining(self, window_max_requests: int, window_seconds: float, *, now: float | None = None) -> int:
        """Requests still available in the window, floored at 0."""
        used = self.window_used(window_seconds, now=now)
        return max(0, window_max_requests - used)
