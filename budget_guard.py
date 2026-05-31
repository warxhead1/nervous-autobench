"""Budget guard for autobench RSI sessions.

A safety net that halts a long-running session when cumulative LLM cost or
wall-clock time exceeds an operator-set cap. Emits `autobench.budget.warning.v1`
events at 50% / 80% / 100% threshold crossings so the pulse dashboard can show
the cap creeping up in real time.

Usage:
    guard = BudgetGuard(max_cost_dollars=1.00, max_wall_time_seconds=1800)
    # Inside the RSI loop:
    guard.record_cost(0.012)             # cents from one improver call
    guard.record_iteration_complete()    # checkpoint
    ok, reason = guard.check()
    if not ok:
        guard.halt(reason)               # raises BudgetExceeded

Integration: ``SelfImprovingHarness`` accepts an optional ``budget_guard`` field;
when set, the loop calls ``record_iteration_complete()`` after each iteration and
``halt()`` if ``check()`` returns False.

Cost source: the guard does NOT subscribe to the bus directly — that would
couple it to async event delivery. Instead, callers feed it costs via
``record_cost()``. The wrapper (e.g. ``MiniMaxLLMWrapper``) reports
``cost_dollars`` on every call, and ``SelfImprovingHarness.improve()`` forwards
that into the guard.

Source of bus events: ``autobench.budget.warning.v1`` (schema in
``schemas/autobench.budget.warning.v1.json``).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class BudgetExceeded(RuntimeError):
    """Raised by :meth:`BudgetGuard.halt` when the budget cap is reached."""


class RateBudgetExceeded(RuntimeError):
    """Raised by :meth:`RateBudgetGuard.halt` when the request-rate cap is reached."""


# --------------------------------------------------------------------------- #
# Constants — match observability.py for emission compatibility
# --------------------------------------------------------------------------- #

_CHANNEL_WARNING = "autobench.budget.warning.v1"
_CHANNEL_RATE = "autobench.budget.rate.v1"
_SOURCE = "/autobench"
_THRESHOLDS = (0.5, 0.8, 1.0)

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_DEBUG_CACHE = Path.home() / ".cache" / "nervous-bus"
_DEBUG_FILE = _DEBUG_CACHE / "debug.jsonl"

# Module-level zellij pipe state. Once a probe fails (timeout or non-zero
# exit), all subsequent emits skip straight to the debug file. Mirrors the
# probe-once-and-cache pattern in observability.py (commit 4c9d56f) so a
# guard never blocks the RSI loop more than once when zellij is unreachable
# or the nervous-bus plugin isn't loaded.
import shutil as _shutil  # noqa: E402

_PIPE_DISABLED: bool = (
    os.environ.get("AUTOBENCH_OBS_DISABLE_PIPE", "").lower() in {"1", "true", "yes"}
    or _shutil.which("zellij") is None
)


def _ulid() -> str:
    ts_ms = int(time.time() * 1000)
    time_part = ""
    n = ts_ms
    for _ in range(10):
        time_part = _CROCKFORD[n & 0x1F] + time_part
        n >>= 5
    rand_part = "".join(random.choice(_CROCKFORD) for _ in range(16))
    return time_part + rand_part


def _iso_now() -> str:
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


# --------------------------------------------------------------------------- #
# BudgetGuard
# --------------------------------------------------------------------------- #


@dataclass
class BudgetGuard:
    """Cost + wall-time safety net for one autobench session.

    Args:
        max_cost_dollars: Cumulative-cost cap (USD).
        max_wall_time_seconds: Wall-clock cap from construction time.
        on_exceed: ``"halt"`` (raise) or ``"warn"`` (emit warning only).
        session_id: Optional ULID for bus correlation. Auto-generated when None.
        publisher: Optional callable ``(channel, payload_dict) -> None`` for
            unit tests. When None, uses the default zellij-pipe + debug-file
            fallback (mirrors :class:`AutobenchObservability`).

    Attributes are mutated in-place by ``record_cost`` /
    ``record_iteration_complete``. The guard is single-session — make a fresh
    one per RSI run.
    """

    max_cost_dollars: float
    max_wall_time_seconds: float = 3600.0
    on_exceed: str = "halt"
    session_id: str = field(default_factory=_ulid)
    publisher: Any = None  # callable(channel: str, payload: dict) -> None

    current_cost_dollars: float = field(default=0.0, init=False)
    iterations_completed: int = field(default=0, init=False)
    _start_time: float = field(init=False)
    _thresholds_fired: set[float] = field(default_factory=set, init=False)
    _halted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def record_cost(self, usd: float) -> None:
        """Accumulate cost from an improver call."""
        if usd is None or usd < 0:
            return
        self.current_cost_dollars += float(usd)
        self._maybe_emit_threshold()

    def record_iteration_complete(self) -> None:
        """Checkpoint after a full RSI iteration."""
        self.iterations_completed += 1
        self._maybe_emit_threshold()

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def elapsed_wall_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def cost_fraction(self) -> float:
        if self.max_cost_dollars <= 0:
            return 0.0
        return self.current_cost_dollars / self.max_cost_dollars

    def wall_fraction(self) -> float:
        if self.max_wall_time_seconds <= 0:
            return 0.0
        return self.elapsed_wall_seconds() / self.max_wall_time_seconds

    def fraction_used(self) -> float:
        """Largest of cost-fraction and wall-fraction."""
        return max(self.cost_fraction(), self.wall_fraction())

    def check(self) -> tuple[bool, str | None]:
        """Return ``(ok, reason_if_exceeded)``.

        ``ok=False`` when either cost or wall-time fully consumed.

        nervous-bus-dq7l: ``max_cost_dollars <= 0`` means the dollar guard
        is DISABLED. Callers on the MiniMax coding plan should set the cap
        to 0 and rely on ``RateBudgetGuard`` for the real billing unit
        (requests-per-5h). Wall-time still applies as a safety net.
        """
        if self.max_cost_dollars > 0 and self.current_cost_dollars >= self.max_cost_dollars:
            return False, (
                f"cost {self.current_cost_dollars:.4f} USD "
                f"reached cap {self.max_cost_dollars:.4f} USD"
            )
        if self.elapsed_wall_seconds() >= self.max_wall_time_seconds:
            return False, (
                f"wall-time {self.elapsed_wall_seconds():.1f}s "
                f"reached cap {self.max_wall_time_seconds:.1f}s"
            )
        return True, None

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def halt(self, reason: str) -> None:
        """Emit a halt event and raise :class:`BudgetExceeded`.

        Idempotent — repeated calls still raise but only emit the first time.
        """
        if not self._halted:
            self._halted = True
            self._emit_warning(threshold=1.0, action="halt", reason=reason)
        raise BudgetExceeded(reason)

    # ------------------------------------------------------------------ #
    # Threshold emission
    # ------------------------------------------------------------------ #

    def _maybe_emit_threshold(self) -> None:
        frac = self.fraction_used()
        for t in _THRESHOLDS:
            if frac >= t and t not in self._thresholds_fired:
                self._thresholds_fired.add(t)
                action = "halt" if t >= 1.0 else "warning"
                reason = (
                    f"crossed {int(t * 100)}% of budget "
                    f"(cost={self.current_cost_dollars:.4f}/{self.max_cost_dollars:.4f} USD, "
                    f"wall={self.elapsed_wall_seconds():.1f}/{self.max_wall_time_seconds:.1f}s)"
                )
                self._emit_warning(threshold=t, action=action, reason=reason)

    def _emit_warning(self, threshold: float, action: str, reason: str) -> None:
        payload = {
            "session_id": self.session_id,
            "fraction_used": self.fraction_used(),
            "current_cost_usd": self.current_cost_dollars,
            "max_cost_usd": self.max_cost_dollars,
            "elapsed_wall_seconds": self.elapsed_wall_seconds(),
            "max_wall_seconds": self.max_wall_time_seconds,
            "threshold": threshold,
            "action": action,
            "reason": reason,
        }
        try:
            if self.publisher is not None:
                self.publisher(_CHANNEL_WARNING, payload)
            else:
                _default_publish(_CHANNEL_WARNING, payload, self.session_id)
        except Exception as e:  # never let observability crash the run
            print(f"[budget_guard] emit failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Default publisher (mirrors AutobenchObservability._publish)
# --------------------------------------------------------------------------- #


def _default_publish(channel: str, data: dict[str, Any], session_id: str) -> None:
    global _PIPE_DISABLED
    envelope = {
        "specversion": "1.0",
        "id": _ulid(),
        "source": _SOURCE,
        "type": channel,
        "datacontenttype": "application/json",
        "time": _iso_now(),
        "data": data,
    }
    payload = json.dumps(envelope, default=str)

    # Try zellij pipe first (short 0.5s probe), fall back to debug file.
    # After any probe failure, cache the disabled state to avoid N×timeout
    # pathology that would otherwise drift the RateBudgetGuard sliding window.
    if not _PIPE_DISABLED:
        try:
            proc = subprocess.run(
                ["zellij", "pipe", "-p", "nervous-bus", "-n", channel, "--"],
                input=payload.encode(),
                timeout=0.5,
                capture_output=True,
            )
            if proc.returncode == 0:
                return
            _PIPE_DISABLED = True
        except Exception:
            _PIPE_DISABLED = True

    try:
        _DEBUG_CACHE.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_FILE, "a") as fh:
            fh.write(payload + "\n")
    except Exception as e:
        print(f"[budget_guard] debug-file write failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# RateBudgetGuard — sliding-window request-rate safety net
# --------------------------------------------------------------------------- #


@dataclass
class RateBudgetGuard:
    """Sliding-window request-rate cap.

    Designed for plans like MiniMax coding (15,000 requests / 5h = 3,000 req/h).
    Under those plans the dollar cost is $0; the binding constraint is request
    count within a rolling window. This guard counts timestamps, not money.

    Args:
        max_requests: Maximum requests allowed in the window *after* safety
            margin is applied. The plan's hard cap (e.g. 15000) is multiplied
            by ``(1 - safety_margin)`` so the guard halts before the wall.
        window_seconds: Sliding-window length in seconds. 18000 = 5h.
        safety_margin: Fraction (0..1) to deduct from ``max_requests``. With
            default 0.05 and ``max_requests=15000``, the effective cap becomes
            14250.
        on_exceed: ``"halt"`` (raise) or ``"warn"`` (emit warning only).
        session_id: Optional ULID for bus correlation. Auto-generated when None.
        publisher: Optional callable ``(channel, payload_dict) -> None`` for
            unit tests. When None, uses the default zellij-pipe + debug-file
            fallback (mirrors :class:`AutobenchObservability`).
        clock: Optional callable returning current time in seconds; injected
            for deterministic tests. Defaults to :func:`time.monotonic`.
    """

    max_requests: int = 15000
    window_seconds: float = 18000.0
    safety_margin: float = 0.05
    on_exceed: str = "halt"
    session_id: str = field(default_factory=_ulid)
    publisher: Any = None
    clock: Any = None  # callable() -> float

    _timestamps: "deque[float]" = field(default_factory=deque, init=False)
    _thresholds_fired: set[float] = field(default_factory=set, init=False)
    _effective_max: int = field(init=False, default=0)
    _halted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = time.monotonic
        # Apply safety margin once at construction time.
        margin = max(0.0, min(1.0, float(self.safety_margin)))
        self._effective_max = max(1, int(self.max_requests * (1.0 - margin)))

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _prune(self) -> None:
        """Drop timestamps outside the sliding window."""
        cutoff = self.clock() - self.window_seconds
        ts = self._timestamps
        while ts and ts[0] <= cutoff:
            ts.popleft()

    def record_request(self) -> None:
        """Record one request at the current clock time."""
        now = self.clock()
        self._timestamps.append(now)
        self._prune()
        self._maybe_emit_threshold()

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def current_count(self) -> int:
        """Number of requests within the sliding window."""
        self._prune()
        return len(self._timestamps)

    def available(self) -> int:
        """Remaining request slots until the effective cap is reached."""
        return max(0, self._effective_max - self.current_count())

    def fraction_used(self) -> float:
        if self._effective_max <= 0:
            return 0.0
        return self.current_count() / self._effective_max

    def time_until_available(self) -> float:
        """Seconds until the oldest in-window timestamp leaves the window.

        Returns 0.0 when under the cap (a slot is already available).
        """
        if self.current_count() < self._effective_max:
            return 0.0
        if not self._timestamps:
            return 0.0
        oldest = self._timestamps[0]
        elapsed = self.clock() - oldest
        remaining = self.window_seconds - elapsed
        return max(0.0, remaining)

    def check(self) -> tuple[bool, str | None]:
        """Return ``(ok, reason_if_exceeded)``.

        ``ok=False`` once the current count reaches the effective cap.
        """
        count = self.current_count()
        if count >= self._effective_max:
            return False, (
                f"rate {count}/{self._effective_max} requests in last "
                f"{self.window_seconds:.0f}s reached cap"
                + (
                    f" (margin {self.safety_margin:.2f}, raw cap {self.max_requests})"
                    if self.safety_margin > 0
                    else ""
                )
            )
        return True, None

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def halt(self, reason: str) -> None:
        """Emit a halt event and raise :class:`RateBudgetExceeded`."""
        if not self._halted:
            self._halted = True
            self._emit_rate(threshold=1.0, action="halt", reason=reason)
        raise RateBudgetExceeded(reason)

    # ------------------------------------------------------------------ #
    # Threshold emission
    # ------------------------------------------------------------------ #

    def _maybe_emit_threshold(self) -> None:
        frac = self.fraction_used()
        for t in _THRESHOLDS:
            if frac >= t and t not in self._thresholds_fired:
                self._thresholds_fired.add(t)
                action = "halt" if t >= 1.0 else "warning"
                reason = (
                    f"crossed {int(t * 100)}% of request-rate budget "
                    f"({self.current_count()}/{self._effective_max} in "
                    f"{self.window_seconds:.0f}s window)"
                )
                self._emit_rate(threshold=t, action=action, reason=reason)

    def _emit_rate(self, threshold: float, action: str, reason: str) -> None:
        payload = {
            "session_id": self.session_id,
            "current_count": self.current_count(),
            "max_requests": self._effective_max,
            "window_seconds": self.window_seconds,
            "fraction_used": self.fraction_used(),
            "time_until_available": self.time_until_available(),
            "action": action,
            "warned_at_threshold": threshold,
            "reason": reason,
        }
        try:
            if self.publisher is not None:
                self.publisher(_CHANNEL_RATE, payload)
            else:
                _default_publish(_CHANNEL_RATE, payload, self.session_id)
        except Exception as e:  # never let observability crash the run
            print(f"[budget_guard] rate emit failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# CompositeBudgetGuard — combines dollar-cost and request-rate guards
# --------------------------------------------------------------------------- #


@dataclass
class CompositeBudgetGuard:
    """Wraps a :class:`BudgetGuard` and a :class:`RateBudgetGuard`.

    Used when both a dollar cap AND a request-rate cap are in effect — under
    free-tier plans (MiniMax coding) the rate is the binding constraint while
    the dollar guard remains as a defensive secondary check.

    ``check()`` returns the first violation found: the binding constraint
    matters. ``record_request(cost_usd)`` forwards to both inner guards in
    one call.
    """

    cost_guard: BudgetGuard
    rate_guard: RateBudgetGuard

    def record_request(self, cost_usd: float = 0.0) -> None:
        """Record one improver call: increments rate counter, accumulates cost."""
        self.rate_guard.record_request()
        self.cost_guard.record_cost(cost_usd)

    def record_iteration_complete(self) -> None:
        """Forward to the dollar guard so wall-clock thresholds still fire."""
        self.cost_guard.record_iteration_complete()

    def check(self) -> tuple[bool, str | None]:
        """Return the first violation found across the two inner guards.

        Rate is checked first because, under the MiniMax coding plan, requests
        are the binding constraint.
        """
        ok, reason = self.rate_guard.check()
        if not ok:
            return False, reason
        ok, reason = self.cost_guard.check()
        if not ok:
            return False, reason
        return True, None

    def halt(self, reason: str) -> None:
        """Halt with the matching exception type based on which guard tripped."""
        # Determine which guard tripped so the right exception type is raised.
        ok_rate, _ = self.rate_guard.check()
        if not ok_rate:
            self.rate_guard.halt(reason)
        else:
            self.cost_guard.halt(reason)


__all__ = [
    "BudgetGuard",
    "BudgetExceeded",
    "RateBudgetGuard",
    "RateBudgetExceeded",
    "CompositeBudgetGuard",
]
