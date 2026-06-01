"""autobench.daemons — long-running supervisor and event-driven trigger processes.

Phase 2C of the autobench restructuring. Two sibling modules that used to
live at the autobench package root are regrouped here: continuous (24/7
self-improvement supervisor) and trigger_daemon (event-driven consumer).
Public re-exports are listed in ``__all__``.
"""

from __future__ import annotations

# continuous.py — long-running autonomous self-improvement supervisor
from .continuous import (
    ContinuousModeDaemon,
    Digest,
    PromotionDecision,
    Surprise,
    SurpriseDigest,
)

# trigger_daemon.py — event-driven cycle trigger consumer
from .trigger_daemon import (
    CycleConfig,
    TriggerDaemon,
    build_cycle_config,
)

__all__ = [
    "ContinuousModeDaemon",
    "CycleConfig",
    "Digest",
    "PromotionDecision",
    "Surprise",
    "SurpriseDigest",
    "TriggerDaemon",
    "build_cycle_config",
]
