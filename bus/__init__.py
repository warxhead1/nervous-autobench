"""autobench.bus — nervous-bus integration: publishers, subscribers, helpers.

Phase 2A of the autobench restructuring. This subpackage consolidates the
four bus-adjacent modules that used to live as sibling modules at the
autobench package root, plus two new tiny helper modules:

    autobench/bus/
    ├── idgen.py        # NEW: unified ulid() + iso_now() helpers
    ├── envelope.py     # NEW: unified build_event() (CloudEvents-lite)
    ├── signal_bus.py   # AutobenchResultPublisher + 2 subscribers
    ├── gpu_types.py    # GPUJob / GPUResult dataclasses
    ├── gpu_publisher.py # GPUResultPublisher
    └── integration.py  # RoleSpec, BudgetViolation, NervousBusPublisher, …

Public re-exports:

    from autobench.bus import (
        # signal_bus.py
        AutobenchResultPublisher, AutobenchResultSubscriber, DeerFlowResultSubscriber,
        make_publisher, make_subscriber, make_deerflow_subscriber,
        # gpu_types.py + gpu_publisher.py
        GPUJob, GPUResult, GPUResultPublisher,
        # integration.py
        RoleSpec, BudgetViolation, BudgetViolationMiddleware,
        DeerFlowEvaluator, NervousBusPublisher,
        # helpers (Phase 2A NEW)
        ulid, iso_now, build_event,
    )
"""

from __future__ import annotations

# signal_bus.py — zellij-pipe publishers + JSONL-fallback subscribers
from .signal_bus import (
    AutobenchResultPublisher,
    AutobenchResultSubscriber,
    DeerFlowResultSubscriber,
    make_deerflow_subscriber,
    make_publisher,
    make_subscriber,
)

# gpu_types.py — dataclasses
from .gpu_types import GPUJob, GPUResult

# gpu_publisher.py — GPU result publisher
from .gpu_publisher import GPUResultPublisher

# integration.py — role routing, budget middleware, deer-flow glue, nervous-bus publisher
from .integration import (
    BudgetViolation,
    BudgetViolationMiddleware,
    DeerFlowEvaluator,
    NervousBusPublisher,
    RoleSpec,
)

# Phase 2A NEW: unified id + timestamp + envelope helpers
from .envelope import build_event
from .idgen import iso_now, ulid

__all__ = [
    "AutobenchResultPublisher",
    "AutobenchResultSubscriber",
    "BudgetViolation",
    "BudgetViolationMiddleware",
    "DeerFlowEvaluator",
    "DeerFlowResultSubscriber",
    "GPUJob",
    "GPUResult",
    "GPUResultPublisher",
    "NervousBusPublisher",
    "RoleSpec",
    "build_event",
    "iso_now",
    "make_deerflow_subscriber",
    "make_publisher",
    "make_subscriber",
    "ulid",
]
