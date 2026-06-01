"""Shared path helpers for the autobench test suite.

Resolves NBUS_ROOT (the nervous-bus repo root) regardless of whether autobench
is running as a standalone repo or as a submodule inside nervous-bus.

Priority:
  1. NBUS_ROOT env var (explicit override)
  2. Parent directory has schemas/ (submodule context: nervous-bus/autobench/)
  3. nervous-bus/ sibling in the same projects directory (standalone clone)
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT: Path = Path(__file__).resolve().parent.parent  # nervous-autobench/


def _resolve_nbus_root() -> Path:
    if env := os.environ.get("NBUS_ROOT"):
        p = Path(env)
        if (p / "schemas").is_dir():
            return p
    # submodule: nervous-bus/autobench/ → parent is nervous-bus/
    if (_PKG_ROOT.parent / "schemas").is_dir():
        return _PKG_ROOT.parent
    # standalone clone alongside nervous-bus
    sibling = _PKG_ROOT.parent / "nervous-bus"
    if (sibling / "schemas").is_dir():
        return sibling
    raise RuntimeError(
        f"Cannot locate nervous-bus schemas. "
        f"Set NBUS_ROOT=/path/to/nervous-bus or clone nervous-bus next to {_PKG_ROOT.name}."
    )


NBUS_ROOT: Path = _resolve_nbus_root()
SCHEMA_DIR: Path = NBUS_ROOT / "schemas"
PKG_ROOT: Path = _PKG_ROOT
BENCHMARKS_DIR: Path = _PKG_ROOT / "benchmarks"
