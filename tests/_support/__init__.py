"""Test-only support modules — moved from autobench root in Phase 3.

These modules are not part of the production package. They live under
``tests/_support/`` so ``pytest tests/`` is the canonical entry point and
``pyproject.toml`` can exclude them from the wheel via the existing
``tests*`` exclude pattern.
"""
