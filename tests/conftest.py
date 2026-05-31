"""Pytest configuration for autobench tests."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_tools: integration test that needs external CLI tools "
        "(ast-grep, difft); auto-skipped when binaries are absent.",
    )
