"""Pytest conftest — racing_kernel in-package test discovery.

Ensures autobench.racing_kernel is imported before pytest collects tests
from racing_kernel/tests/, so that @register_kernel("racing") fires and
all absolute imports in the test modules resolve via the editable install.
"""
# Pre-import via the installed package to fire @register_kernel side-effect.
import autobench.racing_kernel  # noqa: F401
