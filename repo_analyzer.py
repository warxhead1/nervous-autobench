"""Back-compat shim. Prefer ``autobench.audit.repo_analyzer``."""

from autobench.audit.repo_analyzer import (  # noqa: F401
    ARCH_DEPTH_THRESHOLD,
    COUPLING_HIGH,
    ChangeType,
    RepoMetrics,
    TEST_PATTERNS,
    analyze_repo,
    classify_change,
    detect_architecture_shift,
)

__all__ = [
    "ARCH_DEPTH_THRESHOLD",
    "COUPLING_HIGH",
    "ChangeType",
    "RepoMetrics",
    "TEST_PATTERNS",
    "analyze_repo",
    "classify_change",
    "detect_architecture_shift",
]
