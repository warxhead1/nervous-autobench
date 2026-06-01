"""Back-compat shim. Prefer ``autobench.audit.harness_diff``."""

from autobench.audit.harness_diff import (  # noqa: F401
    diff_harnesses,
)

__all__ = ["diff_harnesses"]
