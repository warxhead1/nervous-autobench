"""Back-compat shim. Prefer `autobench.evaluation.registry`.

Phase 2B of the autobench restructuring moved ``benchmark_registry.py``
into the ``autobench.evaluation`` subpackage. This module re-exports
the public surface so legacy ``from autobench.benchmark_registry
import …`` call sites keep working.
"""

from .evaluation.registry import (  # noqa: F401
    DEFAULT_DOMAIN,
    DEFAULT_WEIGHTS,
    BenchmarkDomain,
    BenchmarkRegistry,
    _read_enabled_domains_env,
    _read_weight_overrides_env,
)
