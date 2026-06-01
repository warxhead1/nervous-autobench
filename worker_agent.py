"""Back-compat shim. Prefer `autobench.llm.worker`.

Phase 2B of the autobench restructuring moved ``worker_agent.py`` into
the ``autobench.llm`` subpackage. This module re-exports the public
surface so legacy ``from autobench.worker_agent import …`` call sites
keep working.
"""

import httpx  # noqa: F401 — re-exported for test monkeypatches
import time  # noqa: F401 — re-exported for test monkeypatches

from .llm.worker import (  # noqa: F401
    DEFAULT_SYSTEM_PROMPT,
    MiniMaxWorker,
    WorkerResult,
    _calc_backoff_ms,
    _estimate_cost,
    _extract_code,
    _parse_response,
)
