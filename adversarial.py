"""Back-compat shim. Prefer `autobench.rsi.adversarial`.

Phase 2B of the autobench restructuring moved ``adversarial.py`` into
the ``autobench.rsi`` subpackage. This module re-exports the public
surface so legacy ``from autobench.adversarial import …`` call sites
keep working.
"""

import httpx  # noqa: F401 — re-exported for test monkeypatches

from .rsi.adversarial import (  # noqa: F401
    DEFAULT_FAILURE_MODES,
    AdversarialCase,
    AdversarialDual,
    AdversarialGenerator,
    AdversarialRoundResult,
    _STATIC_FALLBACK,
    _round_up_ratio,
    generate_adversarial_case_mix,
    mine_failure_modes_from_result,
)
