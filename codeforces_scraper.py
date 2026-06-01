"""Back-compat shim. Prefer `autobench.evaluation.codeforces`.

Phase 2B of the autobench restructuring moved ``codeforces_scraper.py``
into the ``autobench.evaluation`` subpackage. This module re-exports
the public surface so legacy ``from autobench.codeforces_scraper
import …`` call sites keep working.
"""

from .evaluation.codeforces import (  # noqa: F401
    CodeForcesScraper,
)
