"""Back-compat shim. Prefer `autobench.evaluation.diversity`.

Phase 2B of the autobench restructuring moved ``diversity.py`` into
the ``autobench.evaluation`` subpackage. This module re-exports the
public surface so legacy ``from autobench.diversity import …`` call
sites keep working.
"""

from .evaluation.diversity import (  # noqa: F401
    FINGERPRINT_DIM,
    DiversityTracker,
    LineageSignature,
    StructuralFingerprint,
    _cosine,
    lineage_signature,
    pairwise_lineage_distance,
    pairwise_lineage_similarity,
    run_ab_comparison,
    sacs_similarity,
)
