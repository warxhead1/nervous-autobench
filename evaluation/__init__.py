"""Evaluation subpackage: registry, assembly, curriculum, distillation, diversity, etc.

Public surface (root-level ``autobench.benchmark_registry`` etc. are
deprecated back-compat shims; ``autobench.evaluator`` stays at the root):

  - :class:`BenchmarkEvaluator`, :class:`BenchmarkCase`, :class:`BenchmarkResult`
    (from the root ``autobench.evaluator`` — the main entry point)
  - :class:`BenchmarkRegistry`, :func:`DEFAULT_DOMAIN` (from ``registry``)
  - :func:`assemble_benchmark_cases` (from ``assembly``)
  - :class:`CurriculumAgent` (from ``curriculum``)
  - :class:`CycleDistiller` (from ``distillation``)
  - :class:`DiversityTracker`, :func:`lineage_signature`,
    :func:`pairwise_lineage_similarity` (from ``diversity``)
  - :class:`CodeForcesScraper` (from ``codeforces``)
  - :func:`run_noise_floor_measurement`, :class:`NoiseFloorReport` (from ``noise_floor``)
"""

from __future__ import annotations

from .assembly import assemble_benchmark_cases
from .codeforces import CodeForcesScraper
from .curriculum import CurriculumAgent, CurriculumGoals, CurriculumScheduler
from .distillation import CycleDistiller
from .diversity import (
    DiversityTracker,
    LineageSignature,
    StructuralFingerprint,
    lineage_signature,
    pairwise_lineage_similarity,
    run_ab_comparison,
)
from .noise_floor import (
    LanguageStats,
    NoiseFloorReport,
    NoiseFloorResult,
    run_noise_floor_measurement,
)
from .registry import (
    DEFAULT_DOMAIN,
    BenchmarkDomain,
    BenchmarkRegistry,
)

__all__ = [
    "BenchmarkDomain",
    "BenchmarkRegistry",
    "CodeForcesScraper",
    "CurriculumAgent",
    "CurriculumGoals",
    "CurriculumScheduler",
    "CycleDistiller",
    "DEFAULT_DOMAIN",
    "DiversityTracker",
    "LanguageStats",
    "LineageSignature",
    "NoiseFloorReport",
    "NoiseFloorResult",
    "StructuralFingerprint",
    "assemble_benchmark_cases",
    "lineage_signature",
    "pairwise_lineage_similarity",
    "run_ab_comparison",
    "run_noise_floor_measurement",
]
