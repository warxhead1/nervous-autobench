"""Cross-domain benchmark registry (wire-pop Phase 6, nervous-bus-qp91).

The legacy RSI cycle runs exclusively against ``codeforces_tier1`` (20
fixed CodeForces problems). Long runs overfit to that single
distribution. This module wires a registry of benchmark domains so each
cycle evaluates against MULTIPLE domains and aggregates per-domain
scores into a single cross-domain figure of merit. Set
``AUTOBENCH_CROSS_DOMAIN=1`` to enable the dispatch (off by default).

Public surface: :class:`BenchmarkRegistry`, :class:`BenchmarkDomain`,
:data:`DEFAULT_DOMAIN`, :data:`DEFAULT_WEIGHTS`.

The env-knob table (``AUTOBENCH_DOMAINS`` / ``AUTOBENCH_DOMAIN_WEIGHTS``),
the multifile_refactor thesis-failure caveat, and the legacy single-
domain shim behaviour are preserved in
``_checkpoints/architecture-history.md``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..evaluator import BenchmarkCase


DEFAULT_DOMAIN = "<default>"
"""Name used when a legacy single-domain caller passes ``list[BenchmarkCase]``.

Single-key per_domain map keyed by this string preserves bit-for-bit
behavior on the cf-tier-1 single-domain path.
"""


# Default weights — see module docstring. These sum to 1.0 when all three
# are enabled; the registry renormalizes when any are missing.
DEFAULT_WEIGHTS: dict[str, float] = {
    "codeforces_tier1": 0.5,
    "multifile_refactor": 0.3,
    "shader_tier1": 0.2,
}


# --------------------------------------------------------------------------- #
# Domain dataclass
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkDomain:
    """A named benchmark domain participating in cross-domain aggregation.

    Attributes:
        name: Stable identifier. Lowercase, snake_case, matches the
            directory under ``autobench/benchmarks/`` when possible.
        case_loader: Zero-arg callable returning a list of BenchmarkCase
            objects. May be expensive (file I/O, network); the registry
            calls it at most once per :meth:`BenchmarkRegistry.load_all_cases`.
        weight: Non-negative weight in the cross-domain aggregate. Coerced
            to 0.0 when negative; values are renormalized at aggregation
            time so callers don't need to track the sum.
        optional: When True, a ``case_loader`` exception is logged to
            stderr and the domain silently drops out of this cycle.
            Required domains propagate the exception.
    """

    name: str
    case_loader: Callable[[], list[BenchmarkCase]]
    weight: float = 0.0
    optional: bool = False


# --------------------------------------------------------------------------- #
# Built-in loaders for each shipping benchmark
# --------------------------------------------------------------------------- #


def _load_codeforces_tier1_cases() -> list[BenchmarkCase]:
    """Load the 20 curated CodeForces tier-1 cases from cases.jsonl."""
    # Phase 2B: moved into autobench/evaluation/ — benchmarks/ still lives
    # at the autobench root, so the parent traversal is now TWO levels up.
    here = Path(__file__).resolve().parent.parent / "benchmarks" / "codeforces_tier1"
    cases_file = here / "cases.jsonl"
    cases: list[BenchmarkCase] = []
    with open(cases_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cases.append(
                BenchmarkCase(
                    id=d["id"],
                    prompt=d["prompt"],
                    language=d.get("language", "python"),
                    expected_output=d.get("expected_output", ""),
                    expected_outputs=d.get("expected_outputs", []),
                    constraints=d.get("constraints", {}),
                    starter_code=d.get("starter_code", ""),
                    test_inputs=d.get("test_inputs", []),
                    metadata=d.get("metadata", {}),
                )
            )
    return cases


def _load_multifile_refactor_cases() -> list[BenchmarkCase]:
    """Load the multifile_refactor cases as ``BenchmarkCase`` objects.

    The multifile cases.jsonl is dict-shaped (case_id/description/oracle_*
    fields, not BenchmarkCase's prompt/expected_output). We adapt by:

        * Mapping ``case_id``     → ``BenchmarkCase.id``
        * Mapping ``description`` → ``BenchmarkCase.prompt``
        * Stashing the multifile-specific fields under ``metadata`` so a
          domain-aware evaluator (future work) can recover them.

    This keeps the registry shape uniform (list[BenchmarkCase] everywhere)
    while allowing domain-specific scoring to be wired in later. The
    current cross-domain aggregation treats each domain's BenchmarkResult
    via the standard BenchmarkEvaluator; multifile cases that have no
    BenchmarkEvaluator-style verifier simply score 0 at that layer — which
    is fine: the *domain* still exists, the *aggregate* still has a slot,
    and the head-to-head benchmark in ``run.py`` remains the canonical
    scorer for the multifile thesis (this registry never claims to replace
    it). Future work (out of scope for qp91): plug a multifile-specific
    evaluator into the cross-domain dispatch.
    """
    here = Path(__file__).resolve().parent.parent / "benchmarks" / "multifile_refactor"
    cases_file = here / "cases.jsonl"
    cases: list[BenchmarkCase] = []
    with open(cases_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            metadata = {
                "domain": "multifile_refactor",
                "fixture_dir": d.get("fixture_dir", ""),
                "oracle_files": d.get("oracle_files", []),
                "oracle_tests": d.get("oracle_tests", []),
                "max_files_changed": d.get("max_files_changed", 6),
                "difficulty": d.get("difficulty", "unknown"),
                "expected_files": d.get("expected_files", 0),
            }
            cases.append(
                BenchmarkCase(
                    id=d["case_id"],
                    prompt=d.get("description", ""),
                    language="python",
                    expected_output="",
                    constraints={"max_time_seconds": 120, "max_memory_mb": 512},
                    metadata=metadata,
                )
            )
    return cases


def _load_shader_tier1_cases() -> list[BenchmarkCase]:
    """Load shader_tier1 cases from per-case JSON files.

    Unlike the other domains, shader_tier1 does NOT ship a cases.jsonl —
    each case is its own NN_name.json file. We enumerate them in sorted
    order and adapt the GLSL-prompt/PNG-reference shape into BenchmarkCase
    metadata.

    Returns an empty list if the directory doesn't exist (so optional=True
    handling skips the domain cleanly).
    """
    here = Path(__file__).resolve().parent.parent / "benchmarks" / "shader_tier1"
    if not here.exists():
        return []
    cases: list[BenchmarkCase] = []
    for case_file in sorted(here.glob("*.json")):
        try:
            d = json.loads(case_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict) or "id" not in d or "prompt" not in d:
            continue
        metadata = dict(d.get("metadata", {}) or {})
        metadata.update(
            {
                "domain": "shader_tier1",
                "reference_shader": d.get("reference_shader", ""),
            }
        )
        cases.append(
            BenchmarkCase(
                id=d["id"],
                prompt=d["prompt"],
                language=d.get("language", "glsl"),
                expected_output=d.get("expected_output", ""),
                constraints=d.get("constraints", {}),
                metadata=metadata,
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# Env parsing helpers
# --------------------------------------------------------------------------- #


def _read_enabled_domains_env() -> list[str] | None:
    """Parse ``AUTOBENCH_DOMAINS``. Returns None when unset (= "all enabled")."""
    raw = os.environ.get("AUTOBENCH_DOMAINS")
    if raw is None or raw.strip() == "":
        return None
    names = [tok.strip() for tok in raw.split(",")]
    return [n for n in names if n]


def _read_weight_overrides_env() -> dict[str, float]:
    """Parse ``AUTOBENCH_DOMAIN_WEIGHTS``. Returns ``{name: weight}``."""
    raw = os.environ.get("AUTOBENCH_DOMAIN_WEIGHTS")
    if raw is None or raw.strip() == "":
        return {}
    out: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, weight_s = pair.partition("=")
        name = name.strip()
        try:
            w = float(weight_s.strip())
        except ValueError:
            continue
        if w < 0.0:
            w = 0.0
        out[name] = w
    return out


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkRegistry:
    """Container for the enabled benchmark domains this cycle.

    Construct via :meth:`default` for the standard cf-tier-1 + multifile +
    shader trio honoring ``AUTOBENCH_DOMAINS`` / ``AUTOBENCH_DOMAIN_WEIGHTS``,
    or pass an explicit ``domains`` list for tests.
    """

    domains: list[BenchmarkDomain] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def default(cls) -> "BenchmarkRegistry":
        """Build the registry from the project-wide defaults + env overrides.

        Registers:
            * ``codeforces_tier1`` (required, weight 0.5)
            * ``multifile_refactor`` (optional, weight 0.3) — Phase 1 cases
              even though the analytical thesis failed (see module docstring).
            * ``shader_tier1`` (optional, weight 0.2) — pulled from per-case
              JSON files; gracefully empty when the directory is gone.

        Applies ``AUTOBENCH_DOMAINS`` to filter which participate and
        ``AUTOBENCH_DOMAIN_WEIGHTS`` to override default weights.
        """
        enabled = _read_enabled_domains_env()
        overrides = _read_weight_overrides_env()

        registered: list[BenchmarkDomain] = [
            BenchmarkDomain(
                name="codeforces_tier1",
                case_loader=_load_codeforces_tier1_cases,
                weight=overrides.get(
                    "codeforces_tier1",
                    DEFAULT_WEIGHTS["codeforces_tier1"],
                ),
                optional=False,
            ),
            BenchmarkDomain(
                name="multifile_refactor",
                case_loader=_load_multifile_refactor_cases,
                weight=overrides.get(
                    "multifile_refactor",
                    DEFAULT_WEIGHTS["multifile_refactor"],
                ),
                optional=True,
            ),
            BenchmarkDomain(
                name="shader_tier1",
                case_loader=_load_shader_tier1_cases,
                weight=overrides.get(
                    "shader_tier1",
                    DEFAULT_WEIGHTS["shader_tier1"],
                ),
                optional=True,
            ),
        ]

        # Apply AUTOBENCH_DOMAINS filter.
        if enabled is not None:
            allowed = set(enabled)
            registered = [d for d in registered if d.name in allowed]

        return cls(domains=registered)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enabled_domains(self) -> list[str]:
        """Return the names of domains that will participate this cycle."""
        return [d.name for d in self.domains]

    def weights(self) -> dict[str, float]:
        """Return ``{name: weight}`` for enabled domains (raw, un-normalized)."""
        return {d.name: float(d.weight) for d in self.domains}

    def load_all_cases(self) -> dict[str, list[BenchmarkCase]]:
        """Eagerly load every enabled domain's cases.

        Optional domains that raise ``Exception`` are logged to stderr and
        dropped from the returned mapping. Required domains propagate.

        An optional domain that loads to an empty list IS kept in the map
        (so the aggregate still surfaces a 0.0 slot for it). Use the empty
        list as the signal that the domain shipped but had no test inputs.
        """
        out: dict[str, list[BenchmarkCase]] = {}
        for d in self.domains:
            try:
                cases = d.case_loader()
            except Exception as exc:  # noqa: BLE001
                if not d.optional:
                    raise
                print(
                    f"[benchmark_registry] optional domain {d.name!r} "
                    f"failed to load ({type(exc).__name__}: {exc}); skipping",
                    file=sys.stderr,
                )
                continue
            out[d.name] = list(cases)
        return out

    def aggregate_score(self, per_domain_scores: dict[str, float]) -> float:
        """Compute the weighted-average aggregate score across domains.

        Weights are renormalized across the keys actually present in
        ``per_domain_scores`` so callers don't need to track which domains
        survived loading. An empty input → 0.0 (no domains contributed,
        no score). A single-domain input with the only weight at 0.0 → the
        raw score (degenerate-weight defensive path; matches the legacy
        single-domain semantics where aggregate == single-domain score).
        """
        if not per_domain_scores:
            return 0.0

        # Pull weight for each key that's actually present.
        weight_map = self.weights()
        used: list[tuple[str, float, float]] = []
        for name, score in per_domain_scores.items():
            w = float(weight_map.get(name, 0.0))
            used.append((name, max(0.0, w), float(score)))

        total_weight = sum(w for _, w, _ in used)
        if total_weight <= 0.0:
            # All weights zero/missing — fall back to a uniform mean so
            # the aggregate is still a meaningful figure of merit.
            if not used:
                return 0.0
            return sum(s for _, _, s in used) / len(used)

        weighted = sum(w * s for _, w, s in used)
        return weighted / total_weight


__all__ = [
    "DEFAULT_DOMAIN",
    "DEFAULT_WEIGHTS",
    "BenchmarkDomain",
    "BenchmarkRegistry",
    "_load_codeforces_tier1_cases",
    "_load_multifile_refactor_cases",
    "_load_shader_tier1_cases",
    "_read_enabled_domains_env",
    "_read_weight_overrides_env",
]
