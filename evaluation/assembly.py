"""Benchmark case assembly — wire-pop Phase 3 (nervous-bus-gdzo).

PopulationRunner consumes a list of :class:`BenchmarkCase` objects each cycle.
Phase 4 (curriculum.py) already supplies fresh problems; Phase 3 (this module)
mixes adversarially-generated *gotchas* into that set, keyed to the previous
cycle's failure modes.

Public surface
==============
- :func:`assemble_benchmark_cases` — replace ``~adversarial_ratio`` of cases
  with curveballs and return the mixed list. Idempotent. Defaults to 20%.

Wiring
======
Callers of :class:`autobench.population.PopulationRunner` invoke this helper
*before* ``runner.run(cases)``. The runner stays generic — it doesn't need to
know about adversarial machinery. The caller passes the *previous cycle's*
:class:`PopulationResult` (or ``None`` on the first cycle) so failure-mode
mining can ground the curveballs in observed weaknesses.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from ..rsi.adversarial import (
    AdversarialGenerator,
    DEFAULT_FAILURE_MODES,
    _round_up_ratio,
    generate_adversarial_case_mix,
    mine_failure_modes_from_result,
)
from ..evaluator import BenchmarkCase
from ..observability import AutobenchObservability

logger = logging.getLogger(__name__)


DEFAULT_ADVERSARIAL_RATIO: float = 0.20


def assemble_benchmark_cases(
    base_cases: list[BenchmarkCase],
    prior_result: Any = None,
    adversarial_ratio: float = DEFAULT_ADVERSARIAL_RATIO,
    generator: AdversarialGenerator | None = None,
    obs: AutobenchObservability | None = None,
    language: str = "python",
    rng: random.Random | None = None,
) -> list[BenchmarkCase]:
    """Mix adversarial gotchas into ``base_cases`` and return the assembled list.

    Steps:
      1. Compute ``n_adversarial = ceil(len(base_cases) * adversarial_ratio)``.
      2. Mine failure modes from ``prior_result``. Fall back to
         :data:`DEFAULT_FAILURE_MODES` when no prior cycle or no signal.
      3. Generate ``n_adversarial`` curveball BenchmarkCases.
      4. Replace ``n_adversarial`` randomly-chosen entries from ``base_cases``
         with the curveballs (preserves total count). Replacement (not
         append) keeps cycle cost constant.

    Emissions (when ``obs`` is provided):
      * ``autobench.adversarial.curveball_generated.v1`` — one per curveball.
      * ``autobench.adversarial.round_complete.v1`` — one summary for the
        assembled batch.

    Args:
        base_cases: The seed list (e.g. cf-tier-1 cases or curriculum output).
            When empty or ``None``, the function returns an empty list.
        prior_result: Previous cycle's :class:`PopulationResult` (or any
            duck-typed equivalent with ``.advocates`` → ``[advocate.history]``).
            ``None`` on the first cycle.
        adversarial_ratio: Fraction of cases to replace, in ``[0, 1]``.
            Defaults to ``0.20`` per bead spec. ``0.0`` disables the mix.
        generator: Optional :class:`AdversarialGenerator`. When ``None`` one
            is constructed with ``obs`` so emissions land on the same session.
        obs: Optional observability emitter. When provided, the generator
            inherits it and the assembly emits its own ``round_complete``.
        language: BenchmarkCase language tag for the curveballs.
        rng: Optional :class:`random.Random` for deterministic replacement.

    Returns:
        New list of :class:`BenchmarkCase` with the same length as
        ``base_cases`` (minus zero or more depending on rounding). The original
        list is *not* mutated.
    """
    if not base_cases:
        return []
    if adversarial_ratio <= 0.0:
        return list(base_cases)

    n_total = len(base_cases)
    n_adversarial = _round_up_ratio(n_total, adversarial_ratio)
    if n_adversarial == 0:
        return list(base_cases)

    failure_modes = mine_failure_modes_from_result(prior_result)
    if not failure_modes:
        failure_modes = list(DEFAULT_FAILURE_MODES)

    # Share obs with the generator so curveball_generated events emit on the
    # caller's session.
    if generator is None:
        generator = AdversarialGenerator(obs=obs)
    elif obs is not None and generator.obs is None:
        generator.obs = obs

    curveballs = generate_adversarial_case_mix(
        n_cases=n_adversarial,
        failure_modes=failure_modes,
        generator=generator,
        obs=obs,
        language=language,
    )

    if not curveballs:
        # Generator failed entirely (shouldn't happen — it has a static
        # fallback) — degrade by returning the base set unchanged.
        return list(base_cases)

    # Replace random positions in a copy of base_cases. Using replacement
    # (not append) keeps the total cycle cost stable.
    rng = rng or random.Random()
    assembled = list(base_cases)
    positions = rng.sample(range(n_total), k=min(len(curveballs), n_total))
    for pos, curveball in zip(positions, curveballs):
        assembled[pos] = curveball

    return assembled


__all__ = [
    "DEFAULT_ADVERSARIAL_RATIO",
    "assemble_benchmark_cases",
]
