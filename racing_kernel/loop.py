"""RacingKernel — FunSearchKernel subclass for racing-line evolution.

Island assignment: one island per track layout (island_id % n_tracks).
Bus channel prefix: "racing" — events emit as racing.kernel.started.v1 etc.,
which pass through _unify_kernel_channel as non-kernel-domain events (the
racing domain is not in KERNEL_DOMAINS for backward-compat); they still go
through _publish → debug.jsonl + nervous CLI.
"""

from __future__ import annotations

import logging
import time

from autobench.kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    register_kernel,
)
from autobench.racing_kernel.instance import RacingInstance, generate_instance, TRACK_LAYOUTS
from autobench.racing_kernel.oracle import (
    SEED_RACING_PROGRAMS,
    build_llm_prompt,
    evaluate_on_instance,
    parse_llm_response,
)

logger = logging.getLogger(__name__)


@register_kernel("racing")
class RacingKernel(FunSearchKernel):
    """FunSearch racing-line controller evolution kernel.

    Evolves a Python ``racing_line(u, curvature, half_width, speed_limit)``
    policy for each track instance using a generative-membership oracle that
    scores:
      - speed    (45%) — lap time vs physics ceiling
      - smoothness (30%) — low curvature variation in the driven line
      - track    (25%) — fraction of samples within the corridor

    Island assignment: island_id % n_tracks → each island specialises on one
    track, reusing the SDF-kernel island-per-instance pattern.
    """

    BUS_CHANNEL_PREFIX = "racing"

    def __init__(self, config: KernelConfig):
        if not config.instances:
            config.instances = list(TRACK_LAYOUTS.keys())

        # Island-per-track routing (same pattern as SDF island-per-instance)
        config.island_instance_assignment = True

        super().__init__(config)

        # Load track instances immediately (no sandbox needed — pure Python oracle)
        self.problem_instances: list[RacingInstance] = self.load_instances()
        logger.info(
            "RacingKernel: loaded %d track(s): %s",
            len(self.problem_instances),
            [inst.name for inst in self.problem_instances],
        )

    # ------------------------------------------------------------------
    # FunSearchKernel abstract interface
    # ------------------------------------------------------------------

    def load_instances(self) -> list[RacingInstance]:
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info(
                "Track '%s': %d centerline pts, lap=%.1fm, ref_time=%.1fs",
                inst.name, len(inst.centerline), inst.lap_length, inst.ref_lap_time_s,
            )
            instances.append(inst)
        return instances

    def evaluate_candidate(self, code: str, instance: RacingInstance) -> float | None:
        """Evaluate racing_line code against one track instance."""
        return evaluate_on_instance(code, instance)

    def evaluate_fitness(self, program: CandidateProgram) -> tuple[float, float, float]:
        """Evaluate against this island's assigned track (island-per-instance)."""
        t0 = time.time()
        if self.config.island_instance_assignment and self.problem_instances:
            inst_idx = program.island % len(self.problem_instances)
            instances = [self.problem_instances[inst_idx]]
        else:
            instances = self.problem_instances

        ratios: list[float] = []
        for inst in instances:
            r = self.evaluate_candidate(program.code, inst)
            ratios.append(r if r is not None else 0.0)
        elapsed = (time.time() - t0) * 1000

        if not ratios:
            return 0.0, 0.0, 0.0
        mean = sum(ratios) / len(ratios)
        var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        worst = min(ratios)
        program.fitness = mean
        program.fitness_variance = var
        program.worst_fitness = worst
        program.computation_time_ms = elapsed
        program.evaluated = True
        return mean, var, worst

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        if self.config.island_instance_assignment and self.problem_instances:
            inst_idx = island.id % len(self.problem_instances)
            inst_name = self.problem_instances[inst_idx].name
        else:
            inst_name = ""
        return build_llm_prompt(island, top_programs, generation, inst_name, hint)

    def parse_response(self, response: str) -> str:
        return parse_llm_response(response)

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        """Three parametric baseline policies — one per island variant."""
        programs = []
        for name, code in SEED_RACING_PROGRAMS:
            programs.append(CandidateProgram(
                id=f"island{island_id}_gen{generation}_{name}",
                code=code,
                island=island_id,
                generation=generation,
                source="baseline",
            ))
        return programs
