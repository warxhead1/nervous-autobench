"""Oasis FunSearch kernel — evolves a shallow-water flow law into a breathing oasis."""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, register_kernel,
)
from .instance import OasisInstance, generate_instance
from .oracle import OASIS_FUNCTION_SIGNATURE, get_seed_programs
from .scoring import evaluate_on_instance
from .render import render_oasis

logger = logging.getLogger(__name__)


@register_kernel("oasis")
class OasisKernel(FunSearchKernel):
    """Evolves `flux(dhead, depth, visc)` — the constitutive law of a 2D
    shallow-water oasis. The oracle drives a dune-basin field with an artesian
    spring + day/night evaporation and measures whether the flow produces a
    stable, basin-captured pool that breathes with the day cycle.

    Fitness = basin_capture · (0.4+0.6·stability) · pool_fraction ·
              (0.4+0.6·breathing).
    """

    kernel_name = "oasis"
    BUS_CHANNEL_PREFIX = "oasis"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        logger.info("Oasis kernel: %d instances, sandbox=%s",
                    len(self.problem_instances), self.executor.sandbox_type)

    def load_instances(self) -> list[OasisInstance]:
        out = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info("Oasis instance %r: visc=%.3f spring=%.1f artesian=%.2f steps=%d",
                        name, inst.viscosity, inst.spring, inst.artesian, inst.n_steps)
            out.append(inst)
        return out

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(code, instance, self.executor,
                                    run_timeout=self.config.run_timeout)

    def build_prompt(self, island: Island, top_programs: list[CandidateProgram],
                     generation: int, hint: str = "") -> str:
        inst = self.problem_instances[0] if self.problem_instances else None
        inst_desc = (f"{inst.name} (viscosity={inst.viscosity:.3f} — {inst.description})"
                     if inst else "unknown")
        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )
        hint_block = f"\n## Strategic advice (plateau breaker)\n{hint}\n" if hint else ""
        return (
            "You are a fluid-dynamics expert evolving the flow law of a desert oasis.\n"
            "Evolve a better shallow-water flux(dhead, depth, visc).\n\n"
            f"{OASIS_FUNCTION_SIGNATURE}\n\n"
            f"Target: {inst_desc}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            "Goal: maximise Fitness = basin_capture · (0.4+0.6·stability) · "
            "pool_fraction · (0.4+0.6·breathing).\n"
            "  basin_capture: water pools in the low basin (not spread to ridges)\n"
            "  stability:     the pool converges to a steady level (no drift/flood)\n"
            "  pool_fraction: a pool a few % of the basin (not a spike, not a sea)\n"
            "  breathing:     the shoreline oscillates gently with the day cycle\n\n"
            "Rules:\n"
            "- Return ONLY the flux() function in a single ```cpp code block\n"
            "- Signature: float flux(float dhead, float depth, float visc)\n"
            "- Must return a finite value >= 0 for all dhead>=0, depth>=0, visc>0\n"
            "- No loops, no static state, no allocation. Under 15 lines.\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m and "flux" in m.group(1):
            return m.group(1).strip()
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        name = self.config.instances[0] if self.config.instances else "clear_spring"
        return [
            CandidateProgram(id=str(uuid.uuid4()), code=code,
                             island=island_id, generation=generation)
            for _label, code in get_seed_programs(name)
        ]

    # ------------------------------------------------------------------
    # Gallery render
    # ------------------------------------------------------------------

    def _render_best_program(self, best: CandidateProgram, out_path: Path) -> bool:
        inst = self.problem_instances[0] if self.problem_instances else None
        if inst is None:
            return False
        try:
            return render_oasis(best.code, inst, self.executor, out_path)
        except Exception as e:
            logger.debug("oasis render failed: %s", e)
            return False

    def _artifact_render_type(self) -> str:
        return "oasis_strip"

    # ------------------------------------------------------------------
    # Bus publishing
    # ------------------------------------------------------------------

    def _publish_started(self) -> None:
        from ..kernels.base import _git_commit_short
        self._publish("oasis.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": self.config.instances,
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "sandbox_type": self.executor.sandbox_type,
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        best = programs[0] if programs else None
        self._publish("oasis.kernel.completed.v1", {
            "run_id": self.run_id,
            "total_generations": self.generation,
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "best_program": {
                "id": best.id, "fitness": best.fitness, "island": best.island,
                "generation": best.generation, "flux_code": best.code,
            } if best else None,
            "history": self.history,
        })

    def save_results(self, programs: list[CandidateProgram]) -> Optional[Path]:
        if not self.config.output_dir:
            return None
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"oasis_results_gen{self.generation:02d}.json"
        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        out_path.write_text(json.dumps({
            "kernel": "oasis",
            "run_id": self.run_id,
            "generation": self.generation,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {"id": p.id, "fitness": p.fitness, "worst_fitness": p.worst_fitness,
                 "island": p.island, "generation": p.generation, "flux_code": p.code}
                for p in top[:20]
            ],
        }, indent=2))
        logger.info("Oasis results → %s (best=%.4f)", out_path, top[0].fitness if top else 0)
        return out_path
