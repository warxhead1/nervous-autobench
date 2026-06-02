"""ThermalKernel — FunSearch loop class (split from thermal_kernel/__init__.py)."""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional
import json

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, register_kernel,
)
from .instance import ThermalInstance, generate_instance
from .scoring import THERMAL_FUNCTION_SIGNATURE, evaluate_on_instance
from .oracle import get_seed_programs

logger = logging.getLogger("autobench.thermal_kernel")


# ---------------------------------------------------------------------------
# ThermalKernel
# ---------------------------------------------------------------------------

@register_kernel("thermal")
class ThermalKernel(FunSearchKernel):
    """FunSearch kernel: 2D Allen-Cahn phase field — freeze/melt spot dynamics.

    Oracle: 2D PDE time-stepper on a 64×64 grid with a fixed temperature field.
    Fitness = 0.4·phase_coverage + 0.4·retention + 0.2·radial_corr.

    Uses CORRECT sign convention: m = 2*(0.5-T), so cold (T<0.5) nucleates solid.
    """

    kernel_name = "thermal"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        for inst in self.problem_instances:
            logger.info("Thermal instance %r: cold=%.2f hot=%.2f R=%.0f D=%.0f dt=%.4f",
                        inst.name, inst.cold_temp, inst.hot_temp,
                        inst.cold_radius, inst.D, inst.dt)

    def load_instances(self) -> list[ThermalInstance]:
        return [generate_instance(n) for n in self.config.instances]

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(code, instance, self.executor)

    def build_prompt(self, island: Island, top_programs: list[CandidateProgram],
                     generation: int, hint: str = "") -> str:
        instance = self.problem_instances[0] if self.problem_instances else None
        inst_desc = (f"{instance.name} — {instance.description}" if instance else "unknown")

        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )

        hint_block = f"\n## Strategic hint (plateau breaker)\n{hint}\n" if hint else ""

        cold = instance.cold_temp if instance else 0.25
        hot  = instance.hot_temp  if instance else 0.75
        mode = "solid grows from cold zone" if (instance and instance.initial_phi < 0.5) else "liquid grows from hot zone"

        return (
            f"You are a computational physicist specialising in phase-field models.\n"
            f"Evolve a better Allen-Cahn driving force reaction(phi, temp).\n\n"
            f"{THERMAL_FUNCTION_SIGNATURE}\n"
            f"Target: {inst_desc}\n"
            f"  cold_temp={cold:.2f}  hot_temp={hot:.2f}  mode: {mode}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Fitness = 0.4·phase_coverage + 0.4·retention + 0.2·radial_corr\n"
            f"  phase_coverage: target phase fills its zone\n"
            f"  retention: other zone stays in its phase\n"
            f"  radial_corr: interface profile matches ideal tanh at cold_radius\n\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            f"Return ONLY a single ```cpp code block.\n"
            f"- Signature: float reaction(float phi, float temp)\n"
            f"- |reaction| < 50 for all phi,temp ∈ [0,1]\n"
            f"- No loops, no static state, under 20 lines\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "reaction" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        programs = []
        for name, code in get_seed_programs(
            self.config.instances[0] if self.config.instances else "freeze_spot"
        ):
            prog = CandidateProgram(
                id=str(uuid.uuid4()),
                code=code,
                island=island_id,
                generation=generation,
            )
            programs.append(prog)
        return programs

    def save_results(self, programs: list[CandidateProgram]) -> Optional[Path]:
        if not self.config.output_dir:
            return None
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"thermal_results_gen{self.generation:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "thermal",
            "run_id": self.run_id,
            "generation": self.generation,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {
                    "id": p.id,
                    "fitness": p.fitness,
                    "worst_fitness": p.worst_fitness,
                    "island": p.island,
                    "generation": p.generation,
                    "reaction_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Thermal results → %s (best=%.4f)", out_path,
                    top[0].fitness if top else 0)
        return out_path
