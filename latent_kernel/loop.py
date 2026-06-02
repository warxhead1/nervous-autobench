"""Latent kernel evolution loop.

LatentKernel (FunSearchKernel subclass). Importing this module fires
@register_kernel("latent"). Moved verbatim from the package __init__
(behaviour-preserving file split).
"""
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
from .instance import LatentInstance, generate_instance
from .scoring import LATENT_FUNCTION_SIGNATURE, evaluate_on_instance
from .oracle import get_seed_programs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LatentKernel
# ---------------------------------------------------------------------------

@register_kernel("latent")
class LatentKernel(FunSearchKernel):
    """FunSearch kernel: 2D coupled phase-thermal PDE with latent heat.

    Evolves reaction(phi, temp, lap_T) against an oracle where temperature
    EVOLVES — it rises as ice forms (latent heat) and the freeze front is
    thermodynamically self-limiting at φ_eq = (0.5 − T_cold) / L.

    Fitness v2 = 0.30·T_balance + 0.30·velocity + 0.25·sharpness + 0.15·retention

    velocity_score rewards Stefan self-regulation: the oracle tracks the front position
    at the midpoint step and final step, comparing observed velocity to the analytical
    Stefan prediction v_stefan = D_T * |undercooling| / (L * r_mid). Retreating fronts
    score 0 (hard step function on v > 0).

    D_phi reduced to 4.0 from 10.0: natural interface width sqrt(4/0.5) ≈ 2.8 cells
    makes target_width=3 achievable, unblocking the sharpness score from dead weight.
    """

    kernel_name = "latent"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        for inst in self.problem_instances:
            logger.info(
                "Latent instance %r: cold=%.2f hot=%.2f R=%.0f L=%.2f φ_eq=%.3f",
                inst.name, inst.cold_temp, inst.hot_temp,
                inst.cold_radius, inst.L, inst.phi_eq
            )

    def load_instances(self) -> list[LatentInstance]:
        return [generate_instance(n) for n in self.config.instances]

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(code, instance, self.executor)

    def build_prompt(self, island: Island, top_programs: list[CandidateProgram],
                     generation: int, hint: str = "") -> str:
        instance = self.problem_instances[0] if self.problem_instances else None
        phi_eq_str = f"{instance.phi_eq:.3f}" if instance else "0.625"
        inst_desc  = (f"{instance.name} — {instance.description}" if instance else "unknown")
        cold = instance.cold_temp if instance else 0.25
        hot  = instance.hot_temp  if instance else 0.75
        L    = instance.L         if instance else 0.4
        D_T  = instance.D_T       if instance else 5.0

        # Compute reference Stefan velocity for prompt context
        undercooling = 0.5 - cold if instance and instance.initial_phi < 0.5 else (hot - 0.5)
        v_stefan_ref = f"{D_T * undercooling / (L * 15.0):.3f}"  # at r~15 cells

        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )
        hint_block = f"\n## Strategic hint\n{hint}\n" if hint else ""

        return (
            f"You are a computational physicist specialising in phase-field models "
            f"and thermodynamic coupling.\n"
            f"Evolve a better Allen-Cahn driving force with latent heat feedback.\n\n"
            f"{LATENT_FUNCTION_SIGNATURE}\n\n"
            f"Target: {inst_desc}\n"
            f"  cold_temp={cold:.2f}  hot_temp={hot:.2f}  L={L:.2f}  φ_eq={phi_eq_str}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Fitness = 0.30·T_balance + 0.30·velocity + 0.25·sharpness + 0.15·retention\n"
            f"  T_balance:  Gaussian(mean_T_cold − 0.5, σ=0.10) — latent heat equilibrium\n"
            f"  velocity:   exp(-(log(v/v_stefan))^2/0.25) if v>0, else 0\n"
            f"              v_stefan ≈ {v_stefan_ref} cells/time at r~15 cells\n"
            f"              KEY: retreating fronts (v<0) score 0 — self-regulation required\n"
            f"  sharpness:  Gaussian reward for interface width ≈ 3 cells\n"
            f"  retention:  opposite zone stays in correct phase\n\n"
            f"Top programs:\n\n{exemplars}\n\n"
            f"Rules:\n"
            f"- Return ONLY a single ```cpp code block\n"
            f"- Signature: float reaction(float phi, float temp, float lap_T)\n"
            f"- |reaction| < 50 for all inputs\n"
            f"- No loops, no static state, under 20 lines\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "reaction" in code and "lap_T" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        programs = []
        for name, code in get_seed_programs(
            self.config.instances[0] if self.config.instances else "freeze_latent"
        ):
            prog = CandidateProgram(
                id=str(uuid.uuid4()), code=code,
                island=island_id, generation=generation,
            )
            programs.append(prog)
        return programs

    def save_results(self, programs: list[CandidateProgram]) -> Optional[Path]:
        if not self.config.output_dir:
            return None
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"latent_results_gen{self.generation:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "latent",
            "run_id": self.run_id,
            "generation": self.generation,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {
                    "id": p.id, "fitness": p.fitness, "worst_fitness": p.worst_fitness,
                    "island": p.island, "generation": p.generation,
                    "reaction_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Latent results → %s (best=%.4f)", out_path,
                    top[0].fitness if top else 0)
        return out_path
