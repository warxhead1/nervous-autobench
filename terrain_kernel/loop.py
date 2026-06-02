"""Terrain FunSearch kernel — the registered Kernel class and its run loop."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    ensure_sandboxed_executor, register_kernel,
)
from .instance import TerrainInstance, generate_instance
from .scoring import evaluate_on_instance
from .oracle import get_seed_programs, TERRAIN_FUNCTION_SIGNATURE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terrain FunSearch kernel
# ---------------------------------------------------------------------------

@register_kernel("terrain")
class TerrainKernel(FunSearchKernel):
    """FunSearch kernel that evolves geological height functions terrain(vec2 p).

    Oracle: Hurst-exponent correctness + normalised slope — measures whether
    the evolved function has the statistical signature of real terrain at a
    specified geological class (rolling hills, mountain peaks, etc.).

    Fitness = 0.6·hurst_score + 0.3·slope_score + 0.1·range_score.
    """

    kernel_name = "terrain"
    BUS_CHANNEL_PREFIX = "terrain"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        logger.info("Terrain kernel: %d instances, sandbox=%s",
                    len(self.problem_instances), self.executor.sandbox_type)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def load_instances(self) -> list[TerrainInstance]:
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info("Terrain instance %r: H_target=%.2f, ns_target=%.2f",
                        name, inst.target_hurst, inst.target_norm_slope)
            instances.append(inst)
        return instances

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(
            code, instance, self.executor,
            run_timeout=self.config.run_timeout,
        )

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        instance = self.problem_instances[0] if self.problem_instances else None
        inst_desc = (f"{instance.name} (H_target={instance.target_hurst:.2f}, "
                     f"ns_target={instance.target_norm_slope:.2f}  — {instance.description})"
                     if instance else "unknown")

        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )

        hint_block = ""
        if hint:
            hint_block = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        return (
            f"You are a terrain generation expert and procedural noise researcher.\n"
            f"Evolve a better geological height function terrain(vec2 p).\n\n"
            f"{TERRAIN_FUNCTION_SIGNATURE}\n\n"
            f"Target: {inst_desc}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            f"Your goal: maximise Fitness = 0.6·hurst_score + 0.3·slope_score + 0.1·range_score\n"
            f"  hurst_score = exp(−8·(H_est − {instance.target_hurst if instance else 0.7:.2f})²)\n"
            f"  slope_score = exp(−5·(norm_slope − {instance.target_norm_slope if instance else 1.5:.2f})² / …)\n"
            f"  H_est is from structure-function log-log regression on a 32×32 sample grid.\n\n"
            f"Rules:\n"
            f"- Return ONLY the terrain() function (+ any helpers) in a single ```cpp code block\n"
            f"- Signature: float terrain(vec2 p)\n"
            f"- You MAY define private helper functions above terrain()\n"
            f"- Fixed-iteration loops (≤ 10 iters) are OK — needed for fBm octaves\n"
            f"- No static mutable globals, no dynamic allocation, no infinite loops\n"
            f"- All finite-float inputs must produce finite-float output\n"
            f"- Under 50 lines total\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+|glsl)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "terrain" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        seeds = get_seed_programs(
            self.config.instances[0] if self.config.instances else "rolling_hills"
        )
        programs = []
        for name, code in seeds:
            prog = CandidateProgram(
                id=str(uuid.uuid4()),
                code=code,
                island=island_id,
                generation=generation,
            )
            programs.append(prog)
        return programs

    # ------------------------------------------------------------------
    # Bus publishing — terrain.kernel.* channels
    # ------------------------------------------------------------------

    def _publish_started(self) -> None:
        """Emit terrain.kernel.started.v1 when the run begins."""
        from ..kernels.base import _git_commit_short
        self._publish("terrain.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": self.config.instances,
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
            "sandbox_type": self.executor.sandbox_type,
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        """Emit terrain.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        self._publish("terrain.kernel.completed.v1", {
            "run_id": self.run_id,
            "total_generations": self.generation,
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "best_program": {
                "id": best.id if best else "",
                "fitness": best.fitness if best else 0.0,
                "island": best.island if best else -1,
                "generation": best.generation if best else -1,
                "source": best.source if best else "",
                "terrain_code": best.code if best else "",
            } if best else None,
            "history": self.history,
        })

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def save_results(self, programs: list[CandidateProgram]) -> Optional[Path]:
        if not self.config.output_dir:
            return None
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"terrain_results_gen{self.generation:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "terrain",
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
                    "terrain_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Terrain results → %s (best=%.4f)", out_path,
                    top[0].fitness if top else 0)

        # Publish best-of-generation to shader studio vault (fire-and-forget)
        if top:
            try:
                from autobench.kernels.bridge import NervousKernelBridge
                bridge = NervousKernelBridge()
                instance = self.config.instances[0] if self.config.instances else "terrain"
                bridge.publish_to_shader_vault(
                    top[0].code,
                    biome=instance,
                    fitness=top[0].fitness,
                    generation=self.generation,
                )
            except Exception as _e:
                pass  # vault publish is best-effort, never blocks evolution

        return out_path
