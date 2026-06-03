"""SVDAGBeautyKernel — FunSearch kernel evolving compute_density for volcanic SVDAG terrain."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    register_kernel,
)
from . import bridge_eval
from .instance import VolcanoInstance, generate_instance
from .oracle import build_llm_prompt, evaluate_on_instance, get_seed_programs, parse_llm_response

logger = logging.getLogger(__name__)


@register_kernel("svdag_beauty")
class SVDAGBeautyKernel(FunSearchKernel):
    """Evolve a C ``compute_density(world_x, world_z, cell_y, seed)`` that yields
    rocky / porous / volcanic SVDAG terrain.

    Fitness is a CPU membership oracle over the sampled occupancy field (no GPU).
    The best candidate is optionally rendered for real on tengine via the
    tengine.shadergen.eval contract (confirmation only; gated by
    AUTOBENCH_SVDAG_EVAL_RENDER).
    """

    # Emitting on the 'svdag.*' prefix means base._publish unifies to kernel.*
    # with domain='svdag' (svdag must be in kernels.base.KERNEL_DOMAINS).
    BUS_CHANNEL_PREFIX = "svdag"

    def __init__(self, config: KernelConfig):
        if not config.instances:
            config.instances = ["stratovolcano", "lava_field", "caldera"]
        config.island_instance_assignment = True

        from . import ensure_executor
        self.executor = ensure_executor(
            allow_unsandboxed=config.allow_unsandboxed,
            max_memory_mb=config.max_memory_mb,
        )
        logger.info("svdag_beauty sandbox: %s", self.executor.sandbox_type)

        super().__init__(config)
        self.problem_instances = self.load_instances()
        logger.info(
            "Loaded %d volcanic instance(s): %s",
            len(self.problem_instances), [i.name for i in self.problem_instances],
        )

    # ---- FunSearchKernel abstract interface -----------------------------

    def load_instances(self) -> list[VolcanoInstance]:
        return [generate_instance(name) for name in self.config.instances]

    def evaluate_candidate(self, code: str, instance: VolcanoInstance) -> float | None:
        return evaluate_on_instance(code, instance, self.executor, run_timeout=self.config.run_timeout)

    def build_prompt(self, island: Island, top_programs, generation: int, hint: str = "") -> str:
        if self.config.island_instance_assignment and self.problem_instances:
            names = [self.problem_instances[island.id % len(self.problem_instances)].name]
        else:
            names = [i.name for i in self.problem_instances]
        return build_llm_prompt(island, top_programs, generation, instance_names=names, hint=hint)

    def parse_response(self, response: str) -> str:
        return parse_llm_response(response)

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        seeds = get_seed_programs("")
        return [
            CandidateProgram(
                id=f"island{island_id}_gen{generation}_{name}",
                code=code, island=island_id, generation=generation, source="baseline",
            )
            for name, code in seeds
        ]

    def evaluate_fitness(self, program: CandidateProgram) -> tuple[float, float, float]:
        t0 = time.time()
        if self.config.island_instance_assignment and self.problem_instances:
            instances = [self.problem_instances[program.island % len(self.problem_instances)]]
        else:
            instances = self.problem_instances
        scores = [self.evaluate_candidate(program.code, inst) or 0.0 for inst in instances]
        elapsed = (time.time() - t0) * 1000
        if not scores:
            return 0.0, 0.0, 0.0
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        program.fitness = mean
        program.fitness_variance = var
        program.worst_fitness = min(scores)
        program.computation_time_ms = elapsed
        program.evaluated = True
        # Stash the volcanic diagnostics so extract_t_vector (cross-run prior) and
        # the candidate event carry WHAT structure worked, not just a scalar.
        diag = getattr(instances[0], "_last_diag", {}) if instances else {}
        program._svdag_diag = diag if isinstance(diag, dict) else {}
        return mean, var, program.worst_fitness

    def extract_t_vector(self, program: CandidateProgram) -> dict:
        """Sufficient-statistic vector for the consolidated cross-run prior:
        the structural signature (porosity, roughness, spectral slope, relief)
        so successive runs accumulate which volcanic structure scores well."""
        d = getattr(program, "_svdag_diag", {}) or {}
        return {
            "fitness": program.fitness,
            "pore_frac": float(d.get("pore_frac", 0.0)),
            "rough": float(d.get("rough", 0.0)),
            "beta": float(d.get("beta", 0.0)),
            "relief": float(d.get("relief", 0.0)),
            "code_length": float(len(program.code)),
        }

    # ---- artifact render via the tengine eval contract (confirmation) ---

    def _artifact_render_type(self) -> str:
        return "svdag_voxel_iso"

    def _render_best_program(self, best: CandidateProgram, out_path: Path) -> bool:
        """Render the best candidate to a PNG artifact.

        Default: in-house CPU voxel-isometric render (always works, no GPU). If
        AUTOBENCH_SVDAG_EVAL_RENDER is set, additionally request a real tengine
        render via the eval contract and prefer that screenshot when it lands.
        """
        seed = 1337.0
        if self.problem_instances:
            seed = self.problem_instances[best.island % len(self.problem_instances)].seed

        # Optional high-fidelity tengine render (confirmation path).
        if bridge_eval.render_enabled():
            try:
                data = bridge_eval.request_render(self._publish, best.code, best.id)
                src = (data or {}).get("screenshot_path")
                if data and data.get("status") == "ok" and src and Path(src).exists():
                    shutil.copyfile(src, out_path)
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("svdag eval render failed, falling back to CPU: %s", exc)

        # Default CPU render — sample occupancy, isometric voxel projection.
        try:
            from .render import render_density_to_png
            return render_density_to_png(best.code, self.executor, out_path,
                                         res=96, seed=seed, run_timeout=120)
        except Exception as exc:  # noqa: BLE001
            logger.debug("svdag CPU render failed: %s", exc)
            return False

    # ---- results -------------------------------------------------------

    def save_results(self, programs: list[CandidateProgram], path: Path | None = None) -> None:
        if path is None and self.config.output_dir:
            path = self.config.output_dir / f"svdag_beauty_results_gen{self.generation}.json"
        if path is None:
            return
        best = programs[0] if programs else None
        out = {
            "config": {
                "instances": self.config.instances,
                "n_islands": self.config.n_islands,
                "population_per_island": self.config.population_per_island,
                "generations": self.generations,
            },
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "history": self.history,
            "best_program": {
                "id": best.id, "fitness": best.fitness, "density_code": best.code,
                "island": best.island, "generation": best.generation, "source": best.source,
            } if best else None,
            "top_programs": [
                {
                    "id": p.id, "fitness": p.fitness, "worst_fitness": p.worst_fitness,
                    "island": p.island, "generation": p.generation,
                    "computation_time_ms": p.computation_time_ms, "source": p.source,
                    "density_code": p.code,
                    "diag": getattr(self.problem_instances[p.island % len(self.problem_instances)],
                                    "_last_diag", None) if self.problem_instances else None,
                }
                for p in programs[:10]
            ],
        }
        path.write_text(json.dumps(out, indent=2))
        logger.info("Saved svdag_beauty results to %s", path)
