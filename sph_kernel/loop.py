"""SPH FunSearch kernel — the registered Kernel class.

Moved verbatim from ``sph_kernel/__init__.py`` as part of a behavior-preserving
file split. Importing this module registers ``SPHKernel`` under the ``"sph"``
kernel name via ``@register_kernel``.
"""
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
    ensure_sandboxed_executor, UnsafeSandboxError, register_kernel,
)
from .instance import SPHInstance, generate_instance
from .oracle import get_seed_programs
from .scoring import SPH_FUNCTION_SIGNATURE, evaluate_on_instance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPH FunSearch kernel
# ---------------------------------------------------------------------------

@register_kernel("sph")
class SPHKernel(FunSearchKernel):
    """FunSearch kernel that evolves SPH smoothing functions W(r, h).

    Oracle: density-reconstruction MSE — how accurately Σ mⱼ·W(|x-xⱼ|,h)
    recovers a known Gaussian blob density field at off-lattice probe points.
    Fitness = 1/(1+MSE); higher is better.
    """

    kernel_name = "sph"
    BUS_CHANNEL_PREFIX = "sph"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        logger.info("SPH kernel sandbox: %s", self.executor.sandbox_type)

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def load_instances(self) -> list[SPHInstance]:
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info("Generated SPH instance %r: %d particles, %d probes, h=%.3f",
                        name, inst.n_particles, inst.n_probes, inst.h)
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
        instance_names = [i.name for i in self.problem_instances]
        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )

        hint_block = ""
        if hint:
            hint_block = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        return (
            f"You are a numerical methods expert specialising in SPH fluid simulation kernels.\n"
            f"Evolve a better smoothing kernel W(r,h) for SPH density reconstruction.\n\n"
            f"{SPH_FUNCTION_SIGNATURE}\n\n"
            f"Target instance(s): {', '.join(instance_names)}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            f"Your goal: write a sph_kernel() that MINIMISES density-reconstruction MSE.\n"
            f"Fitness = 1/(1+MSE). Higher fitness = better.\n\n"
            f"Rules:\n"
            f"- Return ONLY the C++ sph_kernel() in a single ```cpp code block\n"
            f"- Signature: extern \"C\" float sph_kernel(float r, float h)\n"
            f"- Include <cmath> functions (sqrtf, fabsf, fmaxf, fminf, powf, expf)\n"
            f"- W(r,h) MUST be 0 for r >= h (compact support — checked as precondition)\n"
            f"- W(0,h) MUST be >= 0 (positivity — checked as precondition)\n"
            f"- No loops, no static arrays — must be evaluable in a single expression path\n"
            f"- Normalisation: 4π∫₀ʰ W(r,h)r²dr should ≈ 1 (partition of unity)\n"
            f"- Keep it under 20 lines\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "sph_kernel" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        seeds = get_seed_programs(
            self.config.instances[0] if self.config.instances else "gauss_blobs_3d"
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
    # Bus publishing — sph.kernel.* channels
    # ------------------------------------------------------------------

    def _find_nervous_bin(self) -> str | None:
        candidates = [
            Path(__file__).parent.parent.parent / "sdk" / "shell" / "nervous",
            Path.home() / "projects" / "nervous-bus" / "sdk" / "shell" / "nervous",
            Path("/usr/local/bin/nervous"),
            Path("/usr/bin/nervous"),
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
        return None

    def _publish(self, channel: str, data: dict) -> bool:
        """Write event to bus debug log and optionally the nervous CLI.

        Source is hardcoded to '/autobench/sph_kernel' to match the schema
        const.
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/sph_kernel",
            "type": channel,
            "datacontenttype": "application/json",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data": data,
        }
        payload = json.dumps(envelope)

        debug_path = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(debug_path, "a") as f:
                f.write(payload + "\n")
            if self.config.bus_verbose:
                logger.info("bus: published %s", channel)
        except Exception as e:
            logger.debug("bus: write to debug log failed: %s", e)

        if self._nervous_bin:
            try:
                env = dict(os.environ)
                env["NBUS_SKIP_VALIDATION"] = "1"
                env["NERVOUS_NO_ZELLIJ"] = "1"
                env["NERVOUS_NO_REDIS"] = "1"
                proc = subprocess.Popen(
                    [self._nervous_bin, "publish", channel, payload],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
                proc.wait(timeout=2)
            except Exception:
                pass

        return True

    def _publish_started(self) -> None:
        """Emit sph.kernel.started.v1 when the run begins."""
        from ..kernels.base import _git_commit_short
        self._publish("sph.kernel.started.v1", {
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
        """Emit sph.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        self._publish("sph.kernel.completed.v1", {
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
                "sph_code": best.code if best else "",
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
        gen = self.generation
        out_path = out_dir / f"sph_results_gen{gen:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "sph",
            "run_id": self.run_id,
            "generation": gen,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {
                    "id": p.id,
                    "fitness": p.fitness,
                    "worst_fitness": p.worst_fitness,
                    "island": p.island,
                    "generation": p.generation,
                    "instance": getattr(p, "instance", ""),
                    "sph_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("SPH results saved to %s (top fitness=%.4f)", out_path,
                    top[0].fitness if top else 0)
        return out_path
