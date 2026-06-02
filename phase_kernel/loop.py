"""Phase field FunSearch kernel — the registered Kernel class.

Behavior-preserving split of phase_kernel/__init__.py: the PhaseKernel class
(and its @register_kernel("phase") registration) moved here verbatim.
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
    ensure_sandboxed_executor, register_kernel,
)
from .instance import PhaseInstance, generate_instance
from .oracle import PHASE_FUNCTION_SIGNATURE, get_seed_programs
from .scoring import evaluate_on_instance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase FunSearch kernel
# ---------------------------------------------------------------------------

@register_kernel("phase")
class PhaseKernel(FunSearchKernel):
    """FunSearch kernel that evolves Allen-Cahn phase field driving forces.

    Oracle: 1D time-stepper — measures whether the evolved reaction(phi, temp)
    drives a sharp liquid/solid interface to a smooth tanh equilibrium profile
    at the correct temperature response.

    Fitness = 0.5·tanh_score + 0.3·equil_score + 0.2·width_score.

    This is the mathematical substrate for matter state transitions in TEngine:
    the "freeze the river with a spell" mechanic.
    """

    kernel_name = "phase"
    BUS_CHANNEL_PREFIX = "phase"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_sandboxed_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        logger.info("Phase kernel: %d instances, sandbox=%s",
                    len(self.problem_instances), self.executor.sandbox_type)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def load_instances(self) -> list[PhaseInstance]:
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info("Phase instance %r: T=%.2f, D=%.2f, steps=%d",
                        name, inst.temperature, inst.D, inst.n_steps)
            instances.append(inst)
        return instances

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(
            code, instance, self.executor,
            run_timeout=self.config.run_timeout,
            generation=self.generation,  # oracle phase scheduling
        )

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        instance = self.problem_instances[0] if self.problem_instances else None
        inst_desc = (f"{instance.name} (T={instance.temperature:.2f}, "
                     f"D={instance.D:.2f}, steps={instance.n_steps}  — {instance.description})"
                     if instance else "unknown")

        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )

        hint_block = ""
        if hint:
            hint_block = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        temp_str = f"{instance.temperature:.2f}" if instance else "0.5"
        goal_desc = ("solid phase grows (φ→1 spreads)" if instance and instance.temperature < 0.5
                     else ("liquid grows (φ→0 spreads)" if instance and instance.temperature > 0.5
                           else "interface sharpens at equilibrium"))

        return (
            f"You are a computational physics expert specialising in phase field models.\n"
            f"Evolve a better Allen-Cahn driving force reaction(phi, temp).\n\n"
            f"{PHASE_FUNCTION_SIGNATURE}\n\n"
            f"Target: {inst_desc}\n"
            f"Physics goal at T={temp_str}: {goal_desc}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            f"Your goal: maximise Fitness = 0.5·tanh_score + 0.3·equil_score + 0.2·width_score\n"
            f"  tanh_score:  how closely final phi(x) matches a tanh profile\n"
            f"  equil_score: how well the domain ends reach phi=0 and phi=1\n"
            f"  width_score: whether interface width ≈ {instance.target_width if instance else 8} cells\n\n"
            f"Rules:\n"
            f"- Return ONLY the reaction() function in a single ```cpp code block\n"
            f"- Signature: float reaction(float phi, float temp)\n"
            f"- No loops, no static state, no dynamic allocation\n"
            f"- |reaction(phi, temp)| < 50 for all phi,temp ∈ [0,1] (stability checked)\n"
            f"- Under 15 lines\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "reaction" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        seeds = get_seed_programs(
            self.config.instances[0] if self.config.instances else "water_ice_freezing"
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
    # Bus publishing — phase.kernel.* channels
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

        Source is hardcoded to '/autobench/phase_kernel' to match the schema
        const.
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/phase_kernel",
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
        """Emit phase.kernel.started.v1 when the run begins."""
        from ..kernels.base import _git_commit_short
        self._publish("phase.kernel.started.v1", {
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
        """Emit phase.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        self._publish("phase.kernel.completed.v1", {
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
                "reaction_code": best.code if best else "",
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
        out_path = out_dir / f"phase_results_gen{self.generation:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "phase",
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
        logger.info("Phase results → %s (best=%.4f)", out_path,
                    top[0].fitness if top else 0)
        return out_path
