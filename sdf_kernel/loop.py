"""SDFKernel — the FunSearchKernel subclass and its evolution/bus helpers."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
import uuid
from pathlib import Path

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    register_kernel,
)
from .instance import SDFInstance, generate_instance
from .oracle import (
    build_llm_prompt, evaluate_on_instance, get_seed_programs,
    parse_llm_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main kernel class — real FunSearchKernel subclass (instance #2)
# ---------------------------------------------------------------------------

@register_kernel("sdf")
class SDFKernel(FunSearchKernel):
    """FunSearch-style SDF heuristic discovery kernel.

    Subclasses FunSearchKernel and implements the five abstract methods.
    This is the second concrete FunSearch kernel after TSP (instance #1).

    The kernel evolves compact C++ sdf(x,y,z) functions that approximate
    target signed-distance fields. Fitness = 1/(1+MSE), where MSE is
    computed over a fixed set of precomputed sample points per instance.
    The target function is never exposed to the LLM — it only sees sample
    coordinates and expected distances.
    """

    BUS_CHANNEL_PREFIX = "sdf"

    def __init__(self, config: KernelConfig):
        if not config.instances:
            config.instances = ["gyroid", "round_box", "warped_sphere"]

        # Enable island-per-instance routing: each island evaluates against
        # only one instance (island_id % n_instances), specialising its search.
        config.island_instance_assignment = True

        # Build executor BEFORE super().__init__() — fail fast if no sandbox.
        # (The base class does not build an executor; subclasses own this.)
        # Resolve ensure_executor through the package namespace at call time so
        # tests that patch("autobench.sdf_kernel.ensure_executor") take effect.
        from . import ensure_executor
        self.executor = ensure_executor(
            allow_unsandboxed=config.allow_unsandboxed,
            max_memory_mb=config.max_memory_mb,
        )
        logger.info("SDF kernel sandbox: %s", self.executor.sandbox_type)

        super().__init__(config)

        # The base class leaves self.problem_instances = [] (load_instances is
        # abstract and intentionally not called in __init__). We populate it now.
        self.problem_instances = self.load_instances()
        logger.info(
            "Loaded %d SDF instance(s): %s",
            len(self.problem_instances),
            [inst.name for inst in self.problem_instances],
        )

    # ------------------------------------------------------------------
    # FunSearchKernel abstract interface — all five methods
    # ------------------------------------------------------------------

    def load_instances(self) -> list[SDFInstance]:
        """Generate synthetic benchmark instances from config.instances list."""
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info(
                "Generated SDF instance '%s': %d samples, bbox=[%.1f,%.1f]",
                inst.name, inst.n_samples, inst.bbox[0], inst.bbox[1],
            )
            instances.append(inst)
        return instances

    def evaluate_candidate(self, code: str, instance: SDFInstance) -> float | None:
        """Evaluate sdf_code on one SDF instance. Returns fitness in (0,1] or None."""
        return evaluate_on_instance(
            code,
            instance,
            self.executor,
            run_timeout=self.config.run_timeout,
        )

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        """Build LLM prompt for evolving a sdf() function.

        With island_instance_assignment=True, each island evaluates only its
        assigned instance — pass only that name so the LLM knows its target.
        """
        if self.config.island_instance_assignment and self.problem_instances:
            inst_idx = island.id % len(self.problem_instances)
            instance_names = [self.problem_instances[inst_idx].name]
        else:
            instance_names = [inst.name for inst in self.problem_instances]
        return build_llm_prompt(
            island,
            top_programs,
            generation,
            instance_names=instance_names,
            hint=hint,
        )

    def parse_response(self, response: str) -> str:
        """Extract C++ sdf() from an LLM response."""
        return parse_llm_response(response)

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        """Return 3 analytical baseline programs for a new island.

        With island_instance_assignment=True, pick seeds relevant to the
        instance this island will specialise on.
        """
        if self.config.island_instance_assignment and self.problem_instances:
            inst_idx = island_id % len(self.problem_instances)
            first_name = self.problem_instances[inst_idx].name
        else:
            first_name = self.problem_instances[0].name if self.problem_instances else "generic"
        seeds = get_seed_programs(first_name)

        programs = []
        for variant_name, code in seeds:
            programs.append(CandidateProgram(
                id=f"island{island_id}_gen{generation}_{variant_name}",
                code=code,
                island=island_id,
                generation=generation,
                source="baseline",
            ))
        return programs

    def evaluate_fitness(self, program: CandidateProgram) -> tuple[float, float, float]:
        """Evaluate program against one or all instances.

        With island_instance_assignment=True, each island evaluates only the
        single instance whose index matches (island_id % n_instances).  This
        specialises the search so that islands compete for different instances
        instead of averaging over all three, which would blur the per-topology
        gradient signal.
        """
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
        # Snapshot per-instance diagnostics onto the program now, while the instance
        # attributes are current for this evaluation. extract_t_vector() reads these
        # per-program attrs rather than the stale instance attrs, giving correct T-vectors
        # when iterating over multiple programs in the prior append loop.
        g_errs = [getattr(i, "_last_grad_err", None) for i in instances]
        t_scores = [getattr(i, "_last_topology_score", None) for i in instances]
        valid_ge = [g for g in g_errs if g is not None]
        valid_t = [t for t in t_scores if t is not None]
        program._eikonal_score = math.exp(-0.5 * sum(valid_ge) / len(valid_ge)) if valid_ge else 0.0
        program._topology_score = sum(valid_t) / len(valid_t) if valid_t else 0.0
        return mean, var, worst

    def extract_t_vector(self, program: CandidateProgram) -> dict:
        """Extract sufficient-statistic T-vector for this program.

        Reads eikonal and topology scores stashed per-program by evaluate_fitness(),
        giving correct per-program values rather than stale instance attributes.
        """
        return {
            "fitness": program.fitness,
            "eikonal_score": getattr(program, "_eikonal_score", 0.0),
            "topology_score": getattr(program, "_topology_score", 0.0),
            "code_length": len(program.code),
        }

    # ------------------------------------------------------------------
    # Bus publishing — emit on schema'd channels.
    # _publish_candidate: overridden below (adds topology + eikonal fields).
    # _publish_generation: not overridden — base class handles generation events.
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
        """Publish to bus debug log and optionally the nervous CLI.

        Source is hardcoded to '/autobench/sdf_kernel' to match the schema
        const — the base class uses the class name dynamically, which would
        produce '/autobench/sdfkernel' (no underscore).
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/sdf_kernel",
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

    def _render_best_program(self, best: "CandidateProgram", out_path: "Path") -> bool:
        """Render the evolved SDF via in-house CPU sphere tracer (fallback: GLSL probe)."""
        # Prefer the CPU tracer — it works headless with zero VRAM, and takes
        # the C++ code directly without translation. Fall back to the GLSL path
        # (ShaderExecutor) if the CPU tracer isn't available.
        if self.config.allow_unsandboxed:
            try:
                from ..engines.sdf_tracer import render_sdf_cpp_to_png  # type: ignore
                from ..artifact_store import _INSTANCE_CAMERA_DIST  # type: ignore
                inst_name = (
                    self.problem_instances[best.island % len(self.problem_instances)].name
                    if self.problem_instances else ""
                )
                cam_dist = _INSTANCE_CAMERA_DIST.get(inst_name, 3.5)
                return render_sdf_cpp_to_png(
                    best.code, out_path, viewport=(256, 256), camera_dist=cam_dist
                )
            except Exception as e:
                logger.debug("CPU tracer unavailable (%s), trying GLSL path", e)

        from ..artifact_store import render_sdf_to_png  # type: ignore
        glsl = best.code.replace(
            "extern \"C\" float sdf(float x, float y, float z)",
            "float sdf(vec3 p)"
        )
        if "float x=p.x" not in glsl:
            glsl = glsl.replace(
                "float sdf(vec3 p) {",
                "float sdf(vec3 p) {\n    float x=p.x, y=p.y, z=p.z;"
            )
        return render_sdf_to_png(glsl, out_path)

    def _artifact_render_type(self) -> str:
        return "sdf_raymarch"

    def _publish_started(self) -> None:
        """Emit sdf.kernel.started.v1 when the run begins."""
        from ..kernels.base import _git_commit_short
        self._publish("sdf.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": [inst.name for inst in self.problem_instances],
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
            "sandbox_type": self.executor.sandbox_type,
        })

    def _publish_candidate(self, candidate: CandidateProgram) -> None:
        """Emit sdf.candidate.evaluated.v1 with eikonal and topology diagnostics."""
        if not candidate.evaluated or candidate.fitness <= 0:
            return
        # Collect per-instance diagnostics stashed by evaluate_on_instance.
        grad_errs = [getattr(inst, "_last_grad_err", None)
                     for inst in self.problem_instances]
        mses = [getattr(inst, "_last_mse", None)
                for inst in self.problem_instances]
        topo_scores = [getattr(inst, "_last_topology_score", None)
                       for inst in self.problem_instances]
        sign_densities = [getattr(inst, "_last_sign_change_density", None)
                          for inst in self.problem_instances]

        valid_ge = [g for g in grad_errs if g is not None]
        valid_mse = [m for m in mses if m is not None]
        valid_topo = [t for t in topo_scores if t is not None]
        valid_density = [d for d in sign_densities if d is not None]

        mean_grad_err = sum(valid_ge) / len(valid_ge) if valid_ge else None
        mean_mse = sum(valid_mse) / len(valid_mse) if valid_mse else None
        mean_topo = sum(valid_topo) / len(valid_topo) if valid_topo else None
        mean_density = sum(valid_density) / len(valid_density) if valid_density else None

        self._publish("sdf.candidate.evaluated.v1", {
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": candidate.id,
            "island": candidate.island,
            "fitness": round(candidate.fitness, 6),
            "fitness_variance": round(candidate.fitness_variance, 6),
            "worst_fitness": round(candidate.worst_fitness, 6),
            "eikonal_score": round(math.exp(-0.5 * mean_grad_err), 4) if mean_grad_err is not None else None,
            "mean_grad_err": round(mean_grad_err, 4) if mean_grad_err is not None else None,
            "mean_mse": round(mean_mse, 6) if mean_mse is not None else None,
            "topology_score": round(mean_topo, 4) if mean_topo is not None else None,
            "sign_change_density": round(mean_density, 6) if mean_density is not None else None,
            "source": candidate.source,
            "code_length": len(candidate.code),
            "computation_time_ms": round(candidate.computation_time_ms, 1),
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        """Emit sdf.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        # Include eikonal diagnostics in the best_program block.
        best_grad_err = None
        if best:
            grad_errs = [getattr(inst, "_last_grad_err", None)
                         for inst in self.problem_instances]
            valid = [g for g in grad_errs if g is not None]
            best_grad_err = sum(valid) / len(valid) if valid else None
        self._publish("sdf.kernel.completed.v1", {
            "run_id": self.run_id,
            "total_generations": self.generation,
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "best_program": {
                "id": best.id if best else "",
                "fitness": best.fitness if best else 0.0,
                "eikonal_score": round(math.exp(-0.5 * best_grad_err), 4)
                                 if best_grad_err is not None else None,
                "mean_grad_err": round(best_grad_err, 4) if best_grad_err is not None else None,
                "island": best.island if best else -1,
                "generation": best.generation if best else -1,
                "source": best.source if best else "",
                "sdf_code": best.code if best else "",
            } if best else None,
            "history": self.history,
        })

    def save_results(self, programs: list[CandidateProgram], path: Path | None = None) -> None:
        """Save results to JSON (mirrors TSP convention)."""
        if path is None and self.config.output_dir:
            path = self.config.output_dir / f"sdf_results_gen{self.generation}.json"
        if path is None:
            return
        best = programs[0] if programs else None
        output = {
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
                "id": best.id,
                "fitness": best.fitness,
                "sdf_code": best.code,
                "source": best.source,
                "island": best.island,
                "generation": best.generation,
            } if best else None,
            "top_programs": [
                {
                    "id": p.id,
                    "fitness": p.fitness,
                    "fitness_variance": p.fitness_variance,
                    "worst_fitness": p.worst_fitness,
                    "island": p.island,
                    "generation": p.generation,
                    "computation_time_ms": p.computation_time_ms,
                    "source": p.source,
                    "sdf_code": p.code,
                }
                for p in programs[:10]
            ],
        }
        path.write_text(json.dumps(output, indent=2))
        logger.info("Saved SDF results to %s", path)
