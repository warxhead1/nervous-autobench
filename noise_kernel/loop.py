"""NoiseKernel — the FunSearch kernel class for GPU-evaluated GLSL noise.

Moved verbatim from noise_kernel/__init__.py — no logic changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ..kernels import (
    FunSearchKernel, KernelConfig, CandidateProgram, Island,
    register_kernel,
)
from .instance import (
    NoiseInstance,
    _ISLAND_PERSONAS,
    _REFERENCE_SHADERS,
    _SEEDS,
    build_probe_shader,
    build_reference_shader,
)
from .oracle import _extract_noise_fn
from .spectral import compute_spectral_fitness, _raps_from_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NoiseKernel
# ---------------------------------------------------------------------------

@register_kernel("noise")
class NoiseKernel(FunSearchKernel):
    """FunSearch kernel for GPU-evaluated GLSL noise functions.

    Evolves `float noise(vec3 p)` candidates rendered headless on the GPU via
    ShaderExecutor (moderngl/EGL). Fitness is SSIM against a pre-rendered
    reference PNG for each benchmark instance.

    GLSL runs through the GPU driver, which provides its own isolation. This
    kernel does not apply an OS-level process sandbox (unlike the CPU kernels).

    Subclasses FunSearchKernel and implements the five abstract methods.
    """

    BUS_CHANNEL_PREFIX = "noise"

    def __init__(self, config: KernelConfig):
        super().__init__(config)

        # Lazy-init executor; only imported when needed so the module is cheap to import.
        self._executor: Any = None  # ShaderExecutor, created on first use

        # Populate instances immediately (reference PNGs generated here).
        self.problem_instances = self.load_instances()
        logger.info(
            "Loaded %d noise instance(s): %s",
            len(self.problem_instances),
            [inst.name for inst in self.problem_instances],
        )

    # ------------------------------------------------------------------
    # ShaderExecutor accessor (lazy init, graceful on no GPU)
    # ------------------------------------------------------------------

    def _get_executor(self) -> Any:
        """Return a ShaderExecutor, creating it on first call."""
        if self._executor is not None:
            return self._executor
        try:
            from ..engines.shader_executor import ShaderExecutor  # type: ignore
            self._executor = ShaderExecutor(viewport=(256, 256))
            return self._executor
        except ImportError as e:
            logger.warning("ShaderExecutor import failed (moderngl/numpy missing?): %s", e)
            return None

    # ------------------------------------------------------------------
    # FunSearchKernel abstract interface
    # ------------------------------------------------------------------

    def load_instances(self) -> list[NoiseInstance]:
        """Load benchmark instances, generating reference PNGs if absent."""
        output_dir = self.config.output_dir or Path(tempfile.mkdtemp(prefix="noise_kernel_"))

        instances: list[NoiseInstance] = []
        for name in self.config.instances:
            if name not in _REFERENCE_SHADERS:
                raise ValueError(
                    f"Unknown noise instance '{name}'. "
                    f"Available: {sorted(_REFERENCE_SHADERS)}"
                )
            ref_glsl, description = _REFERENCE_SHADERS[name]
            ref_png = output_dir / f"{name}_reference.png"

            if not ref_png.exists():
                self._render_reference(name, ref_glsl, ref_png)

            ref_raps = _raps_from_image(ref_png) if ref_png.exists() else None

            instances.append(NoiseInstance(
                name=name,
                description=description,
                reference_shader=build_reference_shader(name),
                reference_png=ref_png,
                reference_raps=ref_raps,
            ))

        return instances

    def _render_reference(self, name: str, ref_glsl: str, out_path: Path) -> None:
        """Render the reference PNG for one instance. Logs warning on failure."""
        executor = self._get_executor()
        if executor is None:
            logger.warning(
                "No GPU executor available; skipping reference render for '%s'", name
            )
            return

        shader_body = build_probe_shader(ref_glsl)
        try:
            success, log, render_ms = executor.render(
                shader_body, out_path, viewport=(256, 256), i_time=0.0
            )
            if success:
                logger.info(
                    "Rendered reference '%s' → %s (%.1fms)", name, out_path, render_ms
                )
            else:
                logger.warning(
                    "Reference render failed for '%s': %s", name, log[:400]
                )
        except NotImplementedError as e:
            logger.warning("GPU not available for reference render '%s': %s", name, e)
        except Exception as e:
            logger.warning("Unexpected error rendering reference '%s': %s", name, e)

    def evaluate_candidate(self, code: str, instance: NoiseInstance) -> float | None:
        """Inject noise() into the probe shader, render once at t=0, score via spectral oracle.

        Fitness = spectral class-membership score in [0, 1] (see compute_spectral_fitness).
        Falls back to max(0.0, ssim) if the spectral oracle returns None (missing Pillow,
        unrecognised instance name, etc.).

        Returns None (not 0.0) if:
          - The reference PNG does not exist (no GPU at init time)
          - The candidate fails to compile (CE) or crashes (RE)
          - The GPU render backend is unavailable

        GLSL evaluation trusts the GPU driver's own isolation boundary.
        """
        if not instance.reference_png.exists():
            logger.debug(
                "Skipping evaluation of '%s': reference PNG missing at %s",
                instance.name, instance.reference_png,
            )
            return None

        executor = self._get_executor()
        if executor is None:
            return None

        shader_body = build_probe_shader(code)

        # Evaluate at 3 Z-slices (iTime=0, 3, 7 → Z=0.0, 0.3, 0.7) and average spectral
        # scores. A 2D function that only varies in XY will score poorly at non-zero Z.
        # Single Z=0 evaluation misses 2/3 of the 3D noise structure.
        _Z_SLICES = [0.0, 3.0, 7.0]
        spectral_scores: list[float] = []
        ssim_score = 0.0

        with tempfile.TemporaryDirectory(prefix="noise_eval_") as _tmp:
            from ..engines.shader_executor import Verdict  # type: ignore

            for i_time in _Z_SLICES:
                candidate_png = Path(_tmp) / f"candidate_t{int(i_time*10):02d}.png"
                try:
                    result = executor.run(
                        shader_body,
                        reference_path=instance.reference_png,
                        viewport=(256, 256),
                        i_time=i_time,
                        out_path=candidate_png,
                    )
                except TypeError:
                    try:
                        result = executor.run(
                            shader_body,
                            reference_path=instance.reference_png,
                            viewport=(256, 256),
                            i_time=i_time,
                        )
                        candidate_png = None  # type: ignore[assignment]
                    except Exception as e:
                        logger.warning("Unexpected error on '%s' t=%.1f: %s", instance.name, i_time, e)
                        return None
                except Exception as e:
                    logger.warning("Unexpected error on '%s' t=%.1f: %s", instance.name, i_time, e)
                    return None

                if result.verdict in (Verdict.CE, Verdict.RE):
                    logger.debug("Candidate CE/RE on '%s': %s", instance.name, result.error[:200])
                    return None

                ssim_score = max(ssim_score, max(0.0, result.ssim))

                if candidate_png is not None and Path(candidate_png).exists():
                    spectral = compute_spectral_fitness(
                        candidate_png, instance.name, _stash_target=self
                    )
                    if spectral is not None:
                        spectral_scores.append(spectral)

            if spectral_scores:
                self._last_ssim_fallback = ssim_score
                return sum(spectral_scores) / len(spectral_scores)

            # Spectral oracle unavailable — fall back to SSIM
            logger.debug(
                "Spectral oracle unavailable for '%s'; using SSIM fallback=%.4f",
                instance.name, ssim_score,
            )
            self._last_beta = 0.0
            self._last_hr = 0.0
            self._last_raps_fitness = 0.0
            self._last_ssim_fallback = ssim_score
            return ssim_score

    def evaluate_fitness(self, program: CandidateProgram) -> tuple[float, float, float]:
        """Evaluate program and snapshot spectral features per-program.

        Calls the base-class evaluation loop, then stashes the spectral values
        that compute_spectral_fitness wrote onto self onto the program object.
        This lets extract_t_vector() return correct per-program T-vectors rather
        than kernel-level _last_* attributes that reflect the last evaluated program.
        """
        result = super().evaluate_fitness(program)
        program._spectral_beta = getattr(self, "_last_beta", 0.0)
        program._harmonic_ratio = getattr(self, "_last_hr", 0.0)
        return result

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        """Build the LLM prompt for evolving a new noise() function."""
        persona_name, persona_hint = _ISLAND_PERSONAS[island.id % len(_ISLAND_PERSONAS)]

        exemplars = ""
        for i, p in enumerate(sorted(top_programs, key=lambda x: -x.fitness)[:3]):
            exemplars += f"\n// Exemplar {i+1} (SSIM={p.fitness:.4f}):\n{p.code}\n"

        instance_names = [inst.name for inst in self.problem_instances]
        instance_desc = ", ".join(instance_names)

        hint_section = ""
        if hint:
            hint_section = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        return f"""You are a GLSL shader expert acting as a "{persona_name}".
{persona_hint}
{hint_section}
Write a new GLSL `float noise(vec3 p)` function.

## Signature and contract
```glsl
float noise(vec3 p);   // returns a value in [-1, 1]
```
- Input: a 3D point `p`.
- Output: continuous noise in [-1, 1]. Output range matters — values outside this
  will be clamped in rendering, flattening contrast.
- No textures, no samplers — pure computation only.
- **You may define private helper functions before `noise()`** (e.g. `float hash(vec3 p)`,
  `float valueNoise(vec3 p)`, `float fbm(vec3 p)`). Include them all in the same code block.
  This is required to implement fBm, Worley, or any multi-octave structure.

## Evaluation — spectral oracle (NOT pixel SSIM)
Your function is rendered at 256×256 at **three different Z slices** (iTime=0, 3, 7).
Fitness = average spectral slope match across all three slices:
- Target spectral slope β (power law exponent of log-log RAPS)
- Target harmonic ratio (ratio of 1st and 2nd spectral harmonics)
- Entropy guard: output must not be near-constant or near-binary

**Low-scoring failure modes:**
- All-black or all-white output (constant or clipped → no spectral structure)
- Identical output at all Z values (2D function disguised as 3D → fails Z-slice test)
- Visible axis-aligned grid lines or hard-edged repetition (wrong spectral slope)
- NaN/Inf output

## Benchmark instances
{instance_desc}

## Available GLSL builtins (no textures, no external functions)
`sin`, `cos`, `fract`, `floor`, `ceil`, `round`, `abs`, `mod`,
`mix`, `smoothstep`, `step`, `clamp`, `dot`, `cross`, `length`,
`normalize`, `sqrt`, `pow`, `exp`, `log`, `min`, `max`, `sign`

## Island {island.id} population (generation {generation})
{exemplars if exemplars else "  (no exemplars yet — this is the first generation)"}

## Your task
Write a SINGLE ```glsl code block. Include helper functions first, then `float noise(vec3 p)` last.
Do NOT include `void mainImage`, `#version`, `uniform`, or any other declarations.
Keep the total under 60 lines.
Aim to match the target spectral slope β — smooth, 3D-varying, multi-scale noise wins.
The Z-slice test means a function that ignores `p.z` will fail. Use all three components.
"""

    def parse_response(self, response: str) -> str:
        """Extract a GLSL noise() function body from an LLM response.

        Tries markdown fences first, then bare function definition.
        Returns '' on failure — treated as 0.0 fitness candidate.
        """
        if not response or not response.strip():
            return ""
        text = response.strip()

        # 1. Markdown fenced code block (glsl, glsl|GLSL, or generic)
        fence = re.search(r"```[a-zA-Z0-9_]*[ \t]*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence and fence.group(1).strip():
            code = fence.group(1).strip()
            # If it contains mainImage (LLM over-generated), extract just noise()
            if "void mainImage" in code:
                noise_part = _extract_noise_fn(code)
                if noise_part:
                    return noise_part
            return code

        # 2. Bare float noise( signature
        sig = re.search(r"float\s+noise\s*\(\s*vec3", text)
        if sig:
            return text[sig.start():].strip()

        # 3. Last-resort strip
        return text.replace("```glsl", "").replace("```GLSL", "").replace("```", "").strip()

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        """Return the two baseline seed programs for a fresh island."""
        programs = []
        for variant_name, code in _SEEDS:
            programs.append(CandidateProgram(
                id=f"island{island_id}_gen{generation}_{variant_name}",
                code=code,
                island=island_id,
                generation=generation,
                source="baseline",
            ))
        return programs

    # ------------------------------------------------------------------
    # T-vector extraction — spectral features for ConsolidatedPrior
    # ------------------------------------------------------------------

    def extract_t_vector(self, program: CandidateProgram) -> dict:
        """Return spectral features for this program as a T-vector dict.

        Reads features stashed per-program by evaluate_fitness(), giving
        correct per-program values rather than kernel-level _last_* attributes.
        Falls back to 0.0 for seed programs evaluated before the spectral oracle ran.
        """
        return {
            "fitness": program.fitness,
            "spectral_beta": getattr(program, "_spectral_beta", 0.0),
            "harmonic_ratio": getattr(program, "_harmonic_ratio", 0.0),
        }

    # ------------------------------------------------------------------
    # Candidate event publishing override — adds spectral fields
    # ------------------------------------------------------------------

    def _publish_candidate(self, program: CandidateProgram) -> None:
        """Emit noise.candidate.evaluated.v1 with spectral oracle fields."""
        spectral_beta = getattr(self, "_last_beta", 0.0)
        harmonic_ratio = getattr(self, "_last_hr", 0.0)
        raps_fitness = getattr(self, "_last_raps_fitness", 0.0)
        ssim_fallback = getattr(self, "_last_ssim_fallback", 0.0)

        self._publish(
            "noise.candidate.evaluated.v1",
            {
                "run_id": self.run_id,
                "program_id": program.id,
                "island": program.island,
                "generation": program.generation,
                "fitness": program.fitness,
                "source": program.source,
                "spectral_beta": spectral_beta,
                "harmonic_ratio": harmonic_ratio,
                "raps_fitness": raps_fitness,
                "ssim_fallback": ssim_fallback,
            },
        )

    # ------------------------------------------------------------------
    # Bus publishing — noise.kernel.* channels
    # ------------------------------------------------------------------

    def _render_best_program(self, best: "CandidateProgram", out_path: "Path") -> bool:
        """Render the best noise function to PNG."""
        executor = self._get_executor()
        if executor is None:
            return False
        try:
            shader_body = build_probe_shader(best.code)
            result = executor.render_only(shader_body, out_path=str(out_path),
                                          viewport=(512, 512), i_time=0.0)
            return bool(result.frame_path) and Path(result.frame_path).exists()
        except Exception as e:
            logger.debug("_render_best_program failed: %s", e)
            return False

    def _artifact_render_type(self) -> str:
        return "noise_render"

    def _publish_started(self) -> None:
        """Emit noise.kernel.started.v1 when the run begins."""
        from ..kernels.base import _git_commit_short
        self._publish("noise.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": [inst.name for inst in self.problem_instances],
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
            "gpu_backend": "moderngl/egl",
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        """Emit noise.kernel.completed.v1 when the run ends."""
        best = programs[0] if programs else None
        self._publish("noise.kernel.completed.v1", {
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
                "noise_glsl": best.code if best else "",
            } if best else None,
            "history": self.history,
        })

    def save_results(self, programs: list[CandidateProgram], path: Path | None = None) -> None:
        """Save results to JSON."""
        if path is None and self.config.output_dir:
            path = self.config.output_dir / f"noise_results_gen{self.generation}.json"
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
                "noise_glsl": best.code,
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
                    "noise_glsl": p.code,
                }
                for p in programs[:10]
            ],
        }
        path.write_text(json.dumps(output, indent=2))
        logger.info("Saved noise results to %s", path)
