"""TSPKernel FunSearch loop + island/evolution helpers.

The kernel class drives generation steps, migration, plateau handling, and bus
publishing. The module-level helpers (code diversity, baseline seeding, island
init, migration, island evaluation) are the population-management primitives the
loop is built from.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .instance import (
    CandidateProgram,
    Island,
    KNOWN_OPTIMALS,
    TSPInstance,
    fetch_tsplib_instance,
    new_ulid,
)
from .oracle import build_llm_prompt, mutate_priority, parse_llm_response
from .scoring import (
    FunSearchKernel,
    KernelConfig,
    UnsafeSandboxError,
    ensure_sandboxed_executor,
    evaluate_fitness,
    evaluate_on_instance,
    register_kernel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code diversity helpers — module-level, used in step() diagnostics
# ---------------------------------------------------------------------------

def _code_tokens(code: str) -> frozenset:
    return frozenset(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', code))


def _code_diversity(a: str, b: str) -> float:
    ta, tb = _code_tokens(a), _code_tokens(b)
    if not (ta | tb):
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def _population_code_diversity(codes: list) -> float:
    if len(codes) < 2:
        return 1.0
    dists = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            dists.append(_code_diversity(a, b))
    return sum(dists) / len(dists) if dists else 1.0


# ---------------------------------------------------------------------------
# Island model population management
# ---------------------------------------------------------------------------

def init_baseline_programs(island_id: int, generation: int) -> list[CandidateProgram]:
    """Return seeded baseline programs for an island."""
    return [
        CandidateProgram(
            id=f"island{island_id}_gen{generation}_nn",
            priority_code="""extern "C" double priority(int node, const Instance* inst, const State* state) {
    double best = 1e100;
    for (int t : state->current_tour) best = min(best, inst->dist[node][t]);
    return -best;
}""",
            island=island_id,
            generation=generation,
            source="baseline",
        ),
        CandidateProgram(
            id=f"island{island_id}_gen{generation}_fi",
            priority_code="""extern "C" double priority(int node, const Instance* inst, const State* state) {
    double worst = 0.0;
    for (int t : state->current_tour) worst = max(worst, inst->dist[node][t]);
    return worst;
}""",
            island=island_id,
            generation=generation,
            source="baseline",
        ),
        CandidateProgram(
            id=f"island{island_id}_gen{generation}_angle",
            priority_code="""extern "C" double priority(int node, const Instance* inst, const State* state) {
    double cx = 0, cy = 0;
    for (int i = 0; i < inst->n; i++) { cx += inst->coords[i].first; cy += inst->coords[i].second; }
    cx /= inst->n; cy /= inst->n;
    double dx = inst->coords[node].first - cx;
    double dy = inst->coords[node].second - cy;
    return atan2(dy, dx);
}""",
            island=island_id,
            generation=generation,
            source="baseline",
        ),
    ]


def initialize_islands(n_islands: int, pop_size: int, generation: int = 0) -> list[Island]:
    """Initialize islands with seeded baselines."""
    islands = []
    for i in range(n_islands):
        island = Island(id=i)
        # Add baseline seeds
        island.population.extend(init_baseline_programs(i, generation))
        # Fill rest with mutated copies of best baseline
        while len(island.population) < pop_size:
            base = island.population[i % len(island.population)]
            mutated = mutate_priority(base.priority_code, i)
            p = CandidateProgram(
                id=f"island{i}_gen{generation}_mut{len(island.population)}",
                priority_code=mutated,
                island=i,
                generation=generation,
                source="mutated",
            )
            island.population.append(p)
        island.best_program = max(island.population, key=lambda p: p.fitness)
        islands.append(island)
    return islands


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(islands: list[Island], interval: int = 10) -> None:
    """Migrate best programs between islands every `interval` generations."""
    for i, island in enumerate(islands):
        donor = islands[(i + 1) % len(islands)]
        if donor.best_program and island.population:
            # Replace a random individual — avoids always evicting the weakest
            # which can homogenize the island when all members are near-equal.
            import random
            replace_idx = random.randint(0, len(island.population) - 1)
            migrant = CandidateProgram(
                id=f"migrated_from_island{donor.id}_to_island{island.id}",
                priority_code=donor.best_program.priority_code,
                island=island.id,
                generation=island.population[0].generation,
                source="migrated",
            )
            island.population[replace_idx] = migrant


# ---------------------------------------------------------------------------
# Island-level evaluation (evaluate entire island population)
# ---------------------------------------------------------------------------

def evaluate_island(
    island: Island,
    instances: list[TSPInstance],
    work_dir: Path | None = None,
    executor: "SandboxedExecutor | None" = None,
    run_timeout: float = 10.0,
) -> None:
    """Evaluate not-yet-evaluated programs in an island, update fitness + best.

    Programs carry an ``evaluated`` flag, so a candidate is scored exactly once
    across its lifetime — surviving programs are not re-run every generation.
    Only freshly inserted programs (new migrants, new candidates) are evaluated.
    """
    # Resolve evaluate_fitness through the package namespace so that
    # monkeypatch.setattr(autobench.tsp_kernel, "evaluate_fitness", ...) — used
    # in the throughput tests and any caller that patched the old monolithic
    # module — still intercepts the call after the file split.
    import autobench.tsp_kernel as _pkg
    _evaluate_fitness = getattr(_pkg, "evaluate_fitness", evaluate_fitness)

    for prog in island.population:
        if prog.evaluated:
            continue
        _evaluate_fitness(prog, instances, work_dir, executor=executor, run_timeout=run_timeout)

    valid = [p for p in island.population if p.fitness > 0]
    if valid:
        island.best_program = max(valid, key=lambda p: p.fitness)


# ---------------------------------------------------------------------------
# Main kernel class
# ---------------------------------------------------------------------------

# KernelConfig is now imported from autobench.kernels (single source of truth).
# The duplicate definition that used to live here was removed in Phase 1 of
# the autobench kernels restructuring.


@register_kernel("tsp")
class TSPKernel(FunSearchKernel):
    """FunSearch-style TSP heuristic discovery kernel."""

    BUS_CHANNEL_PREFIX = "tsp"

    def __init__(self, config: KernelConfig):
        self.config = config
        self.islands: list[Island] = []
        self.generation = 0
        self.generations = config.generations  # total generations to run
        self.history: list[dict] = []
        self.llm_requests = 0  # MiniMax billing unit is requests, not dollars
        self._llm_lock = threading.Lock()  # guards llm_requests across threads
        self._plateau_count = 0   # generations since the last fitness improvement
        self._run_start = 0.0     # wall-clock start of run()
        self.stop_reason = ""     # why the run loop ended (horizon governance)
        self.run_id = new_ulid()  # sortable ULID, per the tsp.* schemas
        self._active_hint: str = ""  # strategic hint from deer query on plateau
        self._last_published_best_fitness: float = 0.0
        self._island_age: dict[int, int] = {}
        self._island_plateau_counts: dict[int, int] = {}

        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect nervous CLI
        self._nervous_bin = config.nervous_bin or self._find_nervous_bin()

        # Build the isolating sandbox up front so we fail fast (before any
        # untrusted code is compiled) if the host can't provide isolation.
        self.executor = ensure_sandboxed_executor(
            allow_unsandboxed=config.allow_unsandboxed,
            max_memory_mb=config.max_memory_mb,
        )
        logger.info("TSP kernel sandbox: %s", self.executor.sandbox_type)

        # Load TSPLIB instances
        self.instances: list[TSPInstance] = []
        for name in config.instances:
            inst = fetch_tsplib_instance(name)
            # Load known optimal tour lengths
            opt = KNOWN_OPTIMALS.get(name.lower())
            if opt:
                inst.optimal_tour_length = opt
            self.instances.append(inst)
            logger.info("Loaded %s: %d nodes, optimal=%s", name, inst.n, opt)
        # Sync to kernel_base's problem_instances so inherited utilities
        # (evaluate_fitness, evaluate_island, island_reset) work for TSP.
        self.problem_instances = self.instances

    # ------------------------------------------------------------------
    # FunSearchKernel abstract method implementations
    # The TSP loop (run/step/initialize) is fully overridden here, so
    # these five methods exist primarily to satisfy the ABC contract and
    # allow kernel_base.FunSearchKernel.run() to drive TSP if ever called.
    # ------------------------------------------------------------------

    def load_instances(self) -> list[TSPInstance]:
        """Return the already-loaded TSPInstance list (loaded in __init__)."""
        return self.instances

    def evaluate_candidate(self, code: str, instance: "TSPInstance") -> float | None:
        """Evaluate a priority() code string on one TSPInstance via sandbox.

        Returns an approx-ratio in (0, 1] (higher = better), or None on failure.
        """
        result = evaluate_on_instance(code, instance, self.executor, self.config.run_timeout)
        if result is None:
            return None
        if instance.optimal_tour_length:
            return instance.optimal_tour_length / result.length
        return 1.0  # unknown optimal

    def build_prompt(self, island: "Island", top_programs: "list[CandidateProgram]", generation: int, hint: str = "") -> str:
        """Build the LLM prompt for the given island and generation."""
        return build_llm_prompt(island, top_programs, generation, hint=hint)

    def parse_response(self, response: str) -> str:
        """Extract a C++ priority() function from an LLM response string."""
        return parse_llm_response(response)

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        """Return deterministic baseline programs for a fresh island."""
        return init_baseline_programs(island_id, generation)

    def _default_llm_call(self, prompt: str) -> str:
        """Generate a candidate via the LLM. Returns "" on any failure.

        Every attempt counts as one MiniMax request (tracked in self.llm_requests
        — MiniMax billing is per-request, not dollars). Timeouts and non-zero
        exits are swallowed into "" so a flaky model call scores the candidate
        0.0 rather than crashing the generation.
        """
        with self._llm_lock:
            self.llm_requests += 1  # called from a thread pool — keep the count exact

        if self.config.llm_call_fn:
            try:
                return self.config.llm_call_fn(prompt)
            except Exception as e:  # noqa: BLE001
                logger.warning("custom llm_call_fn failed: %s", e)
                return ""

        try:
            result = subprocess.run(
                [
                    "deer", "query", "--model", "minimax-m2.7", "--terse",
                    "--temperature", str(self.config.temperature),
                    "TSP priority function generation. Respond with ONLY a single "
                    "```cpp code block containing the function, no explanation.\n"
                    f"Prompt:\n{prompt}",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.llm_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("LLM call timed out after %ss", self.config.llm_timeout)
            return ""
        except Exception as e:  # noqa: BLE001 — deer missing, OS error, etc.
            logger.warning("LLM call failed: %s", e)
            return ""

        if result.returncode != 0:
            logger.warning("deer query exit %d: %s", result.returncode, result.stderr[:200])
            return ""
        return result.stdout

    def _get_plateau_hint(self, best_program: "CandidateProgram") -> str:
        """Call deer query for strategic advice when evolution plateaus."""
        prompt = (
            f"A TSP nearest-neighbor construction heuristic has plateaued at fitness "
            f"{best_program.fitness:.4f} (approx ratio = optimal/candidate_tour_length). "
            f"The current best priority function is:\n\n"
            f"```cpp\n{best_program.priority_code}\n```\n\n"
            f"Identify ONE specific structural weakness in this function and suggest a concrete "
            f"algorithmic change (e.g. a different signal to add, a normalization to apply, "
            f"a term to remove, a non-linearity to introduce). Be specific — name the change "
            f"and explain why it should improve the approx ratio. Under 4 sentences."
        )
        try:
            result = subprocess.run(
                ["deer", "query", "--model", self.config.plateau_hint_model, "--terse", prompt],
                capture_output=True, text=True, timeout=60.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.warning("plateau hint call failed: %s", e)
        return ""

    def initialize(self) -> None:
        """Initialize island population with baselines."""
        self.islands = initialize_islands(
            self.config.n_islands,
            self.config.population_per_island,
            generation=0,
        )
        for island in self.islands:
            evaluate_island(
                island, self.instances,
                executor=self.executor, run_timeout=self.config.run_timeout,
            )
        logger.info("Initialized %d islands with %d programs each",
                    self.config.n_islands, self.config.population_per_island)

        from ..kernels.base import _git_commit_short
        self._publish("tsp.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": [inst.name for inst in self.instances],
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
            "sandbox_type": self.executor.sandbox_type,
        })

    def _generate_candidates(self) -> list[list[CandidateProgram]]:
        """Generate candidates for every island concurrently.

        Fires ``candidates_per_island`` LLM calls per island, up to
        ``max_concurrent_llm`` at once. The calls are network-bound subprocesses
        (deer query) and don't touch the sandbox executor, so a thread pool is
        safe and turns N serial ~28s calls into one concurrent wave — the point
        of MiniMax's high request budget. Each result is parsed; failures become
        empty code that scores 0.0 downstream.
        """
        from concurrent.futures import ThreadPoolExecutor

        tasks: list[tuple[int, int, str]] = []
        for i, island in enumerate(self.islands):
            top_programs = sorted(island.population, key=lambda p: -p.fitness)[:3]
            prompt = build_llm_prompt(island, top_programs, self.generation, hint=self._active_hint)
            for k in range(max(1, self.config.candidates_per_island)):
                tasks.append((i, k, prompt))

        new_programs: list[list[CandidateProgram]] = [[] for _ in self.islands]
        workers = max(1, min(self.config.max_concurrent_llm, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda t: (t[0], t[1], self._default_llm_call(t[2])), tasks)
            for island_idx, cand_idx, response in results:
                code = parse_llm_response(response)
                new_programs[island_idx].append(CandidateProgram(
                    id=f"island{island_idx}_gen{self.generation}_llm{cand_idx}",
                    priority_code=code,
                    island=island_idx,
                    generation=self.generation,
                    source="llm",
                ))
        return new_programs

    def step(self) -> None:
        """One generation: generate new candidates, evaluate, migrate, collect."""
        t_gen0 = time.time()
        new_programs = self._generate_candidates()
        gen_seconds = time.time() - t_gen0

        t_eval0 = time.time()
        # Evaluate all new candidates
        for i, island in enumerate(self.islands):
            for candidate in new_programs[i]:
                try:
                    evaluate_fitness(
                        candidate, self.instances,
                        executor=self.executor, run_timeout=self.config.run_timeout,
                    )
                    # Publish candidate evaluated event
                    self._publish("tsp.candidate.evaluated.v1", {
                        "run_id": self.run_id,
                        "program_id": candidate.id,
                        "island": candidate.island,
                        "generation": candidate.generation,
                        "source": candidate.source,
                        "fitness": candidate.fitness,
                        "fitness_variance": candidate.fitness_variance,
                        "worst_fitness": candidate.worst_fitness,
                        "instance_names": [inst.name for inst in self.instances],
                    })
                    # Soft elitism: accept if better than weakest or diverse+good
                    worst = min(island.population, key=lambda p: p.fitness)
                    if candidate.fitness > worst.fitness:
                        island.population.append(candidate)
                    # Trim to population size
                    island.population.sort(key=lambda p: -p.fitness)
                    island.population = island.population[:self.config.population_per_island]
                except Exception as e:
                    logger.warning("Candidate failed: %s", e)
                    continue

        # Ring migration removed — collapsed multi-basin search into single basin
        # within ~20 generations. island_reset() in run() culls weak islands at
        # plateau midpoint instead, preserving search diversity.

        # Re-evaluate islands after changes — eval-once means only freshly
        # inserted migrants actually run here, not the whole population.
        for island in self.islands:
            evaluate_island(
                island, self.instances,
                executor=self.executor, run_timeout=self.config.run_timeout,
            )
        eval_seconds = time.time() - t_eval0

        # Record generation stats
        all_best = [island.best_program for island in self.islands if island.best_program]
        if all_best:
            best = max(all_best, key=lambda p: p.fitness)

            # Fitness standard deviation across all active programs
            all_fitness = [p.fitness for isl in self.islands for p in isl.population if p.fitness > 0]
            if len(all_fitness) > 1:
                mean_f = sum(all_fitness) / len(all_fitness)
                fitness_std = (sum((f - mean_f) ** 2 for f in all_fitness) / (len(all_fitness) - 1)) ** 0.5
            else:
                fitness_std = 0.0

            # Code token diversity: mean pairwise Jaccard distance
            all_codes = [p.priority_code for isl in self.islands for p in isl.population if p.priority_code]
            code_diversity = _population_code_diversity(all_codes)

            self.history.append({
                "generation": self.generation,
                "best_fitness": best.fitness,
                "best_id": best.id,
                "best_island": best.island,
                "mean_pop_fitness": sum(
                    max(p.fitness for p in isl.population) for isl in self.islands
                ) / len(self.islands),
                "gen_seconds": round(gen_seconds, 2),
                "eval_seconds": round(eval_seconds, 2),
                "llm_requests": self.llm_requests,
                "fitness_std": round(fitness_std, 6),
                "code_diversity": round(code_diversity, 6),
                "island_reset_fired": False,  # set True in run() after island_reset fires
            })
            logger.debug(
                "Gen %d: fitness_std=%.4f code_diversity=%.3f",
                self.generation, fitness_std, code_diversity,
            )
            logger.info(
                "Gen %3d | best=%.4f (island%d) | mean_pop=%.4f | gen=%.1fs eval=%.1fs reqs=%d",
                self.generation, best.fitness, best.island,
                self.history[-1]["mean_pop_fitness"],
                gen_seconds, eval_seconds, self.llm_requests,
            )

            # Publish generation completed event
            self._publish("tsp.generation.completed.v1", {
                "run_id": self.run_id,
                "generation": self.generation,
                "best_fitness": best.fitness,
                "best_island": best.island,
                "best_program_id": best.id,
                "mean_pop_fitness": self.history[-1]["mean_pop_fitness"],
                "fitness_std": self.history[-1]["fitness_std"],
                "code_diversity": self.history[-1]["code_diversity"],
                "island_summaries": [
                    {
                        "island": isl.id,
                        "best_fitness": isl.best_program.fitness if isl.best_program else 0.0,
                        "best_program_id": isl.best_program.id if isl.best_program else "",
                        "population_size": len(isl.population),
                    }
                    for isl in self.islands
                ],
            })

            # Cross-cutting events (mirrored from kernels/base.py — TSP overrides
            # step() without calling super(), so these must be duplicated here)
            for island in self.islands:
                self._island_age[island.id] = self._island_age.get(island.id, 0) + 1
                self._publish("autobench.island.health.v1", {
                    "run_id": self.run_id,
                    "generation": self.generation,
                    "island": island.id,
                    "plateau_count": self._island_plateau_counts.get(island.id, 0),
                    "population_size": len(island.population),
                    "age_since_last_reset": self._island_age.get(island.id, 0),
                })

            if self.generation % 5 == 0 and self.config.max_requests:
                gen_est = max(1, self.generation)
                rate = self.llm_requests / gen_est
                remaining_gens = int((self.config.max_requests - self.llm_requests) / rate) if rate > 0 else 0
                self._publish("autobench.budget.gauge.v1", {
                    "run_id": self.run_id,
                    "generation": self.generation,
                    "requests_used": self.llm_requests,
                    "max_requests": self.config.max_requests,
                    "estimated_remaining_generations": remaining_gens,
                })

            if best.fitness - self._last_published_best_fitness > 1e-6:
                improvement_delta = best.fitness - self._last_published_best_fitness
                self._publish("tsp.best_fitness_improved.v1", {
                    "run_id": self.run_id,
                    "generation": self.generation,
                    "best_fitness": best.fitness,
                    "improvement_delta": round(improvement_delta, 6),
                    "best_program_id": best.id,
                    "best_island": best.island,
                })
                self._last_published_best_fitness = best.fitness

        self.generation += 1

    def global_best_fitness(self) -> float:
        """Best fitness across all islands right now (0.0 if none scored)."""
        bests = [isl.best_program.fitness for isl in self.islands if isl.best_program]
        return max(bests) if bests else 0.0

    def _next_wave_requests(self) -> int:
        """How many LLM requests the next generation will issue."""
        return len(self.islands) * max(1, self.config.candidates_per_island)

    def _horizon_reason(self) -> str | None:
        """Return a human-readable stop reason if any horizon is hit, else None.

        Checked at the TOP of each generation so we never start a generation we
        don't need (and the request budget is never overshot by a whole wave).
        """
        c = self.config
        best = self.global_best_fitness()
        if c.target_fitness is not None and best >= c.target_fitness:
            return f"target_fitness reached (best={best:.4f} >= {c.target_fitness})"
        if c.max_requests is not None and self.llm_requests + self._next_wave_requests() > c.max_requests:
            return (f"request budget {c.max_requests} reached "
                    f"(used={self.llm_requests}, next wave={self._next_wave_requests()})")
        if c.max_wall_seconds is not None and (time.time() - self._run_start) >= c.max_wall_seconds:
            return f"wall-clock budget {c.max_wall_seconds:.0f}s reached"
        if c.plateau_generations is not None and self._plateau_count >= c.plateau_generations:
            return (f"plateau: no improvement > {c.plateau_epsilon} "
                    f"in {c.plateau_generations} generations")
        return None

    def run(self) -> list[CandidateProgram]:
        """Run the FunSearch loop until the generation cap OR a horizon is hit."""
        self.initialize()
        self._run_start = time.time()
        self._plateau_count = 0
        self._island_plateau_counts = {}
        self._last_published_best_fitness = 0.0
        self._island_age = {}
        self.stop_reason = ""

        for _ in range(self.generations):
            reason = self._horizon_reason()
            if reason:
                self.stop_reason = reason
                logger.info("Horizon stop: %s", reason)
                break
            prev_best = self.global_best_fitness()
            self.step()
            if self.global_best_fitness() - prev_best > self.config.plateau_epsilon:
                self._plateau_count = 0
                self._active_hint = ""  # clear hint when a new best is found
            else:
                self._plateau_count += 1
                midpoint = (self.config.plateau_generations or 6) // 2
                if self._plateau_count == midpoint:
                    pre_best = self.global_best_fitness()
                    n_culled = max(1, len(self.islands) * 2 // 5)
                    self.island_reset()
                    self._publish_island_reset(n_culled, pre_best)
                    if self.history:
                        self.history[-1]["island_reset_fired"] = True
                    if self.config.plateau_hint and not self._active_hint:
                        best = max(
                            (p for isl in self.islands for p in isl.population if p.fitness > 0),
                            key=lambda p: p.fitness,
                            default=None,
                        )
                        if best:
                            hint = self._get_plateau_hint(best)
                            if hint:
                                self._active_hint = hint
                                self._publish_plateau_hint(hint, best.fitness)
                                logger.info("plateau hint: %s", hint[:120])
        if not self.stop_reason:
            self.stop_reason = f"generation cap {self.generations} reached"

        # Return all programs sorted by fitness
        all_programs = []
        for island in self.islands:
            all_programs.extend(island.population)
        all_programs.sort(key=lambda p: -p.fitness)

        logger.info(
            "Run ended: %s | generations=%d requests=%d best=%.4f",
            self.stop_reason, self.generation, self.llm_requests, self.global_best_fitness(),
        )

        # Publish kernel completed event
        best = all_programs[0] if all_programs else None
        self._publish("tsp.kernel.completed.v1", {
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
                "priority_code": best.priority_code if best else "",
            } if best else None,
            "history": self.history,
        })

        return all_programs

    def save_results(self, programs: list[CandidateProgram], path: Path | None = None) -> None:
        """Save results to JSON."""
        if path is None and self.config.output_dir:
            path = self.config.output_dir / f"results_gen{self.generation}.json"

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
            "best_program": None,
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
                    "priority_code": p.priority_code,
                }
                for p in programs[:10]
            ],
        }

        if programs:
            best = programs[0]
            output["best_program"] = {
                "id": best.id,
                "fitness": best.fitness,
                "priority_code": best.priority_code,
            }

        if path:
            path.write_text(json.dumps(output, indent=2))
            logger.info("Saved results to %s", path)
