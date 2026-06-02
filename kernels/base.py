"""base — FunSearch/EoH harness for auto-kernel problems (kernel_base relocated).

Phase 1 of the kernel restructuring. Behaviour is byte-identical to
``autobench.kernel_base`` — this is a relocation, not a rewrite.

Provides the complete island-model evolution loop, concurrent LLM generation,
horizon governance, sandbox gate, bus publishing, and analysis hooks. A second
problem domain (bin-packing, scheduling, graph-colouring…) plugs in by
subclassing FunSearchKernel and implementing five abstract methods:

  load_instances()          → load benchmark problem instances
  evaluate_candidate()      → run one code candidate on one instance → float
  build_prompt()            → build the LLM prompt for an island/generation
  parse_response()          → extract code from an LLM response string
  seed_programs()           → return baseline CandidateProgram list for an island

Everything else — ULID run ids, eval-once, concurrent generation,
ThreadPoolExecutor, plateau/budget/target/wall-clock horizon stops, bus
events, save_results, phase telemetry — is inherited unchanged.

Public re-exports from ``autobench.kernels``:
  FunSearchKernel, KernelConfig, CandidateProgram, Island, ConsolidatedPrior,
  new_ulid, make_local_llm_fn, ensure_sandboxed_executor, UnsafeSandboxError.
"""

from __future__ import annotations

import abc
import json
import logging
import math
import os
import random
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import KernelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kernel registry — used by the unified CLI in kernels/cli.py
# ---------------------------------------------------------------------------

KERNEL_REGISTRY: dict[str, type["FunSearchKernel"]] = {}


def register_kernel(name: str):
    """Class decorator: register a FunSearchKernel subclass under ``name``.

    The kernel CLI (``autobench.kernels run --kernel NAME``) uses this
    registry to look up the right subclass. Subclasses that want to be
    dispatchable through the unified CLI must be decorated::

        @register_kernel("sdf")
        class SDFKernel(FunSearchKernel):
            ...
    """

    def deco(cls: type["FunSearchKernel"]) -> type["FunSearchKernel"]:
        if name in KERNEL_REGISTRY:
            raise ValueError(f"kernel '{name}' already registered ({KERNEL_REGISTRY[name]})")
        KERNEL_REGISTRY[name] = cls
        cls.kernel_name = name
        return cls

    return deco


# ---------------------------------------------------------------------------
# ULID generator (shared with tsp_kernel; duplicated here so kernel_base is
# importable standalone without importing the full tsp_kernel package)
# ---------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    value = ((int(time.time() * 1000) & ((1 << 48) - 1)) << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def _code_diversity(a: str, b: str) -> float:
    """Jaccard distance on token sets (0=identical, 1=disjoint)."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def _population_code_diversity(codes: list[str]) -> float:
    """Mean pairwise Jaccard distance across all code strings."""
    if len(codes) < 2:
        return 0.0
    dists = [_code_diversity(codes[i], codes[j])
             for i in range(len(codes)) for j in range(i + 1, len(codes))]
    return sum(dists) / len(dists)


def _git_commit_short() -> str:
    """Return short git SHA of the repo at the kernel source location, or 'unknown'."""
    import subprocess as _sp
    try:
        r = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2.0,
            cwd=Path(__file__).parent,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# ConsolidatedPrior — cross-run T-vector accumulation (the neocortex layer)
# ---------------------------------------------------------------------------

class ConsolidatedPrior:
    """Cross-run T-vector accumulation — the neocortex layer.

    Stores sufficient-statistic T-vectors from high-fitness programs across runs.
    Fits a Gaussian over the accumulated T-vectors to build a prior distribution.
    Seeds the next run's initial population with programs sampled from (or near)
    the mode of the prior.

    Storage: ~/.cache/funsearch/<prefix>/prior.jsonl — one JSON line per T-vector.
    Format: {"run_id": ..., "fitness": ..., "t_vector": {...}, "code_hash": ...}
    """

    def __init__(self, prefix: str, path: "Path | None" = None):
        self.prefix = prefix
        self.path: Path = path or (Path.home() / ".cache" / "funsearch" / prefix / "prior.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []

    def load(self) -> int:
        """Load existing T-vectors. Returns count loaded."""
        self._entries = []
        if not self.path.exists():
            return 0
        try:
            seen: set[str] = set()
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            ch = entry.get("code_hash", "")
                            if ch and ch in seen:
                                continue
                            seen.add(ch)
                            self._entries.append(entry)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.warning("ConsolidatedPrior.load failed: %s", e)
        return len(self._entries)

    def append(self, run_id: str, fitness: float, t_vector: dict, code_hash: str) -> None:
        """Append one T-vector entry."""
        entry = {
            "run_id": run_id,
            "fitness": fitness,
            "t_vector": t_vector,
            "code_hash": code_hash,
        }
        self._entries.append(entry)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("ConsolidatedPrior.append failed: %s", e)

    def fit(self) -> "dict[str, tuple[float, float]]":
        """Fit Gaussian over T-vectors of high-fitness programs (top 25%).
        Returns {key: (mean, std)} for each T component."""
        if not self._entries:
            return {}
        sorted_entries = sorted(self._entries, key=lambda e: e.get("fitness", 0.0), reverse=True)
        top_n = max(1, len(sorted_entries) // 4)
        top_entries = sorted_entries[:top_n]

        # Gather all keys across the top entries
        all_keys: set[str] = set()
        for entry in top_entries:
            tv = entry.get("t_vector", {})
            all_keys.update(tv.keys())

        result: dict[str, tuple[float, float]] = {}
        for key in all_keys:
            values = [entry["t_vector"][key] for entry in top_entries if key in entry.get("t_vector", {})]
            if not values:
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            result[key] = (mean, std)
        return result

    def diversity_score(self, t_vector: dict) -> float:
        """How far is this T-vector from the accumulated prior?
        Higher = more novel = worth exploring.
        Returns Mahalanobis-like distance from prior mean."""
        stats = self.fit()
        if not stats:
            return 1.0
        dists: list[float] = []
        for key, (mean, std) in stats.items():
            if key not in t_vector:
                continue
            if std > 1e-9:
                dists.append(((t_vector[key] - mean) / std) ** 2)
            else:
                dists.append(0.0)
        if not dists:
            return 0.0
        return (sum(dists) / len(dists)) ** 0.5


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateProgram:
    """A single evolved program — problem-domain-agnostic."""
    id: str
    code: str                    # the evolved function/snippet
    island: int
    generation: int
    fitness: float = 0.0
    fitness_variance: float = 0.0
    worst_fitness: float = 0.0
    computation_time_ms: float = 0.0
    source: str = "llm"          # "llm" | "baseline" | "mutated" | "migrated"
    evaluated: bool = False      # set once; prevents redundant re-evaluation


@dataclass
class Island:
    id: int
    population: list[CandidateProgram] = field(default_factory=list)
    best_program: CandidateProgram | None = None


# ---------------------------------------------------------------------------
# Abstract base kernel
# ---------------------------------------------------------------------------

class FunSearchKernel(abc.ABC):
    """Domain-agnostic FunSearch island-model loop.

    Subclass this and implement the five abstract methods. See tsp_kernel for
    a worked example.
    """

    # Bus channel prefix — subclasses override to emit on their own channels
    BUS_CHANNEL_PREFIX: str = "kernel"

    def __init__(self, config: KernelConfig):
        self.config = config
        self.islands: list[Island] = []
        self.generation = 0
        self.generations = config.generations
        self.history: list[dict] = []
        self.llm_requests = 0
        self._llm_lock = threading.Lock()
        self._plateau_count = 0
        self._island_plateau_counts: dict[int, int] = {}
        self._island_prev_best: dict[int, float] = {}
        self._run_start = 0.0
        self.stop_reason = ""
        self.run_id = new_ulid()
        self._active_hint: str = ""

        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        self._nervous_bin = config.nervous_bin or self._find_nervous_bin()
        self.problem_instances: list[Any] = []

        self._prior: ConsolidatedPrior | None = None
        if config.use_consolidated_prior:
            self._prior = ConsolidatedPrior(self.BUS_CHANNEL_PREFIX)

        # Cross-cutting bus event state
        self._last_published_best_fitness: float = 0.0
        self._island_age: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Diversity helpers
    # ------------------------------------------------------------------

    def _sample_exemplars(self, population: list[CandidateProgram], k: int = 3) -> list[CandidateProgram]:
        """Softmax-weighted exemplar sampling — surfaces diverse programs, not just the top-k."""
        valid = [p for p in population if p.fitness > 0]
        if not valid:
            return []
        max_f = max(p.fitness for p in valid)
        weights = [math.exp((p.fitness - max_f) / 0.5) for p in valid]
        result: list[CandidateProgram] = []
        pool = list(zip(weights, valid))
        while len(result) < min(k, len(valid)) and pool:
            total = sum(w for w, _ in pool)
            r = random.random() * total
            cum = 0.0
            for i, (w, p) in enumerate(pool):
                cum += w
                if r <= cum:
                    result.append(p)
                    pool.pop(i)
                    break
        return result

    def extract_t_vector(self, program: "CandidateProgram") -> "dict[str, float]":
        """Extract sufficient-statistic T-vector. Subclasses override for domain-specific stats."""
        import zlib
        code_bytes = program.code.encode() if program.code else b""
        comp_ratio = len(zlib.compress(code_bytes)) / (len(code_bytes) + 1)
        return {
            "fitness": program.fitness,
            "code_length": len(program.code),
            "compression_ratio": round(comp_ratio, 4),
        }

    def _get_plateau_hint(self, best_program: CandidateProgram) -> str:
        """Call deer query for strategic advice when evolution plateaus."""
        import subprocess
        prompt = (
            f"An evolved program has plateaued at fitness {best_program.fitness:.4f}. "
            f"The current best code is:\n\n```\n{best_program.code}\n```\n\n"
            f"Identify ONE specific structural weakness and suggest a concrete algorithmic change. "
            f"Be specific — name the change and explain why it should improve fitness. Under 4 sentences."
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

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these five
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_instances(self) -> list[Any]:
        """Load and return the benchmark problem instances."""

    @abc.abstractmethod
    def evaluate_candidate(self, code: str, instance: Any) -> float | None:
        """Evaluate code on one instance. Return approx-ratio-style score in
        (0, 1] (higher = better, 1.0 = optimal), or None on failure."""

    @abc.abstractmethod
    def build_prompt(self, island: Island, top_programs: list[CandidateProgram], generation: int, hint: str = "") -> str:
        """Build the LLM prompt for this island/generation."""

    @abc.abstractmethod
    def parse_response(self, response: str) -> str:
        """Extract executable code from an LLM response. Return '' on failure."""

    @abc.abstractmethod
    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        """Return deterministic baseline programs for a fresh island."""

    # ------------------------------------------------------------------
    # Shared machinery — usually no need to override
    # ------------------------------------------------------------------

    def evaluate_fitness(self, program: CandidateProgram) -> tuple[float, float, float]:
        """Evaluate across all instances. Returns (mean, variance, worst)."""
        ratios: list[float] = []
        t0 = time.time()
        for inst in self.problem_instances:
            result = self.evaluate_candidate(program.code, inst)
            ratios.append(result if result is not None else 0.0)
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
        return mean, var, worst

    def evaluate_island(self, island: Island) -> None:
        for prog in island.population:
            if prog.evaluated:
                continue
            self.evaluate_fitness(prog)
        valid = [p for p in island.population if p.fitness > 0]
        if valid:
            island.best_program = max(valid, key=lambda p: p.fitness)

    def initialize_islands(self) -> list[Island]:
        islands = []
        for i in range(self.config.n_islands):
            island = Island(id=i)
            island.population.extend(self.seed_programs(i, 0))
            fill_source = island.population[i % len(island.population)]
            while len(island.population) < self.config.population_per_island:
                island.population.append(CandidateProgram(
                    id=f"island{i}_gen0_fill{len(island.population)}",
                    code=fill_source.code,
                    island=i, generation=0, source="mutated",
                ))
            island.best_program = max(island.population, key=lambda p: p.fitness)
            islands.append(island)
        return islands

    def migrate(self) -> None:
        # Ring migration: inject global best into each island's weakest slot.
        # NOTE: with ≥4 islands and migration_interval≤5, all islands share a
        # common ancestor within ~20 generations — collapsing the multi-basin
        # search into single-basin local search. Kept for back-compat; the
        # run() loop calls island_reset() on plateau instead of migrate() on
        # regular intervals.
        for i, island in enumerate(self.islands):
            donor = self.islands[(i + 1) % len(self.islands)]
            if not (donor.best_program and island.population):
                continue
            weakest = min(range(len(island.population)), key=lambda j: island.population[j].fitness)
            island.population[weakest] = CandidateProgram(
                id=f"migrated_from_island{donor.id}_to_island{island.id}",
                code=donor.best_program.code,
                island=island.id,
                generation=island.population[0].generation,
                source="migrated",
            )

    def island_reset(self) -> None:
        """Cull the weakest islands and reseed them from scratch.

        Replaces ring-migration as the diversity-preservation mechanism.
        Triggered at plateau midpoint by run(). Culls the bottom 40% of
        islands (by best_program.fitness) and reseeds them from seed_programs()
        with fresh ULIDs — restoring fresh search basins without contaminating
        the surviving islands that are still climbing.
        """
        if not self.islands:
            return
        ranked = sorted(self.islands, key=lambda isl: (isl.best_program.fitness if isl.best_program else 0.0))
        n_cull = max(1, len(ranked) * 2 // 5)  # bottom 40%
        for island in ranked[:n_cull]:
            import copy as _copy
            old_best = island.best_program.fitness if island.best_program else 0.0
            fresh_seeds = self.seed_programs(island.id, self.generation)
            fill_src = fresh_seeds[island.id % len(fresh_seeds)]
            population = list(fresh_seeds)
            while len(population) < self.config.population_per_island:
                # copy.copy preserves the subclass type (e.g. TSP's CandidateProgram
                # which carries priority_code in addition to the base code field).
                dup = _copy.copy(fill_src)
                dup.id = f"island{island.id}_gen{self.generation}_reset{len(population)}"
                dup.fitness = 0.0
                dup.evaluated = False
                dup.source = "mutated"
                population.append(dup)
            island.population = population
            island.best_program = None
            self.evaluate_island(island)
            self._island_age[island.id] = 0
            logger.info("Island %d reset (was best=%.4f, reseeded from scratch)",
                        island.id, old_best)
        logger.info("island_reset: culled %d/%d islands", n_cull, len(self.islands))

    def global_best_fitness(self) -> float:
        bests = [isl.best_program.fitness for isl in self.islands if isl.best_program]
        return max(bests) if bests else 0.0

    def _next_wave_requests(self) -> int:
        return len(self.islands) * max(1, self.config.candidates_per_island)

    def _horizon_reason(self) -> str | None:
        c = self.config
        best = self.global_best_fitness()
        if c.target_fitness is not None and best >= c.target_fitness:
            return f"target_fitness reached (best={best:.4f} >= {c.target_fitness})"
        if c.max_requests is not None and self.llm_requests + self._next_wave_requests() > c.max_requests:
            return (f"request budget {c.max_requests} reached "
                    f"(used={self.llm_requests}, next wave={self._next_wave_requests()})")
        if c.max_wall_seconds is not None and (time.time() - self._run_start) >= c.max_wall_seconds:
            return f"wall-clock budget {c.max_wall_seconds:.0f}s reached"
        if c.plateau_generations is not None:
            if len(self.problem_instances) > 1 and self.config.island_instance_assignment:
                # Specialized-island mode: each island evaluates a single instance and
                # gets its FULL plateau budget independently. Skip the global plateau
                # check — it fires on the easy instance's ceiling and would terminate
                # hard-instance islands before they can search.
                if len(self._island_plateau_counts) >= len(self.islands):
                    if all(self._island_plateau_counts.get(isl.id, 0) >= c.plateau_generations
                           for isl in self.islands):
                        return (f"plateau: all {len(self.islands)} islands plateaued "
                                f">= {c.plateau_generations} gens each")
            elif self._plateau_count >= c.plateau_generations:
                return (f"plateau: no improvement > {c.plateau_epsilon} "
                        f"in {c.plateau_generations} generations")
        return None

    def _llm_call(self, prompt: str) -> str:
        with self._llm_lock:
            self.llm_requests += 1
        if self.config.llm_call_fn:
            try:
                return self.config.llm_call_fn(prompt)
            except Exception as e:
                logger.warning("llm_call_fn failed: %s", e)
                return ""
        try:
            import subprocess
            result = subprocess.run(
                ["deer", "query", "--model", "minimax-m2.7", "--terse",
                 f"Respond with ONLY a single ```cpp code block.\nPrompt:\n{prompt}"],
                capture_output=True, text=True, timeout=self.config.llm_timeout,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""

    def _generate_candidates(self) -> list[list[CandidateProgram]]:
        tasks: list[tuple[int, int, str]] = []
        for i, island in enumerate(self.islands):
            exemplars = self._sample_exemplars(island.population, k=3)
            prompt = self.build_prompt(island, exemplars, self.generation, hint=self._active_hint)
            for k in range(max(1, self.config.candidates_per_island)):
                tasks.append((i, k, prompt))

        new_programs: list[list[CandidateProgram]] = [[] for _ in self.islands]
        workers = max(1, min(self.config.max_concurrent_llm, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for island_idx, cand_idx, response in pool.map(
                lambda t: (t[0], t[1], self._llm_call(t[2])), tasks
            ):
                code = self.parse_response(response)
                new_programs[island_idx].append(CandidateProgram(
                    id=f"island{island_idx}_gen{self.generation}_llm{cand_idx}",
                    code=code, island=island_idx,
                    generation=self.generation, source="llm",
                ))
        return new_programs

    def step(self) -> None:
        t_gen0 = time.time()
        new_programs = self._generate_candidates()
        gen_seconds = time.time() - t_gen0

        t_eval0 = time.time()
        for i, island in enumerate(self.islands):
            for candidate in new_programs[i]:
                try:
                    self.evaluate_fitness(candidate)
                    self._publish_candidate(candidate)
                    worst = min(island.population, key=lambda p: p.fitness)
                    if candidate.fitness > worst.fitness:
                        island.population.append(candidate)
                    island.population.sort(key=lambda p: -p.fitness)
                    island.population = island.population[:self.config.population_per_island]
                except Exception as e:
                    logger.warning("Candidate failed: %s", e)

        if self.generation > 0 and self.generation % self.config.migration_interval == 0:
            self.migrate()

        for island in self.islands:
            self.evaluate_island(island)
        for island in self.islands:
            prev = self._island_prev_best.get(island.id, -1.0)
            curr = island.best_program.fitness if island.best_program else 0.0
            if curr - prev > self.config.plateau_epsilon:
                self._island_plateau_counts[island.id] = 0
            else:
                self._island_plateau_counts[island.id] = self._island_plateau_counts.get(island.id, 0) + 1
            self._island_prev_best[island.id] = max(prev, curr)
        eval_seconds = time.time() - t_eval0

        all_best = [isl.best_program for isl in self.islands if isl.best_program]
        if all_best:
            best = max(all_best, key=lambda p: p.fitness)

            all_fitness = [p.fitness for isl in self.islands for p in isl.population if p.fitness > 0]
            if len(all_fitness) > 1:
                mean_f = sum(all_fitness) / len(all_fitness)
                fitness_std = (sum((f - mean_f) ** 2 for f in all_fitness) / (len(all_fitness) - 1)) ** 0.5
            else:
                fitness_std = 0.0

            all_codes = [p.code for isl in self.islands for p in isl.population if p.code]
            code_diversity = _population_code_diversity(all_codes)

            entry = {
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
                "island_reset_fired": False,  # set True in run() when reset happens
            }
            self.history.append(entry)
            logger.info(
                "Gen %3d | best=%.4f (island%d) | mean=%.4f | std=%.4f | div=%.3f | gen=%.1fs reqs=%d",
                self.generation, best.fitness, best.island,
                entry["mean_pop_fitness"], fitness_std, code_diversity,
                gen_seconds, self.llm_requests,
            )
            self._publish_generation(best, entry)

            # Per-island health snapshot
            for island in self.islands:
                self._island_age[island.id] = self._island_age.get(island.id, 0) + 1
                self._publish("autobench.island.health.v1", {
                    "run_id": self.run_id,
                    "generation": self.generation,
                    "island": island.id,
                    "best_fitness": (island.best_program.fitness
                                     if island.best_program else 0.0),
                    "plateau_count": self._island_plateau_counts.get(island.id, 0),
                    "population_size": len(island.population),
                    "age_since_last_reset": self._island_age.get(island.id, 0),
                })

            # Budget gauge every 5 generations
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

            # best_fitness_improved — fires when global best improves by > 1e-6
            if best.fitness - self._last_published_best_fitness > 1e-6:
                improvement_delta = best.fitness - self._last_published_best_fitness
                self._publish(f"{self.BUS_CHANNEL_PREFIX}.best_fitness_improved.v1", {
                    "run_id": self.run_id,
                    "generation": self.generation,
                    "best_fitness": best.fitness,
                    "improvement_delta": round(improvement_delta, 6),
                    "best_program_id": best.id,
                    "best_island": best.island,
                })
                self._last_published_best_fitness = best.fitness

        self.generation += 1

    def initialize(self) -> None:
        prior_entries = 0
        if self._prior is not None:
            prior_entries = self._prior.load()
            if prior_entries > 0:
                logger.info("Loaded %d T-vectors from consolidated prior (%s)",
                            prior_entries, self._prior.prefix)
                self._publish_prior_loaded(prior_entries)

        self.islands = self.initialize_islands()
        for island in self.islands:
            self.evaluate_island(island)
        logger.info("Initialized %d islands × %d programs",
                    self.config.n_islands, self.config.population_per_island)
        self._publish_started()
        self._save_best_artifact()  # capture initial baseline

    def run(self) -> list[CandidateProgram]:
        self.initialize()
        self._run_start = time.time()
        self._plateau_count = 0
        self._island_plateau_counts = {}
        self._island_prev_best = {}
        # Seed from the post-initialize global best, NOT 0.0: the islands are
        # already seeded+evaluated by initialize(), so starting the watermark at
        # 0 would make the first generation's best_fitness_improved fire with a
        # delta equal to the entire seed fitness — a phantom "discovery" that is
        # really just the seed baseline. Measuring from the seed means the event
        # only fires for genuine improvement over it.
        self._last_published_best_fitness = self.global_best_fitness()
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
                self._active_hint = ""
                self._save_best_artifact()
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
                        best_p = max(
                            (p for isl in self.islands for p in isl.population if p.fitness > 0),
                            key=lambda p: p.fitness,
                            default=None,
                        )
                        if best_p:
                            hint = self._get_plateau_hint(best_p)
                            if hint:
                                self._active_hint = hint
                                self._publish_plateau_hint(hint, best_p.fitness)
                                logger.info("plateau hint: %s", hint[:120])

        if not self.stop_reason:
            self.stop_reason = f"generation cap {self.generations} reached"

        all_programs = []
        for island in self.islands:
            all_programs.extend(island.population)
        all_programs.sort(key=lambda p: -p.fitness)

        if self._prior is not None:
            import hashlib
            seen_hashes: set[str] = set()
            saved = 0
            for prog in all_programs[:self.config.prior_top_k]:
                if prog.fitness > 0:
                    code_hash = hashlib.md5(prog.code.encode()).hexdigest()[:8]
                    if code_hash not in seen_hashes:
                        seen_hashes.add(code_hash)
                        self._prior.append(self.run_id, prog.fitness,
                                           self.extract_t_vector(prog), code_hash)
                        saved += 1
            self._publish_prior_updated(saved)

        logger.info("Run ended: %s | gens=%d reqs=%d best=%.4f",
                    self.stop_reason, self.generation, self.llm_requests,
                    self.global_best_fitness())
        self._publish_completed(all_programs)
        return all_programs

    def save_results(self, programs: list[CandidateProgram], path: Path | None = None) -> None:
        if path is None and self.config.output_dir:
            path = self.config.output_dir / f"results_gen{self.generation}.json"
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
                "code": best.code,
                "source": best.source,
                "island": best.island,
                "generation": best.generation,
            } if best else None,
            "top_programs": [
                {"id": p.id, "fitness": p.fitness, "fitness_variance": p.fitness_variance,
                 "worst_fitness": p.worst_fitness, "island": p.island, "generation": p.generation,
                 "computation_time_ms": p.computation_time_ms, "source": p.source, "code": p.code}
                for p in programs[:10]
            ],
        }
        path.write_text(json.dumps(output, indent=2))
        logger.info("Saved results to %s", path)

    def _save_best_artifact(self) -> None:
        """Persist a visual artifact for the current global best program.

        Called every time a new best fitness is reached. Subclasses override
        _render_best_program() to produce a domain-specific PNG.
        Falls back to saving the code only (no render) if subclass doesn't implement.
        """
        best = max(
            (p for isl in self.islands for p in isl.population if p.fitness > 0),
            key=lambda p: p.fitness,
            default=None,
        )
        if best is None:
            return
        try:
            from autobench.artifact_store import (  # type: ignore
                ArtifactRecord, save_artifact_record, artifact_path_for
            )
            out_path = artifact_path_for(self.BUS_CHANNEL_PREFIX, self.run_id,
                                         self.generation, best.fitness)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            rendered = self._render_best_program(best, out_path)
            render_type = "none" if not rendered else self._artifact_render_type()

            record = ArtifactRecord(
                kernel=self.BUS_CHANNEL_PREFIX,
                run_id=self.run_id,
                generation=self.generation,
                fitness=best.fitness,
                instance=self.config.instances[0] if self.config.instances else "unknown",
                artifact_path=str(out_path.relative_to(
                    out_path.parents[out_path.parts.index("benchmarks")-1]
                    if "benchmarks" in out_path.parts else out_path.parent
                )),
                render_type=render_type,
                sdf_code=best.code,
                metadata={
                    "code_length": len(best.code),
                    "best_program_code": best.code[:500],
                    **self.extract_t_vector(best),
                },
            )
            save_artifact_record(record, self._nervous_bin)
        except Exception as e:
            logger.debug("_save_best_artifact failed (non-fatal): %s", e)

    def _render_best_program(self, best: CandidateProgram, out_path: Path) -> bool:
        """Render best program to out_path. Return True on success.
        Default: no render. Subclasses override for visual kernels."""
        return False

    def _artifact_render_type(self) -> str:
        """What kind of render does this kernel produce? Subclasses override."""
        return "none"

    # ------------------------------------------------------------------
    # Bus publishing — override to publish on different channels
    # ------------------------------------------------------------------

    def _publish_started(self) -> None:
        """Emit ``<prefix>.kernel.started.v1`` at run start.

        Base default — config-level run parameters common to every kernel.
        Kernels with a richer payload (e.g. ``sandbox_type``) override this;
        kernels that don't (latent, thermal) get a uniform started event for
        free instead of emitting nothing.
        """
        self._publish(f"{self.BUS_CHANNEL_PREFIX}.kernel.started.v1", {
            "run_id": self.run_id,
            "git_commit": _git_commit_short(),
            "instances": list(self.config.instances),
            "n_islands": self.config.n_islands,
            "population_per_island": self.config.population_per_island,
            "generations": self.config.generations,
            "plateau_generations": self.config.plateau_generations,
            "temperature": self.config.temperature,
            "plateau_hint": self.config.plateau_hint,
        })

    def _publish_candidate(self, candidate: CandidateProgram) -> None:
        """Emit <prefix>.candidate.evaluated.v1 for every scored candidate.

        Lightweight event — fitness, island, source, code_length. Full code
        is in the completed event. Subclasses with richer candidate schemas
        can override (TSP already does via its own step() loop).
        """
        if not candidate.evaluated or candidate.fitness <= 0:
            return
        prefix = self.BUS_CHANNEL_PREFIX
        self._publish(f"{prefix}.candidate.evaluated.v1", {
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": candidate.id,
            "island": candidate.island,
            "fitness": round(candidate.fitness, 6),
            "fitness_variance": round(candidate.fitness_variance, 6),
            "worst_fitness": round(candidate.worst_fitness, 6),
            "source": candidate.source,
            "code_length": len(candidate.code),
            "computation_time_ms": round(candidate.computation_time_ms, 1),
        })

    def _publish_generation(self, best: CandidateProgram, stats: dict) -> None:
        """Emit <prefix>.generation.completed.v1 each generation.

        Uses self._publish() so the call dispatches through the subclass's
        publish implementation (which calls nervous), not just debug.jsonl.
        Subclasses that define a richer generation event can still override.
        """
        prefix = self.BUS_CHANNEL_PREFIX
        self._publish(f"{prefix}.generation.completed.v1", {
            "run_id": self.run_id,
            "generation": stats["generation"],
            "best_fitness": stats["best_fitness"],
            "best_island": stats["best_island"],
            "mean_pop_fitness": stats["mean_pop_fitness"],
            "gen_seconds": stats["gen_seconds"],
            "eval_seconds": stats.get("eval_seconds", 0.0),
            "llm_requests": stats["llm_requests"],
            "fitness_std": stats.get("fitness_std", 0.0),
            "code_diversity": stats.get("code_diversity", 0.0),
            "island_reset_fired": stats.get("island_reset_fired", False),
        })

    def _publish_island_reset(self, n_culled: int, pre_best: float) -> None:
        """Emit <prefix>.island_reset.v1 when plateau culling fires."""
        prefix = self.BUS_CHANNEL_PREFIX
        self._publish(f"{prefix}.island_reset.v1", {
            "run_id": self.run_id,
            "generation": self.generation,
            "n_islands_culled": n_culled,
            "pre_reset_best_fitness": pre_best,
            "plateau_count": self._plateau_count,
        })

    def _publish_plateau_hint(self, hint: str, best_fitness: float) -> None:
        """Emit <prefix>.plateau_hint.v1 when deer advice is injected."""
        prefix = self.BUS_CHANNEL_PREFIX
        self._publish(f"{prefix}.plateau_hint.v1", {
            "run_id": self.run_id,
            "generation": self.generation,
            "plateau_count": self._plateau_count,
            "best_fitness": best_fitness,
            "hint_preview": hint[:200],
        })

    def _publish_prior_updated(self, n_vectors: int) -> None:
        """Emit <prefix>.prior.updated.v1 after saving T-vectors at end of run."""
        self._publish(f"{self.BUS_CHANNEL_PREFIX}.prior.updated.v1", {
            "run_id": self.run_id,
            "kernel": self.BUS_CHANNEL_PREFIX,
            "vectors_added": n_vectors,
            "total_vectors": len(self._prior._entries) if self._prior else 0,
            "prior_path": str(self._prior.path) if self._prior else "",
        })

    def _publish_prior_loaded(self, n_vectors: int) -> None:
        """Emit <prefix>.prior.loaded.v1 when a prior is loaded at initialize()."""
        self._publish(f"{self.BUS_CHANNEL_PREFIX}.prior.loaded.v1", {
            "run_id": self.run_id,
            "kernel": self.BUS_CHANNEL_PREFIX,
            "vectors_loaded": n_vectors,
            "prior_fitted": True,
        })

    def _publish_completed(self, programs: list[CandidateProgram]) -> None:
        """Emit ``<prefix>.kernel.completed.v1`` at run end.

        Base default — best-program summary + run stats. Kernels with a richer
        payload (e.g. domain source code, full history) override this; kernels
        that don't (latent, thermal) get a uniform completed event for free.
        """
        best = programs[0] if programs else None
        self._publish(f"{self.BUS_CHANNEL_PREFIX}.kernel.completed.v1", {
            "run_id": self.run_id,
            "total_generations": self.generation,
            "stop_reason": self.stop_reason,
            "llm_requests": self.llm_requests,
            "best_program": {
                "id": best.id,
                "fitness": best.fitness,
                "island": best.island,
                "generation": best.generation,
                "source": best.source,
            } if best else None,
        })

    def _publish(self, channel: str, data: dict) -> bool:
        """Publish a CloudEvents-lite event to nervous-bus. Fail-silent.

        The single bus-publish path for every kernel (consolidated from the
        per-kernel copies). Writes to ``~/.cache/nervous-bus/debug.jsonl``
        (durable history) first, then best-effort fire-and-forget to the
        ``nervous`` CLI for zellij/Redis consumers. The CloudEvents ``source``
        is derived from ``BUS_CHANNEL_PREFIX`` so it matches the per-kernel
        contract (``/autobench/<prefix>_kernel``); ``type`` is the channel.
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": f"/autobench/{self.BUS_CHANNEL_PREFIX}_kernel",
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

        # Best-effort live delivery to Redis Streams via the shell SDK's
        # --json pass-through. We feed the ALREADY-BUILT envelope on stdin so
        # nervous forwards it verbatim — the non-json path would re-wrap the
        # envelope as the `data` field of a fresh envelope (double-wrap, with a
        # bogus source from CWD). NERVOUS_DEBUG_LOG=/dev/null prevents a second
        # debug.jsonl write (the durable record above is the single source of
        # truth). Redis is left ENABLED: kernel channels are not in
        # redis-mirror's tail list (mirror_all=false), so this direct XADD is
        # the only Redis path — no duplicate, despite redis-mirror having no
        # $id dedup. Zellij pane fan-out stays off to avoid a per-event
        # subprocess in tight evolution loops.
        if self._nervous_bin:
            try:
                env = dict(os.environ)
                env["NERVOUS_NO_ZELLIJ"] = "1"
                env["NERVOUS_DEBUG_LOG"] = os.devnull
                proc = subprocess.Popen(
                    [self._nervous_bin, "publish", "--json"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
                proc.communicate(payload.encode(), timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        return True

    def _find_nervous_bin(self) -> str | None:
        """Locate the ``nervous`` shell SDK — PATH first, then the repo path."""
        import shutil
        found = shutil.which("nervous")
        if found:
            return found
        repo = Path.home() / "projects" / "nervous-bus" / "sdk" / "shell" / "nervous"
        return str(repo) if repo.is_file() else None


# ---------------------------------------------------------------------------
# CPU-fallback LLM factory — zero GPU VRAM via CUDA_VISIBLE_DEVICES=""
# ---------------------------------------------------------------------------

def make_local_llm_fn(
    base_url: str = "http://localhost:8081",
    model: str = "local",
    temperature: float = 0.9,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> Callable[[str], str]:
    """Return an llm_call_fn that hits a local OpenAI-compatible server.

    Use with llama-server or ollama to run entirely on CPU (zero GPU VRAM):

        # llama-server (recommended — more control):
        CUDA_VISIBLE_DEVICES="" /path/to/llama-server \\
            --model ~/models/qwen2.5-coder-3b-instruct-Q4_K_M.gguf \\
            -ngl 0 --parallel 4 -cb --port 8081 --ctx-size 2048 --threads 6

        # ollama (simpler):
        OLLAMA_NUM_GPU=0 ollama serve   # in one terminal
        ollama pull qwen2.5-coder:3b   # once

    Then pass to KernelConfig:
        config = KernelConfig(..., llm_call_fn=make_local_llm_fn())

    IMPORTANT: CUDA_VISIBLE_DEVICES="" (not -ngl 0 alone) is required for zero
    VRAM on CUDA builds — the CUDA backend is loaded via dlopen regardless of
    --ngl flags; the env var prevents the CUDA context from being created.
    """
    try:
        import httpx as _httpx
    except ImportError:
        raise ImportError("httpx required for local LLM: pip install httpx")

    _client = _httpx.Client(base_url=base_url, timeout=timeout)

    def _call(prompt: str) -> str:
        resp = _client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _call
