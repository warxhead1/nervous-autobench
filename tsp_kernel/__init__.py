"""TSP FunSearch kernel — LLM-driven heuristic discovery for routing.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │                     FunSearch Loop                            │
  │                                                               │
  │  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐  │
  │  │  LLM Caller  │──▶│  Priority Fn  │──▶│  C++ Sandbox     │  │
  │  │  (deer-flow) │   │  Generator    │   │  (compile+run)  │  │
  │  └──────────────┘   └───────────────┘   └────────┬─────────┘  │
  │                                                     │          │
  │                          ┌─────────────────────────┘          │
  │                          ▼                                      │
  │                   ┌──────────────┐                            │
  │                   │  Fitness     │                            │
  │                   │  (approx     │                            │
  │                   │   ratio)     │                            │
  │                   └──────┬───────┘                            │
  │                          │                                     │
  │         ┌────────────────┼────────────────┐                   │
  │         ▼                ▼                ▼                   │
  │  ┌───────────┐   ┌───────────┐   ┌───────────┐              │
  │  │  Island 0 │   │  Island 1 │ ...│ Island N │              │
  │  │  (20 progs)│   │  (20 progs)│   │  (20 progs)│             │
  │  └───────────┘   └───────────┘   └───────────┘              │
  │         │                │                │                   │
  │         └────────────────┼────────────────┘                   │
  │                    migration (every 10 gens)                  │
  └───────────────────────────────────────────────────────────────┘

Key design choices:
  - C++ skeleton + injected LLM priority(); compiled+run per candidate
  - Untrusted candidate code is EXECUTED only inside an isolating sandbox
    (autobench.sandbox.SandboxedExecutor → rootless gVisor/runsc, --network=none).
    Compilation runs on the host by deliberate design: the untrusted input is
    source-only, compiler flags are fixed (no -fplugin injection), g++ does not
    run the program at compile time, and a separate `runsc do` cannot persist its
    binary to the host anyway (ephemeral overlay). The host compile is still
    timeout/memory-bounded. The kernel REFUSES to run if no isolating sandbox is
    available, unless allow_unsandboxed=True is set for a trusted, attended run.
  - Island model: 10 islands × 20 programs — avoids premature convergence
  - Fitness = mean approximation ratio over 3-5 training instances
  - Baseline seeds: nearest-neighbor, farthest insertion, cheapest insertion
  - LLM only evolves priority(); solve() and evaluate() are fixed

Usage:
  python -m autobench.tsp_kernel run --instances berlin52,kroA100,eil101 \
           --generations 100 --islands 10 --population 20

Reference:
  FunSearch (DeepMind, 2024): https://github.com/google-deepmind/funsearch
  EoH (ICML 2024): https://arxiv.org/abs/2401.02051
  EoH-S (2025): https://arxiv.org/html/2508.03082v2
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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


# Crockford base32 (no I, L, O, U) — ULID alphabet.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """Return a 26-char ULID: 48-bit ms timestamp + 80 random bits.

    Sortable and monotonic-ish by creation time, unlike uuid4 — the schemas
    declare run_id as a ULID, so run ordering by id is meaningful.
    """
    value = ((int(time.time() * 1000) & ((1 << 48) - 1)) << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


# ---------------------------------------------------------------------------
# TSPLIB instance format
# ---------------------------------------------------------------------------

TSPLIB_URL_BASE = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"


def fetch_tsplib_instance(name: str, cache_dir: Path | None = None) -> "TSPInstance":
    """Load a TSPLIB instance from the local cache (or download if missing)."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "instances"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for .tsp.gz first (compressed archive format)
    gz_path = cache_dir / f"{name}.tsp.gz"
    tsp_path = cache_dir / f"{name}.tsp"

    if gz_path.exists():
        path = gz_path
    elif tsp_path.exists():
        path = tsp_path
    else:
        # No local copy — embedded benchmark data (no network fetch in kernel)
        raise FileNotFoundError(
            f"Instance '{name}' not found in {cache_dir}. "
            f"Run 'python -m autobench.tsp_kernel bootstrap' first to download."
        )

    inst = TSPInstance.from_file(path)

    # Load optimal tour length if .opt.tour.gz exists
    opt_gz = cache_dir / f"{name}.opt.tour.gz"
    if opt_gz.exists():
        import gzip
        with gzip.open(opt_gz, 'rt', encoding='utf-8', errors='replace') as f:
            content = f.read()
        # Optimal tour format: one integer per line, after header
        reading_tour = False
        tour = []
        for line in content.splitlines():
            if reading_tour and line.strip():
                try:
                    tour.append(int(line.strip()))
                except ValueError:
                    pass
            if "TOUR_SECTION" in line:
                reading_tour = True
        if tour and len(tour) > 1:
            # Compute optimal length from coords
            opt_len = 0.0
            for i in range(len(tour) - 1):
                opt_len += inst.distance(tour[i] - 1, tour[i + 1] - 1)
            inst.optimal_tour_length = opt_len
            inst.optimal_tour = tour

    return inst


@dataclass
class TSPInstance:
    name: str
    n: int
    coords: list[tuple[float, float]]
    dist_matrix: list[list[float]] | None = None
    optimal_tour_length: float | None = None
    optimal_tour: list[int] | None = None

    def distance(self, i: int, j: int) -> float:
        if self.dist_matrix is not None:
            return self.dist_matrix[i][j]
        cx, cy = self.coords[i]
        dx, dy = self.coords[j]
        return ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5

    def compute_dist_matrix(self) -> None:
        n = self.n
        self.dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                cx, cy = self.coords[i]
                dx, dy = self.coords[j]
                d = ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5
                self.dist_matrix[i][j] = d
                self.dist_matrix[j][i] = d

    @classmethod
    def from_file(cls, path: Path) -> "TSPInstance":
        # Handle .tsp.gz and .tsp extensions
        name = path.stem  # 'berlin52.tsp' or 'berlin52'
        if name.endswith('.tsp'):
            name = name[:-4]
        # Decompress if needed
        if path.suffix == '.gz':
            import gzip
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
                content = f.read()
        else:
            content = path.read_text(encoding='utf-8', errors='replace')
        lines = content.splitlines()

        coords: list[tuple[float, float]] = []
        dimension: int | None = None
        reading_coords = False

        for line in lines:
            line = line.strip()
            if line.startswith("DIMENSION"):
                dimension = int(line.split()[-1])
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                reading_coords = False
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                        coords.append((x, y))
                    except ValueError:
                        continue
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True

        # If dimension was declared but coords don't match, truncate or pad
        if dimension and len(coords) != dimension:
            if len(coords) < dimension:
                coords = coords + [(0.0, 0.0)] * (dimension - len(coords))

        n = len(coords)
        inst = cls(name=name, n=n, coords=coords)
        inst.compute_dist_matrix()
        return inst


@dataclass
class Tourevaluation:
    tour: list[int]
    length: float
    instance_name: str


# ---------------------------------------------------------------------------
# C++ skeleton — fixed architecture, LLM injects priority()
# ---------------------------------------------------------------------------

CPP_SKELETON = """// Auto-generated TSP skeleton. LLM writes priority() only.
#include <bits/stdc++.h>
using namespace std;

struct State {{
    vector<int> current_tour;
    vector<int> remaining;
    double tour_length_so_far;
    int current_node;
}};

struct Instance {{
    int n;
    vector<vector<double>> dist;
    vector<pair<double,double>> coords;
    string name;
}};

extern "C" double priority(int node, const Instance* inst, const State* state);

extern "C" void solve(const Instance* inst, vector<int>* out_tour, double* out_len) {{
    State state;
    state.tour_length_so_far = 0.0;
    state.current_node = 0;

    // Start at node 0
    vector<bool> visited(inst->n, false);
    state.current_tour.push_back(0);
    visited[0] = true;
    state.remaining.clear();
    for (int i = 1; i < inst->n; i++) state.remaining.push_back(i);
    state.current_node = 0;

    while (!state.remaining.empty()) {{
        double best_score = -1e100;
        int best_node = -1;

        for (int v : state.remaining) {{
            double sc = priority(v, inst, &state);
            if (sc > best_score) {{
                best_score = sc;
                best_node = v;
            }}
        }}

        state.current_tour.push_back(best_node);
        state.tour_length_so_far += inst->dist[state.current_node][best_node];
        state.current_node = best_node;

        vector<int> new_rem;
        for (int v : state.remaining) if (v != best_node) new_rem.push_back(v);
        state.remaining.swap(new_rem);
        visited[best_node] = true;
    }}

    // Close tour
    state.tour_length_so_far += inst->dist[state.current_node][0];
    *out_tour = state.current_tour;
    *out_len = state.tour_length_so_far;
}}

// Built-in priority functions (baselines)
extern "C" double baseline_nearest(int node, const Instance* inst, const State* state) {{
    double best = 1e100;
    for (int t : state->current_tour) {{
        best = min(best, inst->dist[node][t]);
    }}
    return -best;
}}

extern "C" double baseline_farthest(int node, const Instance* inst, const State* state) {{
    double worst = 0.0;
    for (int t : state->current_tour) {{
        worst = max(worst, inst->dist[node][t]);
    }}
    return worst;
}}

extern "C" double baseline_angle(int node, const Instance* inst, const State* state) {{
    double cx = 0, cy = 0;
    for (int i = 0; i < inst->n; i++) {{ cx += inst->coords[i].first; cy += inst->coords[i].second; }}
    cx /= inst->n; cy /= inst->n;
    double dx = inst->coords[node].first - cx;
    double dy = inst->coords[node].second - cy;
    return atan2(dy, dx);
}}
"""


CPP_MAIN_TEMPLATE = """#include <bits/stdc++.h>
using namespace std;

{extras}

extern "C" double priority(int node, const Instance* inst, const State* state);
extern "C" void solve(const Instance* inst, vector<int>* out_tour, double* out_len);

int main(int argc, char** argv) {
    // Instance JSON arrives on stdin (the sandbox executor feeds it there).
    string s((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());

    // ---- Hand-rolled JSON parser ----
    int pos = 0;
    auto skip_ws = [&]() {
        while (pos < (int)s.size() && isspace(s[pos])) pos++;
    };
    auto expect_char = [&](char c) -> bool {
        skip_ws();
        if (pos < (int)s.size() && s[pos] == c) { pos++; return true; }
        return false;
    };
    auto parse_string = [&]() -> string {
        skip_ws();
        if (pos >= (int)s.size() || s[pos] != '"') return string();
        pos++; string out;
        while (pos < (int)s.size() && s[pos] != '"') {
            if (s[pos] == '\\\\') pos++;
            if (pos < (int)s.size()) out += s[pos++];
        }
        if (pos < (int)s.size()) pos++;
        return out;
    };
    auto parse_number = [&]() -> double {
        skip_ws();
        double v = 0; bool neg = false;
        if (pos < (int)s.size() && s[pos] == '-') { neg = true; pos++; }
        while (pos < (int)s.size() && isdigit(s[pos])) { v = v * 10 + (s[pos] - '0'); pos++; }
        if (pos < (int)s.size() && s[pos] == '.') {
            pos++; double frac = 1;
            while (pos < (int)s.size() && isdigit(s[pos])) { frac /= 10; v += (s[pos] - '0') * frac; pos++; }
        }
        return neg ? -v : v;
    };

    if (!expect_char('{')) return 1;

    Instance inst;
    inst.n = 0;

    while (true) {
        skip_ws();
        if (pos >= (int)s.size()) break;
        if (s[pos] == '}') { pos++; break; }

        string key = parse_string();
        if (key.empty()) {
            // Unrecognized content after dist array — stop parsing
            break;
        }
        skip_ws();
        if (pos < (int)s.size() && s[pos] == ':') pos++;
        skip_ws();

        if (key == "name") {
            inst.name = parse_string();
        } else if (key == "n") {
            inst.n = (int)parse_number();
        } else if (key == "coords") {
            if (s[pos] == '[') {
                pos++;
                inst.coords.resize(inst.n);
                for (int c = 0; c < inst.n; c++) {
                    skip_ws();
                    if (s[pos] == ']') { pos++; break; }
                    if (s[pos] == '[') pos++;
                    inst.coords[c].first = parse_number();
                    skip_ws();
                    if (s[pos] == ',') pos++;
                    inst.coords[c].second = parse_number();
                    skip_ws();
                    if (s[pos] == ']') pos++;
                    skip_ws();
                    if (s[pos] == ',') pos++;
                }
                skip_ws();
                if (s[pos] == ']') pos++;
            }
        } else if (key == "dist") {
            if (s[pos] == '[') {
                pos++;
                inst.dist.resize(inst.n, vector<double>(inst.n, 0));
                for (int r = 0; r < inst.n; r++) {
                    skip_ws();
                    if (s[pos] == ']') { pos++; break; }
                    if (s[pos] == '[') pos++;
                    for (int c = 0; c < inst.n; c++) {
                        skip_ws();
                        if (s[pos] == ']') { pos++; break; }
                        if (s[pos] == ',') { pos++; continue; }
                        inst.dist[r][c] = parse_number();
                        skip_ws();
                        if (s[pos] == ',') pos++;
                    }
                    if (s[pos] == ']') {
                        pos++;
                        if (s[pos] == ',') { pos++; }
                        else { break; }
                    }
                }
            }
        }
        skip_ws();
        if (pos < (int)s.size() && s[pos] == ',') pos++;
    }

    vector<int> tour;
    double len = 0.0;
    solve(&inst, &tour, &len);

    cout << "{\\\"tour\\":[";
    for (int k = 0; k < (int)tour.size(); k++) {
        if (k > 0) cout << ",";
        cout << tour[k];
    }
    cout << "],\\"length\\":" << len << ",\\"instance\\":\\"" << inst.name << "\\"}";
    return 0;
}
"""


@dataclass
class CandidateProgram:
    id: str
    priority_code: str
    island: int
    generation: int
    fitness: float = 0.0
    fitness_variance: float = 0.0
    worst_fitness: float = 0.0
    computation_time_ms: float = 0.0
    source: str = "llm"  # "llm" | "baseline" | "mutated" | "migrated"
    evaluated: bool = False  # set once fitness is computed; skip re-evaluation
    per_instance_fitness: dict = field(default_factory=dict)  # instance_name → fitness ratio

    @property
    def code(self) -> str:
        """Alias for priority_code — satisfies FunSearchKernel.evaluate_fitness interface."""
        return self.priority_code


@dataclass
class Island:
    id: int
    population: list[CandidateProgram] = field(default_factory=list)
    best_program: CandidateProgram | None = None


# ---------------------------------------------------------------------------
# Sandboxed C++ runner
#
# Candidate priority() functions are LLM-generated and therefore UNTRUSTED. The
# dangerous part — running the produced binary with arbitrary syscalls — happens
# only inside autobench.sandbox.SandboxedExecutor in gVisor mode (rootless runsc,
# --network=none), the same isolation the codeforces/shader benchmarks use.
# Compilation runs on the host (fixed flags, source-only untrusted input,
# resource-capped); see _run_subprocess for why a sandboxed compile cannot
# persist its binary. The kernel refuses to run unless an isolating sandbox is
# active for EXECUTION (see ensure_sandboxed_executor); set allow_unsandboxed=True
# only for trusted, attended experiments.
# ---------------------------------------------------------------------------

from ..core import Verdict  # noqa: E402  (kept local to the runner section)
from ..sandbox import SandboxedExecutor, compile_and_run  # noqa: E402
# FunSearchKernel + ensure_sandboxed_executor + KernelConfig come from autobench.kernels.
# We re-export them below for back-compat with existing imports inside this file.
from ..kernels import (  # noqa: E402
    FunSearchKernel,
    KernelConfig,
    ensure_sandboxed_executor,
    UnsafeSandboxError,
    register_kernel,
)


def build_candidate_source(priority_code: str, extra_code: str = "") -> str:
    """Assemble the full C++ program: skeleton + LLM priority() + main."""
    main_src = CPP_MAIN_TEMPLATE.replace("{extras}", extra_code)
    return CPP_SKELETON.format(extra_code=extra_code) + "\n" + priority_code + "\n" + main_src


def _instance_stdin(instance: TSPInstance) -> str:
    """Serialize an instance to the JSON the C++ runner reads from stdin."""
    return json.dumps({
        "name": instance.name,
        "n": instance.n,
        "coords": [[x, y] for x, y in instance.coords],
        "dist": instance.dist_matrix,
    })


def evaluate_on_instance(
    priority_code: str,
    instance: TSPInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 10.0,
) -> Tourevaluation | None:
    """Compile+run one candidate against one instance inside the sandbox.

    Returns a Tourevaluation on success, or ``None`` when the candidate fails
    (compile error, timeout, crash, or unparseable output) — callers treat None
    as fitness 0.0. The Verdict from the executor is the authority on failure;
    we only parse stdout when the verdict is OK.
    """
    source = build_candidate_source(priority_code)
    stdout, verdict, _latency = compile_and_run(
        source,
        "cpp",
        constraints={"max_time_seconds": run_timeout, "max_memory_mb": executor.max_memory_mb},
        stdin=_instance_stdin(instance),
        executor=executor,
    )
    if verdict != Verdict.OK:
        logger.debug("candidate non-OK verdict %s on %s", verdict, instance.name)
        return None
    try:
        out = json.loads(stdout)
        length = float(out["length"])
        if length <= 0:
            return None
        return Tourevaluation(tour=out["tour"], length=length, instance_name=instance.name)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.debug("candidate output unparseable on %s: %s", instance.name, exc)
        return None


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_fitness(
    program: CandidateProgram,
    instances: list[TSPInstance],
    work_dir: Path | None = None,
    executor: SandboxedExecutor | None = None,
    run_timeout: float = 10.0,
) -> tuple[float, float, float]:
    """Evaluate candidate across instances. Returns (mean_ratio, variance, worst_ratio).

    All compilation and execution happens inside ``executor`` (an isolating
    sandbox). ``work_dir`` is accepted for backward compatibility and ignored —
    the executor manages its own scratch space. A failing candidate scores 0.0
    on the instance it failed, so a single bad instance cannot crash the loop.
    """
    if executor is None:
        executor = ensure_sandboxed_executor()

    ratios: list[float] = []
    total_time = 0.0
    for inst in instances:
        t0 = time.time()
        result = evaluate_on_instance(program.priority_code, inst, executor, run_timeout)
        total_time += (time.time() - t0) * 1000

        if result is None:
            ratios.append(0.0)
        elif inst.optimal_tour_length:
            ratios.append(inst.optimal_tour_length / result.length)
        else:
            ratios.append(1.0)  # unknown optimal

    if not ratios:
        return 0.0, 0.0, 0.0

    mean_ratio = sum(ratios) / len(ratios)
    variance = sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)
    worst = min(ratios)
    program.fitness = mean_ratio
    program.fitness_variance = variance
    program.worst_fitness = worst
    program.computation_time_ms = total_time
    program.evaluated = True
    program.per_instance_fitness = {inst.name: ratio for inst, ratio in zip(instances, ratios)}
    return mean_ratio, variance, worst


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


def mutate_priority(code: str, seed: int) -> str:
    """Make a small syntactic mutation to a priority function."""
    import random
    rng = random.Random(seed)
    mutations = [
        ("return -best;", "return -best * 1.1;"),
        ("return worst;", "return worst * 0.9;"),
        ("return atan2(dy, dx);", "return atan2(dy, dx) * 1.05;"),
        ("best = min(best,", "best = min(best * 0.95,"),
        ("worst = max(worst,", "worst = max(worst * 1.05,"),
    ]
    a, b = rng.choice(mutations)
    return code.replace(a, b)


# ---------------------------------------------------------------------------
# Algorithmic sketch library — one sketch per island×generation combination
# ---------------------------------------------------------------------------

PROMPT_SKETCHES = [
    # Sketch 0: Regret-based (Or-opt-style)
    ("regret-based insertion",
     "Score each candidate node by its REGRET: how much worse would the tour be if we deferred visiting this node to later? "
     "Regret ≈ (second-nearest tour distance) - (nearest tour distance). High-regret nodes are urgent — visit them early."),

    # Sketch 1: MST-inspired (Christofides)
    ("MST-guided construction",
     "Approximate minimum spanning tree construction: prefer nodes that would form low-cost MST edges from the current tour frontier. "
     "A node is a good next choice if selecting it now 'extends' the tour along a direction that will minimize future backtracking."),

    # Sketch 2: Convex hull peeling
    ("convex hull insertion",
     "Build the tour from the outside in: prefer nodes on or near the convex hull of remaining nodes first, then fill interior nodes. "
     "Hull nodes visited early prevent the common failure of long cross-edges to isolated exterior nodes."),

    # Sketch 3: k-opt anticipation
    ("crossing-avoidance",
     "Penalize tour crossings: prefer nodes whose insertion avoids creating edge intersections with the current tour. "
     "A crossing always indicates a 2-opt improvement exists — constructing crossing-free tours from the start approaches LK quality."),

    # Sketch 4: LK k-neighbor lists
    ("k-neighbor locality",
     "Exploit the Lin-Kernighan insight: the best next node is almost always within the 5 nearest neighbors of the current node. "
     "Apply an exponential penalty to nodes ranked outside the k=7 nearest-neighbor list of the current position."),

    # Sketch 5: Normalized multi-signal fusion
    ("normalized signal fusion",
     "Combine distance, angular, and density signals, but NORMALIZE each before weighting. "
     "Compute max_dist = max over remaining nodes of dist_to_tour[node], then use dist_to_tour[node]/max_dist (range 0–1). "
     "Similarly normalize angle_to_centroid by dividing by π. Then a weighted sum like "
     "-w1*(dist/max_dist) + w2*(angle/π) + w3*density keeps all terms contributing at the same order of magnitude."),
]


# ---------------------------------------------------------------------------
# LLM prompt generation
# ---------------------------------------------------------------------------

def _sample_exemplars(population: list, k: int = 3, temperature: float = 0.5) -> list:
    """Softmax-weighted exemplar sampling — FunSearch style.

    Lets diverse lower-fitness programs appear in prompts instead of always
    anchoring on the incumbent. Uses a fixed temperature (0.5) distinct from
    the generation temperature so sampling diversity is decoupled from model
    creativity.
    """
    valid = [p for p in population if p.fitness > 0]
    if not valid:
        return []
    max_f = max(p.fitness for p in valid)
    weights = [math.exp((p.fitness - max_f) / temperature) for p in valid]
    result = []
    # Sample without replacement using softmax weights
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


ISLAND_PERSONAS = [
    # Island 0: Control (nearest-neighbor greedy)
    ("nearest-neighbor strategist",
     "Optimize for immediate proximity — the best next node minimizes the current edge length while considering local density."),
    # Island 1: Graph theorist
    ("graph theorist",
     "Think in terms of graph structure: connectivity, degree centrality, spanning tree proximity. Prefer nodes that would form low-cost MST edges from the current tour frontier."),
    # Island 2: Geometric
    ("geometric analyst",
     "Use angular sweep, convex hull peeling, and centroid-orbit strategies. Prefer nodes on the convex hull of remaining nodes; fill interior last."),
    # Island 3: Physicist
    ("physicist",
     "Frame nodes as particles: minimize tour 'potential energy' by avoiding long-range interactions. Penalize nodes that would create field lines (tour edges) crossing the existing tour."),
    # Island 4+: Lookahead (for 5+ island runs)
    ("lookahead planner",
     "Score nodes by 2-step anticipation: prefer nodes where selecting them now leaves a favorable next-step choice. Consider the second-nearest unvisited node's accessibility."),
]


PRIORITY_SIGNATURE = """extern "C" double priority(int node, const Instance* inst, const State* state);

// inst->n           : number of nodes
// inst->dist[i][j]  : distance between nodes i and j
// inst->coords[i]   : (x, y) coordinates of node i
// inst->name        : instance name (string)
// state->current_tour    : list of visited nodes in order
// state->remaining       : list of unvisited nodes
// state->tour_length_so_far : current tour length
// state->current_node    : last node added to tour

// Available precomputed signals (add these to your code):
// dist_to_tour[node] = min_{v in state->current_tour} inst->dist[node][v]
// centroid_x, centroid_y, angle_to_centroid, local_density

// SIGNAL SCALES (berlin52/kroA100 typical ranges — critical for weighting):
//   inst->dist[i][j]       : 0 – 600   (raw coordinate distance)
//   dist_to_tour[node]     : 0 – 400   (distance to nearest tour node)
//   state->tour_length_so_far : 0 – 15000
//   angle_to_centroid      : -π – +π   (~-3.14 to +3.14)
//   local_density          : 0.0 – 1.0
// WARNING: mixing raw distances (0–600) with angles (-π,+π) without
// normalization lets the distance term dominate by ~100×. Either divide
// each signal by its max or use (signal - mean) / std before weighting.
"""


def build_llm_prompt(island: Island, best_programs: list[CandidateProgram], generation: int, hint: str = "") -> str:
    """Build the LLM prompt for generating a new priority function."""

    # Softmax-weighted exemplar sampling — diverse programs, not just the incumbent
    sampled = _sample_exemplars(island.population, k=3, temperature=0.5)
    exemplars = ""
    for i, p in enumerate(sampled):
        exemplars += f"\n// Example {i+1} (fitness={p.fitness:.4f}):\n{p.priority_code}\n"

    sketch_name, sketch_desc = PROMPT_SKETCHES[(island.id + generation) % len(PROMPT_SKETCHES)]
    sketch_hint = f"\nApproach hint ({sketch_name}): {sketch_desc}\n"

    # Per-island persona for diversity across the island population
    persona_name, persona_desc = ISLAND_PERSONAS[island.id % len(ISLAND_PERSONAS)]
    persona_header = f"You are a {persona_name}. {persona_desc}\n\n"

    hint_section = f"\nStrategic improvement hint: {hint}\n" if hint else ""

    prompt = persona_header + f"""You are a TSP heuristic engineer. Generate a new `priority()` function for the Traveling Salesperson Problem.

The function receives a candidate node and must return a HIGHER score for more promising nodes to visit next.

{PRIORITY_SIGNATURE}
{sketch_hint}
Island {island.id} — Generation {generation}
Current population bests: {exemplars}
{hint_section}
Produce ONLY the C++ priority function implementation, inside a single ```cpp
code block. No prose before or after.

Rules:
- The function must be callable as `double priority(int node, const Instance* inst, const State* state)`
- Use `extern "C"` linkage
- Do NOT redefine structs — use the ones provided
- Keep it under 50 lines
- Composing multiple signals (distance + angle + density) usually beats single-signal heuristics
- NORMALIZE before combining: raw distances (~0–600) dominate angles (~-π,+π) by 100× unless you divide each signal by its range or max value
"""
    return prompt


def parse_llm_response(response: str) -> str:
    """Extract a C++ priority() function from an LLM response.

    MiniMax (and most models) return the code inside a markdown ```cpp fence,
    ignoring any "use these markers" instruction, so fences are tried first.
    Falls back to explicit PRIORITY_FUNCTION markers, then a bare `double
    priority(...)` definition. Returns "" when nothing usable is found — the
    caller treats an empty candidate as a failed one scoring 0.0, so a junk or
    empty LLM response degrades gracefully instead of crashing the loop.
    """
    if not response or not response.strip():
        return ""
    text = response.strip()

    # 1. Markdown code fence: ```cpp ... ``` (the common case).
    fence = re.search(r"```[a-zA-Z0-9_+\-]*[ \t]*\n?(.*?)```", text, re.DOTALL)
    if fence and fence.group(1).strip():
        return fence.group(1).strip()

    # 2. Explicit PRIORITY_FUNCTION markers (if the model honored the format).
    if "PRIORITY_FUNCTION" in text:
        parts = text.split("PRIORITY_FUNCTION")
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip().replace("```", "").strip()

    # 3. Bare definition: keep from the priority() signature onward.
    sig = re.search(r'(?:extern\s+"C"\s+)?(?:inline\s+)?double\s+priority\s*\(', text)
    if sig:
        return text[sig.start():].strip()

    # 4. Last resort: strip stray fences and hope it compiles (else fitness 0.0).
    return text.replace("```cpp", "").replace("```c++", "").replace("```", "").strip()


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
    for prog in island.population:
        if prog.evaluated:
            continue
        evaluate_fitness(prog, instances, work_dir, executor=executor, run_timeout=run_timeout)

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

    def _find_nervous_bin(self) -> str | None:
        """Search for the nervous CLI in common locations."""
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

    def _publish(self, channel: str, data: dict) -> bool:
        """Publish a CloudEvents-lite event to the nervous-bus debug log.

        Writes directly to ~/.cache/nervous-bus/debug.jsonl. This is the primary
        publish path. Optionally tries the ``nervous`` CLI for zellij/Redis
        distribution if it can be invoked without blocking.

        Args:
            channel: Event channel name (e.g. ``tsp.kernel.started.v1``).
            data: Event payload (becomes the ``data`` field of the envelope).

        Returns:
            True always (fail-silent for high-frequency use).
        """
        envelope = {
            "specversion": "1.0",
            "id": uuid.uuid4().urn,
            "source": "/autobench/tsp_kernel",
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

        # Best-effort nervous CLI for zellij/Redis consumers.
        # Runs non-blocking via Popen; failures are silently ignored.
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


# Known optimal tour lengths for benchmark instances
KNOWN_OPTIMALS: dict[str, float] = {
    "berlin52": 7542.0,
    "eil101": 629.0,
    "kroa100": 21282.0,
    "ch130": 6110.0,
    "ts225": 126643.0,
    "pr1002": 259045.0,
}