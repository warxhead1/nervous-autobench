"""LLM prompt construction, response parsing, and syntactic mutation.

This is the "oracle" edge of the TSP kernel — everything that turns the island
state into a prompt, extracts a priority() function from a model response, and
performs deterministic baseline mutations. No sandbox, no kernel loop.
"""

from __future__ import annotations

import math
import random
import re

from .instance import CandidateProgram, Island


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
