"""Sandboxed C++ compile/run/score for TSP candidate priority() functions.

Candidate priority() functions are LLM-generated and therefore UNTRUSTED. The
dangerous part — running the produced binary with arbitrary syscalls — happens
only inside autobench.sandbox.SandboxedExecutor in gVisor mode (rootless runsc,
--network=none), the same isolation the codeforces/shader benchmarks use.
Compilation runs on the host (fixed flags, source-only untrusted input,
resource-capped); see _run_subprocess for why a sandboxed compile cannot
persist its binary. The kernel refuses to run unless an isolating sandbox is
active for EXECUTION (see ensure_sandboxed_executor); set allow_unsandboxed=True
only for trusted, attended experiments.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .instance import CandidateProgram, TSPInstance, Tourevaluation

from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run
# FunSearchKernel + ensure_sandboxed_executor + KernelConfig come from autobench.kernels.
# We re-export them below for back-compat with existing imports inside this file.
from ..kernels import (
    FunSearchKernel,
    KernelConfig,
    ensure_sandboxed_executor,
    UnsafeSandboxError,
    register_kernel,
)

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Sandboxed C++ runner
# ---------------------------------------------------------------------------

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
