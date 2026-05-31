"""Tests for the throughput rework: eval-once + concurrent generation.

These use a mock llm_call_fn so they don't spend MiniMax requests, and a
monkeypatched evaluate_fitness so they don't invoke the gVisor sandbox — the
logic under test is the scheduling/accounting, not the C++ run.
"""

from __future__ import annotations

import threading

from autobench import tsp_kernel as tk
from autobench.tsp_kernel import CandidateProgram, Island, evaluate_island


def test_eval_once_skips_already_evaluated(monkeypatch):
    """evaluate_island runs only programs whose `evaluated` flag is False."""
    ran: list[str] = []

    def fake_eval(prog, instances, work_dir=None, executor=None, run_timeout=10.0):
        ran.append(prog.id)
        prog.fitness = 0.5
        prog.evaluated = True
        return 0.5, 0.0, 0.5

    monkeypatch.setattr(tk, "evaluate_fitness", fake_eval)

    island = Island(id=0)
    already = CandidateProgram(
        id="already", priority_code="", island=0, generation=0,
        fitness=0.9, evaluated=True,
    )
    fresh = CandidateProgram(id="fresh", priority_code="", island=0, generation=0)
    island.population = [already, fresh]

    evaluate_island(island, [], executor=None)

    assert ran == ["fresh"], "only the un-evaluated program should be scored"
    assert island.best_program.id == "already"  # higher fitness, not re-run


def test_llm_request_count_exact_under_concurrency():
    """candidates_per_island * n_islands calls per generation, counted exactly."""
    nn = (
        'extern "C" double priority(int node, const Instance* inst, const State* state)'
        '{ double b=1e100; for(int t:state->current_tour) b=std::min(b,inst->dist[node][t]); return -b; }'
    )
    seen = {"n": 0}
    lock = threading.Lock()

    def mock(prompt):
        with lock:
            seen["n"] += 1
        return f"```cpp\n{nn}\n```"

    cfg = tk.KernelConfig(
        instances=["berlin52"], n_islands=2, population_per_island=4,
        generations=2, migration_interval=1,
        candidates_per_island=3, max_concurrent_llm=6, llm_call_fn=mock,
    )
    kernel = tk.TSPKernel(cfg)
    kernel.run()

    expected = 2 * 3 * 2  # islands * candidates_per_island * generations
    assert kernel.llm_requests == expected
    assert seen["n"] == expected
    # Phase timing is recorded for every generation.
    assert all("gen_seconds" in h and "eval_seconds" in h for h in kernel.history)
