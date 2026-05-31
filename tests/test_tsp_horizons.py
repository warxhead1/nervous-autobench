"""Tests for horizon governance — budget-driven stopping.

Mock llm_call_fn so no MiniMax requests are spent; evaluation still runs in the
sandbox (these assert on the stop reason / counts, which don't depend on the
exact fitness value).
"""

from __future__ import annotations

from autobench.tsp_kernel import KernelConfig, TSPKernel

_NN = (
    'extern "C" double priority(int node, const Instance* inst, const State* state)'
    '{ double b=1e100; for(int t:state->current_tour) b=std::min(b,inst->dist[node][t]); return -b; }'
)


def _mock(_prompt: str) -> str:
    return f"```cpp\n{_NN}\n```"


def test_request_budget_is_not_overshot():
    cfg = KernelConfig(
        instances=["berlin52"], n_islands=2, population_per_island=4,
        generations=100, candidates_per_island=2, max_requests=8,
        migration_interval=5, llm_call_fn=_mock,
    )
    kernel = TSPKernel(cfg)
    kernel.run()
    # A wave is n_islands * candidates_per_island = 4; the loop must stop before
    # a wave would exceed the budget, so requests never exceed max_requests.
    assert kernel.llm_requests <= 8
    assert "request budget" in kernel.stop_reason


def test_plateau_stops_when_no_improvement():
    cfg = KernelConfig(
        instances=["berlin52"], n_islands=2, population_per_island=4,
        generations=100, candidates_per_island=1, plateau_generations=2,
        migration_interval=5, llm_call_fn=_mock,
    )
    kernel = TSPKernel(cfg)
    kernel.run()
    assert "plateau" in kernel.stop_reason
    assert kernel.generation <= 4  # a couple of stale gens, not the full 100


def test_target_fitness_stops_without_wasting_requests():
    # Baselines already exceed 0.3, so the loop should stop before any LLM call.
    cfg = KernelConfig(
        instances=["berlin52"], n_islands=2, population_per_island=4,
        generations=100, candidates_per_island=1, target_fitness=0.3,
        llm_call_fn=_mock,
    )
    kernel = TSPKernel(cfg)
    kernel.run()
    assert "target_fitness reached" in kernel.stop_reason
    assert kernel.llm_requests == 0


def test_generation_cap_is_the_hard_bound():
    cfg = KernelConfig(
        instances=["berlin52"], n_islands=1, population_per_island=3,
        generations=2, candidates_per_island=1, migration_interval=5,
        llm_call_fn=_mock,
    )
    kernel = TSPKernel(cfg)
    kernel.run()
    assert "generation cap" in kernel.stop_reason
    assert kernel.generation == 2
