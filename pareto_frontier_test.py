"""
Tests for pareto_frontier module.
"""

import json
import math
import os
import tempfile
import pytest

from autobench.pareto_frontier import (
    dominance_check,
    dominance_check_for_point,
    update_frontier,
    query_frontier,
    compute_aaii,
    evaluate_weight_space,
    rank_frontier_by_preset,
    WeightPreset,
    WEIGHT_PRESETS,
    ParetoFrontier,
    StorageBackend,
    FrontierConfig,
    UpdateResult,
    benchmark_result_to_metrics,
    frontier_config_from_benchmark_result,
)


# ---------------------------------------------------------------------------
# dominance_check tests
# ---------------------------------------------------------------------------

class TestDominanceCheck:
    def test_dominates_all_better(self):
        """A config that is strictly better in all dimensions dominates."""
        a = {"quality": 0.8, "cost": 0.2, "speed": 0.9}
        b = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        assert dominance_check(a, b) is True

    def test_dominates_one_better_rest_equal(self):
        """A config that is better in one dimension and equal in others dominates."""
        a = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        b = {"quality": 0.5, "cost": 0.5, "speed": 0.4}
        assert dominance_check(a, b) is True

    def test_does_not_dominate_worse_in_one_dimension(self):
        """A config that is worse in any dimension does not dominate."""
        a = {"quality": 0.8, "cost": 0.6, "speed": 0.5}  # cost worse
        b = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        assert dominance_check(a, b) is False

    def test_does_not_dominate_equal(self):
        """A config that is equal in all dimensions does not dominate (needs strict >)."""
        a = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        b = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        assert dominance_check(a, b) is False

    def test_dominates_with_lower_cost(self):
        """Lower cost is better — a config with half the cost dominates."""
        a = {"quality": 0.5, "cost": 0.25, "speed": 0.5}
        b = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        assert dominance_check(a, b) is True

    def test_scalar_version_matches(self):
        """The scalar version matches the dict version."""
        cases = [
            ({"quality": 0.8, "cost": 0.2, "speed": 0.9}, {"quality": 0.5, "cost": 0.5, "speed": 0.5}),
            ({"quality": 0.5, "cost": 0.5, "speed": 0.5}, {"quality": 0.5, "cost": 0.5, "speed": 0.4}),
            ({"quality": 0.8, "cost": 0.6, "speed": 0.5}, {"quality": 0.5, "cost": 0.5, "speed": 0.5}),
        ]
        for a, b in cases:
            dict_result = dominance_check(a, b)
            scalar_result = dominance_check_for_point(
                a["quality"], a["cost"], a["speed"],
                b["quality"], b["cost"], b["speed"],
            )
            assert dict_result == scalar_result


# ---------------------------------------------------------------------------
# update_frontier tests
# ---------------------------------------------------------------------------

class TestUpdateFrontier:
    def test_adds_first_config(self):
        """First config becomes the frontier."""
        frontier = []
        new_config = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        updated, dominated = update_frontier(frontier, new_config)
        assert len(updated) == 1
        assert dominated == []
        assert updated[0] == new_config

    def test_does_not_add_dominated_config(self):
        """A config dominated by existing frontier is rejected."""
        frontier = [{"quality": 0.8, "cost": 0.2, "speed": 0.9}]
        new_config = {"quality": 0.5, "cost": 0.5, "speed": 0.5}
        updated, dominated = update_frontier(frontier, new_config)
        assert len(updated) == 1  # frontier unchanged
        assert dominated == []

    def test_removes_dominated_configs(self):
        """New config that dominates existing configs removes them."""
        frontier = [
            {"quality": 0.5, "cost": 0.5, "speed": 0.5},
        ]
        new_config = {"quality": 0.8, "cost": 0.2, "speed": 0.9}
        updated, dominated = update_frontier(frontier, new_config)
        assert len(updated) == 1
        assert updated[0] == new_config
        assert len(dominated) == 1
        assert dominated[0] == {"quality": 0.5, "cost": 0.5, "speed": 0.5}

    def test_removes_multiple_dominated(self):
        """New config can dominate multiple existing configs."""
        frontier = [
            {"quality": 0.5, "cost": 0.5, "speed": 0.5},
            {"quality": 0.6, "cost": 0.6, "speed": 0.6},
        ]
        new_config = {"quality": 0.9, "cost": 0.1, "speed": 0.9}
        updated, dominated = update_frontier(frontier, new_config)
        assert len(updated) == 1
        assert len(dominated) == 2

    def test_does_not_remove_non_dominated(self):
        """New config that is worse in some dimension doesn't remove others."""
        frontier = [
            {"quality": 0.8, "cost": 0.2, "speed": 0.5},
            {"quality": 0.5, "cost": 0.8, "speed": 0.8},
        ]
        new_config = {"quality": 0.6, "cost": 0.6, "speed": 0.6}
        updated, dominated = update_frontier(frontier, new_config)
        assert len(updated) == 3  # all three on frontier
        assert dominated == []


# ---------------------------------------------------------------------------
# query_frontier tests
# ---------------------------------------------------------------------------

class TestQueryFrontier:
    def setup_method(self):
        self.frontier = [
            {"quality": 0.9, "cost": 0.1, "speed": 0.9},   # excellent all around
            {"quality": 0.8, "cost": 0.05, "speed": 0.7}, # cheap, decent quality
            {"quality": 0.7, "cost": 0.02, "speed": 0.95},# very cheap, fast
            {"quality": 0.95, "cost": 0.3, "speed": 0.6}, # high quality, expensive, slow
            {"quality": 0.6, "cost": 0.01, "speed": 0.5}, # cheapest, slowest
        ]

    def test_no_filters_returns_all(self):
        """With no filters, returns all frontier configs."""
        results = query_frontier(self.frontier)
        assert len(results) == 5

    def test_cost_budget_filters(self):
        """Cost budget filters out configs above threshold."""
        results = query_frontier(self.frontier, cost_budget=0.1)
        # Costs: 0.1, 0.05, 0.02, 0.3, 0.01 -> 4 pass (only 0.3 excluded)
        assert len(results) == 4
        for r in results:
            assert r["cost"] <= 0.1

    def test_time_budget_maps_to_speed(self):
        """Time budget of e.g. 10s maps to speed threshold."""
        # speed = 1/(1+time_budget) for 10s = 0.09
        # Only configs with speed >= 0.09 pass
        results = query_frontier(self.frontier, time_budget=10.0)
        # Most configs should pass this loose threshold
        assert len(results) >= 3

    def test_min_quality_filters(self):
        """Min quality filters out configs below threshold."""
        results = query_frontier(self.frontier, min_quality=0.8)
        # Quality: 0.9, 0.8, 0.7, 0.95, 0.6 -> 3 pass (0.7 and 0.6 excluded)
        assert len(results) == 3
        for r in results:
            assert r["quality"] >= 0.8

    def test_combined_filters(self):
        """Multiple filters can be combined."""
        results = query_frontier(
            self.frontier,
            cost_budget=0.15,
            min_quality=0.7,
        )
        assert len(results) == 3
        for r in results:
            assert r["cost"] <= 0.15
            assert r["quality"] >= 0.7


# ---------------------------------------------------------------------------
# compute_aaii tests
# ---------------------------------------------------------------------------

class TestComputeAAII:
    def test_empty_frontier(self):
        """Empty frontier returns 1.0 (worst)."""
        assert compute_aaii([]) == 1.0

    def test_perfect_frontier(self):
        """Frontier at ideal point returns 0.0."""
        frontier = [{"quality": 1.0, "cost": 0.0, "speed": 1.0}]
        assert compute_aaii(frontier) == 0.0

    def test_monotonic_decreasing(self):
        """More configs on frontier -> lower AAII (better)."""
        simple = [{"quality": 0.5, "cost": 0.5, "speed": 0.5}]
        good = [
            {"quality": 1.0, "cost": 0.0, "speed": 1.0},
            {"quality": 0.5, "cost": 0.5, "speed": 0.5},
        ]
        aaii_simple = compute_aaii(simple)
        aaii_good = compute_aaii(good)
        # Good frontier should have lower (better) AAII
        assert aaii_good < aaii_simple


# ---------------------------------------------------------------------------
# Weight space tests
# ---------------------------------------------------------------------------

class TestWeightSpace:
    def setup_method(self):
        self.frontier = [
            {"quality": 0.9, "cost": 0.1, "speed": 0.9},   # best overall
            {"quality": 0.8, "cost": 0.05, "speed": 0.7}, # cheapest
            {"quality": 0.7, "cost": 0.02, "speed": 0.95},# fastest
        ]

    def test_balanced_preset_weights_sum_to_one(self):
        """All presets should sum to 1.0 (or close)."""
        for preset, weights in WEIGHT_PRESETS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{preset} sums to {total}"

    def test_evaluate_weight_space_returns_scored_configs(self):
        """evaluate_weight_space returns ScoredConfig list sorted by score."""
        results = evaluate_weight_space(self.frontier, WeightPreset.BALANCED)
        assert len(results) == 3
        # Should be sorted highest first
        assert results[0].score >= results[1].score >= results[2].score

    def test_quality_first_prefers_high_quality(self):
        """Quality-first preset ranks high-quality config first."""
        results = evaluate_weight_space(self.frontier, WeightPreset.QUALITY_FIRST)
        assert results[0].config["quality"] == 0.9

    def test_cost_focused_prefers_cheapest(self):
        """Cost-focused preset ranks cheapest config first."""
        results = evaluate_weight_space(self.frontier, WeightPreset.COST_FOCUSED)
        assert results[0].config["cost"] == 0.02  # cheapest

    def test_speed_first_prefers_fastest(self):
        """Speed-first preset ranks fastest config first."""
        results = evaluate_weight_space(self.frontier, WeightPreset.SPEED_FIRST)
        assert results[0].config["speed"] == 0.95  # fastest

    def test_rank_frontier_by_preset_returns_config_list(self):
        """rank_frontier_by_preset returns list of config dicts."""
        results = rank_frontier_by_preset(self.frontier, WeightPreset.BALANCED)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)


# ---------------------------------------------------------------------------
# ParetoFrontier persistence tests
# ---------------------------------------------------------------------------

class TestParetoFrontierJSON:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.tmpdir, "frontier.json")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_save_load_roundtrip_empty(self):
        """Empty frontier saves and loads correctly."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.save()
        pf2 = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf2.load()
        assert pf2.size == 0

    def test_save_load_roundtrip_with_configs(self):
        """Configs survive save/load cycle."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.add_config(
            metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9},
            harness_config={"model": "claude-3-5", "temperature": 0.7},
        )
        pf.save()

        pf2 = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf2.load()
        assert pf2.size == 1
        config = pf2.frontier[0]
        assert config["metrics"]["quality"] == 0.8

    def test_add_config_returns_update_result(self):
        """add_config returns UpdateResult with correct fields."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        result = pf.add_config(
            metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9},
            harness_config={"model": "claude-3-5"},
        )
        assert isinstance(result, UpdateResult)
        assert result.was_added is True
        assert result.was_dominated is False
        assert result.frontier_size == 1

    def test_dominated_config_not_added(self):
        """A dominated config is not added to frontier."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.add_config(
            metrics={"quality": 0.9, "cost": 0.1, "speed": 0.9},
            harness_config={"model": "best"},
        )
        result = pf.add_config(
            metrics={"quality": 0.5, "cost": 0.5, "speed": 0.5},
            harness_config={"model": "worst"},
        )
        assert result.was_added is False
        assert result.was_dominated is True
        assert pf.size == 1

    def test_domination_removes_existing_configs(self):
        """New config that dominates existing configs removes them."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.add_config(
            metrics={"quality": 0.5, "cost": 0.5, "speed": 0.5},
            harness_config={"model": "old"},
        )
        result = pf.add_config(
            metrics={"quality": 0.9, "cost": 0.1, "speed": 0.9},
            harness_config={"model": "new"},
        )
        assert result.was_added is True
        assert len(result.newly_dominated) == 1
        assert pf.size == 1
        assert pf.frontier[0]["harness_config"]["model"] == "new"

    def test_query_with_cost_budget(self):
        """query filters by cost budget."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.add_config(metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9})
        pf.add_config(metrics={"quality": 0.6, "cost": 0.05, "speed": 0.5})
        pf.add_config(metrics={"quality": 0.7, "cost": 0.3, "speed": 0.8})

        results = pf.query(cost_budget=0.25)
        assert len(results) == 2  # excludes the 0.3 cost one
        for r in results:
            assert r["metrics"]["cost"] <= 0.25

    def test_rank_by_preset(self):
        """rank returns configs sorted by weight preset."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.add_config(metrics={"quality": 0.5, "cost": 0.05, "speed": 0.8})
        pf.add_config(metrics={"quality": 0.9, "cost": 0.4, "speed": 0.3})

        ranked = pf.rank(WeightPreset.QUALITY_FIRST)
        assert ranked[0]["metrics"]["quality"] == 0.9

        ranked_cost = pf.rank(WeightPreset.COST_FOCUSED)
        assert ranked_cost[0]["metrics"]["cost"] == 0.05

    def test_aaii_computed(self):
        """aaii() returns a float between 0 and 1."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.add_config(metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9})
        aaii = pf.aaii()
        assert isinstance(aaii, float)
        assert 0.0 <= aaii <= 1.0

    def test_clear_removes_all(self):
        """clear removes all configs including dominated."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        pf.add_config(metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9})
        pf.add_config(metrics={"quality": 0.9, "cost": 0.1, "speed": 0.9})  # dominates first
        pf.clear()
        assert pf.size == 0
        assert pf.aaii() == 1.0  # empty frontier

    def test_remove_config(self):
        """remove_config deletes specific config by ID."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        result = pf.add_config(metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9})
        config_id = result.config_id
        assert pf.size == 1
        removed = pf.remove_config(config_id)
        assert removed is True
        assert pf.size == 0

    def test_get_config(self):
        """get_config returns specific config or None."""
        pf = ParetoFrontier(storage=StorageBackend.JSON, path=self.json_path, auto_save=False)
        result = pf.add_config(metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9})
        config = pf.get_config(result.config_id)
        assert config is not None
        assert config["metrics"]["quality"] == 0.8
        assert pf.get_config("nonexistent") is None


class TestParetoFrontierSQLite:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "frontier.db")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_save_load_roundtrip(self):
        """SQLite backend survives save/load cycle."""
        pf = ParetoFrontier(storage=StorageBackend.SQLITE, path=self.db_path, auto_save=False)
        pf.add_config(
            metrics={"quality": 0.8, "cost": 0.2, "speed": 0.9},
            harness_config={"model": "claude-3-5"},
        )
        pf.save()

        pf2 = ParetoFrontier(storage=StorageBackend.SQLITE, path=self.db_path, auto_save=False)
        pf2.load()
        assert pf2.size == 1

    def test_dominated_ids_preserved(self):
        """Dominated config IDs are persisted (archive)."""
        pf = ParetoFrontier(storage=StorageBackend.SQLITE, path=self.db_path, auto_save=False)
        pf.add_config(metrics={"quality": 0.5, "cost": 0.5, "speed": 0.5})
        pf.add_config(metrics={"quality": 0.9, "cost": 0.1, "speed": 0.9})  # dominates first

        # Save and reload
        pf.save()
        pf2 = ParetoFrontier(storage=StorageBackend.SQLITE, path=self.db_path, auto_save=False)
        pf2.load()
        assert pf2.size == 1
        # The dominated config should still be in all_configs (archive)
        all_configs = pf2.all_configs
        assert len(all_configs) == 2


# ---------------------------------------------------------------------------
# Integration with BenchmarkResult tests
# ---------------------------------------------------------------------------

class TestBenchmarkResultIntegration:
    def test_benchmark_result_to_metrics(self):
        """Converts BenchmarkResult to metrics dict."""
        class FakeResult:
            aggregate_score = 0.85
            total_latency_ms = 1500.0
            verdict_counts = {"OK": 80, "WA": 10, "CE": 10}

        result = FakeResult()
        metrics = benchmark_result_to_metrics(result)

        assert "quality" in metrics
        assert "cost" in metrics
        assert "speed" in metrics
        assert 0.0 <= metrics["quality"] <= 1.0
        assert 0.0 <= metrics["cost"] <= 1.0
        assert 0.0 <= metrics["speed"] <= 1.0

    def test_frontier_config_from_benchmark_result(self):
        """Creates FrontierConfig from BenchmarkResult."""
        class FakeResult:
            aggregate_score = 0.8
            total_latency_ms = 2000.0
            verdict_counts = {"OK": 70, "CE": 30}

        result = FakeResult()
        harness = {"model": "claude-3-5-sonnet", "temperature": 0.7}

        fc = frontier_config_from_benchmark_result(result, harness)

        assert isinstance(fc, FrontierConfig)
        assert fc.harness_config["model"] == "claude-3-5-sonnet"
        assert fc.metrics["quality"] == 0.8
        assert 0.0 <= fc.metrics["speed"] <= 1.0


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])