"""Tests for the racing-line FunSearch kernel.

Covers:
  - Instance generation (all four track layouts, seeded determinism)
  - Oracle generativity: two structurally different valid policies both score high
  - Oracle degenerate cases: off-track, stopped vehicle
  - Baselines derisking: all three seed policies score in [0.70, 0.90]
  - Prompt builder and response parser
  - RacingKernel kernel registration and seed_programs

All tests run without LLM, network, or sandbox — marked not-live.
"""

from __future__ import annotations

import math
import re

import pytest

from autobench.racing_kernel.instance import (
    RacingInstance,
    TRACK_LAYOUTS,
    generate_instance,
)
from autobench.racing_kernel.oracle import (
    SEED_RACING_PROGRAMS,
    _compile_policy,
    _speed_score,
    _smoothness_score,
    _track_membership_score,
    _rhythm_score,
    _simulate_lap,
    evaluate_on_instance,
    build_llm_prompt,
    parse_llm_response,
)
from autobench.kernels import KernelConfig, CandidateProgram, Island


# ---------------------------------------------------------------------------
# Instance tests
# ---------------------------------------------------------------------------

class TestRacingInstance:
    def test_all_layouts_generate(self):
        for name in TRACK_LAYOUTS:
            inst = generate_instance(name)
            assert isinstance(inst, RacingInstance)
            assert inst.name == name
            assert inst.lap_length > 0
            assert len(inst.centerline) > 20
            assert len(inst.curvatures) == len(inst.centerline)
            assert len(inst.speed_limit) == len(inst.centerline)
            assert all(v > 0 for v in inst.speed_limit)

    def test_seeded_determinism(self):
        """Same name → identical instance every time (seeded RNG)."""
        a = generate_instance("chicane")
        b = generate_instance("chicane")
        assert a.centerline == b.centerline
        assert a.curvatures == b.curvatures
        assert abs(a.lap_length - b.lap_length) < 1e-9

    def test_distinct_layouts_differ(self):
        oval = generate_instance("oval")
        complex_ = generate_instance("complex")
        # Oval (6 control pts, low perturbation) vs complex (14 pts, high perturbation):
        # either lap length OR curvature profiles must differ significantly.
        lap_diff = abs(oval.lap_length - complex_.lap_length)
        # complex has many more control points so the curvature signature is richer
        max_curv_oval = max(abs(c) for c in oval.curvatures)
        max_curv_complex = max(abs(c) for c in complex_.curvatures)
        # At least one geometric property should differ noticeably
        assert lap_diff > 5.0 or abs(max_curv_oval - max_curv_complex) > 0.005, (
            f"oval and complex appear identical: lap_diff={lap_diff:.2f}, "
            f"curv_diff={abs(max_curv_oval - max_curv_complex):.6f}"
        )

    def test_unknown_layout_raises(self):
        with pytest.raises(ValueError, match="Unknown track"):
            generate_instance("nonexistent_track")

    def test_half_width_positive(self):
        for name in TRACK_LAYOUTS:
            inst = generate_instance(name)
            assert inst.half_width > 0

    def test_ref_lap_time_reasonable(self):
        for name in TRACK_LAYOUTS:
            inst = generate_instance(name)
            # Should be more than 5 s and less than 5 minutes
            assert 5.0 < inst.ref_lap_time_s < 300.0


# ---------------------------------------------------------------------------
# Oracle unit tests
# ---------------------------------------------------------------------------

class TestOracleTerms:
    def test_speed_score_at_reference(self):
        """At exactly the reference (physics-ceiling) lap time → score = 1.0."""
        s = _speed_score(30.0, 30.0)
        # ratio = ref/lap = 1.0 → exp(-k*0) = 1.0 (at the physics ceiling)
        assert s == 1.0

    def test_speed_score_faster_is_better(self):
        s_fast = _speed_score(25.0, 30.0)
        s_slow = _speed_score(40.0, 30.0)
        assert s_fast > s_slow

    def test_speed_score_clamped(self):
        s = _speed_score(1.0, 30.0)   # impossibly fast
        assert 0.0 <= s <= 1.0
        s2 = _speed_score(1000.0, 30.0)  # impossibly slow
        assert 0.0 <= s2 <= 1.0

    def test_smoothness_score_constant_line(self):
        """A perfectly constant lateral offset gets smoothness ≈ 1.0."""
        offsets = [0.5] * 50
        curvs = [0.0] * 50
        s = _smoothness_score(offsets, curvs)
        assert s > 0.95

    def test_smoothness_score_jerky_line(self):
        """An alternating offset (very jerky) gets a lower smoothness score."""
        offsets = [(-1.0) ** i * 3.0 for i in range(50)]
        curvs = [0.0] * 50
        s_jerky = _smoothness_score(offsets, curvs)
        s_smooth = _smoothness_score([0.5] * 50, curvs)
        assert s_jerky < s_smooth

    def test_track_score_all_in_bounds(self):
        offsets = [0.0, 1.0, -1.0, 2.0, -2.0]
        s = _track_membership_score(offsets, half_width=3.0)
        assert s == 1.0

    def test_track_score_all_out(self):
        offsets = [5.0, 6.0, -5.0]
        s = _track_membership_score(offsets, half_width=3.0)
        assert s == 0.0

    def test_track_score_mixed(self):
        # 3 in, 2 out → 0.6
        offsets = [0.0, 0.0, 0.0, 5.0, -5.0]
        s = _track_membership_score(offsets, half_width=2.0)
        assert abs(s - 0.6) < 0.01


# ---------------------------------------------------------------------------
# Baseline derisking — seed programs must score in [0.70, 0.90]
# ---------------------------------------------------------------------------

class TestBaselineDerisking:
    """Critical: seeds must be 0.70–0.90 (not trivial, not impossibly hard)."""

    @pytest.mark.parametrize("track_name", list(TRACK_LAYOUTS.keys()))
    @pytest.mark.parametrize("seed_name,seed_code", SEED_RACING_PROGRAMS)
    def test_seed_in_band(self, track_name, seed_name, seed_code):
        inst = generate_instance(track_name)
        score = evaluate_on_instance(seed_code, inst)
        assert score is not None, f"{seed_name} returned None on {track_name}"
        assert 0.50 <= score <= 0.95, (
            f"Seed '{seed_name}' on track '{track_name}' scored {score:.4f} — "
            f"outside [0.50, 0.95] (target: 0.70–0.90)"
        )
        # Tight band assertion
        assert 0.70 <= score <= 0.90, (
            f"Seed '{seed_name}' on track '{track_name}' scored {score:.4f} — "
            f"outside the 0.70–0.90 target band"
        )


# ---------------------------------------------------------------------------
# Generativity test — two structurally different valid policies both score high
# ---------------------------------------------------------------------------

class TestOracleGenerativity:
    """CRITICAL: the oracle must be generative, not singleton-matching."""

    _POLICY_APEX = """\
def racing_line(u, curvature, half_width, speed_limit):
    # Apex strategy: aggressively use the inside of corners
    lateral_offset = -curvature * half_width * 0.75
    lateral_offset = max(-half_width * 0.8, min(half_width * 0.8, lateral_offset))
    throttle = max(0.5, min(1.0, speed_limit / 20.0))
    return lateral_offset, throttle
"""

    _POLICY_CENTER = """\
def racing_line(u, curvature, half_width, speed_limit):
    # Conservative center-line strategy — different structure from apex
    import math
    # Sinusoidal offset tuned to track rhythm
    lateral_offset = -math.sin(u * math.pi * 2) * half_width * 0.25
    # Brake earlier, exit faster
    throttle = max(0.6, speed_limit / 22.0)
    return lateral_offset, throttle
"""

    def test_two_different_valid_lines_both_score_high(self):
        """Two structurally different policies must both score > 0.65.

        If either scores < 0.65, the oracle is behaving as a singleton
        (only one 'correct' line).
        """
        inst = generate_instance("oval")
        s_apex = evaluate_on_instance(self._POLICY_APEX, inst)
        s_center = evaluate_on_instance(self._POLICY_CENTER, inst)

        assert s_apex is not None, "apex policy failed to evaluate"
        assert s_center is not None, "center policy failed to evaluate"

        assert s_apex > 0.65, (
            f"Apex policy scored {s_apex:.4f} — oracle too restrictive (singleton trap)"
        )
        assert s_center > 0.65, (
            f"Center policy scored {s_center:.4f} — oracle too restrictive (singleton trap)"
        )

    def test_off_track_policy_scores_low(self):
        """A policy that always drives off-track must score low on track_score."""
        off_track_code = """\
def racing_line(u, curvature, half_width, speed_limit):
    # Always drives far outside the track boundary
    return half_width * 5.0, 0.5
"""
        inst = generate_instance("oval")
        score = evaluate_on_instance(off_track_code, inst)
        assert score is not None
        # track_score term (weight 0.25) should pull the fitness down
        assert score < 0.80, (
            f"Off-track policy scored {score:.4f} — track membership not penalised"
        )

    def test_stopped_policy_scores_low_on_speed(self):
        """A policy that provides zero throttle must score low on speed."""
        stopped_code = """\
def racing_line(u, curvature, half_width, speed_limit):
    return 0.0, 0.0  # zero throttle — should be penalised on speed
"""
        inst = generate_instance("oval")
        score = evaluate_on_instance(stopped_code, inst)
        assert score is not None
        # Speed weight is 0.45; zero throttle → very slow lap → low speed_score
        assert score < 0.70, (
            f"Stopped policy scored {score:.4f} — speed term not penalising slow laps"
        )


# ---------------------------------------------------------------------------
# Failure / robustness cases
# ---------------------------------------------------------------------------

class TestOracleRobustness:
    def test_broken_code_returns_none(self):
        score = evaluate_on_instance("this is not valid python !!!", generate_instance("oval"))
        assert score is None

    def test_no_racing_line_fn_returns_none(self):
        code = "x = 1 + 1\n"
        score = evaluate_on_instance(code, generate_instance("oval"))
        assert score is None

    def test_wrong_return_type_returns_none(self):
        code = "def racing_line(u, c, hw, sl): return 42\n"
        score = evaluate_on_instance(code, generate_instance("oval"))
        assert score is None

    def test_exception_in_policy_graceful(self):
        # Policy raises on first call — should still return None (not crash)
        code = """\
def racing_line(u, curvature, half_width, speed_limit):
    raise RuntimeError("deliberate crash")
"""
        # _compile_policy catches the smoke-test crash
        score = evaluate_on_instance(code, generate_instance("oval"))
        assert score is None


# ---------------------------------------------------------------------------
# Prompt builder / parser
# ---------------------------------------------------------------------------

class TestPromptAndParser:
    def _dummy_island(self, island_id: int = 0) -> Island:
        from autobench.kernels.base import Island
        return Island(id=island_id)

    def test_prompt_contains_function_signature(self):
        island = self._dummy_island(0)
        prompt = build_llm_prompt(island, [], generation=0, instance_name="oval")
        assert "racing_line" in prompt
        assert "lateral_offset" in prompt
        assert "throttle" in prompt
        assert "oval" in prompt

    def test_prompt_persona_varies_by_island(self):
        p0 = build_llm_prompt(self._dummy_island(0), [], generation=0)
        p1 = build_llm_prompt(self._dummy_island(1), [], generation=0)
        # Different islands get different personas (they cycle through ISLAND_PERSONAS)
        assert p0 != p1

    def test_parse_response_fenced_block(self):
        response = (
            "Here is my solution:\n"
            "```python\n"
            "def racing_line(u, curvature, half_width, speed_limit):\n"
            "    return 0.0, 1.0\n"
            "```\n"
        )
        code = parse_llm_response(response)
        assert "def racing_line" in code
        assert "return 0.0, 1.0" in code

    def test_parse_response_bare_def(self):
        response = (
            "Thinking...\n"
            "def racing_line(u, curvature, half_width, speed_limit):\n"
            "    return 0.5, 0.8\n"
        )
        code = parse_llm_response(response)
        assert "def racing_line" in code

    def test_parse_response_no_match_returns_empty(self):
        code = parse_llm_response("No function here at all.")
        assert code == ""


# ---------------------------------------------------------------------------
# RacingKernel integration (no LLM, no sandbox)
# ---------------------------------------------------------------------------

class TestRacingKernel:
    def test_kernel_registered(self):
        from autobench.kernels.base import KERNEL_REGISTRY
        assert "racing" in KERNEL_REGISTRY

    def test_seed_programs_return_three(self):
        from autobench.racing_kernel.loop import RacingKernel
        config = KernelConfig(
            instances=["oval"],
            n_islands=1,
            population_per_island=3,
            generations=1,
            allow_unsandboxed=True,
        )
        kernel = RacingKernel(config)
        seeds = kernel.seed_programs(island_id=0, generation=0)
        assert len(seeds) == 3
        assert all(isinstance(s, CandidateProgram) for s in seeds)
        assert all(s.source == "baseline" for s in seeds)

    def test_evaluate_fitness_on_seed(self):
        from autobench.racing_kernel.loop import RacingKernel
        config = KernelConfig(
            instances=["oval"],
            n_islands=1,
            population_per_island=3,
            generations=1,
            allow_unsandboxed=True,
        )
        kernel = RacingKernel(config)
        seeds = kernel.seed_programs(island_id=0, generation=0)
        prog = seeds[0]
        mean, var, worst = kernel.evaluate_fitness(prog)
        assert 0.0 < mean <= 1.0
        assert var >= 0.0
        assert worst >= 0.0

    def test_load_instances_returns_all_four_tracks(self):
        from autobench.racing_kernel.loop import RacingKernel
        config = KernelConfig(
            instances=["oval", "chicane", "hairpin", "complex"],
            n_islands=4,
            population_per_island=3,
            generations=1,
            allow_unsandboxed=True,
        )
        kernel = RacingKernel(config)
        assert len(kernel.problem_instances) == 4
        names = [inst.name for inst in kernel.problem_instances]
        assert "oval" in names
        assert "complex" in names


# ---------------------------------------------------------------------------
# Rhythm / cadence score tests (nervous-bus-71cn.4)
# ---------------------------------------------------------------------------

class TestRhythmScore:
    """Tests for the spectral cadence term: _rhythm_score.

    CADENCE NOTE: The series is sampled per-centerline-point (uniform
    arc-length, not time), so "rhythm" is spatial-frequency cadence along
    the lap, not wall-clock temporal rhythm.
    """

    # ---- degenerate inputs -------------------------------------------------

    def test_constant_series_returns_neutral(self):
        """A perfectly constant throttle has no AC spectral structure → neutral 0.5."""
        s = _rhythm_score([0.7] * 100)
        assert abs(s - 0.5) < 0.01, (
            f"Constant series should return exactly 0.5, got {s:.4f}"
        )

    def test_short_series_returns_neutral(self):
        """Fewer than 8 samples → not enough to estimate cadence → neutral 0.5."""
        s = _rhythm_score([0.5] * 5)
        assert s == 0.5

    def test_empty_series_returns_neutral(self):
        s = _rhythm_score([])
        assert s == 0.5

    def test_output_in_unit_interval(self):
        """Result is always in [0, 1] regardless of input."""
        import random
        random.seed(0)
        for _ in range(10):
            series = [random.uniform(-2.0, 3.0) for _ in range(80)]
            s = _rhythm_score(series)
            assert 0.0 <= s <= 1.0, f"rhythm_score out of range: {s}"

    # ---- discrimination: rhythmic vs arrhythmic ----------------------------

    def test_rhythmic_beats_constant_by_margin(self):
        """A sinusoidal throttle (periodic cadence) must score meaningfully
        higher than a constant throttle.

        Margin ≥ 0.15 in rhythm-term space (not total fitness).
        """
        N = 120
        # Rhythmic: corner-like brake/throttle cycle, ~4 corners per lap
        rhythmic = [math.sin(2 * math.pi * 4 * i / N) * 0.25 + 0.65 for i in range(N)]
        constant = [0.7] * N  # DC only — no modulation

        s_rhythmic = _rhythm_score(rhythmic)
        s_constant = _rhythm_score(constant)  # should be 0.5 (neutral)

        margin = s_rhythmic - s_constant
        assert margin >= 0.15, (
            f"Rhythmic series ({s_rhythmic:.3f}) should score ≥0.15 above "
            f"constant ({s_constant:.3f}); margin={margin:.3f}"
        )

    def test_rhythmic_beats_white_noise_by_margin(self):
        """A sinusoidal throttle must score meaningfully higher than white noise.

        Margin ≥ 0.10 in rhythm-term space.
        """
        import random
        random.seed(42)
        N = 120
        rhythmic = [math.sin(2 * math.pi * 4 * i / N) * 0.3 + 0.6 for i in range(N)]
        white_noise = [random.uniform(0.0, 1.0) for _ in range(N)]

        s_rhythmic = _rhythm_score(rhythmic)
        s_noise = _rhythm_score(white_noise)

        margin = s_rhythmic - s_noise
        assert margin >= 0.10, (
            f"Rhythmic ({s_rhythmic:.3f}) should score ≥0.10 above white noise "
            f"({s_noise:.3f}); margin={margin:.3f}"
        )

    def test_structured_1f_beats_white_noise(self):
        """A 1/f-structured series (correlated) should score higher than white noise."""
        import random
        random.seed(7)
        N = 120
        # Generate 1/f-like series via cumulative sum of Gaussian noise
        steps = [random.gauss(0, 0.03) for _ in range(N)]
        raw = [0.7]
        for s in steps:
            raw.append(max(0.2, min(1.0, raw[-1] + s)))
        correlated = raw[:N]
        white_noise = [random.uniform(0.0, 1.0) for _ in range(N)]

        s_corr = _rhythm_score(correlated)
        s_noise = _rhythm_score(white_noise)

        assert s_corr > s_noise, (
            f"Correlated (1/f-like) series ({s_corr:.3f}) should score above "
            f"white noise ({s_noise:.3f})"
        )

    # ---- baseline seed integration -----------------------------------------

    def test_seeds_return_throttle_series(self):
        """_simulate_lap now returns (lap_time, offsets, speeds, throttles)."""
        inst = generate_instance("oval")
        for name, code in SEED_RACING_PROGRAMS:
            fn = _compile_policy(code)
            result = _simulate_lap(inst, fn)
            assert len(result) == 4, (
                f"_simulate_lap should return 4-tuple for {name}, got {len(result)}"
            )
            lap_time, offsets, speeds, throttles = result
            assert len(throttles) == len(inst.centerline)
            assert all(0.0 <= t <= 1.0 for t in throttles), (
                f"{name}: throttles out of [0,1] range"
            )

    def test_seed_rhythm_scores_in_reasonable_range(self):
        """Seeds are track-responsive (not constant, not pure noise).

        Their rhythm scores should be in [0.40, 0.99] — not forced to extremes.
        """
        for track in ["oval", "chicane", "hairpin", "complex"]:
            inst = generate_instance(track)
            for name, code in SEED_RACING_PROGRAMS:
                fn = _compile_policy(code)
                _, _, _, throttles = _simulate_lap(inst, fn)
                s = _rhythm_score(throttles)
                assert 0.40 <= s <= 0.99, (
                    f"Seed '{name}' on '{track}' rhythm score {s:.3f} outside "
                    f"[0.40, 0.99] — rhythm term is too extreme"
                )

    def test_total_fitness_baselines_still_in_band(self):
        """Adding the rhythm term (weight 0.10) must not push any seed outside
        the [0.70, 0.90] fitness band on any track.
        """
        for track in ["oval", "chicane", "hairpin", "complex"]:
            inst = generate_instance(track)
            for name, code in SEED_RACING_PROGRAMS:
                score = evaluate_on_instance(code, inst)
                assert score is not None, f"{name} returned None on {track}"
                assert 0.70 <= score <= 0.90, (
                    f"Seed '{name}' on track '{track}' scored {score:.4f} after "
                    f"adding rhythm term — outside [0.70, 0.90] target band"
                )
