"""Racing-kernel oracle — generative-membership scoring for evolved controllers.

GENERATIVE ORACLE DESIGN (avoids the singleton/SSIM trap):
  Score "is this a valid, fast, collision-free racing line?" using four
  independent, physically-grounded terms.  There is NO reference trajectory to
  match — ANY line that is fast + smooth + on-track + rhythmically structured
  scores high.

Score decomposition (all in [0,1], weighted sum → fitness):
  1. speed_score     (0.40) — how close is the estimated lap time to the physics
                              ceiling (speed_limit curve)?  Higher → faster lap.
  2. smoothness_score (0.27) — curvature rate-of-change along the driven line.
                               High curvature variation = swerving = poor line.
  3. track_score     (0.23) — fraction of the driven line that stays within the
                               half-width corridor at every sample point.
  4. rhythm_score    (0.10) — spectral cadence of the throttle control series.
                               Rewards 1/f-structured modulation (human-like
                               corner-brake / straight-throttle cadence) over
                               constant output or white-noise randomness.
                               Calibrated from spectral slope + low-band energy
                               ratio; no human episode artifacts required.

Seed programs are parametric Python functions evaluated in-process (no C++
compilation); the LLM mutates the Python function body.

A "seed program" is a Python string containing:
  def racing_line(u, curvature, half_width, speed_limit):
      ...
      return (lateral_offset, throttle)
where:
  u             — normalized progress in [0,1] around the track
  curvature     — signed curvature at this point
  half_width    — corridor half-width
  speed_limit   — max speed at this point (curvature-derived)
  → lateral_offset in [-half_width, half_width] (outward from centerline)
  → throttle    in [0, 1]

The oracle simulates a lap using a kinematic point-mass model to turn the
(offset, throttle) policy into a lap time, then scores it.

Public surface:
  SEED_RACING_PROGRAMS      — list of (name, code) for baseline policies
  evaluate_on_instance      — code str × RacingInstance → float|None
  build_llm_prompt          — island × top_programs × generation → str
  parse_llm_response        — LLM str → code str | ''
"""

from __future__ import annotations

import cmath
import logging
import math
import re
import textwrap
from typing import Any

from autobench.racing_kernel.instance import RacingInstance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed programs — parametric baseline policies
# ---------------------------------------------------------------------------

# Each seed is a Python function that the oracle executes.
# LLM is asked to improve / mutate it.

_SEED_PURE_PURSUIT = """\
def racing_line(u, curvature, half_width, speed_limit):
    \"\"\"Pure-pursuit: apex at the inside of corners, full throttle on straights.\"\"\"
    # Positive curvature = left-hand bend: move toward the inside (negative offset)
    k_scale = 0.6
    lateral_offset = -curvature * half_width * k_scale
    lateral_offset = max(-half_width * 0.85, min(half_width * 0.85, lateral_offset))
    # Throttle: proportional to available speed headroom
    throttle = min(1.0, speed_limit / 22.0)
    return lateral_offset, throttle
"""

_SEED_COST_BASED = """\
def racing_line(u, curvature, half_width, speed_limit):
    \"\"\"Cost-based: smooth geometric line — proportional curvature response.\"\"\"
    import math
    # Smooth proportional response: larger curvature → larger offset toward inside
    # Use tanh for a smooth, bounded response (avoids the jerkiness of phase-switching)
    lateral_offset = -math.tanh(curvature * 3.5) * half_width * 0.65
    lateral_offset = max(-half_width * 0.80, min(half_width * 0.80, lateral_offset))
    # Throttle: moderate blend — back off slightly in tight corners
    throttle = max(0.45, 1.0 - abs(curvature) * 2.5)
    throttle = min(throttle, speed_limit / 19.0)
    return lateral_offset, throttle
"""

_SEED_TANGENT_SMOOTHING = """\
def racing_line(u, curvature, half_width, speed_limit):
    \"\"\"Tangent-smoothing: lookahead-weighted offset to reduce curvature variation.\"\"\"
    import math
    # Smooth sinusoidal response to curvature
    # Phase angle encodes position around track
    angle = u * 2 * math.pi
    # Damped curvature response
    lateral_offset = -math.tanh(curvature * 4.0) * half_width * 0.7
    # Blend toward center on fast sections
    fast_blend = max(0.0, (speed_limit - 15.0) / 7.0)
    lateral_offset = lateral_offset * (1.0 - fast_blend * 0.4)
    lateral_offset = max(-half_width * 0.85, min(half_width * 0.85, lateral_offset))
    # Graduated throttle: back off in high-curvature zones
    throttle = max(0.3, 1.0 - abs(curvature) * 3.0)
    throttle = min(throttle, speed_limit / 20.0)
    return lateral_offset, throttle
"""

SEED_RACING_PROGRAMS: list[tuple[str, str]] = [
    ("pure_pursuit", _SEED_PURE_PURSUIT),
    ("cost_based", _SEED_COST_BASED),
    ("tangent_smoothing", _SEED_TANGENT_SMOOTHING),
]


# ---------------------------------------------------------------------------
# Kinematic lap simulation
# ---------------------------------------------------------------------------

def _simulate_lap(
    instance: RacingInstance,
    policy_fn: Any,
    dt: float = 0.05,
) -> tuple[float, list[float], list[float], list[float]]:
    """Simulate a lap with the given policy function.

    Uses a kinematic point-mass model:
      - Speed is clamped to the speed_limit at each point.
      - Lateral position follows the policy offset.
      - Time is accumulated via distance / speed.

    Args:
        instance:   RacingInstance with track geometry.
        policy_fn:  Callable (u, curvature, half_width, speed_limit) →
                    (lateral_offset, throttle).
        dt:         Simulation time step (seconds) — unused in kinematic mode,
                    kept for API stability.

    Returns:
        (lap_time_s, offsets, speeds, throttles)
        offsets   — lateral offset at each centerline sample
        speeds    — speed at each centerline sample
        throttles — raw throttle command emitted by the policy at each sample
    """
    n = len(instance.centerline)
    offsets: list[float] = []
    speeds: list[float] = []
    throttles: list[float] = []

    prev_speed = instance.max_speed * 0.5
    lap_time = 0.0

    for i in range(n):
        u = instance.arc_lengths[i] / instance.lap_length
        curvature = instance.curvatures[i]
        half_w = instance.half_width
        sp_limit = instance.speed_limit[i]

        try:
            lat_off, throttle = policy_fn(u, curvature, half_w, sp_limit)
        except Exception:
            lat_off, throttle = 0.0, 0.5

        # Kinematic speed: approach speed_limit * throttle smoothly
        target_speed = sp_limit * max(0.0, min(1.0, throttle))
        speed = min(sp_limit, prev_speed * 0.6 + target_speed * 0.4)
        speed = max(0.5, speed)  # avoid division by zero

        # Segment length to next point
        next_i = (i + 1) % n
        dx = instance.centerline[next_i][0] - instance.centerline[i][0]
        dy = instance.centerline[next_i][1] - instance.centerline[i][1]
        seg_len = math.hypot(dx, dy)
        lap_time += seg_len / speed

        offsets.append(float(lat_off))
        speeds.append(float(speed))
        throttles.append(float(max(0.0, min(1.0, throttle))))
        prev_speed = speed

    return lap_time, offsets, speeds, throttles


# ---------------------------------------------------------------------------
# Oracle scoring terms
# ---------------------------------------------------------------------------

def _speed_score(lap_time_s: float, ref_lap_time_s: float) -> float:
    """Score based on lap time ratio (faster → higher score).

    ref_lap_time_s is the PHYSICS CEILING — the minimum achievable lap time
    if the controller ran at the curvature-derived speed limit everywhere.
    Seeds typically run 5–20% slower than this ceiling (lap_time > ref_lap_time).

    ratio = ref_lap_time / lap_time  ∈ (0, 1] for real controllers:
      ratio = 1.0  → perfectly at the physics ceiling
      ratio = 0.9  → 10% slower than optimal (typical good seed)
      ratio = 0.8  → 20% slower (weak seed)
      ratio = 0.7  → 30% slower (very slow)

    Calibrated so that:
      ratio = 0.90 → score ~0.75  (typical seed target band)
      ratio = 0.95 → score ~0.85  (above seed band — evolved)
      ratio = 1.00 → score ~0.95  (at physics ceiling)
      ratio = 0.75 → score ~0.50  (poor policy)

    This places seeds with ~10% overhead in the 0.70-0.80 range on speed_score,
    which combines with smoothness (~0.95) and track (1.0) to give total ~0.78-0.87.
    """
    ratio = ref_lap_time_s / max(lap_time_s, 1e-3)
    # Exponential scoring: exp(-k * (1 - ratio)) where k controls the decay rate.
    # ratio=1.00 → score=1.00 (at physics ceiling)
    # ratio=0.97 → score~0.76  (3% slower — typical smooth-track seed: oval/cost_based)
    # ratio=0.95 → score~0.64  (5% slower — typical seed on technical track)
    # ratio=0.92 → score~0.51  (8% slower — seed on complex track)
    # ratio=0.85 → score~0.26  (15% slower — weak policy)
    # k=9 empirically places ALL three seed programs on ALL four tracks in [0.70, 0.90]:
    # speed alone lands at 0.51–0.76; smoothness ~0.95 and track 1.0 bring total to 0.73–0.89.
    k = 9.0
    ratio = max(0.0, min(1.1, ratio))
    score = math.exp(-k * max(0.0, 1.0 - ratio))
    return max(0.0, min(1.0, score))


def _smoothness_score(offsets: list[float], curvatures: list[float]) -> float:
    """Score based on steering smoothness — penalise jerky offset changes.

    A smooth racing line has slowly varying lateral position.  We measure the
    RMS of second-difference of offsets (discrete acceleration of the line),
    normalized by the half-width, then invert via an exponential.
    """
    n = len(offsets)
    if n < 3:
        return 1.0
    # Second differences: proxy for steering rate-of-change
    accel: list[float] = []
    for i in range(1, n - 1):
        d2 = offsets[i + 1] - 2 * offsets[i] + offsets[i - 1]
        accel.append(d2)
    rms = math.sqrt(sum(a * a for a in accel) / len(accel))
    # Normalize: rms=0 → score=1.0; rms=2.0 → score≈0.1
    return math.exp(-rms * 1.5)


def _track_membership_score(
    offsets: list[float],
    half_width: float,
) -> float:
    """Fraction of samples within the drivable corridor.

    This is the GENERATIVE membership term: any offset in [-hw, hw] is valid.
    The oracle does NOT compare to a specific reference trajectory — it only
    checks corridor membership.  An evolved line that explores the full width
    for a better geometric line still scores 1.0 here.
    """
    if not offsets:
        return 0.0
    in_bounds = sum(1 for o in offsets if abs(o) <= half_width)
    return in_bounds / len(offsets)


# ---------------------------------------------------------------------------
# Rhythm / cadence scoring — spectral sufficient-statistic
# ---------------------------------------------------------------------------

# Default calibration — target band for human-like 1/f control cadence.
# Calibrated conceptually: a human driver modulates throttle with a
# structured, corner-locked rhythm (brake-in, coast, throttle-out per corner),
# producing power spectra with negative slope (1/f^beta, beta ~1–2) and
# meaningful low-band energy (the corner period sits in low spatial frequencies).
#
# No real episode artifacts are required: these defaults were chosen so that
# track-responsive seeds (which already exhibit structured cadence) score in
# the 0.55–0.95 band, while pathological controllers (constant output or white
# noise) score significantly lower.
#
# Note: series are sampled per-centerline-point (uniform arc-length, NOT time).
# "Cadence" here is spatial-frequency cadence along the lap, not temporal.
# A purely DC throttle (constant) has no spectral structure and scores ~0.5
# (neutral); it is not penalised to zero because the track-membership and
# speed terms already handle degenerate throttle policies.
_RHYTHM_SLOPE_TARGET: float = -1.5   # 1/f^1.5 — typical structured cadence
_RHYTHM_SLOPE_SIGMA: float = 1.5     # wide tolerance; seeds span -0.5 to -2.1
_RHYTHM_LR_HALFLIFE: float = 0.20    # exponential rise constant for low-ratio
_RHYTHM_N_LOW: int = 5               # DFT bins 1–5 count as "low-band"
_RHYTHM_MAX_K: int = 40              # max DFT bin used for slope estimation


def _rhythm_score(throttles: list[float]) -> float:
    """Spectral cadence score for the throttle control series.

    Extracts two sufficient statistics of the DFT power spectrum of the
    mean-centred throttle series:

      1. Spectral slope  — negative log-log slope (beta); rewards 1/f
         structure (human-like acceleration/deceleration rhythm) over flat
         white noise (slope ~ 0) or pure DC (undefined → neutral 0.5).

      2. Low-band ratio  — fraction of AC power in the lowest N_LOW DFT bins;
         rewards controllers that concentrate energy in a few dominant cadence
         frequencies (corner period) rather than spreading it uniformly.

    The two sub-scores are combined as a weighted average (0.5 / 0.5).

    Returns a value in [0, 1].  Constant series → 0.5 (neutral, not zero).
    """
    N = len(throttles)
    if N < 8:
        return 0.5  # too short to measure cadence

    max_k = min(_RHYTHM_MAX_K, N // 2)
    if max_k < 4:
        return 0.5

    # Remove DC; guard against constant series
    mean = sum(throttles) / N
    centered = [x - mean for x in throttles]
    rms = math.sqrt(sum(x * x for x in centered) / N)
    if rms < 1e-8:
        return 0.5  # constant series: neutral cadence score

    # DFT power spectrum (bins 0 .. max_k-1)
    fft_power: list[float] = []
    for k in range(max_k):
        s = sum(
            centered[j] * cmath.exp(-2j * math.pi * k * j / N)
            for j in range(N)
        )
        fft_power.append(abs(s) ** 2)

    # 1. Spectral slope: OLS fit to log P vs log k for k = 1 … max_k-1
    lf = [math.log(k) for k in range(1, max_k)]
    lp = [math.log(fft_power[k] + 1e-12) for k in range(1, max_k)]
    n2 = len(lf)
    sx = sum(lf)
    sy = sum(lp)
    sxx = sum(x * x for x in lf)
    sxy = sum(x * y for x, y in zip(lf, lp))
    denom = n2 * sxx - sx * sx
    slope = (n2 * sxy - sx * sy) / (denom + 1e-12) if abs(denom) > 1e-12 else 0.0

    slope_score = math.exp(
        -0.5 * ((slope - _RHYTHM_SLOPE_TARGET) / _RHYTHM_SLOPE_SIGMA) ** 2
    )

    # 2. Low-band energy ratio: bins 1 .. N_LOW vs 1 .. max_k-1
    band_low = sum(fft_power[1: _RHYTHM_N_LOW + 1])
    total_ac = sum(fft_power[1:max_k]) + 1e-12
    low_ratio = band_low / total_ac
    lr_score = 1.0 - math.exp(-low_ratio / _RHYTHM_LR_HALFLIFE)

    return max(0.0, min(1.0, 0.5 * slope_score + 0.5 * lr_score))


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

# Oracle weights — must sum to 1.0
# Re-balanced from 3-term (0.45 / 0.30 / 0.25) to make room for rhythm (0.10):
#   speed 0.45 → 0.40  (−0.05)
#   smooth 0.30 → 0.27 (−0.03)
#   track  0.25 → 0.23 (−0.02)
#   rhythm  new → 0.10
# Rationale: rhythm weight is small (10%) so baselines stay in [0.70, 0.90].
_W_SPEED = 0.40
_W_SMOOTH = 0.27
_W_TRACK = 0.23
_W_RHYTHM = 0.10


def evaluate_on_instance(code: str, instance: RacingInstance) -> float | None:
    """Evaluate a policy code string on one RacingInstance.

    Returns a fitness score in (0, 1], or None if the code fails to parse/run.

    The score is a weighted combination of:
      speed_score     * 0.40
      smoothness_score * 0.27
      track_score     * 0.23
      rhythm_score    * 0.10

    This is a GENERATIVE oracle: any valid, fast, smooth, rhythmically
    structured line scores high.  It does NOT compare to a reference lap —
    multiple structurally different policies that are fast and on-track will
    all score similarly high.
    """
    policy_fn = _compile_policy(code)
    if policy_fn is None:
        return None

    try:
        lap_time, offsets, speeds, throttles = _simulate_lap(instance, policy_fn)
    except Exception as exc:
        logger.debug("lap simulation failed: %s", exc)
        return None

    s_speed = _speed_score(lap_time, instance.ref_lap_time_s)
    s_smooth = _smoothness_score(offsets, instance.curvatures)
    s_track = _track_membership_score(offsets, instance.half_width)
    s_rhythm = _rhythm_score(throttles)

    fitness = (
        _W_SPEED * s_speed
        + _W_SMOOTH * s_smooth
        + _W_TRACK * s_track
        + _W_RHYTHM * s_rhythm
    )

    logger.debug(
        "racing oracle [%s] lap=%.2fs (ref=%.2fs) speed=%.3f smooth=%.3f "
        "track=%.3f rhythm=%.3f → fitness=%.4f",
        instance.name, lap_time, instance.ref_lap_time_s,
        s_speed, s_smooth, s_track, s_rhythm, fitness,
    )
    return max(0.0, min(1.0, fitness))


def _compile_policy(code: str) -> Any | None:
    """Compile and return the ``racing_line`` function from a code string."""
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<racing_line>", "exec"), ns)  # noqa: S102
    except Exception as exc:
        logger.debug("racing_line compile failed: %s", exc)
        return None
    fn = ns.get("racing_line")
    if not callable(fn):
        return None
    # Smoke-test: call with dummy args
    try:
        result = fn(0.0, 0.0, 5.0, 15.0)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
    except Exception as exc:
        logger.debug("racing_line smoke-test failed: %s", exc)
        return None
    return fn


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

ISLAND_PERSONAS: list[str] = [
    "You are an expert in geometric racing lines — focus on the apex-out-in-out technique.",
    "You are an expert in predictive control — focus on anticipating corner exits to maximize exit speed.",
    "You are an expert in smooth driving — minimize steering variation for stability and tire wear.",
    "You are an expert in aggressive cornering — push the lateral limits while staying on track.",
]

PROMPT_SKETCHES: list[str] = [
    "Consider modifying how lateral_offset responds to the sign and magnitude of curvature.",
    "Experiment with a lookahead: compute the offset not just from current curvature but from the trend.",
    "Try a two-phase approach: different strategies for corners vs straights based on speed_limit.",
    "Explore using trigonometric functions of `u` to add periodic structure to the line.",
]


def build_llm_prompt(
    island: Any,
    top_programs: list[Any],
    generation: int,
    instance_name: str = "",
    hint: str = "",
) -> str:
    """Build LLM prompt for the racing-line evolution."""
    persona = ISLAND_PERSONAS[island.id % len(ISLAND_PERSONAS)]
    sketch = PROMPT_SKETCHES[island.id % len(PROMPT_SKETCHES)]

    exemplar_block = ""
    if top_programs:
        parts = []
        for prog in top_programs[:2]:
            parts.append(
                f"# Fitness: {prog.fitness:.4f}\n"
                f"```python\n{prog.code.strip()}\n```"
            )
        exemplar_block = "\n\n".join(parts)
    else:
        exemplar_block = "(no exemplars yet — start from scratch)"

    track_note = f"Track: {instance_name}." if instance_name else ""
    hint_block = f"\nStrategic hint: {hint}\n" if hint else ""

    return textwrap.dedent(f"""\
        {persona}

        Your task: write a Python function `racing_line(u, curvature, half_width, speed_limit)`
        that returns `(lateral_offset, throttle)` for a racing car.

        Arguments:
          u              — normalized progress in [0,1] around the track
          curvature      — signed curvature at this point (rad/m); positive = left turn
          half_width     — corridor half-width in world units; offset must stay in [-half_width, half_width]
          speed_limit    — max speed at this point (m/s); throttle in [0,1] scales actual speed

        Scoring objective (higher is better, all in [0,1]):
          speed_score     (×0.40): estimated lap time vs physics ceiling
          smoothness_score (×0.27): penalises jerky offset changes (low 2nd derivative = smooth)
          track_score     (×0.23): fraction of samples within the corridor — stay ON track
          rhythm_score    (×0.10): spectral cadence of the throttle series — reward structured
                                   corner-brake/straight-throttle rhythm (1/f-like spectrum)
                                   over constant or erratic throttle profiles

        {track_note}{hint_block}

        Top current programs (generation {generation}):
        {exemplar_block}

        Hint for this island: {sketch}

        Write ONE improved `racing_line` function in a ```python code block. Return ONLY the code block.
        Keep it under 25 lines. Do NOT import anything except `math`.
    """)


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------

_PY_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_DEF_RE = re.compile(r"def\s+racing_line\s*\(")


def parse_llm_response(response: str) -> str:
    """Extract the `racing_line` function from an LLM response.

    Returns the code string if a valid function definition is found, else ''.
    """
    # 1. Try fenced code block
    for m in _PY_BLOCK_RE.finditer(response):
        code = m.group(1)
        if _DEF_RE.search(code):
            return code.strip()

    # 2. Scan for bare function definition (no fences)
    lines = response.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _DEF_RE.search(line):
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()

    return ""
