"""Racing-kernel oracle — generative-membership scoring for evolved controllers.

GENERATIVE ORACLE DESIGN (avoids the singleton/SSIM trap):
  Score "is this a valid, fast, collision-free racing line?" using three
  independent, physically-grounded terms.  There is NO reference trajectory to
  match — ANY line that is fast + smooth + on-track scores high.

Score decomposition (all in [0,1], weighted sum → fitness):
  1. speed_score     (0.45) — how close is the estimated lap time to the physics
                              ceiling (speed_limit curve)?  Higher → faster lap.
  2. smoothness_score (0.30) — curvature rate-of-change along the driven line.
                               High curvature variation = swerving = poor line.
  3. track_score     (0.25) — fraction of the driven line that stays within the
                               half-width corridor at every sample point.

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

import logging
import math
import re
import textwrap
from typing import Any

from .instance import RacingInstance

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
) -> tuple[float, list[float], list[float]]:
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
        (lap_time_s, offsets, speeds)
        offsets — lateral offset at each centerline sample
        speeds  — speed at each centerline sample
    """
    n = len(instance.centerline)
    offsets: list[float] = []
    speeds: list[float] = []

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
        prev_speed = speed

    return lap_time, offsets, speeds


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
# Top-level evaluation
# ---------------------------------------------------------------------------

# Oracle weights — must sum to 1.0
_W_SPEED = 0.45
_W_SMOOTH = 0.30
_W_TRACK = 0.25


def evaluate_on_instance(code: str, instance: RacingInstance) -> float | None:
    """Evaluate a policy code string on one RacingInstance.

    Returns a fitness score in (0, 1], or None if the code fails to parse/run.

    The score is a weighted combination of:
      speed_score     * 0.45
      smoothness_score * 0.30
      track_score     * 0.25

    This is a GENERATIVE oracle: any valid, fast, smooth line scores high.
    It does NOT compare to a reference lap — multiple structurally different
    policies that are fast and on-track will all score similarly high.
    """
    policy_fn = _compile_policy(code)
    if policy_fn is None:
        return None

    try:
        lap_time, offsets, speeds = _simulate_lap(instance, policy_fn)
    except Exception as exc:
        logger.debug("lap simulation failed: %s", exc)
        return None

    s_speed = _speed_score(lap_time, instance.ref_lap_time_s)
    s_smooth = _smoothness_score(offsets, instance.curvatures)
    s_track = _track_membership_score(offsets, instance.half_width)

    fitness = _W_SPEED * s_speed + _W_SMOOTH * s_smooth + _W_TRACK * s_track

    logger.debug(
        "racing oracle [%s] lap=%.2fs (ref=%.2fs) speed=%.3f smooth=%.3f "
        "track=%.3f → fitness=%.4f",
        instance.name, lap_time, instance.ref_lap_time_s,
        s_speed, s_smooth, s_track, fitness,
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
          speed_score     (×0.45): estimated lap time vs physics ceiling
          smoothness_score (×0.30): penalises jerky offset changes (low 2nd derivative = smooth)
          track_score     (×0.25): fraction of samples within the corridor — stay ON track

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
