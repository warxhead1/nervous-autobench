"""racing_kernel/distill.py — distill an evolved racing controller into a tiny runtime brain.

Takes the best evolved ``racing_line(u, curvature, half_width, speed_limit)``
controller, samples its input→output behaviour over the track state space, and
bakes a compact approximation — either a lookup table (LUT) or a tiny MLP —
that can be hot-loaded into the engine at near-zero VRAM cost.

Representation choice:
  LUT (default) — 2-D grid over (u_normalized, curvature_normalized) →
    (lateral_offset_norm, throttle).  Bilinear interpolation at runtime.
    Footprint: U_BINS × K_BINS × 2 float32 = 32×32×2×4 = 8 192 bytes.
    No matrix multiplications; pure table lookup with two lerps.  Chosen
    because the controller's input space is low-dimensional (2 independent
    inputs at evaluation time: u and curvature; half_width and speed_limit
    are per-track constants that are baked separately) and the oracle is
    purely kinematic — no hidden state.

  MLP (fallback, opt-in via ``kind="mlp"``) — 2-input → 2-output network
    with one hidden layer of 16 units and tanh activation.  Hand-rolled
    in pure Python+struct, no framework dependency.  Footprint: ~800 bytes.

Bus event: emits ``tengine.race.brain.v1`` (status="distilled") on completion
via ``nervous publish --json`` exactly like kernels/base.py's _publish path.

CLI subcommand registered at the END of this file; it appends to
``autobench.kernels`` dispatch via ``register_distill_handler``.

Public surface:
    distill_controller(code, instances, output_path, kind, source_run, fitness)
        → DistillResult

    bake_lut(policy_fn, instances, u_bins, k_bins) → bytes
    bake_mlp(policy_fn, instances, hidden) → bytes

    emit_brain_event(brain_id, status, kind, artifact_uri, footprint_bytes,
                     source_run, fitness, nervous_bin) → bool
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LUT geometry
# ---------------------------------------------------------------------------

# Sampled grid dimensions.  32×32 is small enough (8 KiB) and dense enough
# to track a smooth racing line without visible staircase artefacts.
U_BINS: int = 32    # normalized progress [0,1) around the track
K_BINS: int = 32    # curvature, clamped to CURV_RANGE

# Curvature clamp range.  Racing oracle curvatures are typically in [-0.06, 0.06].
# We widen to ±0.15 to handle hairpins; beyond that the controller saturates anyway.
CURV_RANGE: float = 0.15

# MLP default hidden size
MLP_HIDDEN: int = 16

# -------------------------------------------------------------------------
# Normalization helpers
# -------------------------------------------------------------------------

def _u_to_idx(u: float, bins: int) -> int:
    """Map normalized progress u ∈ [0,1) to grid index."""
    return int(u * bins) % bins


def _k_to_idx(k: float, bins: int, k_range: float = CURV_RANGE) -> int:
    """Map curvature k to grid index, clamped."""
    k_clamped = max(-k_range, min(k_range, k))
    frac = (k_clamped + k_range) / (2 * k_range)
    return int(frac * (bins - 1) + 0.5)


# -------------------------------------------------------------------------
# LUT bake
# -------------------------------------------------------------------------

def bake_lut(
    policy_fn: Callable[[float, float, float, float], tuple[float, float]],
    instances: list[Any],
    u_bins: int = U_BINS,
    k_bins: int = K_BINS,
) -> bytes:
    """Sample policy_fn over a (u, curvature) grid and pack as raw float32.

    The LUT is a (u_bins, k_bins, 2) grid.  Output channel 0 = lateral_offset
    normalized to [-1, 1] (divide by half_width), channel 1 = throttle ∈ [0,1].

    We average half_width and speed_limit over all provided instances to build
    a representative constant for the grid sweep — the controller typically
    saturates on these values anyway.

    Binary layout (little-endian):
        [4 bytes] magic "RLUT"
        [4 bytes] u_bins (uint32)
        [4 bytes] k_bins (uint32)
        [4 bytes] CURV_RANGE (float32)
        [u_bins × k_bins × 2 × 4 bytes] float32 grid, row-major (u outer, k inner)

    Returns raw bytes.
    """
    if not instances:
        raise ValueError("bake_lut requires at least one RacingInstance")

    avg_hw = sum(inst.half_width for inst in instances) / len(instances)
    avg_sl = sum(
        sum(inst.speed_limit) / len(inst.speed_limit)
        for inst in instances
    ) / len(instances)

    grid: list[float] = []
    for ui in range(u_bins):
        u = (ui + 0.5) / u_bins
        for ki in range(k_bins):
            k = -CURV_RANGE + (ki / (k_bins - 1)) * 2 * CURV_RANGE
            try:
                lat, thr = policy_fn(u, k, avg_hw, avg_sl)
            except Exception:
                lat, thr = 0.0, 0.5

            # Clamp outputs to valid ranges
            lat_norm = max(-1.0, min(1.0, lat / max(avg_hw, 1e-6)))
            thr = max(0.0, min(1.0, thr))
            grid.append(lat_norm)
            grid.append(thr)

    header = struct.pack("<4sIIf", b"RLUT", u_bins, k_bins, CURV_RANGE)
    body = struct.pack(f"<{len(grid)}f", *grid)
    return header + body


def query_lut(
    lut_bytes: bytes,
    u: float,
    curvature: float,
    half_width: float,
) -> tuple[float, float]:
    """Bilinear lookup into a packed LUT.  Returns (lateral_offset, throttle)."""
    # Parse header
    magic, u_bins, k_bins, k_range = struct.unpack_from("<4sIIf", lut_bytes, 0)
    if magic != b"RLUT":
        raise ValueError("Not a valid RLUT blob")
    header_size = struct.calcsize("<4sIIf")

    # Grid coordinates
    u_norm = u % 1.0
    k_clamped = max(-k_range, min(k_range, curvature))
    k_frac = (k_clamped + k_range) / (2 * k_range)

    u_f = u_norm * u_bins - 0.5
    k_f = k_frac * (k_bins - 1)

    u0 = int(u_f) % u_bins
    u1 = (u0 + 1) % u_bins
    k0 = max(0, min(k_bins - 2, int(k_f)))
    k1 = k0 + 1

    du = u_f - math.floor(u_f)
    dk = k_f - k0

    def _get(ui: int, ki: int, ch: int) -> float:
        idx = header_size + ((ui * k_bins + ki) * 2 + ch) * 4
        return struct.unpack_from("<f", lut_bytes, idx)[0]

    lat_norm = (
        _get(u0, k0, 0) * (1 - du) * (1 - dk)
        + _get(u1, k0, 0) * du * (1 - dk)
        + _get(u0, k1, 0) * (1 - du) * dk
        + _get(u1, k1, 0) * du * dk
    )
    thr = (
        _get(u0, k0, 1) * (1 - du) * (1 - dk)
        + _get(u1, k0, 1) * du * (1 - dk)
        + _get(u0, k1, 1) * (1 - du) * dk
        + _get(u1, k1, 1) * du * dk
    )
    lat_off = lat_norm * half_width
    return lat_off, max(0.0, min(1.0, thr))


# -------------------------------------------------------------------------
# MLP bake (tiny, hand-rolled, pure struct — no framework)
# -------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _tanh(x: float) -> float:
    return math.tanh(x)


def _fit_mlp(
    X: list[list[float]],
    Y: list[list[float]],
    hidden: int = MLP_HIDDEN,
    lr: float = 0.01,
    epochs: int = 500,
    seed: int = 42,
) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
    """Fit a 2→hidden→2 MLP via mini-batch SGD.  Pure Python.

    Returns (W1, b1, W2, b2) — weight matrices as lists of lists (row-major).
    Input features: [u_norm, curvature_norm].
    Output: [lateral_offset_norm, throttle].
    """
    import random
    rng = random.Random(seed)

    n_in, n_out = 2, 2

    def _randn(scale: float = 0.1) -> float:
        u1 = rng.random() or 1e-12
        u2 = rng.random() or 1e-12
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2) * scale

    # He init for tanh (scale = sqrt(2/n_in))
    scale_in = math.sqrt(2.0 / n_in)
    scale_h = math.sqrt(2.0 / hidden)

    W1 = [[_randn(scale_in) for _ in range(n_in)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    W2 = [[_randn(scale_h) for _ in range(hidden)] for _ in range(n_out)]
    b2 = [0.0] * n_out

    n = len(X)
    for epoch in range(epochs):
        # Decay LR
        cur_lr = lr / (1.0 + epoch * 0.005)

        # Shuffle
        indices = list(range(n))
        rng.shuffle(indices)

        total_loss = 0.0
        for i in indices:
            x = X[i]
            y_target = Y[i]

            # Forward
            h_pre = [sum(W1[j][k] * x[k] for k in range(n_in)) + b1[j] for j in range(hidden)]
            h = [_tanh(v) for v in h_pre]
            out_pre = [sum(W2[j][k] * h[k] for k in range(hidden)) + b2[j] for j in range(n_out)]
            # lat via tanh (bounded [-1,1]), throttle via sigmoid (bounded [0,1])
            out = [_tanh(out_pre[0]), _sigmoid(out_pre[1])]

            # Loss (MSE)
            err = [out[j] - y_target[j] for j in range(n_out)]
            total_loss += sum(e * e for e in err) / n_out

            # Output grad
            d_out = [
                err[0] * (1.0 - out[0] ** 2),    # tanh derivative
                err[1] * out[1] * (1.0 - out[1]), # sigmoid derivative
            ]

            # Hidden grad
            d_h = [
                sum(W2[j][k] * d_out[j] for j in range(n_out)) * (1.0 - h[k] ** 2)
                for k in range(hidden)
            ]

            # Update W2, b2
            for j in range(n_out):
                for k in range(hidden):
                    W2[j][k] -= cur_lr * d_out[j] * h[k]
                b2[j] -= cur_lr * d_out[j]

            # Update W1, b1
            for j in range(hidden):
                for k in range(n_in):
                    W1[j][k] -= cur_lr * d_h[j] * x[k]
                b1[j] -= cur_lr * d_h[j]

        if epoch % 100 == 0:
            logger.debug("MLP train epoch %d loss=%.6f lr=%.5f", epoch, total_loss / n, cur_lr)

    return W1, b1, W2, b2


def bake_mlp(
    policy_fn: Callable[[float, float, float, float], tuple[float, float]],
    instances: list[Any],
    hidden: int = MLP_HIDDEN,
    n_samples: int = 1024,
) -> bytes:
    """Sample policy_fn and fit a 2→hidden→2 MLP.  Returns packed bytes.

    Binary layout (little-endian):
        [4 bytes]  magic "RMLP"
        [4 bytes]  n_in = 2  (uint32)
        [4 bytes]  n_hidden  (uint32)
        [4 bytes]  n_out = 2 (uint32)
        [4 bytes]  CURV_RANGE (float32)
        [n_hidden × n_in × 4 bytes] W1 (float32, row-major)
        [n_hidden × 4 bytes] b1 (float32)
        [n_out × n_hidden × 4 bytes] W2 (float32, row-major)
        [n_out × 4 bytes] b2 (float32)
    """
    if not instances:
        raise ValueError("bake_mlp requires at least one RacingInstance")

    avg_hw = sum(inst.half_width for inst in instances) / len(instances)
    avg_sl = sum(
        sum(inst.speed_limit) / len(inst.speed_limit)
        for inst in instances
    ) / len(instances)

    import random
    rng = random.Random(0)

    X: list[list[float]] = []
    Y: list[list[float]] = []

    for _ in range(n_samples):
        u = rng.random()
        k = rng.uniform(-CURV_RANGE, CURV_RANGE)
        try:
            lat, thr = policy_fn(u, k, avg_hw, avg_sl)
        except Exception:
            lat, thr = 0.0, 0.5
        lat_norm = max(-1.0, min(1.0, lat / max(avg_hw, 1e-6)))
        thr = max(0.0, min(1.0, thr))
        X.append([u, k / CURV_RANGE])
        Y.append([lat_norm, thr])

    W1, b1, W2, b2 = _fit_mlp(X, Y, hidden=hidden)

    n_in, n_out = 2, 2
    header = struct.pack("<4sIIIf", b"RMLP", n_in, hidden, n_out, CURV_RANGE)

    def _pack_matrix(M: list[list[float]]) -> bytes:
        flat = [v for row in M for v in row]
        return struct.pack(f"<{len(flat)}f", *flat)

    def _pack_vec(v: list[float]) -> bytes:
        return struct.pack(f"<{len(v)}f", *v)

    return header + _pack_matrix(W1) + _pack_vec(b1) + _pack_matrix(W2) + _pack_vec(b2)


def query_mlp(
    mlp_bytes: bytes,
    u: float,
    curvature: float,
    half_width: float,
) -> tuple[float, float]:
    """Forward pass through a packed RMLP blob.  Returns (lateral_offset, throttle)."""
    magic, n_in, n_hidden, n_out, k_range = struct.unpack_from("<4sIIIf", mlp_bytes, 0)
    if magic != b"RMLP":
        raise ValueError("Not a valid RMLP blob")
    offset = struct.calcsize("<4sIIIf")

    def _read_matrix(n_rows: int, n_cols: int) -> list[list[float]]:
        nonlocal offset
        n = n_rows * n_cols
        flat = list(struct.unpack_from(f"<{n}f", mlp_bytes, offset))
        offset += n * 4
        return [flat[i * n_cols:(i + 1) * n_cols] for i in range(n_rows)]

    def _read_vec(n: int) -> list[float]:
        nonlocal offset
        v = list(struct.unpack_from(f"<{n}f", mlp_bytes, offset))
        offset += n * 4
        return v

    W1 = _read_matrix(n_hidden, n_in)
    b1 = _read_vec(n_hidden)
    W2 = _read_matrix(n_out, n_hidden)
    b2 = _read_vec(n_out)

    x = [u % 1.0, max(-k_range, min(k_range, curvature)) / k_range]
    h = [_tanh(sum(W1[j][k] * x[k] for k in range(n_in)) + b1[j]) for j in range(n_hidden)]
    out_pre = [sum(W2[j][k] * h[k] for k in range(n_hidden)) + b2[j] for j in range(n_out)]
    lat_norm = _tanh(out_pre[0])
    thr = _sigmoid(out_pre[1])
    return lat_norm * half_width, max(0.0, min(1.0, thr))


# -------------------------------------------------------------------------
# Tolerance check — compare brain approximation to oracle
# -------------------------------------------------------------------------

def _measure_tolerance(
    policy_fn: Callable[[float, float, float, float], tuple[float, float]],
    query_fn: Callable[[float, float, float], tuple[float, float]],
    instances: list[Any],
    n_samples: int = 256,
) -> dict[str, float]:
    """Measure mean absolute error between brain and oracle outputs.

    query_fn signature: (u, curvature, half_width) → (lateral_offset, throttle)

    Returns {'lat_mae': ..., 'thr_mae': ..., 'lat_max': ..., 'thr_max': ...}
    """
    import random
    rng = random.Random(1)

    avg_hw = sum(inst.half_width for inst in instances) / len(instances)
    avg_sl = sum(
        sum(inst.speed_limit) / len(inst.speed_limit)
        for inst in instances
    ) / len(instances)

    lat_errors: list[float] = []
    thr_errors: list[float] = []

    for _ in range(n_samples):
        u = rng.random()
        k = rng.uniform(-CURV_RANGE, CURV_RANGE)
        try:
            lat_true, thr_true = policy_fn(u, k, avg_hw, avg_sl)
        except Exception:
            continue
        try:
            lat_approx, thr_approx = query_fn(u, k, avg_hw)
        except Exception:
            continue
        lat_errors.append(abs(lat_true - lat_approx))
        thr_errors.append(abs(thr_true - thr_approx))

    if not lat_errors:
        return {"lat_mae": float("nan"), "thr_mae": float("nan"),
                "lat_max": float("nan"), "thr_max": float("nan")}

    return {
        "lat_mae": sum(lat_errors) / len(lat_errors),
        "thr_mae": sum(thr_errors) / len(thr_errors),
        "lat_max": max(lat_errors),
        "thr_max": max(thr_errors),
    }


# -------------------------------------------------------------------------
# ULID generator (mirrors kernels/base.py — no cross-import to avoid cycles)
# -------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _new_ulid() -> str:
    value = ((int(time.time() * 1000) & ((1 << 48) - 1)) << 80) | int.from_bytes(
        os.urandom(10), "big"
    )
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


# -------------------------------------------------------------------------
# Bus event emit
# -------------------------------------------------------------------------

def _find_nervous_bin() -> str | None:
    import shutil
    found = shutil.which("nervous")
    if found:
        return found
    repo = Path.home() / "projects" / "nervous-bus" / "sdk" / "shell" / "nervous"
    return str(repo) if repo.is_file() else None


def emit_brain_event(
    brain_id: str,
    status: str,
    kind: str,
    artifact_uri: str,
    footprint_bytes: int,
    source_run: str,
    fitness: float,
    nervous_bin: str | None = None,
    params: dict | None = None,
) -> bool:
    """Emit tengine.race.brain.v1 via nervous publish --json.

    Follows the exact same pattern as kernels/base.py _publish():
      - write envelope to debug.jsonl first (durable)
      - fire-and-forget to nervous CLI (live Redis delivery)
      - NERVOUS_NO_ZELLIJ=1  (no pane fan-out)
      - NERVOUS_DEBUG_LOG=/dev/null (suppress second debug write)
    """
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data: dict[str, Any] = {
        "brain_id": brain_id,
        "status": status,
        "kind": kind,
        "controller_artifact": artifact_uri,
        "footprint_bytes": footprint_bytes,
        "source_run": source_run,
        "fitness": fitness,
        "created_at": created_at,
    }
    if params:
        data["params"] = params

    envelope = {
        "specversion": "1.0",
        "id": uuid.uuid4().urn,
        "source": "/autobench/race",
        "type": "tengine.race.brain.v1",
        "datacontenttype": "application/json",
        "time": created_at,
        "data": data,
    }
    payload = json.dumps(envelope)

    debug_path = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(debug_path, "a") as f:
            f.write(payload + "\n")
    except Exception as e:
        logger.debug("brain event: debug.jsonl write failed: %s", e)

    bin_path = nervous_bin or _find_nervous_bin()
    if bin_path:
        try:
            env = dict(os.environ)
            env["NERVOUS_NO_ZELLIJ"] = "1"
            env["NERVOUS_DEBUG_LOG"] = os.devnull
            proc = subprocess.Popen(
                [bin_path, "publish", "--json"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            proc.communicate(payload.encode(), timeout=3)
        except Exception as e:
            logger.debug("brain event: nervous publish failed: %s", e)

    return True


# -------------------------------------------------------------------------
# DistillResult
# -------------------------------------------------------------------------

@dataclass
class DistillResult:
    """Result of distilling an evolved controller into a brain artifact."""
    brain_id: str
    kind: str                   # "lut" | "mlp"
    artifact_path: str          # absolute path to the baked artifact
    footprint_bytes: int
    fitness: float
    source_run: str
    tolerance: dict             # {lat_mae, thr_mae, lat_max, thr_max}


# -------------------------------------------------------------------------
# Top-level distill_controller
# -------------------------------------------------------------------------

def _compile_policy(code: str) -> Callable | None:
    """Compile a racing_line function from a code string (mirrors oracle._compile_policy)."""
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<racing_line>", "exec"), ns)  # noqa: S102
    except Exception as exc:
        logger.debug("distill: compile failed: %s", exc)
        return None
    fn = ns.get("racing_line")
    if not callable(fn):
        return None
    try:
        result = fn(0.0, 0.0, 5.0, 15.0)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
    except Exception as exc:
        logger.debug("distill: smoke-test failed: %s", exc)
        return None
    return fn


def distill_controller(
    code: str,
    instances: list[Any],
    output_path: str | Path,
    *,
    kind: str = "lut",
    source_run: str = "unknown",
    fitness: float = 0.0,
    nervous_bin: str | None = None,
    emit_event: bool = True,
) -> DistillResult:
    """Distill an evolved controller into a tiny brain artifact.

    Args:
        code:        Python source of the ``racing_line`` function.
        instances:   List of RacingInstance objects (from generate_instance).
        output_path: Path to write the baked artifact (.lut or .mlp file).
        kind:        "lut" (default) or "mlp".
        source_run:  autobench run_id that produced this controller.
        fitness:     Oracle fitness at distill time.
        nervous_bin: Path to nervous CLI (auto-detected if None).
        emit_event:  Whether to emit tengine.race.brain.v1 (default True).

    Returns:
        DistillResult with artifact path, footprint, and tolerance stats.

    Raises:
        ValueError: If the code cannot be compiled or instances is empty.
    """
    if not instances:
        raise ValueError("distill_controller requires at least one RacingInstance")

    policy_fn = _compile_policy(code)
    if policy_fn is None:
        raise ValueError("Failed to compile racing_line from provided code")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    brain_id = _new_ulid()

    if kind == "lut":
        artifact_bytes = bake_lut(policy_fn, instances)
        suffix = ".lut"
        params = {"u_bins": U_BINS, "k_bins": K_BINS, "curv_range": CURV_RANGE,
                  "inputs": ["u_norm", "curvature_norm"],
                  "outputs": ["lateral_offset_norm", "throttle"]}

        def _query(u: float, k: float, hw: float) -> tuple[float, float]:
            return query_lut(artifact_bytes, u, k, hw)

    elif kind == "mlp":
        artifact_bytes = bake_mlp(policy_fn, instances)
        suffix = ".mlp"
        params = {"hidden": MLP_HIDDEN, "activation": "tanh/sigmoid",
                  "inputs": ["u_norm", "curvature_norm"],
                  "outputs": ["lateral_offset_norm", "throttle"]}

        def _query(u: float, k: float, hw: float) -> tuple[float, float]:
            return query_mlp(artifact_bytes, u, k, hw)

    else:
        raise ValueError(f"Unknown kind '{kind}'. Use 'lut' or 'mlp'.")

    # Write artifact
    if not output_path.suffix:
        output_path = output_path.with_suffix(suffix)
    output_path.write_bytes(artifact_bytes)
    footprint = len(artifact_bytes)

    logger.info(
        "distill: baked %s brain → %s (%d bytes)",
        kind.upper(), output_path, footprint,
    )

    # Measure approximation tolerance
    tolerance = _measure_tolerance(policy_fn, _query, instances)
    logger.info(
        "distill: tolerance lat_mae=%.4f thr_mae=%.4f lat_max=%.4f thr_max=%.4f",
        tolerance["lat_mae"], tolerance["thr_mae"],
        tolerance["lat_max"], tolerance["thr_max"],
    )

    artifact_uri = f"file://{output_path.resolve()}"

    if emit_event:
        emit_brain_event(
            brain_id=brain_id,
            status="distilled",
            kind=kind,
            artifact_uri=artifact_uri,
            footprint_bytes=footprint,
            source_run=source_run,
            fitness=fitness,
            nervous_bin=nervous_bin,
            params=params,
        )

    return DistillResult(
        brain_id=brain_id,
        kind=kind,
        artifact_path=str(output_path.resolve()),
        footprint_bytes=footprint,
        fitness=fitness,
        source_run=source_run,
        tolerance=tolerance,
    )


# -------------------------------------------------------------------------
# CLI subcommand — additive registration only (no edits to cli.py)
# -------------------------------------------------------------------------

def _register_cli() -> None:
    """Register the 'distill' subcommand with the kernels CLI if possible.

    Called at module import time.  Safe to call multiple times (idempotent).
    Adds a ``distill`` handler to ``autobench.kernels.cli._EVAL_HANDLERS``
    under the key "racing" so:
        python -m autobench.kernels eval --kernel racing --results-file …
    can also distill the top program.  The dedicated distill path is via:
        python -m autobench.racing_kernel distill …
    (see __main__.py extension or __main__ block below).
    """
    try:
        from autobench.kernels.cli import _EVAL_HANDLERS  # noqa: PLC0415

        if "racing_distill" not in _EVAL_HANDLERS:
            def _eval_handler(args: Any, kernel_cls: Any) -> int:
                return _cli_distill_main(args)

            _EVAL_HANDLERS["racing_distill"] = _eval_handler
    except ImportError:
        pass  # CLI not available — standalone mode, safe to skip


def _cli_distill_main(args: Any = None) -> int:
    """Entry point for ``python -m autobench.racing_kernel distill ...``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Distill an evolved racing controller into a tiny LUT/MLP brain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--code-file", required=False, default=None,
                        help="Path to Python file containing racing_line function. "
                             "If omitted, uses the best seed program.")
    parser.add_argument("--results-file", required=False, default=None,
                        help="Path to kernel results JSON (use top program's code).")
    parser.add_argument("--tracks", default="oval,chicane,hairpin,complex",
                        help="Comma-separated track names to sample over.")
    parser.add_argument("--kind", choices=["lut", "mlp"], default="lut",
                        help="Brain representation: lut (smallest) or mlp.")
    parser.add_argument("--output", required=False, default=None,
                        help="Output artifact path. Defaults to /tmp/brain_<ulid>.<kind>.")
    parser.add_argument("--source-run", default="manual",
                        help="Run ID to record in the brain event.")
    parser.add_argument("--fitness", type=float, default=0.0,
                        help="Fitness score of the source controller.")
    parser.add_argument("--no-event", action="store_true",
                        help="Skip emitting tengine.race.brain.v1.")
    parser.add_argument("-v", "--verbose", action="store_true")

    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve code
    code: str | None = None
    fitness = parsed.fitness

    if parsed.results_file:
        results_path = Path(parsed.results_file)
        if not results_path.exists():
            print(f"Error: results file not found: {results_path}", flush=True)
            return 1
        try:
            with open(results_path) as f:
                results = json.load(f)
            top = results.get("top_programs") or []
            if top:
                code = top[0].get("code", "")
                fitness = float(top[0].get("fitness", fitness))
            elif results.get("best_program"):
                code = results["best_program"].get("code", "")
                fitness = float(results["best_program"].get("fitness", fitness))
        except Exception as e:
            print(f"Error reading results file: {e}", flush=True)
            return 1

    if parsed.code_file:
        code_path = Path(parsed.code_file)
        if not code_path.exists():
            print(f"Error: code file not found: {code_path}", flush=True)
            return 1
        code = code_path.read_text()

    if not code:
        # Default: use the best seed program
        from autobench.racing_kernel.oracle import SEED_RACING_PROGRAMS
        _name, code = SEED_RACING_PROGRAMS[0]
        print(f"[distill] No code provided — using baseline '{_name}'")

    # Load instances
    from autobench.racing_kernel.instance import generate_instance
    track_names = [t.strip() for t in parsed.tracks.split(",") if t.strip()]
    try:
        instances = [generate_instance(t) for t in track_names]
    except ValueError as e:
        print(f"Error loading tracks: {e}", flush=True)
        return 1

    # Output path
    output = parsed.output
    if output is None:
        output = f"/tmp/brain_{_new_ulid()}.{parsed.kind}"

    try:
        result = distill_controller(
            code=code,
            instances=instances,
            output_path=output,
            kind=parsed.kind,
            source_run=parsed.source_run,
            fitness=fitness,
            emit_event=not parsed.no_event,
        )
    except Exception as e:
        print(f"Distillation failed: {e}", flush=True)
        return 1

    print(f"\n[distill] Brain distilled successfully")
    print(f"  brain_id     : {result.brain_id}")
    print(f"  kind         : {result.kind}")
    print(f"  artifact     : {result.artifact_path}")
    print(f"  footprint    : {result.footprint_bytes} bytes")
    print(f"  fitness      : {result.fitness:.6f}")
    print(f"  lat_mae      : {result.tolerance.get('lat_mae', float('nan')):.4f} world-units")
    print(f"  thr_mae      : {result.tolerance.get('thr_mae', float('nan')):.4f}")
    print(f"  lat_max_err  : {result.tolerance.get('lat_max', float('nan')):.4f} world-units")
    if not parsed.no_event:
        print(f"  event        : tengine.race.brain.v1 (distilled) emitted")
    return 0


# Register CLI handler on import (additive, idempotent)
_register_cli()


if __name__ == "__main__":
    import sys
    sys.exit(_cli_distill_main())
