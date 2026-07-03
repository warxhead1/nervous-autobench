"""cycle — one budgeted greenhouse evolution cycle.

Each invocation of ``run_cycle`` is meant to be a single ``python -m
greenhouse cycle`` process (see ``greenhouse.timer``): acquire a single-
instance lock, pick the next goal, spend at most ``per_cycle_max_requests``
(clamped to whatever is left in the sliding window) running that goal's
kernel, export validated GLSL candidates up to its ``want``, record the
spend, and emit exactly one ``greenhouse.cycle.completed.v1`` no matter how
the cycle ends — lock contention, an unusable goals manifest, and budget
exhaustion are all reportable outcomes, not silent no-ops.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import bus, export
from .goals import Goal, GoalsManifest, GoalsManifestError, load_manifest
from .ledger import Ledger

DEFAULT_LOCK_PATH = Path.home() / ".cache" / "nervous-bus" / "greenhouse" / "cycle.lock"
DEFAULT_RUNS_ROOT = Path.home() / ".cache" / "nervous-bus" / "greenhouse" / "runs"

# Below this many requests remaining in the window, a cycle isn't worth
# spinning up a kernel run for (island seeding alone costs several requests).
MIN_CYCLE_REQUESTS = 50

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_DRY_RUN_FIXTURES = {"sdf": "dry_run_sdf.json", "terrain": "dry_run_terrain.json", "noise": "dry_run_noise.json"}


@dataclass
class CycleResult:
    skipped: bool
    stop_reason: str
    window_requests_used: int
    window_requests_budget: int
    goal_id: str | None = None
    domain: str | None = None
    run_id: str | None = None
    generations: int | None = None
    best_fitness: float | None = None
    llm_requests: int = 0
    candidates_dropped: int = 0
    drop_paths: list[Path] = field(default_factory=list)
    export_errors: dict[str, list[str]] = field(default_factory=dict)


@contextlib.contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def select_goal(manifest: GoalsManifest, drops_root: Path | None = None) -> Goal | None:
    """Priority-weighted round-robin over goals whose `want` isn't satisfied yet.

    Goals are ranked by (dropped_count / priority): the goal furthest behind
    its priority-weighted share of drops goes next. This lets a want:3
    priority:5 goal get evolved roughly 5x as often as a want:3 priority:1
    goal, while every eligible goal still eventually gets a turn (unlike a
    strict priority-then-FIFO scheme, which would starve low-priority goals
    for as long as any higher-priority goal remains unsatisfied).
    """
    eligible = [g for g in manifest.goals if export.dropped_count(g.id, drops_root) < g.want]
    if not eligible:
        return None

    def _key(g: Goal) -> tuple[float, int, str]:
        dropped = export.dropped_count(g.id, drops_root)
        weight = max(1, g.priority)
        return (dropped / weight, -g.priority, g.id)

    eligible.sort(key=_key)
    return eligible[0]


def _load_fixture_run(domain: str) -> dict | None:
    name = _DRY_RUN_FIXTURES.get(domain)
    if name is None:
        return None
    return json.loads((_FIXTURES_DIR / name).read_text())


def _run_kernel_for_real(goal: Goal, *, max_requests: int, runs_root: Path) -> tuple[str, dict]:
    """Construct and run the domain kernel the same way its cli.py does, return (run_id, results-shaped dict)."""
    # Importing kernels.cli triggers its module-level eager-import of all 8
    # kernel subpackages, which is what populates KERNEL_REGISTRY via each
    # package's @register_kernel side effect on import. Reusing that import
    # instead of duplicating the registration list here.
    import autobench.kernels.cli  # noqa: F401
    from autobench.kernels.base import KERNEL_REGISTRY
    from autobench.kernels.config import KernelConfig

    kernel_cls = KERNEL_REGISTRY.get(goal.domain)
    if kernel_cls is None:
        raise export.UnsupportedDomain(f"no kernel registered for domain '{goal.domain}'")

    output_dir = runs_root / goal.id
    config = KernelConfig(
        instances=list(goal.instances),
        output_dir=output_dir,
        max_requests=max_requests,
        target_fitness=goal.target_fitness,
    )
    kernel = kernel_cls(config)
    programs = kernel.run()
    if hasattr(kernel, "save_results"):
        kernel.save_results(programs)

    results = {
        "stop_reason": kernel.stop_reason,
        "llm_requests": kernel.llm_requests,
        "config": {"generations": kernel.generation},
        "top_programs": [
            {"id": p.id, "fitness": p.fitness, "generation": p.generation, "code": p.code, "source": p.source}
            for p in programs
        ],
    }
    return kernel.run_id, results


def _finish(
    *,
    skipped: bool,
    stop_reason: str,
    window_requests_used: int,
    window_requests_budget: int,
    goal_id: str | None = None,
    domain: str | None = None,
    run_id: str | None = None,
    generations: int | None = None,
    best_fitness: float | None = None,
    llm_requests: int = 0,
    candidates_dropped: int = 0,
    drop_paths: list[Path] | None = None,
    export_errors: dict[str, list[str]] | None = None,
) -> CycleResult:
    """Emit greenhouse.cycle.completed.v1 (the required heartbeat, whatever the outcome) and build the result."""
    data = {
        "goal_id": goal_id or "",
        "domain": domain or "",
        "run_id": run_id or "",
        "stop_reason": stop_reason,
        "llm_requests": llm_requests,
        "candidates_dropped": candidates_dropped,
        "window_requests_used": window_requests_used,
        "window_requests_budget": window_requests_budget,
        "skipped": skipped,
    }
    # generations/best_fitness are optional integer/number fields in the schema —
    # omit them entirely rather than sending `null` when a cycle never ran a kernel.
    if generations is not None:
        data["generations"] = generations
    if best_fitness is not None:
        data["best_fitness"] = best_fitness
    bus.publish("greenhouse.cycle.completed.v1", data)
    return CycleResult(
        skipped=skipped, stop_reason=stop_reason, goal_id=goal_id, domain=domain, run_id=run_id,
        generations=generations, best_fitness=best_fitness, llm_requests=llm_requests,
        candidates_dropped=candidates_dropped, drop_paths=drop_paths or [], export_errors=export_errors or {},
        window_requests_used=window_requests_used, window_requests_budget=window_requests_budget,
    )


def run_cycle(
    *,
    dry_run: bool = False,
    manifest: GoalsManifest | None = None,
    ledger: Ledger | None = None,
    drops_root: Path | None = None,
    runs_root: Path | None = None,
    lock_path: Path | None = None,
    now: float | None = None,
) -> CycleResult:
    lock_path = lock_path or DEFAULT_LOCK_PATH
    ledger = ledger or Ledger()
    now = now if now is not None else time.time()

    with _lock(lock_path) as acquired:
        if not acquired:
            window = manifest.budget.window_seconds if manifest else 18000.0
            budget = manifest.budget.window_max_requests if manifest else 0
            return _finish(skipped=True, stop_reason="lock_held",
                            window_requests_used=ledger.window_used(window, now=now),
                            window_requests_budget=budget)
        return _run_cycle_locked(
            dry_run=dry_run, manifest=manifest, ledger=ledger,
            drops_root=drops_root, runs_root=runs_root, now=now,
        )


def _run_cycle_locked(
    *,
    dry_run: bool,
    manifest: GoalsManifest | None,
    ledger: Ledger,
    drops_root: Path | None,
    runs_root: Path | None,
    now: float,
) -> CycleResult:
    if manifest is None:
        try:
            manifest = load_manifest()
        except GoalsManifestError as e:
            return _finish(skipped=True, stop_reason=f"goals_manifest_error: {e}",
                            window_requests_used=0, window_requests_budget=0)

    goal = select_goal(manifest, drops_root)
    if goal is None:
        used = ledger.window_used(manifest.budget.window_seconds, now=now)
        return _finish(skipped=True, stop_reason="no_eligible_goal",
                        window_requests_used=used, window_requests_budget=manifest.budget.window_max_requests)

    remaining = ledger.remaining(manifest.budget.window_max_requests, manifest.budget.window_seconds, now=now)
    if remaining < MIN_CYCLE_REQUESTS:
        used = ledger.window_used(manifest.budget.window_seconds, now=now)
        return _finish(skipped=True, stop_reason="budget_exhausted", goal_id=goal.id, domain=goal.domain,
                        window_requests_used=used, window_requests_budget=manifest.budget.window_max_requests)

    max_requests = min(manifest.budget.per_cycle_max_requests, remaining)
    runs_root = runs_root or DEFAULT_RUNS_ROOT

    if dry_run:
        fixture = _load_fixture_run(goal.domain)
        if fixture is None:
            used = ledger.window_used(manifest.budget.window_seconds, now=now)
            return _finish(skipped=True, stop_reason=f"dry_run_no_fixture_for_domain:{goal.domain}",
                            goal_id=goal.id, domain=goal.domain,
                            window_requests_used=used, window_requests_budget=manifest.budget.window_max_requests)
        run_id = f"dryrun-{int(now)}"
        stop_reason = fixture["stop_reason"]
        llm_requests = fixture["llm_requests"]
        generations = fixture["config"]["generations"]
        top_programs = fixture["top_programs"]
    else:
        run_id, results = _run_kernel_for_real(goal, max_requests=max_requests, runs_root=runs_root)
        stop_reason = results["stop_reason"]
        llm_requests = results["llm_requests"]
        generations = results["config"]["generations"]
        top_programs = results["top_programs"]

    if llm_requests > 0:
        ledger.record(run_id=run_id, goal_id=goal.id, requests=llm_requests, ts=now)

    already_dropped = export.dropped_count(goal.id, drops_root)
    slots = max(0, goal.want - already_dropped)
    instance = goal.instances[0] if goal.instances else ""

    drop_paths: list[Path] = []
    export_errors: dict[str, list[str]] = {}
    for program in sorted(top_programs, key=lambda p: -p["fitness"])[:slots]:
        try:
            result = export.export_candidate(
                domain=goal.domain, goal_id=goal.id, goal_notes=goal.notes, goal_tags=goal.tags,
                instance=instance, program=program, run_id=run_id, drops_root=drops_root,
            )
        except export.UnsupportedDomain:
            export_errors[program["id"]] = [f"unsupported domain '{goal.domain}' — no GLSL export path wired"]
            continue
        if result.validated and result.drop_path is not None:
            drop_paths.append(result.drop_path)
            bus.publish("greenhouse.candidate.ready.v1", {
                "goal_id": goal.id,
                "domain": goal.domain,
                "run_id": run_id,
                "candidate_id": result.candidate_id,
                "generation": program.get("generation", 0),
                "instance": instance,
                "fitness": float(program["fitness"]),
                "drop_path": str(result.drop_path),
                "glsl_validated": True,
                "glsl_bytes": result.glsl_bytes,
            })
        else:
            export_errors[result.candidate_id] = result.errors

    best_fitness = max((p["fitness"] for p in top_programs), default=None)
    window_used_after = ledger.window_used(manifest.budget.window_seconds, now=now)

    return _finish(
        skipped=False, stop_reason=stop_reason, goal_id=goal.id, domain=goal.domain, run_id=run_id,
        generations=generations, best_fitness=best_fitness, llm_requests=llm_requests,
        candidates_dropped=len(drop_paths), drop_paths=drop_paths, export_errors=export_errors,
        window_requests_used=window_used_after, window_requests_budget=manifest.budget.window_max_requests,
    )
