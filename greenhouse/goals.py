"""goals — manifest loader + validator for greenhouse evolution goals.

The manifest tells the greenhouse WHAT to evolve toward: a prioritized list of
named goals, each targeting one kernel domain with a ``want`` count of
validated GLSL candidates for Shader Garden. Budget lives alongside the goals
because the two are coupled — goal selection needs to know how much of the
shared request plan it is allowed to spend per cycle.

Path resolution: ``$GREENHOUSE_GOALS`` env var, else
``~/.config/nervous-bus/greenhouse-goals.json``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GOALS_PATH = Path.home() / ".config" / "nervous-bus" / "greenhouse-goals.json"

# Kept in sync with the `domain` enum in
# nervous-bus/schemas/greenhouse.candidate.ready.v1.json.
VALID_DOMAINS = frozenset({
    "sdf", "noise", "terrain", "phase", "latent", "sph", "thermal", "oasis",
})


class GoalsManifestError(ValueError):
    """Raised when a goals manifest is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class Budget:
    window_seconds: float
    window_max_requests: int
    per_cycle_max_requests: int


@dataclass(frozen=True)
class Goal:
    id: str
    domain: str
    instances: list[str]
    priority: int = 1
    want: int = 1
    target_fitness: float | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class GoalsManifest:
    version: int
    budget: Budget
    goals: list[Goal]


def goals_path() -> Path:
    env = os.environ.get("GREENHOUSE_GOALS")
    return Path(env) if env else DEFAULT_GOALS_PATH


def _require(d: dict, key: str, ctx: str) -> object:
    if key not in d:
        raise GoalsManifestError(f"{ctx}: missing required field '{key}'")
    return d[key]


def _parse_budget(raw: dict) -> Budget:
    if not isinstance(raw, dict):
        raise GoalsManifestError("'budget' must be an object")
    window_seconds = _require(raw, "window_seconds", "budget")
    window_max_requests = _require(raw, "window_max_requests", "budget")
    per_cycle_max_requests = _require(raw, "per_cycle_max_requests", "budget")
    try:
        window_seconds = float(window_seconds)
        window_max_requests = int(window_max_requests)
        per_cycle_max_requests = int(per_cycle_max_requests)
    except (TypeError, ValueError) as e:
        raise GoalsManifestError(f"budget: fields must be numeric ({e})") from e
    if window_seconds <= 0:
        raise GoalsManifestError("budget.window_seconds must be > 0")
    if window_max_requests <= 0:
        raise GoalsManifestError("budget.window_max_requests must be > 0")
    if per_cycle_max_requests <= 0:
        raise GoalsManifestError("budget.per_cycle_max_requests must be > 0")
    if per_cycle_max_requests > window_max_requests:
        raise GoalsManifestError(
            "budget.per_cycle_max_requests cannot exceed budget.window_max_requests "
            f"({per_cycle_max_requests} > {window_max_requests})"
        )
    return Budget(window_seconds, window_max_requests, per_cycle_max_requests)


def _parse_goal(raw: dict, seen_ids: set[str]) -> Goal:
    if not isinstance(raw, dict):
        raise GoalsManifestError("each goal must be an object")
    goal_id = _require(raw, "id", "goal")
    if not isinstance(goal_id, str) or not goal_id:
        raise GoalsManifestError("goal.id must be a non-empty string")
    if goal_id in seen_ids:
        raise GoalsManifestError(f"duplicate goal id '{goal_id}'")
    domain = _require(raw, "domain", f"goal '{goal_id}'")
    if domain not in VALID_DOMAINS:
        raise GoalsManifestError(
            f"goal '{goal_id}': unknown domain '{domain}' "
            f"(valid: {', '.join(sorted(VALID_DOMAINS))})"
        )
    instances = raw.get("instances", [])
    if not isinstance(instances, list) or not all(isinstance(i, str) for i in instances):
        raise GoalsManifestError(f"goal '{goal_id}': instances must be a list of strings")
    if not instances:
        raise GoalsManifestError(f"goal '{goal_id}': instances must be non-empty")
    priority = raw.get("priority", 1)
    if not isinstance(priority, int) or priority < 1:
        raise GoalsManifestError(f"goal '{goal_id}': priority must be an int >= 1")
    want = raw.get("want", 1)
    if not isinstance(want, int) or want < 0:
        raise GoalsManifestError(f"goal '{goal_id}': want must be an int >= 0")
    target_fitness = raw.get("target_fitness")
    if target_fitness is not None:
        try:
            target_fitness = float(target_fitness)
        except (TypeError, ValueError) as e:
            raise GoalsManifestError(f"goal '{goal_id}': target_fitness must be numeric ({e})") from e
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise GoalsManifestError(f"goal '{goal_id}': tags must be a list of strings")
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        raise GoalsManifestError(f"goal '{goal_id}': notes must be a string")
    return Goal(
        id=goal_id, domain=domain, instances=list(instances), priority=priority,
        want=want, target_fitness=target_fitness, tags=list(tags), notes=notes,
    )


def parse_manifest(raw: dict) -> GoalsManifest:
    if not isinstance(raw, dict):
        raise GoalsManifestError("manifest root must be an object")
    version = raw.get("version", 1)
    if version != 1:
        raise GoalsManifestError(f"unsupported manifest version {version!r} (only 1 is supported)")
    budget = _parse_budget(_require(raw, "budget", "manifest"))
    goals_raw = _require(raw, "goals", "manifest")
    if not isinstance(goals_raw, list) or not goals_raw:
        raise GoalsManifestError("manifest.goals must be a non-empty list")
    seen: set[str] = set()
    goals: list[Goal] = []
    for g in goals_raw:
        goal = _parse_goal(g, seen)
        seen.add(goal.id)
        goals.append(goal)
    return GoalsManifest(version=version, budget=budget, goals=goals)


def load_manifest(path: Path | None = None) -> GoalsManifest:
    p = path or goals_path()
    if not p.is_file():
        raise GoalsManifestError(
            f"goals manifest not found: {p} "
            "(set GREENHOUSE_GOALS or see greenhouse/goals.example.json)"
        )
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise GoalsManifestError(f"goals manifest is not valid JSON ({p}): {e}") from e
    return parse_manifest(raw)
