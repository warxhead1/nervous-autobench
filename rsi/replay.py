"""Counterfactual replay — the operator's "I bet you would have lost here" tool.

Given a captured autobench session (CloudEvents JSONL on the nervous-bus
debug file), reconstruct the harness config that was in flight at a
specific iteration, apply a forced override, re-run the benchmark
cases against the modified harness, and emit a comparison report
showing how many cases flipped verdicts and how the aggregate score
moved.

Public surface: :class:`ReplayLoader`, :class:`CounterfactualRunner`,
:class:`ReplayComparison`, :func:`parse_override`,
:func:`load_cases_from_dir`, :func:`harness_dict_to_config`.

Reconstruction caveats (event-stream fidelity + ``--benchmark-dir``
requirement) are preserved in ``_checkpoints/architecture-history.md``.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..core import ContextManager, HarnessConfig, RolloutProtocol, Verdict


# Channel names we know about — kept inline so the replay module has no hard
# dependency on observability (it might run against a capture from an older
# version of autobench that didn't yet have all four channels).
CHANNEL_PHASE = "autobench.phase.v1"
CHANNEL_ITERATION = "autobench.iteration.v1"
CHANNEL_SANDBOX = "autobench.sandbox.v1"
CHANNEL_IMPROVER = "autobench.improver.v1"


# --------------------------------------------------------------------------- #
# Override parsing
# --------------------------------------------------------------------------- #

def _coerce_scalar(raw: str) -> Any:
    """Coerce a CLI string value into the most natural Python scalar.

    Tries int, float, bool, JSON literal, then falls back to the bare string.
    """
    s = raw.strip()
    lowered = s.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        return float(s)
    except (TypeError, ValueError):
        pass
    # Try JSON for lists / dicts / quoted strings
    if s and s[0] in "[{\"":
        try:
            return json.loads(s)
        except Exception:
            pass
    return s


def parse_override(spec: str) -> dict[str, Any]:
    """Parse a single ``KEY=VAL`` override into a (possibly nested) dict.

    Examples::

        parse_override("context_manager=BUDGETED")
            -> {"context_manager": "BUDGETED"}
        parse_override("budget.max_tokens=4096")
            -> {"budget": {"max_tokens": 4096}}
        parse_override("rollout_protocol=ITERATIVE")
            -> {"rollout_protocol": "ITERATIVE"}
    """
    if "=" not in spec:
        raise ValueError(f"invalid override spec (missing '='): {spec!r}")
    key, _, raw_val = spec.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"invalid override spec (empty key): {spec!r}")
    value = _coerce_scalar(raw_val)

    parts = key.split(".")
    out: dict[str, Any] = {}
    cur = out
    for part in parts[:-1]:
        nxt: dict[str, Any] = {}
        cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return out


def merge_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    """Merge a list of ``KEY=VAL`` overrides into one nested dict."""
    merged: dict[str, Any] = {}
    for spec in overrides:
        single = parse_override(spec)
        _deep_merge(merged, single)
    return merged


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """In-place deep merge of ``src`` into ``dst``."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


# --------------------------------------------------------------------------- #
# Replay loader
# --------------------------------------------------------------------------- #

class ReplayLoader:
    """Index a JSONL bus capture by ``session_id`` and ``iteration``.

    The loader reads every line, ignores anything that isn't a recognised
    autobench channel, and groups events into per-session, per-iteration
    buckets. Events with no iteration tag (phase/sandbox/improver) are
    attached to the most recent iteration-start seen for that session — this
    matches how the RSI loop emits events.
    """

    def __init__(self, debug_file: str | Path) -> None:
        self.path = Path(debug_file)
        # session_id -> iteration -> list[event]
        self._by_session_iter: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # session_id -> list[(iteration, harness_version, improvement_delta)] in event order
        self._iteration_completes: dict[str, list[tuple[int, str, dict[str, Any] | None]]] = (
            defaultdict(list)
        )
        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.path.exists():
            return
        # Track the active iteration per session so we can attach phase /
        # sandbox / improver events to the correct bucket.
        active_iter: dict[str, int] = {}
        with open(self.path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                channel = ev.get("type", "")
                if not channel.startswith("autobench."):
                    continue
                data = ev.get("data") or {}
                sid = data.get("session_id")
                if not sid:
                    continue

                if channel == CHANNEL_ITERATION:
                    iter_num = int(data.get("iteration", 0))
                    active_iter[sid] = iter_num
                    self._by_session_iter[sid][iter_num].append(ev)
                    if data.get("status") == "complete":
                        self._iteration_completes[sid].append(
                            (
                                iter_num,
                                str(data.get("harness_version", "")),
                                data.get("improvement_delta"),
                            )
                        )
                else:
                    # Attach to the most recent iteration we saw for this session.
                    iter_num = active_iter.get(sid)
                    if iter_num is None:
                        # Pre-iteration events (e.g. setup phases) — bucket them
                        # under iteration -1 so they remain queryable without
                        # polluting any real iteration.
                        iter_num = -1
                    self._by_session_iter[sid][iter_num].append(ev)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def sessions(self) -> list[str]:
        """All distinct session IDs in the capture, in lexical order."""
        return sorted(self._by_session_iter.keys())

    def iterations(self, session_id: str) -> list[int]:
        """All iterations seen for ``session_id``, sorted ascending."""
        buckets = self._by_session_iter.get(session_id, {})
        return sorted(i for i in buckets.keys() if i >= 0)

    def events_for(self, session_id: str, iteration: int) -> list[dict[str, Any]]:
        """Every event tied to ``(session_id, iteration)``, in file order."""
        return list(self._by_session_iter.get(session_id, {}).get(iteration, []))

    def case_ids(self, session_id: str, iteration: int) -> list[str]:
        """Distinct case IDs that ran during this iteration (from sandbox events)."""
        seen: list[str] = []
        for ev in self.events_for(session_id, iteration):
            if ev.get("type") != CHANNEL_SANDBOX:
                continue
            cid = (ev.get("data") or {}).get("case_id")
            if cid and cid not in seen:
                seen.append(cid)
        return seen

    def original_verdicts(self, session_id: str, iteration: int) -> dict[str, str]:
        """Map of case_id -> verdict from sandbox-complete events for this iter."""
        out: dict[str, str] = {}
        for ev in self.events_for(session_id, iteration):
            if ev.get("type") != CHANNEL_SANDBOX:
                continue
            data = ev.get("data") or {}
            if data.get("status") != "complete":
                continue
            cid = data.get("case_id")
            verdict = data.get("verdict")
            if cid and verdict:
                out[cid] = verdict
        return out

    def aggregate_score(self, session_id: str, iteration: int) -> float | None:
        """The aggregate_score from this iteration's complete event, if recorded."""
        for ev in self.events_for(session_id, iteration):
            if ev.get("type") != CHANNEL_ITERATION:
                continue
            data = ev.get("data") or {}
            if data.get("status") == "complete":
                v = data.get("aggregate_score")
                if v is not None:
                    return float(v)
        return None

    def harness_at(self, session_id: str, iteration: int) -> dict[str, Any]:
        """Reconstruct the harness config *at the start of* ``iteration``.

        Best effort: walks every iteration-complete event whose iteration is
        less than ``iteration`` and replays its ``improvement_delta`` onto a
        default base config. Returns a plain dict so callers can pass it back
        through ``HarnessConfig`` if they want a strict type, or just inspect /
        diff it freely.

        The reconstruction is intentionally non-strict: if a delta only carries
        a boolean flip flag (``rollout_protocol_changed``) with no new value,
        we leave the field on the base value but tag it with
        ``_unresolved_flips`` so the caller can warn the user.
        """
        base: dict[str, Any] = {
            "system_prompt": "",
            "rollout_protocol": RolloutProtocol.SINGLE.value,
            "context_manager": ContextManager.FULL.value,
            "tool_surface": "",
            "budget": {
                "max_tokens": 8192,
                "max_time_seconds": 30,
                "max_cost_dollars": 0.10,
                "max_memory_mb": 512,
            },
            "_unresolved_flips": [],
        }
        for prev_iter, _hv, delta in self._iteration_completes.get(session_id, []):
            if prev_iter >= iteration:
                break
            if not delta:
                continue
            spd = delta.get("system_prompt_delta") or ""
            if spd:
                base["system_prompt"] = (
                    (base["system_prompt"] + "\n" + spd) if base["system_prompt"] else spd
                )
            if delta.get("rollout_protocol_changed"):
                base["_unresolved_flips"].append(
                    f"iter={prev_iter}:rollout_protocol_changed (new value not in event payload)"
                )
            if delta.get("context_manager_changed"):
                base["_unresolved_flips"].append(
                    f"iter={prev_iter}:context_manager_changed (new value not in event payload)"
                )
            ts = delta.get("tool_surface_delta") or ""
            if ts:
                base["tool_surface"] = (base["tool_surface"] + "\n" + ts) if base["tool_surface"] else ts
            bd = delta.get("budget_delta") or {}
            if isinstance(bd, dict) and bd:
                base["budget"].update(bd)
        return base


# --------------------------------------------------------------------------- #
# Comparison + runner
# --------------------------------------------------------------------------- #

@dataclass
class ReplayComparison:
    """Result of a counterfactual replay.

    ``flipped_cases`` is a list of ``(case_id, original_verdict, replay_verdict)``
    tuples — only cases whose verdict changed between the original capture and
    the replay run.
    """

    session_id: str = ""
    iteration: int = 0
    override: dict[str, Any] = field(default_factory=dict)
    original_harness: dict[str, Any] = field(default_factory=dict)
    replay_harness: dict[str, Any] = field(default_factory=dict)
    original_score: float = 0.0
    replay_score: float = 0.0
    original_verdicts: dict[str, int] = field(default_factory=dict)
    replay_verdicts: dict[str, int] = field(default_factory=dict)
    flipped_cases: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.replay_score - self.original_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "override": self.override,
            "original_harness": self.original_harness,
            "replay_harness": self.replay_harness,
            "original_score": self.original_score,
            "replay_score": self.replay_score,
            "delta": self.delta,
            "original_verdicts": self.original_verdicts,
            "replay_verdicts": self.replay_verdicts,
            "flipped_cases": [
                {"case_id": c, "original": o, "replay": r}
                for (c, o, r) in self.flipped_cases
            ],
        }

    def render_text(self) -> str:
        """Pretty human-readable summary for stdout."""
        lines: list[str] = []
        lines.append(f"autobench replay — session {self.session_id} iteration {self.iteration}")

        def _fmt(h: dict[str, Any]) -> str:
            cm = h.get("context_manager", "?")
            mt = (h.get("budget") or {}).get("max_tokens", "?")
            rp = h.get("rollout_protocol", "?")
            return f"context_manager={cm}, rollout_protocol={rp}, budget.max_tokens={mt}"

        lines.append(f"original harness: {_fmt(self.original_harness)}")
        lines.append(f"replay harness:   {_fmt(self.replay_harness)}  ({len(self.override)} override(s))")
        lines.append("")
        orig_ok = self.original_verdicts.get("OK", 0)
        rep_ok = self.replay_verdicts.get("OK", 0)
        total = sum(self.original_verdicts.values()) or sum(self.replay_verdicts.values())
        lines.append(
            f"original: {orig_ok} / {total} OK, score={self.original_score:.3f}"
        )
        sign = "+" if self.delta >= 0 else ""
        lines.append(
            f"replay:   {rep_ok} / {total} OK, score={self.replay_score:.3f}   (Δ {sign}{self.delta:.3f})"
        )
        lines.append("")
        if self.flipped_cases:
            lines.append(f"flipped cases ({len(self.flipped_cases)}):")
            for cid, orig, rep in self.flipped_cases:
                lines.append(f"  {cid}: {orig:<4} → {rep}")
        else:
            lines.append("flipped cases: (none)")
        lines.append("")
        lines.append("verdict counts (orig → replay):")
        keys = sorted(set(self.original_verdicts) | set(self.replay_verdicts))
        for k in keys:
            o = self.original_verdicts.get(k, 0)
            r = self.replay_verdicts.get(k, 0)
            lines.append(f"  {k:<4} {o:>3} → {r:<3}")
        unresolved = self.original_harness.get("_unresolved_flips") or []
        if unresolved:
            lines.append("")
            lines.append("warnings: harness reconstruction was incomplete:")
            for w in unresolved:
                lines.append(f"  - {w}")
        return "\n".join(lines)


class CounterfactualRunner:
    """Run a benchmark with a forced harness override.

    Takes a ``BenchmarkEvaluator`` (the same one autobench uses for live runs),
    a starting ``HarnessConfig`` (typically reconstructed via
    ``ReplayLoader.harness_at``), an override dict, and the list of benchmark
    cases to replay. Returns a ``ReplayComparison``.
    """

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    def apply_override(
        self,
        harness: HarnessConfig,
        override: dict[str, Any],
    ) -> HarnessConfig:
        """Return a deep-copied harness with ``override`` applied.

        Override semantics:
            * ``context_manager`` / ``rollout_protocol`` — case-insensitive
              enum lookups; accepts either ``"BUDGETED"`` or ``"budgeted"``.
            * ``budget`` — shallow-merged into the existing budget dict.
            * Everything else is set verbatim on the dataclass.
        """
        new = copy.deepcopy(harness)
        for k, v in override.items():
            if k == "context_manager":
                new.context_manager = self._coerce_enum(ContextManager, v)
            elif k == "rollout_protocol":
                new.rollout_protocol = self._coerce_enum(RolloutProtocol, v)
            elif k == "budget" and isinstance(v, dict):
                merged = dict(new.budget)
                merged.update(v)
                new.budget = merged
            elif hasattr(new, k):
                setattr(new, k, v)
            else:
                # Unknown field — stash on the dict-like budget so callers can
                # inspect it via to_dict() if they need to. We deliberately do
                # NOT raise here; the override may target a future field.
                new.budget[f"_replay_extra_{k}"] = v
        return new

    @staticmethod
    def _coerce_enum(enum_cls: Any, raw: Any) -> Any:
        if isinstance(raw, enum_cls):
            return raw
        if isinstance(raw, str):
            # Try as-is, then upper, then lower.
            for candidate in (raw, raw.upper(), raw.lower()):
                try:
                    return enum_cls(candidate)
                except ValueError:
                    continue
            # Try by-name lookup
            try:
                return enum_cls[raw.upper()]
            except (KeyError, AttributeError):
                pass
        raise ValueError(f"cannot coerce {raw!r} into {enum_cls.__name__}")

    def run(
        self,
        original_harness: HarnessConfig,
        override: dict[str, Any],
        cases: list[Any],
        original_verdicts: dict[str, str] | None = None,
        original_score: float | None = None,
    ) -> ReplayComparison:
        """Apply override, run benchmark, return comparison.

        Args:
            original_harness: Harness config as it was at the start of the
                              captured iteration (best-effort reconstruction).
            override:         Nested dict produced by ``merge_overrides``.
            cases:            List of ``BenchmarkCase`` instances to replay.
            original_verdicts: Optional captured ``case_id -> verdict`` map
                              from the original run. Used to compute
                              ``flipped_cases``; if omitted, flipped_cases is
                              computed against the original_score baseline only.
            original_score:    Optional captured aggregate score. Used in the
                              report when the evaluator can't rerun the original.
        """
        replay_harness = self.apply_override(original_harness, override)
        replay_result = self.evaluator.run(replay_harness, cases)

        replay_verdicts_by_case: dict[str, str] = {}
        for r in replay_result.case_results:
            cid = r.metadata.get("case_id") if isinstance(r.metadata, dict) else None
            if cid:
                replay_verdicts_by_case[cid] = (
                    r.verdict.value if isinstance(r.verdict, Verdict) else str(r.verdict)
                )

        flipped: list[tuple[str, str, str]] = []
        for cid, orig_v in (original_verdicts or {}).items():
            rep_v = replay_verdicts_by_case.get(cid)
            if rep_v is None:
                continue
            if rep_v != orig_v:
                flipped.append((cid, orig_v, rep_v))

        # Verdict count maps for the report header
        orig_counts: dict[str, int] = {}
        for v in (original_verdicts or {}).values():
            orig_counts[v] = orig_counts.get(v, 0) + 1
        replay_counts = {k: int(v) for k, v in replay_result.verdict_counts.items()}

        return ReplayComparison(
            override=override,
            original_harness=_harness_to_dict(original_harness),
            replay_harness=_harness_to_dict(replay_harness),
            original_score=float(original_score) if original_score is not None else 0.0,
            replay_score=float(replay_result.aggregate_score),
            original_verdicts=orig_counts,
            replay_verdicts=replay_counts,
            flipped_cases=flipped,
        )


def _harness_to_dict(h: HarnessConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(h, dict):
        return dict(h)
    if hasattr(h, "to_dict"):
        return h.to_dict()
    return dict(h.__dict__)


# --------------------------------------------------------------------------- #
# Benchmark case loading
# --------------------------------------------------------------------------- #

def load_cases_from_dir(benchmark_dir: str | Path) -> list[Any]:
    """Load every ``*.json`` BenchmarkCase from ``benchmark_dir``.

    The directory is expected to follow the layout under
    ``autobench/benchmarks/<name>/`` — flat JSON files, one case per file.
    """
    from ..evaluator import BenchmarkCase  # local import: heavy module

    path = Path(benchmark_dir)
    cases: list[Any] = []
    if not path.exists():
        raise FileNotFoundError(f"benchmark directory not found: {path}")
    for f in sorted(path.glob("*.json")):
        try:
            raw = json.load(open(f))
        except Exception:
            continue
        if not isinstance(raw, dict) or "id" not in raw or "prompt" not in raw:
            continue
        cases.append(
            BenchmarkCase(
                id=raw["id"],
                prompt=raw["prompt"],
                language=raw.get("language", "python"),
                expected_output=raw.get("expected_output", ""),
                constraints=raw.get("constraints", {}),
                starter_code=raw.get("starter_code", ""),
                test_inputs=raw.get("test_inputs", []),
                metadata=raw.get("metadata", {}),
            )
        )
    return cases


def filter_cases_by_id(cases: list[Any], case_ids: list[str]) -> list[Any]:
    """Keep only cases whose ``id`` is in ``case_ids`` (preserves capture order)."""
    if not case_ids:
        return list(cases)
    by_id = {c.id: c for c in cases}
    out: list[Any] = []
    for cid in case_ids:
        if cid in by_id:
            out.append(by_id[cid])
    return out


def harness_dict_to_config(h: dict[str, Any]) -> HarnessConfig:
    """Convert a reconstructed-harness dict back into a HarnessConfig."""
    cfg = HarnessConfig(
        system_prompt=h.get("system_prompt", ""),
        tool_surface=h.get("tool_surface", ""),
    )
    rp = h.get("rollout_protocol")
    if rp:
        try:
            cfg.rollout_protocol = RolloutProtocol(rp)
        except ValueError:
            pass
    cm = h.get("context_manager")
    if cm:
        try:
            cfg.context_manager = ContextManager(cm)
        except ValueError:
            pass
    if isinstance(h.get("budget"), dict):
        cfg.budget = dict(h["budget"])
    return cfg
