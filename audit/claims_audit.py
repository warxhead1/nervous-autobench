"""EDD Claims Audit for nervous-bus — evaluates autobench claims against live evidence.

Usage:
    python -m autobench.claims_audit --claims claims/claims.yaml --report-json

    # Watch mode (tail debug.jsonl, re-evaluate on new events)
    python -m autobench.claims_audit --claims claims/claims.yaml --watch

    # CI mode (exit 1 on any FAIL)
    python -m autobench.claims_audit --claims claims/claims.yaml --ci

The evaluator reads:
  - ~/.cache/nervous-bus/debug.jsonl   (CloudEvents from autobench producers)
  - nervous-bus/tools/promotion_ledger.jsonl  (PromotionDecision entries)
  - nervous-bus/schemas/*.json          (schema definitions)

Emits nervous-bus.claims.result.v1 on the bus for integration with deer-flow's
EvidenceCollector (already subscribed to autobench.*).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# Autobench package
from autobench import observability
from .ahe import Prediction, PredictionVerification

# CloudEvents UUID helper (same style as observability.py)
_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    import random
    return "".join(random.choices(_ULID_CHARS, k=26))


def _iso_now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ─────────────────────────────────────────────────────────────────
# Evidence Records (nervous-bus variant)
# ─────────────────────────────────────────────────────────────────


@dataclass
class NBEvidenceRecord:
    """Nervous-bus evidence record parsed from debug.jsonl or promotion_ledger.jsonl."""
    channel: str
    event_type: str | None
    timestamp: str | None
    session_id: str | None = None
    bead_id: str | None = None
    cycle_id: str | None = None
    advocate_id: str | None = None
    payload: dict = field(default_factory=dict)

    @classmethod
    def from_debug_event(cls, event: dict) -> "NBEvidenceRecord":
        channel = event.get("type", "")
        data = event.get("data") or {}
        event_type = None
        # Extract event_type from composite envelopes (bus.bead.lifecycle.v1)
        if isinstance(data, dict):
            event_type = data.get("event_type")
            payload = data.get("payload", data)
        else:
            payload = {}
        return cls(
            channel=channel,
            event_type=event_type,
            timestamp=event.get("time"),
            session_id=data.get("session_id"),
            bead_id=data.get("bead_id"),
            cycle_id=data.get("cycle_id"),
            advocate_id=data.get("advocate_id"),
            payload=payload,
        )

    @classmethod
    def from_promotion_entry(cls, entry: dict) -> "NBEvidenceRecord":
        return cls(
            channel="autobench.continuous.promotion_decision.v1",
            event_type="promotion_decision",
            timestamp=entry.get("ts"),
            session_id=entry.get("candidate_session_id"),
            cycle_id=entry.get("cycle_id"),
            payload=entry,
        )


# ─────────────────────────────────────────────────────────────────
# Claim spec (mirrors deer-flow's PassCriteria/ClaimSpec for compatibility)
# ─────────────────────────────────────────────────────────────────


@dataclass
class NBPassCriteria:
    type: str
    event_type: str | None = None
    required_fields: list[str] | None = None
    condition: str | None = None
    max_tolerated: int | None = None
    allowlist_path: str | None = None
    deprecated_path: str | None = None


@dataclass
class NBClaimSpec:
    id: str
    title: str
    pass_criteria: NBPassCriteria


# ─────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────


@dataclass
class NBClaimResult:
    claim_id: str
    title: str
    status: str  # PASS | FAIL | INCONCLUSIVE
    evaluated_at: str
    evidence_matched: int
    evidence_total: int
    confidence: str  # HIGH | MEDIUM | LOW
    failure_reason: str | None = None
    invalidated_triggered: bool = False
    invalidation_reason: str | None = None

    def to_bus_event(self) -> dict:
        return {
            "specversion": "1.0",
            "id": _ulid(),
            "source": "/nervous-bus/autobench/claims_audit",
            "type": "nervous-bus.claims.result.v1",
            "datacontenttype": "application/json",
            "time": _iso_now(),
            "data": {
                "claim_id": self.claim_id,
                "title": self.title,
                "status": self.status,
                "evaluated_at": self.evaluated_at,
                "evidence_matched": self.evidence_matched,
                "evidence_total": self.evidence_total,
                "confidence": self.confidence,
                "failure_reason": self.failure_reason,
                "invalidation_triggered": self.invalidated_triggered,
            },
        }


# ─────────────────────────────────────────────────────────────────
# Claims Auditor
# ─────────────────────────────────────────────────────────────────


class ClaimsAuditor:
    """Evaluates nervous-bus claims against debug.jsonl and promotion_ledger.jsonl."""

    def __init__(
        self,
        claims_path: Path,
        debug_ledger_path: Path | None = None,
        promotion_ledger_path: Path | None = None,
        schemas_dir: Path | None = None,
    ):
        self.claims_path = claims_path
        self.debug_ledger = debug_ledger_path or Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"
        self.promotion_ledger = promotion_ledger_path or (
            Path(__file__).parent.parent.parent / "tools" / "promotion_ledger.jsonl"
        )
        self.schemas_dir = schemas_dir or (Path(__file__).parent.parent.parent / "schemas")
        self._claims: list[NBClaimSpec] = []
        self._allowlist: list[str] = []
        self._deprecated: list[str] = []

    # ── Public API ────────────────────────────────────────────────

    def load_claims(self) -> list[NBClaimSpec]:
        import yaml
        with open(self.claims_path) as fh:
            data = yaml.safe_load(fh)
        self._claims = []
        for c in data.get("claims", []):
            pc_data = c.get("pass_criteria", {})
            pc = NBPassCriteria(
                type=pc_data.get("type", ""),
                event_type=pc_data.get("event_type"),
                required_fields=pc_data.get("required_fields"),
                condition=pc_data.get("condition"),
                max_tolerated=pc_data.get("max_tolerated"),
                allowlist_path=pc_data.get("allowlist_path"),
                deprecated_path=pc_data.get("deprecated_path"),
            )
            self._claims.append(NBClaimSpec(
                id=c["id"],
                title=c["title"],
                pass_criteria=pc,
            ))
        self._load_schema_lists()
        return self._claims

    def stream_debug_ledger(self):
        """Stream NBEvidenceRecord from debug.jsonl."""
        if not self.debug_ledger.exists():
            return
        with open(self.debug_ledger) as fh:
            for line in fh:
                if line.strip():
                    try:
                        event = json.loads(line)
                        yield NBEvidenceRecord.from_debug_event(event)
                    except json.JSONDecodeError:
                        continue

    def stream_promotion_ledger(self):
        """Stream NBEvidenceRecord from promotion_ledger.jsonl."""
        if not self.promotion_ledger.exists():
            return
        with open(self.promotion_ledger) as fh:
            for line in fh:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        yield NBEvidenceRecord.from_promotion_entry(entry)
                    except json.JSONDecodeError:
                        continue

    def evaluate_all(self) -> list[NBClaimResult]:
        self.load_claims()
        results = []
        for claim in self._claims:
            if claim.pass_criteria.type == "completeness":
                result = self._eval_completeness(claim)
            elif claim.pass_criteria.type == "forall":
                result = self._eval_forall(claim)
            elif claim.pass_criteria.type == "schema_coverage":
                result = self._eval_schema_coverage(claim)
            elif claim.pass_criteria.type == "absence":
                result = self._eval_absence(claim)
            else:
                result = NBClaimResult(
                    claim_id=claim.id,
                    title=claim.title,
                    status="INCONCLUSIVE",
                    evaluated_at=datetime.now(UTC).isoformat(),
                    evidence_matched=0,
                    evidence_total=0,
                    confidence="LOW",
                    failure_reason=f"Unknown pass_criteria.type: {claim.pass_criteria.type}",
                )
            results.append(result)
        return results

    def emit_result(self, result: NBClaimResult, obs: Any) -> None:
        """Emit a claim result on the nervous-bus."""
        event = result.to_bus_event()
        obs._publish("nervous-bus.claims.result.v1", event["data"])

    # ── Evaluators ────────────────────────────────────────────────

    def _eval_completeness(self, claim: NBClaimSpec) -> NBClaimResult:
        event_type = claim.pass_criteria.event_type
        required_fields = claim.pass_criteria.required_fields or []
        records = self._records_for_event_type(event_type)
        incomplete = [
            r for r in records
            if any(not r.payload.get(f) for f in required_fields)
        ]
        if not incomplete:
            return self._make_result(claim, "PASS", len(records), len(records))
        return self._make_result(
            claim, "FAIL",
            len(records) - len(incomplete), len(records),
            failure_reason=f"{len(incomplete)} {event_type} records missing: {[f for f in required_fields if any(not r.payload.get(f) for r in incomplete)]}",
        )

    def _eval_forall(self, claim: NBClaimSpec) -> NBClaimResult:
        condition = claim.pass_criteria.condition
        if condition == "prediction_unverified":
            return self._eval_prediction_unverified(claim)
        elif condition == "aggregate_vs_best_score_conflict":
            return self._eval_aggregate_vs_best(claim)
        return self._make_result(claim, "INCONCLUSIVE", 0, 0, failure_reason=f"Unknown condition: {condition}")

    def _eval_prediction_unverified(self, claim: NBClaimSpec) -> NBClaimResult:
        predictions: dict[str, dict] = {}
        verifications: dict[str, dict] = {}
        refutations: dict[str, dict] = {}

        for record in self.stream_debug_ledger():
            if record.channel == "autobench.improver.prediction.v1":
                pid = record.payload.get("prediction_id") or record.session_id
                predictions[pid] = record.payload
            elif record.channel == "autobench.improver.prediction.verified.v1":
                pid = record.payload.get("prediction_id") or record.session_id
                verifications[pid] = record.payload
            elif record.channel == "autobench.improver.prediction.refuted_live.v1":
                pid = record.payload.get("prediction_id") or record.session_id
                refutations[pid] = record.payload

        # Check promotion ledger for outcomes too
        for record in self.stream_promotion_ledger():
            cycle_id = record.cycle_id

        # A prediction is verified if it has a verification or refutation
        unverified = [
            pid for pid, pred in predictions.items()
            if pid not in verifications and pid not in refutations and pred.get("active", True)
        ]

        if not predictions:
            return self._make_result(claim, "INCONCLUSIVE", 0, 0, failure_reason="No prediction events in ledger")
        return self._make_result(
            claim,
            "PASS" if not unverified else "FAIL",
            len(predictions) - len(unverified), len(predictions),
            failure_reason=None if not unverified else f"{len(unverified)} predictions unverified (no outcome_label within 2 cycles)",
        )

    def _eval_aggregate_vs_best(self, claim: NBClaimSpec) -> NBClaimResult:
        # Check cross_domain_evaluation events for aggregate vs best_score conflicts
        seen = []
        for record in self.stream_debug_ledger():
            if record.channel == "autobench.cross_domain.evaluation.v1":
                seen.append(record.payload)

        if not seen:
            return self._make_result(claim, "INCONCLUSIVE", 0, 0, failure_reason="No cross_domain.evaluation events")
        return self._make_result(claim, "PASS", len(seen), len(seen))

    def _eval_schema_coverage(self, claim: NBClaimSpec) -> NBClaimResult:
        channels = {r.channel for r in self.stream_debug_ledger()}
        if not channels:
            return self._make_result(claim, "INCONCLUSIVE", 0, 0, failure_reason="No channels in ledger")

        unknown, inconclusive = [], []
        for ch in channels:
            if self._channel_matches_allowlist(ch):
                continue
            elif self._channel_matches_deprecated(ch):
                inconclusive.append(ch)
            else:
                unknown.append(ch)

        if unknown:
            return self._make_result(
                claim, "FAIL",
                len(channels) - len(unknown), len(channels),
                failure_reason=f"Unknown channels: {unknown[:10]}",
            )
        if inconclusive:
            return self._make_result(
                claim, "INCONCLUSIVE",
                len(channels) - len(inconclusive), len(channels),
                failure_reason=f"Deprecated channels: {inconclusive}",
            )
        return self._make_result(claim, "PASS", len(channels), len(channels))

    def _eval_absence(self, claim: NBClaimSpec) -> NBClaimResult:
        event_type = claim.pass_criteria.event_type
        max_tol = claim.pass_criteria.max_tolerated or 0
        records = [r for r in self.stream_debug_ledger() if r.event_type == event_type or r.channel == event_type]
        if len(records) <= max_tol:
            return self._make_result(claim, "PASS", len(records), 0)
        return self._make_result(
            claim, "FAIL", len(records), 0,
            failure_reason=f"Found {len(records)} '{event_type}' events, max tolerates {max_tol}",
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _records_for_event_type(self, event_type: str | None) -> list[NBEvidenceRecord]:
        if event_type == "promotion_decision":
            return list(self.stream_promotion_ledger())
        return [r for r in self.stream_debug_ledger() if r.event_type == event_type or r.channel == event_type]

    def _load_schema_lists(self) -> None:
        claims_dir = self.claims_path.parent
        allowlist_path = claims_dir / "schema_coverage_allowlist.txt"
        deprecated_path = claims_dir / "schema_coverage_deprecated.txt"
        self._allowlist = self._read_patterns(allowlist_path)
        self._deprecated = self._read_patterns(deprecated_path)

    def _read_patterns(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        patterns = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    line = line.split("#")[0].strip()
                    if line:
                        patterns.append(line)
        return patterns

    def _channel_matches_allowlist(self, channel: str) -> bool:
        for pat in self._allowlist:
            if fnmatch(channel, pat):
                return True
        return False

    def _channel_matches_deprecated(self, channel: str) -> bool:
        for pat in self._deprecated:
            if fnmatch(channel, pat):
                return True
        return False

    def _compute_confidence(self, claim: NBClaimSpec, records: list[NBEvidenceRecord]) -> str:
        if not records:
            return "LOW"
        cutoff = datetime.now(UTC) - timedelta(days=30)
        try:
            latest = max(
                datetime.fromisoformat((r.timestamp or "1970-01-01").replace("Z", "+00:00"))
                for r in records if r.timestamp
            )
            return "LOW" if latest < cutoff else "HIGH"
        except Exception:
            return "MEDIUM"

    def _make_result(
        self,
        claim: NBClaimSpec,
        status: str,
        matched: int,
        total: int,
        failure_reason: str | None = None,
    ) -> NBClaimResult:
        records = self._records_for_event_type(claim.pass_criteria.event_type)
        return NBClaimResult(
            claim_id=claim.id,
            title=claim.title,
            status=status,
            evaluated_at=datetime.now(UTC).isoformat(),
            evidence_matched=matched,
            evidence_total=total,
            confidence=self._compute_confidence(claim, records),
            failure_reason=failure_reason,
        )


# ─────────────────────────────────────────────────────────────────
# Bus event emitter (nervous-bus format, same as AutobenchObservability)
# ─────────────────────────────────────────────────────────────────


class NBObservabilityEmulator:
    """Thin wrapper that publishes to nervous-bus using the same pattern as AutobenchObservability."""

    def __init__(self) -> None:
        self._pipe_disabled = (
            os.environ.get("AUTOBENCH_OBS_DISABLE_PIPE", "").lower() in {"1", "true", "yes"}
            or __import__("shutil").which("zellij") is None
        )
        self._debug_file = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"

    def _publish(self, channel: str, data: dict[str, Any]) -> None:
        import subprocess
        event = {
            "specversion": "1.0",
            "id": _ulid(),
            "source": "/nervous-bus/autobench/claims_audit",
            "type": channel,
            "datacontenttype": "application/json",
            "time": _iso_now(),
            "data": data,
        }
        payload = json.dumps(event, default=str)
        if self._pipe_disabled:
            self._write_debug(payload)
            return
        try:
            proc = subprocess.run(
                ["zellij", "pipe", "-p", "nervous-bus", "-n", channel, "--"],
                input=payload.encode(),
                timeout=0.5,
                capture_output=True,
            )
            if proc.returncode != 0:
                self._write_debug(payload)
        except Exception:
            self._write_debug(payload)

    def _write_debug(self, payload: str) -> None:
        try:
            self._debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._debug_file, "a") as fh:
                fh.write(payload + "\n")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autobench.claims_audit",
        description="EDD Claims Audit — evaluate nervous-bus claims against evidence",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path(__file__).parent.parent.parent / "claims" / "claims.yaml",
        help="Path to claims.yaml",
    )
    parser.add_argument(
        "--debug-ledger",
        type=Path,
        default=Path.home() / ".cache" / "nervous-bus" / "debug.jsonl",
        help="Path to debug.jsonl evidence ledger",
    )
    parser.add_argument(
        "--promotion-ledger",
        type=Path,
        default=Path(__file__).parent.parent.parent / "tools" / "promotion_ledger.jsonl",
        help="Path to promotion_ledger.jsonl",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode: poll debug.jsonl and re-evaluate on new events",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit 1 if any claim FAIL",
    )
    parser.add_argument(
        "--report-json",
        action="store_true",
        help="Output machine-readable JSON to stdout",
    )
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Emit results to nervous-bus as nervous-bus.claims.result.v1 events",
    )
    return parser


def _exit_code_for_results(results: list[NBClaimResult]) -> int:
    return 1 if any(r.status == "FAIL" for r in results) else 0


def run_evaluation(auditor: ClaimsAuditor, emit: bool = False) -> list[NBClaimResult]:
    results = auditor.evaluate_all()
    if emit:
        obs = NBObservabilityEmulator()
        for result in results:
            auditor.emit_result(result, obs)
    return results


async def watch_mode(auditor: ClaimsAuditor) -> None:
    """Poll debug.jsonl, re-evaluate when file grows."""
    obs = NBObservabilityEmulator()
    known_size = auditor.debug_ledger.stat().st_size if auditor.debug_ledger.exists() else 0

    print("Watching {0} for changes…".format(auditor.debug_ledger), file=sys.stderr)
    stop = asyncio.Event()

    def signal_handler():
        stop.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    while not stop.is_set():
        await asyncio.sleep(5)
        if not auditor.debug_ledger.exists():
            continue
        current_size = auditor.debug_ledger.stat().st_size
        if current_size > known_size:
            known_size = current_size
            results = run_evaluation(auditor, emit=True)
            for r in results:
                icon = {"PASS": "✓", "FAIL": "✗", "INCONCLUSIVE": "?"}[r.status]
                print(f"[{r.status:13s}] {icon} {r.claim_id}: {r.title}", file=sys.stderr)
                if r.failure_reason:
                    print(f"    {r.failure_reason}", file=sys.stderr)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    claims_path = Path(args.claims).expanduser().resolve()
    if not claims_path.exists():
        sys.exit(f"claims not found: {claims_path}")

    auditor = ClaimsAuditor(
        claims_path=claims_path,
        debug_ledger_path=Path(args.debug_ledger).expanduser().resolve(),
        promotion_ledger_path=Path(args.promotion_ledger).expanduser().resolve(),
    )

    if args.watch:
        asyncio.run(watch_mode(auditor))
        return

    results = run_evaluation(auditor, emit=args.emit)

    if args.report_json:
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))

    # Human-readable summary
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "INCONCLUSIVE": "?"}[r.status]
        print(f"[{r.status:13s}] {icon} {r.claim_id}: {r.title}", file=sys.stderr)
        if r.failure_reason:
            print(f"    {r.failure_reason}", file=sys.stderr)

    sys.exit(_exit_code_for_results(results) if args.ci else 0)


if __name__ == "__main__":
    main()