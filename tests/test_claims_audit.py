"""Tests for autobench.audit.claims_audit offline-batch mode.

Covers the read path claimed to already be zero-live-bus (stream_debug_ledger
/ stream_promotion_ledger never import Redis or open a bus connection — they
only ever read local files) plus the newly explicit ``--offline
--evidence-path`` contract: plain jsonl, gzip-compressed jsonl, a directory of
rotated windows (mixed plain + .gz), and --ci exit-code semantics (0 pass / 1
fail) matching the live path.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from autobench.audit.claims_audit import (
    ClaimsAuditor,
    _build_parser,
    _exit_code_for_results,
    _resolve_ledger_files,
    main,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


CLAIMS_YAML = """\
claims_version: "1.0"
claims:
  - id: test.completeness
    title: "Test completeness claim"
    pass_criteria:
      type: completeness
      event_type: test.evidence.v1
      required_fields: [foo]
"""


def _event(foo=True) -> dict:
    data = {"foo": "bar"} if foo else {"other": 1}
    return {
        "specversion": "1.0",
        "id": "01AAAAAAAAAAAAAAAAAAAAAAAA",
        "source": "/test",
        "type": "test.evidence.v1",
        "time": "2026-07-19T00:00:00.000000Z",
        "data": data,
    }


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _write_jsonl_gz(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _claims_file(tmp_path: Path) -> Path:
    p = tmp_path / "claims.yaml"
    p.write_text(CLAIMS_YAML)
    return p


# --------------------------------------------------------------------------- #
# _resolve_ledger_files
# --------------------------------------------------------------------------- #


def test_resolve_single_plain_file_passthrough(tmp_path):
    f = tmp_path / "debug.jsonl"
    f.write_text("{}\n")
    assert _resolve_ledger_files(f) == [f]


def test_resolve_single_gz_file_passthrough(tmp_path):
    f = tmp_path / "debug.jsonl.gz"
    _write_jsonl_gz(f, [_event()])
    assert _resolve_ledger_files(f) == [f]


def test_resolve_directory_collects_rotated_windows_only(tmp_path):
    d = tmp_path / "archive"
    d.mkdir()
    plain = d / "debug.jsonl"
    rotated_gz = d / "debug.jsonl.1.gz"
    unrelated = d / "notes.txt"
    plain.write_text("{}\n")
    _write_jsonl_gz(rotated_gz, [_event()])
    unrelated.write_text("ignore me")

    resolved = _resolve_ledger_files(d)
    assert set(resolved) == {plain, rotated_gz}


def test_resolve_directory_orders_oldest_first(tmp_path):
    d = tmp_path / "archive"
    d.mkdir()
    older = d / "debug.jsonl.2"
    newer = d / "debug.jsonl.1"
    older.write_text("{}\n")
    newer.write_text("{}\n")
    # Force distinguishable mtimes regardless of write order above.
    import os
    import time

    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    resolved = _resolve_ledger_files(d)
    assert resolved == [older, newer]


# --------------------------------------------------------------------------- #
# ClaimsAuditor offline evaluation — plain / gz / directory all agree
# --------------------------------------------------------------------------- #


def test_offline_plain_jsonl_pass(tmp_path):
    claims_path = _claims_file(tmp_path)
    ledger = tmp_path / "debug.jsonl"
    _write_jsonl(ledger, [_event(foo=True)])

    auditor = ClaimsAuditor(claims_path=claims_path, debug_ledger_path=ledger,
                             promotion_ledger_path=tmp_path / "no_promotion.jsonl")
    results = auditor.evaluate_all()
    assert len(results) == 1
    assert results[0].status == "PASS"
    assert _exit_code_for_results(results) == 0


def test_offline_gz_jsonl_matches_plain(tmp_path):
    claims_path = _claims_file(tmp_path)
    ledger = tmp_path / "debug.jsonl.gz"
    _write_jsonl_gz(ledger, [_event(foo=True)])

    auditor = ClaimsAuditor(claims_path=claims_path, debug_ledger_path=ledger,
                             promotion_ledger_path=tmp_path / "no_promotion.jsonl")
    results = auditor.evaluate_all()
    assert results[0].status == "PASS"
    assert results[0].evidence_matched == 1
    assert results[0].evidence_total == 1


def test_offline_rotated_directory_combines_windows(tmp_path):
    claims_path = _claims_file(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_jsonl(archive / "debug.jsonl", [_event(foo=True)])
    _write_jsonl_gz(archive / "debug.jsonl.1.gz", [_event(foo=True)])

    auditor = ClaimsAuditor(claims_path=claims_path, debug_ledger_path=archive,
                             promotion_ledger_path=tmp_path / "no_promotion.jsonl")
    results = auditor.evaluate_all()
    assert results[0].status == "PASS"
    # Both windows' records were read and matched.
    assert results[0].evidence_matched == 2
    assert results[0].evidence_total == 2


def test_offline_directory_with_one_bad_window_fails(tmp_path):
    claims_path = _claims_file(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_jsonl(archive / "debug.jsonl", [_event(foo=True)])
    _write_jsonl_gz(archive / "debug.jsonl.1.gz", [_event(foo=False)])

    auditor = ClaimsAuditor(claims_path=claims_path, debug_ledger_path=archive,
                             promotion_ledger_path=tmp_path / "no_promotion.jsonl")
    results = auditor.evaluate_all()
    assert results[0].status == "FAIL"
    assert _exit_code_for_results(results) == 1


# --------------------------------------------------------------------------- #
# CLI: --offline --evidence-path, --ci exit codes, incompatible-flag gating
# --------------------------------------------------------------------------- #


def test_cli_offline_pass_exits_zero(tmp_path, monkeypatch, capsys):
    claims_path = _claims_file(tmp_path)
    ledger = tmp_path / "debug.jsonl.gz"
    _write_jsonl_gz(ledger, [_event(foo=True)])

    monkeypatch.setattr("sys.argv", [
        "claims_audit", "--claims", str(claims_path),
        "--offline", "--evidence-path", str(ledger),
        "--promotion-ledger", str(tmp_path / "no_promotion.jsonl"),
        "--ci", "--report-json",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["status"] == "PASS"


def test_cli_offline_fail_exits_one(tmp_path, monkeypatch, capsys):
    claims_path = _claims_file(tmp_path)
    ledger = tmp_path / "debug.jsonl"
    _write_jsonl(ledger, [_event(foo=False)])

    monkeypatch.setattr("sys.argv", [
        "claims_audit", "--claims", str(claims_path),
        "--offline", "--evidence-path", str(ledger),
        "--promotion-ledger", str(tmp_path / "no_promotion.jsonl"),
        "--ci",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_cli_offline_rejects_watch(tmp_path, monkeypatch):
    claims_path = _claims_file(tmp_path)
    ledger = tmp_path / "debug.jsonl"
    _write_jsonl(ledger, [_event(foo=True)])

    monkeypatch.setattr("sys.argv", [
        "claims_audit", "--claims", str(claims_path),
        "--offline", "--evidence-path", str(ledger), "--watch",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2  # argparse.error() exit code


def test_cli_offline_rejects_emit(tmp_path, monkeypatch):
    claims_path = _claims_file(tmp_path)
    ledger = tmp_path / "debug.jsonl"
    _write_jsonl(ledger, [_event(foo=True)])

    monkeypatch.setattr("sys.argv", [
        "claims_audit", "--claims", str(claims_path),
        "--offline", "--evidence-path", str(ledger), "--emit",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_build_parser_has_offline_flags():
    parser = _build_parser()
    args = parser.parse_args(["--offline", "--evidence-path", "/tmp/x.jsonl"])
    assert args.offline is True
    assert args.evidence_path == Path("/tmp/x.jsonl")
