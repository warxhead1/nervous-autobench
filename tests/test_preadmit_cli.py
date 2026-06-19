"""Tests for the `preadmit` CLI wire into the two-tier admission loop.

These drive :func:`autobench.cli.cmd_preadmit` directly with an
``argparse.Namespace`` and a monkeypatched tier-1 gate, so the tests are
GPU-free and deterministic. They cover the admission-record mode that fuses the
tier-1 ``PreAdmitResult`` with a tier-2 shadow-dispatch verdict and forwards the
honest exit code — the wire that turns the assembler from dead code into a live
producer. The emit path is exercised with ``--emit`` only via a monkeypatched
emitter (no bus required).
"""

import argparse
import json

import pytest

from autobench import cli
from autobench.engines.preadmit import PreAdmitResult


def _ns(**overrides):
    base = dict(
        shader="-",
        language="glsl",
        kind="fragment",
        work_type=None,
        source_run_id=None,
        shader_id=None,
        shadow=None,
        require_full_coverage=False,
        emit=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def fake_gate(monkeypatch):
    """Monkeypatch the tier-1 gate to a caller-chosen PreAdmitResult."""

    def _install(result):
        monkeypatch.setattr(
            "autobench.engines.preadmit.pre_admit",
            lambda *a, **k: result,
        )

    return _install


def _partial_shadow(path):
    path.write_text(
        json.dumps(
            {
                "candidate": 644,
                "subset_context": [643, 645],
                "decision": "Admit",
                "reason": "CleanPartialCoverage",
                "verdict": "PASS",
                "coverage": "partial",
                "frames_stepped": 16,
                "quarantined": False,
                "violation_slots": [],
                "unobservable_slots": [466, 490],
            }
        )
    )
    return str(path)


def test_default_mode_unchanged(fake_gate, capsys, monkeypatch):
    """No --work-type ⟹ plain tier-1 report, exit on safety (legacy behavior)."""
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=True, verdict="OK"))
    rc = cli.cmd_preadmit(_ns())
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["verdict"] == "OK" and "admitted" not in out


def test_unsafe_tier1_exits_2(fake_gate, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", _Stub("garbage"))
    fake_gate(PreAdmitResult(safe=False, verdict="CE"))
    rc = cli.cmd_preadmit(_ns())
    capsys.readouterr()
    assert rc == 2


def test_admission_mode_requires_source_run_id(fake_gate, monkeypatch):
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=True, verdict="OK"))
    rc = cli.cmd_preadmit(_ns(work_type=644))
    assert rc == 1


def test_admission_mode_safe_none_refuses(fake_gate, monkeypatch):
    """safe is None (no GPU for tier-1 dynamic gate) ⟹ refuse, exit 1."""
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=None, verdict="SKIP"))
    rc = cli.cmd_preadmit(_ns(work_type=644, source_run_id="run-x"))
    assert rc == 1


def test_admission_partial_shadow_admits_and_carries_blindspot(
    fake_gate, capsys, monkeypatch, tmp_path
):
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=True, verdict="OK"))
    shadow = _partial_shadow(tmp_path / "v.json")
    rc = cli.cmd_preadmit(
        _ns(work_type=644, source_run_id="run-x", shadow=shadow)
    )
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["admitted"] is True
    assert data["shadow_dispatch"]["coverage"] == "partial"
    assert 466 in data["shadow_dispatch"]["unobservable_slots"]


def test_admission_partial_blocked_under_require_full(
    fake_gate, capsys, monkeypatch, tmp_path
):
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=True, verdict="OK"))
    shadow = _partial_shadow(tmp_path / "v.json")
    rc = cli.cmd_preadmit(
        _ns(
            work_type=644,
            source_run_id="run-x",
            shadow=shadow,
            require_full_coverage=True,
        )
    )
    data = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert data["admitted"] is False


def test_admission_tier1_only_no_shadow_not_admitted(
    fake_gate, capsys, monkeypatch
):
    """No --shadow ⟹ null tier-2 verdict; tier-1 alone never admits."""
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=True, verdict="OK"))
    rc = cli.cmd_preadmit(_ns(work_type=644, source_run_id="run-x"))
    data = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert data["shadow_dispatch_verdict"] is None
    assert data["admitted"] is False


def test_admission_unreadable_shadow_errors(fake_gate, monkeypatch):
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=True, verdict="OK"))
    rc = cli.cmd_preadmit(
        _ns(work_type=644, source_run_id="run-x", shadow="/nonexistent/v.json")
    )
    assert rc == 1


def test_admission_emit_prints_envelope(
    fake_gate, capsys, monkeypatch, tmp_path
):
    """--emit goes through the (monkeypatched) emitter and prints the envelope."""
    monkeypatch.setattr("sys.stdin", _Stub("void main(){}"))
    fake_gate(PreAdmitResult(safe=True, verdict="OK"))
    shadow = _partial_shadow(tmp_path / "v.json")

    captured = {}

    def _fake_emit(**kwargs):
        from autobench.bus.shader_admission import build_shader_admission

        data = build_shader_admission(**kwargs)
        captured["data"] = data
        return {"specversion": "1.0", "type": "tengine.shader.admission.v1", "data": data}

    monkeypatch.setattr(
        "autobench.bus.shader_admission.emit_shader_admission", _fake_emit
    )
    rc = cli.cmd_preadmit(
        _ns(work_type=644, source_run_id="run-x", shadow=shadow, emit=True)
    )
    env = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert env["type"] == "tengine.shader.admission.v1"
    assert env["data"]["admitted"] is True
    assert captured["data"]["shader_id"] == "stdin"


class _Stub:
    """Minimal stdin stub returning a fixed source once."""

    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text
