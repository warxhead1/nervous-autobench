"""Tests for autobench.refactor_verifier — tier-1 symbol-rename verification."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from autobench.core import Verdict as CoreVerdict
from autobench.audit.refactor_verifier import (
    RenameVerdict,
    RenameVerifier,
    Verdict,
    _dump_normalized,
    _fallback_ast_drift,
    _python_identifier_index,
    _substitute_identifier,
    _which,
    run_ast_grep,
    run_difft,
    run_test_suite,
)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "refactor_tier1"


def _case_dirs() -> list[Path]:
    return sorted(p for p in BENCHMARK_ROOT.iterdir() if p.is_dir() and p.name.startswith("case_"))


def _has_ast_grep() -> bool:
    return _which("ast-grep") is not None


def _has_difft() -> bool:
    return _which("difft") is not None


# --------------------------------------------------------------------------- #
# Core enum & dataclass invariants
# --------------------------------------------------------------------------- #


def test_verdict_strings_match_core_enum():
    """RV/RD/RT in this module mirror the canonical Verdict enum in core.py."""
    assert Verdict.RV == CoreVerdict.RV.value
    assert Verdict.RD == CoreVerdict.RD.value
    assert Verdict.RT == CoreVerdict.RT.value


def test_rename_verdict_to_dict_round_trip():
    rv = RenameVerdict(
        verdict=Verdict.RV,
        ast_changes_outside_rename=["a.py"],
        test_pass=True,
        test_output="3 passed",
        details={"x": 1},
    )
    d = rv.to_dict()
    assert d["verdict"] == "RV"
    assert d["ast_changes_outside_rename"] == ["a.py"]
    assert d["test_pass"] is True
    assert d["details"] == {"x": 1}


# --------------------------------------------------------------------------- #
# Python-AST fallback helpers (always exercised)
# --------------------------------------------------------------------------- #


def test_python_identifier_index_counts(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return foo\n")
    (tmp_path / "b.py").write_text("x = 'foo'  # string literal — not counted\n")
    idx = _python_identifier_index(tmp_path, "foo")
    # a.py: FunctionDef + Name = 2 occurrences. b.py: 0 (string literal).
    assert idx == {"a.py": 2}


def test_python_identifier_index_skips_strings_and_comments(tmp_path):
    (tmp_path / "a.py").write_text('"docstring mentioning foo"\nx = "foo"  # foo in comment\n')
    idx = _python_identifier_index(tmp_path, "foo")
    assert idx == {}


def test_dump_normalized_equivalence():
    before = "def foo(x):\n    return foo(x) + 1\n"
    after = "def bar(x):\n    return bar(x) + 1\n"
    # Normalizing `after` by mapping new->old should equal `before`'s identity dump.
    assert _dump_normalized(before, "foo", "foo") == _dump_normalized(after, "foo", "bar")


def test_dump_normalized_detects_default_arg_drift():
    before = "def foo(x, factor=2):\n    return x * factor\n"
    after = "def bar(x, factor=3):\n    return x * factor\n"
    assert _dump_normalized(before, "foo", "foo") != _dump_normalized(after, "foo", "bar")


def test_fallback_ast_drift_clean_rename(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "m.py").write_text("def foo():\n    return 1\n")
    (after / "m.py").write_text("def bar():\n    return 1\n")
    assert _fallback_ast_drift(before, after, "foo", "bar") == []


def test_fallback_ast_drift_detects_semantic_change(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "m.py").write_text("def foo(x):\n    return x + 1\n")
    (after / "m.py").write_text("def bar(x):\n    return x + 2\n")
    drift = _fallback_ast_drift(before, after, "foo", "bar")
    assert drift == ["m.py"]


def test_substitute_identifier_preserves_strings():
    src = 'x = "foo"\nfoo = 1\nreturn foo\n'
    out = _substitute_identifier(src, "foo", "bar")
    # The identifier `foo` should change, the string literal "foo" must not.
    assert '"foo"' in out
    assert "bar = 1" in out
    assert "return bar" in out


# --------------------------------------------------------------------------- #
# Tool wrappers — mocked subprocess
# --------------------------------------------------------------------------- #


def test_run_ast_grep_parses_jsonl(tmp_path):
    fake_stdout = (
        json.dumps({"file": "a.py", "text": "foo"}) + "\n" + json.dumps({"file": "b.py", "text": "foo"}) + "\n"
    )
    with mock.patch("subprocess.run") as runmock:
        runmock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_stdout, stderr=""
        )
        res = run_ast_grep("/usr/bin/fake-ast-grep", "foo", tmp_path)
    assert res["count"] == 2
    assert set(res["files"]) == {"a.py", "b.py"}


def test_run_ast_grep_handles_missing_binary(tmp_path):
    with mock.patch("subprocess.run", side_effect=FileNotFoundError("nope")):
        res = run_ast_grep("/missing", "foo", tmp_path)
    assert res["count"] == 0
    assert res["ok"] is False
    assert "error" in res


def test_run_difft_captures_output(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("x = 1\n")
    with mock.patch("subprocess.run") as runmock:
        runmock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        res = run_difft("/usr/bin/fake-difft", a, b)
    assert res["ok"] is True
    assert res["identical"] is True


def test_run_test_suite_pass(tmp_path):
    with mock.patch("subprocess.run") as runmock:
        runmock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="3 passed", stderr=""
        )
        r = run_test_suite(["pytest", "-q"], tmp_path)
    assert r["pass"] is True
    assert "3 passed" in r["output"]


def test_run_test_suite_fail(tmp_path):
    with mock.patch("subprocess.run") as runmock:
        runmock.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="NameError"
        )
        r = run_test_suite(["pytest", "-q"], tmp_path)
    assert r["pass"] is False
    assert "NameError" in r["output"]


def test_run_test_suite_timeout(tmp_path):
    with mock.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=1),
    ):
        r = run_test_suite(["pytest", "-q"], tmp_path, timeout=1)
    assert r["pass"] is False
    assert "TIMEOUT" in r["output"]


# --------------------------------------------------------------------------- #
# Tool-detection degradation
# --------------------------------------------------------------------------- #


def test_check_scope_falls_back_when_ast_grep_missing(tmp_path):
    (tmp_path / "m.py").write_text("def bar():\n    return bar\n")
    v = RenameVerifier(target_repo=tmp_path, old_name="foo", new_name="bar")
    with mock.patch("autobench.refactor_verifier._which", return_value=None):
        scope = v.check_scope()
    assert scope["tool"] == "python-ast-fallback"
    assert scope["new_count"] >= 1
    assert scope["residual_old_count"] == 0


def test_check_drift_falls_back_when_difft_missing(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "m.py").write_text("def foo():\n    return 1\n")
    (after / "m.py").write_text("def bar():\n    return 1\n")
    v = RenameVerifier(target_repo=after, before_repo=before, old_name="foo", new_name="bar")
    with mock.patch("autobench.refactor_verifier._which", return_value=None):
        drift = v.check_drift()
    assert drift["tool"] == "python-ast-fallback"
    assert drift["drift_files"] == []


def test_verify_skips_tests_when_no_command(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "m.py").write_text("def foo():\n    return 1\n")
    (after / "m.py").write_text("def bar():\n    return 1\n")
    v = RenameVerifier(target_repo=after, before_repo=before, old_name="foo", new_name="bar")
    verdict = v.verify()
    assert verdict.verdict == Verdict.RV
    assert verdict.test_pass is None


# --------------------------------------------------------------------------- #
# Integration: every benchmark case matches its expected verdict
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case_dir", _case_dirs(), ids=lambda p: p.name)
def test_benchmark_case_matches_expected(case_dir: Path):
    task = json.loads((case_dir / "task.json").read_text())
    expected = json.loads((case_dir / "expected_verdict.json").read_text())["verdict"]

    # The benchmark integration test exercises whatever tools are available.
    # The Python-AST fallback is precise enough for these fixtures, so we
    # do NOT skip when ast-grep/difft are missing — that's the whole point
    # of the fallback path.
    verifier = RenameVerifier(
        target_repo=case_dir / "after",
        before_repo=case_dir / "before",
        old_name=task["old_name"],
        new_name=task["new_name"],
        test_command=task.get("test_command"),
        test_timeout=task.get("test_timeout", 60),
    )
    verdict = verifier.verify()
    assert verdict.verdict == expected, (
        f"{case_dir.name}: got {verdict.verdict}, expected {expected}. "
        f"details={verdict.details}"
    )


# --------------------------------------------------------------------------- #
# Optional: ast-grep / difft integration when actually installed
# --------------------------------------------------------------------------- #


@pytest.mark.requires_tools
@pytest.mark.skipif(not _has_ast_grep(), reason="ast-grep not installed")
def test_real_ast_grep_invocation(tmp_path):
    (tmp_path / "m.py").write_text("def foo():\n    return foo()\n")
    res = run_ast_grep(_which("ast-grep"), "foo", tmp_path)
    assert res["ok"] is True
    assert res["count"] >= 1


@pytest.mark.requires_tools
@pytest.mark.skipif(not _has_difft(), reason="difft not installed")
def test_real_difft_invocation(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("x = 1\n")
    res = run_difft(_which("difft"), a, b)
    assert res["ok"] is True
