"""Autobench Tier-1 refactor verifier — symbol rename.

Three-step pipeline (per ``refactor_verifiers_2026.md`` §2 / §10):
  1. ast-grep: confirm only the renamed symbol changed across files.
  2. difftastic: confirm the structural AST diff is identity-modulo-identifier.
  3. Test suite: full project tests must pass post-rename.

Returns a :class:`RenameVerdict` with ``RV`` (verified), ``RD`` (drift), or
``RT`` (test failure) plus structured evidence in ``details``.

Tool degradation: when ``ast-grep`` or ``difft`` are missing, falls back to a
Python-AST walk + a token-level diff that normalises ``new_name`` → ``old_name``
in the after tree. Fallback path is recorded in ``details['tools_used']``.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


__all__ = ["RenameVerifier", "RenameVerdict", "Verdict"]


# Mirror the relevant Verdict values from autobench.core so callers that only
# import this module still get useful enum-like strings without circular import
# risk. The authoritative enum lives in autobench/core.py.
class Verdict:
    RV = "RV"
    RD = "RD"
    RT = "RT"


@dataclass
class RenameVerdict:
    """Structured outcome of a tier-1 rename verification.

    Attributes:
        verdict: ``RV`` (verified), ``RD`` (semantic drift), or ``RT`` (test fail).
        ast_changes_outside_rename: Paths (relative to repo root) where the
            AST/structural diff observed changes beyond the declared
            ``old_name → new_name`` rename. Empty when verdict is ``RV``.
        test_pass: Whether the project test suite passed against the ``after``
            tree. ``None`` when no test command was configured.
        test_output: Captured stdout+stderr from the test runner (truncated).
        details: Free-form evidence dict — populated with the tool outputs
            (ast-grep counts, difftastic JSON, fallback diff lines), the
            ``tools_used`` matrix, and ``old_name`` / ``new_name``.
    """

    verdict: str
    ast_changes_outside_rename: list[str] = field(default_factory=list)
    test_pass: bool | None = None
    test_output: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ast_changes_outside_rename": list(self.ast_changes_outside_rename),
            "test_pass": self.test_pass,
            "test_output": self.test_output,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Tool wrappers
# --------------------------------------------------------------------------- #


def _which(name: str) -> str | None:
    """Locate a CLI binary on PATH (also tries ``sg`` alias for ast-grep)."""
    p = shutil.which(name)
    if p:
        return p
    if name == "ast-grep":
        # ast-grep ships as `sg` on some installs but `sg` is also coreutils.
        # Only accept `sg` if it self-identifies as ast-grep.
        sg = shutil.which("sg")
        if sg:
            try:
                out = subprocess.run([sg, "--version"], capture_output=True, text=True, timeout=2)
                if "ast-grep" in (out.stdout + out.stderr).lower():
                    return sg
            except Exception:
                return None
    return None


def run_ast_grep(binary: str, pattern: str, root: Path) -> dict[str, Any]:
    """Run ``ast-grep run --pattern <pattern> --json=stream <root>``.

    Returns a dict with ``ok``, ``count`` (number of matches), ``files``
    (set of file paths), and ``raw`` (the parsed records list).
    """
    cmd = [binary, "run", "--pattern", pattern, "--json=stream", str(root)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return {"ok": False, "count": 0, "files": [], "raw": [], "error": str(exc)}

    records: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, list):
            records.extend(rec)
        else:
            records.append(rec)

    files = sorted({r.get("file", "") for r in records if r.get("file")})
    return {
        "ok": proc.returncode in (0, 1),  # ast-grep returns 1 on "no matches"
        "count": len(records),
        "files": files,
        "raw": records,
        "stderr": proc.stderr[-1024:],
    }


def run_difft(binary: str, before: Path, after: Path) -> dict[str, Any]:
    """Run ``difft --check-only --exit-code <before> <after>``.

    ``difft`` exits 0 on identical, 1 on different. We capture both so callers
    can interpret. JSON output is also requested when available.
    """
    cmd = [binary, "--display=json", str(before), str(after)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return {"ok": False, "identical": None, "raw": "", "error": str(exc)}

    raw = proc.stdout
    identical = (raw.strip() == "" or proc.returncode == 0) and not raw.strip()
    # difft emits per-file JSON records on lines; parse what we can
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return {
        "ok": True,
        "identical": identical,
        "raw": raw[:4096],
        "records": records,
        "stderr": proc.stderr[-1024:],
        "returncode": proc.returncode,
    }


def run_test_suite(test_command: list[str], cwd: Path, timeout: int = 60) -> dict[str, Any]:
    """Run the project's test command in ``cwd``.

    Returns dict with ``pass`` (bool), ``returncode``, and ``output`` (truncated).
    """
    try:
        proc = subprocess.run(
            test_command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "pass": False,
            "returncode": -1,
            "output": f"[TIMEOUT after {timeout}s]\n{exc.stdout or ''}{exc.stderr or ''}"[-8192:],
        }
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return {"pass": False, "returncode": -1, "output": f"[runner error] {exc}"}

    combined = (proc.stdout + "\n" + proc.stderr)[-8192:]
    return {"pass": proc.returncode == 0, "returncode": proc.returncode, "output": combined}


# --------------------------------------------------------------------------- #
# Python-AST fallback helpers
# --------------------------------------------------------------------------- #


def _python_identifier_index(root: Path, name: str) -> dict[str, int]:
    """Count occurrences of ``name`` as a Python identifier per file.

    Skips comments and string literals — those are not AST ``Name`` nodes.
    Returns ``{relative_path: count}``.
    """
    out: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        try:
            src = path.read_text()
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                count += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                count += 1
            elif isinstance(node, ast.Attribute) and node.attr == name:
                count += 1
            elif isinstance(node, ast.arg) and node.arg == name:
                count += 1
        if count:
            out[str(path.relative_to(root))] = count
    return out


def _dump_normalized(src: str, old_name: str, new_name: str) -> str:
    """Parse Python source, substitute ``new_name`` → ``old_name`` in identifier
    positions, and dump the AST canonically.

    Identifier substitution is word-boundary-aware: when ``old_name`` and
    ``new_name`` differ, any identifier matching ``new_name`` exactly OR
    containing it as a whole word (e.g. ``test_<new_name>``,
    ``<new_name>_helper``, ``test_<new_name>_case``) is rewritten to the
    corresponding ``old_name``-form. This lets the verifier tolerate the
    standard convention that a rename also propagates to derived names
    (test functions, doc-fixture identifiers, etc.) without falsely
    flagging them as drift.

    Two files that differ only by the declared rename will produce identical
    normalized dumps; any other AST delta will surface as a textual diff.
    """
    import re

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return f"<SYNTAX_ERROR>\n{src}"

    if old_name == new_name:
        # identity mode: no substitutions
        substitute = lambda s: s  # noqa: E731
    else:
        # Match `new_name` as a whole identifier OR as an underscore-separated
        # token inside a compound identifier (e.g. test_<new>, <new>_helper).
        # We treat `_` as a token separator so `test_foo` -> `test_bar` is
        # tolerated even though `_` is technically an identifier character.
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(new_name) + r"(?![A-Za-z0-9])")

        def substitute(s: str) -> str:
            if not isinstance(s, str):
                return s
            return pattern.sub(old_name, s)

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            node.id = substitute(node.id)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.name = substitute(node.name)
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            node.name = substitute(node.name)
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            node.name = substitute(node.name)
            self.generic_visit(node)
            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            node.attr = substitute(node.attr)
            self.generic_visit(node)
            return node

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = substitute(node.arg)
            return node

        def visit_alias(self, node: ast.alias) -> ast.AST:
            node.name = substitute(node.name)
            if node.asname:
                node.asname = substitute(node.asname)
            return node

        def visit_Global(self, node: ast.Global) -> ast.AST:
            node.names = [substitute(n) for n in node.names]
            return node

        def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
            node.names = [substitute(n) for n in node.names]
            return node

    tree = Renamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _fallback_ast_drift(before: Path, after: Path, old_name: str, new_name: str) -> list[str]:
    """Identify files whose Python AST differs by more than the declared rename.

    For each file present in either tree:
    * If only on one side → flag (file structure changed).
    * If both: build the normalized dump (after substituting new→old) and
      compare. Any inequality is "drift beyond rename".

    Returns list of relative paths with drift.
    """
    drift: list[str] = []
    rels: set[str] = set()
    for path in before.rglob("*.py"):
        rels.add(str(path.relative_to(before)))
    for path in after.rglob("*.py"):
        rels.add(str(path.relative_to(after)))

    for rel in sorted(rels):
        b = before / rel
        a = after / rel
        if not b.exists() or not a.exists():
            drift.append(rel)
            continue
        try:
            b_src = b.read_text()
            a_src = a.read_text()
        except (OSError, UnicodeDecodeError):
            drift.append(rel)
            continue
        b_norm = _dump_normalized(b_src, old_name, old_name)  # identity on before
        a_norm = _dump_normalized(a_src, old_name, new_name)
        if b_norm != a_norm:
            drift.append(rel)
    return drift


# --------------------------------------------------------------------------- #
# RenameVerifier
# --------------------------------------------------------------------------- #


@dataclass
class RenameVerifier:
    """Verifier for tier-1 symbol-rename refactors.

    Args:
        target_repo: Path to the post-refactor (``after``) tree. The verifier
            runs the test suite against this directory.
        old_name: The symbol's pre-refactor name.
        new_name: The symbol's post-refactor name.
        before_repo: Optional path to the pre-refactor (``before``) tree. When
            supplied, structural-drift detection is enabled (difftastic or the
            Python-AST fallback). Without it, only the AST-scope and test-suite
            signals run.
        test_command: Optional list-form command to invoke the test runner
            (e.g. ``["pytest", "-q"]``). When ``None`` the verifier defers
            from a test signal — useful for fixture-style cases that ship a
            pre-recorded expected verdict.
        test_timeout: Wall-clock timeout for the test command, seconds.
    """

    target_repo: Path
    old_name: str
    new_name: str
    before_repo: Path | None = None
    test_command: list[str] | None = None
    test_timeout: int = 60

    def __post_init__(self) -> None:
        self.target_repo = Path(self.target_repo)
        if self.before_repo is not None:
            self.before_repo = Path(self.before_repo)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def verify(self) -> RenameVerdict:
        """Run the full three-step verifier pipeline and return a verdict."""
        details: dict[str, Any] = {
            "old_name": self.old_name,
            "new_name": self.new_name,
            "target_repo": str(self.target_repo),
            "before_repo": str(self.before_repo) if self.before_repo else None,
            "tools_used": {},
        }

        # 1. AST scope check ------------------------------------------------
        scope = self.check_scope()
        details["scope"] = scope
        details["tools_used"]["scope"] = scope["tool"]

        # 2. AST drift check ------------------------------------------------
        drift_files: list[str] = []
        if self.before_repo is not None:
            drift = self.check_drift()
            details["drift"] = drift
            details["tools_used"]["drift"] = drift["tool"]
            drift_files = list(drift.get("drift_files", []))
        else:
            details["drift"] = {"skipped": "no before_repo provided"}
            details["tools_used"]["drift"] = "skipped"

        # 3. Test suite -----------------------------------------------------
        test_pass: bool | None = None
        test_output = ""
        if self.test_command:
            tres = self.run_tests()
            details["tests"] = tres
            details["tools_used"]["tests"] = " ".join(self.test_command)
            test_pass = bool(tres.get("pass"))
            test_output = tres.get("output", "")
        else:
            details["tests"] = {"skipped": "no test_command provided"}
            details["tools_used"]["tests"] = "skipped"

        # ---- verdict logic ------------------------------------------------
        # Priority: tests fail → RT; otherwise non-rename AST drift → RD;
        # otherwise → RV.
        if test_pass is False:
            verdict = Verdict.RT
        elif drift_files:
            verdict = Verdict.RD
        else:
            # If scope check found residual old_name uses, treat as drift.
            if scope.get("residual_old_count", 0) > 0 and not scope.get("tool", "").startswith("skipped"):
                verdict = Verdict.RD
                drift_files = sorted(scope.get("residual_old_files", []))
            else:
                verdict = Verdict.RV

        return RenameVerdict(
            verdict=verdict,
            ast_changes_outside_rename=drift_files,
            test_pass=test_pass,
            test_output=test_output,
            details=details,
        )

    # ------------------------------------------------------------------ #
    # Step 1 — AST scope check
    # ------------------------------------------------------------------ #

    def check_scope(self) -> dict[str, Any]:
        """Count occurrences of old/new names in the target tree.

        Returns a dict with:
            tool: which backend was used ("ast-grep" or "python-ast-fallback")
            new_count, residual_old_count: integer counts in ``after``
            residual_old_files: list of files where old name still appears
        """
        ast_grep = _which("ast-grep")
        if ast_grep:
            new_res = run_ast_grep(ast_grep, self.new_name, self.target_repo)
            old_res = run_ast_grep(ast_grep, self.old_name, self.target_repo)
            return {
                "tool": "ast-grep",
                "new_count": new_res["count"],
                "new_files": new_res["files"],
                "residual_old_count": old_res["count"],
                "residual_old_files": old_res["files"],
                "ast_grep_ok": new_res.get("ok", False) and old_res.get("ok", False),
            }

        # Fallback: Python AST walk
        new_idx = _python_identifier_index(self.target_repo, self.new_name)
        old_idx = _python_identifier_index(self.target_repo, self.old_name)
        return {
            "tool": "python-ast-fallback",
            "new_count": sum(new_idx.values()),
            "new_files": sorted(new_idx.keys()),
            "residual_old_count": sum(old_idx.values()),
            "residual_old_files": sorted(old_idx.keys()),
        }

    # ------------------------------------------------------------------ #
    # Step 2 — AST drift check (requires before_repo)
    # ------------------------------------------------------------------ #

    def check_drift(self) -> dict[str, Any]:
        """Detect AST-level changes beyond the declared rename.

        Uses difftastic when available; falls back to a Python-AST normalized
        dump comparison (which is precise for .py files and silent on others).
        """
        if self.before_repo is None:
            return {"tool": "skipped", "drift_files": []}

        difft = _which("difft")
        if difft:
            drift_files: list[str] = []
            difft_records: list[dict[str, Any]] = []
            rels: set[str] = set()
            for path in self.before_repo.rglob("*"):
                if path.is_file():
                    rels.add(str(path.relative_to(self.before_repo)))
            for path in self.target_repo.rglob("*"):
                if path.is_file():
                    rels.add(str(path.relative_to(self.target_repo)))
            # We need the rename-normalized comparison too, so we feed difft
            # a temp copy where new_name is renamed back to old_name. That
            # way, an "identical" difft result means "only the rename changed".
            with tempfile.TemporaryDirectory() as tmp:
                tmp_root = Path(tmp)
                for rel in sorted(rels):
                    a = self.target_repo / rel
                    if not a.exists() or not a.is_file():
                        continue
                    try:
                        src = a.read_text()
                    except (OSError, UnicodeDecodeError):
                        continue
                    if rel.endswith(".py"):
                        # token-substitute new→old; identifier-level only.
                        normalized = _substitute_identifier(src, self.new_name, self.old_name)
                    else:
                        normalized = src.replace(self.new_name, self.old_name)
                    dst = tmp_root / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(normalized)

                for rel in sorted(rels):
                    b = self.before_repo / rel
                    a_norm = tmp_root / rel
                    if not b.exists() or not a_norm.exists():
                        drift_files.append(rel)
                        continue
                    r = run_difft(difft, b, a_norm)
                    difft_records.append({"file": rel, "summary": r.get("raw", "")[:512]})
                    if not r.get("identical"):
                        drift_files.append(rel)

            return {
                "tool": "difftastic",
                "drift_files": sorted(set(drift_files)),
                "records": difft_records[:50],
            }

        # Fallback: Python AST normalized dump.
        drift_files = _fallback_ast_drift(
            self.before_repo, self.target_repo, self.old_name, self.new_name
        )
        return {"tool": "python-ast-fallback", "drift_files": drift_files}

    # ------------------------------------------------------------------ #
    # Step 3 — Test suite
    # ------------------------------------------------------------------ #

    def run_tests(self) -> dict[str, Any]:
        if not self.test_command:
            return {"pass": None, "skipped": True}
        return run_test_suite(self.test_command, self.target_repo, timeout=self.test_timeout)


def _substitute_identifier(src: str, src_name: str, dst_name: str) -> str:
    """Rename ``src_name`` → ``dst_name`` in identifier positions only.

    Uses Python's tokenize module to avoid touching string literals and
    comments. Falls back to no-op when the source can't be tokenized.
    """
    import io
    import tokenize

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenizeError, IndentationError):
        return src

    out_parts: list[str] = []
    prev_end = (1, 0)
    for tok in tokens:
        ttype, tstr, start, end, _line = tok
        # Emit whitespace between tokens
        if start[0] > prev_end[0]:
            out_parts.append("\n" * (start[0] - prev_end[0]))
            out_parts.append(" " * start[1])
        elif start[1] > prev_end[1]:
            out_parts.append(" " * (start[1] - prev_end[1]))
        if ttype == tokenize.NAME and tstr == src_name:
            out_parts.append(dst_name)
        else:
            out_parts.append(tstr)
        prev_end = end
    return "".join(out_parts)
