"""Auto-generate test scaffolding for autobench.

Classes:
    generate_scaffolding(repo_path, change_type) — for feature/bug/refactor/redesign
    CurveballGenerator — generates adversarial inputs, boundary conditions, race conditions
    generate_curveballs(baseline_inputs) — returns adversarial test cases
"""

from __future__ import annotations

import os
import random
import re
import string
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Language-appropriate test file templates
TEST_TEMPLATES: dict[str, str] = {
    "python": textwrap.dedent("""\
        # Auto-generated test scaffolding
        import pytest
        import sys
        sys.path.insert(0, "{module_dir}")

        from {module_import} import {function_name}


        class Test{function_name_cap}:

            def test_basic(self):
                result = {function_name}(*self.basic_args)
                assert result == self.basic_expected

            def test_edge_cases(self):
                pass

            def test_adversarial(self):
                pass

            @pytest.fixture(autouse=True)
            def setup(self):
                self.basic_args = {basic_args}
                self.basic_expected = {basic_expected}
        """),
    "rust": textwrap.dedent("""\
        // Auto-generated test scaffolding
        #[cfg(test)]
        mod tests {{
            use super::*;

            #[test]
            fn test_basic() {{
                let result = {function_name}({basic_args});
                assert_eq!(result, {basic_expected});
            }}

            #[test]
            fn test_edge_cases() {{
                // TODO
            }}

            #[test]
            fn test_adversarial() {{
                // TODO
            }}
        }}
        """),
    "go": textwrap.dedent("""\
        // Auto-generated test scaffolding
        package {package}

        import "testing"

        func Test{function_name_cap}(t *testing.T) {{
            got := {function_name}({basic_args})
            want := {basic_expected}
            if got != want {{
                t.Errorf("{function_name}() = %v, want %v", got, want)
            }}
        }}

        func Test{function_nameCap}_EdgeCases(t *testing.T) {{
            // TODO
        }}

        func Test{function_nameCap}_Adversarial(t *testing.T) {{
            // TODO
        }}
        """),
    "javascript": textwrap.dedent("""\
        // Auto-generated test scaffolding
        const {{ {function_name} }} = require('{module_path}');

        describe('{function_name}', () => {{
            it('basic case', () => {{
                const result = {function_name}({basic_args});
                expect(result).toBe({basic_expected});
            }});

            it('edge cases', () => {{ TODO }});
            it('adversarial', () => {{ TODO }});
        }});
        """),
    "typescript": textwrap.dedent("""\
        // Auto-generated test scaffolding
        import {{ {function_name} }} from '{module_path}';

        describe('{function_name}', () => {{
            it('basic case', () => {{
                const result = {function_name}({basic_args});
                expect(result).toBe({basic_expected});
            }});

            it('edge cases', () => {{ TODO }});
            it('adversarial', () => {{ TODO }});
        }});
        """),
    "java": textwrap.dedent("""\
        // Auto-generated test scaffolding
        import org.junit.jupiter.api.Test;
        import static org.junit.jupiter.api.Assertions.*;

        class {class_name_cap}Test {{
            @Test
            void testBasic() {{
                var result = {function_name}({basic_args});
                assertEquals({basic_expected}, result);
            }}

            @Test
            void testEdgeCases() {{
                // TODO
            }}

            @Test
            void testAdversarial() {{
                // TODO
            }}
        }}
        """),
}


@dataclass
class CurveballCase:
    """A single adversarial test case."""

    name: str
    input: Any
    expected_behavior: str  # "pass", "timeout", "crash", "wrong_answer"
    description: str
    category: str  # "boundary", "race_condition", "adversarial_input", "resource_exhaustion"


@dataclass
class CurveballGenerator:
    """Generates adversarial inputs, boundary conditions, and race conditions.

    Produces test cases designed to stress agents that generate code:
    - Boundary values (empty, very large, very small)
    - Race conditions (concurrent access, timing-sensitive)
    - Adversarial inputs (malformed, crafted edge cases)
    - Resource exhaustion (stack overflow, heap exhaustion, infinite loops)
    """

    seed: int = 42
    _random: random.Random = field(init=False)

    def __post_init__(self):
        self._random = random.Random(self.seed)

    def generate_curveballs(
        self,
        baseline_inputs: list[Any],
        num_cases: int = 20,
    ) -> list[CurveballCase]:
        """Generate adversarial test cases based on baseline inputs.

        Args:
            baseline_inputs: Representative inputs from the problem domain.
            num_cases: Number of curveball cases to generate.

        Returns:
            List of CurveballCase objects.
        """
        cases: list[CurveballCase] = []

        # Boundary cases
        cases.extend(self._generate_boundary_cases(baseline_inputs))

        # Adversarial input cases
        cases.extend(self._generate_adversarial_cases(baseline_inputs))

        # Race condition cases
        cases.extend(self._generate_race_condition_cases(baseline_inputs))

        # Resource exhaustion cases
        cases.extend(self._generate_resource_exhaustion_cases(baseline_inputs))

        # Trim to num_cases
        if len(cases) > num_cases:
            cases = self._random.sample(cases, num_cases)

        return cases

    def _generate_boundary_cases(self, baselines: list[Any]) -> list[CurveballCase]:
        """Generate boundary condition cases."""
        cases: list[CurveballCase] = []

        for baseline in baselines:
            btype = type(baseline)

            # Empty cases
            if btype == str:
                cases.append(CurveballCase(
                    name="empty_string",
                    input="",
                    expected_behavior="pass",
                    description="Empty string input",
                    category="boundary",
                ))
                cases.append(CurveballCase(
                    name="unicode_boundary",
                    input="​﻿",
                    expected_behavior="pass",
                    description="Unicode boundary characters",
                    category="boundary",
                ))
            elif btype in (int, float):
                cases.append(CurveballCase(
                    name="zero",
                    input=0,
                    expected_behavior="pass",
                    description="Zero value",
                    category="boundary",
                ))
                cases.append(CurveballCase(
                    name="max_int_boundary",
                    input=2**63 - 1,
                    expected_behavior="pass",
                    description="Maximum 64-bit integer",
                    category="boundary",
                ))
                cases.append(CurveballCase(
                    name="negative_boundary",
                    input=-2**63,
                    expected_behavior="pass",
                    description="Minimum 64-bit integer",
                    category="boundary",
                ))
            elif btype in (list, tuple):
                cases.append(CurveballCase(
                    name="empty_list",
                    input=[],
                    expected_behavior="pass",
                    description="Empty list",
                    category="boundary",
                ))
                cases.append(CurveballCase(
                    name="single_element",
                    input=[baselines[0] if baselines else 1],
                    expected_behavior="pass",
                    description="Single element list",
                    category="boundary",
                ))
                cases.append(CurveballCase(
                    name="huge_list",
                    input=[1] * 10_000_000,
                    expected_behavior="timeout",
                    description="Very large list (10M elements)",
                    category="boundary",
                ))

        return cases

    def _generate_adversarial_cases(self, baselines: list[Any]) -> list[CurveballCase]:
        """Generate adversarial/malformed input cases."""
        cases: list[CurveballCase] = []

        # SQL injection-style strings (for string-handling code)
        adversarial_strings = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "\x00\x00\x00NULL_BYTES",
            "日本語" * 1000,
            "🎉" * 10000,
            "{{ .TrustedTemplate }}",
            "${ENV_VAR}",
            "A" * 1_000_000,  # Very long string
        ]

        for s in adversarial_strings:
            cases.append(CurveballCase(
                name=f"adversarial_{len(s)}_chars",
                input=s,
                expected_behavior="wrong_answer",  # Should be handled gracefully, not crash
                description=f"Adversarial string: {s[:50]}...",
                category="adversarial_input",
            ))

        # Malformed JSON-like structures
        malformed_jsons = [
            '{"key": "value",}',
            '{"key": }',
            '{"key": "unclosed}',
            '{{"nested": "malformed"}}',
            '[1, 2, 3,]',
            '{"a": [1, 2, {"b": }]}',
        ]
        for json_str in malformed_jsons:
            cases.append(CurveballCase(
                name=f"malformed_json_{len(json_str)}",
                input=json_str,
                expected_behavior="wrong_answer",
                description=f"Malformed JSON: {json_str[:40]}",
                category="adversarial_input",
            ))

        return cases

    def _generate_race_condition_cases(self, baselines: list[Any]) -> list[CurveballCase]:
        """Generate race condition test cases."""
        cases: list[CurveballCase] = []

        # Timing-sensitive cases
        cases.append(CurveballCase(
            name="rapid_fire_inputs",
            input=list(range(1000)),
            expected_behavior="pass",
            description="1000 rapid sequential inputs",
            category="race_condition",
        ))

        # Concurrent modification
        cases.append(CurveballCase(
            name="concurrent_modification",
            input={"shared": [1, 2, 3]},
            expected_behavior="crash",
            description="Simulated concurrent shared-state modification",
            category="race_condition",
        ))

        return cases

    def _generate_resource_exhaustion_cases(self, baselines: list[Any]) -> list[CurveballCase]:
        """Generate resource exhaustion test cases."""
        cases: list[CurveballCase] = []

        # Deeply nested structures
        cases.append(CurveballCase(
            name="deep_recursion_10000",
            input=list(range(10000)),
            expected_behavior="timeout",
            description="Input that triggers 10000 recursive calls",
            category="resource_exhaustion",
        ))

        # Huge input
        cases.append(CurveballCase(
            name="massive_input_100mb",
            input="A" * 100_000_000,
            expected_behavior="timeout",
            description="100MB input string",
            category="resource_exhaustion",
        ))

        # Exponential behavior trigger
        cases.append(CurveballCase(
            name="exponential_behavior",
            input={"nested": [{"x": i} for i in range(20)]},
            expected_behavior="timeout",
            description="Deeply nested structure that could trigger exponential parsing",
            category="resource_exhaustion",
        ))

        return cases


def generate_scaffolding(
    repo_path: str | Path,
    change_type: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate test scaffolding for a repository based on change type.

    Args:
        repo_path: Path to the repository root.
        change_type: One of feature | bug_fix | refactor | redesign.
        output_dir: Optional directory to write scaffold files to.
                    Defaults to a 'tests/autobench' directory in the repo.

    Returns:
        Dict with keys: scaffold_files (list of written paths),
                        scaffold_config (dict of generation metadata),
                        detected_language, function_signatures.
    """
    repo_path = Path(repo_path)
    if output_dir is None:
        output_dir = repo_path / "tests" / "autobench"
    else:
        output_dir = Path(output_dir)

    if not repo_path.exists():
        return {
            "scaffold_files": [],
            "scaffold_config": {},
            "detected_language": None,
            "function_signatures": [],
        }

    # Detect language
    from .engines.sandbox import detect_language
    lang = detect_language(repo_path) or "python"

    # Find function signatures (Python focus)
    signatures = _find_function_signatures(repo_path, lang)

    # Determine template
    template = TEST_TEMPLATES.get(lang, TEST_TEMPLATES["python"])

    scaffold_files: list[str] = []
    generated: list[dict[str, Any]] = []

    for fn in signatures[:10]:  # Scaffold top 10 functions
        content = _fill_template(template, fn, lang, repo_path)
        test_filename = _test_filename(fn["name"], lang, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(test_filename).write_text(content)
        scaffold_files.append(test_filename)
        generated.append({
            "function": fn["name"],
            "filename": test_filename,
            "category": change_type,
        })

    # Change-type specific: add integration test for redesign/redesign
    if change_type in ("redesign", "refactor"):
        integration_test = _generate_integration_test(repo_path, lang, output_dir, signatures)
        if integration_test:
            Path(integration_test["path"]).write_text(integration_test["content"])
            scaffold_files.append(integration_test["path"])
            generated.append(integration_test)

    return {
        "scaffold_files": scaffold_files,
        "scaffold_config": {
            "change_type": change_type,
            "language": lang,
            "output_dir": str(output_dir),
            "num_functions": len(signatures),
        },
        "detected_language": lang,
        "function_signatures": signatures,
        "generated": generated,
    }


def _find_function_signatures(repo_path: Path, lang: str) -> list[dict[str, Any]]:
    """Extract function signatures from source files."""
    signatures: list[dict[str, Any]] = []

    if lang == "python":
        import ast

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", "target", ".venv", "venv"}]
            for f in files:
                if not f.endswith(".py") or f.startswith("test_") or f.endswith("_test.py"):
                    continue
                filepath = Path(root) / f
                try:
                    source = filepath.read_text()
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                            args = [a.arg for a in node.args.args]
                            signatures.append({
                                "name": node.name,
                                "args": args,
                                "module": str(filepath.relative_to(repo_path).with_suffix("")).replace("/", "."),
                                "file": str(filepath.relative_to(repo_path)),
                            })
                except (OSError, SyntaxError):
                    pass

    elif lang in ("javascript", "typescript"):
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules"}]
            for f in files:
                if not (f.endswith(".js") or (lang == "typescript" and f.endswith(".ts"))):
                    continue
                filepath = Path(root) / f
                try:
                    content = filepath.read_text()
                    # Simple regex extraction for function declarations
                    for m in re.finditer(r"(?:function|const|let|export)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)", content):
                        signatures.append({
                            "name": m.group(1),
                            "module": str(filepath.relative_to(repo_path)),
                            "file": str(filepath.relative_to(repo_path)),
                        })
                except OSError:
                    pass

    elif lang == "go":
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {".git", "vendor"}]
            for f in files:
                if not f.endswith(".go") or f.endswith("_test.go"):
                    continue
                filepath = Path(root) / f
                try:
                    content = filepath.read_text()
                    for m in re.finditer(r"func\s+(\w+)\s*\([^)]*\)", content):
                        signatures.append({
                            "name": m.group(1),
                            "module": filepath.stem,
                            "file": str(filepath.relative_to(repo_path)),
                        })
                except OSError:
                    pass

    elif lang == "rust":
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {".git", "target"}]
            for f in files:
                if not f.endswith(".rs") or f.endswith("_test.rs"):
                    continue
                filepath = Path(root) / f
                try:
                    content = filepath.read_text()
                    for m in re.finditer(r"(?:pub\s+)?fn\s+(\w+)\s*<[^>]*>\s*\([^)]*\)", content):
                        signatures.append({
                            "name": m.group(1),
                            "module": filepath.stem,
                            "file": str(filepath.relative_to(repo_path)),
                        })
                except OSError:
                    pass

    return signatures


def _fill_template(template: str, fn: dict[str, Any], lang: str, repo_path: Path) -> str:
    """Fill in a test template with function details."""
    name = fn["name"]
    args = fn.get("args", [])
    module = fn.get("module", "")
    module_dir = str(repo_path / module.replace(".", "/")).rsplit("/", 1)[0]

    # Simple placeholder values for basic args
    basic_args = ", ".join(f"test_{i}" for i in range(len(args)))
    basic_expected = "expected_result"

    return template.format(
        module_dir=module_dir,
        module_import=module,
        module_path=f"./{fn.get('file', '')}".replace(".py", "").replace("/", "/"),
        function_name=name,
        function_name_cap=name.capitalize(),
        class_name_cap=name.capitalize(),
        basic_args=basic_args,
        basic_expected=basic_expected,
        package=module.split(".")[0] if module else "main",
    )


def _test_filename(function_name: str, lang: str, output_dir: Path) -> str:
    """Generate a test filename for a function."""
    suffixes = {
        "python": "_test.py",
        "rust": "_test.rs",
        "go": "_test.go",
        "javascript": ".test.js",
        "typescript": ".test.ts",
        "java": "Test.java",
    }
    suffix = suffixes.get(lang, "_test.py")
    return str(output_dir / f"test_{function_name}{suffix}")


def _generate_integration_test(
    repo_path: Path,
    lang: str,
    output_dir: Path,
    signatures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Generate a change-type-specific integration test."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if lang == "python":
        content = textwrap.dedent("""\
            # Auto-generated integration test
            import pytest
            import sys
            sys.path.insert(0, '{repo_rel}')

            class TestIntegration:
                '''Integration tests for refactored/redesigned components.'''

                def test_all_exports_load(self):
                    '''Smoke test: all modules load without import errors.'''
                    import importlib
                    for sig in SIGNATURES:
                        try:
                            importlib.import_module(sig['module'])
                        except Exception as e:
                            pytest.fail(f"Failed to import {{sig['module']}}: {{e}}")

                def test_no_regression(self):
                    '''Ensure existing function signatures still exist.'''
                    import inspect
                    for sig in SIGNATURES:
                        module = importlib.import_module(sig['module'])
                        assert hasattr(module, sig['name']), \\
                            f"{{sig['module']}}.{{sig['name']}} missing"

                SIGNATURES = {signatures}
            }
        """).format(repo_rel=str(repo_path), signatures=signatures[:10])
        path = output_dir / "test_integration.py"

    elif lang == "go":
        content = textwrap.dedent("""\
            // Auto-generated integration test
            package integration

            import "testing"

            func TestAllPackagesLoad(t *testing.T) {{
                // Smoke test: packages load without errors
            }}

            func TestNoRegression(t *testing.T) {{
                // Ensure exported functions still exist
            }}
        """)
        path = output_dir / "integration_test.go"

    else:
        return None

    return {"path": str(path), "content": content, "type": "integration"}
