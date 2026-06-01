"""Repository complexity analysis for autobench.

Classes:
    RepoMetrics — cyclomatic_complexity, coupling, test_coverage, num_languages, architecture_depth
    classify_change(pr_diff) — feature / bug_fix / refactor / redesign
    detect_architecture_shift(pr_diff) — detect fundamental architecture changes
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Threshold for considering a change as "architecture shift"
ARCH_DEPTH_THRESHOLD = 4  # modules deeper than this = architectural
COUPLING_HIGH = 0.7  # above this ratio = highly coupled


@dataclass
class RepoMetrics:
    """Metrics describing repository complexity and structure.

    Attributes:
        cyclomatic_complexity: Average cyclomatic complexity across functions.
        coupling: Inter-module coupling ratio (0.0–1.0).
        test_coverage: Estimated test coverage percentage (0.0–1.0).
        num_languages: Number of distinct programming languages detected.
        architecture_depth: Deepest module path depth.
        num_modules: Total number of source modules.
        num_files: Total number of source files.
        test_file_ratio: Ratio of test files to source files.
    """

    cyclomatic_complexity: float = 0.0
    coupling: float = 0.0
    test_coverage: float = 0.0
    num_languages: int = 0
    architecture_depth: int = 0
    num_modules: int = 0
    num_files: int = 0
    test_file_ratio: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "coupling": self.coupling,
            "test_coverage": self.test_coverage,
            "num_languages": self.num_languages,
            "architecture_depth": self.architecture_depth,
            "num_modules": self.num_modules,
            "num_files": self.num_files,
            "test_file_ratio": self.test_file_ratio,
            "metadata": self.metadata,
        }


class ChangeType(str):
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    REDESIGN = "redesign"


# File patterns that indicate test files
TEST_PATTERNS = [
    r"_test\.py$",
    r"test_.*\.py$",
    r"\.test\.py$",
    r"\.spec\.py$",
    r"_tests\.py$",
    r"tests?\.go$",
    r"_spec\.rb$",
    r"\.test\.ts$",
    r"\.test\.js$",
    r"\.spec\.ts$",
    r"\.spec\.js$",
]


def classify_change(pr_diff: str) -> ChangeType:
    """Classify a PR diff as feature / bug_fix / refactor / redesign.

    Uses heuristics on the diff: file paths, commit messages in the diff,
    and patterns of added/removed lines.

    Args:
        pr_diff: The full diff text of a PR (unified diff format preferred).

    Returns:
        One of: feature | bug_fix | refactor | redesign
    """
    diff_lower = pr_diff.lower()

    # Bug fix indicators
    bug_indicators = [
        r"fixes?:",
        r"bug:",
        r"patch:",
        r"hotfix:",
        r"hotfix",
        r"bugfix",
        r"closes?:?\s+#?\d+",
        r"resolves?:?\s+#?\d+",
        r"error:",
        r"exception:",
        r"crash",
        r"null pointer",
        r"race condition",
        r"deadlock",
    ]
    bug_score = sum(1 for p in bug_indicators if re.search(p, diff_lower))

    # Feature indicators
    feature_indicators = [
        r"feat(ure)?:",
        r"adds?:",
        r"new:",
        r"implements?:",
        r"introduces?:",
        r"\+{3}.*\.(py|rs|go|ts|js|java)$",  # new files added
    ]
    feature_score = sum(1 for p in feature_indicators if re.search(p, diff_lower))

    # Refactor indicators
    refactor_indicators = [
        r"refactors?:",
        r"renames?:",
        r"moves?:",
        r"extracts?:",
        r"inline:",
        r"consolidates?:",
        r"deprecates?:",
        r"^-\s+.*function\s+",  # removed functions
        r"^-\s+class\s+",  # removed classes
    ]
    refactor_score = sum(1 for p in refactor_indicators if re.search(p, diff_lower, re.MULTILINE))

    # Count added vs removed lines (refactor tends to be balanced)
    added = len(re.findall(r"^\+[^+]", pr_diff, re.MULTILINE))
    removed = len(re.findall(r"^-[^-]", pr_diff, re.MULTILINE))
    total_lines = added + removed

    if total_lines > 0:
        balance_ratio = min(added, removed) / max(added, removed)
    else:
        balance_ratio = 0.0

    # Balanced add/remove with low test coverage change = refactor
    if balance_ratio > 0.7 and refactor_score > 0:
        return ChangeType.REFACTOR

    # Bug keywords dominate
    if bug_score >= 2 or (bug_score >= 1 and "fix" in diff_lower):
        return ChangeType.BUG_FIX

    # New files + feature keywords
    if feature_score >= 1 and added > removed * 2:
        return ChangeType.FEATURE

    # Lots of new architecture levels touched (deep module changes)
    if refactor_score >= 3 or (added > 500 and refactor_score >= 1):
        return ChangeType.REDESIGN

    # Default: feature (most common)
    return ChangeType.FEATURE


def detect_architecture_shift(pr_diff: str) -> bool:
    """Detect if a diff represents a fundamental architecture change.

    Architecture shifts include:
    - Moving/renaming deep modules (architecture_depth changes)
    - Adding/removing entire layers (e.g., adding a new service boundary)
    - Introducing new patterns (e.g., adding an ORM, message queue, API layer)
    - Large-scale interface changes (many function signatures changing)

    Args:
        pr_diff: Unified diff text.

    Returns:
        True if the diff represents an architecture shift.
    """
    # Check for deep module movements (changes to files in deep directories)
    deep_moves = _find_deep_module_changes(pr_diff)
    if deep_moves:
        return True

    # Check for architectural pattern changes
    arch_patterns = [
        r"models?/.*\.py",  # Django/Flask models
        r"services?/.*\.py",  # Service layer
        r"controllers?/.*\.(py|ts|js)",  # Controllers
        r"views?/.*\.(py|ts|js|html)",  # Views
        r"middleware.*\.(py|ts|js)",  # Middleware
        r"routers?/.*\.(py|ts|go)",  # Routers
        r"handlers?/.*\.(py|ts|go|java)",  # Handlers
        r"jobs?/.*\.(py|rb|js)",  # Job queues
        r"events?/.*\.(py|ts|go)",  # Event systems
        r"sagas?/.*\.(py|ts|go|java)",  # Sagas / workflows
        r"adapters?/.*\.(py|ts|go|rs)",  # Hexagonal adapters
        r"ports?/.*\.(py|ts|go|rs)",  # Ports (hexagonal)
        r"entities?/.*\.(py|ts|go|java)",  # Domain entities
        r"repositories?/.*\.(py|ts|go|java)",  # Repositories
        r"schemas?/.*\.(py|ts|go|sql)",  # Schemas
    ]

    # Count how many distinct architectural layers are touched
    layers_touched: set[str] = set()
    for pattern in arch_patterns:
        if re.search(pattern, pr_diff):
            layer = pattern.split("/")[0].replace("?", "")
            layers_touched.add(layer)

    if len(layers_touched) >= 3:
        return True

    # Check for interface changes (many function/class signature changes)
    signature_changes = _count_signature_changes(pr_diff)
    if signature_changes >= 5:
        return True

    # Check for new entry points
    entry_point_patterns = [
        r"new file:.*main\.(py|rs|go|java|ts|js)",
        r"new file:.*index\.(py|ts|js|html)",
        r"new file:.*app\.(py|rs|ts|js|go)",
        r"new file:.*server\.(py|rs|go|ts|js|java)",
        r"new file:.*api.*\.(py|ts|go|rs)",
    ]
    new_entry_points = sum(1 for p in entry_point_patterns if re.search(p, pr_diff))
    if new_entry_points >= 2:
        return True

    return False


def _find_deep_module_changes(diff: str) -> int:
    """Find occurrences of deep module path changes."""
    # Match lines like "diff --git a/deep/nested/module/file.py"
    path_depths: dict[str, int] = {}
    for line in diff.splitlines():
        m = re.match(r"^[+-]{3} [ab]/(.+)$", line)
        if m:
            path = m.group(1)
            depth = len(Path(path).parts)
            path_depths[path] = depth

    deep_count = sum(1 for d in path_depths.values() if d >= ARCH_DEPTH_THRESHOLD)
    return deep_count


def _count_signature_changes(diff: str) -> int:
    """Count function/method signature changes in a diff."""
    # Look for patterns like "def foo(..." or "func foo(" or "function foo("
    signature_patterns = [
        r"^[+-]\s*def\s+\w+\(",  # Python def
        r"^[+-]\s*async\s+def\s+\w+\(",  # Python async def
        r"^[+-]\s*func\s+\w+\(",  # Go func
        r"^[+-]\s*fn\s+\w+\(",  # Rust fn
        r"^[+-]\s*fun\s+\w+\(",  # Kotlin fun
        r"^[+-]\s*public\s+\w+\s+\w+\(",  # Java/C# methods
    ]
    count = 0
    for pat in signature_patterns:
        count += len(re.findall(pat, diff, re.MULTILINE))
    return count


def analyze_repo(repo_path: str | Path) -> RepoMetrics:
    """Compute RepoMetrics for a repository.

    Performs static analysis on the source tree:
    - Cyclomatic complexity via AST parsing (Python)
    - Inter-module coupling via import graph
    - Test coverage via file ratio heuristics
    - Language count via file extension survey
    - Architecture depth via deepest path

    Args:
        repo_path: Root path of the repository.

    Returns:
        RepoMetrics instance.
    """
    repo_path = Path(repo_path)
    if not repo_path.exists():
        return RepoMetrics()

    # Detect languages
    languages = _detect_languages(repo_path)

    # Count files and modules
    num_files, num_modules, test_file_ratio = _count_files_and_modules(repo_path)

    # Cyclomatic complexity (Python focus; other languages use heuristics)
    avg_cyclomatic = _compute_cyclomatic_complexity(repo_path)

    # Coupling
    coupling = _compute_coupling(repo_path)

    # Test coverage (heuristic via test file ratio)
    test_coverage = min(test_file_ratio * 100, 100.0) / 100.0

    # Architecture depth
    arch_depth = _compute_architecture_depth(repo_path)

    return RepoMetrics(
        cyclomatic_complexity=avg_cyclomatic,
        coupling=coupling,
        test_coverage=test_coverage,
        num_languages=len(languages),
        architecture_depth=arch_depth,
        num_modules=num_modules,
        num_files=num_files,
        test_file_ratio=test_file_ratio,
        metadata={
            "languages": list(languages),
        },
    )


def _detect_languages(repo_path: Path) -> set[str]:
    """Detect languages present in a repo."""
    from autobench.engines.sandbox import LANGUAGE_MAP

    languages: set[str] = set()
    try:
        for entry in os.scandir(repo_path):
            if entry.is_file():
                ext = entry.name.rsplit(".", 1)[-1] if "." in entry.name else ""
                for lang, exts in LANGUAGE_MAP.items():
                    if ext in exts:
                        languages.add(lang)
            elif entry.is_dir() and entry.name not in {".git", "node_modules", "__pycache__", "target", ".venv", "venv"}:
                languages |= _detect_languages(Path(entry.path))
    except PermissionError:
        pass
    return languages


def _count_files_and_modules(repo_path: Path) -> tuple[int, int, float]:
    """Count total files, modules, and test file ratio."""
    total = 0
    test_files = 0
    modules = set()

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", "target", ".venv", "venv"}]

        for f in files:
            if f.startswith("."):
                continue
            total += 1
            if any(re.search(p, f) for p in TEST_PATTERNS):
                test_files += 1

            # Module detection: directory with __init__.py or package marker
            if f == "__init__.py" or f.endswith(".go") or f.endswith(".rs") or f.endswith(".ts"):
                modules.add(root)

    num_modules = len(modules)
    test_ratio = test_files / total if total > 0 else 0.0
    return total, num_modules, test_ratio


def _compute_cyclomatic_complexity(repo_path: Path) -> float:
    """Compute average cyclomatic complexity via AST for Python files."""
    complexities: list[int] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", "target", ".venv", "venv"}]

        for f in files:
            if not f.endswith(".py"):
                continue
            filepath = Path(root) / f
            try:
                source = filepath.read_text()
                complexity = _cyclomatic_complexity_of_source(source)
                if complexity is not None:
                    complexities.append(complexity)
            except (OSError, SyntaxError):
                pass

    if not complexities:
        return 0.0
    return sum(complexities) / len(complexities)


def _cyclomatic_complexity_of_source(source: str) -> int | None:
    """Compute McCabe cyclomatic complexity for a Python source string.

    Complexity = 1 + number of decision points (if, for, while, except, etc.)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    complexity = 1  # base complexity

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):  # and/or expressions
            complexity += len(node.values) - 1
        elif isinstance(node, ast.comprehension):  # list/set/dict comprehensions
            if node.ifs:
                complexity += len(node.ifs)

    return complexity


def _compute_coupling(repo_path: Path) -> float:
    """Compute inter-module coupling ratio.

    Coupling is defined as the ratio of import edges to module count.
    Lower is better (more cohesive). Returns 0.0–1.0 normalized.
    """
    import_edges = 0
    modules_seen: set[str] = set()

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", "target", ".venv", "venv"}]

        for f in files:
            if not f.endswith(".py"):
                continue
            filepath = Path(root) / f
            try:
                source = filepath.read_text()
                module_name = str(filepath.relative_to(repo_path).with_suffix(""))
                modules_seen.add(module_name)

                # Count imports
                for line in source.splitlines():
                    m = re.match(r"^\s*(?:from|import)\s+(\S+)", line)
                    if m:
                        import_edges += 1
            except (OSError, SyntaxError):
                pass

    num_modules = len(modules_seen)
    if num_modules <= 1:
        return 0.0

    # Normalize: 0 imports = 0 coupling; one import per module = moderate
    normalized = min(import_edges / (num_modules * 2), 1.0)
    return normalized


def _compute_architecture_depth(repo_path: Path) -> int:
    """Compute the deepest module path depth."""
    max_depth = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", "target", ".venv", "venv"}]
        depth = len(Path(root).relative_to(repo_path).parts)
        max_depth = max(max_depth, depth)

    return max_depth
