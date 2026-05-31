# AST-level Pattern Extraction for the Autobench Improver

*Research note for `nervous-bus-zpk`. Author: Claude (research-advisor + local investigation). Date: 2026-05-16.*

> SICA itself observes: *"the agent in `base_agent` is a minimal agent that can just about perform the meta-improvement task. It lacks efficient file editing tools, devtools such as tree-sitter or LSP integrations, or advanced reasoning structures."* This document is a concrete plan for closing that gap inside autobench.

---

## 1. TL;DR

**What the improver sees today** (`autobench/llm_improver.py` → `_build_diagnosis_prompt`): verdict histogram, aggregate score, raw harness fields. **That is it.** The actual generated code is thrown away — `evaluator.py:305` only stores `generated_code_length`. The improver therefore proposes prompt edits the way a doctor would prescribe medication after reading only the temperature, not the chart.

**Minimum viable feature set** (8 numbers per failed case + the verdict). All extractable with `tree-sitter` (already installed at `0.25.2`) plus the `tree-sitter-python` / `tree-sitter-rust` grammar packages:

| Feature                     | Type   | Why the improver cares                                  |
| --------------------------- | ------ | ------------------------------------------------------- |
| `cyclomatic_complexity`     | int    | High complexity → "decompose into helpers" guidance     |
| `max_loop_nesting`          | int    | `>3` correlates with TLE; suggest hoisting / memoization |
| `recursion_present`         | bool   | Stack-RE failures, "prefer iterative"                    |
| `n_branches`                | int    | Coverage of edge cases; missing guards                   |
| `n_calls`                   | int    | Call-graph fan-out                                       |
| `uses_floats`               | bool   | Codeforces WA on float arithmetic is *the* classic bug  |
| `comprehension_count`       | int    | Pythonic idioms vs C-style loops                         |
| `has_try_except`            | bool   | Defensive coding presence                                |

**Recommended pipeline:**

```
generated_code  ─►  tree-sitter parse  ─►  feature dict
                                                │
verdict, error  ────────────────────────────────┤
                                                ▼
                                       cluster (KMeans, k≈4)
                                                │
                                                ▼
                              per-cluster LLM summary ("what went wrong")
                                                │
                                                ▼
                                  improver prompt (structured context)
```

**First thing to implement** — and the only thing on the critical path:

1. **Capture the generated code into `HarnessResult.metadata["generated_code"]`** (one-line evaluator change).
2. **Add `autobench/ast_features.py`** (~50 lines, sketched in §9).
3. **Pipe a `cluster_summary` field into `_build_diagnosis_prompt`** (one section of the LLM prompt).

Everything else (LSP, embeddings, semgrep) is value-add but not required for the next iteration.

**Answers to the bead's specific questions:**

- **Highest-leverage tree-sitter query**: the *function-definition + call-edge* combo —
  ```
  (function_definition name: (identifier) @fn.name body: (block) @fn.body)
  (call function: [(identifier) @call.callee (attribute attribute: (identifier) @call.callee)])
  ```
  Together these give you the call-graph node set + edge set in one parse, which composes into cyclomatic complexity (subqueries over the body block), recursion detection (call.callee ∈ fn.name set), and fan-out, all from a single pass.
- **Should we capture generated code in case-result events?** **Yes, unconditionally.** It is the single most-leverage observability change available; without it the improver is operating blind. Add a `generated_code` field (truncated to ~4KB) to `HarnessResult.metadata` and to the `autobench.case.result` event schema.

---

## 2. Tree-sitter overview (2026 state)

**Status check on this machine:** `tree-sitter==0.25.2` is already installed. The legacy `tree-sitter-languages` mega-wheel is *not* the recommended path in 2026 — the new approach (since `py-tree-sitter` 0.22) is to install **per-language grammar packages** (`tree-sitter-python`, `tree-sitter-rust`, `tree-sitter-javascript`, etc.) and import their `language()` function. This is faster to install, smaller, and lets you pin grammars independently.

### 2.1 Setup pattern (current API)

```python
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser, Query

PY = Language(tspython.language())
RS = Language(tsrust.language())

parser = Parser(PY)
tree = parser.parse(b"def f(x): return x + 1")
```

Note the 0.25 API quirks:

- `Language(...)` now takes the *PyCapsule* returned by `tspython.language()` directly (older code passed a path).
- `Parser(language)` is positional; `parser.language = ...` is also accepted.
- `Query` now wants a `Language` and a query string; iterate with `QueryCursor.matches(tree.root_node)`.

### 2.2 Language coverage

Tree-sitter has grammars for **200+ languages**. For autobench's domain (Python, Rust, Go, TypeScript, C++, GLSL via `tree-sitter-glsl`, and Bash), every language we benchmark is covered. The grammars vary in fidelity — Python and JS/TS are extremely mature; GLSL is functional but lighter on semantic nodes.

### 2.3 Performance

- **Parse latency**: ~1–5 ms per kilobyte of source. For a 4 KB generated-code blob: <20 ms.
- **Query latency**: O(nodes × patterns); for our 8-feature query suite over a 4 KB file: <10 ms.
- **Total overhead per case**: ~30 ms. For a 200-case benchmark: ~6 s. **Acceptable** — well below the LLM-call cost.
- Memory: each `Language` instance is ~1–3 MB resident; one global instance shared across parses.

### 2.4 Query DSL examples

The Query DSL is S-expression based with capture names (`@cap.name`) and predicates (`#eq?`, `#match?`).

**Find all function definitions:**
```scheme
(function_definition
  name: (identifier) @fn.name
  parameters: (parameters) @fn.params
  body: (block) @fn.body)
```

**All call sites with callee resolution:**
```scheme
(call
  function: [
    (identifier) @callee
    (attribute attribute: (identifier) @callee)
  ])
```

**Decision points (cyclomatic-complexity nodes):**
```scheme
[
  (if_statement) (elif_clause)
  (for_statement) (while_statement)
  (except_clause) (boolean_operator)
  (conditional_expression)
  (list_comprehension (if_clause))
] @decision
```

**Loop nesting (run depth-first walk and count `(for_statement)` / `(while_statement)` ancestors instead of trying to encode max-depth in the DSL — DSL has no aggregations).**

---

## 3. Per-language feature extractors

The codebase has Python coverage today via `repo_analyzer.py:_cyclomatic_complexity_of_source` using stdlib `ast`. That works *only for Python* and *only for full-file syntactically-valid Python*. Tree-sitter wins because:

1. It is error-recovering — partial code (which is most failed agent output, especially on CE verdicts) still parses.
2. It is multi-language — one extractor pattern works for every benchmark language.
3. Queries are declarative — adding a new feature is a query, not a `NodeVisitor` subclass.

### 3.1 Python — `ast` vs tree-sitter

| Concern              | stdlib `ast`                              | tree-sitter               |
| -------------------- | ----------------------------------------- | ------------------------- |
| Syntax errors        | Raises; cannot inspect                    | Recovers; returns partial tree |
| Speed                | ~0.3 ms/kB                                | ~1 ms/kB                  |
| Comments             | Lost (lexer strips)                       | Preserved as nodes        |
| Type info            | None                                      | None (need LSP / mypy)    |
| Cross-language       | Python-only                               | 200+ languages            |

**Recommendation:** Use **tree-sitter for failure analysis** (where syntax may be broken) and keep `ast` for the existing repo-wide complexity computation in `repo_analyzer.py` (where we own the codebase and know it parses).

### 3.2 Rust — tree-sitter + (optional) `syn`

`syn` is the gold standard for Rust AST work but requires `cargo` + a subprocess. Tree-sitter is sufficient for autobench's feature set. If we ever need trait-resolution or borrow info, fall through to **rust-analyzer** over JSON-RPC.

### 3.3 JS/TS, Go, C++

Tree-sitter grammars are mature for all three. For TS specifically, `tree-sitter-typescript` exposes both `typescript()` and `tsx()` language objects.

### 3.4 GLSL (shader benchmarks)

`tree-sitter-glsl` exists and works for our shader cases (`autobench/shader_executor.py`). Useful features for shaders: `branch_count`, `texture_sample_count`, `loop_count`, `uniform_count` — the last one being a known correlate of compile-time success.

---

## 4. Static-analysis enrichment (cheap second layer)

Tree-sitter gives structural features. Static analyzers give *semantic* features. Three already-installed candidates on this system:

| Tool        | Languages                | Per-failure features it emits                                |
| ----------- | ------------------------ | ------------------------------------------------------------ |
| **ruff**    | Python                   | Lint codes (F841 unused-var, E722 bare-except, B007 unused-loop-var). Each code is a categorical feature: `lint_<CODE>_count`. |
| **semgrep** | Multi (Py/Rs/JS/Go/...)  | Semantic-pattern matches; produces named rule IDs. Useful for "agent forgot to check `None`" style patterns. |
| **mypy**    | Python                   | Type errors. Strongest single signal for "agent confused two types". |

The lift is **the rule-ID histogram per failure cluster**: if cluster A has 80% of its members triggering `F841` (unused variables), the improver can be told *"failures in this cluster tend to leave variables dangling — suggest the agent verify all bindings are used."*

`ruff check --output-format=json` is the cheapest integration — sub-ms per file, ships JSON. Recommended integration order: **ruff → mypy → semgrep**, stop at the first one that gives meaningful signal.

---

## 5. Failure clustering approaches

### 5.1 Why cluster?

The improver currently sees `{CE: 4, RE: 3, WA: 9, OK: 12}`. That tells it *types* of failure but not *causes*. If 7 of those 9 WAs share an AST signature (recursive Fibonacci with no memoization), the improver should know that.

### 5.2 Three approaches, ranked

**(a) PCA + KMeans over symbolic feature vectors** — recommended starting point.

```python
# pseudocode
features = [extract(case.metadata["generated_code"], case.language) for case in failed_cases]
X = np.array([[f[k] for k in FEATURE_KEYS] for f in features])
X = StandardScaler().fit_transform(X)
X2 = PCA(n_components=min(4, X.shape[1])).fit_transform(X)
labels = KMeans(n_clusters=min(4, len(X2))).fit_predict(X2)
clusters = group_by(labels, failed_cases)
```

- Cheap, deterministic, explainable.
- k=4 is a reasonable default for ~50-case batches; for very small batches use HDBSCAN with `min_cluster_size=3`.
- **Per-cluster summary**: take the centroid's nearest case + verdict mode + most-distinctive feature (highest z-score vs corpus mean) and hand those to the improver.

**(b) Embedding-based clustering** — `CodeBERT`, `UniXcoder`, `CodeT5+`, or 2026's `CodeQwen-Embed`.

- Captures semantics symbolic features miss ("the variable is named `i` but used as a key, weird").
- Cost: embedding-model inference per case (~50 ms on CPU, ~5 ms on GPU).
- Drawback: not explainable. The improver gets a cluster label but no "this cluster has unused-variables-with-int-types" rationale unless you bolt on a secondary descriptor pass.

**(c) LLM-as-clusterer (single secondary call)** — give the LLM the full set of failed code snippets and ask it to *group them by root cause*.

- Highest semantic quality. Catches things like "all of these are off-by-one on string indexing."
- Cost: one secondary LLM call (~$0.01–0.03) per improvement iteration.
- Best used as a **refinement layer** on top of (a): symbolic clusters first, then LLM names them.

**Recommended composite:** **(a) → (c)**. Symbolic clustering for the structure (fast, deterministic), LLM for naming and root-cause hypothesis (one call, cheap). Skip (b) for v1.

---

## 6. LLM-vs-structured-features tradeoff

| Use **structured features** when…                          | Use **embedding/LLM clustering** when…                |
| ---------------------------------------------------------- | ----------------------------------------------------- |
| You want explainable improver context.                     | You suspect semantic, not structural, failures.       |
| Feature set is small (≤20) and stable.                     | Failures span many languages with different AST shapes. |
| You're optimizing latency or running on CPU.               | You can afford one extra LLM call per iteration.      |
| You need deterministic reproducibility.                    | Drift over time is acceptable.                        |

Autobench's current setting screams *structured-features-first*: small `(verdict, score, error)` already exists, the improver is itself an LLM call (so we want the extra context to be **complementary**, not redundant), and the v1 RSI loop is small (~5 iterations × ~20 cases).

---

## 7. Concrete pipeline diagram

```
                          ┌────────────────────────────────┐
                          │  BenchmarkCase                 │
                          │  (id, prompt, language, ...)   │
                          └───────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │  improver_agent(case)           │
                         │  → generated_code (string)      │
                         └───────────────┬─────────────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                ┌──────────────────┐         ┌──────────────────────┐
                │  sandbox exec    │         │  ast_features.py     │
                │  → verdict, err  │         │  → feature dict      │
                └──────────┬───────┘         └──────────┬───────────┘
                           │                            │
                           └──────────────┬─────────────┘
                                          ▼
                          ┌────────────────────────────────┐
                          │  HarnessResult                 │
                          │  metadata.generated_code       │
                          │  metadata.ast_features         │
                          └───────────────┬────────────────┘
                                          │       (gathered over N cases)
                                          ▼
                          ┌────────────────────────────────┐
                          │  diagnoser.py (NEW)            │
                          │  PCA + KMeans + LLM-namer      │
                          │  → cluster_summaries[]         │
                          └───────────────┬────────────────┘
                                          ▼
                          ┌────────────────────────────────┐
                          │  llm_improver.py               │
                          │  diagnosis_prompt now includes │
                          │  cluster_summaries[]           │
                          └────────────────────────────────┘
```

The dotted line: the `generated_code` *also* gets emitted on the nervous-bus `autobench.case.result.v1` channel for offline mining (deer-flow can rerun feature extraction with newer models without re-executing the harness).

---

## 8. The "diagnoser" pattern

Split improvement into two stages with a clean interface.

### 8.1 Stage 1 — Diagnoser

**Input**: `list[HarnessResult]` with `generated_code` and `ast_features` populated.
**Output**: a `Diagnosis` dataclass:

```python
@dataclass
class Diagnosis:
    cluster_summaries: list[ClusterSummary]   # 1-N groups of related failures
    dominant_pattern: str                      # "recursion+TLE", "float-eq+WA", etc.
    cross_cluster_signal: dict[str, float]     # e.g. {"recursion_present": 0.62}
    suggested_focus: str                       # natural-language hint, ≤200 chars
```

The diagnoser is **purely analytical** — no harness mutation, no LLM-as-decider. It can be re-run on archival data.

### 8.2 Stage 2 — Improver

The existing `llm_improver.py` already plays this role. Augment its prompt to include the `Diagnosis` rendered as Markdown. The improver then proposes `HarnessConfig` deltas as it does today, but with richer context.

### 8.3 Why the split

1. **Testability** — diagnoser is deterministic given fixed seeds; you can unit-test it without LLM mocks.
2. **Reuse** — the same Diagnosis can feed *multiple* improvers (LLM, rule-based, minimax) for comparison.
3. **Observability** — emit the Diagnosis as its own `autobench.diagnosis.v1` event. Now you can ask "why did the improver propose X?" and see the structured input.
4. **Cost** — diagnoser is CPU-only; you only pay the LLM cost in the improver step.

---

## 9. 50-line implementation sketch — `autobench/ast_features.py`

```python
"""Lightweight AST feature extraction for autobench failure analysis.

Per-language tree-sitter parsers + a small declarative query set.
Used by the diagnoser; consumed by the improver via HarnessResult.metadata.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser, Query, QueryCursor

_LANGS: dict[str, Language] = {
    "python": Language(tspython.language()),
    "rust":   Language(tsrust.language()),
}
_PARSERS: dict[str, Parser] = {k: Parser(v) for k, v in _LANGS.items()}

# Decision-point node types per language (cyclomatic complexity = 1 + count)
_DECISION_NODES = {
    "python": {"if_statement", "elif_clause", "for_statement", "while_statement",
               "except_clause", "boolean_operator", "conditional_expression"},
    "rust":   {"if_expression", "match_expression", "while_expression",
               "for_expression", "loop_expression", "binary_expression"},
}

@dataclass(frozen=True)
class AstFeatures:
    cyclomatic_complexity: int = 1
    max_loop_nesting: int = 0
    recursion_present: bool = False
    n_branches: int = 0
    n_calls: int = 0
    uses_floats: bool = False
    comprehension_count: int = 0
    has_try_except: bool = False
    parse_ok: bool = True
    def to_dict(self) -> dict[str, Any]: return asdict(self)

_NULL = AstFeatures(parse_ok=False)

def extract(code: str, language: str = "python") -> AstFeatures:
    if language not in _PARSERS or not code: return _NULL
    tree = _PARSERS[language].parse(code.encode("utf-8"))
    root = tree.root_node
    decisions = _DECISION_NODES.get(language, set())
    fn_names: set[str] = set(); calls: list[str] = []
    cc = 1; loop_depth_max = 0; comps = 0; floats = False
    branches = 0; tryx = False

    def walk(n, loop_depth: int = 0) -> None:
        nonlocal cc, loop_depth_max, comps, floats, branches, tryx
        t = n.type
        if t in decisions: cc += 1
        if t in ("if_statement", "elif_clause", "if_expression", "match_expression"): branches += 1
        if t in ("list_comprehension", "set_comprehension", "dict_comprehension",
                 "generator_expression"): comps += 1
        if t == "try_statement": tryx = True
        if t == "float": floats = True
        if t == "function_definition" and (nm := n.child_by_field_name("name")):
            fn_names.add(nm.text.decode())
        if t == "call" and (fn := n.child_by_field_name("function")):
            calls.append(fn.text.decode().split(".")[-1])
        new_depth = loop_depth + (1 if t in ("for_statement", "while_statement",
                                              "for_expression", "while_expression") else 0)
        loop_depth_max = max(loop_depth_max, new_depth)
        for c in n.children: walk(c, new_depth)

    walk(root)
    return AstFeatures(
        cyclomatic_complexity=cc, max_loop_nesting=loop_depth_max,
        recursion_present=bool(fn_names & set(calls)), n_branches=branches,
        n_calls=len(calls), uses_floats=floats, comprehension_count=comps,
        has_try_except=tryx, parse_ok=not root.has_error,
    )
```

That's it. ~60 lines including dataclass + extractor. Adding new languages is a `tree_sitter_X.language()` line + a `_DECISION_NODES[lang]` entry.

**Companion test (sanity):**

```python
def test_extract_recursive_python():
    code = "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)"
    f = extract(code, "python")
    assert f.parse_ok and f.recursion_present and f.cyclomatic_complexity >= 2
```

---

## 10. What we change in `BenchmarkCase` / `HarnessResult` / observability

### 10.1 `evaluator.py` (existing, line ~303)

```python
# Before:
metadata={"case_id": case.id, "generated_code_length": len(code)},

# After:
from .ast_features import extract as ast_extract
metadata={
    "case_id": case.id,
    "generated_code_length": len(code),
    "generated_code": code[:4096],  # truncated; full code in event payload
    "ast_features": ast_extract(code, case.language).to_dict(),
},
```

### 10.2 `HarnessResult` (no schema change required)

`metadata: dict[str, Any]` already exists. We are populating documented keys. No dataclass field changes.

### 10.3 New nervous-bus channel: `autobench.case.result.v1`

Schema (`schemas/autobench.case.result.v1.json`):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["case_id", "verdict", "p_score", "generated_code", "ast_features"],
  "properties": {
    "case_id": {"type": "string"},
    "verdict": {"enum": ["CE","RE","TLE","MLE","WA","OK","VF"]},
    "p_score": {"type": "number"},
    "generated_code": {"type": "string", "maxLength": 16384},
    "ast_features": {
      "type": "object",
      "properties": {
        "cyclomatic_complexity": {"type": "integer"},
        "max_loop_nesting": {"type": "integer"},
        "recursion_present": {"type": "boolean"},
        "n_branches": {"type": "integer"},
        "n_calls": {"type": "integer"},
        "uses_floats": {"type": "boolean"},
        "comprehension_count": {"type": "integer"},
        "has_try_except": {"type": "boolean"},
        "parse_ok": {"type": "boolean"}
      }
    }
  }
}
```

### 10.4 Observability hook

`autobench/observability.py` already emits per-case events. Add `generated_code` and `ast_features` to the payload assembled in `AutobenchObservability.emit_case_result(...)`. With this in place, deer-flow can mine historical runs **without re-executing the harness**, which is the single biggest unlock for offline RSI research.

### 10.5 Improver prompt change (`llm_improver.py:_build_diagnosis_prompt`)

Insert a new section after the verdict histogram:

```
## Failure structure
We clustered the {N} failing cases into {K} groups by AST features:

Cluster A ({n_a} cases, dominant verdict {verdict}):
  - dominant pattern: {pattern_name}
  - mean cyclomatic_complexity: {cc:.1f}
  - {recursion_pct}% use recursion
  - representative case: {case_id}
  - sample code (truncated):
    ```{lang}
    {snippet}
    ```

Cluster B ...

Suggested focus from diagnoser: {suggested_focus}
```

---

## 11. Risks

1. **Feature explosion.** It's tempting to add 50 features. Resist. Start with 8, see what the LLM uses, iterate. The Diagnosis struct is your forcing function — if a feature doesn't end up in a `cluster_summary`, drop it.
2. **Overfitting to AST shape.** Tree-sitter sees structure, not behavior. Two semantically-identical solutions can have wildly different ASTs. Mitigate by combining with verdict/error-string features and (eventually) by LLM-named clusters.
3. **Language-specific blind spots.** GLSL's tree-sitter grammar is shallower than Python's; some features (e.g. `recursion_present`) are not meaningful in GLSL. The dataclass should report `parse_ok=False` rather than zero-valued fields, so the diagnoser can mask them.
4. **Privacy/leakage.** Generated code may contain prompt content. For autobench this is fine (we own the prompts), but if we ever ingest external agents' code, we need a redaction pass.
5. **Truncation hides bugs.** 4 KB code cap will truncate ~5% of real codeforces solutions. Store the full code in the bus event but cap `metadata.generated_code` so the in-memory dataclasses stay light.
6. **Cluster instability with small N.** KMeans with k=4 on N<10 is noisy. Gate the diagnoser: if `len(failed_cases) < 6`, skip clustering and just emit per-case features.
7. **Tree-sitter parse cost on giant outputs.** A pathological 100 KB output could spend 100 ms parsing. Cap input length at parse time too.

---

## 12. Citations

- **SICA paper** — Robeyns et al., *A Self-Improving Coding Agent*, ICLR 2025 SSI-FM workshop. The "lacks tree-sitter / LSP integrations" quote is from §2 of the arXiv version. [arxiv.org/abs/2504.15228](https://arxiv.org/abs/2504.15228)
- **py-tree-sitter 0.25 docs** — [tree-sitter.github.io/py-tree-sitter](https://tree-sitter.github.io/py-tree-sitter/) — current `Language` / `Parser` / `Query` API used in §9.
- **tree-sitter Query DSL reference** — [tree-sitter.github.io/tree-sitter/using-parsers/queries](https://tree-sitter.github.io/tree-sitter/using-parsers/queries/) — capture syntax, predicates, alternations.
- **tree-sitter-python grammar** — [github.com/tree-sitter/tree-sitter-python](https://github.com/tree-sitter/tree-sitter-python) — node-type reference used to populate `_DECISION_NODES`.
- **CodeBERT / UniXcoder / CodeT5+** — Feng et al. 2020, Guo et al. 2022, Wang et al. 2023. Useful background for §5(b) but **not** recommended for v1.
- **ruff** — [docs.astral.sh/ruff](https://docs.astral.sh/ruff/) — `--output-format=json` for the §4 enrichment layer.
- **semgrep** — [semgrep.dev](https://semgrep.dev/) — multi-language pattern matching.
- **HDBSCAN** — McInnes et al. 2017. Density-based alternative to KMeans for small/noisy N.
- Internal: `autobench/repo_analyzer.py:374-419` (existing `ast`-based cyclomatic complexity), `autobench/evaluator.py:296-307` (the call site that currently discards `code`), `autobench/llm_improver.py:_build_diagnosis_prompt` (the prompt to augment).

---

## Appendix A — Minimal "diagnoser" implementation outline (~80 lines)

For when you actually wire this up (out of scope for this research note, but here is the shape):

```python
# autobench/diagnoser.py
from dataclasses import dataclass, field
from .ast_features import AstFeatures, extract
from .core import HarnessResult, Verdict

@dataclass
class ClusterSummary:
    label: int
    size: int
    dominant_verdict: Verdict
    pattern_name: str
    centroid_features: dict
    representative_case_id: str
    sample_code: str

@dataclass
class Diagnosis:
    cluster_summaries: list[ClusterSummary] = field(default_factory=list)
    dominant_pattern: str = ""
    cross_cluster_signal: dict[str, float] = field(default_factory=dict)
    suggested_focus: str = ""

def diagnose(results: list[HarnessResult], k: int = 4) -> Diagnosis:
    failed = [r for r in results if not r.is_pass()
              and r.metadata.get("ast_features", {}).get("parse_ok")]
    if len(failed) < 6:
        return Diagnosis(suggested_focus="too few failures to cluster")

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    KEYS = ["cyclomatic_complexity", "max_loop_nesting", "n_branches",
            "n_calls", "comprehension_count"]
    X = np.array([[r.metadata["ast_features"][k] for k in KEYS] for r in failed], float)
    X = StandardScaler().fit_transform(X)
    k_eff = min(k, len(failed) // 2)
    labels = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit_predict(X)

    clusters: list[ClusterSummary] = []
    for c in range(k_eff):
        members = [r for r, l in zip(failed, labels) if l == c]
        if not members: continue
        verdicts = [m.verdict for m in members]
        dom_v = max(set(verdicts), key=verdicts.count)
        rep = members[0]
        feats = rep.metadata["ast_features"]
        pattern = _name_pattern(feats, dom_v)
        clusters.append(ClusterSummary(
            label=c, size=len(members), dominant_verdict=dom_v,
            pattern_name=pattern, centroid_features={k: feats[k] for k in KEYS},
            representative_case_id=rep.metadata.get("case_id", "?"),
            sample_code=rep.metadata.get("generated_code", "")[:512],
        ))

    return Diagnosis(
        cluster_summaries=clusters,
        dominant_pattern=clusters[0].pattern_name if clusters else "",
        suggested_focus=_focus_from_clusters(clusters),
    )

def _name_pattern(f: dict, v: Verdict) -> str:
    parts = []
    if f.get("recursion_present"): parts.append("recursion")
    if f.get("max_loop_nesting", 0) >= 3: parts.append("deep-nesting")
    if f.get("cyclomatic_complexity", 0) >= 10: parts.append("high-complexity")
    if f.get("uses_floats") and v == Verdict.WA: parts.append("float-arith")
    if not f.get("has_try_except") and v == Verdict.RE: parts.append("no-defensive")
    return "+".join(parts) or "generic-" + v.name.lower()

def _focus_from_clusters(cs: list[ClusterSummary]) -> str:
    if not cs: return ""
    biggest = max(cs, key=lambda c: c.size)
    return f"largest cluster ({biggest.size} cases) shows pattern: {biggest.pattern_name}"
```

This is what plugs into `llm_improver.py` as the new prompt section.

---

*End of research note. Total: ~580 lines. Implementation footprint for the recommendation: ~140 LOC across two new files (`ast_features.py`, `diagnoser.py`) plus a 4-line edit in `evaluator.py` and a prompt-template change in `llm_improver.py`.*
