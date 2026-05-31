# Behavioral Equivalence Verifiers for Autobench Refactor Tiers (2026)

**Bead:** nervous-bus-skn
**Date:** 2026-05-16
**Scope:** Survey of the SOTA for "are these two programs behaviorally equivalent?" verifiers
applicable to autobench refactor tiers 1–4. Honest about hype vs substance.

---

## 1. TL;DR — Recommended Verifier Stacks by Tier

| Tier | Refactor class | Verifier stack | Confidence |
|------|----------------|----------------|-----------|
| 1 | Symbol rename | `ast-grep` scope check + `git diff --stat` budget + full test suite + (optional) RefactoringMiner spot-check | **High** — ship now |
| 2 | Extract / inline function | RefactoringMiner-style detector + `difftastic` to confirm call-site AST shape + full test suite + Hypothesis property tests on the extracted function signature | **High–Medium** — ship in 1–2 months |
| 3 | Sync → async migration | Differential testing harness with seeded RNG + deterministic-simulation runner (turmoil / madsim / tokio loom for Rust; asyncio-compatible recorder for Python) + race detector (TSan / Go `-race`) + full integration tests | **Medium** — partial coverage, declare unsolved cases explicitly |
| 4 | Framework migration | JudgingPool (multi-LLM rubric) + golden-output replay of integration test fixtures + perf-regression gate + human escalation on disagreement | **Low** — verifier *augments*, does not *replace*, human review |

**Underrated technique I think the field is sleeping on:**
**Deterministic Simulation Testing (DST)** applied to *refactor benchmarks*. FoundationDB and
TigerBeetle proved DST catches concurrency bugs that years of production never surface. The
autobench tier-3 framing (sync → async) is exactly where DST shines — but no public refactor
benchmark currently uses it. We could be the first.

The runner-up underrated technique is **e-graph / equality saturation** (egg, [PLDI 2026 EGRAPHS
workshop](https://pldi26.sigplan.org/home/egraphs-2026)) as a *bidirectional* verifier for
peephole-scale tier-1/tier-2 refactors. Verified rewrites give you a soundness guarantee
without a full SMT call, but tool maturity for general-purpose languages is still thin.

---

## 2. Tier-1 — Symbol Rename

### Problem statement
LLM proposes renaming `foo` → `bar` across a repo. Verifier must confirm:
- The rename is scope-correct (no captures, no shadowing introduced)
- No semantic side effect (no unrelated changes)
- All tests still pass

### Recommended stack

```
LLM patch
  │
  ├─→ ast-grep:  pattern-match all references to old/new name, confirm 1:1
  ├─→ git diff --stat:  enforce "only these files" budget
  ├─→ difftastic:  confirm structural diff is identity-modulo-identifier
  ├─→ Pytest / Cargo test / Vitest:  full suite green
  └─→ (optional) RefactoringMiner:  classifies the change as "Rename"
```

### Why this is enough
A rename that changes program semantics is detectable via:
1. **Test suite regression** — any test that asserts on the symbol's identity (rare) or
   behavior (common) will fail.
2. **AST-diff invariant** — `difftastic` will show exactly one node type changed
   (`identifier` → `identifier`). Anything else is a flag.
3. **Scope check** — `ast-grep` confirms the new name doesn't collide with a closure
   binding or import in any modified file.

### Implementation sketch (~50 lines of glue Python)

```python
def verify_rename(repo, old_name, new_name, llm_patch):
    apply_patch(repo, llm_patch)

    # 1. AST scope check
    matches_old = ast_grep(f'pattern: "{old_name}"', repo).hits
    matches_new = ast_grep(f'pattern: "{new_name}"', repo).hits
    if matches_old != 0:
        return Failure("residual old-name references")
    if matches_new == 0:
        return Failure("rename did not apply")

    # 2. Structural diff budget — only identifier-class changes
    sd = run("difft", "--display=json", "HEAD~1", "HEAD")
    non_identifier_changes = [c for c in sd.changes
                              if c.kind not in {"Atom.Identifier", "Whitespace"}]
    if non_identifier_changes:
        return Failure(f"unexpected change kinds: {non_identifier_changes}")

    # 3. Test suite
    if not run_tests(repo).ok:
        return Failure("tests regressed")

    return Success()
```

**Tool versions:**
- [`ast-grep`](https://github.com/ast-grep/ast-grep) — 20+ languages, YAML rule files,
  produces valid code snippets (unlike Semgrep metavariables).
- [`difftastic`](https://github.com/Wilfred/difftastic) — 30+ languages via tree-sitter,
  v0.69.0 (Apr 2026). JSON output available.
- [RefactoringMiner](https://github.com/tsantalis/RefactoringMiner) 3.0 — Java only;
  precision 99.8% / recall 99.6% on rename detection ([Alikhanifard et al.
  2024](https://users.encs.concordia.ca/~nikolaos/theses/Pouria_Alikhanifard.pdf)).
  PyRef ([Atwi 2021](https://www.inf.usi.ch/lanza/Downloads/MSc/Atwi2021a.pdf)) is the
  Python analog; less battle-tested but usable.

### Confidence: HIGH
This tier is *solved*. Ship it.

---

## 3. Tier-2 — Function Extraction / Inlining

### Problem statement
LLM extracts a block from `foo()` into a new function `helper()`, replacing the block
with a call. Verifier must confirm:
- The extracted body is semantically the original block
- Argument passing is correct (no captured-variable bugs)
- The call site behaves identically
- All tests still pass

### Recommended stack

```
LLM patch
  │
  ├─→ RefactoringMiner-style detector:  classify as "Extract Method"
  │     ↳ confirms helper() exists, foo() calls it, body lines moved
  ├─→ difftastic on the call site:  diff shows "block → call(args)" pattern only
  ├─→ Hypothesis / fast-check / proptest:  generate inputs for helper()
  │     and confirm pre-refactor foo(x) == post-refactor foo(x) on those inputs
  ├─→ Test suite:  green
  └─→ (optional, advanced) ast-grep rule:  no free variables in extracted body
                                            that aren't in helper's signature
```

### Why this is harder than tier-1
Function extraction can silently break:
- **Captured-variable bugs**: original block used `self.cache`; extracted version uses
  a stale copy.
- **Side-effect ordering**: if extracted block modifies state read later in `foo()`,
  reordering breaks things.
- **Exception propagation**: extracted helper catches/wraps an exception the original
  didn't.

Test suites often miss these because the property "this slice of `foo()` behaves the
same as before" isn't usually a test.

### Property-based testing as the gap-filler

**Recommendation:** generate a property test *automatically* from the extracted function's
signature, then run it against both the pre- and post-refactor versions of `foo()`.

```python
# Pseudo-code of the autogenerated property
from hypothesis import given, strategies as st

@given(x=st.integers(), y=st.text())
def test_refactor_equivalence(x, y):
    pre = pre_refactor.foo(x, y)
    post = post_refactor.foo(x, y)
    assert pre == post  # or deep-equal w/ tolerance
```

[Hypothesis](https://hypothesis.readthedocs.io/) has a `ghostwriter` mode that
infers strategies from type hints automatically; in 2026 it now integrates with
Claude Code as a slash-command for writing tests.

For TypeScript/JavaScript: [fast-check](https://github.com/dubzzz/fast-check) — actively
maintained, ~1M downloads/week. For Rust: [proptest](https://github.com/proptest-rs/proptest).
For Java/Kotlin: jqwik — note it entered maintenance mode in March 2026
([baeldung](https://www.baeldung.com/java-jqwik-property-based-testing)),
so for new Kotlin code, prefer kotest's property module instead.

### Implementation sketch

```python
def verify_extract(repo, helper_name, llm_patch):
    apply_patch(repo, llm_patch)

    # 1. Classify the refactor
    rmine = run("refactoring-miner", "-c", "HEAD~1", "HEAD")
    if "Extract Method" not in rmine.refactorings:
        return Failure("not classified as extract-method")

    helper = find_function(repo, helper_name)
    if helper is None:
        return Failure("helper not found")

    # 2. Free-variable scope check via ast-grep
    free_vars = find_free_vars(helper.body) - set(helper.params)
    if free_vars:
        return Failure(f"helper has free vars: {free_vars}")

    # 3. Property test on call site
    call_sites = find_call_sites(helper_name, repo)
    for cs in call_sites:
        prop = synthesize_property(cs.parent_function, helper.signature)
        if not run_hypothesis(prop, n=1000, pre=PRE_REPO, post=repo).ok:
            return Failure(f"behavioral divergence at {cs.file}:{cs.line}")

    # 4. Tests
    return run_tests(repo)
```

### Confidence: HIGH-MEDIUM
Most failure modes are catchable. The remaining hard case is **non-pure functions**
(filesystem, network, time). For those, push to tier-3 verification or require an
explicit mock/fixture.

---

## 4. Tier-3 — Semantic Preservation Under Nondeterminism (async migration)

### Problem statement
LLM converts `requests.get(url)` to `await aiohttp.get(url)`, sprinkles `async`/`await`,
maybe changes a `for` loop to `asyncio.gather()`. Verifier must confirm:
- Same observable outputs
- No new races, deadlocks, or ordering bugs
- Performance does not catastrophically regress

### The honest truth
**This is not fully solvable today.** Concurrency bugs are the canonical "rare-event in
huge state space" problem. Even with TLA+ specs, finding a race usually requires either
(a) a model checker run for hours, or (b) chaos to surface it in production.

But you can get *very far* with deterministic simulation.

### Recommended stack

```
LLM patch
  │
  ├─→ AST-diff:  confirm only async/await/gather idioms changed
  │              (Semgrep rule library)
  ├─→ Test suite under deterministic simulation:
  │     - Python:  pytest-asyncio + freezegun + seeded asyncio loop
  │     - Rust:    turmoil / madsim / tokio loom
  │     - Go:      goleak + -race + custom simulator
  │   Run the integration tests N times with N different RNG seeds
  ├─→ Race detector:  TSan / -race / pytest-asyncio strict mode
  ├─→ Differential output:  same inputs, compare structured outputs
  │     allowing for stable-sort tolerance on collections
  └─→ Perf gate:  p99 latency / throughput within X% of baseline
```

### Deterministic Simulation Testing — the killer technique
DST works by controlling *every* source of nondeterminism (clocks, network, disk,
RNG, scheduler) so the same seed produces the same execution. With a seeded
fuzzer feeding inputs, you simulate years of operation in minutes.

- **FoundationDB**: 18 months building a DST framework before shipping a single byte;
  result is regarded as one of the most reliable databases in existence
  ([eatonphil notes](https://notes.eatonphil.com/2024-08-20-deterministic-simulation-testing.html)).
- **TigerBeetle**: 3.3s of VOPR simulation = 39 minutes of real-world testing.
- **WarpStream**: applies DST to an entire SaaS
  ([blog](https://www.warpstream.com/blog/deterministic-simulation-testing-for-our-entire-saas)).
- **[Antithesis](https://antithesis.com/)** sells DST-as-a-service.

For Rust: [`turmoil`](https://github.com/tokio-rs/turmoil) (network injection) +
[`madsim`](https://github.com/madsim-rs/madsim) (tokio-compatible deterministic runtime)
+ [`loom`](https://github.com/tokio-rs/loom) (concurrency permutations).

For Python: less mature. Best you can do is pytest with `freezegun`, seeded
`random.Random`, and `asyncio` with a single-threaded executor. Network is hard to
sandbox deterministically.

### TLA+ — when to use it (and when not to)
TLA+ ([Lamport, Wikipedia](https://en.wikipedia.org/wiki/TLA%2B)) is exceptional for
*designing* a concurrent algorithm but a poor fit for *verifying a refactor diff*.
You'd need an existing TLA+ spec of the original code to diff against, and almost
no real codebase has that.

**Recommendation:** skip TLA+ for autobench unless we're benchmarking refactors of
code that already ships with a TLA+ spec (e.g., parts of MongoDB, Confluent, AWS S3).

### Symbolic execution — also a poor fit
KLEE / angr / CBMC are stunning research tools but practical limitations
([Poeplau & Francillon 2019](https://www.s3.eurecom.fr/docs/acsac19_poeplau.pdf),
[arXiv:2508.06643](https://arxiv.org/html/2508.06643)) make them unsuitable for
multi-language refactor verification:
- KLEE supports the smallest set of programs; struggles with heap-manipulating code.
- angr is too slow on large binaries.
- All struggle with external environment (sockets, file I/O) and unbounded loops.

For a tier-3 *function-level* refactor with a pure-data pipeline (e.g., parsing,
serialization), symbolic execution is a powerful equivalence oracle. For anything
involving network, threads, or stateful objects: skip.

### Differential testing — the practical workhorse

```python
@hypothesis.given(workload=st.lists(st.text(), min_size=1, max_size=50))
@hypothesis.settings(deadline=None, max_examples=200)
def test_differential(workload, seed=0):
    random.seed(seed)
    pre_out = run_pre_refactor(workload)
    random.seed(seed)
    post_out = run_post_refactor(workload)

    # Tolerance: order-insensitive on sets, exact on sequences
    assert canonicalize(pre_out) == canonicalize(post_out)
```

The *canonicalize* step is the secret sauce: define equivalence classes (sets vs
sequences, float tolerance, timestamp normalization). This is where domain knowledge
enters; an LLM can suggest canonicalizers but a human signs off once per project.

### Confidence: MEDIUM
You can catch ~80% of async-migration regressions. The other 20% — subtle race
conditions surfacing only under specific scheduler patterns — needs DST or production.
**Be explicit in the benchmark report that tier-3 verifier is best-effort.**

---

## 5. Tier-4 — Framework Migration (Flask → FastAPI, Webpack → Vite, etc.)

### Problem statement
Cross-cutting change: new dependencies, new APIs, new idioms, possibly new file
structure. No single AST diff is meaningful. Test suite may itself need rewriting.

### The fundamental insight
At this tier, the question shifts from "is this *equivalent*?" to **"is this a
competent, idiomatic translation that preserves business behavior?"** That's a
judgment call, not a verification.

### Recommended stack — JudgingPool

```
LLM patch
  │
  ├─→ Integration-test replay:
  │     Golden HTTP request/response fixtures recorded against pre-refactor.
  │     Run against post-refactor; diff status, headers (sans server/date),
  │     body (canonicalized JSON).
  │
  ├─→ Perf regression gate:
  │     p50/p95/p99 within ±15%; memory within +25%.
  │
  ├─→ Test suite:  green (allowing for the test suite itself being part of the patch)
  │
  ├─→ JudgingPool — 3+ LLM judges score on rubric:
  │     1. Functional preservation (golden outputs match)         (0–5)
  │     2. Idiomatic use of new framework                          (0–5)
  │     3. No dropped error-handling cases                         (0–5)
  │     4. Migration completeness (no half-translated files)       (0–5)
  │     5. Backward-compat where promised (e.g., URL routes)       (0–5)
  │   Pass if median ≥ 4 across all dimensions.
  │
  └─→ Human gate:  any judge flag < 3 → human review
```

### Why JudgingPool, not a single judge

**[EquiBench (arXiv:2502.12466)](https://arxiv.org/html/2502.12466v2)** is the 2026 reference
benchmark for LLM-as-judge on code equivalence. Headline numbers (best model = o4-mini):

| Category | Accuracy |
|---|---|
| Variable rename (OJ_V) | 96.5% |
| Algorithmic differences (OJ_A) | 89.0% |
| x86-64 superoptimization | 83.0% |
| Dead code elimination | 76.2% |
| CUDA scheduling | 60.8% |
| **Overall** | **82.3%** |

Hardest categories barely beat random (49–53% mean across all 19 models tested).
LLMs "often rely on syntactic similarity rather than exhibiting robust reasoning over
execution semantics." **Single-judge LLM equivalence is unreliable for non-trivial
refactors.**

Pool of N judges with disagreement-triggered escalation buys you robustness:
- 3 judges, unanimous → high confidence
- 2/3 agree → medium confidence
- Split → human

### Golden-output replay — the load-bearing part

For web frameworks, record real HTTP requests during pre-refactor test runs:

```python
# Pre-refactor: record
@pytest.fixture
def golden_recorder(request):
    recorder = VCRRecorder(f"goldens/{request.node.name}.json")
    yield recorder
    recorder.save()

# Post-refactor: replay
def test_endpoint_matches_golden(golden):
    for req, expected_resp in golden:
        actual = client.request(req.method, req.url, **req.kwargs)
        assert canonicalize(actual) == canonicalize(expected_resp)
```

`canonicalize` strips `Server`, `Date`, `Set-Cookie` timestamps, and sorts JSON keys.

### Rubric design principles
1. **Each rubric dimension must be evaluable from the diff + outputs alone.**
   No "is this code elegant?" — non-falsifiable.
2. **Rubrics must compose.** Pass = all dimensions ≥ threshold, not average.
3. **Calibration data**: hand-label 50 framework migrations, measure judge accuracy
   per dimension, drop or rewrite low-accuracy dimensions.

### Confidence: LOW
This tier is *not* about replacing human judgment. The verifier filters out the
worst 80% of bad migrations so humans can focus on the gray-area 20%.

---

## 6. Property-Based Testing — Concrete Recommendations

| Language | Library | Status (2026) | Notes |
|----------|---------|---------------|-------|
| Python | [Hypothesis](https://hypothesis.readthedocs.io/) (6.152+) | Active, IR-layer shrinker 1.38x faster | Ghostwriter mode auto-generates tests from type hints. Best-in-class. |
| TypeScript / JS | [fast-check](https://github.com/dubzzz/fast-check) | Active, ~1M dl/week | Idiomatic ES2020+, integrates with Vitest/Jest. |
| Rust | [proptest](https://github.com/proptest-rs/proptest) | Active | Macro-driven. For QuickCheck-style: `quickcheck` crate. |
| Java | jqwik | **Maintenance mode (Mar 2026)** | Use only for legacy. New Kotlin: kotest-property. |
| Go | rapid (github.com/flyingmutant/rapid) | Active | Stateful machines supported. |
| Haskell | QuickCheck / Hedgehog | Stable | Hedgehog has integrated shrinking. |

**Three properties every refactor verifier should test:**
1. **Equivalence**: `pre(x) == post(x)` for all `x` in the input strategy.
2. **Idempotence (if pure)**: `post(post(x).inputs) == post(x)` — catches subtle state bugs.
3. **Inverse roundtrip (where defined)**: `parse(serialize(x)) == x` — catches data-shape drift.

**Caveat on coverage:** PBT shrinks counterexamples beautifully but doesn't *explore*
adversarial inputs without help. For a function with two `int` parameters Hypothesis
will find boundary bugs (0, MAX, MIN, -1) within ~50 examples. For a function taking
a complex nested structure, you need explicit strategies — auto-derivation hits a wall.

---

## 7. AST-Diff Tools — Comparison Matrix

| Tool | Languages | Algorithm | Output | Refactor-aware? | License | Best for |
|------|-----------|-----------|--------|-----------------|---------|----------|
| [difftastic](https://github.com/Wilfred/difftastic) | 30+ (tree-sitter) | Tree edit distance, custom heuristics | Terminal / JSON | No (structural only) | MIT/Apache | Human-readable diffs, CI gates |
| [GumTree](https://github.com/GumTreeDiff/gumtree) | Java, JS, C, Python, more | Top-down isomorphic + bottom-up greedy (Falleri et al. 2014) | Edit script (HTML/XML) | No | LGPL | Research, base for other tools |
| [RefactoringMiner](https://github.com/tsantalis/RefactoringMiner) 3.0 | Java (best), C++ (RM++) | Refactoring-aware AST diff | Refactoring classification + mappings | **Yes (40+ types)** | MIT | Detecting refactorings in commits |
| [PyRef](https://www.inf.usi.ch/lanza/Downloads/MSc/Atwi2021a.pdf) | Python | RefactoringMiner-style | Refactoring classification | **Yes (subset)** | MIT | Python refactor detection |
| [semgrep](https://github.com/semgrep/semgrep) | 30+ | Pattern matching on AST | Match locations + autofix | Indirect (rule-based) | LGPL | Custom refactor rules, autofix |
| [ast-grep](https://github.com/ast-grep/ast-grep) | 20+ (tree-sitter) | Pattern matching | Match locations + rewrite | Indirect (rule-based) | MIT | Structural rewriting, valid metavariables |
| [SemanticDiff](https://semanticdiff.com/) | 14 (curated) | Semantic-aware diff | VS Code / GitHub plugin | Yes (filters cosmetic changes) | Commercial | IDE/GitHub integration |

**Cross-language semantic equivalence (Python ↔ TypeScript)**: none of the above. This is
research-grade territory; you'd need an intermediate representation (e.g., translating
both into a common bytecode or pseudocode) and that's a project unto itself. For
autobench, **don't attempt cross-language equivalence in v1.**

**For `x = 5; y = x` ≡ `y = 5`** (semantic-equivalent constant propagation): only
SemanticDiff filters this out of view; even it doesn't *prove* equivalence. True semantic
equivalence here needs e-graph rewriting (egg) with a known-sound rule set.

**Recommended pick for autobench:** `difftastic` + `ast-grep` + `RefactoringMiner` (for
Java) / `PyRef` (for Python). Three tools, each adding orthogonal signal.

---

## 8. Differential Testing — Patterns with Code

### Pattern A — Pure-function equivalence

```python
import hypothesis as h
from hypothesis import strategies as st

@h.given(input=st.builds(MyInput))
def test_equiv(input):
    assert pre.transform(input) == post.transform(input)
```

Failure mode handled: any deterministic divergence.
Failure modes *not* handled: nondeterministic, side-effectful, or order-dependent.

### Pattern B — Order-insensitive collection equivalence

```python
def canonicalize(x):
    if isinstance(x, dict):
        return {k: canonicalize(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, set, tuple)):
        return sorted((canonicalize(v) for v in x), key=str)
    return x

assert canonicalize(pre.run()) == canonicalize(post.run())
```

### Pattern C — Seeded RNG differential

```python
def run_both(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    a = pre.run()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    b = post.run()
    return a, b

@h.given(seed=st.integers(0, 2**31 - 1))
def test_seeded_equiv(seed):
    a, b = run_both(seed)
    assert a == b
```

Failure mode handled: nondeterminism from RNG.
Not handled: scheduler nondeterminism, network jitter, time.

### Pattern D — Recorded-fixture replay (for tier-4)

```python
@pytest.mark.parametrize("scenario", load_goldens("integration/"))
def test_golden(scenario):
    response = post_client.request(scenario.request)
    assert canonicalize_http(response) == canonicalize_http(scenario.response)
```

### Pattern E — Deterministic-simulation under chaos

```rust
#[madsim::test]
async fn test_async_refactor_equiv() {
    madsim::runtime::Handle::current().add_node(...);
    let pre_out = madsim::runtime::seed(42, || pre::run()).await;
    let post_out = madsim::runtime::seed(42, || post::run()).await;
    assert_eq!(pre_out, post_out);
}
```

Failure modes handled: scheduler ordering, network reordering (madsim/turmoil
inject these deterministically), task interleavings.

---

## 9. Risks — Where Verifiers Can Lie to Us

1. **Test-suite coverage gaps** → tests pass, behavior diverges.
   *Mitigation:* require coverage delta ≥ 95% on changed lines via `coverage.py` /
   `tarpaulin` / `c8`.

2. **PBT strategy doesn't reach the bug** → 1000 random inputs miss the boundary.
   *Mitigation:* dedicated boundary strategies (`st.just(0)`, `st.just(MAX_INT)`),
   `@example` decorators, mutation testing with `cosmic-ray` / `mutmut` to
   confirm tests can detect injected mutations.

3. **Canonicalization too lossy** → differences hidden by "tolerance".
   *Mitigation:* version-control the canonicalizer; reject any verifier run that
   *modifies* it as part of the patch.

4. **AST-diff blind to identifier-level semantic change** → renaming `delete` to
   `update` passes a rename verifier; the function now does the wrong thing.
   *Mitigation:* require that renames preserve "predicate" annotations from a
   docstring/type stub; otherwise upgrade to tier-2 verification.

5. **LLM-judge collusion** → if JudgingPool members share training data, they fail
   together.
   *Mitigation:* diversify judges (Anthropic + OpenAI + Google), report per-judge
   scores not just consensus.

6. **Performance regressions disguised as equivalence wins** → output identical, but
   100x slower.
   *Mitigation:* mandatory perf-gate in every tier ≥ 3.

7. **Flaky baselines** → if pre-refactor tests are flaky, the verifier can't
   distinguish a regression from noise.
   *Mitigation:* "stable baseline" gate — pre-refactor must pass test suite N times
   in a row before verifier accepts a patch.

8. **EquiBench-style "syntactic similarity" deception** → LLM produces code that
   *looks* like the original but does something different. Most likely in tier-4.
   *Mitigation:* JudgingPool dimension "no syntactic-near-miss" plus
   golden-output replay.

---

## 10. Concrete First Deliverable — autobench Tier-1 Verifier (1 month)

**Goal:** ship a working tier-1 (symbol rename) refactor verifier that gates real LLM
PRs. Target throughput: 100 refactors/hr on a single workstation.

### Components

```
autobench/refactor/
├── verifiers/
│   ├── __init__.py
│   ├── tier1_rename.py       # The verifier
│   ├── ast_grep_wrapper.py   # Subprocess wrapper
│   ├── difft_wrapper.py      # JSON parser for difftastic
│   └── test_runner.py        # Multi-language test invocation
├── benchmarks/
│   ├── rename_corpus.yaml    # 50 hand-curated rename refactors w/ ground truth
│   └── adversarial/          # Renames that *look* safe but break things
├── cli.py                    # autobench refactor verify --tier=1 <repo> <patch>
└── tests/
    └── test_tier1.py
```

### Bead breakdown
1. `autobench-N+1`: scaffold `verifiers/tier1_rename.py` with the four-step
   pipeline above. AC: pass on 5 hand-coded examples.
2. `autobench-N+2`: build `rename_corpus.yaml` (50 examples: 30 Python, 15 TS, 5
   Rust). Each has `before/`, `after/`, `expected_verdict`.
3. `autobench-N+3`: build adversarial corpus — 20 renames that should *fail*
   verification (scope captures, partial renames, semantic-bearing names).
4. `autobench-N+4`: integration test. AC: verifier achieves ≥95% precision and
   ≥90% recall on the combined corpus.
5. `autobench-N+5`: CLI + JSON output, emit `autobench.refactor.verified` events
   onto the nervous bus.
6. `autobench-N+6`: harness 3 SOTA LLMs (Claude 4.7, GPT-5, Gemini 3) on the
   corpus; report per-model pass rates as the first published autobench
   refactor leaderboard.

### Tools to install / vendor
- `ast-grep` (cargo install or homebrew)
- `difftastic` (cargo install or homebrew)
- `RefactoringMiner` (jar, for Java spot-checks only)
- `PyRef` (pip, for Python spot-checks)
- `hypothesis`, `pytest`, `coverage` (already in autobench deps)

### Definition of done
- `autobench refactor verify --tier=1` returns `{verdict, evidence[], confidence}`
- Adversarial corpus catches all 20 "should fail" examples
- ≥95% precision / ≥90% recall on benign corpus
- Bus event emitted per run; deer-flow consumes for leaderboard updates
- README explains tier semantics + known limitations

---

## 11. Citations

### Tools
- ast-grep — https://github.com/ast-grep/ast-grep
- difftastic — https://github.com/Wilfred/difftastic
- GumTree — https://github.com/GumTreeDiff/gumtree
- RefactoringMiner — https://github.com/tsantalis/RefactoringMiner
- semgrep — https://github.com/semgrep/semgrep
- Hypothesis — https://hypothesis.readthedocs.io/
- fast-check — https://github.com/dubzzz/fast-check
- proptest — https://github.com/proptest-rs/proptest
- KLEE — https://klee-se.org/
- angr — https://github.com/angr/angr
- CBMC — https://github.com/diffblue/cbmc
- turmoil — https://github.com/tokio-rs/turmoil
- madsim — https://github.com/madsim-rs/madsim
- loom — https://github.com/tokio-rs/loom
- egg — https://github.com/egraphs-good/egg
- Antithesis — https://antithesis.com/

### Papers / benchmarks
- EquiBench: Benchmarking LLMs' Reasoning about Program Semantics — arXiv:2502.12466
  (https://arxiv.org/html/2502.12466v2)
- Falleri, J.-R. et al. "Fine-grained and accurate source code differencing" (GumTree),
  ASE 2014 — https://hal.science/hal-01054552/document
- Alikhanifard et al. "A Novel Refactoring and Semantic Aware AST Differencing Tool"
  (RefactoringMiner 3.0), TOSEM 2024 — DOI 10.1145/3696002
- Willsey, M. et al. "egg: Fast and Extensible Equality Saturation," PACMPL 2021 —
  DOI 10.1145/3434304
- Poeplau & Francillon, "Systematic Comparison of Symbolic Execution Systems," ACSAC 2019
- "Symbolic Execution in Practice: A Survey" — arXiv:2508.06643
- "LLM-as-a-Judge for Software Engineering" — arXiv:2510.24367
- "Assessing Code Understanding in LLMs" — arXiv:2504.00065
- RESpecBench — OpenReview eFwJZIN9eI
- HEC: Equivalence Verification Checking for Code — USENIX ATC 2025
  (https://www.usenix.org/system/files/atc25-yin.pdf)
- LLM-Guided Strategy Synthesis for Scalable Equality Saturation — arXiv:2604.17364

### Industry write-ups
- WarpStream DST — https://www.warpstream.com/blog/deterministic-simulation-testing-for-our-entire-saas
- TigerBeetle simulator — https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/ARCHITECTURE.md
- eatonphil on DST — https://notes.eatonphil.com/2024-08-20-deterministic-simulation-testing.html
- PLDI 2026 EGRAPHS workshop — https://pldi26.sigplan.org/home/egraphs-2026
- SemanticDiff vs Difftastic — https://semanticdiff.com/blog/semanticdiff-vs-difftastic/
