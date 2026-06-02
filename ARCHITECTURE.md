# Autobench Architecture

**Version:** 0.1.0
**Date:** 2026-05-15
**Status:** Draft

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Harness Architecture](#2-harness-architecture)
3. [RSI Loop Architecture](#3-rsi-loop-architecture)
4. [Sandbox Architecture](#4-sandbox-architecture)
5. [Benchmark Suite Architecture](#5-benchmark-suite-architecture)
6. [Evaluator Architecture](#6-evaluator-architecture)
7. [Integration with Deer-Flow](#7-integration-with-deer-flow)
8. [Multi-Agent Topology](#8-multi-agent-topology)
9. [Curveball System](#9-curveball-system)
10. [Pareto Optimization](#10-pareto-optimization)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Unknowns and Research Gaps](#12-unknowns-and-research-gaps)

---

## 1. System Overview

### What Autobench Is

Autobench is a recursive self-improvement coding agent harness system. It evaluates coding agents against benchmark suites, collects verdict-level signals (CE/RE/TLE/MLE/WA/OK), and iteratively improves the harness configuration through an RSI (Recursive Self-Improvement) loop.

### Position in Nervous-Bus Ecosystem

```
nervous-bus
├── autobench/           # This project — harness evaluation + RSI loop
├── adapters/            # Protocol adapters (OSC133, etc.)
├── schemas/             # CloudEvents schema definitions
├── plugin/              # Zellij WASM plugin (routing only)
└── sdk/                 # Shell/Rust/Python SDKs

deer-flow                # Consumer of nervous-bus; hosts the supervisor topology
└── role_router.py       # Adaptive model selection with SQLite observations
```

Autobench is the **evaluation substrate** — it generates problems, runs agents against them, collects verdicts, and feeds results back into harness improvement. Deer-flow provides the **supervisor/subagent topology** and **middleware pipeline** (BudgetViolationMiddleware, RoleSpec activation_predicates) that autobench configures and evaluates.

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Autobench System                          │
│                                                                 │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │   Curveball  │───▶│   Benchmark  │───▶│   Harness       │  │
│  │   Generator  │    │   Suite      │    │   Config        │  │
│  └──────────────┘    └───────────────┘    └────────┬─────────┘  │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │   Pareto     │◀───│   Evaluator   │◀───│   Sandboxed     │  │
│  │   Frontier   │    │   (SICA)      │    │   Executor      │  │
│  └──────────────┘    └───────────────┘    └──────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│                    ┌───────────────┐                            │
│                    │  RSI Loop     │                            │
│                    │  g(C) = perf  │                            │
│                    └───────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼ (improved config)
              ┌────────────────────────┐
              │  deer-flow Supervisor  │
              │  RoleSpec activation  │
              │  BudgetViolation MW   │
              └────────────────────────┘
```

### Key Properties

- **Temporal separation**: Problems released in waves; ground-truth withheld to prevent overfitting (LCB-style)
- **Multi-objective**: Optimizes for quality + cost + speed simultaneously (AAII metric)
- **Contamination-free**: Fresh problem generation every run; cannot be memorized
- **Adversarial**: Curveball system generates boundary conditions and race conditions

---

## 2. Harness Architecture

The harness is the configurable environment in which a coding agent operates. It is decomposed into 8 components, each with a defined self-improvement surface.

### 8-Component Model

```python
@dataclass
class HarnessConfig:
    system_prompt: str          # Agent instructions / persona
    rollout_protocol: RolloutProtocol  # SINGLE | ITERATIVE | SELF_REVISION | MONTE_CARLO
    context_manager: ContextManager   # FULL | BUDGETED | SEMANTIC | HIERARCHICAL
    tool_surface: str           # Tool / function descriptions
    verifiers: list[Verifier]  # Output validation callables
    budget: dict[str, Any]      # max_tokens, max_time_seconds, max_cost_dollars
```

### Component Improvement Surface

| # | Component | Surface | Rationale |
|---|-----------|---------|-----------|
| 1 | **System prompt / persona** | **HIGH** | Poetiq meta-system showed this is the highest-leverage improvement target. Prompt engineering outperforms fine-tuning. |
| 2 | **Tool surface** | **HIGH** | Determines what the agent can do. Clear tool descriptions reduce RE/WA rates. |
| 3 | **Context manager** | **MEDIUM** | HIERARCHICAL reduces WA on complex problems; FULL wastes context on simple ones. |
| 4 | **Memory** | **MEDIUM** | Session memory enables cross-problem learning; must avoid poisoning. |
| 5 | **Rollout protocol** | **MEDIUM** | ITERATIVE/SELF_REVISION helps on hard problems but adds latency. |
| 6 | **Sub-agent topology** | **MEDIUM** | Supervisor→workers→judges pattern; parallelizable but increases overhead. |
| 7 | **Guardrails / gates** | **LOW** | Necessary but not a primary improvement vector. |
| 8 | **Verifiers / judges** | **HIGH** | Accurate verdicts are critical for reliable scoring; bad judges poison RSI. |

### Source Files

- **Core types**: `core.py`
  - `HarnessConfig` (lines 57-90)
  - `HarnessResult` (lines 93-133)
  - `Verdict` enum (lines 18-26)
  - `ContextManager` enum (lines 29-35)
  - `RolloutProtocol` enum (lines 38-44)

### Key Code Patterns

```python
# Harness configuration with budget constraints
harness = HarnessConfig(
    system_prompt="You are a Python coding assistant. Write efficient, correct code.",
    rollout_protocol=RolloutProtocol.ITERATIVE,
    context_manager=ContextManager.HIERARCHICAL,
    tool_surface="read_file(path), write_file(path, content), run(cmd)",
    verifiers=[Verifier(name="exact_match", check=exact_match_fn)],
    budget={
        "max_tokens": 8192,
        "max_time_seconds": 30,
        "max_cost_dollars": 0.10,
    },
)
```

---

## 3. RSI Loop Architecture

### The Core Iteration

The RSI loop implements: **g(C) = perf_metric(improver_agent(harness_v1))**

Where `C` is the harness configuration, `improver_agent` is an LLM that analyzes results and generates improvements, and `perf_metric` is the SICA utility score.

```python
@dataclass
class RSILoop:
    max_iterations: int = 10
    improvement_threshold: float = 0.01
    history: list[tuple[HarnessConfig, HarnessResult]] = field(default_factory=list)
```

### Iteration Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      RSI Loop                                 │
│                                                               │
│  for i in range(max_iterations):                              │
│    1. benchmark_fn(harness) ──────▶ list[HarnessResult]      │
│           │                                                     │
│           ▼                                                     │
│    2. improver_fn(harness, results) ──▶ HarnessConfig v2     │
│           │                                                     │
│           ▼                                                     │
│    3. convergence_check(history) ──▶ bool (stop/continue)    │
│                                                               │
│  return final_harness, final_results                          │
└──────────────────────────────────────────────────────────────┘
```

### Convergence Criteria

Convergence is declared when the last 3 iterations show no improvement above `improvement_threshold` (default 0.01):

```python
def convergence_check(self, iteration_history) -> bool:
    if len(iteration_history) < 3:
        return False
    recent = [h[1].p_score for h in iteration_history[-3:]]
    return all(abs(recent[i] - recent[i-1]) < self.improvement_threshold
               for i in range(1, len(recent)))
```

### Two Improvement Paths

**Rule-based improver** (`rsi_loop.py`, lines 186-228):
- Analyzes verdict distribution
- Adjusts budget proportionally to dominant verdict type
- Switches rollout protocol on TLE patterns
- Escalates context manager on WA patterns

**LLM-based improver** (`rsi_loop.py`, lines 180-184):
- Feeds benchmark results + current config to LLM
- LLM generates targeted modifications as JSON
- Parsed into `ImprovementDelta` for auditability

### Source Files

- **RSI loop core**: `rsi_loop.py`
  - `RSILoop` (lines 136-265)
  - `SelfImprovingHarness` (lines 32-136)
  - `improve_harness()` (lines 138-231)
  - `ImprovementDelta` (lines 19-29)

### Verdict-Driven Strategy Table

| Dominant Verdict | Threshold | Action |
|-----------------|-----------|--------|
| CE >= 30% | ce_count >= total * 0.3 | Reduce `max_tokens` by 20% |
| TLE >= 20% | tle_count >= total * 0.2 | Tighten `max_time_seconds`, switch to ITERATIVE |
| WA >= 30% | wa_count >= total * 0.3 | Escalate context_manager to HIERARCHICAL |
| RE >= 20% | re_count >= total * 0.2 | Append error-handling guidance to tool_surface |
| OK >= 90% | ok_count >= total * 0.9 | Minor refinement; increase `max_tokens` by 10% |

---

## 4. Sandbox Architecture

### Multi-Language Execution

The `SandboxedExecutor` class provides polyglot code execution with verdict detection:

```
Language Support (from sandbox.py, lines 25-134):
  python, rust, go, javascript, typescript, java,
  c, cpp, ruby, bash, php, swift, kotlin, zig
```

### Sandbox Tier Strategy

```
┌─────────────────────────────────────────────────────────┐
│           Sandbox Selection (cold-start priority)        │
│                                                          │
│  1. gVisor          (~50-100ms)  ─ Strong syscall filter │
│     └── Fallback if unavailable                          │
│                                                          │
│  2. Firecracker     (<125ms)     ─ Snapshot/restore      │
│     └── Fallback if unavailable                          │
│                                                          │
│  3. namespace (unshare)  (~1-5ms) ─ Fastest; no filtering│
│     └── Always available fallback                        │
│                                                          │
│  4. raw subprocess   (minimal)  ─ Last resort           │
└─────────────────────────────────────────────────────────┘
```

**Note**: WebAssembly is NOT used for compiled languages — it adds compilation overhead without performance benefit.

### Verdict Detection Precedence

```
CE > TLE > MLE > RE > WA > OK

Priority order (first match wins):
1. CE — stderr contains "error:", "SyntaxError", "ParseError", "cannot find symbol"
2. TLE — runtime_ms > max_time_seconds
3. MLE — stderr matches oom indicators ("OutOfMemoryError", "std::bad_alloc", "killed", "oom_kill")
4. RE — stderr contains "Traceback", "panic:", "Exception in thread", "SIGSEGV"
5. WA — exit_code == 0 but stdout != expected_output
6. OK — all checks passed
```

### Runtime Error Pattern Detection (per language)

From `sandbox.py` lines 354-406:

| Language | Patterns |
|----------|----------|
| python | `Traceback`, `RuntimeError`, `TypeError`, `ValueError`, `IndexError`, `KeyError` |
| rust | `thread '.*' panicked`, `error[E\d+]`, `rust_backtrace` |
| go | `panic:`, `runtime error:`, `fatal error:` |
| javascript | `ReferenceError:`, `TypeError:`, `SyntaxError:`, `RangeError:` |
| java | `Exception in thread`, `java.*Exception`, `Error:` |
| c/cpp | `Segmentation fault`, `core dumped`, `SIGSEGV`, `std::exception` |

### Memory Limit Detection

RSS (Resident Set Size) is not directly available from Python's `subprocess`. Current implementation uses heuristics from stderr:
- "out of memory", "MemoryError", "cannot allocate"
- "std::bad_alloc", "OutOfMemoryError", "Java heap space"
- "Killed" (dmesg), "oom_kill"

**Gap**: Direct RSS tracking requires cgroup integration or `psutil` usage.

### Source Files

- **Executor**: `sandbox.py`
  - `SandboxedExecutor` class (lines 153-424)
  - `LANGUAGE_RUNNERS` dict (lines 42-134)
  - `ExecutionResult` dataclass (lines 137-150)
  - `detect_language()` (lines 427-468)
  - `compile_and_run()` (lines 471-494)
  - `verify_output()` (lines 497-548)

---

## 5. Benchmark Suite Architecture

### LCB-Style Contamination-Free Evaluation

```
┌─────────────────────────────────────────────────────────┐
│           LiveCodeBench (LCB) Model                      │
│                                                          │
│  - Problems released in temporal waves                   │
│  - Ground-truth withheld from agents                    │
│  - Tests 4 planes: generation, self-repair, execution,   │
│    test-output-prediction                               │
│                                                          │
│  Autobench adopts:                                      │
│  - Fresh problem generation each run (cannot memorize)  │
│  - Ground-truth stored separately, revealed post-run    │
│  - Difficulty distribution: easy/medium/hard            │
└─────────────────────────────────────────────────────────┘
```

### Difficulty Distribution

Benchmark cases carry metadata for stratified sampling:

```python
@dataclass
class BenchmarkCase:
    id: str
    prompt: str
    language: str = "python"
    expected_output: str = ""
    constraints: dict[str, Any] = field(default_factory=lambda: {
        "max_time_seconds": 10,
        "max_memory_mb": 512,
    })
    starter_code: str = ""
    test_inputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys: difficulty, category, source (leetcode/atcoder/codeforces)
```

### Problem Sourcing (Future)

```
Phase 1 (current):   Static benchmark files (JSON/YAML)
Phase 2 (planned):   API ingestion from LeetCode/AtCoder/CodeForces
Phase 3 (future):    LLM-generated fresh problems (deer-flow integration)
```

### Source Files

- **Benchmark types**: `evaluator.py`
  - `BenchmarkCase` (lines 29-68)
  - `BenchmarkResult` (lines 71-102)

---

## 6. Evaluator Architecture

### SICA Utility Function

The evaluator uses the SICA utility function from arXiv:2504.15228v2:

```
U = 0.5·p_score + 0.25·(1 - min(1, p_cost/10)) + 0.25·(1 - min(1, p_time/300))

Where:
  p_score  = normalized score [0.0, 1.0]     (1.0 = correct, 0.0 = incorrect)
  p_cost   = cost in dollars (normalized against 10-dollar ceiling)
  p_time   = latency in seconds (normalized against 300-second ceiling)
```

### Multi-Objective Scoring

Default weights (from `evaluator.py` lines 22-26):
```python
DEFAULT_WEIGHTS = {
    "score": 0.5,
    "cost": 0.25,
    "time": 0.25,
}
```

The `BenchmarkEvaluator` scores are normalized internally to sum to 1.0:

```python
w_score = utility_weights.get("score", 0.5) / total_weight
w_cost  = utility_weights.get("cost", 0.25) / total_weight
w_time  = utility_weights.get("time", 0.25) / total_weight

utility = w_score * avg_score + w_cost * (1.0 - avg_cost) + w_time * (1.0 - avg_time)
```

### Verdict Signal Collection

The evaluator collects per-case verdict signals and aggregates into `BenchmarkResult`:

```python
@dataclass
class BenchmarkResult:
    case_results: list[HarnessResult]
    aggregate_score: float          # SICA utility
    total_latency_ms: float
    verdict_counts: dict[str, int]  # CE/RE/TLE/MLE/WA/OK counts
```

### Pareto Frontier Tracking

Pareto optimality is tracked across quality/cost/speed dimensions:
- A configuration is Pareto-optimal if no other configuration dominates it
- Dominance: configuration A dominates B if A is better in all three dimensions

### Source Files

- **Evaluator**: `evaluator.py`
  - `BenchmarkEvaluator` (lines 105-397)
  - `emit_verdict()` (lines 245-335)
  - `score_harness()` (lines 356-396)
  - `BenchmarkResult` (lines 71-102)

---

## 7. Integration with Deer-Flow

### Supervisor Topology Integration

Autobench configures deer-flow's supervisor topology, which provides:

```
Deer-flow Supervisor
├── role_router.py       # Adaptive model selection with SQLite observations
├── BudgetViolationMiddleware   # Token excess detection + logging + context adjustment
└── 12+ middleware layers
```

### Key Integration Points

| Autobench Concern | Deer-Flow Mechanism |
|-------------------|---------------------|
| Model selection | `role_router.py` adaptive model selection |
| Budget enforcement | `BudgetViolationMiddleware` — token excess detection + context trim |
| Verdict signals | `CE/RE/TLE/MLE/WA` verdict-level signals needed in evaluator |
| Session termination | Session termination field needed in deer-flow |
| Agent handoff | `RoleSpec` with `activation_predicate` for supervisor→workers→judges |

### RoleSpec Activation Predicate

The supervisor→workers→judges topology uses RoleSpec activation predicates:

```python
# Hypothetical RoleSpec configuration
supervisor_role = RoleSpec(
    name="supervisor",
    activation_predicate=lambda ctx: ctx.complexity > 5,
    max_retries=3,
)

worker_role = RoleSpec(
    name="worker",
    activation_predicate=lambda ctx: ctx.complexity <= 5,
    max_retries=2,
)

judge_role = RoleSpec(
    name="judge",
    activation_predicate=lambda ctx: ctx.phase == "evaluation",
)
```

### BudgetViolationMiddleware Requirements

Based on deer-flow architecture review, the middleware needs:
1. **Token excess detection**: Compare `usage.total_tokens` against `budget.max_tokens`
2. **Logging**: Record violations with case_id, harness version, and excess amount
3. **Context adjustment**: Trim context or switch to budgeted context manager on violation

### Signal Emitter Interface

For verdict signals to flow back to autobench:

```
Agent code ──▶ verdict signal (CE/RE/TLE/MLE/WA/OK) ──▶ autobench evaluator
                     │
                     ▼
              deer-flow event bus
                     │
                     ▼
              nervous-bus channel: autobench.verdict.*
```

### Source Files

- **Deer-flow integration (planned)**: Integration via deer-flow's `deer obs bus` command
- **Existing bus adapter**: `nervous-bus/sdk/python/nbus.py` (bash shim to `deer obs bus`)

---

## 8. Multi-Agent Topology

### Supervisor → Workers → Judges Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                    Multi-Agent Topology                       │
│                                                               │
│  Supervisor                                                  │
│    │                                                         │
│    ├──▶ Worker Pool (parallel, n=30+)                        │
│    │      ├── Worker 1 ──▶ verdict                           │
│    │      ├── Worker 2 ──▶ verdict                           │
│    │      └── Worker N ──▶ verdict                           │
│    │                                                         │
│    └──▶ Judges Pool (anonymous, peer judging)                │
│           ├── Judge 1 (model A) ──▶ scores Worker 2         │
│           ├── Judge 2 (model B) ──▶ scores Worker 1         │
│           └── Judge N (model N) ──▶ scores Worker M          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Hand-Off Protocol

```
1. Supervisor receives benchmark case
2. Supervisor selects model based on role_router.py observations
3. Supervisor dispatches to Worker with:
   - case_id, prompt, language, constraints, harness_config
4. Worker generates code, submits to sandbox
5. Sandbox returns ExecutionResult with verdict
6. Supervisor forwards to Judges pool for peer evaluation
7. Judges anonymously score workers (AAII metric)
8. Aggregate scores returned to RSI loop
```

### Collective LLM-as-Judge (AutoBench 2.0)

- Every model anonymously judges peers
- Judging prompt does not reveal model identity
- Changes at each run to resist gaming
- AAII (Area Above Ideal) correlation: 87.08%

### Source Files

- **CLI topology (future)**: `cli.py` (currently single-agent)
- **Repo analyzer for change type**: `repo_analyzer.py`
  - `RepoAnalyzer` (lines 25-62)
  - `classify_change()` (lines 88-173)

---

## 9. Curveball System

### Purpose

The Curveball system generates adversarial inputs designed to stress-test agents and expose edge-case failures. It is implemented in `test_scaffolder.py`.

### Curveball Categories

```python
@dataclass
class CurveballCase:
    name: str
    input: Any
    expected_behavior: str   # "pass", "timeout", "crash", "wrong_answer"
    description: str
    category: str            # "boundary", "race_condition", "adversarial_input", "resource_exhaustion"
```

### Four Adversarial Generation Strategies

From `test_scaffolder.py` lines 176-208:

```
┌─────────────────────────────────────────────────────────┐
│              Curveball Generator                         │
│                                                          │
│  generate_curveballs(baseline_inputs, num_cases=20)     │
│                                                          │
│  ├── _generate_boundary_cases()                         │
│  │     ├── empty_string / unicode_boundary              │
│  │     ├── zero / max_int_boundary / negative_boundary  │
│  │     ├── empty_list / single_element / huge_list     │
│  │                                                        │
│  ├── _generate_adversarial_cases()                     │
│  │     ├── SQL injection strings                        │
│  │     ├── XSS payloads                                 │
│  │     ├── Path traversal ("../../../etc/passwd")        │
│  │     ├── NULL bytes, unicode DoS                      │
│  │     ├── Malformed JSON                               │
│  │                                                        │
│  ├── _generate_race_condition_cases()                  │
│  │     ├── rapid_fire_inputs (1000 sequential)          │
│  │     └── concurrent_modification (shared state)       │
│  │                                                        │
│  └── _generate_resource_exhaustion_cases()             │
│        ├── deep_recursion_10000                          │
│        ├── massive_input_100mb                           │
│        └── exponential_behavior (nested structures)      │
└─────────────────────────────────────────────────────────┘
```

### Example Curveball Cases

| Category | Name | Input | Expected |
|----------|------|-------|----------|
| boundary | `empty_string` | `""` | pass |
| boundary | `unicode_boundary` | `​﻿` (zero-width chars) | pass |
| boundary | `max_int_boundary` | `2**63 - 1` | pass |
| boundary | `huge_list` | `[1] * 10_000_000` | timeout |
| adversarial | `adversarial_sql` | `"'; DROP TABLE users; --"` | wrong_answer |
| adversarial | `adversarial_xss` | `"<script>alert('xss')</script>"` | wrong_answer |
| adversarial | `malformed_json` | `'{"key": "value",}'` | wrong_answer |
| race_condition | `rapid_fire_inputs` | `list(range(1000))` | pass |
| resource_exhaustion | `massive_input_100mb` | `"A" * 100_000_000` | timeout |

### Source Files

- **Curveball generator**: `test_scaffolder.py`
  - `CurveballGenerator` (lines 159-381)
  - `CurveballCase` (lines 148-157)
  - `generate_curveballs()` (lines 176-208)

---

## 10. Pareto Optimization

### Multi-Objective Problem

Autobench optimizes across three dimensions:

```
         Maximize  quality (p_score)
         Minimize  cost (p_cost in $)
         Minimize  time (p_time in seconds)
```

### Pareto Frontier

A harness configuration is **Pareto-optimal** if no other configuration is better in all three dimensions.

```
Pareto Frontier (example):
─────────────────────────────────────────────────────
  Config A: score=0.95, cost=$0.05, time=8s  ← Pareto
  Config B: score=0.90, cost=$0.03, time=5s  ← Pareto
  Config C: score=0.92, cost=$0.08, time=12s ← Pareto
  Config D: score=0.93, cost=$0.04, time=9s  ← Dominated (B dominates D)
─────────────────────────────────────────────────────
```

### Pareto Dominance

Configuration X dominates Y if:
```
X.p_score >= Y.p_score  AND
X.p_cost  <= Y.p_cost   AND
X.p_time  <= Y.p_time

AND at least one inequality is strict.
```

### Frontier Tracking Algorithm

```python
def update_pareto_frontier(frontier: list[HarnessConfig], new_config: HarnessConfig) -> list[HarnessConfig]:
    """Add new_config to frontier, removing any configurations it dominates."""

    # Remove configs dominated by new_config
    frontier = [c for c in frontier if not dominates(new_config, c)]

    # Don't add if new_config is dominated by any existing
    for c in frontier:
        if dominates(c, new_config):
            return frontier

    frontier.append(new_config)
    return frontier
```

### Weight Space Exploration

Rather than fixing weights, autobench explores the weight space:

```
Weight configurations to evaluate:
  - Balanced:      (0.5, 0.25, 0.25)
  - Cost-focused:  (0.3, 0.5, 0.2)
  - Quality-first: (0.7, 0.15, 0.15)
  - Speed-first:   (0.3, 0.15, 0.55)
```

This produces a Pareto surface rather than a single optimal point.

### Source Files

- **Pareto tracking (future)**: Not yet implemented
- **Current scoring**: `evaluator.py` lines 356-396 (`score_harness`)

---

## 11. Implementation Roadmap

### Phase 1: Core Sandbox (Current)

**Goal**: Single-language sandbox execution with verdict detection.

**Status**: Implemented in `sandbox.py`

**Delivered**:
- `SandboxedExecutor` with 14-language support
- Verdict detection (CE/RE/TLE/MLE/WA/OK)
- `BenchmarkCase` and `BenchmarkEvaluator` types
- CLI with `run`, `eval`, `scaffold`, `sandbox` commands
- psutil RSS tracking: polls memory every 50ms, kills with SIGKILL when limit exceeded
- cgroup2 memory enforcement via `run_in_cgroup()` with fallback to psutil-only
- gVisor integration stub (`sandbox_type="gvisor"` → `runsc run`)
- Interpreted language fix: `_build_run_cmd` now appends source_path for Python/JS/Ruby/PHP

**Remaining**:
- [ ] Firecracker integration (snapshot/restore for <125ms cold start)
- [ ] Direct RSS via cgroup2 `memory.max` enforcement (psutil is fallback)

### Phase 2: RSI Loop

**Goal**: Closed-loop harness improvement with convergence detection.

**Status**: Partially implemented in `rsi_loop.py`

**Delivered**:
- `RSILoop` class with convergence check
- `SelfImprovingHarness` with iteration history
- Rule-based improver with verdict-driven strategy
- LLM-based improver (`AnthropicLLMWrapper` in `llm_improver.py`) with retry/backoff and JSON parsing

**Remaining**:
- [ ] `ImprovementDelta` audit trail persisted to disk
- [ ] Multi-harness A/B evaluation infrastructure

**RSI Convergence Empirical Findings** (from `rsi_simulation.py`):
- **Threshold=0.01/window=2** is optimal for good harnesses (100% convergence, mean plateau iter=7.14)
- **Noise floor σ>0.08** causes convergence failure (80% at 0.08 noise, drops to 45% at 0.15)
- **Initial quality matters enormously**: start at 0.80 → 99% convergence in 3.7 iters; start at 0.20 → 5% convergence in 11+ iters
- **Use aggregate_score over pass_rate** for convergence detection (4.6x more stable, CV=0.029 vs CV=0.134)
- **False positive risk**: threshold=0.01 with window=3 on flat trajectories has high false-positive rate when noise ≈ threshold. Minimum recommended threshold is 0.02 with noise-adaptive fallback.
- **15 iterations is tight** for bad harnesses — 20-25 recommended for initial quality < 0.35

### Phase 3: Deer-Flow Integration

**Goal**: Autobench configures deer-flow supervisor topology.

**Status**: In Progress

**Delivered**:
- `SessionState` dataclass with session termination fields (`session_id`, `started_at`, `terminated_at`, `termination_reason`, `status`)
- `finish_session()` and `is_session_complete()` helpers for session lifecycle
- `ActivationPredicate` dataclass with operators: eq, ne, gt, lt, contains, regex
- `evaluate_predicate()` for context-based predicate evaluation
- `RoleSpecActivationBuilder` class for building RoleSpec dicts with `activation_predicate` fields
- Verdict-based routing predicates that route CE/RE to error-handler, TLE to timeout-handler, WA to debug-agent, OK to worker
- Test coverage in `role_predicate_test.py`
- `AutobenchResultPublisher` + `AutobenchResultSubscriber` in `signal_bus.py` with CloudEvents-lite schema
- `emit_signals: bool` parameter on `BenchmarkEvaluator` — lazily publishes results to nervous-bus

**Remaining**:
- [ ] `role_router.py` integration — wired `RoleSpecActivationBuilder` to deer-flow's adaptive model selection
- [ ] Middleware wiring — connect `BudgetViolationMiddleware` to session lifecycle
- [ ] `role_router.py` observation database integration

**Key Files**:
- Session state: `session_state.py`
- Role predicate: `role_predicate.py`
- Tests: `role_predicate_test.py`
- Deer-flow: external consumer (separate repo)
- Adapter: `nervous-bus/sdk/python/nbus.py` → `deer obs bus`

### Phase 4: Production

**Goal**: Multi-agent evaluation with Pareto optimization and collective judging.

**Status**: Blueprint

**Requirements**:
- [ ] Supervisor→workers→judges topology implemented
- [ ] 30+ simultaneous model evaluation
- [ ] Collective LLM-as-Judge (anonymous peer scoring)
- [ ] Pareto frontier tracking across weight space
- [ ] Curveball generation integrated into benchmark pipeline
- [ ] AAII metric correlation measurement (target: 87.08%)
- [ ] Contamination-free problem sourcing (LCB API or LLM generation)

---

## 12. Unknowns and Research Gaps

### High Priority

1. **cgroup-based memory enforcement**: DONE — psutil RSS tracking implemented (polls every 50ms, SIGKILL on limit). cgroup2 helper `run_in_cgroup()` written with fallback.

2. **gVisor/Firecracker integration**: PARTIAL — gVisor stub written (`_find_runsc`, `runsc run` wrap). Firecracker not yet started.

3. **LLM improver wiring**: PARTIAL — `AnthropicLLMWrapper` in `llm_improver.py` is complete and committed. Not yet wired as the default improver in `rsi_loop.py`. Needs `ANTHROPIC_API_KEY` env var or direct injection.

4. **Verdict signal bus**: DONE — `signal_bus.py` with `AutobenchResultPublisher` (CloudEvents-lite, `zellij pipe -p nervous-bus -n autobench.result`) and `AutobenchResultSubscriber`. Schema at `schemas/autobench.result.v1.json`.

5. **Session termination field**: DONE — `SessionState` dataclass in `session_state.py` with `finish_session()`, `is_session_complete()`. Commit `252e34e`.

### Medium Priority

6. **RSS tracking**: DONE — psutil RSS tracking implemented, stderr heuristic still as fallback.

7. **Pareto frontier persistence**: DONE — `pareto_frontier.py` with SQLite persistence, dominance checking, query_frontier, weight presets, AAII correlation.

8. **LCB API integration**: DONE — `codeforces_scraper.py` with `CodeForcesScraper` class. `fetch_all()` hits `/api/problemset.problems` (11186 problems), deduplicates by `cf_id`, maps rating→difficulty. Rate-limit aware (backoff on 429). Cache writes to `autobench/data/codeforces_problems.json`. `to_benchmark_cases()` converts to `BenchmarkCase` dicts. Schema at `schemas/codeforces_problem.v1.json`.

9. **Multi-model evaluation infrastructure**: DONE — `multi_model.py` with `ModelClient` Protocol, `ModelClientRegistry`, `ScoreNormalizer` (rank + zscore modes), `MultiModelBenchmark.run_sweep()` for parallel model evaluation, `ModelLeaderboard` for cross-model comparison. `AnthropicModelClient` and `DeepSeekModelClient` adapters included. Normalized scores plug into `ParetoFrontier.update_frontier()`.

10. **Collective judging**: DONE — `JudgingPool` class in `evaluator.py` (line ~447). Anonymous peer judging with majority verdict aggregation + mean for continuous scores. `calibrate_ensemble_size()` for variance-based ensemble sizing. 8 tests passing in `tests/test_judging.py`.

**Verdict noise floor (empirical from `noise_floor.py`)**:
- PHP: binary not installed → false positive for all PHP samples
- Rust: needs Cargo.toml in the sandbox directory — `rustc` works but `cargo` needs a project context. All Rust samples produce CE without a Cargo.toml.
- Go: parsing errors in the sandbox environment (missing imports parsed as syntax errors). Go compiler `go run` works but parsing `package main` requires proper module structure.
- Languages with clean false-positive rates: Python (OK), JavaScript (OK), Java (OK), C (OK), C++ (OK), Ruby (OK), Swift (OK), Kotlin (OK), Bash (OK)
- **Recommendation**: For Rust, use `rustc` directly instead of `cargo`. For Go, use `go run` with proper module. For PHP, install `php` binary or skip PHP testing.

### Research Questions

11. **Optimal RSI convergence threshold**: 0.01 is arbitrary. Empirical analysis of convergence curves needed.

12. **Weight space granularity**: How many weight configurations should be evaluated per benchmark run? 4 (current) vs N (exploration budget)?

13. **Harness transfer**: Poetiq showed harness optimized on Gemini improved GPT 5.5 by 4.3 percentage points. What is the cross-model transfer function? Is a "universal harness" feasible?

14. **Curveball density**: How many curveball cases per benchmark case? 20 (current) may be insufficient for adversarial robustness testing.

15. **Verdict noise floor**: What is the false-positive rate for RE detection? The `_is_runtime_error` patterns are heuristics; calibrated recall needed.

### Lower Priority / Deferred

16. **WebAssembly for JIT languages**: Research suggests WASM overhead outweighs benefit for compiled languages. Revisit for interpreted languages (Python, Ruby) if a clear benefit emerges.

17. **Snapshot/restore for Firecracker**: Firecracker's snapshot mechanism could enable <125ms cold boot. Not needed until Phase 3+.

18. **Monte Carlo rollout protocol**: `RolloutProtocol.MONTE_CARLO` is defined but not implemented. MCTS requires significant infrastructure.

---

## Appendix: File Index

| File | Purpose | Key Classes |
|------|---------|-------------|
| `autobench/core.py` | Core types | `HarnessConfig`, `HarnessResult`, `RSILoop`, `Verdict` |
| `autobench/evaluator.py` | Benchmark evaluation | `BenchmarkEvaluator`, `BenchmarkCase`, `BenchmarkResult` |
| `autobench/sandbox.py` | Sandboxed execution | `SandboxedExecutor`, `ExecutionResult`, `LANGUAGE_RUNNERS`, `run_in_cgroup` |
| `autobench/rsi_loop.py` | Self-improvement loop | `SelfImprovingHarness`, `improve_harness`, `ImprovementDelta` |
| `autobench/cli.py` | CLI commands | `cmd_run`, `cmd_improve`, `cmd_eval`, `cmd_scaffold` |
| `autobench/repo_analyzer.py` | Repo complexity analysis | `RepoAnalyzer`, `classify_change`, `detect_architecture_shift` |
| `autobench/test_scaffolder.py` | Test generation + curveballs | `CurveballGenerator`, `generate_scaffolding` |
| `autobench/llm_improver.py` | Anthropic LLM improver | `AnthropicLLMWrapper`, `LLMImprovementResult`, `VERDICT_STRATEGIES` |
| `autobench/signal_bus.py` | Nervous-bus integration | `AutobenchResultPublisher`, `AutobenchResultSubscriber`, `make_publisher`, `make_subscriber` |
| `autobench/signal_bus_test.py` | Signal bus tests | 19 tests |
| `autobench/session_state.py` | Session lifecycle | `SessionState` |
| `autobench/role_predicate.py` | RoleSpec predicates | `ActivationPredicate`, `RoleSpecActivationBuilder` |
| `autobench/role_predicate_test.py` | Predicate tests | 35 tests |
| `autobench/integration.py` | Deer-flow integration | `DeerFlowEvaluator`, `NervousBusPublisher`, `BudgetViolationMiddleware` |
| `autobench/__init__.py` | Package init | `__version__ = "0.1.0"` |

---

## Appendix: ASCII Reference Diagrams

### RSI Loop
```
┌─────────────────────────────────────────────┐
│                 RSI Loop                     │
│  g(C) = perf_metric(improver_agent(harness)) │
│                                              │
│  ┌─────────────┐    ┌─────────────────┐     │
│  │  harness    │───▶│ benchmark_fn()  │     │
│  │  config Cn   │    └────────┬────────┘     │
│  └─────────────┘             │              │
│        ▲                     ▼              │
│        │              ┌─────────────┐       │
│        │              │  results    │       │
│        │              └──────┬──────┘       │
│        │                     │              │
│        │              ┌──────▼──────┐       │
│        │              │ improver_fn │       │
│        │              └──────┬──────┘       │
│        │                     │              │
│        └─────────────────────┘              │
└─────────────────────────────────────────────┘
```

### Verdict Precedence
```
CE ──▶ TLE ──▶ MLE ──▶ RE ──▶ WA ──▶ OK
 │       │       │       │       │      │
 ▼       ▼       ▼       ▼       ▼      ▼
compile  time   memory  runtime  wrong  pass
error   exceeded exceeded error   answer
```

### Multi-Agent Topology
```
Supervisor
    ├──▶ Worker 1 ──▶ verdict
    ├──▶ Worker 2 ──▶ verdict
    ├──▶ ...
    └──▶ Worker N ──▶ verdict
          │
          └──▶ Judges Pool
                   ├── Judge A ──▶ scores Worker B
                   ├── Judge B ──▶ scores Worker A
                   └── Judge N ──▶ scores Worker M
```