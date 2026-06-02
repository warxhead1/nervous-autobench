# Autobench Architecture

**Version:** 0.1.0 · **Status:** Active

Autobench is a self-improving harness with two cooperating loops: a **FunSearch
island-model evolver** that discovers GPU/compute kernels, and an **RSI
(recursive self-improvement) loop** that benchmarks coding-agent harness
configurations and proposes improvements. Both loops share one sandbox executor,
one multi-objective evaluator, and one event substrate.

---

## 1. System Overview

Autobench lives as a git submodule of [nervous-bus](https://github.com/warxhead1/nervous-bus)
and is the ecosystem's **evaluation substrate**: it generates problems, runs code
against them, collects verdict-level signals (CE/RE/TLE/MLE/WA/OK), scores them on
quality/cost/speed, and feeds the results back into the next iteration.

Every generation, island reset, candidate evaluation, sandbox verdict, and run
completion fires a typed CloudEvents-lite event to nervous-bus. deer-flow consumes
those events; the `autobench-pulse` dashboard (in the parent repo) visualises them
live. The bus integration is fire-and-forget and never blocks the compute path.

```
            ┌──────────────────────────────────────────────┐
            │                  Autobench                    │
problem ──▶ │  evolver (kernels)  ┐                         │
            │  RSI loop (rsi)     ┴─▶ sandbox ─▶ evaluator  │ ──▶ verdicts + scores
            │                              │         │      │
            └──────────────────────────────┼─────────┼──────┘
                                           ▼         ▼
                                    nervous-bus  (consumed by deer-flow)
```

### Key properties

- **Contamination-free** — kernel oracles and benchmark problems are generated /
  scored fresh; there is no fixed answer key to memorise.
- **Multi-objective** — every result is scored on quality, cost, and time at once
  (SICA utility), not a single pass/fail.
- **Verdict-driven** — improvement strategy is selected from the *distribution* of
  verdicts, not aggregate score alone.
- **Polyglot sandbox** — one executor compiles and runs 14 languages with a single
  verdict-detection precedence shared by both loops.

---

## 2. Package Layout

The `autobench` package is organised into shared subpackages plus eight kernel
packages. Everything imports through its subpackage path
(`from autobench.engines.sandbox import ...`, `from autobench.rsi.loop import ...`);
the old flat top-level modules were removed in favour of this layout.

| Subpackage | Responsibility | Key surface |
|---|---|---|
| `kernels/` | FunSearch base + registry + shared loop machinery | `base.py` (`FunSearchKernel`, `@register_kernel`, `KERNEL_REGISTRY`), `cli.py`, `config.py`, `bridge.py`, `sandbox.py` |
| `engines/` | Code execution | `sandbox.py` (`SandboxedExecutor`), `firecracker_vm.py`, `guest_agent.py`, `shader_executor.py`, `sdf_tracer.py` |
| `evaluation/` | Benchmark assembly + datasets | `assembly.py`, `registry.py`, `curriculum.py`, `diversity.py`, `distillation.py`, `codeforces.py`, `noise_floor.py` |
| `evaluator/` | Scoring + judging | `engine.py` (`BenchmarkEvaluator`), `judging.py` (`JudgingPool`), `types.py` (`BenchmarkCase`, `BenchmarkResult`) |
| `rsi/` | Self-improvement | `loop.py` (`SelfImprovingHarness`, `ImprovementDelta`), `simulation.py`, `adversarial.py`, `population.py`, `replay.py` |
| `llm/` | Model wrappers | `anthropic.py`, `minimax.py`, `ensemble.py`, `worker.py`, `judge.py`, `models.py`, `base.py` |
| `bus/` | Event substrate | `envelope.py` (`build_event`), `idgen.py` (`ulid`, `iso_now`), `signal_bus.py`, `gpu_publisher.py`, `integration.py` |
| `observability/` | Run telemetry | `core.py` (`AutobenchObservability`), `channels.py`, `events.py`, `_util.py` |
| `daemons/` | Long-running drivers | `continuous.py` (supervisor), `trigger_daemon.py` |
| `audit/` | Verification + lifecycle | `claims_audit.py`, `refactor_verifier.py`, `budget_guard.py`, `oracle_calibration.py`, `post_run_assess.py`, `ahe.py`, `session_state.py`, `role_predicate.py`, `repo_analyzer.py` |

Core dataclasses (`HarnessConfig`, `HarnessResult`, `Verdict`, `RolloutProtocol`,
`ContextManager`) live in the top-level `core.py`.

---

## 3. The Kernel Suite (FunSearch Evolution)

Each kernel is a small package implementing one domain. They all subclass
`FunSearchKernel` (`kernels/base.py`) and register themselves with
`@register_kernel("<name>")`, so any kernel is reachable from the unified
dispatcher (`kernels/cli.py`) or its own entry point (`python -m <name>_kernel`).

A kernel package splits into:

- `instance.py` — the problem instances / fixtures for the domain
- `scoring.py` (or `topology.py` for sdf, `spectral.py` for noise) — the oracle's
  numeric scoring helpers
- `oracle.py` — the domain oracle that maps a candidate program to a fitness score
- `loop.py` — the registered `FunSearchKernel` subclass (the public class)
- `__init__.py` — re-exports the public surface

| Kernel (`@register_kernel`) | Domain | Oracle |
|---|---|---|
| `sdf` (`sdf_kernel/`) | Signed-distance geometry | MSE + eikonal gradient penalty; gyroid / round-box / warped-sphere instances |
| `tsp` (`tsp_kernel/`) | TSP heuristics | Tour length ratio vs known optimum (berlin52, eil101, kroA100) |
| `noise` (`noise_kernel/`) | Procedural noise | RAPS spectral-slope match |
| `sph` (`sph_kernel/`) | SPH fluid kernel | Density-reconstruction MSE on jittered particles |
| `phase` (`phase_kernel/`) | Allen-Cahn PDE | Interface width + front velocity (water→ice) |
| `terrain` (`terrain_kernel/`) | Terrain generation | Valley coherence + elevation statistics |
| `thermal` (`thermal_kernel/`) | 2D Allen-Cahn + heat | Fixed-temperature equilibrium |
| `latent` (`latent_kernel/`) | Stefan problem | Phase + latent-heat coupling |

### Evolution loop (island model)

`FunSearchKernel` runs a generational island model. Each `Island` holds a
population of `CandidateProgram`s; per generation the loop samples exemplars,
prompts an LLM to mutate them, sandboxes the result, and scores it through the
oracle. Plateau detection triggers a diversity reset (drop/repopulate the worst
island) so a stuck island can't dominate. Best programs migrate across islands.

```
generation:
  for each island:
    exemplars = sample(population)            # few-shot context
    child     = llm.mutate(exemplars)         # candidate program
    result    = sandbox.run(child)            # compile + execute
    fitness   = oracle.score(result)          # domain-specific
    population.insert(child, fitness)
  if plateau(best): island.reset()            # diversity injection
  migrate_best_across_islands()
```

---

## 4. The RSI Loop

The RSI loop improves a *harness configuration* — the configurable environment an
agent runs in (`HarnessConfig`: system prompt, rollout protocol, context manager,
tool surface, verifiers, budget). It implements
`g(C) = perf_metric(improver(C))`: benchmark the current config, propose a new
one, repeat until convergence.

```
for i in range(max_iterations):
    results  = evaluator.run(harness)          # list[HarnessResult]
    harness  = improver(harness, results)      # next config
    if converged(history): break               # plateau over recent window
```

Convergence is declared when recent iterations stop improving past a threshold
(empirically ~0.02 with a noise-adaptive fallback; aggregate score is more stable
than raw pass-rate). Two improver paths exist: a **rule-based** improver that reads
the verdict distribution and adjusts budget / rollout / context accordingly, and an
**LLM-based** improver that emits targeted edits as a structured `ImprovementDelta`
for auditability.

| Dominant verdict | Action |
|---|---|
| CE high | reduce `max_tokens` |
| TLE high | tighten time limit, switch to ITERATIVE rollout |
| WA high | escalate context manager to HIERARCHICAL |
| RE high | append error-handling guidance to tool surface |
| OK ≥ 90% | minor refinement; loosen budget slightly |

---

## 5. Sandbox Execution Model

`engines/sandbox.py` (`SandboxedExecutor`) compiles and runs code across 14
languages (python, rust, go, javascript, typescript, java, c, cpp, ruby, bash, php,
swift, kotlin, zig). It selects an isolation tier by availability, preferring
stronger isolation but always falling back so a run never fails for lack of a
sandbox:

```
gVisor (syscall filter)  ─▶  Firecracker (microVM snapshot)
        ─▶  namespace (unshare)  ─▶  raw subprocess (last resort)
```

WebAssembly is intentionally not used for compiled languages — the compile
overhead outweighs the isolation benefit. Memory limits are enforced via cgroup2
where available, with psutil RSS polling (SIGKILL on breach) as the fallback.

### Verdict precedence

Every execution resolves to exactly one verdict, first match wins:

```
CE  ─▶  TLE  ─▶  MLE  ─▶  RE  ─▶  WA  ─▶  OK
compile  time   memory  runtime  wrong  pass
error    exceeded exceeded error  answer
```

CE comes from compiler/parse errors in stderr; TLE from wall-clock exceeding the
case limit; MLE from OOM indicators; RE from per-language runtime-error patterns
(panics, tracebacks, segfaults); WA from a clean exit with mismatched output.

---

## 6. Evaluation and Scoring

`evaluator/engine.py` (`BenchmarkEvaluator`) runs each `BenchmarkCase` through the
sandbox, collects per-case `HarnessResult`s, and aggregates a `BenchmarkResult`
with verdict counts and a multi-objective utility score.

Scoring follows the SICA utility (arXiv:2504.15228): a weighted blend of normalised
quality, cost, and time, with default weights 0.5 / 0.25 / 0.25 normalised to sum to
1.0. Weights can be swept (balanced, cost-focused, quality-first, speed-first) to
trace a Pareto surface rather than commit to a single operating point.

**Pareto optimisation** keeps the set of configs that no other config beats on all
three axes simultaneously — when a new config arrives, anything it dominates is
dropped and it is added only if nothing already dominates it. This yields a frontier
of trade-offs instead of one "best" config.

**Judging.** `evaluator/judging.py` (`JudgingPool`) supports collective
LLM-as-judge: multiple models score outputs anonymously and verdicts are aggregated
by majority (continuous scores by mean). The pool can size its ensemble from
observed score variance. Anonymity and per-run rotation resist gaming.

---

## 7. Adversarial Inputs (Curveballs)

The curveball generator produces adversarial cases to stress edge-case handling
across four categories: **boundary** (empty/unicode/min-max/huge inputs),
**adversarial input** (injection strings, malformed JSON, null bytes, unicode DoS),
**race condition** (rapid-fire and shared-state cases), and **resource exhaustion**
(deep recursion, massive inputs, exponential structures). Each case carries an
expected behaviour (pass / timeout / crash / wrong-answer) so the evaluator can
score robustness directly.

---

## 8. Bus Integration

Producers build a CloudEvents-lite envelope via `bus/envelope.py` (`build_event`,
ULID ids, RFC3339 UTC timestamps) and publish through nervous-bus's shell SDK;
`observability/` owns the non-blocking per-run emitter. Autobench publishes ~30
event types (iteration progress, sandbox verdicts, evolved-kernel candidates,
plateau/reset signals, run completion). **deer-flow consumes these events** as a
bus subscriber; autobench does not call into deer-flow directly.

---

## Open questions

- Optimal RSI convergence threshold and weight-space granularity are still
  empirical; defaults are conservative.
- Cross-model harness transfer (does a config tuned on one model help another?) is
  unmeasured.
- Curveball density per case and calibrated RE false-positive rates need more data.
