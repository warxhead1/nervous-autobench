# nervous-autobench

FunSearch-style island-model evolution harness for GPU kernel research, paired with an RSI (recursive self-improvement) evaluation loop for coding agents.

Lives as a git submodule of [nervous-bus](https://github.com/warxhead1/nervous-bus) and publishes ~30 event types to the bus covering iteration progress, sandbox verdicts, evolved kernel candidates, and plateau/reset signals.

---

## What it does

**Kernel evolution:** Given a target domain (SDF geometry, fluid SPH kernel, Allen-Cahn PDE reaction, terrain generation, TSP heuristic), runs island-model LLM evolution to discover programs that score well on a domain-specific oracle. Multiple islands evolve in parallel; plateau detection triggers diversity resets.

**RSI eval loop:** Benchmarks coding agent harness configurations against a suite of competitive programming and refactoring problems, then uses a supervisor to propose configuration improvements — feeding scored results back in for the next iteration.

**Bus integration:** Every generation, island reset, candidate evaluation, and run completion fires a typed event to nervous-bus. The `autobench-pulse` dashboard (`adapters/dashboard/autobench-pulse/` in the parent repo) visualises these live.

---

## Project structure

The `autobench` package is organised into 9 subpackages, each handling a distinct concern. The 8 kernel subpackages (`tsp_kernel/`, `sdf_kernel/`, `latent_kernel/`, `phase_kernel/`, `sph_kernel/`, `terrain_kernel/`, `thermal_kernel/`, `noise_kernel/`) stay at the package root because they are registered into the kernel registry and ship their own CLI entry points (`python -m <name>_kernel ...`).

| Subpackage | Responsibility |
|---|---|
| `kernels/` | FunSearch evolution loop, `KernelConfig`, sandbox bridge — the 8 kernel subpackages re-export from here |
| `engines/` | `SandboxedExecutor`, Firecracker VM pool, guest agent, shader executor, SDF tracer |
| `bus/` | CloudEvents-lite envelope (`build_event`), idgen (`ulid`/`iso_now`), signal + GPU publishers, nervous-bus integration |
| `llm/` | LLM wrappers (Anthropic, MiniMax), ensemble + worker, model registry, judge, shared `base.py` helpers |
| `rsi/` | Recursive self-improvement: loop, simulation, adversarial, population, replay |
| `evaluation/` | `BenchmarkEvaluator` (still at the package root as the public entry point), registry, assembly, curriculum, distillation, diversity, codeforces, noise_floor |
| `daemons/` | Long-running supervisor (`continuous.py`) and event-driven trigger daemon |
| `audit/` | Claim verification, refactor verifier, budget guard, oracle calibration, post-run assessment, AHE, session state, role predicate, invalidation, repo analyzer |

Back-compat shims live at the old root import paths (e.g. `from autobench.sandbox import ...` continues to work) but the canonical import is the subpackage path: `from autobench.engines.sandbox import ...`.

---

## Kernel domains

| Domain | Oracle | Notes |
|---|---|---|
| `sdf_kernel` | MSE + eikonal penalty `exp(-0.5·mean_grad_err)` | Gyroid, round-box, warped-sphere instances; single-instance focus avoids multi-basin collapse |
| `tsp_kernel` | Tour ratio vs known optimum | berlin52, eil101, kroA100; diversity + normalization |
| `phase_kernel` | Allen-Cahn interface width + front velocity | Water→ice freezing; gen16 fit=1.000 |
| `latent_kernel` | Phase + latent heat coupling | Stefan problem; oracle fix: `target_width=5`, velocity score |
| `sph_kernel` | Density reconstruction MSE on jittered particles | Gen31: `sigma*(1-q²)⁵`, non-standard normalisation |
| `terrain_kernel` | Valley coherence + elevation statistics | river, rolling hills, mountain, volcanic, badlands instances |
| `noise_kernel` | RAPS spectral slope | Single-instance; SSIM multi-instance hits arithmetic ceiling |
| `thermal_kernel` | 2D Allen-Cahn + fixed-T equilibrium | Coupled heat diffusion |

---

## Quick start

```bash
# Run one evolution pass (TSP, 10 generations)
python -m tsp_kernel --generations 10

# Run SDF evolution with eikonal penalty
python -m sdf_kernel --instances gyroid --generations 20

# Live dashboard (requires nervous-bus + redis-mirror running)
python -m pulse_app --prefer-bus

# Offline replay
python -m pulse_app --debug-file ~/.cache/nervous-bus/debug.jsonl
```

Set `AUTOBENCH_BENCH_DIR` to point at your benchmark results directory (default: `~/.cache/nervous-bus/benchmarks/`).

---

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `AUTOBENCH_BENCH_DIR` | `~/.cache/nervous-bus/benchmarks/` | Benchmark result output |
| `ANTHROPIC_API_KEY` | — | Required for LLM generation |
| `MINIMAX_API_KEY` | — | Optional adversarial judge (MiniMax API) |
| `NERVOUS_HOME` | `~/.config/nervous-bus/` | Schema overlay + adapter config |

---

## Architecture

See `ARCHITECTURE.md` for the full harness design. Key files:

| File | Role |
|---|---|
| `kernels/base.py` | Island-model loop, plateau detection, bus publishing, consolidated prior |
| `evaluator.py` | SICA scoring (score/cost/time), AHE prediction tracking |
| `engines/sandbox.py` | Firecracker VM + gVisor sandbox for untrusted code execution |
| `engines/firecracker_vm.py` | Firecracker VM pool (vsock transport, port 8888) |
| `rsi/loop.py` | RSI supervisor loop — proposes harness config improvements |
| `composition_oracle/` | Joint terrain+phase oracle for TEngine readiness gating |

---

## Bus events emitted

All events follow CloudEvents-lite envelopes. Schemas live in the parent `nervous-bus/schemas/` directory (schema-first rule — schemas precede producers).

```
autobench.iteration.v1          — RSI iteration start/complete + aggregate score
autobench.sandbox.v1            — per-case sandbox verdict + latency
autobench.improver.v1           — LLM improver call boundaries
autobench.phase.v1              — phase boundary (benchmark/improver/judge)
kernel.candidate.evaluated.v1   — per-candidate fitness score
kernel.generation.completed.v1  — generation summary across all islands
kernel.island_reset.v1          — diversity reset triggered
kernel.plateau_hint.v1          — plateau detection signal
```

---

## Security

This repo uses [gitleaks](https://github.com/gitleaks/gitleaks) for secret scanning.

```bash
# Scan working tree
gitleaks detect --source . --config .gitleaks.toml

# Scan staged changes (pre-commit)
gitleaks protect --staged --config .gitleaks.toml
```

The pre-commit hook is installed automatically when you run `nervous setup --install-hooks` from the parent nervous-bus repo.

---

## License

MIT
