# AutoBench 2.0 Research

*Status: Incomplete — key details (AAII formula, Pareto algorithm, dynamic generation specifics) could not be extracted from public sources. Evidence-based gaps noted throughout.*

---

## 1. AutoBench 2.0 Architecture

### Dynamic Problem Generation

AutoBench's core differentiating claim is **"resists gaming by changing at each run."** The engine writes "difficulty-balanced prompts" dynamically, preventing models from memorising benchmark solutions.

**What is known:**
- Problems are auto-generated rather than static
- A pool of 30+ LLM models generates "virtual agentic environments" for testing
- The system uses collective LLM judging: models anonymously evaluate each other's responses
- A weighting algorithm iteratively refines scores until the leaderboard stabilises

**What is NOT publicly documented:**
- The exact algorithm for generating fresh unseen problems per run
- How "difficulty-balanced" is determined and enforced
- Whether problems are drawn from a latent space, templated, or generated from scratch each run
- The prompting strategy used for problem generation

### Multi-Objective Scoring Formula

AutoBench tracks three primary metrics per model:

| Metric | Description |
|--------|-------------|
| **Quality Score** | Domain-specific Autobench score (higher is better) |
| **Average Cost** | Average cost per answer in US cents |
| **Average Latency** | Average answer duration in seconds |

**The SICA formula** (confirmed from local autobench codebase, `evaluator.py`):

```
U = w_score * avg_p_score + w_cost * avg_p_cost + w_time * avg_p_time
```

Where:
- `w_score = 0.5`, `w_cost = 0.25`, `w_time = 0.25` (default weights)
- `p_score` = 1.0 if OK verdict, 0.0 otherwise
- `p_cost` = normalised cost utility (1.0 = cheapest)
- `p_time` = normalised time utility: `p_time = max(0, min(1, 1.0 - runtime/max_time))`
- Final `U` is clamped to `[0.0, 1.0]`

**Critical note:** The SICA formula above is from the local `nervous-bus/autobench/` implementation, which may or may not match AutoBench 2.0's live production scoring. AutoBench's public leaderboard shows a single composite "Score" (e.g., 3.30 for Claude Opus 4.7) — the exact normalisation and aggregation is not publicly documented.

---

## 2. AAII (Area Above Ideal) Metric

### Definition

AAII stands for **Area Above the Ideal** — a metric borrowed from multi-criteria decision analysis that measures how far a solution lies from the ideal (perfect quality at zero cost and zero latency).

### Computation

In the autobench codebase, the scoring uses a weighted-sum approach:
```
U = w_score * avg_p_score + w_cost * (1 - avg_p_cost) + w_time * (1 - avg_p_time)
```

AAII in the general multi-objective optimisation sense is computed as the integral of the hypervolume between the Pareto frontier and the ideal point (origin in cost/time space, maximum in quality space). However, **the exact AAII formula used by AutoBench's production system is not publicly documented.**

### Practical Interpretation: 87% Correlation

AutoBench reports **87.08% correlation with AAII** (and 77.16% with LMArena, 80.68% with MMLU-Plus). This means:

- ~87% of the variance in AutoBench scores is explained by/aligned with the AAII metric
- Practically: AutoBench rankings are broadly consistent with a metric that rewards solutions closest to the ideal quality/cost/speed trade-off surface
- The remaining ~13% of variance captures something AutoBench measures that pure AAII does not (agentic capabilities, dynamic problem difficulty, collective judging effects)

**To calculate AAII for a benchmark:**
1. For each model i, compute `(quality_i, cost_i, latency_i)`
2. Normalise all three dimensions to [0, 1] where higher is better
3. The "ideal" point is `(1, 1, 1)` — perfect quality at zero cost and zero latency
4. The "worst" point defines the search space boundary
5. AAII = hypervolume of the region dominated by the Pareto frontier but above the ideal point (lower AAII = closer to ideal)
6. Alternatively, AAII = sum over all models of `(1 - normalized_cost) * (1 - normalized_time) * normalized_quality`

---

## 3. Leaderboard Data: April 2026

*Source: autobench.org leaderboard (32 ranked models)*

| Rank | Model | Score | Avg Cost (cents) | Avg Latency (s) | P99 Latency (s) | Iterations |
|------|-------|-------|-----------------|-----------------|-----------------|------------|
| 1 | Claude Opus 4.7 | 3.30 | 2.72 | 21 | — | — |
| 2 | Claude Opus 4.6 | 3.24 | 2.58 | 38 | — | — |
| 3 | Gemini 3.1 Pro Preview | 3.21 | 1.33 | 26 | — | — |
| 4 | Claude Sonnet 4.6 | 3.16 | 1.98 | 47 | — | — |
| 5 | GLM 5.1 | 3.15 | 0.51 | 60 | — | — |
| 6 | GPT-5.4 (xhigh) | 3.13 | — | — | — | — |
| 7 | Mimo V2 Pro | 3.10 | — | — | — | — |
| 8 | Qwen3.6 Plus | 3.07 | — | — | — | — |
| 9 | Kimi K2.5 | 3.02 | — | — | — | — |
| 10 | MiniMax M2.7 | 3.01 | 0.10 | — | — | — |
| 11 | Grok 4.20 | 3.00 | — | — | — | — |
| 12 | Claude haiku 4.5 | 2.99 | — | — | — | — |
| 13 | Gemini 3 Flash Preview | 2.98 | — | — | — | — |
| 14 | GLM 4.7 | 2.92 | — | — | — | — |
| 15 | GPT-5.4 Mini (xhigh) | 2.91 | — | — | — | — |
| 16 | Grok 4.1 fast | 2.84 | — | — | — | — |
| 17 | Qwen3.5 122B A10B | 2.84 | — | — | — | — |
| 18 | Qwen3.5 35B A3B | 2.82 | — | — | — | — |
| 19 | Gemini 3.1 Flash Lite Preview | 2.82 | — | — | — | — |
| 20 | Nemotron 3 Super 120B A12B | 2.80 | — | — | — | — |
| 21 | Gemma 4 31B IT | 2.79 | — | — | — | — |
| 22 | MiniMax M2.5 | 2.79 | — | — | — | — |
| 23 | GPT-5.4 Nano (xhigh) | 2.78 | — | — | — | — |
| 24 | GPT oss 120b | 2.76 | — | — | — | — |
| 25 | Nemotron 3 Nano 30B A3B | 2.71 | — | — | — | — |
| 26 | Mistral Small 4 | 2.69 | — | — | — | — |
| 27 | Nova 2 lite v1 | 2.66 | — | — | — | — |
| 28 | GPT oss 20b | 2.65 | — | — | — | — |
| 29 | Deepseek v3.2 | 2.64 | — | — | — | — |
| 30 | Mistral large 2512 | 2.62 | — | — | — | — |
| 31 | Gemma 4 26B A4B IT | 2.61 | — | — | — | — |
| 32 | Llama 4 Maverick | 2.27 | — | — | — | — |

**Notable:** MiniMax M2.7 at rank #10 scores 3.01 with an exceptionally low cost of **0.10 cents** per answer.

**Scale:** 32 ranked models, 16 ranking models, 320k ranks, 10,512 iterations, 336k answers, ~300M generated tokens, ~8B evaluated tokens.

---

## 4. 30+ Model Testing Infrastructure

**What is known:**
- AutoBench uses "30+ LLM models to generate virtual agentic environments"
- The 4-step process: (1) Submit Models, (2) Run Benchmarks, (3) Collect Metrics via anonymous peer judging + weighting algorithm, (4) Analyse Results

**Architecture gaps in public documentation:**
- How parallel evaluation is actually orchestrated (batch processing, queue workers, etc.)
- Whether models are evaluated concurrently or sequentially
- Infrastructure specifics (GPU fleet, API rate limiting, etc.)
- The anonymous peer judging mechanism (how anonymity is enforced)

The local autobench codebase includes:
- `SandboxedExecutor` for code execution
- Verdict detection (CE/RE/TLE/MLE/WA/OK)
- gVisor/Firecracker/WASM isolation options
- Sub-second compilation strategies for Python/Go/Rust/C++

---

## 5. Pareto Frontier Computation

**Pareto-optimal solutions** are those where no other solution dominates in all three dimensions (quality, cost, speed). A solution dominates another if it is better or equal in all dimensions and strictly better in at least one.

**AutoBench Pareto frontier:** Not publicly documented. The local autobench codebase does **not** contain a Pareto frontier computation — the SICA formula is a simple weighted sum, which does not produce Pareto-optimal solutions in the multi-objective sense.

The production AutoBench leaderboard likely uses the SICA-weighted score for ranking, which may or may not correspond to actual Pareto frontier identification.

---

## 6. Anti-Gaming Techniques

AutoBench claims "resists gaming by changing at each run." Specific techniques confirmed:

| Technique | Description |
|-----------|-------------|
| **Dynamic prompt generation** | New difficulty-balanced prompts generated each run — not from a fixed problem bank |
| **Collective judging** | Multiple models judge responses anonymously, preventing single-model manipulation |
| **Iterative weight refinement** | Scoring weights stabilise across peer judgements, making it harder to game via single submissions |
| **Domain flexibility** | Custom domain selection prevents针对性 overfitting to specific problem categories |

**Gaming vectors that remain理论上 possible:**
- Overfitting to the judging model pool's preferences
- Exploiting the weighting algorithm's convergence properties
- Specialising to "difficulty-balanced" problem characteristics

---

## 7. AutoBench vs LiveCodeBench

**AutoBench does NOT use LiveCodeBench problems.** Evidence from local autobench research doc (`sandbox_environments.md`):

> LiveCodeBench: Code is compiled/run with strict constraints — Time limit enforced via `ulimit -t` or `timeout`, Memory limit via `ulimit -m`, verification against test cases.

LiveCodeBench is referenced as a **separate existing system** in the autobench research doc, not as a data source. The key distinction:

| Aspect | AutoBench | LiveCodeBench |
|--------|-----------|---------------|
| Problems | Dynamically generated | Static from known problem sets |
| Judging | Collective LLM-as-judge | Test case execution |
| Anti-gaming | Changes each run | Fixed problems (can memorise) |
| Scope | Quality/cost/speed multi-objective | Code execution only |
| Agentic tasks | Yes (virtual environments) | Limited |

---

## 8. Agentic Benchmarking Specifics

What makes AutoBench "agentic" vs traditional static benchmarks:

**Agentic capabilities tested (inferred from "virtual agentic environments" framing):**
- Multi-step task completion requiring planning
- Tool use and function calling
- Context management across extended conversations
- Iterative refinement based on feedback
- Self-revision protocols

**From the local autobench codebase (`core.py`):**
- `RolloutProtocol` enum: `SINGLE`, `ITERATIVE`, `SELF_REVISION`, `MONTE_CARLO`
- `ContextManager` enum: `FULL`, `BUDGETED`, `SEMANTIC`, `HIERARCHICAL`
- `RSILoop`: Recursive self-improvement loop `g(C) = perf_metric(improver_agent(harness_v1))`

**What static benchmarks (MMLU, HumanEval, etc.) do NOT test:**
- Sustained multi-turn interactions
- Adaptive problem solving (模型 can askclarifying questions)
- Real-time tool integration
- Context window management under load
- Cost-efficiency trade-off decisions

---

## Key Gaps and Unknowns

1. **Exact dynamic generation algorithm** — how are fresh problems created? What is the latent space?
2. **AAII formula** — exact computation not publicly documented
3. **Pareto frontier algorithm** — may not be implemented as true multi-objective optimisation
4. **30+ model parallel infrastructure** — orchestration details not public
5. **LiveCodeBench integration** — confirmed separate; no problem sharing
6. **Anonymous peer judging mechanism** — how anonymity is enforced technically

---

## Sources

- [AutoBench (autobench.org)](https://autobench.org) — leaderboard, methodology
- [AutoBench Leaderboard on HuggingFace Spaces](https://huggingface.co/spaces/AutoBench-Leaderboard/AutoBench-Leaderboard)
- This repository — `core.py`, `evaluator.py`, `engines/sandbox.py`, `research/sandbox_environments.md`