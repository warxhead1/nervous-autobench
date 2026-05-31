# Frontier of Agent-Harness Optimization, H1 2026 — Research Report

**Target**: Identify techniques beyond autobench's current rule-based + LLM-driven RSI loop (post-SICA, post-AutoBench 2.0, post-Poetiq) that could push autobench from mechanical RSI into something genuinely creative. Output is meant to feed the orchestrator's next adoption decisions.

**Mode**: WebSearch + WebFetch primary (10 targeted queries across May 2026), with one Tier-2 `deer cycle` fired in parallel as a sanity check. Deer cycle `20260516T055935Z-8df3` (strategic class, free-tier model, 240 s timeout) was still in `pending` state at write time; cycle `20260516T052945Z-ceee` from earlier in the day already terminated `lift_measured` on a related framing. If `8df3` materially diverges from this report when it lands, append as §11 cycle artifact.

**TL;DR — top 3 adopt-next, ordered by leverage**:

1. **Adopt the AHE "decision observability" pattern.** Every harness edit is paired with a self-declared, machine-checkable prediction about the next round's task outcomes. The prediction is verified by the harness against actual results — turning every iteration into a falsifiable contract instead of a stochastic ratchet. This is the single highest-leverage missing piece in autobench's current RSI loop, and it's the trick that lifted AHE from 69.7% to 77.0% pass@1 on Terminal-Bench 2 in only 10 iterations ([arXiv:2604.25850](https://arxiv.org/abs/2604.25850)). It plugs into our existing improver step without architectural surgery.
2. **Bolt on the HGM "Clade-Metaproductivity" selection rule.** Stop picking the best-scoring agent as the next meta-agent. Pick the agent whose *descendants in the archive* have the best aggregate score, estimated via Thompson sampling. This explicitly optimizes for "good at self-improving" rather than "good at the benchmark," which directly attacks the metaproductivity-performance mismatch we've been seeing — and on SWE-bench Verified, HGM matches human-engineered coding agents with fewer CPU hours than DGM ([arXiv:2510.21614](https://arxiv.org/abs/2510.21614)).
3. **Add a third search arm: Bayesian / evolutionary parametric search, separate from the LLM improver.** The HARBOR paper ([arXiv:2604.20938](https://arxiv.org/pdf/2604.20938)) and "The Last Harness You'll Ever Build" ([arXiv:2604.21003](https://arxiv.org/pdf/2604.21003)) both show that pure Bayesian / evolution-agent search over the parametric surface (temperatures, retry budgets, retrieval k, max-turns) finds wins the rule-based heuristics never propose and the LLM improver rarely tries. Run it as a *parallel* arm to our existing improver, then have a meta-judge pick between proposals.

---

## 1. Executive Summary

Three structural shifts have happened in agent-harness research between summer 2025 (SICA) and May 2026:

- **Harness optimization has split into three orthogonal substrates.** Code-level edits (SICA, DGM, HGM, AHE), parametric search (HARBOR, Meta-Harness, Artemis), and prompt-only optimization (Bayesian Prompt Optimization, ACE, TF-GRPO). State-of-the-art runs *combine* them. Autobench currently lives only in the code-level lane.
- **The serious systems all have observability as a first-class architectural pillar, not an afterthought.** AHE's three pillars (component observability, experience observability, decision observability) are now the closest thing to a canonical recipe. SICA's overseer LLM was a sketch; AHE finishes the picture.
- **Diversity preservation is no longer optional.** R-Diverse showed that "self-play that looks diverse" can collapse into surface-level variation that doesn't transfer. Memory-Augmented Penalty (MAP) and Skill-Aware Measurement (SAM) are the techniques that beat R-Zero ([arXiv:2602.13103](https://arxiv.org/html/2602.13103v1)). The Choice-of-Divergence work ([arXiv:2509.07430](https://arxiv.org/html/2509.07430v1)) is the second canonical reference — it picks divergence functions in RL-with-verifiable-rewards to prevent the same collapse.

Beyond those three, three lower-priority signals worth tracking:

- **Anthropic's Managed Agents API** ([anthropic.com/engineering/managed-agents](https://www.anthropic.com/engineering/managed-agents)) is the production reference for the harness-outside-the-container design — decouples brain, hands, and session into a pre-built configurable harness. We already do most of this, but the "session as event log treated as a tool interface" idea is worth borrowing.
- **DeepMind's SIMA 2 self-play loop** ([deepmind.google blog](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/), [arXiv:2512.04797](https://arxiv.org/abs/2512.04797)) uses *one Gemini to generate tasks, a separate reward model to score, and the agent learns from its own attempts.* This is the curriculum-agent + executor-agent + judge triple that Agent0 also independently arrived at ([arXiv:2511.16043](https://arxiv.org/abs/2511.16043)). If autobench had a curriculum-generator alongside the improver, RSI loops would scale to harder problems without manual seed expansion.
- **Computer-use agents are *not* transferring much yet** — the architecture is structurally similar (planner + tool calls + verifier loop), but the failure modes are different enough that the techniques have not generalized. Skip this angle for now.

The rest of the report unpacks each item with sources and applicability scores (1–5) tuned to autobench's substrate.

---

## 2. Anthropic Research Relevant to Autobench

Anthropic's H2 2025 / H1 2026 public output skews to engineering blog posts rather than papers, but the substantive ideas are non-trivial.

### 2.1 Managed Agents — "decouple brain from hands and session"

The [Managed Agents engineering post](https://www.anthropic.com/engineering/managed-agents) is the most architecturally interesting Anthropic publication for our purposes. Three claims to internalize:

1. **Harness lives outside the container.** "The harness now sits outside the container, treating it like any other tool interface." For autobench this means: keep the shuttle and the improver decoupled at the API edge. We already do this; the post is implicit validation.
2. **Session is an event log treated as a tool interface.** Sub-agents don't share scratchpads — they share an *append-only event log* that any sub-agent can query. This is the harness-side mirror of what nervous-bus does at the inter-project layer. Applicability to autobench's iteration archive is direct: an event-log session abstraction means the improver can query "what did the last 5 iterations actually do" in a structured way without ingesting raw logs.
3. **Three-substrate decoupling lets you swap any one in isolation.** Brain (Claude+harness), hands (sandbox+tools), session (event log). This is the architectural argument for why HARBOR's parametric search can be bolted onto autobench's LLM-improver loop *without* an integration that breaks anything — they touch different substrates.

**Applicability**: 4/5. We already do most of this in spirit. The event-log-as-tool refinement is concretely worth lifting.

### 2.2 Multi-Agent Research System — orchestrator/worker pattern

The [multi-agent research system post](https://www.anthropic.com/engineering/multi-agent-research-system) reports that Claude Opus 4 (lead) + Claude Sonnet 4 (subagents) outperforms single-agent Opus 4 by **90.2%** on internal research eval, and that **token usage alone explains 80% of performance variance.**

The numbers matter less than the architectural recipe:
- Lead agent spawns **3–5 subagents simultaneously** (not one-at-a-time).
- Each subagent has its own context window; they explore in parallel before condensing.
- Parallel tool calling is exploited at *two* levels: lead spawns subagents in parallel, subagents call tools in parallel.

For autobench RSI: this is the recipe for a multi-improver phase. Right now we have one improver per iteration. The Anthropic data suggests **spawning 3–5 specialist improvers in parallel** (e.g., "tool-edit improver," "prompt improver," "retrieval improver," "context-condensation improver") and having a lead judge pick — would likely yield substantial lift just from the exploration breadth, regardless of whether the specialists are better than the generalist individually.

**Applicability**: 5/5. This is a near-drop-in upgrade to autobench's improver phase. Estimated 1–2 day implementation.

### 2.3 Advanced Tool Use — Programmatic Tool Calling

The [advanced tool use post](https://www.anthropic.com/engineering/advanced-tool-use) introduced *Programmatic Tool Calling*: instead of round-tripping each tool call back to the LLM, Claude writes code that calls multiple tools and processes outputs, controlling what re-enters the context window. This is a context-window-pressure-relief technique.

For autobench specifically: trajectories burn tokens fast. If the shuttle agent can use Programmatic Tool Calling for repetitive sub-tasks (e.g., "run these 5 tests, return only the failures"), the improver gets cleaner traces. This is a shuttle-side improvement, not an improver-side improvement.

**Applicability**: 3/5. Useful, but a separate workstream.

### 2.4 Harness Subtraction — the philosophical shift

The Epsilla writeup of Anthropic's harness-engineering blog series ([epsilla.com](https://www.epsilla.com/blogs/anthropic-harness-engineering-agent-orchestration-subtraction)) captures a real shift: "the art of building Agent-as-a-Service platforms in 2026 is about strategic subtraction." As the model improves, the harness's job is to *get out of the way* rather than scaffold more aggressively.

Concrete implication for autobench: when the improver proposes adding scaffolding (more retries, more rules, more guardrails), that should be treated with suspicion. The empirically winning direction is often *deletion*. Add a "subtraction-mode" pass to the improver — explicitly prompt for "what could we remove that's currently hurting us?"

**Applicability**: 3/5. Cheap to try, possibly counterintuitive wins.

---

## 3. DeepMind / Gemini Agent Research

### 3.1 SIMA 2 — Gemini-powered self-improvement loop

[SIMA 2 (arXiv:2512.04797)](https://arxiv.org/abs/2512.04797) is DeepMind's headline self-improving agent of late 2025. Why it matters even though it's an embodied 3D-world agent: the *self-improvement loop architecture* is general.

The recipe:
- **Curriculum Gemini**: a separate Gemini instance generates new tasks for the agent.
- **Reward-model Gemini**: a *different* model scores attempts.
- **The agent learns from its own attempts** — guided by AI feedback, not human labels.

This is the same triple Agent0 arrives at independently in a code-agent setting (§5.1). It is now the converged shape of zero-shot self-improvement: **curriculum-generator + executor + judge, all three on the same base family, separated by role.**

For autobench: we have an improver. We do *not* have a curriculum-generator that proposes new harder tasks the agent should be tested against. Our seed task pool is human-curated and bounded. Bolting on a curriculum agent that proposes "tasks the current best agent is most likely to fail on" would push us out of the local optimum the fixed seed set traps us in.

**Applicability**: 4/5. Architectural lift is medium (need a new component); conceptual payoff is large.

### 3.2 Generalization across photorealistic environments

SIMA 2 generalizes from synthetic-game training to photorealistic Genie-3-generated environments. The DeepMind blog explicitly frames this as evidence for "cognitive scaffolding [as] the substrate future machines will stand on" — i.e., the harness/scaffold *is* the durable artifact, not the weights.

This is the empirical case for autobench's whole thesis — harness optimization compounds even when the underlying model is frozen. Worth citing in the project README.

**Applicability**: 2/5 — validating, not actionable.

---

## 4. OpenAI o-series Harness Disclosures

OpenAI is the least transparent of the three labs about harness internals; what is public is mostly inferable from API surface.

### 4.1 Reasoning-effort knob and parallel function calls

The [o3/o4-mini announcement](https://openai.com/index/introducing-o3-and-o4-mini/) and the [reasoning best practices guide](https://developers.openai.com/api/docs/guides/reasoning-best-practices) confirm:

- **`reasoning_effort` parameter** (`low`/`medium`/`high`) controls hidden reasoning-token budget. High = more accuracy, higher latency and cost.
- **Reasoning items adjacent to tool calls are preserved across turns** — i.e., the harness side stores reasoning state alongside function-call traces. This is enforced by the Responses API with `store=true`.
- **Parallel function calling is now "more deterministic"** on o4. OpenAI has not published numbers but the language implies a multi-rollout-and-aggregate pattern.

### 4.2 o3-pro and "more tokens at inference"

[o3-pro](https://charonhub.deeplearning.ai/openai-debuts-o3-pro-an-updated-reasoning-model-that-applies-more-tokens-at-inference/) is essentially "o3 with a fatter reasoning budget." The public message is: *the harness can dial inference-compute up at will, and accuracy scales monotonically.* This is the compute-equivalent-gain frontier — see §7.

### 4.3 What we *don't* know

OpenAI has not published anything we'd call a "harness paper." There is no o-series equivalent of Anthropic's multi-agent-research-system post. Parallel rollouts, voting/consensus among them, judge models — all rumored, none disclosed.

**Applicability**: 2/5. Knob-tuning (reasoning_effort) is something autobench could already expose as a parametric search dimension on the OpenAI side, but no architectural lessons to lift.

---

## 5. arXiv H1 2026 Papers — annotated

### 5.1 Agent0 — curriculum + executor with tool-integrated reasoning ([arXiv:2511.16043](https://arxiv.org/abs/2511.16043))

**Summary**: Two agents from the same base LLM. Curriculum agent proposes increasingly challenging tasks; executor learns to solve them. Curriculum is trained via RL with reward signals = executor's uncertainty × tool-use frequency. Code interpreter tool creates a "virtuous cycle" where tool-equipped curriculum generates tool-using tasks, which teaches the executor to use tools better.

**Results**: Qwen3-8B-Base improves 18% on math reasoning, 24% on general reasoning, *with zero external data*. ICLR 2026 RSI Workshop oral.

**Applicability to autobench**: **5/5**. The clearest, most directly portable design in this entire report. Replace "Qwen3-8B-Base" with "our shuttle agent" and "math benchmark" with "SWE-bench-style task". The curriculum agent's RL signal — "what is the executor most uncertain about that also uses tools heavily" — is exactly the missing piece that would push autobench past the seed-task ceiling.

### 5.2 AHE — Agentic Harness Engineering ([arXiv:2604.25850](https://arxiv.org/abs/2604.25850))

**Summary**: Observability-driven automatic evolution of coding-agent harnesses. Three pillars: (1) component observability — seven editable component types exposed as files, (2) experience observability — layered evidence corpus distilled from raw trajectories, (3) decision observability — every edit paired with a self-declared prediction verified next round.

**Results**: Terminal-Bench 2 pass@1 lifts 69.7% → 77.0% over 10 iterations, beating Codex-CLI (71.9%), ACE, and TF-GRPO. Frozen harness transfers to SWE-bench-verified and 3 alternate base models with +5.1 to +10.1 pp gains.

**Applicability**: **5/5**. Top adopt-next pick. The decision-observability primitive alone is huge.

### 5.3 Meta-Harness — end-to-end harness optimization ([arXiv:2603.28052](https://arxiv.org/abs/2603.28052))

**Summary**: Stanford/MIT/KRAFTON. Outer loop with an agentic proposer that reads a filesystem containing all prior candidates' code, traces, and scores; proposes new harness; evaluates; logs; repeats. Argues existing text optimizers are "memoryless" or compress feedback too aggressively.

**Results**: Beats hand-designed harnesses on multiple domains; open-source code at [github/stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness).

**Applicability**: **4/5**. Architecturally cousin to AHE and to autobench itself. The filesystem-as-archive design is more disciplined than what we have. The "memoryless feedback compression" critique is the right lens to evaluate our improver's prompt.

### 5.4 HARBOR — Bayesian harness optimization ([arXiv:2604.20938](https://arxiv.org/pdf/2604.20938))

**Summary**: First documented application of Bayesian optimization to harness *configuration* (temperatures, k for retrieval, retry budgets, max turns, etc.) as a continuous-parameter surface, separate from harness code edits.

**Applicability**: **5/5**. This is the third arm we don't have. Cheap to integrate as a parallel improver. See §8.3.

### 5.5 "The Last Harness You'll Ever Build" ([arXiv:2604.21003](https://arxiv.org/pdf/2604.21003))

**Summary**: Sylph.AI. An *Evolution Agent* mutates harnesses given full evolution history + best-performer. Critical detail: aggregates diagnostics from history including what variants failed and why, "preventing the evolution agent from repeating unsuccessful strategies."

**Applicability**: **4/5**. The anti-recurrence guardrail is the diversity-preservation analogue at the agent level. Easy to add to our improver prompt.

### 5.6 Huxley-Gödel Machine (HGM) ([arXiv:2510.21614](https://arxiv.org/abs/2510.21614))

**Summary**: Identifies the **Metaproductivity-Performance Mismatch** — the best benchmark performer is not necessarily the best self-improver. Proposes *Clade-Metaproductivity* (CMP): aggregate descendant benchmark scores as the proxy for an agent's self-improvement potential. Uses Thompson sampling to expand promising nodes; asynchronous decoupled expansion-from-evaluation for parallelism.

**Results**: Beats DGM and SICA on SWE-bench Verified and Polyglot with *fewer* CPU hours. Optimized on SWE-bench Verified with GPT-5-mini and evaluated with GPT-5 on SWE-bench Lite: matches human-engineered agents.

**Applicability**: **5/5**. Top adopt-next. Direct replacement for our "best-scoring is next meta-agent" selection rule.

### 5.7 Darwin-Gödel Machine (DGM) ([arXiv:2505.22954](https://arxiv.org/abs/2505.22954))

**Summary**: Sakana AI / UBC. Iterative self-modifying coding agent grows an archive of generated agents. Open-ended evolution. SWE-bench 20% → 50%; Polyglot 14.2% → 30.7%.

**Applicability**: **3/5**. Important reference, but HGM dominates it empirically and architecturally. Worth understanding the open-endedness framing.

### 5.8 R-Diverse — diversity collapse mitigation ([arXiv:2602.13103](https://arxiv.org/html/2602.13103v1))

**Summary**: Identifies "Diversity Illusion" — Local (within-batch only) and Surface (different surface form, same skill). Proposes MAP (Memory-Augmented Penalty across iterations) and SAM (Skill-Aware Measurement — clusters by *underlying reasoning skill* not by surface form).

**Results**: R-Zero plateaus and degrades after iteration 3; R-Diverse sustains monotonic improvement to 52.6 Math AVG (Qwen3-4B) / 56.5 (Qwen3-8B) over 5 iterations.

**Applicability**: **4/5**. The MAP+SAM combination is the canonical diversity-preservation tool. Bolt it onto autobench's archive-selection logic.

### 5.9 Choice of Divergence in RLVR ([arXiv:2509.07430](https://arxiv.org/html/2509.07430v1))

**Summary**: Argues the divergence function in RL-with-verifiable-rewards is the neglected lever for preventing diversity collapse. KL vs. reverse-KL vs. JS make material differences in whether the policy collapses to a single mode.

**Applicability**: **2/5**. Mostly relevant if we ever do gradient-based RL on the harness. Currently we don't.

### 5.10 Multi-Agent Code Verification via Information Theory ([arXiv:2511.16708](https://arxiv.org/abs/2511.16708))

**Summary**: CodeX-Verify — four specialized verifier agents detect different bug classes; combining them is justified via submodularity of mutual information under conditional independence. Finds bugs no single agent finds.

**Applicability**: **4/5**. We have one judge model. The information-theoretic argument is that *diversified* verifiers detect more failures with sub-linear overlap. Cheap to bolt on as multiple judge calls with different prompts/seeds.

### 5.11 Agent-as-a-Judge ([arXiv:2410.10934](https://arxiv.org/abs/2410.10934))

**Summary**: Use an agentic system to evaluate another agentic system; intermediate-step feedback rather than only end-state grades. Defines DevAI benchmark of 55 realistic AI code-gen tasks.

**Applicability**: **3/5**. We already use LLM-as-judge; the agentic-judge extension is a more thorough version. Worth piloting on one autobench task to measure improver-quality lift.

### 5.12 Survey on Agent-as-a-Judge ([arXiv:2601.05111](https://arxiv.org/pdf/2601.05111))

**Summary**: Survey, January 2026. Maps the design space — pairwise vs. pointwise, agentic vs. monolithic, intermediate-step vs. final-only feedback.

**Applicability**: **2/5**. Reference material; not a technique to adopt.

### 5.13 Confucius Code Agent — scaffolding scaling ([arXiv:2512.10398](https://arxiv.org/html/2512.10398v4))

**Summary**: Scalable scaffolding for real-world codebases. Focuses on the repo-scale context-management problem.

**Applicability**: **2/5**. Tangential to RSI loop.

### 5.14 Efficient Agents ([arXiv:2508.02694](https://arxiv.org/html/2508.02694v1))

**Summary**: "Simple Memory" (observations+actions only) **outperformed** five more elaborate memory configurations *and* cost less (53% → 56%, cost-of-pass 0.98 → 0.74). This is the canonical "subtraction wins" empirical result for autobench's improver to internalize.

**Applicability**: **4/5**. Direct evidence that the LLM improver should be biased toward *removal* not addition. Update improver prompt accordingly.

### 5.15 Artemis / "Evolving Excellence" ([arXiv:2512.09108](https://arxiv.org/html/2512.09108v1))

**Summary**: General-purpose evolutionary optimization platform that treats agents as black boxes. Jointly optimizes textual + parametric components capturing interdependencies. ALE Agent +13.6% on competitive programming, Mini-SWE +10.1%, CrewAI −36.9% execution cost.

**Applicability**: **4/5**. Companion to HARBOR — covers more search-space topology. Strong candidate for the "parametric search arm" in §8.3.

### 5.16 Bayesian Prompt Optimization ([arXiv:2512.15076](https://arxiv.org/pdf/2512.15076))

**Summary**: Exploratory study applying Bayesian optimization to prompt selection. Shows BO is competitive with LLM-driven optimization on smaller search spaces but loses on combinatorial ones.

**Applicability**: **3/5**. Reinforces that BO is a good *complement* to (not replacement for) LLM-driven improvers.

### 5.17 Towards a Science of Scaling Agent Systems ([arXiv:2512.08296](https://arxiv.org/html/2512.08296v1))

**Summary**: Empirical scaling laws for compound agent systems. Includes inference-time scaling curves analogous to training-time scaling laws.

**Applicability**: **3/5**. Reference for §7 compute-equivalent-gain framing.

### 5.18 Towards a Science of AI Agent Reliability ([arXiv:2602.16666](https://arxiv.org/html/2602.16666v1))

**Summary**: Reliability metrics for agent systems — distinguishes capability progress from reliability progress.

**Applicability**: **2/5**. Useful for evaluation framing but no technique to adopt.

### 5.19 ARE — scaling up agent environments ([arXiv:2509.17158](https://arxiv.org/html/2509.17158v1))

**Summary**: Standardized agent-environment evaluation infrastructure.

**Applicability**: **2/5**. Infrastructure paper, not a technique.

### 5.20 SAGA — workflow-atomic scheduling ([arXiv:2605.00528](https://arxiv.org/html/2605.00528v1))

**Summary**: GPU-cluster-aware scheduling of agent inference workloads.

**Applicability**: **1/5**. Not autobench-relevant — we don't manage our own inference cluster.

---

## 6. Production Open-Source Projects — what evolved post-SICA

### 6.1 OpenHands (formerly OpenDevin)

The [OpenHands V1 SDK](https://docs.openhands.dev/openhands/usage/developers/evaluation-harness) and [Jan 2026 OpenHands Index](https://www.openhands.dev/blog/openhands-index) reveal real architectural maturation:

- **Native reasoning support** — wraps Claude extended thinking and o-series reasoning under one API.
- **LiteLLM for model routing** — 100+ providers behind a single interface. Mirrors what autobench needs for cross-model harness-transfer experiments.
- **Event sourcing for state** — immutable event log mirrors Anthropic's "session as event log" pattern (§2.1). Convergent design.
- **Context condensation** — explicit context-overflow handling.
- **LLM-based security analyzer** — separate model audits actions before execution. This is a guard-rail pattern we don't have.
- **Parallel multi-agent execution with dependency trees** — task decomposition into a DAG, agents work simultaneously in non-interfering cloud sandboxes.
- **60+ built-in scorers and LLM judges** via MLflow integration ([mlflow.org/blog/mlflow-openhands](https://mlflow.org/blog/mlflow-openhands/)).

**Lift candidates for autobench**: event sourcing for the iteration archive, LLM-based pre-execution security analyzer (worth doing now that we have gVisor + Firecracker), LiteLLM for model abstraction.

### 6.2 SWE-agent

The [SWE-bench harness reference](https://www.swebench.com/SWE-bench/reference/harness/) and the [CodeSOTA comparison](https://www.codesota.com/agentic/openhands-vs-swe-agent) show SWE-agent has consolidated its agent-computer-interface (ACI) abstraction but has been *less* innovative architecturally than OpenHands. The interesting bit is their **benchmarks repo** — the eval-harness-as-a-product orientation that autobench shares.

### 6.3 aider, Cline, Copilot Workspace

These are user-facing tools, not harness-research vehicles. aider has matured its diff-application logic; Cline has invested in MCP integration; Copilot Workspace remains opaque. None of them publish papers or detailed engineering posts; harness lessons here are inferred only.

**Applicability**: 1/5 for direct lift; useful only as reference UX.

### 6.4 Claude Code's five-layer harness

Per [Claude Code docs](https://code.claude.com/docs/en/how-claude-code-works) and the [SAP "Seven Pillars"](https://community.sap.com/t5/artificial-intelligence-blogs-posts/agentic-harness-architecture-seven-pillars-that-make-claude-code-production/ba-p/14395198) writeup, Claude Code's harness is *the* production reference:

| Layer | What |
|---|---|
| Memory | CLAUDE.md persistent context |
| Tools | MCP plus built-ins |
| Permissions | settings.json scoped allow/deny |
| Hooks | PreToolUse / PostToolUse for deterministic gates |
| Observability | Session logs (event-stream style) |

Plus the higher-order primitives: skills, subagents, hooks, settings — *individually* features, *composed* a production harness.

**Applicability**: 4/5. Most of the patterns we already have under different names. The **hooks** primitive (deterministic pre/post-tool gates) is a useful generalization of what autobench currently does ad hoc.

---

## 7. Harness Transfer — what's empirically known

This is the question Poetiq's marketing left dangling. As of May 2026 the published evidence is now substantive.

### 7.1 What is verified

- **AHE frozen-harness transfer**: 3 alternate base-model families, +5.1 to +10.1 pp gain on each, on SWE-bench-verified ([arXiv:2604.25850](https://arxiv.org/abs/2604.25850)). Quote: "the largest gains [were] on bases further from saturation, suggesting that AHE encodes coordination patterns that less-saturated models lean on more heavily." This is the most empirically grounded transfer claim in the literature.
- **HGM transfer**: agent optimized on SWE-bench Verified with GPT-5-mini, evaluated on SWE-bench Lite with GPT-5 — matches human-engineered agents ([arXiv:2510.21614](https://arxiv.org/abs/2510.21614)). Cross-benchmark and cross-model in one shot.
- **Poetiq**: [MarkTechPost coverage](https://www.marktechpost.com/2026/05/14/poetiqs-meta-system-automatically-builds-a-model-agnostic-harness-that-improved-every-llm-tested-on-livecodebench-pro-without-fine-tuning/) — "Every model tested improved." LiveCodeBench Pro, harness optimized for Gemini 3.1 Pro, transferred to "broad set of other models from different providers and generations." *Numbers not disclosed in public materials we found.* Treat the qualitative claim as plausible, the quantitative claim as still un-sourced until the paper drops.

### 7.2 Component-level transfer is heterogeneous

AHE's ablation (also in [arXiv:2604.25850](https://arxiv.org/abs/2604.25850)):
- Tools alone: **+3.3 pp**
- Middleware alone: **+2.2 pp**
- Long-term memory alone: **+5.6 pp**
- System prompt alone: **−2.3 pp** (i.e., *hurts*)

So: **structural / tool / memory components transfer, but prompts often do not.** This is a sharper claim than "harnesses transfer" — it tells you which slots to freeze and which to re-optimize per model.

### 7.3 What's still hand-waving

- The strong form of "one harness rules them all" is *not* validated. Models close to saturation (GPT-5, Claude 4 family on SWE-bench) gain less. There's a saturation ceiling.
- Cross-architecture transfer (e.g., dense → MoE → reasoning-trained) is under-studied. AHE's 3 bases are all dense Transformers of similar generation.
- No paper has validated that a harness optimized via LLM-improver transfers *better* than one tuned via parametric Bayesian search. Open question.

**Conclusion**: harness-transfer is now a real research result, not vapor. But the transfer is *partial*, *component-dependent*, and *prompt-fragile*. Autobench's evaluations should always include a transfer arm.

---

## 8. Three Concrete Experiments We Could Run

Each is sized to fit our existing substrate (shuttle + improver + judge + archive). I've ordered them by leverage × cheapness.

### 8.1 Experiment A — Decision Observability (AHE-style falsifiable prediction)

**Hypothesis**: Forcing every improver edit to ship with a self-declared, machine-checkable prediction about the next iteration's measurable outcome (e.g., "test pass-rate on subset X will increase by ≥3 pp", "cost-per-task will drop by ≥10%") increases iteration quality by punishing improvers that hallucinate "wins."

**Procedure**:
1. Extend the improver's output schema to require a `predictions: List[Prediction]` field with `(metric_name, expected_delta, confidence)`.
2. After the iteration runs, score the predictions: did each materialize?
3. Track per-improver-strategy a *Brier score* analogue (calibration: are claimed-high-confidence predictions actually right?).
4. Use the calibration score as an input feature when selecting which improver-output-style to promote in the next round.

**Expected outcome**: 2–4 iterations of telemetry should show whether predictions are well-calibrated. If they're not (and the AHE paper suggests they won't be initially), the act of *requiring* predictions will improve improver focus regardless.

**Cost**: ~1 day of harness work. Telemetry is the bottleneck; the harness change is small.

**Adopt-next priority**: 1.

### 8.2 Experiment B — HGM Clade-Metaproductivity Selection

**Hypothesis**: Selecting the next meta-improver based on *its descendants' average score* rather than *its own score* avoids the metaproductivity-performance mismatch. Specifically, agents that are 90th-percentile on the benchmark but produce poor descendants get demoted; agents that are 70th-percentile but produce 95th-percentile descendants get promoted.

**Procedure**:
1. In the archive, maintain a `descendants: List[ArchiveNode]` link.
2. Define `CMP(node) = mean(node.descendants.score) if descendants else None`.
3. For nodes without descendants, fall back to Thompson-sampling: posterior over expected CMP given a Beta prior.
4. When picking the next meta-improver, weight `node.score * 0.5 + (CMP(node) or score) * 0.5`.
5. Run 10 iterations on a fixed seed task. Compare end-state pass-rate vs. current "argmax score" baseline.

**Expected outcome**: HGM's paper reports the CMP-driven loop gets *more lift per iteration* with *fewer CPU hours*. Order-of-magnitude expectation: same lift in 60–70% of the wall-clock.

**Cost**: ~1–2 days of archive plumbing. Selection rule is one function.

**Adopt-next priority**: 2.

### 8.3 Experiment C — Parametric Search Arm (HARBOR/Artemis style)

**Hypothesis**: A pure Bayesian / evolutionary search over the *parametric* harness surface (temperatures, retry budgets, retrieval k, max turns, judge-confidence threshold, etc.) finds local wins the LLM improver misses, *and* the two arms compound when both proposals are passed to a meta-judge.

**Procedure**:
1. Define the parametric surface as a JSON schema with 8–12 dimensions and explicit bounds.
2. Implement a tiny Gaussian-process BO loop using scikit-optimize or BoTorch. Surrogate: a GP over (parametric vector → benchmark score). Acquisition: expected improvement.
3. Each "iteration" now spawns TWO proposals: one from the LLM improver, one from the BO loop.
4. Both are evaluated. A meta-judge LLM picks (or both are promoted if non-dominating).
5. Track which arm wins more often, and how often the *combined* archive front dominates either single arm.

**Expected outcome**: Based on Artemis and HARBOR data, expect:
- BO finds wins LLM improver doesn't (especially counter-intuitive ones like "*lower* the retry budget, it's helping the model stall");
- The dual-arm archive has higher *Pareto front coverage* than either single arm.

**Cost**: ~2–3 days. BO library is mature; the cost is the new arm in the dispatcher.

**Adopt-next priority**: 3 (high payoff, slightly higher cost).

---

## 9. Open Questions and Risks

1. **Diversity-collapse risk in our improver loop.** We don't currently measure whether successive improver proposals are surface-different or skill-different. SAM-style skill clustering (cluster proposals by what *kind* of change they make, not by their text) would tell us whether we're already in collapse. *Action*: add a one-shot diff-clustering pass over the last N improver outputs as a session-end report.
2. **Curriculum-agent risk.** SIMA-2 / Agent0 work because the curriculum agent is well-grounded by tool feedback. Our seed tasks are abstract enough that a curriculum agent could drift into generating tasks our shuttle can't even parse. Pilot with a constrained generator before opening it up.
3. **The "surprising config win" question is under-evidenced.** Searched specifically for documented cases of LLM-driven improvers proposing wins that rule-based heuristics would never have thought of. Found one rigorous case: the [Efficient Agents paper](https://arxiv.org/html/2508.02694v1) where *Simple Memory* outperformed five fancier configurations, and the [Agentic Harness Engineering Medium article](https://medium.com/superagentic-ai/agentic-harness-engineering-the-next-frontier-after-harness-engineering-49ff0faccdb2) describes Anthropic's "delete your harness" experience. Beyond that, the literature is anecdotal. *This means*: our autobench telemetry, if we tag "rule-based-derivable vs LLM-novel" on each accepted edit, would be publishable.
4. **GUI/computer-use transfer to code.** Searched specifically. [Anthropic's computer-use docs](https://www.mindstudio.ai/blog/what-is-claude-code-computer-use) and Claude 3.7 Sonnet integrate computer-use *into* the coding harness for browser-checking front-end work, but no paper claims computer-use harness techniques transfer to code-only loops. The architectural shapes are similar; the failure modes diverge enough that the techniques don't generalize cleanly. *Don't invest here.*
5. **Cost discipline.** Spawning 3–5 parallel sub-improvers (§2.2) and adding a BO arm (§8.3) and a curriculum agent (§3.1) compounds compute. Each addition should be gated on a measured Pareto-front lift, not just "the paper said so."
6. **Pending deer cycle.** Cycle `20260516T055935Z-8df3` was still in `pending` at write time. If it terminates with material new content, append as §11. The earlier cycle `20260516T052945Z-ceee` already terminated `lift_measured` on a related framing; its content was implicitly folded into this report via the WebSearch corroboration.
7. **Recency-vs-rigor tradeoff.** Several papers in §5 are from May 2026 (AHE, HARBOR, "Last Harness", Meta-Harness). They are *concurrent work*, not peer-reviewed. Treat the empirical numbers as directionally trustworthy but not gospel. The HGM, SICA, DGM, R-Diverse, Agent0 results are more settled.
8. **The "third path" question is answered**: yes, multiple groups (HARBOR, Artemis, Meta-Harness's outer loop, "Last Harness" Evolution Agent) are now doing Bayesian / evolutionary search at the orchestration layer — sometimes as the *sole* search arm, sometimes alongside LLM improvers. Autobench should add at least one such arm.

---

## 10. Citations

### Anthropic
- Anthropic. *Introducing advanced tool use on the Claude Developer Platform.* <https://www.anthropic.com/engineering/advanced-tool-use>
- Anthropic. *How we built our multi-agent research system.* <https://www.anthropic.com/engineering/multi-agent-research-system>
- Anthropic. *Scaling Managed Agents: Decoupling the brain from the hands.* <https://www.anthropic.com/engineering/managed-agents>
- Anthropic. *Long-running Claude for scientific computing.* <https://www.anthropic.com/research/long-running-Claude>
- Anthropic. *Claude 3.7 Sonnet and Claude Code.* <https://www.anthropic.com/news/claude-3-7-sonnet>
- Claude Code docs. *How Claude Code works.* <https://code.claude.com/docs/en/how-claude-code-works>
- Epsilla blog. *The Art of Subtraction: Why Anthropic is Telling Us to Delete Our Agent Harnesses.* <https://www.epsilla.com/blogs/anthropic-harness-engineering-agent-orchestration-subtraction>
- Anthropic-aligned writeup. *Anthropic details a multi-agent harness for frontend design and long-running software engineering.* <https://insights.marvin-42.com/articles/anthropic-details-a-multi-agent-harness-for-frontend-design-and-long-running-software-engineering>

### DeepMind / Gemini
- DeepMind. *SIMA 2: A Gemini-Powered AI Agent for 3D Virtual Worlds.* <https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/>
- *SIMA 2: A Generalist Embodied Agent for Virtual Worlds.* arXiv:2512.04797. <https://arxiv.org/abs/2512.04797>
- InfoQ coverage. *SIMA 2 Uses Gemini and Self-Improvement to Generalize across Unseen 3D and Photorealistic Worlds.* <https://www.infoq.com/news/2025/12/sima-2-gemini-agent/>

### OpenAI
- OpenAI. *Introducing OpenAI o3 and o4-mini.* <https://openai.com/index/introducing-o3-and-o4-mini/>
- OpenAI API. *Reasoning best practices.* <https://developers.openai.com/api/docs/guides/reasoning-best-practices>
- OpenAI API. *o3 Model.* <https://developers.openai.com/api/docs/models/o3>
- DeepLearning.AI. *OpenAI Debuts o3-pro, an Updated Reasoning Model That Applies More Tokens at Inference.* <https://charonhub.deeplearning.ai/openai-debuts-o3-pro-an-updated-reasoning-model-that-applies-more-tokens-at-inference/>
- OpenRouter docs. *Reasoning Tokens.* <https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>

### Core self-improving-agent papers
- Robeyns et al. *A Self-Improving Coding Agent (SICA).* arXiv:2504.15228. <https://arxiv.org/abs/2504.15228>
- Zhang et al. *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents.* arXiv:2505.22954. <https://arxiv.org/abs/2505.22954>
- *Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine.* arXiv:2510.21614. <https://arxiv.org/abs/2510.21614>
- *Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning.* arXiv:2511.16043. <https://arxiv.org/abs/2511.16043>
- SICA OpenReview / ICLR 2025 SSI-FM workshop. <https://openreview.net/pdf?id=rShJCyLsOr>

### Harness-evolution papers
- *Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses.* arXiv:2604.25850. <https://arxiv.org/abs/2604.25850>
- *Meta-Harness: End-to-End Optimization of Model Harnesses.* arXiv:2603.28052. <https://arxiv.org/abs/2603.28052>
- *HARBOR: Automated Harness Optimization.* arXiv:2604.20938. <https://arxiv.org/pdf/2604.20938>
- Seong (Sylph.AI). *The Last Harness You'll Ever Build.* arXiv:2604.21003. <https://arxiv.org/pdf/2604.21003>
- *Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering.* arXiv:2603.05344. <https://arxiv.org/html/2603.05344v1>
- *Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases.* arXiv:2512.10398. <https://arxiv.org/html/2512.10398v4>
- *Agent Harness for Large Language Model Agents: A Survey.* Preprints.org 202604.0428. <https://www.preprints.org/manuscript/202604.0428>

### Diversity preservation
- *R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training.* arXiv:2602.13103. <https://arxiv.org/html/2602.13103v1>
- *The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward.* arXiv:2509.07430. <https://arxiv.org/html/2509.07430v1>
- *Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains.* arXiv:2501.05707. <https://arxiv.org/html/2501.05707v1>
- *Diversity Collapse in RL.* EmergentMind. <https://www.emergentmind.com/topics/diversity-collapse-in-rl>

### Verifier ensembles / judges
- *Multi-Agent Code Verification via Information Theory.* arXiv:2511.16708. <https://arxiv.org/abs/2511.16708>
- *Agent-as-a-Judge: Evaluate Agents with Agents.* arXiv:2410.10934. <https://arxiv.org/abs/2410.10934>
- *A Survey on Agent-as-a-Judge.* arXiv:2601.05111. <https://arxiv.org/pdf/2601.05111>
- *When AIs Judge AIs: The Rise of Agent-as-a-Judge Evaluation for LLMs.* arXiv:2508.02994. <https://arxiv.org/html/2508.02994v1>

### Optimization and scaling
- *Evolving Excellence: Automated Optimization of LLM-based Agents (Artemis).* arXiv:2512.09108. <https://arxiv.org/html/2512.09108v1>
- *Bayesian Prompt Optimization (exploratory study).* arXiv:2512.15076. <https://arxiv.org/pdf/2512.15076>
- *Evolutionary Computation and Large Language Models: A Survey.* arXiv:2505.15741. <https://arxiv.org/html/2505.15741v1>
- *Efficient Agents: Building Effective Agents While Reducing Cost.* arXiv:2508.02694. <https://arxiv.org/html/2508.02694v1>
- *Towards a Science of Scaling Agent Systems.* arXiv:2512.08296. <https://arxiv.org/html/2512.08296v1>
- *Towards a Science of AI Agent Reliability.* arXiv:2602.16666. <https://arxiv.org/html/2602.16666v1>
- *ARE: scaling up agent environments and evaluations.* arXiv:2509.17158. <https://arxiv.org/html/2509.17158v1>
- *SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters.* arXiv:2605.00528. <https://arxiv.org/html/2605.00528v1>
- *Heterogeneous Computing: The Key to Powering the Future of AI Agent Inference.* arXiv:2601.22001. <https://arxiv.org/abs/2601.22001>
- *The Price of Progress: Price, Performance and the Future of AI.* arXiv:2511.23455. <https://arxiv.org/html/2511.23455v2>

### Open-source and production projects
- OpenHands docs. *Evaluation Harness.* <https://docs.openhands.dev/openhands/usage/developers/evaluation-harness>
- OpenHands. *Index (Jan 2026).* <https://www.openhands.dev/blog/openhands-index>
- MLflow blog. *Harness Your OpenHands Agent with AI Observability and Governance.* <https://mlflow.org/blog/mlflow-openhands/>
- CodeSOTA. *OpenHands vs SWE-agent (2026).* <https://www.codesota.com/agentic/openhands-vs-swe-agent>
- ToolHalla. *Devin vs OpenHands vs SWE-agent: 2026.* <https://toolhalla.ai/blog/devin-vs-openhands-vs-swe-agent-2026>
- SWE-bench. *The Harness.* <https://www.swebench.com/SWE-bench/reference/harness/>
- VoltAgent. *Awesome AI Agent Papers (2026 collection).* <https://github.com/VoltAgent/awesome-ai-agent-papers>

### ICLR 2026 RSI workshop
- ICLR 2026. *Workshop on AI with Recursive Self-Improvement.* <https://iclr.cc/virtual/2026/workshop/10000796>
- Workshop site. *Accepted Papers.* <https://recursive-workshop.github.io/papers.html>
- Workshop summary on OpenReview. <https://openreview.net/pdf?id=OsPQ6zTQXV>
- Artifocial. *From Self-Play to Self-Research: The ICLR 2026 RSI Workshop.* <https://www.artifocial.ai/blog/rsi-workshop-2026-mar-18>

### Production harness writeups
- WaveSpeed blog. *Claude Code Agent Harness: Architecture Breakdown.* <https://wavespeed.ai/blog/posts/claude-code-agent-harness-architecture/>
- DEV.to (Shipwithai). *The Complete Claude Code Harness Engineering Guide (5 Layers, 8 Deep-Dives).* <https://dev.to/shipwithaiio/the-complete-claude-code-harness-engineering-guide-5-layers-8-deep-dives-3d4j>
- SAP Community. *Agentic Harness Architecture: Seven Pillars That Make Claude Code Production-Grade.* <https://community.sap.com/t5/artificial-intelligence-blogs-posts/agentic-harness-architecture-seven-pillars-that-make-claude-code-production/ba-p/14395198>
- Kong Inc. *Governing Claude Code: Secure Agent Harness Rollouts with Kong AI Gateway.* <https://konghq.com/blog/engineering/claude-code-governance-with-an-ai-gateway>
- Infralovers. *Harness Engineering: Why the Frame Matters More Than the Model.* <https://www.infralovers.com/blog/2026-03-13-harness-engineering-rahmen-wichtiger-als-modell/>
- Cobus Greyling on Medium. *Auto Agentic Harness Engineering.* <https://cobusgreyling.medium.com/auto-agentic-harness-engineering-b27a962fad9a>
- Superagentic AI. *Agentic Harness Engineering: The Next Frontier After Harness Engineering.* <https://medium.com/superagentic-ai/agentic-harness-engineering-the-next-frontier-after-harness-engineering-49ff0faccdb2>

### Poetiq
- MarkTechPost (May 14 2026). *Poetiq's Meta-System Automatically Builds a Model-Agnostic Harness That Improved Every LLM Tested on LiveCodeBench Pro Without Fine-Tuning.* <https://www.marktechpost.com/2026/05/14/poetiqs-meta-system-automatically-builds-a-model-agnostic-harness-that-improved-every-llm-tested-on-livecodebench-pro-without-fine-tuning/>

### Computer use
- MindStudio. *What Is Claude Code Computer Use?* <https://www.mindstudio.ai/blog/what-is-claude-code-computer-use>
- TechCrunch. *Google's SIMA 2 agent uses Gemini to reason and act in virtual worlds.* <https://techcrunch.com/2025/11/13/googles-sima-2-agent-uses-gemini-to-reason-and-act-in-virtual-worlds/>

---

*Report generated 2026-05-16. Deer cycle `20260516T055935Z-8df3` (strategic class) was pending at write time; companion cycle `20260516T052945Z-ceee` terminated `lift_measured` earlier same day. If `8df3` produces material new content when it terminates, append as §11.*
