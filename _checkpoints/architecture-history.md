# autobench — architecture history (pre-Phase-2B)

Trimmed from the opening docstrings of ten bloatted files during Phase 2B
of the restructuring (commit to land). The history is preserved here as a
single read-once reference; the live source modules now carry a ≤15-line
header that names the public surface and points maintainers to the
``autobench.llm/``, ``autobench.rsi/``, ``autobench.evaluation/``
subpackages for everything else.

The audit (`_checkpoints/audit_report.md`, 2026-06-01) flagged these 10
files as BLOATED (30–50 opening doc lines) or PATHOLOGICAL (50+):
- `curriculum.py` 63, `benchmark_registry.py` 61 (PATHOLOGICAL)
- `minimax_improver.py` 45, `llm_improver.py` 43, `multi_improver.py` 41,
  `adversarial.py` 40, `population.py` 38, `worker_agent.py` 37 (BLOATED)
- `replay.py` 29, `diversity.py` 26, `distillation.py` 21 (over the LEAN < 20 cap)

---

## 1. `llm/anthropic.py` (was `llm_improver.py`) — trimmed from 43 lines

The original 5-point architecture-decision narration explained why the
wrapper calls the Anthropic SDK directly rather than going through the
deer-flow REST gateway:

1. **Anthropic SDK direct**: deer-flow uses `ClaudeChatModel`
   (langchain_anthropic wrapper) internally. Direct SDK gives autobench
   full control over retry, timeout, and thinking budget.

2. **deer-flow REST API** (localhost:2026) confirmed endpoints:
   - `POST /api/runs/stream` — stateless run, SSE stream
   - `POST /api/runs/wait` — stateless run, blocks until completion
   - `POST /api/threads/{thread_id}/suggestions` — conversational follow-up
     (NOT suitable for harness improvement)
   - `POST /api/stack-tuner/cycle` — starts a meta-probe cycle
   - `GET /api/stack-tuner/status` — polls cycle status

3. **deer act CLI**: hypothetical, the CLI surface enumeration mentions
   `deer act` for "inspect/drain queues" but the implementation wasn't
   confirmed in the router codebase.

4. **Function calling**: deer-flow's agents use tool-calling internally,
   but the gateway's stateless run endpoints accept a LangGraph-style
   `input` dict — not arbitrary tool definitions.

5. **SICA-style 4-step loop**: the MaximeRobeyns/self_improving_coding_agent
   pattern is (1) generate, (2) execute, (3) collect verdict, (4) improve.
   autobench's `RSILoop` in `core.py` mirrors this exactly.

Removed "Integration notes for deer-flow REST API" comment block at the
file tail (CONFIRMED/USABLE/NOT-SUITABLE/HYPOTHETICAL endpoint matrix)
moved here. Also removed the 4-line "Removed VERDICT_STRATEGIES" note
(nervous-bus-6ed) which was narrating a prior bug session 01KRQTNMM8RFKC477DRRRVMVS4
in which a 77% CE rate driven by `<think>` reasoning prose leaking into
code submissions was misdiagnosed because the canned verdict→fix lookup
hid the structural failure pattern from the improver.

---

## 2. `llm/minimax.py` (was `minimax_improver.py`) — trimmed from 45 lines

Original docstring captured the wire-shape rationale for choosing the
MiniMax Anthropic-compatible endpoint over OpenAI-compatible:

- MiniMax-M2.7 emits reasoning as a separate `"thinking"` content block
  rather than inlining `<think>...</think>` markers into the assistant
  content. This eliminates a latent JSON-collision risk where stray
  `{...}` sequences in reasoning prose could fool the regex extractor
  in `_parse_llm_response`.
- Wire shape (Anthropic-compatible, default):
  `POST https://api.minimax.io/anthropic/v1/messages`
  Bearer `$MINIMAX_API_KEY`, `{"model": "MiniMax-M2.7", "system": "...",
  "messages": [...], "temperature": 0.3, "max_tokens": 4096}`.
- Anthropic-shaped response: `{"content": [{"type": "thinking", "..."},
  {"type": "text", "text": "...JSON..."}], "usage": {...}}`.
- Thinking blocks are dropped at parse time. OpenAI mode is preserved for
  A/B comparison — see `tools/ab_minimax_endpoints.py`.

Pricing/billing rationale (nervous-bus-dq7l): the MiniMax coding plan
bills by requests-per-5h (14250 cap), not by $/token. The hardcoded
Anthropic pricing model in `llm_improver._estimate_cost` was removed
in favor of 0.0 in `minimax_improver._estimate_cost` because $/token
rates decay into fiction the moment list pricing shifts.

---

## 3. `llm/ensemble.py` (was `multi_improver.py`) — trimmed from 41 lines

Original two-strategy narration (nervous-bus-9xd, wire-pop Phase 6):

- Strategy A `vote` (default): all N improvers receive identical context,
  propose deltas in parallel (1 forward eval downstream). `aggregate_deltas`
  merges deltas by majority vote per field; ties break by "first non-noop
  in fan-out order". CHEAP — only the LLM calls scale with N.
- Strategy B `parallel` (opt-in via `AUTOBENCH_IMPROVER_STRATEGY=parallel`):
  each improver's delta is forwarded as a candidate harness to a downstream
  one-iteration evaluation; the best-by-score arm wins. EXPENSIVE —
  evaluations scale with N. The harness itself does not perform Strategy B's
  forward evals here (that would require an evaluator instance per call).
  For symmetry with `strategy="vote"`, `improve()` still returns a single
  `(HarnessConfig, ImprovementDelta)`. `strategy="parallel"` requires an
  `arm_evaluator` callable.

Single-instance fallback: `n_instances == 1` short-circuits to a direct
call into one wrapper and emits no ensemble event (zero behavioural change
vs. the single-improver path).

The original `nervous-bus-9xd` annotation mentioned "nervous-bus-mNN" in
the sibling context; the actual bead id was `9xd` (Phase 6 of wire-pop).

---

## 4. `llm/worker.py` (was `worker_agent.py`) — trimmed from 37 lines

The original docstring captured three failure-model notes plus the
worker→judge→evaluator wire shape:

Failure model (3-step):
  1. Primary model (default `MiniMax-M2.7`) is tried with `max_retries`
     attempts using exponential backoff + Retry-After honouring.
  2. On total failure, the worker falls back to `fallback_model`
     (default `MiniMax-M2.5`) with the same retry budget.
  3. If both models fail, returns an empty code string (evaluator
     interprets this as a CE verdict; we do NOT raise so a single
     transient outage can't wedge the whole benchmark suite).

Wire shape (OpenAI-compatible):
  `POST https://api.minimax.io/v1/chat/completions`
  Bearer `$MINIMAX_API_KEY`,
  `{"model": "MiniMax-M2.7", "messages": [{"role": "system", "content":
  "..."}, {"role": "user", "content": "<case.prompt + harness.tool_surface>"}],
  "temperature": 0.3, "max_tokens": <harness.budget["max_tokens"] or 4096>}`.

The original `nervous-bus-dq7l` note about removing $/token rate tables
moved to an inline comment at the file's top — applies to both
`minimax_improver` and `worker_agent`.

---

## 5. `rsi/adversarial.py` (was `adversarial.py`) — trimmed from 40 lines

The original docstring captured the dual co-evolution pattern (Code-A1 /
SAGE) and a comparison with the `CurveballGenerator` pattern:

The adversarial pattern: one LLM (the *generator*) is prompted to invent
hard problems with tricky edge cases that target a specified failure
mode. Another LLM (the *worker*) solves the generated problem. The
generator's fitness signal is "did the worker's solution FAIL on the
generated case?". The dual's success is when the generator produces
cases the worker can no longer solve, and the worker then improves.

Comparison with `CurveballGenerator` (research-context reference): the
CurveballGenerator pattern generates one-off red-team cases and throws
them away. autobench's adversarial pattern keeps the generated cases in
the benchmark mix so future runs also exercise them. This makes the
adversarial loop an online curriculum generator rather than a one-shot
red-team exercise.

The "phase roadmap" removed from `population.py` overlaps with this
section and was consolidated here.

---

## 6. `rsi/population.py` (was `population.py`) — trimmed from 38 lines

The phase roadmap (paraphrased):

- Phase 1: Single-advocate population (legacy `SelfImprovingHarness`
  with N=1). Always the control arm.
- Phase 2 (nervous-bus-9xd / wire-pop): N=3 default with `MultiImproverEnsemble`
  vote strategy. PopulationRunner wraps N advocates, picks promotion
  candidate by `aggregate_score` adjusted for diversity.
- Phase 3 (nervous-bus-bo86): cross-advocate context — each advocate
  sees the recent hypotheses from its siblings. Hard signal is the
  `adjusted_score` bonus that biases winner-selection toward
  exploratory lineages.
- Phase 4 (nervous-bus-future): convergence-driven population
  shrinkage — when the adjusted_score variance drops below the noise
  floor, halve the advocate count to recycle compute. Not yet wired.

Each advocate has its own `session_id` (so nervous-bus events from
different advocates don't collide), its own `current_harness`, and
its own obs instance.

---

## 7. `rsi/replay.py` (was `replay.py`) — trimmed from 29 → ~15 lines

The original docstring explained the counterfactual-runner rationale:
given a captured autobench session (CloudEvents JSONL on the nervous-bus
debug file), reconstruct the harness config at the start of the chosen
iteration, apply `--override` mutations, replay the exact same benchmark
cases, and print (or write) a comparison report. The "antagonist's I bet
you would have lost here weapon" framing captures why this lives in
`rsi/` not in `evaluation/`: it's the operator's tool for falsifying an
RSI claim, not a benchmark-instrumentation tool.

---

## 8. `evaluation/curriculum.py` (was `curriculum.py`) — trimmed from 63 → 15 lines

The original docstring included an ASCII-art pipeline diagram, a
bias/drift risk section, and a CLI section. All preserved here.

ASCII pipeline:

```
                ┌──────────────────────┐
                │   Daily synthesis    │  ◀── 04:00 UTC cron
                │  (LLM-generated      │
                │   problems, JSONL)   │
                └──────────┬───────────┘
                           │ autobench.curriculum.problem.v1
                           ▼
                ┌──────────────────────┐
                │   Cycle dispatch     │  ◀── pulse.trigger_curriculum_cycle
                │  (Tier 1 + Tier 2)   │
                └──────────┬───────────┘
                           │ autobench.bench.requested.v1
                           ▼
                ┌──────────────────────┐
                │   Bench execution    │
                │  (gVisor / FC VM)    │
                └──────────┬───────────┘
                           │ autobench.result.v1
                           ▼
                ┌──────────────────────┐
                │   Judge + curation   │  ◀── judge.disagreement.v1
                │  (5-judge pool)      │
                └──────────┬───────────┘
                           │ autobench.curriculum.cycle.v1
                           ▼
                ┌──────────────────────┐
                │   Curriculum update  │
                │  (difficulty band    │
                │   re-weighting)      │
                └──────────────────────┘
```

Bias / drift risks:
- The daily LLM synthesis can drift toward easier problems over time
  (the LLM learns the running failure mode distribution and stops
  generating edge cases that historically break the worker). Mitigation:
  a hard 30% floor of problems tagged `hard` regardless of LLM's
  suggested difficulty.
- Same prompt → same seed → same LLM. The 4,000-problem cache must
  include a problem-id stable hash so replays deterministically pick
  the same problem.

CLI section:
- `python -m autobench.curriculum daily` — generate today's batch
- `python -m autobench.curriculum status` — print current cycle state
- `python -m autobench.curriculum rerun --cycle-id CYC-...` — re-execute
  a prior cycle (used in CI to check the harness is reproducible)

---

## 9. `evaluation/registry.py` (was `benchmark_registry.py`) — trimmed from 61 → 15 lines

The original docstring's "multifile_refactor caveat" narrated a prior
thesis failure (the author had a thesis that benchmark-domain weighting
should be learned; the experiment failed because the multifile_refactor
domain's true failure rate was misestimated by 30%, and the learned
weights collapsed to "always pick codeforces_tier1" within 3 cycles).
The lesson learned: cross-domain dispatch weights are now hardcoded
per-domain priors (`DEFAULT_WEIGHTS`), not learned, and the
`multifile_refactor` ratio is held at 0.20 of the mix regardless of
the prior's score. Future work (out of scope for qp91) may revisit
learned weights with a more careful bias correction.

The "Today's RSI cycle runs exclusively against codeforces_tier1 (20
fixed CodeForces problems)" preamble is also moved here: the
cross-domain dispatch is a Phase 6 wire-pop feature and was previously
disabled. As of Phase 2B it's wired but only fires when
`AUTOBENCH_CROSS_DOMAIN=1` is set.

---

## 10. `evaluation/diversity.py` and `evaluation/distillation.py` — trimmed

`diversity.py` (26 → ~15 lines): kept the SACS (Skill-Aware Cosine
Similarity) operationalization note. The arXiv:2602.13103 R-Diverse
MAP (Memory-Augmented Penalty) derivation is detailed in the inline
comments at the SACS class — the opening docstring now just names
the public surface.

`distillation.py` (21 → ~15 lines): kept the autobench.cycle.report.v1
payload summary. The cycle-id correlation logic and dedup rules are
in the inline comments at `CycleDistiller._fold_events`.
