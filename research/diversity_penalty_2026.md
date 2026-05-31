# Diversity Penalty for autobench's RSI Improver Loop

**Bead:** `nervous-bus-b9h`
**Date:** 2026-05-16
**Author:** research-advisor (Claude Opus 4.7 1M)
**Deer cycle:** `b2f07c94-866a-4302-bb7d-e6a1080db61a` (research-advisor agent on nervous-bus; in-flight at time of writing — used as a redundant breadth check, not as a primary citation)

---

## 1. Executive Summary

The autobench RSI improver currently has **no mechanism to detect or prevent strategy collapse**. The rule-based path has only 5 verdict-driven branches, and the LLM path (MiniMax-M2.7 at temperature=0.3) tends to anchor on whichever delta worked last iteration. Over a 10-iteration cycle this gives us a textbook R-Diverse "Surface Diversity Illusion" failure mode: the deltas *look* different (different rationales, different fields touched), but the underlying *strategy* — e.g., "shrink `max_tokens`" — repeats.

**What we should implement, in plain English:** track every proposed `ImprovementDelta` in a persistent memory bank, embed each delta into a low-dimensional "skill signature" (which fields it touched + a hash of *how* it touched them), compute the cosine similarity between a candidate delta and the bank, and subtract a thresholded penalty from the SICA utility *only when the candidate is too similar to recent history*. Combined with a 5–10% mask on the LLM improver's allowable fields per iteration (the "vocabulary dropout" analog for our action space), this gives us R-Diverse's MAP + a curriculum-collapse-resistance layer in ~80 lines of Python.

The single biggest lever is **the skill signature**, not the penalty math. If the signature is wrong (e.g., pure lexical hash of the rationale string), the penalty fires on the wrong axis and the system fights itself. Section 4 nails this down.

**Proposed primary metric:** *Skill-Aware Cosine Similarity (SACS) against a rolling memory bank of the last 20 deltas*, with the penalty term `β · max(0, SACS - τ)²` subtracted from utility. Default `β=0.15`, `τ=0.7`.

**Honest confidence:** ~70% that this lifts autobench's 10-iteration end-state score by ≥3 points on a CodeForces-20 benchmark *if* the skill signature is grounded in field-coverage + value-direction rather than rationale text. ~30% it's neutral or slightly negative (the rule-based improver only has 5 strategies, so the diversity ceiling is low — most of the gain will come from the LLM path). Below ~50% confidence that it helps if we naively embed the JSON delta with a generic encoder; the R-Diverse paper is explicit that **lexical embeddings produce the diversity illusion**, they don't cure it.

---

## 2. R-Diverse Method (extracted in detail)

Source: arXiv:2602.13103 "R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training". Methods extracted via WebFetch of `arxiv.org/html/2602.13103` on 2026-05-16.

### 2.1 The two failure modes R-Diverse names

1. **Local Diversity Illusion** — diversity is enforced *within-batch* (e.g., by self-BLEU on a single sampling step), but across iterations the proposer cycles through the same few modes. The single-batch view looks healthy; the cross-iteration view collapses.
2. **Surface Diversity Illusion** — proposals differ in surface form (different wording, different variable names, different field ordering) but exercise the *same underlying skill*. Self-BLEU and lexical novelty miss this entirely.

Autobench's improver suffers more from #2 than #1 (an LLM at T=0.3 with no memory will write different prose each time but make structurally identical edits).

### 2.2 Memory-Augmented Penalty (MAP)

A persistent memory bank `M` stores embeddings of previously generated questions (in our case: proposed deltas):

```
M_{t+1} ← M_t ∪ { φ(q) | q ∈ Q_t, valid(q) }
```

A proposal is "valid" if its uncertainty score is in `[0.3, 0.8]` — i.e., not trivially easy and not impossible. (In our adaptation: a delta is valid if it actually applies cleanly and produces a measurable score change.)

**Dual-perspective penalty:**

- **Max-similarity penalty** (anti-recycling): `P_max(q, M) = max_{e∈M} cos(φ(q), e)`
- **Mean-similarity penalty** (anti-clustering): `P_mean(q, M) = (1/|M|) · Σ_{e∈M} cos(φ(q), e)`

**Combined MAP loss:**

```
P_MAP(q, M) = γ · [P_max(q, M) − τ_max]_+
            + (1 − γ) · [P_mean(q, M) − τ_mean]_+
```

where `[·]_+ = max(0, ·)` (ReLU). Reported hyperparameters: **γ = 0.5, τ_max = 0.5, τ_mean = 0.25**.

The thresholds matter: the penalty only fires when similarity exceeds them. This is critical — a "soft" penalty that always fires creates a constant pressure away from useful regions, which fights the utility objective.

**Experience replay:** historical samples constitute ratio `ρ = 0.3` of each training batch. This is the *anti-forgetting* counterweight: diversity penalty pushes away from history, experience replay pulls toward it. The system finds equilibrium between exploration and competence.

### 2.3 Skill-Aware Measurement (SAM)

This is the part most people miss. R-Diverse's authors are explicit that **a generic text embedder applied to the proposal will produce the Surface Diversity Illusion**, because surface variation dominates the embedding distance.

Their fix: map each question to *canonical Python solver code* via a code LM (Qwen2.5-Coder-7B at temperature=0), then embed the *code* with a code-aware encoder (Jina-Code-Embeddings-1.5B). The intuition: two questions exercise the same skill iff a code model would write structurally similar solver code for both. Embedding distance in code-space tracks skill distance.

```
φ_SAM(q) = Encoder(Code(q))
```

For autobench, the analog isn't "ask a code LM to solve our delta" — it's "extract the *structural fingerprint* of the delta" (which fields it touches, the direction of each scalar change, whether it's a categorical flip vs a numeric tweak). Section 4.2 spells this out.

### 2.4 Integrated R-Diverse objective

```
R_total(q) = R_uncertainty(q)
           − α · P_rep(q, B; φ_SAM)
           − β · P_MAP(q, M; φ_SAM)
```

with `α = 1.0, β = 1.0`. Both penalties use the skill-aware similarity `φ_SAM`, not lexical BLEU. The two terms are complementary: `P_rep` handles within-batch (Local Diversity Illusion), `P_MAP` handles cross-iteration (the persistent bank).

---

## 3. Other Diversity Techniques Surveyed

### 3.1 Vocabulary Dropout (arXiv:2604.03472)

A complementary technique: at each batch, sample a random Bernoulli(α) mask over the proposer's vocabulary and set masked logits to −∞. Retention `α ∈ {0.75, 0.85}` (drop 15–25% of vocab). Format-critical tokens are protected. The mask is **non-stationary** — resampled every batch — so the proposer can't memorize around it.

They measure diversity at three levels:
- **Lexical:** Self-BLEU.
- **Semantic:** Vendi score, novelty rate (cosine > 0.3 from prior iterations).
- **Functional:** Epiplexity (prequential MDL).

Reported gains: **+4.4 points** on Qwen3-8B math reasoning via R-Zero. The mechanism is dirt-cheap: it costs nothing at inference and prevents the proposer from locking onto a fixed token sequence.

**Translation to autobench:** at each LLM-improver call, mask out a random 15% of the improver's *action vocabulary* — i.e., the list of fields it's allowed to touch (`system_prompt`, `rollout_protocol`, `context_manager`, `tool_surface`, `budget.max_tokens`, `budget.max_time_seconds`, etc.). Forced to vary the action, the improver explores fields it would otherwise ignore.

### 3.2 SAGE Critic (arXiv:2603.15255)

SAGE's Critic isn't a diversity term per se — it's a quality filter. It scores questions/plans on a 10-point scale, normalizes to `[0,1]`:

```
Norm(s) = { s              if 0 ≤ s ≤ 1
          ; (s−1)/9        if 1 < s ≤ 10
          ; 0.5            otherwise }
```

with a quality threshold `α = 0.7`. Below threshold, the difficulty term is zeroed out so "hard but ill-posed" tasks don't get rewarded.

**Relevance to autobench:** less direct, but the pattern of "use a separate critic to gate the improver" is worth filing. The minimax-improver could be its own critic by scoring its proposals before they enter the bank; in practice the rule-based improver can serve as a poor-man's critic ("does this delta even parse?").

### 3.3 Quality-Diversity / MAP-Elites (background)

For completeness: MAP-Elites maintains a *grid* of cells indexed by behavior characteristics; in each cell, only the highest-performing solution is kept. Diversity is enforced by construction (you can only fill so many cells with the same behavior signature) rather than by penalty. CMA-ME extends this with covariance-matrix evolution strategies.

For autobench-scale (10 iterations, single-machine, no population), full MAP-Elites is overkill — but the *behavior signature* idea (Section 4.2) is borrowed directly.

### 3.4 Evolutionary methods as an alternative path

GA/CMA-ES over the harness config space *naturally* maintain a population diversity via mutation + crossover, but the cost is brutal: a population of size 30 over 10 generations is 300 evaluations vs autobench's current 10. Even at CodeForces-20 difficulty (~5 min/eval), that's 25 hours vs ~50 minutes. Not viable for the inner loop; potentially viable as an offline "warm-start" step.

---

## 4. Concrete Proposed Implementation for autobench

### 4.1 New module: `autobench/diversity.py`

A single new class `DiversityTracker` with three methods:

```
record(delta)                  → adds delta's signature to the rolling bank
current_diversity_score()      → 1.0 − P_mean (so higher = more diverse)
penalty_for(candidate_delta)   → returns P_MAP(candidate) as a non-negative float
```

The bank is bounded (last 20 deltas — covers two full RSI runs). Older entries fall off the back.

### 4.2 The skill signature `φ_skill(delta)` — the crucial bit

For autobench, the skill signature is **NOT** a text embedding of the rationale (that's the Surface Diversity Illusion trap). Instead, encode the delta as a fixed-length vector capturing **what changed and in which direction**:

| Component | Dim | Encoding |
|---|---|---|
| `system_prompt_delta` length sign | 1 | `sign(len)` (−1, 0, +1) |
| `rollout_protocol_changed` | 4 | one-hot over {single, iterative, self_revision, monte_carlo}, ⊕ "unchanged" |
| `context_manager_changed` | 5 | one-hot ⊕ "unchanged" |
| `tool_surface_delta` length sign | 1 | `sign(len)` |
| `budget.max_tokens` direction | 1 | `sign(new − old) / log1p(|new − old|)` |
| `budget.max_time_seconds` direction | 1 | same |
| `budget.max_cost_dollars` direction | 1 | same |
| `budget.max_memory_mb` direction | 1 | same |
| Rationale hash bucket | 8 | `sha256(rationale)` first 8 nibbles → one-hot mod 8 (very weak text signal as a tiebreaker) |

Total: **~23 dims**. Cosine similarity over this gives us a *structural* diversity measure. Two deltas that both "shrink max_tokens by 20%" will have identical signatures regardless of how the rationales read — exactly the desired property.

This is the autobench analog of R-Diverse's "embed canonical solver code" trick: we don't care that the LLM wrote different prose, we care that the *action* was different.

### 4.3 Where it plugs in

In `autobench/rsi_loop.py`:

- **Line ~58 (`__post_init__`):** instantiate `self._diversity = DiversityTracker(window=20)`.
- **Line ~125 (after `harness, delta = resolved_improver(...)`):**
  ```python
  div_penalty = self._diversity.penalty_for(delta)
  self._diversity.record(delta)
  ```
- **Line ~137 (in `obs.iteration_complete(...)`):** emit `diversity_score` and `div_penalty` so we can A/B them.
- **In the SICA utility composition** (currently `evaluator.py:545` `utility = w_score * avg_score + w_cost * (1 − avg_cost) + w_time * (1 − avg_time)`): we **do not modify `evaluator.score_harness`** — that would couple the evaluator to RSI state. Instead, in `rsi_loop.py` we compute an *adjusted* utility used **only for convergence and improver-history sorting**:
  ```python
  adjusted_utility = result.aggregate_score - 0.15 * div_penalty
  ```
  The raw `aggregate_score` is what we report and what the A/B compares. The penalty is internal to the RSI loop's *selection* behavior — it changes which deltas are kept and how convergence is detected, not the evaluator's contract.

This separation is intentional: the evaluator must remain a pure function of (harness, cases). Diversity is a *property of the improver's trajectory*, not of any single harness.

### 4.4 The penalty term (matches R-Diverse Eq.)

```
P_MAP(δ, M) = γ · [P_max(δ, M) − τ_max]_+
            + (1 − γ) · [P_mean(δ, M) − τ_mean]_+
```

with `γ = 0.5, τ_max = 0.6, τ_mean = 0.3` (slightly looser than R-Diverse's defaults because our action space is tiny — only 23 dims — so cosines naturally run higher). Penalty weight `β = 0.15` of the aggregate score range, calibrated so that a "fully redundant" delta (P_max = 1.0) loses ~15 percentage points of utility — enough to flip the convergence check but not enough to override a genuinely large score gain.

### 4.5 Optional: action-vocabulary dropout

At each LLM improver call, sample a random 15% subset of the field set `{system_prompt, rollout_protocol, context_manager, tool_surface, budget.max_tokens, budget.max_time_seconds, budget.max_cost_dollars}` and *omit them from the diagnosis prompt*. The improver literally won't see those fields as available levers, forcing it to vary its strategy. Implementation: ~10 lines in `_build_diagnosis_prompt`. Resample mask every iteration.

---

## 5. A/B Experiment Design

### 5.1 Hypothesis

H1: Adding the MAP diversity penalty + action-vocabulary dropout improves the final aggregate_score after 10 RSI iterations on CodeForces-20, vs. baseline (current improver), by **≥3 percentage points** with 80% power.

H0: No effect or negative effect.

### 5.2 Setup

- **Benchmark:** 20 CodeForces problems from `autobench/benchmarks/` (pin specific problem IDs in the experiment config).
- **Improver:** MiniMax-M2.7 (default) — the LLM path is where diversity matters most.
- **Seeds:** 5 random seeds per arm (the MiniMax temperature=0.3 induces some stochasticity).
- **Arms:**
  - A: baseline (current code).
  - B: + DiversityTracker MAP penalty.
  - C: + MAP + action-vocabulary dropout.
  - D: action-vocabulary dropout alone (ablation).
- **Iterations:** 10 per run.
- **Total runs:** 4 × 5 = 20 runs. At ~50 min/run that's ~17 hours wall-clock — overnight feasible.

### 5.3 Primary metric

Final `aggregate_score` at iteration 10. Secondary: AUC of score-vs-iteration curve, count of distinct verdict distributions visited across iterations.

### 5.4 Diagnostic metrics

- `current_diversity_score()` trajectory (should be flat-and-low in A, rising-and-high in B/C).
- Cross-iteration field-coverage entropy: `H({fields touched}) = −Σ p_i log p_i` over the 10 iterations.
- Strategy-frequency distribution: count of each rule-based-strategy branch hit (or, for LLM, count of "which field was the primary change") — Pareto chart over 10 iterations.

### 5.5 Falsification

If arm B's score ≤ arm A's score on ≥3 of 5 seeds, the diversity penalty is **not** carrying its weight on autobench's current improvers — likely because the LLM is already exploring enough at T=0.3, or because our skill signature is too coarse. In that case: re-investigate the signature (Section 4.2) or abandon the penalty in favor of dropout alone (arm D).

---

## 6. Risks and Failure Modes

1. **Penalty too aggressive (β too high, τ too low):** the improver is forced to make *novel-looking* deltas that aren't actually good. SICA score craters. Symptom: aggregate_score in B/C strictly below A from iteration 2 onward. Mitigation: start `β=0.15`, ablate to 0.05 and 0.30.
2. **Penalty too weak (β=0, τ=1):** no effect — equivalent to baseline. Acceptable as a null result, but waste of effort. Mitigation: start with moderate values and only weaken if symptom #1 appears.
3. **Skill signature too coarse (e.g., only 5 dims):** every delta looks similar, penalty fires constantly. Symptom: `current_diversity_score()` near 0 even on early iterations. Mitigation: enrich signature (Section 4.2 already has 23 dims, should be safe).
4. **Skill signature too lexical (e.g., embed rationale string):** Surface Diversity Illusion returns. Symptom: diversity score reads "high" but verdict-distribution-over-iterations is flat. Mitigation: this is *the* R-Diverse trap; the field-direction encoding in Section 4.2 was designed to avoid it.
5. **Improver-fallback collision:** the rule-based improver only has 5 strategies and uses verdict-percentage thresholds. If diversity penalty kicks in, the fallback gets repeatedly suppressed and the system gets stuck in the "balanced adjustments" else-branch. Mitigation: when penalty fires *and* improver is rule-based, the loop should fall through to a *different* rule branch (rotate through strategies), not just emit a low-utility delta.
6. **Bank cold-start:** for the first 2–3 iterations, the bank has too few entries for MAP to be meaningful. R-Diverse handles this by setting penalty to 0 until `|M| ≥ k_min`. Mitigation: skip penalty if `|M| < 3`.
7. **Confounding from MiniMax stochasticity:** MiniMax-M2.7 at T=0.3 introduces noise that may swamp small effects. 5 seeds per arm should give ~80% power for a 3-point effect; if reality is 1-point, we'd need 20+ seeds. Pre-register effect size.
8. **The improver's own JSON parser:** `_parse_llm_improvement` silently drops malformed deltas. If diversity pressure makes the LLM output stranger structures more often, parser dropout rate climbs and net effect is negative. Mitigation: instrument parser dropout per arm.

---

## 7. Citations

- **R-Diverse (primary):** [arXiv:2602.13103](https://arxiv.org/abs/2602.13103) "R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training". Methods extracted from `arxiv.org/html/2602.13103` Section 4 (Method) + Section 5 (Experiments).
- **Vocabulary Dropout:** [arXiv:2604.03472](https://arxiv.org/abs/2604.03472) "Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution". Methods from `arxiv.org/html/2604.03472` Section 3.
- **SAGE:** [arXiv:2603.15255](https://arxiv.org/abs/2603.15255) "SAGE: Multi-Agent Self-Evolution for LLM Reasoning". Critic mechanism from `arxiv.org/html/2603.15255` Section 3.3.
- **Quality with Just Enough Diversity:** [arXiv:2405.04308](https://arxiv.org/html/2405.04308v1) — background on QD with calibrated diversity pressure.
- **EvoLattice:** [arXiv:2512.13857](https://www.arxiv.org/pdf/2512.13857) — MAP-Elites-style internal-population evolution for LLMs.
- **Evolving Excellence:** [arXiv:2512.09108](https://arxiv.org/pdf/2512.09108) — automated optimization of LLM-based agents, evolutionary framing for "agent as black box".
- **SkillRL:** [arXiv:2602.08234](https://arxiv.org/abs/2602.08234) — recursive skill-augmented RL, contemporaneous with R-Diverse, similar concerns.
- **deer cycle:** thread `b2f07c94-866a-4302-bb7d-e6a1080db61a` on nervous-bus (`research-advisor` agent) — used as breadth check, no specific citation needed.

---

## 8. Implementation Sketch (drop-in for `autobench/diversity.py`)

```python
"""Diversity tracking for the autobench RSI improver loop.

Implements R-Diverse-style Memory-Augmented Penalty (MAP) over a rolling
bank of ImprovementDelta signatures. See research/diversity_penalty_2026.md.
"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rsi_loop import ImprovementDelta

# --- hyperparameters (R-Diverse defaults, autobench-tuned thresholds) ---
GAMMA = 0.5         # mix between max-sim and mean-sim
TAU_MAX = 0.6       # max-sim threshold (above → penalty fires)
TAU_MEAN = 0.3      # mean-sim threshold
BETA = 0.15         # penalty weight as fraction of utility
WINDOW = 20         # bank size
MIN_BANK = 3        # skip penalty until bank has at least this many entries

_PROTO_INDEX = {"single": 0, "iterative": 1, "self_revision": 2, "monte_carlo": 3}
_CTX_INDEX = {"full": 0, "budgeted": 1, "semantic": 2, "hierarchical": 3}

def _signed_log(x: float) -> float:
    """Sign-preserving log compression: keeps direction, dampens magnitude."""
    return math.copysign(math.log1p(abs(x)), x)

def skill_signature(delta: "ImprovementDelta", prev_budget: dict | None = None) -> list[float]:
    """Encode a delta as a ~23-dim structural vector. SKILL-LEVEL, not lexical.

    NOTE: We pass prev_budget because the delta only stores the *new* budget;
    we need the *direction* of change for each numeric field.
    """
    vec: list[float] = []
    # 1: system_prompt change direction
    vec.append(math.copysign(1.0, len(delta.system_prompt_delta)) if delta.system_prompt_delta else 0.0)
    # 4 + 1: rollout protocol one-hot (or "unchanged")
    proto_one_hot = [0.0] * 5
    if delta.rollout_protocol_changed:
        # We don't have the new value cleanly on the delta in current code; encode as "changed-unknown".
        proto_one_hot[4] = 0.0  # placeholder; if delta carried the new value, set proto_one_hot[_PROTO_INDEX[v]]=1
        proto_one_hot[4] = 1.0  # "changed" marker
    vec.extend(proto_one_hot)
    # 5 + 1: context manager
    ctx_one_hot = [0.0] * 5
    if delta.context_manager_changed:
        ctx_one_hot[4] = 1.0
    vec.extend(ctx_one_hot)
    # 1: tool surface change direction
    vec.append(math.copysign(1.0, len(delta.tool_surface_delta)) if delta.tool_surface_delta else 0.0)
    # 4: budget direction for each numeric field
    for key in ("max_tokens", "max_time_seconds", "max_cost_dollars", "max_memory_mb"):
        new = delta.budget_delta.get(key) if delta.budget_delta else None
        old = prev_budget.get(key) if prev_budget else None
        if new is None or old is None:
            vec.append(0.0)
        else:
            vec.append(_signed_log(float(new) - float(old)))
    # 8: weak rationale hash bucket (tiebreaker only — should not dominate)
    h = hashlib.sha256(delta.improvement_summary.encode("utf-8")).digest()
    bucket = h[0] % 8
    rat_one_hot = [0.0] * 8
    rat_one_hot[bucket] = 0.25  # downweighted: structural fields should drive cosine
    vec.extend(rat_one_hot)
    return vec

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)

@dataclass
class DiversityTracker:
    """R-Diverse-style MAP over ImprovementDelta signatures."""
    window: int = WINDOW
    _bank: deque = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._bank is None:
            self._bank = deque(maxlen=self.window)

    def record(self, delta: "ImprovementDelta", prev_budget: dict | None = None) -> None:
        sig = skill_signature(delta, prev_budget)
        if any(abs(x) > 1e-9 for x in sig):  # skip no-op deltas
            self._bank.append(sig)

    def penalty_for(self, delta: "ImprovementDelta", prev_budget: dict | None = None) -> float:
        if len(self._bank) < MIN_BANK:
            return 0.0
        sig = skill_signature(delta, prev_budget)
        sims = [_cosine(sig, e) for e in self._bank]
        p_max = max(sims)
        p_mean = sum(sims) / len(sims)
        return (GAMMA * max(0.0, p_max - TAU_MAX)
                + (1.0 - GAMMA) * max(0.0, p_mean - TAU_MEAN))

    def current_diversity_score(self) -> float:
        """1.0 = fully diverse, 0.0 = fully redundant. For observability only."""
        if len(self._bank) < 2:
            return 1.0
        sims = []
        for i, a in enumerate(self._bank):
            for b in list(self._bank)[i+1:]:
                sims.append(_cosine(a, b))
        return 1.0 - (sum(sims) / len(sims) if sims else 0.0)
```

Wiring (in `rsi_loop.py`):

```python
# in __post_init__
self._diversity = DiversityTracker()

# in improve(), after the resolved_improver call
prev_budget = harness.budget.copy()
new_harness, delta = resolved_improver(harness, result)
div_penalty = self._diversity.penalty_for(delta, prev_budget)
self._diversity.record(delta, prev_budget)
adjusted_utility = result.aggregate_score - BETA * div_penalty
# use adjusted_utility for convergence_check; emit raw aggregate_score for reporting
```

Action-vocabulary dropout (in `_build_diagnosis_prompt`):

```python
import random
FIELDS = ["system_prompt", "rollout_protocol", "context_manager",
          "tool_surface", "max_tokens", "max_time_seconds", "max_cost_dollars"]
allowed = [f for f in FIELDS if random.random() > 0.15]
# include only `allowed` fields in the prompt's "fields you may change" list
```

---

## Appendix: Why this matters more than it looks

Autobench's RSI loop is the *substrate* for self-improving agents in this ecosystem. If the loop has a hidden mode-collapse failure, every downstream claim ("autobench improved this harness by X%") is suspect — we cannot tell whether the X% came from real exploration or from the improver finding a single local optimum and parking there for 10 iterations. The R-Diverse paper's whole point is that **without skill-aware diversity tracking, self-improvement metrics are systematically biased upward** (the system reports gains because it's exploiting one strategy harder, not because it's getting smarter).

A 23-dim skill signature is the cheapest possible defense. It costs ~50 µs per iteration and gives us a falsifiable lever. The honest worst case is that the signature is wrong and we learn that empirically in <1 day of A/B; the honest best case is +3 to +8 points of sustained score over 10 iterations, plus a real answer to "is autobench's self-improvement actually self-improvement, or just optimization theater?"
