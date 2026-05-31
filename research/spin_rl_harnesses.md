# SPIN (Self-Play) + RL-Based Harness Optimization for LLMs 2025-2026

Research compiled from arXiv, GitHub, and web sources.

---

## 1. SPIN (Self-Play) with LLMs

### Core Mechanism

Self-play training for LLMs involves a model playing against iterations of itself to improve. The key insight is that a **single model can serve as both proponent and opponent**, generating training signal from self-generated data without human-curated preference data.

### Key Papers

| arXiv ID | Title | Authors | Venue | Key Finding |
|----------|-------|---------|-------|-------------|
| [2605.11636](https://arxiv.org/abs/2605.11636) | Seirenes: Adversarial Self-Play with Evolving Distractions | Zhang et al. | - | +10.2, +9.1, +7.2 point gains across benchmarks via parameter-shared adversarial self-play |
| [2605.09922](https://arxiv.org/abs/2605.09922) | Team-Based Self-Play with Dual Adaptive Weighting (TPAW) | Li et al. | ACL 2026 Main | Policy model collaborates/competes with historical checkpoints; adaptive weighting for responses |
| [2605.07977](https://arxiv.org/abs/2605.07977) | SPEAR: Self-Play Enhancement via Advantage-Weighted Refinement | Lee et al. | - | Efficient online federated LLM fine-tuning using feedback-guided self-play; constructs contrastive pairs without ground-truth |
| [2605.07465](https://arxiv.org/abs/2605.07465) | SEIF: Self-Evolving RL for Instruction Following | Ren et al. | - | Four-role co-evolution: Instructor, Filter, Follower, Judger |
| [2604.20209](https://arxiv.org/abs/2604.20209) | Scaling Self-Play with Self-Guidance | Bailey et al. | - | SGS: model takes three roles (Solver, Conjecturer, Guide) for formal theorem proving in Lean4 |
| [2603.15611](https://arxiv.org/abs/2603.15611) | Code-A1: Adversarial Evolving of Code LLM and Test LLM | Wang et al. | - | Adversarial co-evolution of Code LLM and Test LLM with opposing objectives |
| [2603.15255](https://arxiv.org/abs/2603.15255) | SAGE: Multi-Agent Self-Evolution for LLM Reasoning | Peng et al. | - | Four-agent closed-loop: Challenger, Planner, Solver, Critic co-evolving from shared backbone |
| [2602.21320](https://arxiv.org/abs/2602.21320) | Tool-R0: Self-Evolving LLM Agents for Tool-Learning | Acikgoz et al. | - | Co-evolves Generator and Solver with complementary rewards using self-play RL |
| [2602.13103](https://arxiv.org/abs/2602.13103) | R-Diverse: Mitigating Diversity Illusion in Self-Play | Li et al. | - | Memory-Augmented Penalty + Skill-Aware Measurement to sustain gains over iterations |
| [2604.03472](https://arxiv.org/abs/2604.03472) | Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution | Dineen et al. | - | +4.4 points at 8B by addressing diversity collapse with vocabulary dropout |

### How Self-Play Improves Agent Harnesses

Self-play provides a **task distribution shift** mechanism. The model generates progressively harder versions of tasks by playing against its own solutions:

1. **Proponent** generates candidate solutions
2. **Opponent** identifies failure modes
3. **Iterative refinement** against self-generated difficulty curriculum

This is distinct from RLHF because it requires no human preference data - the model supervises itself through constructed contrastive pairs.

---

## 2. RL-Based Harness Optimization

### Key Concept

Optimizing the *orchestration layer* (prompting strategy, test selection, solution ranking) rather than fine-tuning the model itself.

### Relevant Work

**Post-Training Enhancements Taxonomy** (arXiv:2312.07413)
- Authors: Tom Davidson et al.
- Categories: tool-use, prompting methods, scaffolding, solution selection, data generation
- Key metric: **compute-equivalent gain** - translates performance improvements into "how much additional training compute would be needed for same improvement"
- Finding: Most enhancements improve benchmark performance by **>5x training compute equivalent**, some by **>20x**
- Fine-tuning costs are typically **<1% of original training cost**
- [arXiv:2312.07413](https://arxiv.org/abs/2312.07413)

**Inference Compute Scaling** (arXiv:2403.18432)
- Coverage scales log-linearly with sample count over four orders of magnitude
- SWE-bench Lite: DeepSeek-Coder-V2-Instruct rises from 15.9% (1 sample) to **56%** (250 samples)
- Limitation: In domains without verifiers, majority voting plateaus beyond ~100 samples

---

## 3. Model-Agnostic Evaluation

No specific papers found on truly model-agnostic evaluation frameworks that transfer across families without retuning. The field lacks consensus on this.

Key challenge: Most evaluation frameworks require some form of calibration per model family due to differing response distributions and reasoning styles.

---

## 4. Harness Transfer Learning

**Known Gap**: Little published work on transferring a harness optimized for one model family (e.g., Gemini) to another (e.g., GPT).

What likely transfers:
- Prompting templates (structural)
- Test selection strategies
- Solution ranking heuristics (if calibrated to task type, not model)

What must be re-tuned:
- Temperature/sampling parameters
- Verifier thresholds
- Self-consistency voting parameters
- Model-specific system prompts

---

## 5. Poetiq Meta-System Replication

**Status**: No direct evidence found of replication attempts or follow-up work on the specific 89.6% -> 93.9% improvement claim.

This appears to reference ARC-AGI competition strategies that were allegedly adopted by the Poetiq meta-system. No arXiv IDs, GitHub repos, or benchmark numbers confirmed for this specific claim.

---

## 6. SICA Codebase Analysis

**Repository**: [MaximeRobeyns/self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent)
**Citation**: ICLR 2025 Workshop paper "SICA: A Self-Improving Coding Agent" by Robeyns, Szummer, Aitchison

### (a) Exact Self-Edit Mechanism

Four-step iterative loop:
1. Evaluate agent on benchmark tasks
2. Store results in archive
3. Run agent on its own codebase for improvement
4. Repeat with updated code

Results organization:
```
results/run_{id}/agent_i/benchmarks/{bench_name}/
├── results.jsonl     # Per-problem results
├── perf.jsonl        # Summary metrics
└── traces/           # Detailed execution traces
```

### (b) Benchmark Evaluation

Benchmarks defined in `base_agent/src/benchmarks/__init__.py`. Supports multiple providers (SWEBench, etc.).

### (c) Utility Function Implementation

Not explicitly documented in README. The system evaluates code quality via:
- Task success rate on benchmark problems
- Archived improvement tracking across iterations

### (d) Infrastructure Requirements

- **Docker isolation** (mandatory - agent can execute shell commands)
- **API keys**: Anthropic (Claude), OpenAI (GPT-4o, o1, o3), Google Gemini, GCP Vertex, Fireworks AI, DeepSeek, Modal
- **Python dependencies**: `base_agent/requirements.txt` + `swebench`
- **Build**: `make image` (or `make image-mac` for Apple Silicon)
- **Interactive test**: `make int` then `python -m agent_code.agent --server true -p "<prompt>"`
- **Main runner**: `python runner.py --id <id> --workers <n>`

### Limitations Noted

The base agent "lacks efficient file editing tools, tree-sitter/LSP integrations, and advanced reasoning structures" - it contains "building blocks to bootstrap these features."

---

## 7. ARC-AGI Harness Lessons

**No specific information found** in web research on what Poetiq specifically borrowed from ARC-AGI competition strategies.

ARC-AGI competition is known for:
- Diverse task distributions requiring broad reasoning
- Emphasis on abstract visual reasoning
- Few-shot learning under limited examples
- Competition winners typically employed ensemble methods and careful prompt structuring

---

## 8. Failed Strategies

### When Recursive Improvement Plateaus

Based on arXiv papers studying self-play failure modes:

1. **Diversity Illusion** (arXiv:2602.13103) - Model appears to self-improve but actually converges to narrow strategy distribution. Mitigation: Memory-Augmented Penalty + Skill-Aware Measurement.

2. **Verifier Dependency** - In domains without automatic verifiers, recursive self-improvement plateaus because the model cannot distinguish good from bad solutions.

3. **Overfitting to Self** - Model generates solutions that satisfy its own internal评判 but fail external evaluation. This is the "self-play trap."

### Specific Failure Modes

- **Majority voting saturation**: Beyond ~100-1000 samples, reward-model-based voting stops improving without external feedback
- **Confirmation bias in self-generated data**: Model amplifies its own biases across iterations
- **Curriculum collapse**: Self-generated difficulty curriculum plateaus when model can no longer distinguish hard from easy problems

---

## Summary Table of Measurable Results

| Paper/Repo | Metric | Result |
|------------|--------|--------|
| Seirenes (2605.11636) | Benchmark gains | +10.2, +9.1, +7.2 points |
| TPAW (2605.09922) | ACL 2026 acceptance | Team-based self-play with adaptive weighting |
| SPEAR (2605.07977) | Federated fine-tuning efficiency | Contrastive pairs without ground-truth |
| R-Diverse (2602.13103) | Iteration stability | Sustains gains via diversity metrics |
| Vocabulary Dropout (2604.03472) | 8B model | +4.4 points |
| DeepSeekMath (2402.03300) | MATH benchmark | 51.7% (self-play GRPO) |
| SICA (ICLR 2025 Workshop) | Self-edit loop | Iterative improvement on SWEBench |
| Inference Scaling (2403.18432) | SWE-bench Lite, 250 samples | 56% (from 15.9% base) |
| Post-Training (2312.07413) | Compute-equivalent gain | >5x, some >20x training compute |

---

## Sources

- [arXiv:2605.11636 - Seirenes](https://arxiv.org/abs/2605.11636)
- [arXiv:2605.09922 - TPAW](https://arxiv.org/abs/2605.09922)
- [arXiv:2605.07977 - SPEAR](https://arxiv.org/abs/2605.07977)
- [arXiv:2604.20209 - Scaling Self-Play with Self-Guidance](https://arxiv.org/abs/2604.20209)
- [arXiv:2603.15611 - Code-A1](https://arxiv.org/abs/2603.15611)
- [arXiv:2603.15255 - SAGE](https://arxiv.org/abs/2603.15255)
- [arXiv:2602.13103 - R-Diverse](https://arxiv.org/abs/2602.13103)
- [arXiv:2602.21320 - Tool-R0](https://arxiv.org/abs/2602.21320)
- [arXiv:2604.03472 - Vocabulary Dropout](https://arxiv.org/abs/2604.03472)
- [arXiv:2402.03300 - DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [arXiv:2312.07413 - Post-Training Enhancements](https://arxiv.org/abs/2312.07413)
- [arXiv:2403.18432 - Inference Compute Scaling](https://arxiv.org/abs/2403.18432)
- [MaximeRobeyns/self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent)