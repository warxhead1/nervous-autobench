# autobench continuous-mode daemon

A 24/7 autonomous self-improvement loop for autobench. Runs RSI sessions
indefinitely under a sliding-window request-rate cap (designed for the MiniMax
15k req/5h coding plan, where the cost is effectively $0 — the binding
constraint is request rate, not dollars). At the end of each session, the
improved harness is compared against the canonical one and promoted if it
scores higher. Every 24h, a "biggest surprise" digest is generated and
published to the nervous-bus.

## How it works

```
                   ┌──────────────────────────┐
                   │  ContinuousModeDaemon    │
                   │  .run_forever()          │
                   └─────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        rate-budget       run_one_session     sleep slot
        check()           (RSI cycle)         (window/N)
                                 │
                                 ▼
        ┌────────────────────────────────────────────┐
        │ • load canonical harness from workspace    │
        │ • pick a benchmark (curriculum > codeforces)│
        │ • run SelfImprovingHarness.improve()       │
        │ • append stats, emit session_complete event │
        │ • promote if final > initial               │
        └────────────────────────────────────────────┘
```

A separate `SurpriseDigest` reads the last 24h of `~/.cache/nervous-bus/debug.jsonl`
and flags:

- **Confidently-wrong predictions** — AHE refutations where the improver was
  confident (≥ 0.75) and reality refuted it.
- **Divergence wins** — iterations where the LLM diverged from the rule-based
  heuristic AND the aggregate score improved.
- **Regressions** — sessions whose final score < initial.
- **Verdict-class flips** — new failure modes (or new wins) between iterations.

The most antagonist-valuable signal is the *confidently-wrong prediction*: it
tells the operator exactly where the improver's mental model of the harness is
miscalibrated.

## Quickstart

### Manual

```bash
# Run a single session and exit (good for first-time validation)
python3 -m autobench.continuous once

# Start the daemon in the foreground
python3 -m autobench.continuous run

# Generate today's digest (print + write to workspace/digests/YYYY-MM-DD.md)
python3 -m autobench.continuous digest

# Inspect the canonical harness + last 10 sessions
python3 -m autobench.continuous status
```

### Tweaking rate budget

```bash
python3 -m autobench.continuous \
    --max-requests 14000 \
    --window-seconds 18000 \
    --sessions-per-window 30 \
    run
```

Defaults match the MiniMax 15k/5h plan with a ~7% margin:

| Flag                      | Default | Meaning                            |
|---------------------------|---------|------------------------------------|
| `--max-requests`          | 14000   | hard cap inside the window         |
| `--window-seconds`        | 18000   | 5 hours                            |
| `--sessions-per-window`   | 30      | inter-session pacing target        |
| `--max-iterations`        | 5       | RSI iterations per session         |
| `--improver`              | minimax | improver model (minimax/anthropic/rule_based) |

Each RSI iteration burns ~1 improver request and N case-evaluation requests,
so 30 sessions × 5 iter × 8 cases ≈ 1,200 requests / 5h — well under cap.

### Install as a systemd user service

The repo ships `autobench/continuous.service` as a template. Install it as a
**user** service (no root needed):

```bash
mkdir -p ~/.config/systemd/user
cp autobench/continuous.service ~/.config/systemd/user/autobench-continuous.service

# Sanity-check the unit
systemd-analyze --user verify ~/.config/systemd/user/autobench-continuous.service

# Drop in your API key as a drop-in (don't edit the unit file)
mkdir -p ~/.config/systemd/user/autobench-continuous.service.d
cat >~/.config/systemd/user/autobench-continuous.service.d/env.conf <<'EOF'
[Service]
Environment=MINIMAX_API_KEY=sk-...
EOF

# Enable + start
systemctl --user daemon-reload
systemctl --user enable --now autobench-continuous.service

# Check status / logs
systemctl --user status autobench-continuous.service
journalctl --user -u autobench-continuous.service -f
```

### Halting safely

```bash
# Graceful: finish current session, then stop
systemctl --user stop autobench-continuous.service

# Or, if running interactively: Ctrl-C — the daemon catches SIGINT and finishes
# the active session before exiting.
```

## Workspace layout

```
~/.autobench/continuous/
    harness.json            # canonical (best-so-far) HarnessConfig
    stats.jsonl             # append-only roll of session outcomes
    archive/<ts>.json       # prior canonical harnesses, time-stamped
    digests/YYYY-MM-DD.md   # daily surprise digests
```

Override with `--workspace /path/to/dir`.

## Reading the digest

`workspace/digests/YYYY-MM-DD.md` contains a markdown report:

```markdown
# autobench continuous-mode digest — 2026-05-16

- Sessions: **23** (promoted: **4**)
- Surprises flagged: **6**

## Biggest surprise — `confident_wrong` (score 0.84)

AHE: improver predicted with confidence 0.92 and was REFUTED
(iter 3, model=minimax-m2.7). score_match_ratio=0.15.

## All surprises
- **confident_wrong** [0.84]: ...
- **divergence_win**  [0.93]: LLM diverged from rule-based heuristic and **won** (+0.250 score, iter 1). ...
- **regression**      [0.71]: Score regression in session sess-C…: started at 0.800, ended at 0.500 (Δ=-0.300 over 2 iters).
```

## Cost expectations

Under the MiniMax 15k req/5h plan: **basically free** (the plan is flat-rate).
Outside that plan, expect ~$0.50–$2.00 / day depending on improver model.
`stats.jsonl` reports `total_cost_usd` per session.

## Sibling-aware

The daemon is duck-typed against artifacts from sibling waves:

- **Wave 5-W** (worker agent): if `MiniMaxWorker` is wired into the evaluator,
  generated code is real; otherwise the stub evaluator runs.
- **Wave 5-R** (RateBudgetGuard): constructed by default; falls back to
  conservative pacing if the import fails.
- **Wave 5-A** (adversarial benchmarks): consumed via the curriculum path
  (`~/.autobench/curriculum/today/cases.jsonl`) when present.
- **Wave 5-C** (curriculum): the daemon prefers `curriculum/today/cases.jsonl`
  when it exists, falling back to `codeforces_tier1`.

Nothing crashes if any of the above are missing.
