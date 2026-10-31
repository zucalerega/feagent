# ace: a minimal ACE loop for the FEA agent

A small, readable implementation of ACE (Agentic Context Engineering, Zhang et al., ICLR 2026,
arXiv 2510.04618) on top of `feacore`. The agent never changes; what improves is a **playbook**
of short lessons pasted into its system prompt.

## The loop in plain words

Each generation:

1. **Generator** (`agent.py`, Sonnet). Several copies of a small tool-use agent each attempt an
   FEA problem: `inspect_geometry` → `mesh(h, order, refinements)` → `solve(setup)`. The system
   prompt is a short, deliberately non-expert base prompt plus the current playbook. The agent
   cites bullets it uses, e.g. `[using mesh-0003]`.
2. **Checks** (`checks.py`, no LLM). Each run's final solve is scored:
   `score = log10(e_uniform / e_agent)`. `e_agent` is the relative error of one scored quantity
   (QoI), computed by our solver from the agent's mesh and setup. `e_uniform` is the error of a
   naive uniform linear mesh with the same node budget. 0 = naive, +1 = 10x more accurate.
   Gates score -2: no successful solve, mesh over the node cap, element quality < 0.1. A setup
   diagnostic (missing BCs, sign flips, wrong formulation) is recorded for the Reflector.
3. **Reflector** (`reflector.py`, Opus). One call per training problem: all runs ranked best to
   worst, with checks and condensed traces. Returns validated JSON: where best and worst
   diverged, root causes, 1-3 lessons ("When X, do Y, because Z.") with a scope, and a
   helpful/harmful/neutral vote on every cited bullet.
4. **Curator** (`curator.py`, Sonnet). Turns that into delta ops: `ADD` a bullet, or `VOTE` on
   one. It drops `one_off` lessons and checks each new lesson against the existing bullets in
   its section, so a near-duplicate becomes a vote instead of a new bullet. It only proposes.
5. **Merge** (`merge.py`, no LLM). Applies the ops, deduplicates again with rapidfuzz, prunes
   bullets where harmful ≥ helpful + 2, and saves the new playbook version plus a diff.

Generation 0 runs with an empty playbook as the baseline. The held-out problem (`nafems_le1`)
runs every generation but is never reflected on, so its score shows whether lessons transfer.

Why the node cap: without it, a dense mesh is always more accurate, and nothing is learned.
With a cap, accuracy depends on skill: element order, where to refine, getting the BCs right.

## Run it

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev,ace]"

# free: canned LLM responses, exercises the whole loop
.venv/bin/python -m ace.run_loop --mock
.venv/bin/pytest tests/test_ace.py

# real: reads ANTHROPIC_API_KEY from the environment (never put it in a file)
.venv/bin/python -m ace.run_loop --runs-per-problem 3 --generations 3 --budget-usd 30
```

Useful flags: `--problems a,b` (train), `--holdout c`, `--concurrency 4`, `--top-n 30`,
`--playbook path.json` (start from an earlier playbook), `--out dir`.

The budget is checked before each agent run is started and after each Reflector call. It can
overshoot by the runs already in flight (at most `--concurrency` runs). Mock mode logs fake token
counts so the cost code is exercised; those dollars are not real.

## Watch it evolve

Everything goes to `ace_runs/<timestamp>/` (gitignored):

| file | what |
|---|---|
| `summary.md` | per-generation score table (train vs held-out) and bullets added/voted/pruned |
| `scores.png` | mean score per generation, one line per problem |
| `playbook_gen{N}.json` | the playbook generation N's agents used |
| `diff_gen{N}.md` | what changed from N-1 to N, plus the rendered playbook |
| `gen{N}/reflection_<problem>.json` | the Reflector's analysis |
| `gen{N}/runs/<run_id>.json` | full run record: trace, meshes, solves, cited bullets, checks |
| `cost.jsonl` | one line per LLM call: tokens and estimated $ |

## Knobs that matter (`ace/config.py`)

- `MODELS`: one line per role. The Reflector is the step most worth a strong model.
- `PRICES`: per-model $/MTok used for the cost log. **Verify before a real run.**
- `TASKS`: problem → scored QoI and node cap. `ace/calibration.json` caches `e_uniform`; it is
  recomputed automatically when a cap changes.
- `RUNS_PER_PROBLEM`: more runs = more contrast for the Reflector, less noisy scores.
- `GENERATIONS`: how many playbook updates you get to see.
- `TOP_N_BULLETS`: how much of the playbook the agent sees.
- `MAX_SOLVES`, `MAX_TURNS`: per-run limits (a stand-in for a wall-clock budget later).
- `EFFORT`, `MAX_TOKENS`: per-role thinking effort and output cap.

## Not done yet

- Mesh-refinement convergence check (GCI) as a diagnostic: TODO in `checks.py`.
- The agent works on the problem's own named geometry, not raw STEP. Swapping its tools for
  the `feagent` mesh-harness MCP is the natural next step.
