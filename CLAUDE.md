# CLAUDE.md

Shared context for anyone (human or agent) working in this repo.

## What this is

ScienceClaw hackathon project. A population of (agent, improver) pairs that get better over time at
setting up FEA problems from CAD designs.

## Architecture

- **Agent**: assigned at random to one CAD design problem (many agents per problem). It uses an MCP to
  set up geometry, mesh it, and apply boundary conditions. The solver is fixed, so agents don't touch it.
- **Evaluator**: a separate agent that reviews every solution. It ranks each agent across key areas,
  scores the population as a whole, and writes a summary of what worked, what failed, and the good ideas.
- **Improver**: paired 1:1 with an agent. It reads the evaluator's summary and updates its agent's
  context + harness so the agent does better on future problems.

One train iteration: agents solve → evaluator ranks and summarizes → improvers update their agents.

## Status

- `improver-loop` branch: improver side of the loop.
- `benchmarks` branch: benchmark suites and the deterministic evaluation harness (`feabench`).
  - `benchmarks/`: verification (10 CAD tasks with exact/published/converged references),
    SimJEB (integrated, data blocked by the network policy here), code_aster reference library,
    CalculiX solver-acceptance suite. See `benchmarks/README.md`.
  - `feabench evaluate <case> <submission_dir>` scores a submission. `docs/eval-design.md`
    explains the scoring; `docs/decisions.md` logs the defaults chosen.

## Decided (provisional, see docs/decisions.md)

- **Problems and scoring.** Cases are `benchmarks/*/<case>/task` (STEP + task.json + task.md);
  hidden references are in `reference/`. Score: S = G·A·(1−0.2(1−E)):
  - G: gates (solver ok, materials, geometry volume, load path);
  - A: log-scaled accuracy credit per QoI;
  - E: token/time efficiency.
  Agents are ranked by within-case Bradley-Terry ratings.
- **Fixed solver:** CalculiX ccx 2.21 (`apt install calculix-ccx`).
- **Submission:** a directory with `model.inp` (ccx deck, task units: mm, N, MPa, tonne, s). The
  harness adds output requests and measures QoIs on the agent's own mesh.
- **Evaluator output:** the harness produces all numbers (`evaluation.json`). The evaluator agent
  adds a failure taxonomy and transferable lessons, and never puts reference values into improver
  summaries.

## Open questions

- Which MCP tools does the agent get (geometry / mesh / BCs)? Its "solve" step should emit the
  submission directory above, and the agent harness should log `agent_cost`: tokens from
  `usage`, wall time, tool calls.
- What exactly can an improver change: system prompt, tool descriptions, few-shot examples, harness code?
- Held-out split (proposal: NAFEMS LE1 + angle bracket, 20 % of SimJEB) and per-case reference costs.
- Access to dataverse.harvard.edu (SimJEB) must be allowed in the environment's network settings.

## Conventions

- Branch off `main` per workstream; don't push directly to someone else's branch.
- Update this file when a decision above gets made.
