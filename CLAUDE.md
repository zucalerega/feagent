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

- `make_problem.py`: prints a unique, seeded problem statement per (agent number, call number).
- `improver-loop` branch: improver side of the loop.

## Open questions

- What are the CAD design problems, and how is a solution scored?
- Which MCP tools does the agent get (geometry / mesh / BCs)?
- What exactly can an improver change: system prompt, tool descriptions, few-shot examples, harness code?
- What does the evaluator output look like (schema for rankings + summary)?

## Conventions

- Branch off `main` per workstream; don't push directly to someone else's branch.
- Update this file when a decision above gets made.
