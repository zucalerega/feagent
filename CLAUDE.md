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

- `mesh-harness` branch: the agent's meshing MCP + harness, in `meshing-harness/` (see its README).
  Reads existing STEP/IGES/BREP CAD, names boundary-condition faces by geometric query, meshes with
  gmsh from an editable `mesh.yaml`, and gates mesh quality. No solver (owned separately).
- `improver-loop` branch: improver side of the loop.

## Decisions

- **Agent MCP tools** (`meshing-harness/`, server `uv run feagent-mcp`): `automesh` (zero-config start),
  `inspect_geometry` (digest: features, faces, hazards), `find_faces` (geometric face query with
  expected count), `render_view`, `mesh`, `check_mesh`; prompt `meshing_workflow`. Geometry, mesh and BCs
  are all declared in `mesh.yaml` (`groups:` by query, `bcs:` for fixed/loads or inlet/outlet).
- **Solver hand-off:** `out/mesh.inp` (every group as NSET/ELSET; surface elements for groups must not
  get a section), `out/mesh.msh`/`.su2`, and `out/report.json` → `bcs`.
- **Evaluator input:** `out/report.json` (`schema_version: 1`, per-gate pass/warn/fail with value and
  threshold) — schema in `meshing-harness/docs/report-schema.md`.
- **Improver-editable surface:** `meshing-harness/agent_context/` — `SKILL.md` (workflow prompt),
  `tools.yaml` (MCP tool descriptions), `examples/` (few-shot specs). Give each agent its own copy via
  `FEAGENT_CONTEXT_DIR`. Harness code changes are possible but not required for prompt-level improvement.

## Open questions

- What are the CAD design problems, and how is a solution scored?
- What exactly can an improver change beyond `agent_context/` (harness code? per-agent spec defaults?)
- What does the evaluator output look like (schema for rankings + summary)?

## Conventions

- Branch off `main` per workstream; don't push directly to someone else's branch.
- Update this file when a decision above gets made.
