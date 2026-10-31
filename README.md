# feagent

Multiple (agent, context+harness improver) pairs that get better at CAD/FEA setup tasks.

## Train stage

1. **Agents** are randomly assigned to one of a set of CAD design problems (many agents per problem).
   Each agent gets an MCP for geometry setup, meshing and boundary conditions. The solver is fixed.
2. **Evaluator** agent reviews all solutions, ranks each agent across key areas and the population as a whole,
   and writes a summary of what worked, what failed, and the good ideas.
3. **Improvers** (one per agent) take that summary and update their agent's context + harness for future problems.

## Benchmarks and evaluation (`feabench`)

| | |
|---|---|
| [`benchmarks/`](benchmarks/README.md) | benchmark suites: verification tasks, SimJEB, code_aster references, CalculiX solver acceptance |
| [`docs/eval-design.md`](docs/eval-design.md) | how solutions are scored: accuracy vs speed vs compute cost |
| [`docs/decisions.md`](docs/decisions.md) | decisions taken while building this, with rationale |
| `feabench/` | deterministic harness: runs the fixed solver (CalculiX 2.21), measures QoIs on the agent's mesh, scores |

```bash
sudo apt install calculix-ccx          # the fixed solver (2.21 on Ubuntu 24.04)
pip install -e . h5py pytest           # gmsh, meshio, numpy, scipy come with it
pytest                                 # unit + end-to-end tests
feabench list                          # agent-facing cases
feabench reference verification/lame_cylinder --level 2   # scripted baseline, scored
```
