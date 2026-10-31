# feagent

Multiple (agent, context+harness improver) pairs that get better at CAD/FEA setup tasks.

## Train stage

1. **Agents** are randomly assigned to one of a set of CAD design problems (many agents per problem).
   Each agent gets an MCP for geometry setup, meshing and boundary conditions. The solver is fixed.
2. **Evaluator** agent reviews all solutions, ranks each agent across key areas and the population as a whole,
   and writes a summary of what worked, what failed, and the good ideas.
3. **Improvers** (one per agent) take that summary and update their agent's context + harness for future problems.

## Per-agent problem statements

```sh
python make_problem.py <agent_number> <call_number>
```

Prints the FEA setup prompt with a mesh strategy, solver, BC tuple and force tuple filled in. The call
number is SHA-256 hashed into a seed that shuffles every possible assignment, and the agent number picks
from that shuffle. The same inputs always give the same prompt, and agents on the same call always get
different assignments. Edit the lists at the top of the script to change what can be assigned.
