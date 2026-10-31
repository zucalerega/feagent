# feagent

Multiple (agent, context+harness improver) pairs that get better at CAD/FEA setup tasks.

## Train stage

1. **Agents** are randomly assigned to one of a set of CAD design problems (many agents per problem).
   Each agent gets an MCP for geometry setup, meshing and boundary conditions. The solver is fixed.
2. **Evaluator** agent reviews all solutions, ranks each agent across key areas and the population as a whole,
   and writes a summary of what worked, what failed, and the good ideas.
3. **Improvers** (one per agent) take that summary and update their agent's context + harness for future problems.
