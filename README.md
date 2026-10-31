# feagent

Multiple (agent, context+harness improver) pairs that get better at CAD/FEA setup tasks.

## Train stage

1. **Agents** are randomly assigned to one of a set of CAD design problems (many agents per problem).
   Each agent gets an MCP for geometry setup, meshing and boundary conditions. The solver is fixed.
2. **Evaluator** agent reviews all solutions, ranks each agent across key areas and the population as a whole,
   and writes a summary of what worked, what failed, and the good ideas.
3. **Improvers** (one per agent) take that summary and update their agent's context + harness for future problems.

## FEA core (`feacore/`): fixed solver, verification, benchmarks

The solver agents never touch, the Richardson/GCI verification the evaluator scores with, and the
benchmark problems agents are assigned. Shared decisions and conventions are in `CLAUDE.md`.

```bash
bash scripts/setup_gpu_box.sh        # fresh Ubuntu / Brev / DGX Spark box (docs/brev.md)
pip install -e ".[dev]"              # or: an existing Python >= 3.12 env with gmsh's system libs
python -m feacore.cli case.json --out result.json            # solve one case (mesh + BCs + QoIs)
python -m feacore.adapters.mesh_harness <harness out_dir>    # mesh-harness output -> case.json
python -m feacore.ui --host 0.0.0.0 --port 8000              # UI + HTTP solver service (docs/ui.md)
python scripts/validate.py --jobs 8                          # reference GCI studies of every problem
```

- **Input contract:** a gmsh `.msh` file with named physical groups, plus a case JSON (`feacore/case.py`).
- **Backends:**
  - `skfem`: the fixed solver, verified (docs/solver_skfem.md, docs/code_verification.md);
  - `ccx`: a CalculiX cross-check;
  - `nn`: an optional neural surrogate, never the reference.
- **Benchmarks:** `benchmarks/<id>/public/` (statement + STEP, for agents) and
  `benchmarks/<id>/private/` (references and the reference setup, for the evaluator only). The index
  is in `benchmarks/index.json`. Validation tables are in docs/benchmarks_*.md and docs/test_parts.md.

Suggested first swarm problems (3D, so the mesh harness can mesh them):
- `lame_sphere_3d`: exact solution.
- `nafems_le10`: published target.
- `stepped_shaft_3d_Dd2_rd0p2`: handbook stress concentration factor.
- `l_bracket_r5`: no reference; consensus only.
- `l_bracket_r0`: a sharp corner, where peak stress must be flagged as diverging.

These need `symmetry`, `pressure` or `displacement` BCs as well as `fixed` / `force`; the adapter
already accepts them under `bcs.fea`.
