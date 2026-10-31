# FEAgent wingbox: a Sculptured team

A [Sculptured](https://usesculptured.com) team that runs this repo's system end to end on an aircraft wingbox. It
is stage 1 of the whole-aircraft plan: component agents mesh parts in parallel, one agent integrates and solves,
one verifies, and the GPU is used for the solve (cuDSS) and the visualization (VTK through EGL, OpenUSD).

```
Chief engineer ─▶ Geometry builder ─▶ Skins mesher ─┐
      │                    │        ─▶ Spars mesher ─┼─▶ Integrator ─▶ Verifier
      │                    │        ─▶ Ribs mesher  ─┘        │
      └──────────▶ GPU engineer ────────────────────────────▶ │ ─▶ Visualization engineer
                     (every agent reports to the Chief engineer)
```

| Agent | Does | Writes |
|---|---|---|
| Chief engineer (Opus) | interface control document (geometry, parts, interfaces, load cases, DOF budget, acceptance), coordination, final brief | `RUN/plan/` |
| Geometry builder | the wingbox as STEP parts and one touching assembly (gmsh OCC), checked with `feagent inspect` | `RUN/geometry/` |
| Skins, Spars, Ribs meshers | per part: automesh, name interface/load faces by query, size for ≥ 2 quadratic elements through the thickness, iterate until every gate passes; Sculptured re-checks their `report.json` after each turn | `RUN/skins/`, `RUN/spars/`, `RUN/ribs/` |
| Integrator | assembly spec from the meshers' sizing, conformal mesh, two load cases through `feacore.adapters.mesh_harness` and `feacore.cli` (CPU and cuDSS) | `RUN/integrator/` |
| Verifier | equilibrium, box-beam theory tip deflection (within 10 %), 3-mesh GCI (under 5 %) | `RUN/verifier/` |
| GPU engineer | cuDSS in the venv, CPU vs GPU benchmark and the 8 GB ceiling; stretch: a rib as a superelement (Schur complement) vs the monolithic solve | `RUN/gpu/` |
| Visualization engineer | OpenUSD stage of the assembly (gate status per part, von Mises, deformed shape), GPU renders, GLB + three.js viewer on port 8830 | `RUN/viz/` |

`RUN` is `runs/wingbox-<date>/` in this repo (git-ignored). Every agent gets the `feagent-mcp` component (the
mesh harness's MCP server). [`starter/`](starter/) holds the OpenUSD export and EGL rendering spike the
Visualization engineer starts from.

## Install and run

Needs a running Sculptured (`cargo run --release -p sculptured -- serve`, port 4410), Claude Code signed in, and
the harness on `PATH`:

```bash
uv tool install --editable .                                   # provides feagent and feagent-mcp
uv run python teams/feagent-wingbox/build_team.py --install    # writes the JSON, creates/updates the team, validates
```

Open http://127.0.0.1:4410/p/feagent-wingbox-whole-aircraft-stage-1/board and choose **Start team**. This runs
nine real Claude Code agents on your plan (the Chief on Opus, the rest on Sonnet); several run at once.

`feagent-wingbox.json` is generated and holds this checkout's absolute paths: rerun `build_team.py` after changing
a job, or on another machine.

## Design choices

- **Thick parts (skins 20 mm).** The solver is solid-tet only; real 3-4 mm skins need shell elements, the main
  missing feature for real airframes.
- **Conformal assembly for stage 1.** Parts are fragmented into one conformal mesh; the per-part agents decide
  sizing and interface queries and prove each part passes the gates. Superelements per component (the path to a
  whole aircraft on many GPUs) are the GPU engineer's stretch goal.
- **Evidence over claims.** Mesh gates are re-run by Sculptured after every mesher turn; the Verifier checks the
  answer against equilibrium, beam theory and a convergence study before the Chief may call it done.
