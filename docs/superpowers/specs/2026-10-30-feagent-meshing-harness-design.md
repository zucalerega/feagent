# FEAgent: agentic meshing harness — design

Date: 2026-10-30. Background research: `reports/CAD understanding for agentic meshing.md`.

## Intent

Give an LLM agent (Claude Code first, any MCP client second) the same leverage for meshing that coding
harnesses give it for code: a readable text representation of the part, precise query tools, an editable
"mesh as code" artifact, and a fast, honest test loop. Input is **existing third-party CAD** (STEP, also
IGES/BREP). Output is FEA or CFD volume meshes. The user judges a mesh by running a solver and asking whether
the answer makes sense, so the harness must do the same before claiming success.

## Decisions (from the conversation and research)

- **STEP is the source of truth.** The agent never meshes a reconstruction. Geometry facts come from the
  OpenCascade kernel (via gmsh's OCC API), not from the model's guesses or learned B-rep nets.
- **Geometry digest.** A deterministic, token-budgeted text summary: bbox, volumes, faces with kind/area/
  outward normal/radius/axis, recognized features (holes, bosses, fillets, rounds) collapsed into patterns,
  and a **meshing-hazard report** (small faces, short edges, slivers, thin walls, narrow gaps, small fillets,
  no-volume, multi-volume interfaces) with values relative to the target element size.
- **Faces are selected by geometric query, never by LLM-written selector code or remembered IDs.** Named
  groups bind a name to a query plus an expected count (an assertion). Queries re-resolve on every run, so
  tag renumbering cannot silently move a boundary condition.
- **Mesh as code = a YAML mesh spec** the agent edits with normal file tools; every run is reproducible and
  diffable. The mesher is gmsh (tets, order 1/2). CFD: internal (part is the fluid) and external (box minus
  body) domains. 3D prism boundary layers are **out of scope for v1** (documented gap).
- **Tests for meshes.** A gate suite: geometry, groups, mesh generated, no inverted elements, SICN/gamma,
  volume fidelity, group-area fidelity, connectivity, thin-wall resolution, CFD non-orthogonality/skewness,
  solver smoke test with physics sanity checks, optional 3-grid GCI convergence study.
- **Solver smoke tests are built in** (no external installs): scikit-fem linear elasticity (equilibrium,
  stress-spike-vs-bad-element check) for FEA; potential flow (mass balance, velocity ratio) for CFD.
  Exports (.inp, .msh 2.2, .su2, .unv, .vtk) let real solvers run later.
- **Renders localize, numbers judge.** Face-ID maps, query highlights, group maps, quality clips — each with a
  numeric caption.
- **Packaging:** a Python library + `feagent` CLI (core; JSON output mode), a thin MCP server that shells out
  to the CLI (isolation from gmsh global state/crashes and OCC stdout noise), a Claude Code skill, and a Stop
  hook that blocks finishing while a tracked spec is failing or unchecked (with an explicit waiver escape).
- **Generality:** no specific part/solver was named, so v1 ships six synthetic test parts covering FEA
  (cantilever, plate with hole, bracket, thin-walled cup, sliver hazard part) and CFD (pipe tee internal,
  bullet body external), with analytic or conservation checks.

## Non-goals (v1)

3D boundary-layer prisms, hex meshing, CAD editing/defeaturing operations, CFD Navier–Stokes solves,
running external solvers (CalculiX/OpenFOAM) — exports only, GUI.

## Anti-goals

Reporting "mesh OK" because gmsh exited 0; selecting faces by tag numbers the model memorized; hiding
gate failures from the user.
