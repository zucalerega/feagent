# Mesh system `netgen`

Netgen (`netgen-mesher`, OCC geometry) meshes the STEP directly; the rest is FEAgent tooling. `build.py`
runs `OCCGeometry(step).GenerateMesh(maxh, grading=0.3, curvaturesafety=2)` then `mesh.SecondOrder()`
(tet10, boundary mid-nodes on the CAD surface). Bottom/top are found from the OCC shape by geometry: a face
whose extent along the axis is zero and which sits at the part's min / max along the axis (counts checked
against the contract: 1/1 and 1/2). The mesh is re-ordered to gmsh's tet10/tri6 node order (found geometrically,
not assumed), oriented positive, and written through the gmsh API (`mesh.msh` MSH 4.1 with physical groups
`bottom`, `top`, `domain`; `mesh.inp` from gmsh's CalculiX writer with `*NSET`/`*ELSET` of those names, as the
harness does). Gates: `feagent.quality.assess_mesh` (default `QualitySpec`) on a temporary copy that carries every
CAD face's triangles (so the thin-wall gate can measure); picture: `feagent.render.render_quality`.

- Linear: curved tet10. Elements whose curved mid-nodes would tangle (Jacobian ratio < 0.2 vs straight)
  are straightened (`tangled_elements_straightened` in `report.json`); without this Netgen leaves a few inverted
  elements on thin curved walls.
- Nonlinear: tet10 with all mid-nodes at the corner midpoints (straight edges, CalculiX-safe), coarser `maxh`.
- Extra file `mesh_ccx.inp`: `mesh.inp` without the CPS6 group-surface elements and their ELSETs. CalculiX treats CPS6
  as plane stress and aborts ("gen3delem ... should lie in the z=0 plane") on a 3D model that contains them,
  so use `mesh_ccx.inp` (NSET/ELSET `domain`, NSET `bottom`/`top`) for CalculiX; `mesh.inp` is the harness-shaped file.

## Run

```
# netgen-mesher is not in the project env: pip install it (e.g. uv pip install --target DIR netgen-mesher, drop DIR/numpy*)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=DIR uv run --extra pardiso python experiments/mesh_systems/netgen/build.py \
    benchmarks/gold/abc_00000003_bc0/public/geometry.step z OUT --purpose linear
experiments/mesh_systems/netgen/sanity.py OUT z      # skfem sanity solve (100 N, reaction must be 100)
experiments/mesh_systems/netgen/ccx_smoke.sh OUT 3   # CalculiX smoke, uses OUT/mesh_ccx.inp (axis number 1/2/3)
```

Knobs: `--maxh` (defaults per part/purpose in `DEFAULT_MAXH`), `--grading`, `--curvsafety`, `--opt` (Netgen
`optsteps3d`), `--thin-layers` (per-face `SetFaceMeshsize` = 0.65 t / layers on the harness's thin-wall faces; default
0 = off). Deterministic (identical md5 on a rerun).

## Numbers

| Part | Purpose | nodes | tets | order | DOFs | min SICN / min gamma | gates | build s |
|---|---|---|---|---|---|---|---|---|
| abc_00000003 | linear | 51529 | 28567 | 2 curved | 154587 | 0.153 / 0.153 | all pass except thin_walls warn | 7.8 |
| abc_00000003 | nonlinear | 28537 | 15298 | 2 straight | 85611 | 0.262 / 0.261 | all pass except thin_walls warn | 5.6 |
| abc_00000065 | linear | 59743 | 34523 | 2 curved | 179229 | 0.164 / 0.223 | all pass except thin_walls warn | 9.3 |
| abc_00000065 | nonlinear | 26395 | 14751 | 2 straight | 79185 | 0.247 / 0.188 | all pass except thin_walls warn | 4.4 |

Sanity solve (skfem P2, PARDISO, E = 1879.8 MPa, nu = 0.35, 100 N compression, fixed bottom):

| Part | reaction [N] | top mean \|u\| [m] | peak von Mises [Pa] | strain energy [J] |
|---|---|---|---|---|
| abc_00000003 | 100.0000 | 1.821e-05 | 1.521e+06 | 9.012e-04 |
| abc_00000065 | 100.0000 | 1.072e-04 | 4.711e+06 | 1.835e-03 |

CalculiX smoke (nonlinear meshes, linear-elastic static step, top displaced 1e-5 m): both finish without errors
(5.5 s and 4.2 s).

## Known limits

- Thin walls (2-4 mm) get < 2 elements across at these sizes (thin_walls warn). `--thin-layers 2` fixes the count only at
  2-4x the nodes (over the nonlinear 40k-node budget) and made Netgen's quality erratic (tangled / low-SICN cells), so it is off.
- Peak von Mises is a point value: expect the usual few-percent mesh scatter at re-entrant corners.
- Netgen is not installed by `uv sync`; a `uv sync` by another agent removes it from the shared venv (install to a
  separate `--target` dir and put it on `PYTHONPATH`, dropping the numpy it pulls in).
- The OCC face `.name` setter did not carry into Netgen's BC names here, so groups are assigned geometrically from the
  surface triangles, not via Netgen BC names.
