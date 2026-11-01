# harness_expert: the mesh harness driven like an expert

**Approach.** The exact workflow of `meshing-harness/agent_context/SKILL.md`, frozen into `build.py`: `feagent inspect`
digest (features, thin-wall hazards) -> a `mesh.yaml` per (part, purpose) -> `feagent.check.run_check` (mesh + all
gates + renders) -> `report.json`. No shared code is changed. What the expert spec does that `automesh` does not:

- groups `bottom` / `top` by **normal + box** query with `expect` (1 / 1 for abc_00000003, 1 / 2 for abc_00000065);
  `at: zmin/zmax` is not used (too tight for metre-sized parts);
- `size.thin_wall_layers` for the thin walls (1.9-5 mm), local `refine` at platen faces, fillets / rounds / torus /
  holes (stress concentrators and curvature), and the thin conical foot of abc_00000003 (`kind: cone`,
  distance 12 mm: gmsh's Distance field samples a curve at only 20 points, so a short distance leaves part of a
  thin strip coarse and the `thin_walls` gate warns); coarse `size.global` (6-12 mm) elsewhere;
- tet10, `optimize: netgen`; `bcs.fea` = fixed `bottom`, 100 N compressive on `top`, Alpha Fast E / nu (Pa);
- `mesh.inp` is post-processed: gmsh's CPS6 surface elements and the ELSETs `bottom`/`top` are dropped (CalculiX
  rejects CPS6 in a 3D deck); NSETs and ELSET `domain` stay. `mesh.png` is the harness quality render.

**Knobs** (in `PARTS` in `build.py`): `global`, `min`, `curv` (size.curvature), `layers` (thin_wall_layers),
`min_el` (quality.thin_wall_min_elements), `refine` list (selector, size, distance).

**Linear vs nonlinear.** Linear: 2 quadratic layers through every thin wall (gate threshold 2, default), 375k / 396k
DOFs. Nonlinear: the thin walls cover nearly the whole of both parts, so 2 layers cost 120k-130k nodes, 3x the 40k
budget. So `thin_wall_layers: 1` with `quality.thin_wall_min_elements: 1` (the gate is lowered **deliberately**: ONE
quadratic element through the wall, mean edge <= 0.8 t; it passes at 1.2-1.4x). Honest consequence: bending of the
walls is represented by one quadratic element, adequate for a crush to snap but not a converged peak stress.
Interior edges are straight (`straightness.py`: 0 elements with a curved interior edge; only boundary edges follow
the CAD). Order 1 was not needed.

| part | purpose | order | nodes | tets | DOFs | gates | min SICN | build s |
|---|---|---|---|---|---|---|---|---|
| abc_00000003 | linear | 2 | 124,953 | 72,828 | 374,859 | all 10 pass (thin_walls 2.37 >= 2) | 0.37 | 52 |
| abc_00000003 | nonlinear | 2 | 36,425 | 19,453 | 109,275 | all pass (thin_walls >= 1, lowered) | 0.24 | 19 |
| abc_00000065 | linear | 2 | 132,063 | 80,569 | 396,189 | all pass (thin_walls >= 2) | 0.25 | 87 |
| abc_00000065 | nonlinear | 2 | 39,608 | 22,650 | 118,824 | all pass (thin_walls >= 1, lowered) | 0.26 | 86 |

**Sanity solve** (`sanity.py`, skfem + PARDISO, CPU; 100 N compressive, bottom fixed, E = 1879.8 MPa, nu = 0.35):

| part | reaction | top mean u (m) | peak von Mises (Pa) | solve s |
|---|---|---|---|---|
| abc_00000003 | 100.000 N | 1.843e-5 | 2.077e6 | 29 |
| abc_00000065 | 100.000 N | 1.069e-4 | 7.078e6 | 32 |

**CalculiX smoke** (`ccx_smoke.sh`): both nonlinear `mesh.inp` run a linear-elastic static step (fixed bottom, top
displaced 1e-6 m) with no errors or warnings.

**Known limits.** Nonlinear meshes resolve walls with one quadratic element (see above). Peak von Mises is a
point-stress that scatters between meshes (CLAUDE.md). The nonlinear 65 spec is tuned to be just under 40k nodes and
the SICN gate; holes of r = 3.4 mm need size <= 1.6 mm or the curved tet10 SICN drops below 0.1. Output spec paths in
`mesh.yaml` are absolute (written into OUT_DIR, not committed).
