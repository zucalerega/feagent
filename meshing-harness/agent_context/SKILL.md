---
name: fe-mesh
description: Mesh an existing CAD part (STEP/IGES/BREP) for FEA or CFD with the feagent harness - auto-mesh first, name boundary-condition faces by geometric query, declare bcs, and iterate on mesh.yaml until every quality gate passes. Use whenever a task asks to mesh, set up, or prepare a CAD design for a structural or flow simulation.
---

# fe-mesh: mesh CAD for simulation with feagent

You turn a CAD part into a **checked mesh plus a boundary-condition declaration**. Your deliverable
is a `mesh.yaml` whose `check` passes, its `out/report.json` and `out/REPORT.md`, and a short
report to the user. **Solving is out of scope**: a separate solver owner consumes your exports and
the `bcs` in `report.json`. Do not write solver input decks or estimate stresses or flow fields.

Units: mm, N, MPa.

## Tools

| MCP tool (server `feagent`) | CLI equivalent (`uv run feagent ...`, add `--json` for one JSON doc) |
|---|---|
| `automesh(path, cfd=False)` | `automesh PART [--cfd]` |
| `inspect_geometry(path, size, detail)` | `inspect PATH [--size H] [--detail full]` |
| `find_faces(path, query, expect)` | `faces PATH --query '{...}' --expect N` |
| `render_view(path, view, highlight, query)` | `render PATH --view faces\|groups\|quality [--query ..] [--highlight 1,2]` |
| `mesh(spec_path, scale)` | `mesh SPEC [--scale 2]` (no quality gates; records "unchecked") |
| `check_mesh(spec_path)` | `check SPEC` (mesh + all gates + renders -> report.json, REPORT.md) |
| - | `waive SPEC --reason "..."` (accept a non-passing spec, with a reason) |

CLI exit codes: 0 ok, 1 a gate failed, 2 usage/spec/query/CAD error (message + hint), 3 internal.

## Workflow

1. **Auto-mesh first.** `automesh(part)` (`cfd=True` for flow). It writes an editable
   `<stem>_mesh/mesh.yaml` and checks it. This tells you right away whether the part meshes at all
   and at what size. Read the gate table and the mesher notes, for example thin walls that were
   auto-refined.
2. **Understand the part.** Run `inspect_geometry(part)` (for cfd external, run it on the spec so
   you see the fluid domain). Read FEATURES, PLANAR FACES and HAZARDS. Every hazard line names the
   spec fix. Open `out/REPORT.md` from the automesh run.
3. **Name the physics faces by query.** For each boundary condition (fixed support, load, inlet,
   outlet, body), write a query from geometry: `at` for bbox extremes, `kind`+`radius`+`axis` for
   holes and pins, `feature`, `normal`, `box`/`near` for local faces. Confirm each one:
   - `find_faces(path, query, expect=N)`: N is the number of faces you mean. On a miss, read the
     key-by-key explanation and fix the query; do not drop `expect`.
   - `render_view(path, query=query)`: look at the highlighted faces.
4. **Write the spec.** Add `groups:` (query + `expect`), a `bcs:` section that names those groups,
   and `refine:` around stress raisers (fillets, holes, notches) and flow-critical faces. Start
   from the closest example below.
5. **Check.** `check_mesh(spec)`. Use `mesh(spec, scale=2)` for quick coarse previews while you
   iterate, but only `check_mesh` counts.
6. **Fix failing gates.** Use the gate -> fix table. Change one or two spec keys, then check
   again. The `hint` column names the key to change.
7. **Report to the user.** Give the spec path, PASS/FAIL, every gate that is not `pass` (id,
   value vs threshold), the tet/node counts, the groups with their face counts and the bcs they
   carry, the export files, and any waiver with its reason.

## Rules

- **Never pick faces by a remembered tag.** Tags belong to one gmsh session and change when the
  CAD or the domain changes. Use queries with `expect` in the spec. `ids:` is only for a one-off
  look in `find_faces`/`render_view`.
- **Never claim done while a gate fails.** The Stop hook blocks finishing while any spec you
  meshed or checked is `fail`, `unchecked`, or edited since its last check. If a failure is
  acceptable (for example a coarse preview the user asked for), run
  `feagent waive SPEC --reason "<why>"` and repeat the reason to the user. Editing the spec
  invalidates the waiver.
- Do not relax a `quality:` threshold to make a gate pass unless you can state a physical reason
  (see thin_cup). Say so in the spec comment and in your report.
- Prefer exact `expect` counts. A range `[min, max]` is for genuinely variable features only.
- `analysis: cfd` meshes are linear (order 1). `analysis: fea` defaults to quadratic tets
  (`element: {order: 2}`). Keep them unless you have a reason.

## Spec essentials (mesh.yaml)

```yaml
geometry: ../parts/bracket.step      # relative to this file
analysis: fea                        # fea | cfd
# domain: {type: external, flow_axis: "+x", upstream: 3, downstream: 6, lateral: 2}  # cfd only
groups:                              # name -> {select: <query>, expect: N | [min, max]}
  bolt_holes: {select: {kind: cylinder, radius: 3.25, axis: [0, 0, 1]}, expect: 4}
  load_hole: {select: {kind: cylinder, radius: 5, axis: [1, 0, 0]}, expect: 1}
# default_group: walls               # cfd: faces in no group (default "walls")
size: {global: 2.5, min: 0.1, curvature: 12, thin_wall_layers: 2}
refine:                              # group: <name> or select: <query>
  - {select: {feature: fillet}, size: 1.0, distance: 4}
element: {order: 2, optimize: netgen}
quality: {}                          # thresholds, see table (defaults are fine)
export: [msh, inp]                   # msh msh22 inp vtk unv su2
bcs:
  fea: {fixed: [bolt_holes], loads: [{group: load_hole, force: [0, 0, -500]}], E: 210000, nu: 0.3}
  # cfd: {inlet: [inlet], outlet: [outlet], velocity: 1.0}
```

`force` is the total force on the group in N. `bcs` groups must exist in `groups` (a spec error
otherwise). The old key `solve:` is a spec error: use `bcs:`.

## Gate -> fix table

| gate | fails when | fix (spec keys) |
|---|---|---|
| `spec` | spec changed mid-run (rare) | re-run check |
| `geometry` | CAD does not load / no volume / domain not built | `heal: true`; check the `geometry:` path; for cfd external check `domain:` |
| `groups` | a query matches != `expect` faces, or a face is in 2 groups | read the near-miss explanation; fix the query (tolerances `tol_deg`, `radius_tol`, `within`) or `expect`; confirm with `find_faces` + `render_view` |
| `mesh_generated` | gmsh error, 0 tets, timeout | raise `size.min`; lower `size.curvature`; `heal: true`; try `algorithm: {mesh3d: 10}` (HXT) or `{mesh3d: 1}`; remove an over-fine `refine` |
| `inverted` | any tet has scaled Jacobian <= 0 | lower `size.global`; `algorithm: {mesh3d: 1}`; `element.optimize: netgen`; `heal: true` for CAD defects; for order 2 `refine` tight curved faces |
| `sicn` | min SICN < `quality.min_sicn` | `refine` on the faces the worst tets touch (named in the hint); lower `size.global`; `optimize: netgen`; raise `size.min` if tiny features force slivers |
| `gamma` (warn) | min gamma < `quality.min_gamma` | same as sicn; never fails, so mention it in the report |
| `volume_fidelity` | volume error > `max_volume_error` | raise `size.curvature` (cfd: 24+); lower `size.global`; `refine` curved faces; `order: 2` for fea |
| `area_fidelity` | a group's area error > `max_area_error` | `refine` that group; raise `size.curvature` |
| `connectivity` | mesh pieces != CAD bodies | `heal: true`; lower `size.global`/`size.min` so gaps and thin parts are resolved; check for unfused bodies |
| `thin_walls` (warn) | < `thin_wall_min_elements` elements through a thin wall | raise `size.thin_wall_layers`; `refine` the wall faces to size <= t/2; `order: 2` |
| `non_ortho` (cfd) | max non-orthogonality > `max_non_ortho` (warn > 65 deg) | lower `size.global`; `refine` the region; `optimize: netgen` |
| `skewness` (cfd) | max skewness > `max_skewness` | lower `size.global`; `refine`; `optimize: netgen` |

## Thresholds (defaults, `quality:` keys)

| key | default | gate | meaning |
|---|---|---|---|
| - | 0 | `inverted` | inverted tets allowed |
| `min_sicn` | 0.1 | `sicn` | worst signed inverse condition number (1 = ideal tet) |
| `min_gamma` | 0.05 | `gamma` (warn only) | inscribed/circumscribed radius ratio |
| `max_volume_error` | 0.01 | `volume_fidelity` | relative mesh vs CAD volume error |
| `max_area_error` | 0.02 | `area_fidelity` | relative group area error |
| - | CAD body count | `connectivity` | face-connected mesh components |
| `thin_wall_min_elements` | 2 | `thin_walls` (warn only) | elements through each thin wall (thickness / mean edge) |
| `max_non_ortho` | 70 deg | `non_ortho` (cfd; warns > 65) | face non-orthogonality |
| `max_skewness` | 4.0 | `skewness` (cfd) | cell skewness |

`status` is `pass` when no gate fails. Warnings and skips are allowed, but report them.

## Handoff to the solver (out of scope, but get it right)

- Every group is exported under its own name: physical groups in `.msh`/`.su2`/`.unv`, and NSET
  + ELSET in `mesh.inp`. The volume group is `domain`, a reserved name.
- `mesh.inp` also contains **surface elements** (CPS3/CPS6) for each surface group, so the solver
  can find the faces. They must **not** get a section or material; only the `domain` ELSET is the
  solid. Say this in your report when you export `inp`.
- `report.json` (schema: docs/report-schema.md) carries `bcs`, the gates, the quality numbers and
  the export paths. The evaluator and the solver owner read it, not your chat message.

## Examples (agent_context/examples/, each passes check)

| case | analysis | shows |
|---|---|---|
| `cantilever` | fea | `at:` extremes for clamp and tip load |
| `bracket` | fea | repeated holes by `kind`+`radius`+`axis` (expect 4), fillet refine via `feature:` |
| `plate_hole` | fea | refine around a stress concentration; thin plate (`thin_wall_layers`) |
| `thin_cup` | fea | 1 mm walls: one quadratic layer, threshold lowered **with a stated reason** |
| `tee` | cfd internal | inlet/outlets by `at:`; the rest go to `default_group: walls` |
| `bullet` | cfd external | `domain: external`; tunnel faces by `at:`, body faces by `box:`, body refine |

The STEP files are in `agent_context/examples/parts/`. Copy an example next to your part and
change `geometry:`, the queries and the sizes.
