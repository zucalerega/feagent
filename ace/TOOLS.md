# The meshing agents' tools

What an agent (swarm agent or the single big agent) can do and see in one run. The same text, in
short, is in every role's prompt (`ace/process_brief.py` TOOLS_BRIEF): the agents, the teacher, the
reflector and the organizer, so feedback can name tools and exact calls.

| tool | what it does | what comes back |
|---|---|---|
| `inspect_geometry()` | describes the part (`ace/geometry_info.py`) | bounding box, volume; every face (named group `f<tag>`, type: plane/cylinder/cone/sphere/BSpline, area, centroid, normal if planar, neighbours, whether the statement names it); every edge (tag, length, the two faces, dihedral angle, **re-entrant** or not); the smallest features; plain-language hints ("the clamped face f21 meets f18 at a re-entrant edge") |
| `mesh(h, order, refinements)` | gmsh mesh (`ace/fea.py` make_mesh) | nodes, elements, quality, **repaired_cells** (curved cells straightened automatically) and **folded_cells** (should be 0; a folded cell makes the run invalid), element size near each named group, a hint when over the node cap |
| `solve(setup)` | the fixed solver (feacore skfem) | the scored quantity **plus** (`ace/solve_feedback.py`) every quantity the statement asks for, reactions per support against the applied loads (equilibrium), max displacement and max von Mises with locations, a **reference-free error estimate** (ZZ, relative energy norm) with the **top 5 hotspots** (location, nearest face and re-entrant edge, local element size, share of the error) |
| `view(solve, kind, view)` | renders a picture the agent sees | `kind`: `mesh`, `stress` or `error`; `view`: `iso`, `x`, `y`, `z`, `-x`, `-y`, `-z` |
| `compare(solve_a, solve_b)` | convergence evidence without a reference | how much the stress field changed between the two solves (relative, energy-like) and the 5 regions where it changed most |
| `submit(solve)` | ends the run | the chosen solve is scored |
| `nonlinear_check(mesh)` | only on a task scored by a nonlinear simulation of a physical test (`config.TASKS[...]["score"] == "nonlinear"`, e.g. `abc_00000065_compression`): runs that frozen scoring simulation (`ace/nonlinear.py` run_curve: CalculiX, elastoplastic, large deformation) on mesh number `mesh` (as `mesh` returns it); minutes each, at most `nl_previews` per run, does not use a solve | the force at each checkpoint, simulated vs measured, with the relative error; the mean relative error (the scored error); nodes, DOFs, wall time; the largest plastic strain; hotspots (`peeq` / `s1` peaks with location and nearest faces); `straightened` when curved mid-side nodes had to be straightened for CalculiX |

Refinements in `mesh(...)`: `{group: "f12"}` (a face), `{edge: 37}` (a CAD edge, from
`inspect_geometry`), `{point: [x, y, z]}` or `{box: [xmin, ymin, zmin, xmax, ymax, zmax]}`, each with
`size` (element size there) and `radius` (distance over which the size grows back to `h`; ignored for
a box).

Flow problems (the CFD suite, `ace/README.md`) get `inspect_geometry`, `mesh`, `solve` and `submit`
only (`TOOLS_FLOW`):
- `view`, `compare` and the solve feedback read stress fields, so flow agents don't get them.
- `mesh` takes no `order` (linear finite volumes), but has `quads` for 2D.
- `solve` takes a fluid, a convection scheme and the flow BCs, on the openfoam backend.

## How a careful engineer uses them

1. `inspect_geometry`: find the supported and loaded faces, the **re-entrant edges** (stress
   singularities: the error concentrates there) and small features.
2. Write the setup from the statement; check every load, support and expression against it.
3. First mesh: moderate `h`, quadratic elements, refinements at the re-entrant edges and at the edges
   of supports and loads. Check `folded_cells` is 0 and the node count is within the cap.
4. `solve`: check equilibrium (reactions balance the loads); read the error estimate and its
   hotspots; `view` the error and stress.
5. Next mesh: refine where the hotspots are, coarsen where the error is small; `solve` again.
6. `compare` the two solves: a small change means the field has converged; a large change in one
   region means refine there. Submit the solve with the best accuracy per DOF by this evidence, never
   by closeness to a remembered value.

On a nonlinear (test-scored) task the linear `solve` is a fast proxy only: set it up like the frozen
scoring setup the task states (here displacement-driven, so equilibrium reports the reactions per
support without a balance check), use its error estimate to place refinement, and use
`nonlinear_check` on the most promising meshes to see the scored error before submitting.

None of these tools uses the reference solution, so they work the same on problems with no known
answer.
