# ace_sizing: the ACE loop's mesher

**Approach.** `acemesh.make_mesh` reproduces `ace/fea.make_mesh` (branch `ace-harness`) on
`feacore.geometry.gmsh_tools`: STEP imported with gmsh OCC, physical groups `bottom` / `top` / `domain`
named by geometry (planar faces flat along the axis, at the min / max position), one global size `h`, a
`Min` background field of `Distance+Threshold` (group refinements) and `Ball` (point refinements), a size
floor `diag/100`, `MeshSizeFrom{Points,Curvature,ExtendFromBoundary}=0`, order 2 (curved boundary edges).
`build.py` drives it like an ACE agent: coarse look, refine at the hotspots, read the harness gates,
remesh. The gates come from `feagent.quality.assess_mesh`; the sanity solve is feacore/skfem (PARDISO, CPU).

## Knobs (ACE-style)
| knob | value |
|---|---|
| `h` | per part, from the 1 % study below (`DEFAULT_H` in `build.py`, `--h` overrides) |
| refinement at `bottom`, `top` | size `0.5 h` (0.7 h on retry) growing to `h` over radius `3 h` |
| refinement at stress hotspots | 2 point refinements, size `h/3`, radius `4 h`, at the two worst von Mises nodes of a coarse (`2h`) pass |
| floor | `diag/100` (1 mm for both parts) |
| retry loop | up to 14 remeshes at `h * (1, .93, .86, ...)` and contact fraction (0.5, 0.7) until nothing is inverted and the SICN gate passes; else the best (no inverted, max min-SICN) |
| deviation from ACE | optional `--ho-opt` (gmsh HighOrder optimise); off, because it aborts gmsh on part 65 |

## h study (linear sanity solve, tet10, contact refinement 0.5h/3h, top mean |u|)
Part 3 (h mm: u m): 8: 1.7670e-5, 6: 1.7988e-5, 5: 1.7989e-5, 4: 1.8079e-5, 3: 1.8150e-5, 2.5: 1.8193e-5, 2: 1.8255e-5.
Part 65: 8: 1.0533e-4, 6: 1.0530e-4, 5: 1.0606e-4, 4: 1.0725e-4, 3: 1.0709e-4, 2.5: 1.0723e-4, 2: 1.0718e-4.
Cheapest h within 1 % of the next finer level: part 3 h = 6 mm, part 65 h = 4 mm (both under the 50k DOF target:
20k / 42k DOFs). The delivered meshes use h = 3 mm / 3.5 mm to land in the 50k-400k DOF range, after which the
retry loop settled at 2.58 / 2.38 mm. Peak von Mises does not converge (contact-edge singularities; scatter
+-10 %): compare systems on reaction, top displacement and energy.

## Results
| part | purpose | nodes | elements | order | DOFs | gates | build s | reaction N | top mean u m | peak vM Pa |
|---|---|---|---|---|---|---|---|---|---|---|
| abc_00000003 | linear | 29197 | 16062 | 2 | 87591 | pass (SICN min 0.161); thin_walls warn | 8.5 | 100.000 | 1.8193e-5 | 2.74e6 |
| abc_00000003 | nonlinear | 29128 | 16007 | 2 | 87384 | same | 10.9 | n/a | n/a | n/a |
| abc_00000065 | linear | 39940 | 22975 | 2 | 119820 | sicn FAIL (min 0.025, 0 inverted); thin_walls warn | 36.2 | 100.000 | 1.0719e-4 | 5.60e6 |
| abc_00000065 | nonlinear | 31518 | 17848 | 2 | 94554 | same | 38.3 | n/a | n/a | n/a |

Face counts: bottom 1 / top 1 (part 3), bottom 1 / top 2 (part 65). CalculiX smoke (`ccx_smoke.py`, nonlinear
meshes, fixed bottom, top displacement -1e-5, linear elastic static): both exit 0 with no errors, top |u| = 1e-5 as imposed.

## Known limits
- Part 65: min SICN 0.025 on the curved tet10 file (fails the 0.1 gate) near the thin curved walls; no element
  is inverted. Curving boundary edges at this size is the cause (the straight-sided mesh has min 0.25). A
  larger `h` inverts elements; HighOrder optimise aborts gmsh; Netgen optimisation (other system) would help.
- Retry choice is deterministic (fixed sequence, `Mesh.RandomFactor` 1e-9) but depends on the gmsh version.
- `mesh.inp` keeps the harness convention: CPS6 surface elements for `bottom`/`top`. CalculiX treats them as plane
  stress and errors in a 3D deck; `ccx_smoke.py` strips them. Nonlinear meshes are curved C3D10 (boundary edges
  only), accepted by CalculiX with the usual warning in feacore's backend.
- Nonlinear mesh = same recipe; node cap 38k (not binding here).

Run: `uv run --extra pardiso python experiments/mesh_systems/ace_sizing/build.py STEP AXIS OUT --purpose linear|nonlinear`.
