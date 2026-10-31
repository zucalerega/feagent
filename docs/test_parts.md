# Test parts (`feacore/problems/parts/`)

Parametric "CAD design problems" handed to the agents: realistic machine parts with a STEP file and
a statement (`benchmarks/<id>/public/`), plus a reference setup, references and expected convergence
behaviour for the evaluator (`benchmarks/<id>/private/`). Every part is registered at import time
(`feacore.problems.registry`); `all_problems(tags=["part"])` lists them.

```python
from feacore.problems.registry import get, all_problems
from feacore.verification.study import convergence_study
p = get("stepped_shaft_axi_Dd2_rd0p1")
p.statement()                                   # what the agent gets (plus p.export_step(...))
study = convergence_study(p, order=2, backend="skfem", verbose=True)
```

Units are SI throughout (mm-scale parts written in metres). Steel: E = 200 GPa, nu = 0.3.
Aluminium 6061-T6: E = 69 GPa, nu = 0.33.

## Families and ids

| family | ids | formulation | QoIs | reference |
|---|---|---|---|---|
| `stepped_shaft` | `stepped_shaft_axi_Dd{1p5,2}_rd{0p05,0p1,0p2}` | axisymmetric | `peak_stress_fillet` (max s1 on the fillet), `elongation` (mean axial u of the loaded end) | converged numerical, 2 % (handbook value in the notes, see below) |
| | `stepped_shaft_axi_Dd2_sharp` (r = 0) | axisymmetric | `peak_stress_shoulder` (max s1 on the two faces meeting at the corner), `elongation` | none; **trap**: expectation `diverge` |
| | `stepped_shaft_3d_Dd2_rd0p2` | 3D (full bar in the STEP; reference mesh = 90-degree sector) | as above (`uz`) | axisymmetric converged value of the same part, 3 % |
| `notched_bar` | `notched_bar_t{5,10}_r{2,4}` (depth t, root radius r, mm) | plane stress, 5 mm | `peak_stress_notch` (max s1 on the notch roots), `elongation` | converged numerical, 2 % (handbook value in the notes) |
| `l_bracket` | `l_bracket_r{0,2,5,10}` (inside fillet radius, mm) | 3D (reference mesh = half, symmetry plane) | `max_vm_corner` (max von Mises on the fillet; r = 0: on the corner edge), `load_displacement` (mean ux of the loaded face) | none (GCI consensus); `l_bracket_r0` is the **trap**: expectation `diverge` |
| `clevis_lug` | `clevis_lug_2d_D20`, `clevis_lug_2d_D28`, `clevis_lug_3d_D20` | plane stress / 3D (quarter) | `peak_hoop_stress` (max s1 on the bore), `lug_elongation` (mean uy of the loaded half-bore) | none (GCI consensus) |
| `mounting_bracket` | `mounting_bracket_{vertical,horizontal,diag45}` | 3D (half) | `max_vm_web_root` (max von Mises on the web-root fillets), `load_displacement` (mean \|u\| of the load-hole bore) | none (GCI consensus) |
| `pressure_vessel_axisym` | `pressure_vessel_axi_equal`, `pressure_vessel_axi_thin_head` | axisymmetric (half) | `hoop_stress_mid` (stt at the inner surface, mid-length), `peak_von_mises` (whole wall) | Lamé, analytical, 1 %; `thin_head`: `peak_von_mises` is a **trap** (sharp taper kink, expectation `diverge`) |

Load and support models (stated in each description, so every agent applies the same thing):

- Shaft: small end face axially supported (frictionless); uniform axial traction on the large end
  face, P = 31.416 kN (nominal 100 MPa on d = 20 mm).
- Notched bar: end x = -100 mm held in x, one point held in y; P = 20 kN as a uniform traction on
  x = +100 mm.
- L-bracket: both bolt-hole bores fixed; 1 kN in -x spread uniformly over the top face of the upright.
- Lug: base fixed; pin load P = 20 kN as a **cosine radial pressure** p0 cos(theta) on the upper
  half of the bore, p0 = 2P / (pi a b). The pressure vanishes at theta = +-90 deg, so the peak hoop
  stress at the net section is not polluted by a traction jump (a uniform `force` on the half-bore
  would put a jump exactly there).
- Mounting bracket: four bolt bores fixed; the load is a `force` (uniform traction) on the whole
  load-hole bore: 8 kN vertical, 3 kN horizontal (+y, normal to the web), 5 kN at 45 deg in y-z.
- Vessel: 5 MPa on every inner surface; only the axial rigid-body mode needs constraining.

## Traps (deliberate)

- **Sharp re-entrant corners**: `stepped_shaft_axi_Dd2_sharp` and `l_bracket_r0`. The peak stress at
  the corner is singular (sigma ~ rho^-0.46 for a 270-degree corner), so it grows without bound under
  refinement; a GCI study must report `divergent`. An agent that reports a single number without
  noticing is wrong; a good agent says the value is mesh-dependent (or reports a stress at a
  distance / recommends a fillet). The displacement QoIs of the same parts converge normally.
- **Weak kink singularity**: `pressure_vessel_axi_thin_head` has a 3:1 outside taper whose end meets
  the head at a sharp 18-degree kink (a 198-degree re-entrant corner). `peak_von_mises` (whole wall)
  sits on that kink and creeps up by ~8-10 % per halving of h. Subtle: easy to mistake for slow
  convergence.
- **Support singularities**: fixed bolt bores (L-bracket, mounting bracket) are weakly singular where
  the bore meets the plate faces. The stress QoIs are defined on fillets away from them; an agent
  reporting "max von Mises anywhere" picks up the support singularity instead.
- The mounting bracket's gusset toe and top are left sharp (re-entrant 225-degree edges, also weakly
  singular); `max_vm_web_root` deliberately excludes the gusset fillets that run into them.

## References and a finding about the handbook fits

The plan was handbook references (Peterson / Pilkey curve fits, also in Roark Table 17.1) for the
shaft and notched-bar families. The converged FE peak stresses are **8 to 20 % above those fits**,
for every variant (Richardson GCI 0.01-0.4 %). For the axisymmetric shafts two independent solvers
agree to < 0.1 % on the same meshes (skfem P2 isoparametric and CalculiX CAX6). For the notched
bars CalculiX models plane stress as a 3D slab of the given 5 mm thickness and comes out 1.4-2.1 %
higher (a real through-thickness effect at a 2-4 mm notch root), so there the plane-stress value
rests on the skfem study alone. This is consistent with the literature: the
shoulder-fillet tension charts are based on 1960s photoelastic data and FE studies (Tipton, Sorem &
Rolovic, J. Mech. Des. 118 (1996) 321-327) found them unconservative for shoulder fillets in
tension; the opposite-U-notch fit is close to Neuber's nomogram, which body-force solutions
(Nisitani & Noda) show to underestimate by ~10 %.

So the judged reference for those QoIs is `kind="converged_numerical"` (values in
`feacore/problems/parts/_numerical_refs.py`, regenerate with
`python -m feacore.problems.parts._numerical_refs`), tolerance 2 %. The handbook Kt, the handbook
peak stress and the percentage difference are recorded in each reference's `notes`, and
`problem.handbook_reference()` returns the handbook value as a `Reference` for anyone who wants it.
The transcribed fits are checked in the tests (the two coefficient ranges of each fit meet within 1 %
at depth/radius = 2 for the depths used, and the U-notch fit tends to 3.065 for a shallow semicircular notch).
The coefficients could not be re-checked against the book in this session (web pages blocked); two of
the eight shoulder-fillet coefficient expressions were matched against an online reproduction.

The vessel's mid-length hoop stress is the Lamé value (exact far from the heads; the junction
disturbance decays to ~1e-4 at mid-length); the reference study reproduces it to 0.01 %.

## Reference setups

`case(mesh, order)` is the evaluator's reference setup. Where the part and its loads are symmetric,
`build_geometry()` meshes a symmetric portion (`reference_portion`: L-bracket half, mounting bracket
half, 3D shaft and 3D lug quarter, vessel half) and the case adds `symmetry` BCs and halves any total
`force`. The STEP handed to agents (`export_step`) is always the full part, and the tests check that
the reference geometry is exactly the stated fraction of it. Agents are free to use symmetry or not.

Mesh sizing (`apply_sizing(h)`): a Distance/Threshold field refines fillets, notch roots, bores and
corners. Every *size* is proportional to h and the *distances* are fixed, so the size field scales
uniformly and a sequence of h is a systematic (Richardson-compatible) refinement.

## Validation (reference convergence studies, skfem, P2, 3 meshes)

See the table at the end of this file (filled from `convergence_study`; studies saved under
`/tmp/parts/studies/` during development).

## Tests

`tests/test_problems_parts.py`: for every registered part, the reference case validates, the h0 mesh
has every group the case uses, the STEP re-imports as one solid / face and the reference geometry is
the stated fraction of it, and the statement contains no reference values. One coarse solve per 2D
family runs by default; 3D solves need `FEACORE_SLOW_TESTS=1`.
