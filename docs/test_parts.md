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

`convergence_study(problem, order=2, backend="skfem")` with the default sizes (h0, h0/2, h0/4 in 2D;
h0, h0/1.5, h0/2.25 in 3D), one core, 2026-10-31. Stresses in MPa, displacements in micrometres.
For the shafts and notched bars the reference is the study's own converged value (so the error
column would be 0 by construction); the independent checks are CalculiX (shafts: -0.06 to -0.08 %
on the finest mesh) and the handbook column. The 3D shaft is judged against the axisymmetric value.

| problem | QoI | coarse/medium/fine | extrapolated | p | GCI % | reference | handbook fit (fine vs fit) | verdict | study time, fine dofs |
|---|---|---|---|---|---|---|---|---|---|
| clevis_lug_2d_D20 | peak_hoop_stress [MPa] | 315.2/312.8/312.2 | 312.1 | 2.20 | 0.063 |  |  | monotone | 2 s, 65k |
| clevis_lug_2d_D20 | lug_elongation [um] | 33.4/33.4/33.4 | 33.4 | 2.70 | 0.00042 |  |  | monotone | 2 s, 65k |
| clevis_lug_2d_D28 | peak_hoop_stress [MPa] | 544.1/542.1/541.5 | 541.2 | 1.77 | 0.058 |  |  | monotone | 2 s, 76k |
| clevis_lug_2d_D28 | lug_elongation [um] | 64.66/64.67/64.67 | 64.67 | 3.15 | 0.00019 |  |  | monotone | 2 s, 76k |
| clevis_lug_3d_D20 | peak_hoop_stress [MPa] | 328.8/324.4/323.8 | 323.8 | 5.73 | 0.029 |  |  | monotone | 13 s, 111k |
| clevis_lug_3d_D20 | lug_elongation [um] | 33.28/33.29/33.29 | 33.29 | 2.33 | 0.012 |  |  | monotone | 13 s, 111k |
| l_bracket_r0 | max_vm_corner [MPa] | 196.3/250/307.1 |  |  |  |  |  | diverge seen | 39 s, 217k |
| l_bracket_r0 | load_displacement [um] | -460.1/-461.2/-461.7 | -462.2 | 1.98 | 0.13 |  |  | monotone | 39 s, 217k |
| l_bracket_r10 | max_vm_corner [MPa] | 121.2/121.5/121.5 | 121.5 | 11.25 | 7.2e-05 |  |  | monotone | 89 s, 350k |
| l_bracket_r10 | load_displacement [um] | -391.4/-391.7/-391.9 | -392.1 | 1.73 | 0.061 |  |  | monotone | 89 s, 350k |
| l_bracket_r2 | max_vm_corner [MPa] | 195.4/193/190.6 |  |  |  |  |  | divergent | 77 s, 326k |
| l_bracket_r2 | load_displacement [um] | -449.9/-450.3/-450.5 | -450.7 | 1.97 | 0.051 |  |  | monotone | 77 s, 326k |
| l_bracket_r5 | max_vm_corner [MPa] | 138.4/140.7/139.3 |  | 1.41 | 1.7 |  |  | oscillatory | 39 s, 228k |
| l_bracket_r5 | load_displacement [um] | -429.3/-429.7/-429.9 | -430.1 | 1.81 | 0.061 |  |  | monotone | 39 s, 228k |
| mounting_bracket_diag45 | max_vm_web_root [MPa] | 187.9/184.8/191.9 |  |  |  |  |  | oscillatory_divergent | 44 s, 169k |
| mounting_bracket_diag45 | load_displacement [um] | 298.6/302.5/303.6 | 303.9 | 4.34 | 0.12 |  |  | monotone | 44 s, 169k |
| mounting_bracket_horizontal | max_vm_web_root [MPa] | 160.7/162.4/162.7 | 162.7 | 5.94 | 0.03 |  |  | monotone | 41 s, 169k |
| mounting_bracket_horizontal | load_displacement [um] | 265.9/269.1/269.9 | 270.2 | 4.31 | 0.11 |  |  | monotone | 41 s, 169k |
| mounting_bracket_vertical | max_vm_web_root [MPa] | 183.1/192.4/189.8 |  | 4.01 | 0.54 |  |  | oscillatory | 43 s, 169k |
| mounting_bracket_vertical | load_displacement [um] | 68.83/69.65/69.86 | 69.91 | 4.54 | 0.095 |  |  | monotone | 43 s, 169k |
| notched_bar_t10_r2 | peak_stress_notch [MPa] | 487.1/478.7/476.8 | 476.1 | 2.10 | 0.16 | 476.1 (numerical) | 429.6 (+11.0 %) | pass (monotone) | 5 s, 164k |
| notched_bar_t10_r2 | elongation [um] | 86.75/86.75/86.75 | 86.75 | 3.86 | 5.3e-06 |  |  | monotone | 5 s, 164k |
| notched_bar_t10_r4 | peak_stress_notch [MPa] | 361.3/355.5/354.8 | 354.6 | 3.02 | 0.038 | 354.6 (numerical) | 321.9 (+10.2 %) | pass (monotone) | 5 s, 143k |
| notched_bar_t10_r4 | elongation [um] | 87.12/87.12/87.12 | 87.12 | 3.80 | 7.9e-06 |  |  | monotone | 5 s, 143k |
| notched_bar_t5_r2 | peak_stress_notch [MPa] | 353.9/348.5/347.6 | 347.4 | 2.61 | 0.066 | 347.4 (numerical) | 320.4 (+8.5 %) | pass (monotone) | 5 s, 157k |
| notched_bar_t5_r2 | elongation [um] | 81.76/81.76/81.76 | 81.76 | 3.88 | 2.5e-06 |  |  | monotone | 5 s, 157k |
| notched_bar_t5_r4 | peak_stress_notch [MPa] | 271.1/264.6/264.2 | 264.2 | 4.20 | 0.011 | 264.2 (numerical) | 245.4 (+7.7 %) | pass (monotone) | 4 s, 134k |
| notched_bar_t5_r4 | elongation [um] | 81.9/81.9/81.9 | 81.9 | 3.95 | 4e-06 |  |  | monotone | 4 s, 134k |
| pressure_vessel_axi_equal | hoop_stress_mid [MPa] | 65.11/65.1/65.1 | 65.1 | 1.84 | 0.0024 | 65.1 (analytical) |  | pass (monotone) | 1 s, 41k |
| pressure_vessel_axi_equal | peak_von_mises [MPa] | 61.34/61.35/61.35 |  |  |  |  |  | divergent | 1 s, 41k |
| pressure_vessel_axi_thin_head | hoop_stress_mid [MPa] | 65.11/65.1/65.09 | 65.09 | 1.53 | 0.0033 | 65.1 (analytical) |  | pass (monotone) | 1 s, 33k |
| pressure_vessel_axi_thin_head | peak_von_mises [MPa] | 87.31/94.07/103.7 |  |  |  |  |  | diverge seen | 1 s, 33k |
| stepped_shaft_3d_Dd2_rd0p2 | peak_stress_fillet [MPa] | 182.5/181.7/178.9 |  |  |  | 175.3 (numerical) | 162.9 (+9.8 %) | pass (divergent) | 40 s, 211k |
| stepped_shaft_3d_Dd2_rd0p2 | elongation [um] | 39.05/39.05/39.05 | 39.05 | 3.64 | 0.00044 |  |  | monotone | 40 s, 211k |
| stepped_shaft_axi_Dd1p5_rd0p05 | peak_stress_fillet [MPa] | 271.4/270/269.4 | 269 | 1.31 | 0.19 | 269 (numerical) | 226.1 (+19.1 %) | pass (monotone) | 4 s, 118k |
| stepped_shaft_axi_Dd1p5_rd0p05 | elongation [um] | 41.14/41.14/41.14 | 41.14 | 4.03 | 8.3e-07 |  |  | monotone | 4 s, 118k |
| stepped_shaft_axi_Dd1p5_rd0p1 | peak_stress_fillet [MPa] | 213.8/212/211.5 | 211.3 | 1.79 | 0.13 | 211.3 (numerical) | 186 (+13.7 %) | pass (monotone) | 4 s, 107k |
| stepped_shaft_axi_Dd1p5_rd0p1 | elongation [um] | 40.96/40.96/40.96 | 40.96 | 4.03 | 6.3e-07 |  |  | monotone | 4 s, 107k |
| stepped_shaft_axi_Dd1p5_rd0p2 | peak_stress_fillet [MPa] | 172.1/170.9/170.4 | 170.2 | 1.61 | 0.15 | 170.2 (numerical) | 155.4 (+9.7 %) | pass (monotone) | 2 s, 88k |
| stepped_shaft_axi_Dd1p5_rd0p2 | elongation [um] | 40.59/40.59/40.59 | 40.59 | 3.76 | 1.2e-06 |  |  | monotone | 2 s, 88k |
| stepped_shaft_axi_Dd2_rd0p05 | peak_stress_fillet [MPa] | 291.1/290.1/289.5 | 288.5 | 0.71 | 0.44 | 289.5 (numerical) | 242 (+19.6 %) | pass (monotone) | 4 s, 133k |
| stepped_shaft_axi_Dd2_rd0p05 | elongation [um] | 39.76/39.76/39.76 | 39.76 | 3.85 | 1.4e-06 |  |  | monotone | 4 s, 133k |
| stepped_shaft_axi_Dd2_rd0p1 | peak_stress_fillet [MPa] | 223.9/223.6/223 |  |  |  | 223 (numerical) | 196.5 (+13.5 %) | pass (divergent) | 4 s, 124k |
| stepped_shaft_axi_Dd2_rd0p1 | elongation [um] | 39.54/39.54/39.54 | 39.54 | 3.83 | 1e-06 |  |  | monotone | 4 s, 124k |
| stepped_shaft_axi_Dd2_rd0p2 | peak_stress_fillet [MPa] | 176.9/175.9/175.6 | 175.3 | 1.43 | 0.16 | 175.3 (numerical) | 162.9 (+7.8 %) | pass (monotone) | 3 s, 112k |
| stepped_shaft_axi_Dd2_rd0p2 | elongation [um] | 39.05/39.05/39.05 | 39.05 | 3.79 | 1.2e-06 |  |  | monotone | 3 s, 112k |
| stepped_shaft_axi_Dd2_sharp | peak_stress_shoulder [MPa] | 441.2/618.3/850.5 |  |  |  |  |  | diverge seen | 1 s, 59k |
| stepped_shaft_axi_Dd2_sharp | elongation [um] | 39.95/39.96/39.96 | 39.96 | 1.10 | 0.0074 |  |  | monotone | 1 s, 59k |

Reading the table:

- **Traps behave as designed**: `stepped_shaft_axi_Dd2_sharp` (441 -> 618 -> 851 MPa),
  `l_bracket_r0` (196 -> 250 -> 307 MPa) and the thin-head vessel kink (87 -> 94 -> 104 MPa) diverge,
  while the displacement QoIs of the same parts converge with p near 2-4.
- **2D / axisymmetric peaks converge cleanly** (GCI 0.01-0.4 %). Observed orders for peak stresses
  are 0.7-2.6 rather than 2: the peaks sit next to the fillet tangent lines, where the boundary
  curvature jumps, and "max over nodes" adds a sampling error.
- **3D peak stresses are noisy**: max-over-nodes of the recovered stress on a curved fillet jumps by
  1-4 % between meshes (mounting bracket vertical / 45 deg, L-bracket r = 5, 3D shaft), so the GCI
  reports oscillatory or divergent for QoIs that are physically smooth. The 3D shaft's fine value is
  2.0 % above the axisymmetric reference; on the medium 3D mesh skfem's nodal max is 3.6 % above it
  and CalculiX's 1.9 %, so the excess is partly resolution and partly the L2-recovery overshoot at
  the hottest node (to reproduce: mesh `stepped_shaft_3d_Dd2_rd0p2` at h0/1.5 and solve it with
  both backends). `l_bracket_r2` is still pre-asymptotic (195 -> 193 -> 191 MPa at fillet cells
  r/3 .. r/7). Read the 3D reference peak stresses as uncertain by ~3 %; the displacement QoIs of the
  same parts converge at p ~ 2-4 with GCI ~0.1 %.
- The mounting bracket's hottest web-root point (vertical and 45-degree loads) is where the fillet
  wraps around the web's square vertical end edges; the stress drops ~10 % within one element of it,
  which is why that QoI is the noisiest.
- `pressure_vessel_axi_equal` `peak_von_mises` changes by 3 kPa on 61.35 MPa (6e-5) between meshes:
  converged, but above the verification noise floor (1e-6), so it shows up as "divergent".

## Tests

`tests/test_problems_parts.py`: for every registered part, the reference case validates, the h0 mesh
has every group the case uses, the STEP re-imports as one solid / face and the reference geometry is
the stated fraction of it, and the statement contains no reference values. One coarse solve per 2D
family runs by default; 3D solves need `FEACORE_SLOW_TESTS=1`.
