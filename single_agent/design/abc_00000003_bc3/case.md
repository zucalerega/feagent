# Design package TC-35 / LC-11: shrink-fitting the bore bushing
<!-- part: trunnion_cap -->

## 2. System context and the load case

The bore receives a bronze guide bushing, shrink-fitted at assembly. The fit is done with the cap already
mounted in the assembly fixture by its trunnions. LC-11 is the fit at maximum interference.

- **Trunnion bearing blocks.** Both trunnion pins sit in rigid split bearing blocks that are bonded over each
  pin's whole cylindrical surface, from the shoulder to the pin end. The pins' end faces and the shoulders do
  not touch the blocks.
- **Bushing.** At maximum interference the bushing presses on the whole cylindrical bore wall with a uniform
  contact pressure of 45.6 MPa. The bushing is 0.5 mm shorter than the bore and does not touch the bore's flat
  end face under the cap.
- Nothing else touches the cap. The outer body, the bead, the top face and the open end are free.

## 3. Design review notes (DR-8, assembly review)

- The fixture supplier asked for the axial (x) load that each bearing block sees during the fit, to size the
  block clamping screws.
- The bushing material and its own stresses are the supplier's responsibility; only the cap is analysed.
- Assembly noted that the fit is done at room temperature after the bushing is cooled in liquid nitrogen. The
  thermal transient is not part of this request: analyse the settled fit.
- A reviewer asked for the peak stress in the cap against the 400 MPa allowable, with a value that does not
  depend on the mesh.

## 4. Analysis request (action AI-8.2)

Static, linear-elastic analysis of LC-11 at maximum interference. Report in SI units.

- **Q1.** The elastic strain energy stored in the cap (J).
- **Q2.** The largest displacement magnitude anywhere on the cap (m).
- **Q3.** The volume-averaged von Mises stress over the whole cap (Pa).
- **Q4.** The force that the −x bearing block exerts on the cap along x (N, signed in the global axes).
- **Q5.** The peak von Mises stress (Pa), and whether it meets the 400 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
