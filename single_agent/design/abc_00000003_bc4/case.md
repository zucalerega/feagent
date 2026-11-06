# Design package TC-35 / LC-12: grinding the +x trunnion pin
<!-- part: trunnion_cap -->

## 2. System context and the load case

The +x trunnion pin is finish-ground with the cap mounted on a grinding arbour. A hydraulic steady collet
squeezes the pin while it is ground. LC-12 checks that the cap does not distort the ground pin.

- **Arbour.** The cap is shrunk onto a rigid hollow arbour that fills the bore: the cylindrical bore wall is
  bonded to the arbour over its whole area (no slip, no separation). The arbour does not touch the bore's flat
  end face under the cap.
- **Steady collet.** The collet squeezes the whole cylindrical surface of the +x pin (from the shoulder to the
  pin end) with a uniform 10.8 MPa.
- Nothing else touches the cap: the −x pin, both shoulders, the outer body and the cap's top are free.

## 3. Design review notes (DR-9, machining review)

- Grinding forces themselves are small (a few newtons) and are neglected; the collet pressure dominates.
- The −x pin is ground in a second set-up (LC-13), which is not part of this request.
- The process engineer needs the mean deflection of the +x pin surface under the collet, to set the grinding
  allowance, and the overall stiffness of the set-up.
- A reviewer asked for the peak stress against the 400 MPa allowable, with a value that does not depend on the
  mesh.

## 4. Analysis request (action AI-9.1)

Static, linear-elastic analysis of LC-12. Report in SI units.

- **Q1.** The mean displacement of the +x pin's cylindrical surface: the displacement magnitude averaged over
  that surface (m).
- **Q2.** The elastic strain energy stored in the cap (J).
- **Q3.** The volume-averaged von Mises stress over the whole cap (Pa).
- **Q4.** The peak von Mises stress (Pa), and whether it meets the 400 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
