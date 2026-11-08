# Design package TC-35 / LC-14: sensor carrier side load during −x pin grinding
<!-- part: trunnion_cap -->

## 2. System context and the load case

The −x trunnion pin is finish-ground on the grinding arbour while a position-sensor carrier, already bonded
inside the cap, is held by the machine's probe. LC-14 combines the steady-collet squeeze on the −x pin with the
probe's side load on the sensor carrier.

- **Arbour.** The cap is shrunk onto a rigid hollow arbour that fills the bore: the cylindrical bore wall is
  bonded to the arbour over its whole area (no slip, no separation). The arbour does not touch the bore's flat
  end face.
- **Steady collet.** The collet squeezes the whole cylindrical surface of the −x pin (from the shoulder to the
  pin end) with a uniform 14.2 MPa.
- **Sensor carrier.** The carrier is bonded over the whole flat end face of the bore, under the cap. The probe
  pushes it sideways, in the plane of that face, with 1.2 kN: +635.5 N along x and −1017.9 N along y, spread
  evenly over the face (equal force per unit area).
- Nothing else touches the cap.

## 3. Design review notes (DR-11, machining review, second set-up)

- The probe force is the maximum the probe can apply before it trips; the nominal is about a third of it.
- The +x pin (already ground in LC-12) must not be marked by the set-up; it is free here.
- The process engineer needs the mean deflection of the −x pin surface under the collet (grinding allowance)
  and the largest deflection anywhere (fixture clearances).
- A reviewer asked for the peak stress against the 400 MPa allowable, with a value that does not depend on the
  mesh.

## 4. Analysis request (action AI-11.1)

Static, linear-elastic analysis of LC-14. Report in SI units.

- **Q1.** The mean displacement of the −x pin's cylindrical surface: the displacement magnitude averaged over
  that surface (m).
- **Q2.** The largest displacement magnitude anywhere on the cap (m).
- **Q3.** The peak von Mises stress (Pa), and whether it meets the 400 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
