# Design package TC-35 / LC-15: hydrostatic proof of the cap in the test vessel
<!-- part: trunnion_cap -->

## 2. System context and the load case

Before delivery each cap is proof-loaded in a test vessel. The cap hangs from its rounded bead in the vessel's
socket, and two hydraulic bladders press on it from outside. LC-15 is the proof load.

- **Socket.** The rounded bead of the cap (the toroidal round around the top face, Ø37.3 mm overall) is brazed
  into a rigid socket over the bead's whole surface. The flat top face does not touch the socket (there is a
  vent gap).
- **Body bladder.** A hydraulic bladder surrounds the outer cylindrical surface of the body (the plain
  Ø34.5 mm cylinder between the bottom chamfer and the bead; not the bosses) and presses on it with a uniform
  4 MPa.
- **Pin bladder.** A second bladder presses on the whole cylindrical surface of the −x trunnion pin with a
  uniform 2.1 MPa. The +x pin is not loaded.
- Nothing else touches the cap. The bore is vented to the atmosphere, so no pressure acts inside it.

## 3. Design review notes (DR-12, delivery inspection review)

- Quality wants predicted deformations to compare with the strain-gauge and dial-gauge readings taken during
  the proof: the mean radial motion of the body surface under the bladder and of the −x pin surface.
- The bladders are compliant; their pressure is regulated, so it does not depend on the cap's deformation.
- The systems team needs the stored strain energy for the vessel's safety case.
- The reliability team asked for the volume-averaged von Mises stress.

## 4. Analysis request (action AI-12.4)

Static, linear-elastic analysis of LC-15. Report in SI units.

- **Q1.** The mean displacement of the body's outer cylindrical surface under the body bladder: the
  displacement magnitude averaged over that surface (m).
- **Q2.** The mean displacement of the −x pin's cylindrical surface, averaged the same way (m).
- **Q3.** The elastic strain energy stored in the cap (J).
- **Q4.** The volume-averaged von Mises stress over the whole cap (Pa).

Give each answer with its discretization uncertainty from a mesh convergence study.
