# Design package SC-40 / LC-03: side push on the jaw of a bonded clamp
<!-- part: split_clamp -->

## 2. System context and the load case

In the sensor-mount variant the clamp body is not bolted through its base: it is bonded flat onto a rigid
base plate, and a side pad (part of the mating tube support) pushes on one side of the jaw. LC-03 is the
largest side push.

- **Base plate.** The clamp's back face (the large flat face on the −z side, at z = −8.1 mm) is bonded over its
  whole area to a rigid base plate. The bond neither slips nor separates.
- **Side pad.** A pad shaped to match bears on the convex outer flank of the jaw on the +x side: the cylindrical
  surface of radius 27.8 mm about the bore axis, between the +x waist notch and the +x prong. The pad presses on
  that whole surface with a uniform 2.1 MPa.
- In this variant the bore is empty (the tube is fitted later) and the clamping screw is not fitted. Nothing
  else touches the part.

## 3. Design review notes (DR-2, sensor-mount variant review)

- The bond line is a structural epoxy, 0.1 mm thick; its compliance is negligible compared with the steel.
- The prongs and the clamping-screw hole are kept from the standard part; whether they can be removed in this
  variant is a separate action (AI-2.4), not part of this request.
- The mating-support designer needs the mean deflection of the pushed flank (to check the pad's contact
  stiffness) and the largest deflection anywhere (clearance to the sensor housing).
- A reviewer asked for the peak stress against the allowable of 220 MPa (yield 355 MPa with a 1.6 factor), with
  a value that does not depend on the mesh.

## 4. Analysis request (action AI-2.1)

Static, linear-elastic analysis of LC-03. Report in SI units.

- **Q1.** The mean displacement of the +x jaw flank where the pad pushes: the displacement magnitude averaged
  over that surface (m).
- **Q2.** The largest displacement magnitude anywhere on the part (m).
- **Q3.** The elastic strain energy stored in the part (J).
- **Q4.** The volume-averaged von Mises stress over the whole part (Pa).
- **Q5.** The peak von Mises stress (Pa), and whether it meets the 220 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
