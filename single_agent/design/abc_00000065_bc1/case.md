# Design package SC-40 / LC-06: pressing the clamp onto its locating pins
<!-- part: split_clamp -->

## 2. System context and the load case

In the press-fit variant the clamp body is located by two rigid dowel pins that sit in its waist notches, and
a press pushes the body home along z. LC-06 is the peak press force.

- **Locating pins.** Two rigid cylindrical pins of radius 8.3 mm sit in the two concave waist notches (one on
  each side, where the jaw meets the base block). Each pin is bonded over the whole notch surface with an
  anaerobic retaining compound: no slip, no separation.
- **Press.** A flat press platen bears uniformly on the whole front face (the large flat face on the +z side, at
  z = +8.1 mm) with 21.5 MPa.
- Nothing else touches the part: the back face, the base, the bore and the prongs are free.

## 3. Design review notes (DR-4, press-fit variant review)

- The press stroke is limited by a hard stop on the press, not by the part; the part itself must not yield.
- The dowel supplier needs the forces each pin carries, along the press direction (z) and sideways (x), to
  choose the pin grade and the retaining compound.
- Quality asked for the mean deflection of the pressed face (to set the press's stroke monitoring) and the
  largest deflection anywhere.
- The reliability team asked for the volume-averaged von Mises stress; a reviewer asked for the peak stress
  against the 220 MPa allowable, with a value that does not depend on the mesh.

## 4. Analysis request (action AI-4.2)

Static, linear-elastic analysis of LC-06. Report in SI units.

- **Q1.** The mean displacement of the front face under the platen: the displacement magnitude averaged over
  that face (m).
- **Q2.** The largest displacement magnitude anywhere on the part (m).
- **Q3.** The force that the pin in the +x waist notch exerts on the part along z (N, signed in the global
  axes).
- **Q4.** The force that the same pin exerts on the part along x (N, signed in the global axes).
- **Q5.** The volume-averaged von Mises stress over the whole part (Pa).
- **Q6.** The peak von Mises stress (Pa), and whether it meets the 220 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
