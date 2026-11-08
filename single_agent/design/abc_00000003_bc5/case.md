# Design package TC-35 / LC-09: cantilevered test of the cap with an internal spring pad
<!-- part: trunnion_cap -->

## 2. System context and the load case

In the endurance rig the cap hangs from its +x trunnion pin, which is held in a rigid bearing block, while two
loads are applied at once: a sideways pull at the −x shoulder and the preload of an elastomer spring pad inside
the bore. LC-09 is the static peak of that cycle.

- **Bearing block.** The +x pin sits in a rigid bearing block bonded over the pin's whole cylindrical surface,
  from the shoulder to the pin end. The pin's end face and the +x shoulder do not touch the block.
- **Tie bracket.** A tie bracket is bonded over the whole −x boss shoulder (the flat annular face at
  x = −19.5 mm) and pulls along −x with 6.6 kN, spread evenly over that face (equal force per unit area).
- **Spring pad.** Inside the bore, a compressed elastomer pad presses on the bore's flat end face under the cap
  with a uniform 1.5 MPa. It touches nothing else (it has clearance to the bore wall).
- Nothing else touches the cap.

## 3. Design review notes (DR-10, endurance rig review)

- The rig runs at 2 Hz; inertia is negligible, so the static peak is representative.
- The fatigue assessment will be done later by the reliability team from a stress screening metric they choose;
  for now they asked for the volume-averaged von Mises stress.
- A reviewer asked for the peak stress against the 400 MPa allowable, with a value that does not depend on the
  mesh.

## 4. Analysis request (action AI-10.2)

Static, linear-elastic analysis of LC-09. Report in SI units.

- **Q1.** The volume-averaged von Mises stress over the whole cap (Pa).
- **Q2.** The peak von Mises stress (Pa), and whether it meets the 400 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
