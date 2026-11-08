# Design package TC-35 / LC-02: lift-out of the actuator by the cap
<!-- part: trunnion_cap -->

## 2. System context and the load case

During maintenance the actuator is lifted out of its cradle by a lifting fixture on the cap, while the cap is
still held in the service stand. LC-02 is the worst lift case.

- **Body clamp.** A rigid split clamp grips the outer cylindrical surface of the body over its whole area: the
  plain Ø34.5 mm cylinder between the bottom chamfer and the rounded bead. The bosses pass through cut-outs in
  the clamp and are not gripped. The clamp is shrink-fitted and bonded: no slip, no separation anywhere on that
  surface.
- **Shoulder bracket.** The +x boss shoulder (the flat annular face at x = +19.5 mm) is bolted flat against a
  rigid bracket of the stand, with full-face contact that neither slips nor separates.
- **Lifting fixture.** A lifting plate is bonded to the flat top face of the cap (Ø28.3 mm at z = 61.25 mm; not
  the rounded bead) and pulls straight up, along +z, with 13.8 kN, spread evenly over that face.
- Nothing else touches the cap: the pins, the −x shoulder, the bore and the open end are free.

## 3. Design review notes (DR-5, maintenance tooling review)

- Tooling asked whether the body clamp can stay a shrink-fit clamp, or needs a keyed clamp to carry the lift.
  They need the axial force the clamp must carry.
- The bead was enlarged in rev C for a seal groove on the mating part; the seal is not installed during a lift.
- Paint and the identification plate on the body are not structural.
- A reviewer asked for the peak stress in the cap against the design allowable of 400 MPa (yield 650 MPa with
  a 1.6 factor), and wants a value that does not depend on the mesh.

## 4. Analysis request (action AI-5.1)

Static, linear-elastic analysis of LC-02. Gravity and dynamic effects of the lift are out of scope. Report in
SI units.

- **Q1.** The force that the body clamp exerts on the cap along z (N, signed in the global axes).
- **Q2.** The volume-averaged von Mises stress over the whole cap (Pa).
- **Q3.** The peak von Mises stress in the cap (Pa), and whether it meets the 400 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
