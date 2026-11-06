# Design package TC-35 / LC-04: trunnion clamp rings during transport
<!-- part: trunnion_cap -->

## 2. System context and the load case

For transport the actuator is held in a crate. The cap is restrained by the crate's body clamp and a top stop,
and both trunnion pins are squeezed by elastomer-lined clamp rings that keep the actuator from rotating.

- **Body clamp.** A rigid split clamp grips the outer cylindrical surface of the body over its whole area (the
  plain Ø34.5 mm cylinder between the bottom chamfer and the rounded bead; the bosses pass through cut-outs).
  It is bonded: no slip, no separation.
- **Top stop.** A flat, greased stop plate rests on the cap's flat top face (z = 61.25 mm). It is preloaded, so
  the top face cannot move along z in either direction, but it slides on the plate in x and y without friction.
- **+x clamp ring.** An elastomer-lined ring squeezes the whole cylindrical surface of the +x trunnion pin (from
  the shoulder to the pin end) with a uniform contact pressure of 6 MPa.
- **−x clamp ring.** The same kind of ring on the whole cylindrical surface of the −x pin, at 1 MPa (its bolts
  are torqued lower).
- Nothing else touches the cap. The pin end faces, the shoulders, the bead and the bore are free.

## 3. Design review notes (DR-6, packaging review)

- Logistics proposed to raise the −x ring to 6 MPa as well; that is a later load case, not this one.
- The crate's shock loads are handled by the crate's own dampers and are not part of this request.
- The reviewers want to know how much the pins deform under the rings (the pins are bearing journals and their
  roundness matters), how stiff the cap is overall, and the peak stress against the 400 MPa allowable, with a
  mesh-independent value.

## 4. Analysis request (action AI-6.3)

Static, linear-elastic analysis of LC-04. Report in SI units.

- **Q1.** The mean displacement of the +x pin's cylindrical surface: the displacement magnitude averaged over
  that surface (m).
- **Q2.** The same for the −x pin's cylindrical surface (m).
- **Q3.** The largest displacement magnitude anywhere on the cap (m).
- **Q4.** The elastic strain energy stored in the cap (J).
- **Q5.** The volume-averaged von Mises stress over the whole cap (Pa).
- **Q6.** The peak von Mises stress (Pa), and whether it meets the 400 MPa allowable.

Give each answer with its discretization uncertainty from a mesh convergence study, or say why no
mesh-converged value exists.
