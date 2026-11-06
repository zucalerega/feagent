# Design package TC-35 / LC-07: side pull on the −x trunnion shoulder (rig test)
<!-- part: trunnion_cap -->

## 2. System context and the load case

LC-07 is the qualification rig test that checks the cap's stiffness against a sideways pull at a trunnion.
The rig holds the cap at two places and pulls at a third:

- **Stand post.** The cap sits over a rigid stand post that enters the bore from the open bottom end. The
  post's flat end is bonded with structural epoxy, in full contact, to the bore's flat end face under the cap.
  The post has radial clearance everywhere else: it does not touch the bore wall.
- **Guide plate.** A preloaded guide plate on linear bearings bears on the +x boss shoulder (the flat annular
  face at x = +19.5 mm around the +x pin). The plate stops any motion of that face along x, in both directions,
  but offers no resistance to motion in y or z.
- **Tie bracket.** A tie bracket is bonded over the whole −x boss shoulder (the flat annular face at
  x = −19.5 mm around the −x pin) and pulls the cap along −x with 7.8 kN, spread evenly over that face (equal
  force per unit area). The −x pin itself is not touched.
- Nothing else touches the cap. The pins' end faces, both pins' cylindrical surfaces and the outer body are free.

## 3. Design review notes (DR-3, rig readiness review)

- The test engineer confirmed the bracket bond line covers the full shoulder; the bond is far stiffer than the
  steel, so its compliance is not part of the measurement.
- Manufacturing noted that the production parts are zinc-flake coated; the rig part is uncoated. Coating has
  no structural effect.
- The systems lead reminded the team that the in-service loads (LC-01 to LC-06) are covered by a separate
  report and are not part of this request.
- Concern raised: the rig's displacement transducer reads the mean motion of the pulled shoulder; the predicted
  value is needed to set the transducer range and to compare with the test.

## 4. Analysis request (action AI-3.2)

Static, linear-elastic analysis of LC-07 as described above. Gravity, temperature and the bond's own
flexibility are out of scope. Report in SI units.

- **Q1.** The mean displacement of the −x boss shoulder face, where the tie bracket pulls: the magnitude of the
  displacement vector, averaged over the area of that face (m).
- **Q2.** The volume-averaged von Mises stress over the whole cap (Pa). The reliability team uses it as their
  screening metric.

Give each answer with its discretization uncertainty from a mesh convergence study.
