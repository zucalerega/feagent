# Meshing lessons D (teacher runs: axisymmetric pressure vessels, Lame thick cylinder/sphere, Timoshenko cantilevers)

How to read the score: x = 4A/c. Once g = e_auto/e >= 2, x = 2g/c, so only accuracy per DOF counts
(e x DOFs). Error below the 0.5% floor earns nothing. Aim for e between 0.5% and e_auto/2 with the
fewest DOFs. Below g = 2 the score falls fast (x = g^2/c).

1. When choosing the element order, always use "order": 2, because quadratic elements carry linear
   strain (bending, wall gradients) inside one element. Linear triangles are several times worse per DOF.
   Measured on vessel w20_h30: order 1, h=25 gave e=2.25% at 278 DOFs (0.16x). Order 2, h=60 gave
   e=0.59% at 344 DOFs (1.83x).

2. When an attempt lands below the 0.5% floor, or the automesher is already near it, coarsen the
   whole size field until e is about 0.5-0.6%; never refine further, because DOFs below the floor are
   pure cost. If e_auto is at most about 0.6%, the ceiling is about 1.4x: use a lightly graded mesh at
   about the automesher's DOF count and stop. Measured: on vessel w20_h15, e=0.40% at 528 DOFs gave
   2.87x, while the same refinement idea at e=0.48% and 404 DOFs gave 3.76x. On Lame b/a=1.5, e=0.34% at
   484 DOFs gave 1.01x, and e=0.53% at 334 DOFs gave 1.31x.

3. When a corner is truly singular (re-entrant corner, notch root, wall-thickness step; even a mild
   198 deg corner, with stress about r^-0.17), refine it with a gentle ramp:
   {"corner": k, "size": t/5, "radius": 6t}, where t is the local wall or ligament thickness. The
   growth slope (h - size)/radius should be about 0.5. Add a soft band {"corner": k, "size": 1.2t,
   "radius": 12t} for the decay zone around the corner. Don't use a tiny spot with a steep ramp,
   because the energy error is set by the element size across the whole neighbourhood within 1-2t,
   where the stress is fully 2D, not by the single corner element. The point error follows the band.
   Measured on w20_h15, t=25 u at k7: a spot of size 3 r40 (slope 2.2) plus a band gave field 0.90%
   (3.04x); size 10 r100 (slope 0.8) gave 0.76% (3.63x); size 5 r150 (slope 0.5) plus band 30 r300 gave
   0.48% (3.76x, best).

4. When a thin wall (thickness t much smaller than its length) carries membrane plus bending, mesh it
   with ONE layer of quadratic elements: h of about 2.5-3t, because one quadratic layer already
   represents linear through-wall stress, and h of about 2t forces a second layer that nearly doubles
   DOFs. Measured on w15_h15 (t=18.5 u): h=55 gave 264 DOFs, e 0.75%, 2.68x; h=42 gave 472 DOFs,
   e 0.47%, 2.24x. On w20_h30: h=60 gave 1.83x and h=45 gave 1.62x.

5. When a straight member of CONSTANT section lies more than about 3 decay lengths (shells: 3
   sqrt(R t); solids: about 3 section depths) from every junction, load or support, set h up to about
   6t there. Keep the fine size only near the features, with a curve refinement whose radius spans the
   decay zone, because quadratic elements capture a uniform state exactly. Do NOT do this if the
   section varies along the member (taper, cone): the stress then varies everywhere and the uniform
   h must stay. Measured: w15_h15 (constant wall) with h=110 + {"curve": 2, "size": 50, "radius": 250}
   gave 210 DOFs, 3.17x, against uniform h=55 at 2.68x. The same recipe on w20_h30 (wall tapering
   24->36 u) gave 1.31x, against uniform h=60 at 1.83x.

6. When a boundary is curved and the material behind it is thin (shell heads, fillets with thin
   ligaments, hole edges near another edge), cap the size along it at about R/6 (no more than about
   10 deg of arc per element) with a curve refinement on that arc, because a quadratic element spanning
   more arc curves badly relative to the thickness and the error jumps. Measured: head R=308 u,
   t=18.5 u, with h=120 there (22 deg per element) gave e 2.47% and 0.42x, down from 3.7x. Head R=309
   at size 60 (11 deg) gave 1.01%; size 50 (9 deg) gave 0.79%.

7. When the stress is smooth but decays away from a loaded or concave surface (bore of a thick
   cylinder or sphere, field around a hole), grade outward from that surface:
   {"curve": surface, "size": s, "radius": full width to the far boundary}. Take
   h = s*(b/a)^(4/3) for plane fields decaying like 1/r^2, or h = s*(b/a)^1.5 for axisymmetric or 3D
   fields decaying like 1/r^3 (about 4.3 and 5.2 for b/a = 3). This equidistributes the quadratic
   error (h^2 |u'''|) and halves e x DOFs against uniform. Measured on the Lame cylinder, b/a=3:
   38->165 (ratio 4.3) gave 3.87x; 30->200 (6.7) gave 3.48x; 45->140 (3.1) gave 3.71x; the automesher
   is 1x. On the Lame sphere, b/a=3: 35->180 (5.1) gave 3.95x; 30->210 (7) gave 3.08x. For b/a = 1.5
   the gain is only about 12% (0.53% vs 0.60% at the same 334 DOFs).

8. When the prompt flags a corner as "singular / end of a support" but the support is an axis of
   revolution (u_r = 0), a symmetry line, or a prescribed EXACT displacement (not a rigid clamp), do
   NOT refine it, because the exact field is smooth there: the extra elements cost DOFs and re-mesh
   neighbouring regions. Refine only physically singular spots: re-entrant corners, a rigid clamp
   meeting a free edge (by analogy, treat it as in lesson 3), point loads, sharp notch roots. Measured
   on the Lame sphere: adding {"corner": 5, "size": 12, "radius": 120} moved e from 1.15% to 1.14% and
   the score from 3.95x to 3.75x. An axis strip {"curve": 3, "size": 20, "radius": 60} gave 2.53x
   (point error rose to 1.30%).

9. When a region carries bending with shear across a depth D (beams, ligaments between holes or
   notches, clamped strips), put at least 4 quadratic elements across D: uniform h <= D/4 along the
   whole bending zone. The parabolic shear and the surface stresses resolve only at about 4 elements
   across; below that the point (von Mises) error jumps 4-7x, a cliff rather than a slope. End
   refinements don't replace it. Measured with D=99.5 u: h=26 (D/3.8) gave points 0.29% and 4.78x;
   h=28 (D/3.55) gave 1.99% and 1.48x; h=33 (D/3) gave 1.84%; root+tip refinement at h=40 gave 1.81%.
   With D=287 u: h=72 (D/4) gave 0.29% and 4.35x; h=80 (D/3.6) gave 1.37% and 1.57x.

10. When the point error stalls or jumps (±30%) while the energy error keeps falling, look for a
    missed global threshold (layers across a thickness or depth, the arc cap) instead of stacking
    local refinements, because point stresses scatter with the mesh realization, and a refinement in
    one place re-meshes its neighbours and can make other check points worse. If the field is smooth
    with constant third derivatives (pure bending + shear), go uniform at the largest h that meets the
    threshold, because past g = 2 the score is flat in h. Measured: on cantilever D100, adding a tip
    refinement raised points from 1.36% to 1.81%; uniform h=26 was best. On the sphere, scaling all
    sizes by 0.85 cut the field error from 1.15% to 0.83%, but points stayed at 1.01%.
