# Teacher C meshing lessons (wsp recipe format)

Evidence base: 44 scored attempts on 8 training parts. Six are strips with a central hole (d/W 0.15-0.6,
hole radius r = 36-146 u, net-section ligament 243-97 u) and two are Kirsch infinite-plate quarters (r = 85 and 198 u).
Score reminder: once g >= 2, x_auto = 2g/c, so what is scored is e x DOFs (e floored at 0.5%).
Every part has a 1000 u diagonal, so the sizes below in u carry over between parts.
Notation: "e x N" = e[%] x DOFs, where lower is better. "Slope" k = (h - size)/radius.

1. **When choosing the element order for a stress problem, use `"order": 2`.** The P2 error falls about as
   1/N, but the P1 error falls only as 1/sqrt(N). On the d/W = 0.4 plate, order 1 (hole 8 u, ligament 14 u) gave e = 2.76% at 260 DOFs (x 1.81).
   Order 2 gave e = 0.84% at 472 DOFs (x 5.45). Even on the mildest part (d/W = 0.15), order 1 only reached x 6.42 versus 8.09.

2. **When most of the part carries uniform or slowly varying stress (long bars, regions far from features and
   load ends), set `"h": 400` (at least the thinnest wall) and let the refinement ramps set every size.** P2
   represents a uniform stress exactly, so elements there buy nothing. On d/W = 0.15, raising h from 170 to 400 with the same ramp kept e = 0.51% and
   cut the DOFs from 438 to 422. Capping h at 150 (d/W = 0.25) or 130 (d/W = 0.4) left e unchanged (0.60%) or slightly worse (0.84 to 0.89%), while adding 7-12% DOFs.

3. **When the stress raiser is a smooth curved edge (hole, notch root, fillet), refine the whole curve
   (`{"curve": <arc tag>, ...}`), not corners at its ends or at the peak point.** Around such an edge the stress
   curvature is large all along it (hoop stress at the ends, shear mid-arc), so the energy error is spread over the
   edge. Adding the peak corner k6 at 10 u to the best d/W = 0.4 recipe left e at 0.84% and added 6% DOFs (x 5.45 to 5.10).
   Adding the far end k2 at 12 u on d/W = 0.45 left e at 0.95-0.96% for 8% more DOFs. Coarsening the arc from 18 to 24 u while refining the
   peak region raised e from 0.95% to 1.18%. Corners flagged "end of a support" where the support is a *symmetry* condition are not
   singular, because the mirror image keeps the field smooth, so do not spend DOFs on them. Keep corner refinements for real clamps and
   re-entrant corners. That last point is theory; I did not test it here.

4. **When setting `size` on that curve (radius r in u from the prompt), start at about r/3 for r ~ 35 u, r/4-r/5 for
   r ~ 50-100 u, and r/6-r/8 for r > 100 u.** That works out to about 12-20 u for r = 35-150 u in a strip, or about 2.4*sqrt(r) u for a
   feature inside a large plate. The relative error of a mesh with a fixed number of elements per radius grows with the feature's share of the
   part, so larger features need relatively finer edges to reach the 0.5% floor. The best edge sizes were 12 u (r 36), 12 u (r 48), 13 u
   (r 61), 18 u (r 97), 14 u (r 109), 22 u (plate, r 85) and 35 u (plate, r 198). Coarser starts lost accuracy: e x N rose from 190 to 234 (r 36, 12 to 16 u)
   and from 220 to 257 (r 48, 12 to 16 u). The data cover r = 36-198 u only. For sharper features, keep size <= r/3.

5. **When choosing `radius`, treat it as the distance where the ramp reaches h, not the size of a "zone".** Pick it so
   the slope k = (h - size)/radius is about 0.5 for a compact feature in a long or strip-like part, and about 1.8*size/r when the
   whole part is disturbed (feature in a large plate). With h = 400 that gives radius = (400 - size)/k, which is about 770 u.
   A smooth raiser's disturbance decays like (r/rho)^2, so the ideal P2 size grows like rho^(4/3). A linear ramp of that slope matches
   it over 1-4 radii. Too steep starves the decay zone, and too gentle overspends. Steep ramps scored worst: slope 1.0 gave e x N 310 versus 190-193 at
   0.45-0.55 (d/W = 0.15), 0.6 gave 248 versus 220 at 0.5 (d/W = 0.2), and in the plates 0.55 gave 263 versus 230 at 0.45 (r 85) and 0.4 gave 383 versus 319 at 0.3 (r 198).
   Gentler than 0.5 did not help in strips: 0.4 gave 279 versus 263 at 0.5 (d/W = 0.25), and 463 versus 444 (d/W = 0.45).

6. **When a feature leaves a narrow load path (net-section ligament, web, wall between a hole and an edge) shorter
   than about 1.5x the feature radius, add a curve refinement on the straight edge that crosses that ligament.** Use size of about ligament/6
   and radius of about 400 u. Here that edge was the symmetry line c4, which ends at the peak corner. The load funnels through the ligament (its energy density is several times the far field) and the stress
   drops steeply across it. On d/W = 0.4 (ligament 146 u), c4 at 25 u cut e x N from 424 to 396 (x 5.09 to 5.45) and halved the check-point
   error (0.63% to 0.29%). On d/W = 0.6 (ligament 97 u), c4 at 10 u halved the point error (0.61% to 0.32%).

7. **When the feature dominates the part (ligament < ~0.7x the feature radius, or the feature radius > ~15% of the diagonal),
   switch to a more uniform near field.** Use a coarser edge (about ligament/5) with a gentle slope of 0.3-0.35, plus the fine ligament
   refinement (about ligament/8). The disturbed zone then fills most of the part, and the standard 0.5 ramp starves it. On d/W = 0.6
   (r 146, ligament 97), arc 20 u at slope 0.35 plus c4 12 u gave e x N 672 (x 4.13), versus 817 (x 3.39) for arc 14 u at 0.5 plus c4 19 u. On the
   r = 198 plate, slope 0.3 beat 0.4 (319 versus 383). This did NOT pay at ligament ~ radius (d/W = 0.45: 507 versus 444), so keep 0.5 there.

8. **When the feedback shows e below the 0.5% floor, scale every `size` and `h` up by sqrt(0.5/e) and keep the radii.** This also
   scales the slopes, so the shape of the size field is kept. For P2, e x N stays roughly constant under such scaling, so DOFs spent below the floor
   are pure cost. On d/W = 0.15, e = 0.27% at 762 DOFs (x 4.12) became e = 0.51% at 438 DOFs (x 7.03) after scaling sizes by 1.4. Conversely, when
   0.5% < e <= e_auto/2, refining the same shape does not raise x (d/W = 0.4: 18 to 13 u moved e from 1.00% to 0.67%, but e x N only from 424 to 415), so change
   the shape instead. Never let e exceed e_auto/2: once g < 2 the penalty becomes quadratic (r 198 plate: e 0.77% scored x 4.82 versus 6.52).

9. **When a finer size on a feature gives the same or a worse e, go back to the previous size and work on the slope or the ligament.**
   Below the optimum, the residual error lives in the decay zone, and coarse meshes (200-400 nodes) scatter about 10% from
   mesh topology alone. Examples: arc 14 to 11 u raised e from 0.76% to 0.90% with 20% more DOFs (d/W = 0.45), and 13 to 10 u gave only 0.60% to 0.56% for 21% more DOFs (d/W = 0.25).
   The same 18 to 14 u change helped d/W = 0.45 (469 to 444) but hurt d/W = 0.4 (396 to 438).

10. **When reading feedback, compare `field` (energy) with `points` (check-point von Mises).** If points dominate, refine
    the edge or ligament where the peak stress sits. If field dominates and edge refinement stalls, fix the ramp (lessons 5 and 7) or add the
    ligament refinement. On d/W = 0.6, the arc 20 to 16 u cut the points error from 1.05% to 0.47%. There, c4 19 to 10 u cut points from 0.61% to 0.32%, but field only
    from 1.22% to 1.14%. Field responded only to the gentler ramp (arc 14 u at 0.5 to 20 u at 0.35: 1.14% to 1.03% with 12% fewer DOFs).
