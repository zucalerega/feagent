# Lessons B: graded meshes for stress raisers (from 10 axisymmetric stepped shafts, 55 scored attempts)

Notation: a refinement {tag, size s0, radius R} gives element size s0 + k*d at distance d from the tag,
with slope k = (h - s0)/R, capped at h. Pick the slope first, then set R = (h - s0)/k. R is the slope
knob, not "the size of the zone to refine". Several refinements combine by taking the minimum.
Lengths are in u (part diagonal = 1000 u). Floor = 2.5 u.

1. When choosing the element order, always use order 2, because P2's energy error falls like 1/N
   (P1 like 1/sqrt(N)) and P1's piecewise-constant stresses fail the check-point test.
   Measured: Dd1p2_rd0p05 at matched DOFs, P1 e = 0.91% vs P2 0.50% (best P1 5.03x vs P2 5.79x).
   Dd2p5_rd0p25: P1 1.52x (point error 2.99%) vs P2 4.73x.

2. When grading away from any stress raiser, set the slope to k = 0.5-0.7 (e.g. s0 5.5, h 150 ->
   R 260). Never use k > 1, and rarely k < 0.45, because the load-redistribution zone around a raiser
   is about one section half-width deep and needs ~4-6 elements across.
   Measured: Dd1p2_rd0p05, fillet s0 3:
   - R 60 at h 80 (k 1.3): e 0.73% at 522 DOFs.
   - R 150 (k 0.5): e 0.30% at 826 DOFs, i.e. e*DOFs 2.5 vs 3.8.
   - R 60 at h 150 (k 2.45): e 2.49%, 1.86x.
   Dd1p2_rd0p08: k 0.45 vs 0.56 gives 5.00x vs 5.26x.

3. When a region carries uniform or linearly varying stress (tension or bending away from features,
   ends under uniform traction or frictionless support), let h be large (250, at least 2x the section
   width) and let the graded refinement reach it, because P2 represents linear stress exactly: these
   elements cost DOFs but add no error.
   Measured: h 80 -> 150 at the same slope: e 0.30 -> 0.29% with 7% fewer DOFs (Dd1p2_rd0p05).
   h 150 -> 250: 5.18x -> 5.26x (Dd1p2_rd0p08).

4. When the raiser is a smooth fillet, notch root or hole edge of radius rho, put the curve's size at
   s0 ~ 1.3*sqrt(rho) (+-30%). That is about 0.5*rho for rho 5-12, 0.3*rho at rho ~20, 0.2*rho at
   rho ~36, and the 2.5 floor for rho <= 5. Reason: a bigger fillet stresses a bigger volume and holds
   more of the strain energy, so it needs a smaller size/rho. A small high-Kt root still needs
   ~0.5*rho.
   Measured:
   - rho 20 (Dd1p3_rd0p1): s0 11 -> e 0.88%; s0 6 -> e 0.44% (same slope). Best is s0 6 at k 0.7:
     5.42x vs 4.11x.
   - rho 36.4: s0 13 / 8 / 6 -> e 0.77 / 0.58 / 0.51%.
   - rho 5.2 (Kt ~3): s0 4 vs 2.5 gives 5.01x vs 5.33x.
   Going finer than the rule wastes DOFs: rho 13.1, s0 3 vs 4 gives 5.08x vs 5.25x; rho 20.8, s0 4.5
   vs 6 gives 4.92x vs 5.33x.

5. When load must spread from a narrow section into a much wider one (wide/narrow >= 1.8, or a step
   at least ~0.7 of the narrow half-width; also a ligament feeding a wide plate, or a lug feeding its
   body), add a second refinement on the SAME tag with a larger size and a gentler slope. Example:
   [{"curve": c, "size": s0, "radius": (h-s0)/0.6}, {"curve": c, "size": 14, "radius": 715}],
   i.e. 14 + 0.33d at h 250. For a 3x width jump use 12 + 0.28d (radius 850). The minimum gives
   fine-near, gentle-far, because the spreading field decays over 1-2 wide half-widths, far beyond
   the fillet scale.
   Measured:
   - Dd1p8_rd0p12: dropping the far field raised e 0.76 -> 1.03%. Far 12+0.4d -> 14+0.33d gave
     5.33x -> 6.07x.
   - Dd3_rd0p1: 14+0.33d -> 12+0.28d gave 4.51x -> 4.99x.
   - Dd2p5_rd0p25: relaxing to 18+0.4d raised e 0.51 -> 0.69%.
   On small steps (<= 0.3 half-width) skip it: a gentler slope everywhere lost there (Dd1p3_rd0p1,
   k 0.40 vs 0.56: 3.47x vs 4.11x).

6. When a big step or wide region sits next to a small root (rho much smaller than the step height H),
   also make the near field gentle: k ~0.45-0.5 out to d ~ H/2. Between rho and H the field behaves
   like a re-entrant corner (sigma ~ d^-0.46).
   Measured: Dd2p5_rd0p08 near 6+0.7d -> 5+0.5d gave 6.64x -> 7.12x. Dd3_rd0p1 near 5+0.5d ->
   4+0.45d gave 4.99x -> 5.25x. Going further, to 4+0.4d, over-spends (6.85x).

7. When a corner is re-entrant (270 deg, singular), refine the corner tag ("corner": k) at the 2.5
   floor with ONE linear field. Use k 0.6-0.7 (R ~355-380 at h 250) for a modest step, and k 0.5
   (R 495) when the step is at least the narrow half-width. Do not coarsen the tip, and do not use a
   steep-then-gentle two-field profile there. Reason: the singular field is self-similar, so every
   ring of elements around the tip carries similar error, and resolution must start at the tip and
   grow uniformly. The same physics likely applies where a clamped edge or a load patch ends on a free
   face (not measured here).
   Measured:
   - Dd1p5_sharp: tip 2.5 -> 5 raised e 0.95 -> 1.67% (6.25x -> 4.08x).
   - Dd1p5_sharp: concave 2.5+0.8d / 10+0.4d gave e 1.22% vs 0.82% for a single k 0.6 field (4.27x
     vs 6.18x).
   - Dd1p5_sharp: k 0.4 / 0.6 / 0.7 gave 5.46x / 6.18x / 6.25x.
   - Dd3_sharp: k 0.5 vs 0.6 gave 6.32x vs 5.78x.

8. When a singular corner is present, aim for e ~1% with a lean mesh rather than chasing the 0.5%
   floor. The tip element at the 2.5 floor leaves an error that more DOFs cannot remove, so extra DOFs
   only shrink the rest. The check points avoid the tip, so the point error stays low (0.4-0.6%).
   Measured, Dd3_sharp: near k 0.65 -> 0.45 -> 0.35 gave e 1.22 -> 1.05 -> 0.99% at
   790 -> 878 -> 1092 DOFs (5.78x -> 6.05x -> 5.15x). The leanest good recipe, a single
   {"corner": 3, "size": 2.5, "radius": 495} at 706 DOFs, scored best (6.32x).

9. When e is already near or below the 0.5% floor, stop refining and redistribute. Once g >= 2 the
   score is x = 2*e_auto*N_auto/(e*N), and for P2 e scales like size^2, so e*N stays about constant
   when every size and slope is scaled together. Only a better distribution raises x. To land exactly
   on 0.5%, scale all sizes and slopes by sqrt(0.5%/e).
   Measured: Dd1p2_rd0p05, s0 3 at k 0.49: e 0.29% (floored), 772 DOFs, 3.77x. Scaled to s0 5.5 at
   k 0.56: e 0.50%, 500 DOFs, 5.79x. Dd1p3_rd0p1 s0 7.5: predicted e 0.55%, got 0.54%.

10. When a free face lies next to the raiser (shoulder face, hole flank, notch flank), do not refine it
    separately. The distance field from the root already grades it, and the stress there is low.
    Measured: Dd1p8_rd0p03, adding {"curve": 4, "size": 10, "radius": 200} on the 64 u shoulder face:
    e 0.78 -> 0.77%, DOFs +10%, 5.33x -> 4.92x.
