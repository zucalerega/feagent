# Teacher A: meshing lessons (notched bars + pin-loaded lugs)

Evidence: 57 scored recipes on 8 double-U-notched tension bars (root radius r = 4.9-24 u,
notch depth t = 19-73 u) and 3 pin-loaded clevis lugs (bore radius a = 89-173 u, wall around
the bore w = 51-134 u). Ramp arithmetic: slope = (h - size) / radius, so radius = (h - size) / slope.

1. **When** the stress is smooth apart from a few concentrations (notches, holes, fillets), **do** use
   `"order": 2`, **because** quadratic elements give far more accuracy per DOF. With the same 2.5 u root on
   bar t4_r1, order 1 (with a gentler 0.25 slope to spend the cheaper DOFs) scored 4.7x (e 1.26%,
   982 DOFs). Order 2 scored 7.1x (e 0.50%, 1618 DOFs).

2. **When** a region carries uniform or linearly varying stress far from every feature (a long bar in
   tension away from its notch, a remote plate edge), **do** let `h` reach about the local section width
   (h = 250 u for a 243 u wide bar, roughly one element across), **because** quadratic elements represent
   that field exactly, so those DOFs buy nothing. On t4_r1, going from h 80 to 250 at the same root ramp
   gave e 0.46% to 0.50% and DOFs 1910 to 1618 (6.1x to 7.1x). Capping h at 70 on t12_r4 left e at 1.0%,
   added 11% DOFs and lost score (3.5x to 3.1x).
   Counter-case: a short part stressed throughout is not uniform. The lug shank (clamped base plus load
   spreading below the hole) needed h of about W/5 (going from h 120 to 60 cut e from 1.87% to 1.22%).

3. **When** a concentration sits on a curved boundary of radius r in an otherwise lightly loaded part (notch
   root; by the same logic a fillet or small hole), **do** refine that curve or group with
   size = k*r and slope k, k = 0.4-0.5 (radius = (h - k*r)/k), never below the 2.5 u floor. This makes
   the element size about k times the distance from the centre of curvature, **because** the
   concentration decays over r + d. Every bar's best recipe had size/r and slope both in 0.3-0.5
   (r = 9.7 u: size 4.85, slope 0.5, 4.55x). Unbalanced shapes lost:
   - size r/5 with slope 0.6: 4.4x vs 5.1x (t6_r3);
   - size 0.7r with slope 0.45: 3.9x vs 4.55x (t8_r2).

   If k*r < 2.5 u (r below about 6 u), use the floor size with slope about 0.5. On t4_r1 (r = 4.9 u),
   2.5 u scored 7.1x and 4 u at the same slope scored 5.8x.

4. **When** grading away from a local concentration in a big uniformly loaded body, **do** keep the slope
   between 0.35 and 0.55, never about 1, **because** gmsh does not cut DOFs in proportion to a steep slope
   while the error rises with it. On t4_r1, slope 1.0 against 0.53 removed 37% of the DOFs but nearly
   doubled e (0.50% to 0.96%), so the score fell from 7.1x to 5.9x. Within 0.4-0.5 the score is flat
   (t8_r2: 4.48x vs 4.55x).

5. **When** choosing the overall fineness, **do** aim for e of about 0.5-0.8%, just above the 0.5% floor,
   and stop there, **because** once g >= 2 the score is 2g/c, and e*DOFs stays nearly constant along a
   well-shaped quadratic mesh family (e is proportional to 1/DOFs). A coarser member therefore scores the
   same, while anything below 0.5% is pure cost:
   - t6_r3: four recipes with e from 0.54% to 0.95% all scored 5.0-5.2x;
   - lug D24: slope 0.13 (e 0.47%) scored 2.61x and slope 0.15 (e 0.52%, 6% fewer DOFs) scored 2.68x.

   Deeper or bigger features put more energy at stake and cost more (bars: 7.1x at t = 19 u, 3.4x at
   t = 73 u). Accept that rather than over-refining.

6. **When** the feature zone holds a large share of the strain energy (a pin-loaded eye or lug head, thin
   rings or ligaments around a hole, any short part stressed throughout), **do** grade much more gently,
   slope 0.12-0.2 from the feature, with h of about W/5 (80 u for a 447 u shank), **because** the error
   anywhere in that zone counts at full weight:
   - lug D16: bore slope 0.53 to 0.2 (h 120 to 80) cut e from 1.87% to 0.74% (2.3x to 3.2x);
   - lug D31: slope 0.2 to 0.12 cut e from 0.89% to 0.59% (2.65x to 3.0x);
   - lug D24: slope 0.3 gave e 1.05% (2.2x); slope 0.15 gave e 0.52% (2.7x).

7. **When** a hole carries the load (pin bearing) and is ringed by a wall of width w, **do** refine the
   whole `bore` group (both halves) at size w/5 for thin rings to w/10 for thick ones (D16 with w = 134:
   14 u; D24 with w = 90: 11 u; D31 with w = 51: 10 u), **because** both coarser and finer surfaces lost:
   - D24: bore 18 u scored 2.4x (e 1.06%); bore 8 u scored 2.1x (e 0.68% against 0.66%, for 17% more DOFs);
     11 u scored 2.5-2.7x;
   - D31: 13 u scored 2.6x against 3.0x for 10 u;
   - D16: a steeper ramp on the unloaded half (`bore_free` at slope 0.44 against 0.2) raised e from 0.74%
     to 0.87%, no gain for the DOFs it saved.

8. **When** a corner is singular but sits where little load flows (a clamped edge ending at a free edge,
   like the lug base corners k1 and k2), **do** use only a small local ramp,
   `{"corner": k, "size": 4, "radius": 60-80}` (about 1% of the part), **because** its singular zone holds
   little energy, so harder refinement only adds DOFs:
   - D16: corners 4u/120 to 2.5u/250 moved e from 1.87% to 2.04% (no gain), with 14% more DOFs
     (2.26x to 1.82x);
   - D31: 4u/50 to 2.5u/100 moved e only from 0.86% to 0.87% (2.63x to 2.52x).

   Judge corners by the stress flowing past them: a re-entrant corner in the main load path deserves more.

9. **When** a feature has free surfaces shielded from the load (the flanks of a slot or deep notch),
   **do** leave them to the root ramp and use one ramp per feature, **because** they sit in a near-zero
   stress shadow:
   - adding `notch_flanks` 12u/100u on the deepest notch (t15_r3) cut e only from 1.13% to 1.05% for 8%
     more DOFs (3.41x to 3.39x);
   - stacking two ramps on one root (fine and steep near, coarser and gentle far) matched a single
     balanced ramp (t6_r3: 5.18x vs 5.10x).

   Spend refinement where stress is high and changing, not on every edge near the feature.
