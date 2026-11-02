# fig2_raw_solver_vs_test: "Before post-training, the simulation is 33 % off the real test"

**Caption.** The out-of-the-box nonlinear simulation, before post-training: frictionless platens on gmsh's default
mesh, crosshead travel = platen travel. It starts too stiff (no seating gap) and then gives out: the prongs splay
and the force levels off at 3.3 kN, where the real part carried 4.7 kN
at 2.5 mm. Mean error: **32.7 % off the real test**.

**Key numbers.** 2,937 nodes (8,811 DOFs). Force vs test at 0.5 / 1.0 / 1.5 / 2.0 / 2.5 mm:
1580 / 2566 / 3045 / 3231 / 3298 N vs 860 / 2164 / 3308 / 4171 / 4743 N
(+84 / +19 / -8 / -23 / -30 %).

**Data.** talk_plots/act3_real_data/videos/before_run/curve.csv and summary.json next to it (CalculiX J2 + NLGEOM, the frozen ramp with frictionless
platens; error recomputed from the curve = the recorded 0.3265).

**Caveats.** "Before post-training" means before the swarm was post-trained on this test: the solver and the
swarm were already capable. This run also lacks the test conditions learned from the experiment (sticking
platens, seating gap, load-train compliance); fig3 separates the two steps.
