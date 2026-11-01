## HOW THIS SYSTEM WORKS (read by the agent, the teacher, the reflector and the organizer)

ONE persistent meshing agent learns over many generations to mesh FEA problems well. Every
generation it works on ALL the training tasks in one conversation: for each task it inspects the
geometry, meshes it (global size h, element order, local refinements at named groups or points),
solves within the task's solve budget and submits one solve. Every tool call names its task.

Remembered values. Never use a published or remembered benchmark value (e.g. a NAFEMS target
stress) to judge or choose a solve: scoring is the error of the whole field, and a remembered
number can be wrong for this model. Choose your submitted solve by evidence from your own solves.

Scoring. The submitted solution is compared with a very fine reference solution: the error is the
larger of (a) the energy-norm stress error over the WHOLE part and (b) the RMS von Mises error at a
set of check points spread over the part (a few are disclosed to the agent, the rest are hidden).
The yardstick is an automesher (gmsh default sizes, quadratic elements). The target is half the
automesher's error: below that target the score falls quadratically, above it beating it keeps
paying. DOFs cost: the score is divided by DOFs relative to the automesher, so extra elements must
buy at least proportional accuracy. There are no other agents and no bonus: only this score counts.

Learning. After each generation a teacher looks at the agent's run on every task: the whole mesh,
where the error is (error maps and region grids), the setup and the agent's reasoning. It tells the
agent in depth what went wrong and exactly what to change, and what it did really well. The agent
also looks at its own final results and whole mesh and writes notes for itself. An organizer then
merges, rewrites and orders the agent's notes and its playbook of task-level lessons: it keeps
every useful specific idea and drops only what newer evidence superseded or contradicted.

Notes are tagged by task (source "...@<task>" or "@general"): another task's note is a hint to
test. Address every point: the agent answers every item of its teacher feedback in its first
message, and the teacher checks next generation that each promise was kept and flags repeats.

Rules of the game. Notes are earned evidence from earlier runs: respect them, but test them, and
report when one fails. Every generation the agent must deliberately change at least one thing on
each task compared with its last run, and say what and why. Use solves as controlled experiments
(change one thing, compare). Specific beats generic: name groups, points, sizes, radii and orders.

## THE AGENTS' TOOLS (every agent has them; full reference: ace/TOOLS.md)
- inspect_geometry(): every face (group f<tag>, type, area, centroid, normal, neighbours, named in
  the statement or not) and edge (tag, length, faces, dihedral angle, RE-ENTRANT or not: a likely
  stress singularity), smallest features, plain-language hints.
- mesh(h, order, refinements): refinements at a face {group}, a CAD edge {edge: tag}, a {point} or a
  {box: [xmin,ymin,zmin,xmax,ymax,zmax]}, each with size and radius. Returns quality, repaired_cells
  (folded curved cells straightened automatically) and folded_cells (must be 0, else the run is
  invalid), and element size near each named group.
- solve(setup): the scored quantity plus every statement quantity, reactions vs applied loads, max
  displacement and stress with locations, and a reference-free error estimate with the top 5
  hotspots (location, nearest face/edge, local element size, share of the error).
- view(solve, kind=mesh|stress|error, view=iso|x|y|z|-x|-y|-z): a picture the agent sees.
- compare(solve_a, solve_b): how much the stress field changed between two solves and where
  (convergence evidence without a reference).
- submit(solve).
Good practice: inspect the geometry first; refine re-entrant edges and support/load edges; check
folded_cells is 0 and equilibrium holds; use the hotspots to place the next refinement; compare two
solves before choosing; submit by this evidence.

You are the single agent, working on all of these tasks: (the new problem).

# TASK (the new problem): WHAT HAPPENED LAST TIME

## YOUR LAST RUN
(no previous run)

## YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN
(none yet)

## SETUP WARNINGS FOR THIS TASK ((the new problem))
- (none)

## CONTESTED REGIONS ON THIS TASK ((the new problem)): where last generation's agents disagreed most
- (none)

## YOUR OWN REFLECTION ON YOUR LAST RUN
(none yet)

## YOUR PLAYBOOK (task-level lessons)
#### general notes
### meshing
[mesh-0031] (fix) [general] Read node count from mesh(). If a refinement changes nodes <2-3% (big's bc0 S4 box 0.0035 added 35 nodes), it realised nothing: request ~0.7x desired size or drop it before spending a solve.
[mesh-0010] (highlight) [general] Peer big's recipe: coarse background plus nested boxes over the previous run's hotspot pocket (box spans hotspot x/y +3 mm, size ~0.6x their local h) cut DOFs 32% with no accuracy loss, score 0.42->0.63.
[mesh-0011] (highlight) [general] Closed bores under pressure self-balance (|R|~0 N). Open faces: reaction = p x projected area. Don't chase phantom errors.
[mesh-0020] (highlight) [general] Boxes can't follow annular bands: tile ring with 4 side-strip boxes (inner edge r_in-1mm, outer r_out+1mm, radius 0.002; cut add-on +19.4k to +6.2k DOFs), or refine the single CAD edge the band hugs. Never box a whole disk that is black on the error map.
[mesh-0026] (fix) [general] Boxes realise ~1.2-1.35x requested size. For a target h at a clamp rim use {edge:tag}, or request 0.75x in a THIN slab, never a deep box. Edge radius must exceed hotspot offset +3-4 mm. Read per-group h in mesh().
[mesh-0027] (highlight) [general] Edge refinements along singular lines (re-entrant, clamp rims) beat volume boxes over solid regions: big cut DOFs 32.2k->19.7k and scored best (0.231). Leave black-on-error-map regions at background h.
[mesh-0030] (keep) [general] On thin walls (wall ~ background h), edge refinements reach only ~their radius. Hotspots offset >radius need a box spanning the band ±3 mm. Pair edges (curve singularity) with one band box (bending zone).
[mesh-0029] (highlight) [abc_00000003_bc7] Ablation record: dropping belt box saved 3.9k DOFs but diff +1.9 points (worth it). Spend at most one ablation per refinement; log diff change per 1k DOFs.
### bcs
[bc-0002] (fix) [general] For open loaded faces (bc0: f7) check once that the f12 reaction equals the applied pressure resultant (non-zero). Mismatch means a setup error, showing as error spread over the whole part.
### solver_setup
[solv-0001] (avoid) [general] Before any tool call copy THIS task's LIMITS line verbatim ('at most 4 solves' on both tasks; big wrote 16 three generations running, used 6). Write 'solve budget = 4'. Plan: S1 control, S2 judge, S3/S4 single-factor variants, then submit(argmax est. score). A 5th solve is a violation. quality.min>=0.1.
[solv-0004] (avoid) [general] Run one controlled change per solve and state it. big's S4 changed three things at once (unattributable); S5->S6 single change gave a clean answer. With 4 solves spend one on a judge only if a safe control is already solved.
### postprocessing
[post-0008] (fix) [general] Write a ranking table (solve, DOFs, diff, est. e, est. score) and submit argmax via submit(solve=k), never the last solve by default (big: S5 est 0.240 but S6 went in; on bc0 submitted S4 0.502 while S1/S3 ~0.6).
[post-0001] (keep) [general] Calibrate compare(): true error ~1.2-1.25x diff to a judge with ~2x DOFs (bc0 9.27 vs 7.42; bc7 1.16-1.20). Use sqrt(diff^2+e_judge^2) only if judge error is estimated. Choose by estimated score, not DOFs: a cheap mesh above target can lose.
[post-0004] (highlight) [general] Judge recipe: compare(S1,judge), compare(S2,judge); error ~ calibration x diff (1.2-1.3) or sqrt(diff^2+e_judge^2). Rank by A/c, A=(g/2)^2 if g<2 else g/2, e=max(field, point estimate); submit argmax, not necessarily the finest. Big's estimate matched true error.
[post-0005] (keep) [general] Rank candidates by energy deficit: fit U_ref from your finest solves (or Richardson e~N^-2/3), error ~ sqrt(1-U/U_ref). Predicted 11.0% vs true 10.5%; ZZ reads 1.2-1.4x low.
[post-0006] (avoid) [general] Above target score=g/(2c): refinement pays only if local rate p=ln(e1/e2)/ln(N2/N1)>1, coarsening only if p<1. Below target A=(g/2)^2. When p<0.67 in 3D order 2, extra DOFs lower score: submit the argmax, not the finest (big's S4 0.2315 beat finer S5/S6 ~0.225).
[post-0002] (idea) [general] Strain energy ranks candidates for free (bc0: S3<S4<S1<S2 matched compare). A variant with more nodes but lower energy than control signals noise or a misplaced refinement: discard it. Never submit lowest-energy unless DOF saving beats break-even.
[post-0003] (keep) [general] Name the binding term (field vs check-point RMS) before choosing. If a refinement doesn't touch where check points sit, it can't lower the point term; stop refining a region once its field error is below the point RMS. Where points bind, use hotspot local vM errors in smooth regions.
### tool_use
[tool-0001] (fix) [general] Call view(kind=error) after EVERY solve (big viewed once in 6). Compare each top hotspot's local h with the request; if >1.3-1.5x or outside the refinement radius, move/enlarge it (root hotspots 0.0028-0.0031 vs 0.0021 requested). Box bounds must enclose hotspot coords +~3 mm.
[tool-0002] (fix) [general] Call compare(control, variant) for every candidate before submit (big called it 0 times; energy increments can't show mid-height changes). Check reaction vs true load resultant.
[tool-0003] (highlight) [general] After each compare(), list top-3 regions with nearest edge and distance. Any edge within 3 mm holding >3% gets an {edge} refinement with radius = distance + 2-3 mm.

#### notes for task abc_00000065_bc0
### meshing
[mesh-0001] (keep) [general] Score economics in 3D order 2: e~N^-2/3, so score rises as c^(1/3) below target and falls as c^(-1/3) above. Submit the cheapest mesh near the target error; avoid global refinement once met. Beyond that, move DOFs from low-error to high-error regions.
[mesh-0016] (keep) [abc_00000065_bc0] New control (0.686, 16.7k DOFs, field 9.27%, points 8.57%, ~5571 nodes, q 0.11): h 0.008, order 2, root box [0.012,-0.016,-0.0085,0.031,0.02,-0.0055] 0.0021 r0.003, f7 0.0035 r0.004, f5 0.004 r0.003, upper box [0.012,-0.027,-0.0045,0.026,-0.012,0.0081] 0.005 r0.004. Supersedes 20k control (0.602).
[mesh-0034] (idea) [abc_00000065_bc0] Next (one solve): narrow root box to [0.016,-0.010,-0.0085,0.029,0.010,-0.0055] 0.0018 r0.002, covering hotspots #1-#5 (x 18.5-26.6 mm, z -6.6 to -7.4, realised h 0.0028-0.0031; ~39% of error). Keep only if 1.25 x diff gives higher g/(2c).
### bcs
[bc-0003] (keep) [abc_00000065_bc0] Setup verified: 3d, E 2.1e11, nu 0.3, fixed f12, pressure 2.1e6 into f7, no gravity. f7 open: f12 reaction non-zero (p x projected area); check once.

#### notes for task abc_00000003_bc7
### meshing
[mesh-0014] (keep) [abc_00000003_bc7] New control (g6 best, 23.5k DOFs, true field 16.8%, points 6.3%, score 0.2315): h 0.006, order 2, {group f2,0.0025,r0.005}, {edge 8,0.0025,r0.005}, {edge 39,0.0025,r0.005}, box [-0.021,-0.0175,0.0165,0.021,0.0175,0.032] 0.0032 r0.002. One factor per variant.
[mesh-0032] (idea) [abc_00000003_bc7] Next variant: control + {edge:40,0.0022,r0.004}, or ring strips z 0.052-0.0585 (inner r0.0137, outer r0.018, size 0.0026, r0.0015; not on cap f25). Clamp band holds ~16% error, hotspots #1/#6 1.6-2 mm from edge 40 at h 0.0041-0.0043. Expect diff -0.5 to -1 point for ~+1.5k DOFs.
[mesh-0005] (avoid) [abc_00000003_bc7] AVOID tightening {group f2} to 0.002 r0.006 (+1.9k DOFs, diff 14.6->14.4%): a group refinement stays on the face and misses wall hotspots 1.5-3 mm below. Never full-height clamp boxes (32k DOFs); never drop f2 refinement (f2 h 0.00486, field 16.4->18.4%).
[mesh-0004] (avoid) [abc_00000003_bc7] Never drop the belt box (without it hotspots h 0.0051-0.0054, diff 15.0->16.9%). Do NOT grow it beyond z 0.0165-0.032: down to 0.011 +1.8k DOFs diff 14.0->13.7% (0.225); up to 0.0345 diff 13.9% (0.224). If DOFs free up, test size 0.0029 in the same bounds.
[mesh-0033] (fix) [abc_00000003_bc7] Low priority: upper stub flank z 0.032-0.040 near y -0.013 (2.65% at h 0.0044). Only if DOFs free up try edges 8/39 radius 0.006 instead of 0.005; don't extend the box over it.
[mesh-0021] (keep) [abc_00000003_bc7] Thresholds vs 0.2315 control: target 15.93% needs diff <=~13.3% (calibration 1.2). Field binds (16.8% vs points 6.3%). Rate p~0.45: only placement pays. Order-2 recipes flat at 0.22-0.233 for three generations.
[mesh-0003] (avoid) [abc_00000003_bc7] Coarsen only black areas: stub ends f15,f17, cap f25, bottom wall z<0.012 stay at background 0.006. h>=0.009 fails quality. Spend DOFs within 4 mm of edge 40 and the intersection curves (edges 8/39).
[mesh-0015] (avoid) [abc_00000003_bc7] Thin wall sets DOFs: keep background h 0.006. h 0.007 saved 8% nodes but diff 15.1%->17.0%, quality 0.157; 0.0075 gave ~21%.
[mesh-0022] (avoid) [abc_00000003_bc7] AVOID splitting belt box into two side boxes |x| 0.0155-0.0225: diff 12.6%->16.1%. Belt centre (|x|<0.0155 near f12/f20) holds hotspots; keep refined.
[mesh-0012] (fix) [abc_00000003_bc7] Boxes realise h ~1.2-1.35x requested (0.0032->0.0044). Request ~0.75x desired or use edges; read per-group h in mesh().
[mesh-0013] (idea) [abc_00000003_bc7] Diversity (one solve, untested): order 3, h 0.008, {group f2,0.004,r0.005}, {edge 8,0.004,r0.005}, {edge 39,0.004,r0.005}, {edge 40,0.0035,r0.004}, belt box 0.0045. DOFs<=24k, quality>=0.1. If mesh() rejects order 3, note it.
### bcs
[bc-0001] (fix) [abc_00000003_bc7] Setup correct, reuse verbatim: 3d, E 2.1e11, nu 0.3, fixed f2, pressure 4.0e6 on f23 and 2.1e6 on f16 (into part), no gravity. Closed bores: f2 reaction ~0 N.
### geometry
[geo-0001] (avoid) [abc_00000003_bc7] Wall ~2.5 mm thick (radii ~14.7, 17.2 mm). Two quadratic elements across (h 0.00125) cost ~100k DOFs: unaffordable. Spend extra DOFs only at clamp rim, stub intersections.
### solver_setup
[solv-0003] (keep) [abc_00000003_bc7] Field error binds (16.8% vs points 6.3%). Worst points are low-stress outer-wall spots near stubs (~10% off); don't chase them, only clamp-band and belt field error moves the score.
### postprocessing
[post-0007] (keep) [abc_00000003_bc7] Judge (53.5k DOFs): h 0.0032, order 2, {edge 40,0.0014,r0.006}, {edge 39,0.0016,r0.005}, {edge 8,0.0016,r0.005}, {edge 3,0.0018,r0.004}. Calibration now 1.20 (S4 16.8/14.0; earlier 1.16). e=1.2 x diff; score=(0.3186/e/2)^2/(DOFs/6057). Order: control, judge, 2 variants.

## YOUR NOTES
#### general notes
### meshing
[a0-mesh-0013] (idea) [general] Below target (g<2) score ~ e^-2/N: a refinement pays iff local rate p=ln(e1/e2)/ln(N2/N1) > 0.5 (above target p>1). Measure p per single-factor pair: belt box p~0.58 pays, f2 group vs edge 40 p~0.83 pays. If global p<0.5 submit cheapest sensible mesh.
[a0-mesh-0014] (keep) [general] When DOFs stop responding to global h (features set cost), pick the h that protects check points and spend effort on refinement placement.
[a0-mesh-0001] (idea) [general] When top hotspots each hold <5% the error is diffuse. Use boxes over hot bands with coarser background; coarsen black regions. Once gain exceeds 2, uniform refinement only breaks even or loses.
[a0-mesh-0024] (fix) [general] Group refinement reaches only ~0.6x radius. Edge refinements failed on two tasks. Boxes realise ~1.35x request: ask ~0.75x; a box size >=0.7x realised h adds almost nothing (check mesh() node delta vs control; bc0 +35 nodes). Thin slab boxes at hotspots.
[a0-mesh-0052] (highlight) [general] When a refinement closes one pocket (belt 32% -> 2-3% per cell), re-rank pockets from the new error map before next variant; clamp became top pocket yet two solves still went to belt.
[a0-mesh-0036] (keep) [general] Rank refinements by measured energy gain per DOF (dU/dDOF) from single-factor solves, not intuition; the region you think over-resolved can be the best investment.
### solver_setup
[a0-solv-0003] (fix) [general] REPEATED 3x: misquoted solve limit (16; tasks say 4). First line of every task: copy LIMITS sentence verbatim ('at most 4 times' -> 'solve budget = 4'); number solves k/4. Plan S1 control, S2 judge/variant, S3 one-factor variant, S4 confirm best. Never exceed. Self note saying 16 is wrong.
[a0-solv-0001] (keep) [general] Displacement FE strain energy rises toward the converged value as mesh refines; shrinking increments mean convergence, not a singularity artifact.
### postprocessing
[a0-post-0003] (fix) [general] Before submit write one line per candidate: DOFs, diff, e=calib x diff, g=auto_e/e, A=(g/2)^2 (g<2) or g/2, c=DOFs/auto_DOFs, A/c. Submit the argmax. Name binding term (field vs check-point RMS); if field <~8% points may bind.
[a0-post-0004] (keep) [general] Above 2x gain score is cost-limited: ~1/(e*N), +44% DOFs needs >31% lower error. Tied candidates: submit the cheapest. Use a judge at ~2x DOFs for compare() only.
[a0-post-0008] (fix) [general] Calibrate compare() per task from scored truth: bc0 true field = 1.25x diff to 30k judge (9.27/7.42); low calibration predicted 0.75 vs actual 0.686.
### tool_use
[a0-tool-0008] (fix) [general] Always end with explicit submit(best solve by estimated A/c), not the last solve. g5 submitted last solve (0.502) over S3 (est ~0.69); previously S6 over S5 lost ~0.007.
[a0-tool-0006] (avoid) [general] Never build a mesh you will not solve. If a solve remains, solve the judge mesh and run compare(candidate, judge) before submit.
[a0-tool-0009] (fix) [general] Read compare() 'regions' nearest_reentrant_edge. If hotspot <3 mm from a re-entrant edge, refine that edge, not the nearest face group (g6: edge 40 at 1.6-2.0 mm).
[a0-tool-0010] (avoid) [general] After every variant solve check top hotspots' local h dropped below control's; if not, refinement missed, don't submit it. Avoid near-identical variants; choose bolder changes on different regions.

#### notes for task abc_00000065_bc0
### meshing
[a0-mesh-0042] (keep) [general] [abc_00000065_bc0] New control 0.686 (16.7k DOFs, field 9.27%, points 8.57%): h 0.008, root box [0.012,-0.016,-0.0085,0.031,0.02,-0.0055] 0.0021 r0.003, f7 0.0035 r0.004, f5 0.004 r0.003, upper box [0.012,-0.027,-0.0045,0.026,-0.012,0.0081] 0.005 r0.004. Solve first as safe submission.
[a0-mesh-0049] (idea) [general] [abc_00000065_bc0] Teacher next mesh: control with root box trimmed to [0.015,-0.0115,-0.0085,0.03,0.0135,-0.0055] 0.0021 r0.003 (hotspots x 18.5-26.6, y -6.2..8), ~1.5-3k fewer DOFs at equal error. Second: trimmed at 0.0019 (not below 0.0018). Keep if g/(2c) beats.
[a0-mesh-0007] (fix) [general] [abc_00000065_bc0] Top hotspots at z -7 mm inside f5 (x 18-20) and f7 (x 25-27), y -10..8; root realises h 0.0028-0.0031 under 0.0021 request. Cell (0.0152,0.0014,-0.006) holds 13.6% with 24 elements: test root box x down to 0.010 at 0.0023. Box y must reach 0.020 unless trim tests pass.
[a0-mesh-0043] (keep) [general] [abc_00000065_bc0] Marginal value: root box 4.8e-10 J/DOF, upper box 0.0035->0.0027 2.3e-10, y+ box 0.9e-10. Spend DOFs at f5/f7 root first. Root box not above 0.0024 (0.0028 lost 0.14% U).
[a0-mesh-0012] (avoid) [general] [abc_00000065_bc0] Box 0.0018 realises ~0.0024; never request 0.0015 or 0.0011 (score -0.48 vs 0.56; quality 0.109). Keep box >=0.0018, folded_cells=0.
### solver_setup
[a0-solv-0006] (fix) [abc_00000065_bc0] LIMITS: 'You may call solve at most 4 times.' Write 'solve budget = 4' before inspect_geometry. Plan: S1 control, S2 trimmed root box, S3 trimmed at 0.0019, S4 best. Wrote 16 three generations running.
### tool_use
[a0-tool-0001] (fix) [general] Budget solves before the first mesh. One factor per solve so compare() attributes the difference; group factors only if a judge already ranks them. Never mesh after the last solve. Avoid near-identical variants. Extra solves don't fix diffuse error.
[a0-tool-0005] (fix) [general] After EVERY mesh read 'element size near' per group. After EVERY solve checklist: view(error, iso), copy top-5 hotspots (coords, local h, share), compare(prev,this), table row. If hotspot h >1.5x requested, refinement failed: cover with box/larger radius or drop.
[a0-tool-0002] (fix) [general] Every refinement must name a target (group, box, edge or point); one without was ignored. Check reported size near the group, quality.min >= 0.1, folded_cells = 0.
[a0-tool-0007] (fix) [general] Judge check-point-targeted refinements with compare(control, variant): read where VM changed. Before submit check reaction vs load. Spend solves on bolder experiments, not 0.0004 size tweaks on near-ties.

#### notes for task abc_00000003_bc7
### meshing
[a0-mesh-0038] (keep) [general] [abc_00000003_bc7] New control (g6 S4, true field 16.8%, points 6.3%, 23.5k DOFs, 0.2315, judge diff 14.0%): h 0.006, order 2, f2 0.0025 r0.005, edge 8 and edge 39 0.0025 r0.005, box [-0.021,-0.0175,0.0165,0.021,0.0175,0.032] 0.0032 r0.002.
[a0-mesh-0053] (idea) [abc_00000003_bc7] Teacher's next mesh: control plus {edge 40, 0.0025, r0.004}. Clamp band z 0.0555-0.057 holds ~16% error (4 cells, 210-260%, r 0.0145-0.015, 1.6-2 mm from edge 40, h 0.0041). Pays if diff <=13.4% at <=25.5k DOFs. Optional ring box [-0.017,-0.017,0.050,0.017,0.017,0.059] 0.003 r0.002.
[a0-mesh-0054] (avoid) [abc_00000003_bc7] AVOID f2 group 0.002 r0.006: +1.9k DOFs, diff 14.6->14.4% (p 0.17); clamp hotspots 4-5 mm below f2, spill refines black top disk f25.
[a0-mesh-0055] (keep) [abc_00000003_bc7] Belt box settled at z 0.0165-0.032: bottom 0.020->0.0165 paid (p~1.1); bottom 0.011 (p 0.30), top 0.0345 (p 0.15) did not. Do not widen; belt cells 2.2-3.1% each.
[a0-mesh-0047] (idea) [general] [abc_00000003_bc7] Ablation (one solve): control minus edges 8/39 (belt box covers junction; hotspots on f23 bore flanks ~2 mm from f20). Break-even: -2k DOFs allows diff up to ~14.6%.
[a0-mesh-0044] (avoid) [general] [abc_00000003_bc7] Never drop the belt box: without it (S6, 17.2k) ring z 0.020-0.031 held ~32% of error, hotspots h 0.0052, field 19.6%. With it est 17.4%, local p~0.58. (Self saw S6 score 0.233; belt boxes at +7k DOFs bought 1-3%, but teacher's calibrated S5 is higher.)
[a0-mesh-0040] (keep) [general] [abc_00000003_bc7] Score flat ~0.22-0.23 from 17k-25k DOFs (p~0.49). Only p>0.5 moves pay: +8.5% DOFs needs >=4% relative cut in e. Compute p vs control for every variant.
[a0-mesh-0046] (keep) [general] [abc_00000003_bc7] Clamp: f2 group 0.0025 r0.005 beats edge 40 (0.002 r0.007, est 20.5%) and thin box z 0.0535-0.0585 0.0028 (24.6k, est 17.1%, ~0.215). Keep f2 as only clamp refinement (clamp cells ~11.6%, acceptable). Thin 0.0035 r0.0015 box (~2k DOFs) only as a tested add-on.
[a0-mesh-0048] (idea) [general] [abc_00000003_bc7] Mid-height cells z 0.0255 (x ±0.0174, y ±0.013) hold 2.5-3% each at h 0.0039, 110-190% local error. Uniform 0.004 doesn't resolve; 2.5 mm wall limits grading.
[a0-mesh-0026] (keep) [general] [abc_00000003_bc7] Measured: top box 0.002 r0.003 cut est error 16.4%->~10.5% for +19.5k DOFs (score ~0.17); four ring boxes 0.0022 + f2 ~15.4% at 31.1k (~0.20). Two side boxes instead of junction box raised diff 12.6%->16.1%.
[a0-mesh-0027] (avoid) [general] [abc_00000003_bc7] AVOID changing h, f2 and box at once (h 0.0075, f2 0.003 r0.004, box 0.0032: ~20% at 18.6k, 0.20). Avoid uniform h 0.0035/0.0042 (flat or worse). Never box above z 0.0585 or below 0.0535 thin clamp (full-height box 32.2k DOFs, ~0.19).
[a0-mesh-0029] (avoid) [general] [abc_00000003_bc7] Top disk f25 and stub ends f15/f16/f17/f5/f6 are error-free; coarsen them (coarse background, smaller bleed radii) to free DOFs. Don't refine f25 beyond r<=0.002 above z 0.059.
[a0-mesh-0030] (idea) [general] [abc_00000003_bc7] Hidden worst check point near f3 at x 0.029, z 0.026 (68% local error, h 0.004). Option: box [0.024,-0.010,0.020,0.031,0.010,0.034] 0.0033 only if points bind (they currently don't).
[a0-mesh-0031] (avoid) [general] [abc_00000003_bc7] AVOID global h >0.007 (0.0095 gave quality 0.066; 0.007 saved 8.6% DOFs but diff rose 15.1%->17%) or <0.005. Keep h 0.006; keep body bottom z 0-0.062 not too coarse (~7% error); if bottom cells >3% add box 0.0045.
### bcs
[a0-bc-0001] (fix) [general] [abc_00000003_bc7] Setup verified, reuse verbatim: 3d, E 2.1e11, nu 0.3, fixed f2, 4e6 Pa f23, 2.1e6 Pa f16, no gravity. Record f2 reaction in S1 (expect ~0 N, closed bores); skipped last run.
### geometry
[a0-geo-0001] (keep) [general] [abc_00000003_bc7] Wall f23-f20 is ~2.5 mm: sizes >=0.0032 give <1 quadratic element across; two layers cost ~4x DOFs. Grade in-plane only; no sub-0.003 boxes over large wall areas.
### solver_setup
[a0-solv-0002] (keep) [general] [abc_00000003_bc7] Keep order 2: error is smooth diffuse wall bending; DOF placement is the lever.
### postprocessing
[a0-post-0001] (keep) [general] [abc_00000003_bc7] Calibration: true field = 1.20 x diff to 53.5k judge (S4 16.8/14.0; earlier 1.16). Judge: h 0.0032, {edge 40,0.0014,r0.006}, {edge 39,0.0016,r0.005}, {edge 8,0.0016,r0.005}, {edge 3,0.0018,r0.004}. S1 control, S2 judge, S3/S4 variants.
[a0-post-0005] (keep) [general] [abc_00000003_bc7] Points don't bind (RMS 6.3% vs field 16.8%; hidden worst ~11% near f3 x 0.029, z 0.026 / stub junctions). Every DOF should lower field energy at edge-40 clamp corner or f23 bore flanks.
### tool_use
[a0-tool-0003] (fix) [general] [abc_00000003_bc7] After solve 1 view(error, iso) and (error, -x), list hotspot z-bands before choosing refinements.

## THIS GENERATION
Before you mesh: for EACH item above (every step-critique correction, every mistake, every setup warning, the suggested mesh call, every contested region) write one line: what you will do about it, and answer 'could my solution fail in a way that makes the teacher tell me this again?' If yes, change your plan until the answer is no. Check your setup against every setup item before your first solve. Then follow your plan.
Do this for every task. Use solves as experiments: change one thing and compare.