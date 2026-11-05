# Supersonic flow: the swarm's first generation

**Run.** `ace_runs/supersonic-deep-blunt`: 6 Claude Sonnet agents, the `--deep` loop (Claude Opus teacher, self-reflection,
organizer), field scoring. Training tasks: `wedge_m2_10deg`, `wedge_m3_15deg`, `supersonic_blunt_wedge_2d`; held out:
`wedge_m5_15deg`. Compressible flow and the blunted wedge come from PR #24; the run itself
is work in progress on the branch `claude/supersonic-blunt-swarm`. Generation 0 only.

**Agent a0, generation 0, on `supersonic_blunt_wedge_2d`** (`gen0/runs/g0-supersonic_blunt_wedge_2d-a0.json`, `checks`)
- Mesh: 2,077 nodes, 3,884 linear triangles, 15,536 DOFs, first-order (upwind)
  convection.
- Field error 11.5% (velocity 7.6%, pressure 8.7%); RMS
  speed error at the check points 2.3%.
- The automesher (uniform triangles at the 6,000-node cap, 5,989 nodes, 46,096 DOFs,
  second-order convection): field error 5.7%. So a0 is 0.50x as
  accurate with 3.0x fewer DOFs: score 0.182 without bonuses
  (0.73x the automesher's 0.25), 0.276 with them.

**What the teacher took from it** (`gen0/teacher_supersonic_blunt_wedge_2d.json`, shared lesson): "About 80% of the squared error is on the bow shock, not on the body."

**Figures**
- `fig_supersonic_mesh_g0.png`: a0's mesh with the reference bow shock drawn over it.
- `fig_supersonic_reference.png`: the target flow (Mach number, sonic line).
- `supersonic_g0/g0-supersonic_blunt_wedge_2d-a0_mesh.png`, `_error.png`, `_speed.png`: a0's run report, exactly as the teacher and
  a0's self-reflection saw it. The error map is log10 of the local squared error, with the hotspots numbered
  largest first; the speed panels share one colour scale.

**Caveats**
- One run, generation 0 only: no learning is shown here. The run's generation 1 crashed in scoring and was
  restarted from its exported state (`ace_runs/supersonic-deep-blunt-2`).
- The reference is a fine solve of the problem's own mesh, not an exact solution: a detached shock has no closed
  form (`feacore/problems/cfd/blunt_wedge.py` on that branch).
