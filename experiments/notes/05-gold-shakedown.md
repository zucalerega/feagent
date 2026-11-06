# Full-run shakedown: 20 cases (20 random parts, 1 BC set each), 2026-10-31 13:55-14:18

Driver `~/gold-catalog/scripts/run_all.py --shakedown 20` (12 mesh workers, <= 6 concurrent cuDSS solves, memory-budget
admission, global 12 GiB watchdog: never triggered). Outputs `~/gold-runs/<part>/<set>/status.json`; interim aggregate in
`~/gold-runs/RESULTS.md` / `results.csv` / `cases.csv` (30 cases = these 20 + the 10 sets of abc_00000627 from the driver test).
The full run has NOT been started (held for the GPU video render). Nothing of mine is running.

## Outcome (20 cases)
- ok 15, gate_fail 4 (mesh failed `sicn` after hint-driven fixes; solved and scored anyway), solve_fail 1
  (abc_00000476: mesh 3.90M nodes = 11.7M DOFs, est 330 GB > 92 GB budget -> not admitted; same automesh blow-up as
  abc_00000223 in the pilot; repro saved in ~/gold-catalog/repro/abc_00000223_automesh_blowup/). mesh_fail 0, timeout 0.
- 19 solved cases, all with cuDSS: 63k-1.88M DOFs (median 115k; 0.31 x gold L2 DOFs). Over the 30 scored cases:
  98.2 % of certified QoIs inside their band (excluding 3 near-zero reaction components, see RESULTS.md), median |err| 0.37 %,
  p90 1.5 %, max 5.6 % (a certified vm_max). The generator's own uniform L1 mesh scores 82 % on the same QoIs.

## Time split
- Wall 22.6 min, but 19 of 20 cases were finished after 6.2 min (13:55:36-14:01:50); the last 16 min were one part
  (abc_00000476, 257 faces) meshing twice at ~10 min per mesh. Throughput as recorded: 0.88 cases/min; excluding the
  blow-up part: ~3.1 cases/min with only 20 cases in flight (mesh-limited ramp-up, no BC-set reuse).
- Mesh stage per part (auto_spec + check + fixes): median 9 s, mean 41 s over 19 normal parts (max 357 s, a 150-face
  part needing 3 fixes); 7 of 21 parts needed fixes (sicn), 5 still fail sicn.
- Solve process per case (Python start, assembly, cuDSS, recovery, QoIs): median 16 s, mean 21.5 s, max 99 s (1.88M DOFs);
  queue wait <= 6 s; GPU utilisation 54 % mean while a solve ran (nvidia-smi, 5 s samples); peak host RSS max 8.8 GB.

## Projection for the full set (3,161 imported sets on 349 parts, ~10 sets per part, one mesh per part)
- Meshing: ~345 normal parts x 41 s + ~4-6 blow-up parts (the >=200-face family) x ~10 min = ~5 h worker time / 12
  workers -> ~25-30 min, overlapped with solving. (Fix added after the shakedown: once a mesh is already beyond the
  solvable DOF limit the hint-driven fix loop stops, so a blow-up costs one ~10-min mesh, not two.)
- Solving: 3,161 x ~21.5 s = ~19 h process time. With <= 6 concurrent: ~3.2 h if the small/medium cases parallelise;
  the ~10 % of cases above 1M DOFs (est 30-55 GB each by the 28 KB/DOF rule) run 1-2 at a time and add ~1.5-2 h.
  Projected wall: **~4.5-5.5 h**, solve-bound (~10-12 cases/min for typical cases). Measuring the real GPU-side peak
  of a 1-2M-DOF solve and lowering the 28 KB/DOF estimate accordingly would let 2-3 large solves overlap (-1 h).
- Expected status mix (from the 20): ~75 % ok, ~20 % gate_fail (sicn), ~2-5 % solve_fail from mesh blow-ups.

## Changes made after the pilot / shakedown (all in ~/gold-catalog/scripts)
- One mesh per part for all its BC sets (spec groups = union of every face any set uses); per-set case via the adapter
  with that set's bcs (pilot check: groups do not change the mesh, same node count with/without them).
- auto_spec (exactly what automesh writes) instead of automesh, saving automesh's extra mesh per part.
- Hint-driven gate fixes (sicn/inverted: the hint's refine snippet -> size.min x3 -> size.global x0.7; mesh_generated:
  algorithm.mesh3d 1, heal; area/volume fidelity: refine group / curvature x2), <= 3 fixes, stop after 15 min.
- Unsolvable meshes are rejected before building cases; driver race fixes (case counting, watchdog requeue).
