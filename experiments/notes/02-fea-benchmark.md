# 02 - FEAgent fixed FEA solver on DGX Spark (GB10, aarch64, 121 GB unified)

Date 2026-10-31. Problem identical to the laptop benchmark (cantilever box 1.0x0.2x0.2, tet10, steel, 1 MPa traction,
`feacore.solver.skfem_backend.solve`, old `src/feagent`+`feacore/` layout, branch spark-local-llm). Script: `~/bench/bench_spark.py`
(copy of the laptop bench_gpu.py; changes: mesh cache in ~/bench, nvidia-smi poll replaced by a 0.1 s thread reading
`cupy.cuda.runtime.memGetInfo()` and /proc/meminfo, and every `solve_linear` LinearSolveReport is recorded so the solver actually used and any silent-fallback notes are verified).
Spark: 20 Arm cores, no PARDISO. One fresh process per run, under `systemd-run --user --scope -p MemoryMax=20G`, plus a 1 s watchdog
(`~/bench/run_one.sh`) that stops the scope if MemAvailable < 10 GiB (never triggered).

## IMPORTANT caveat: the vLLM container stopped mid-series
`docker ps -a` is empty now. MemAvailable was ~25-30 GiB through the 129k/200k runs (llm running, as briefed) and jumped to ~117 GiB between
12:44:23 and 12:44:53. I did not touch docker; something else stopped `llm`. So the 37k-200k rows ran WITH vLLM, the 480k and 929k rows WITHOUT it.
The watchdog (10 GiB floor) was active for all runs regardless. The 929k run was only attempted because 480k passed the 16 GB test and memory was free; it exceeded the limit, so the series stopped there.

## Results (seconds)
GPU MiB* = peak drop of cudaMemGetInfo free memory during the run (unified memory: this is a system-wide proxy and may include host-side pages, so RSS+GPU can slightly double count). min avail = minimum MemAvailable (GiB) seen by the watchdog during the run.

| DOFs | cells-ish | method | linear solver verified | mesh | read+basis | assembly | solve (main) | recovery total | total solve() | peak RSS GB | GPU MiB* | min avail GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14.8k | | cudss | cudss (no notes) | 0.05 | 0.04 | 0.08 | 0.33 | 0.06 | 0.53 | 0.82 | 466 | 30.0 |
| 37.8k | | cudss | cudss | 0.10 | 0.11 | 0.21 | 0.51 | 0.15 | 1.01 | 1.11 | 737 | 29.6 |
| 37.8k | | scipy | scipy | - | 0.11 | 0.21 | 3.19 | 0.44 | 3.98 | 0.98 | 616 | 29.7 |
| 128.9k | | cudss | cudss | 0.38 | 0.42 | 0.85 | 1.45 | 0.55 | 3.35 | 1.53 | 1670 | 28.2 |
| 128.9k | | scipy | scipy | - | 0.41 | 0.73 | 190 | 4.9 | 196 | 5.4 | 2761 | 25.6 |
| 128.9k | | cudss_hybrid | cudss (hybrid patch) | - | 0.42 | 0.74 | 1.73 | 0.54 | 3.49 | 1.48 | 2193 | 115 (llm gone) |
| 199.1k | | cudss | cudss | 0.68 | 0.73 | 1.22 | 3.53 | 0.87 | 6.46 | 1.83 | 3014 | 27.1 |
| 199.1k | | scipy | scipy | - | 0.69 | 1.13 | 577 | 17.6 | 596 | 14.4 | 3828 | 25.2 |
| 479.5k | | cudss | cudss | 1.81 | 1.76 | 2.69 | 11.4 | 2.49 | 18.6 | 3.24 | 9283 | 108 (llm gone) |
| 479.5k | | cudss_hybrid | cudss (hybrid patch) | - | 1.87 | 2.83 | 11.0 | 2.58 | 18.6 | 3.24 | 9196 | 108 |
| 928.9k | | cudss | cudss | 4.40 | 3.72 | 5.48 | 36.1 | 5.86 | 51.7 | 5.43 | 20577 | 97.5 (llm gone) |

Tip deflection -0.0005 m in all runs; relative residuals 1e-11..1e-16. No silent fallbacks: every cudss run's two solve_linear calls (main + L2 projection) reported method=cudss, notes=[]. (scipy rows are the explicit scipy method.)
DOF counts differ slightly from the laptop (different gmsh version): 129k vs 129.0k, 199k vs 200k, 480k, 929k vs 934k.
Series stop: 929k run total ~26 GB (RSS 5.4 + GPU 20.6) > 16 GB, so no larger sizes were run. Raw: `~/bench/raw.log`, `~/bench/summary.txt`, `~/bench/watchdog.log`.

## Comparison with the laptop (22-core CPU, MKL PARDISO, numbers from 06-gpu-benchmark.md; not re-run)
| DOFs | laptop PARDISO total (solve) | laptop cuDSS 4060 total (solve) | Spark cuDSS total (solve) | Spark scipy total (solve) |
|---|---|---|---|---|
| 37k | 1.25 (0.49) | 2.10 (1.12) | 1.01 (0.51) | 3.98 (3.19) |
| 129k | 7.51 (2.27) | 8.98 (3.15) | 3.35 (1.45) | 196 (190) |
| 200k | 8.06 (4.48) | 9.82 (5.34) | 6.46 (3.53) | 596 (577) |
| 480k | 32.1 (19.96) | 40.1 (23.2) | 18.6 (11.4) | n/a |
| 934k/929k | 139 (75.6, 18.2 GB RSS, swap-bound) | OOM (8 GB) | 51.7 (36.1, 5.4 GB RSS) | n/a |

## Conclusions
- What the Spark buys vs laptop PARDISO: the GPU path is now faster than the best CPU path, modestly: 1.6-2.2x on the main solve at 129k-480k (1.45 vs 2.27 s; 3.5 vs 4.5; 11.4 vs 20.0), ~2.2x total at 129k and 480k, and 2.1x on the main solve / 2.7x total at ~930k, where the laptop needed 18 GB host RAM and swapped while the Spark used 5.4 GB RSS. Vs the 8 GB laptop GPU, the Spark's cuDSS is 2-2.7x faster and has no 480k wall (laptop failed at 934k). At 37k the Spark cuDSS (1.0 s) is about on par with laptop PARDISO (1.25 s).
- vs SciPy (what feacore falls back to with no PARDISO, i.e. the ARM CPU default): 130-370x faster on the main solve (190 s -> 1.5 s at 129k; 577 s -> 3.5 s at 200k). On ARM without PARDISO, cuDSS is effectively mandatory above ~50k DOFs. SciPy at 200k also took 14.4 GB RSS.
- Non-solve stages (mesh, read, assembly, recovery) are now 40-55% of total below 200k and ~40% at 480k; Python assembly is ~5.7 us/DOF here vs ~13 us on the laptop.
- cuDSS hybrid-memory patch: no effect on unified memory (129k: 3.49 vs 3.35 s; 480k: 18.56 vs 18.58 s; GPU/host figures within noise). Not needed on the Spark.
- Memory model for cuDSS here: ~19-22 KB/DOF GPU-side (superlinear, exponent ~1.2 from 480k->929k: 9.3 -> 20.6 GB) + ~5.8 KB/DOF host RSS.
- DOF ceiling under the 20 GB cap WITH vLLM running (measured basis: ~25-30 GiB available, 10 GiB watchdog floor, so ~15-20 GiB usable): about 480k DOFs measured fine (12.5 GB total; was run after vLLM disappeared, but the footprint is what counts); the 16 GB budget is hit at roughly 550-600k DOFs and 929k (26 GB) is over. Practical ceiling with vLLM resident: ~0.5-0.6M DOFs. Note the cgroup cap did not appear to constrain GPU allocations (not tested to breach), so the watchdog is the real guard.
- EXTRAPOLATION (not measured): without vLLM, ~100 GB usable after a 10 GiB margin, at ~26 GB per 0.93M and exponent ~1.2 gives ~2.5M DOFs for cuDSS in-core tet10 (about 2-3M). Host RSS (~15 GB at 2.5M) would still fit under the 20 GB cap; GPU-side would not be capped by it. Beyond that: multi-GPU cuDSS, domain decomposition or multigrid.

## Commands
```
scp bench_spark.py dell@<spark-host>:~/bench/        # ~/bench/run_one.sh h method does the below + watchdog
cd ~/feagent && systemd-run --user --scope -q -p MemoryMax=20G ~/micromamba/envs/feagent/bin/python ~/bench/bench_spark.py --h 0.012 --method cudss
for h in 0.045 0.03 0.0189 0.0163 0.012 0.0095: ~/bench/run_one.sh $h cudss     (scipy at 0.03/0.0189/0.0163; cudss_hybrid at 0.0189/0.012)
```
Watchdog loop (in run_one.sh): every 1 s, `MemAvailable` from /proc/meminfo; if < 10485760 kB then `systemctl --user stop <unit>.scope`; logs min.
## Watchdog log (~/bench/watchdog.log; killed=0 for every run)
```
12:30:41 h=0.045 method=cudss rc=0 killed=0 min_MemAvailable_GiB=30.24
12:30:49 h=0.03 method=cudss rc=0 killed=0 min_MemAvailable_GiB=30.11
12:30:54 h=0.03 method=scipy rc=0 killed=0 min_MemAvailable_GiB=29.73
12:30:59 h=0.0189 method=cudss rc=0 killed=0 min_MemAvailable_GiB=28.19
12:34:16 h=0.0189 method=scipy rc=0 killed=0 min_MemAvailable_GiB=25.64
12:34:25 h=0.0163 method=cudss rc=0 killed=0 min_MemAvailable_GiB=27.04
12:44:23 h=0.0163 method=scipy rc=0 killed=0 min_MemAvailable_GiB=25.20
12:44:53 h=0.012 method=cudss rc=0 killed=0 min_MemAvailable_GiB=108.53
12:46:05 h=0.0095 method=cudss rc=0 killed=0 min_MemAvailable_GiB=97.45
12:46:16 h=0.0189 method=cudss_hybrid rc=0 killed=0 min_MemAvailable_GiB=115.32
12:46:36 h=0.012 method=cudss_hybrid rc=0 killed=0 min_MemAvailable_GiB=108.07
```
