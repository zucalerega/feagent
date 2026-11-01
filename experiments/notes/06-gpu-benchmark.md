# 06 - What the local RTX 4060 Laptop (8 GB) buys for feacore FEA

Date 2026-10-31. Host: 22 logical cores, 30 GB RAM, RTX 4060 Laptop 8188 MiB, driver 595.99.02 (CUDA 13.2).
Worktree: detached at `mesh-harness` (dd7292c). Nothing committed.

## 1. Install
```
uv sync --extra ui
uv pip install "nvmath-python[cu13]" cupy-cuda13x      # cu13, not cu12 (driver reports CUDA 13.2) -> nvmath 1.0.0, cupy 14.2.0, nvidia-cudss-cu13 0.8.0.10
uv pip install pypardiso                                # pulls MKL; pardiso then shows available
uv run python scripts/gpu_check.py
```
(pyproject's `gpu` extra pins `cupy-cuda12x`; on this driver cu13 worked first try, cu12 not tried.)

gpu_check output (excerpt):
```
gpu:  NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 595.99.02
linear solvers available: {'scipy': True, 'pardiso': False(before pypardiso), 'cudss': True, 'cg': True}
3D Laplacian n=64000:  cudss 1.027 s (res 1.2e-15)   scipy 10.800 s
nafems_le10 h=0.2 (22212 dofs): cudss 1.83 s total, QoI diff vs scipy 0.0 ; scipy 4.12 s total
GPU CHECK PASSED
```

## 2. Benchmark
Problem: cantilever box 1.0 x 0.2 x 0.2, tet10 (gmsh order 2), steel, fixed x=0 face, 1 MPa traction on x=L,
unstructured Delaunay mesh, `feacore.solver.skfem_backend.solve` (fast assembly, L2 stress recovery, 6 projection RHS).
Script: `bench_gpu.py` (this dir), one fresh process per (size, method), stage timings by monkeypatching the backend.
Driver: `run_series.sh "<h list>" "<method list>"`. Raw: `raw.log`.
Methods: `scipy` (SuperLU), `pardiso` (MKL via pypardiso), `cudss` (nvmath, feacore default path),
`cupycg` (cupyx CG + Jacobi, rtol 1e-8, main solve only; L2 projection falls to SciPy), `cudss_hybrid` (cuDSS hybrid-memory mode, factors in host RAM; not exposed by feacore, patched in bench).

Seconds. "solve" = factorise+solve of the static system. "recov" = whole stress recovery (QP stress + mass assembly + 6-RHS L2 projection solve).
asm = stiffness assembly; read = mesh read + basis setup; mesh = gmsh (tet10) generation, only shown when generated in that run.

| DOFs | cells | method | asm | solve | recov | read | total solve() | host peak RSS GB | GPU peak MiB* |
|---|---|---|---|---|---|---|---|---|---|
| 14.7k | 2.7k | cudss | 0.66 | 1.32 | 0.16 | 0.08 | 2.24 | 0.7 | 225 |
| 37k | 7.5k | cudss | 0.43 | 1.12 | 0.35 | 0.18 | 2.10 | 0.85 | 387 |
| 37k | | pardiso | 0.33 | 0.49 | 0.23 | 0.18 | 1.25 | 0.45 | - |
| 37k | | scipy | 0.42 | 5.28 | 1.11 | 0.18 | 7.02 | 0.63 | - |
| 129k | 28k | cudss | 3.31 | 3.15 | 1.55 | 0.72 | 8.98 | 1.28 | 1267 |
| 129k | | pardiso | 3.30 | 2.27 | 0.86 | 0.81 | 7.51 | 1.63 | - |
| 129k | | scipy | 3.30 | **185** | 8.5 | 0.76 | 198 | 4.05 | - |
| 129k | | cupycg (1901 it) | 3.15 | 4.15 | 10.8 | 0.83 | 19.3 | 1.68 | (contaminated) |
| 200k | 44k | cudss | 1.48 | 5.34 | 1.47 | 1.40 | 9.82 | 1.64 | 2111 |
| 200k | | pardiso | 1.34 | 4.48 | 1.07 | 1.04 | 8.06 | 2.72 | - |
| 200k | | cupycg (2231 it) | 1.42 | 3.30 | 14.9 | 1.04 | 20.8 | 2.28 | 428 |
| 480k | 110k | cudss | 6.62 | 23.2 | 7.2 | 2.69 | 40.1 | 3.0 | **6575** |
| 480k | | pardiso | 4.27 | 19.96 | 4.6 | 2.82 | 32.1 | 7.8 | - |
| 480k | | cupycg (2984 it) | 4.18 | 9.92 | **180.7** | 2.56 | 197.7 | 7.0 | 615 |
| 934k | 218k | cudss | - | **fails: "Not enough memory to perform factorization"** (device OOM at 8 GB) | | | | | |
| 934k | | cudss_hybrid | 6.8 | 59.5 | 7.3 | 5.3 | 79.6 | 18.5 | n/a** |
| 934k | | pardiso | 41 (swap-bound) | 75.6 | 10.8 | 5.3 | 139 | **18.2** | - |
| 934k | | cupycg | - | killed by 600 s timeout (SciPy SuperLU mass-matrix projection) | | | | | |
| ~2.0M (h=0.0075) | | cudss | device OOM (see note) | | | | | | |
| ~2.0M | | pardiso | OOM-killed by kernel (exit 137, 30 GB host) | | | | | | |
| ~2.0M | | cudss_hybrid | did not finish in 600 s; GPU then dropped off the bus (see below) | | | | | | |

\* GPU peak is the max `nvidia-smi memory.used` polled at 0.2 s, includes ~11 MiB baseline. Noisy: first-run rows
overlapped with a just-killed process (the 129k pardiso/cupycg rows showed 829/3918 MiB junk and are not trusted).
\*\* nvidia-smi lost the GPU ("Unable to determine the device handle for GPU0 ... No devices were found") before/during this run, so the poll read 0.
tet10 mesh generation (gmsh, 1 thread): 0.1 s @15k, 1.2 s @200k, 2.7 s @480k DOFs (not measured above 480k in the table; small vs everything else).
No ~50k/200k/500k/1M/2M exact match: actual sizes 37k/129k/200k/480k/934k/(~2M attempted); tip deflection identical (-0.0005 m) for all methods.

**Max DOFs on 8 GB**
- cuDSS in-core (feacore's default `_solve_cudss`): worked at 480k DOFs (6.6 GB peak of 8.2 GB), fails at 934k. So practical cap ~ 500-600k DOFs for 3D tet10 unstructured (fill-in heavy; GPU memory ~13 KB/DOF).
- cuDSS hybrid memory mode: solves 934k in 59 s (faster than PARDISO 76 s) but needs ~18 GB host RAM total; 2M did not complete in 10 min.
- Host RAM is the real ceiling: even with the solve on GPU, host RSS was 18 GB at 934k (skfem data structures, K copies); 2M cannot fit in 30 GB (PARDISO was OOM-killed).

## 3. Findings / bottlenecks
1. **GPU speedup is small for this size range.** Against MKL PARDISO (22 cores), cuDSS is *slower* (480k: 23.2 s vs 20.0 s; 200k: 5.3 vs 4.5). Only vs SciPy SuperLU is it huge (129k: 3.1 s vs 185 s; ~60x), and feacore's `auto` falls back to SciPy if PARDISO is not installed - so the shipped "GPU win" is mostly a win over the weakest CPU baseline. cuDSS also has ~1 s fixed startup (14.7k dofs: 1.3 s vs 0.49 s PARDISO at 37k). cuDSS pays off mostly where PARDISO's host memory (7.8 GB at 480k, 18 GB at 934k) or time (hybrid 59 s vs 76 s at 934k) is the limit, and on a weaker CPU.
2. **8 GB VRAM is the hard wall for direct solves: ~0.5M DOFs** (3D, tet10). A whole airplane at 10^7-10^8 DOFs is out of reach for single-GPU direct factorisation; needs multi-GPU/MPI cuDSS, domain decomposition, or iterative/multigrid.
3. **Silent fallback hides OOM**: when cuDSS reports "Not enough memory to perform factorization", `solve_linear` falls back to SciPy SuperLU (note in report) which then dies with MemoryError / runs for minutes. For production, treat it as a failure or try the hybrid mode.
4. **Python assembly dominates once the solve is fast**: at 480k, assembly (4-7 s) + mesh read/basis (2.7 s) + recovery (5-7 s) = about 40% of solve() wall time with cuDSS; at <=200k, non-solve stages are 45-70% of total. Assembly scales ~linearly, ~13 us/DOF (but 41 s at 934k when memory-bound/swapping). Host memory per DOF: ~18-20 KB/DOF in the skfem pipeline, so ~1.5M DOFs would exhaust 30 GB regardless of solver.
5. **Stress recovery solves the L2 mass system with the same `solve_linear` (6 RHS, factorise once)**: cheap with PARDISO/cuDSS (3-5 s at 480k) but with the SciPy path it is 8.5 s @129k and 172 s @480k - a trap when the main solve is iterative/GPU CG and projection silently goes to SuperLU.
6. **Jacobi-CG on GPU is not competitive**: 1900 -> 3000 iterations (grows ~h^-1, ill-conditioned P2 elasticity), 4-10 s per solve; faster than direct at 480k only on the main solve (9.9 s vs 20-23 s) but the total is dominated by recovery. A GPU AMG/smoothed-aggregation (pyamg is CPU only; AMGX/cuPy-based) would be needed. CG relative residual hit rtol 1e-8 (printed as 0.0 due to rounding); tip deflection matched direct.
7. **Stability**: after the 2M-DOF attempts (cuDSS OOM, hybrid mode killed on timeout) the GPU vanished from the driver ("No devices were found"; one run before that already saw cudaErrorNoDevice). Likely driver/GPU reset under memory pressure on a laptop part; reboot/reset may be needed. Treat >7 GB cuDSS runs as risky locally.

## Commands
```
bash run_series.sh "0.030 0.0189 0.014" "cudss pardiso scipy cupycg"     # series 1 (aborted after scipy took 185 s)
bash run_series.sh "0.0189 0.0163 0.012 0.0095 0.0075" "cudss pardiso cupycg"
bash run_series.sh "0.0095 0.0075" "cudss_hybrid"
```
h -> DOFs: 0.045:15k, 0.030:37k, 0.0189:129k, 0.0163:200k, 0.012:480k, 0.0095:934k, 0.0075:~1.8-2M.
Mesh files cached in $TMPDIR/box_<h>.msh (see bench_gpu.py --mesh).

## Raw log excerpt (raw.log, JSON lines)
RESULT {"h": 0.03, "method": "cudss", "dofs": 37410, "cells": 7503, "peak_rss_gb": 0.85, "gpu_peak_mib": 387, "mesh": 0.2458, "read_mesh_basis": 0.1774, "assembly": 0.4266, "linear_solve": 1.1176, "recovery_total": 0.3509, "solve_total": 2.1025, "recovery_proj_solve": 0.2217, "tip_uz": -0.0005, "nodal_vm": 0.0002}
RESULT {"h": 0.03, "method": "pardiso", "dofs": 37410, "cells": 7503, "peak_rss_gb": 0.45, "gpu_peak_mib": 11, "mesh": 0.0, "read_mesh_basis": 0.1775, "assembly": 0.3254, "linear_solve": 0.4881, "recovery_total": 0.231, "solve_total": 1.2481, "recovery_proj_solve": 0.1235, "tip_uz": -0.0005, "nodal_vm": 0.0003}
RESULT {"h": 0.03, "method": "scipy", "dofs": 37410, "cells": 7503, "peak_rss_gb": 0.63, "gpu_peak_mib": 11, "mesh": 0.0002, "read_mesh_basis": 0.177, "assembly": 0.419, "linear_solve": 5.2824, "recovery_total": 1.106, "solve_total": 7.0242, "recovery_proj_solve": 0.9628, "tip_uz": -0.0005, "nodal_vm": 0.0002}
FAIL h=0.030 m=cupycg rc=1: TypeError: cg() got an unexpected keyword argument 'tol'
RESULT {"h": 0.0189, "method": "cudss", "dofs": 129000, "cells": 28059, "peak_rss_gb": 1.28, "gpu_peak_mib": 1267, "mesh": 0.9128, "read_mesh_basis": 0.7162, "assembly": 3.3079, "linear_solve": 3.1455, "recovery_total": 1.55, "solve_total": 8.9798, "recovery_proj_solve": 0.639, "tip_uz": -0.0005, "nodal_vm": 0.0006}
RESULT {"h": 0.0189, "method": "pardiso", "dofs": 129000, "cells": 28059, "peak_rss_gb": 1.63, "gpu_peak_mib": 11, "mesh": 0.002, "read_mesh_basis": 0.8099, "assembly": 3.3021, "linear_solve": 2.2685, "recovery_total": 0.8622, "solve_total": 7.5134, "recovery_proj_solve": 0.4173, "tip_uz": -0.0005, "nodal_vm": 0.001}
RESULT {"h": 0.0189, "method": "scipy", "dofs": 129000, "cells": 28059, "peak_rss_gb": 4.05, "gpu_peak_mib": 266, "mesh": 0.0, "read_mesh_basis": 0.7599, "assembly": 3.304, "linear_solve": 185.2348, "recovery_total": 8.469, "solve_total": 198.0533, "recovery_proj_solve": 8.0685, "tip_uz": -0.0005, "nodal_vm": 0.0014}
FAIL h=0.0189 m=cupycg rc=1: AttributeError: 'R' object has no attribute 'describe'
RESULT {"h": 0.0189, "method": "pardiso", "dofs": 129000, "cells": 28059, "peak_rss_gb": 1.61, "gpu_peak_mib": 829, "mesh": 0.0, "read_mesh_basis": 1.0555, "assembly": 2.3938, "linear_solve": 2.7828, "recovery_total": 1.0354, "solve_total": 7.7866, "recovery_proj_solve": 0.4118, "tip_uz": -0.0005, "nodal_vm": 0.0008}
RESULT {"h": 0.0189, "method": "cupycg", "dofs": 129000, "cells": 28059, "peak_rss_gb": 1.68, "gpu_peak_mib": 3918, "mesh": 0.0, "read_mesh_basis": 0.8311, "assembly": 3.152, "cg_iters": 1901, "cg_info": 0, "cg_relres": 0.0, "linear_solve": 4.1455, "recovery_total": 10.8379, "solve_total": 19.3474, "recovery_proj_solve": 10.0404, "tip_uz": -0.0005, "nodal_vm": 0.0008}
RESULT {"h": 0.0163, "method": "cudss", "dofs": 199809, "cells": 44297, "peak_rss_gb": 1.64, "gpu_peak_mib": 2111, "mesh": 1.1846, "read_mesh_basis": 1.4015, "assembly": 1.4785, "linear_solve": 5.3413, "recovery_total": 1.4702, "solve_total": 9.8189, "recovery_proj_solve": 0.9666, "tip_uz": -0.0005, "nodal_vm": 0.0015}
RESULT {"h": 0.0163, "method": "pardiso", "dofs": 199809, "cells": 44297, "peak_rss_gb": 2.72, "gpu_peak_mib": 11, "mesh": 0.0, "read_mesh_basis": 1.0397, "assembly": 1.34, "linear_solve": 4.4831, "recovery_total": 1.0694, "solve_total": 8.0628, "recovery_proj_solve": 0.506, "tip_uz": -0.0005, "nodal_vm": 0.0012}
RESULT {"h": 0.0163, "method": "cupycg", "dofs": 199809, "cells": 44297, "peak_rss_gb": 2.28, "gpu_peak_mib": 428, "mesh": 0.0, "read_mesh_basis": 1.0373, "assembly": 1.4185, "cg_iters": 2231, "cg_info": 0, "cg_relres": 0.0, "linear_solve": 3.2968, "recovery_total": 14.8965, "solve_total": 20.7905, "recovery_proj_solve": 14.3877, "tip_uz": -0.0005, "nodal_vm": 0.0011}
RESULT {"h": 0.012, "method": "cudss", "dofs": 480351, "cells": 109862, "peak_rss_gb": 3.01, "gpu_peak_mib": 6575, "mesh": 2.6801, "read_mesh_basis": 2.6876, "assembly": 6.6195, "linear_solve": 23.1681, "recovery_total": 7.2344, "solve_total": 40.1174, "recovery_proj_solve": 3.0522, "tip_uz": -0.0005, "nodal_vm": 0.0024}
RESULT {"h": 0.012, "method": "pardiso", "dofs": 480351, "cells": 109862, "peak_rss_gb": 7.84, "gpu_peak_mib": 11, "mesh": 0.0, "read_mesh_basis": 2.8211, "assembly": 4.2719, "linear_solve": 19.9628, "recovery_total": 4.6185, "solve_total": 32.0934, "recovery_proj_solve": 1.7327, "tip_uz": -0.0005, "nodal_vm": 0.0028}
RESULT {"h": 0.012, "method": "cupycg", "dofs": 480351, "cells": 109862, "peak_rss_gb": 6.96, "gpu_peak_mib": 615, "mesh": 0.0, "read_mesh_basis": 2.5582, "assembly": 4.1824, "cg_iters": 2984, "cg_info": 0, "cg_relres": 0.0, "linear_solve": 9.9193, "recovery_total": 180.7238, "solve_total": 197.7146, "recovery_proj_solve": 172.2111, "tip_uz": -0.0005, "nodal_vm": 0.0023}
FAIL h=0.0095 m=cudss rc=1: MemoryError
RESULT {"h": 0.0095, "method": "pardiso", "dofs": 933993, "cells": 217815, "peak_rss_gb": 18.21, "gpu_peak_mib": 11, "mesh": 0.0, "read_mesh_basis": 5.3192, "assembly": 41.0845, "linear_solve": 75.5685, "recovery_total": 10.8296, "solve_total": 139.1101, "recovery_proj_solve": 4.8854, "tip_uz": -0.0005, "nodal_vm": 0.0044}
FAIL h=0.0095 m=cupycg rc=124: 
FAIL h=0.0075 m=cudss rc=1: MemoryError
FAIL h=0.0075 m=pardiso rc=137: 
FAIL h=0.0075 m=cupycg rc=1: cupy_backends.cuda.api.runtime.CUDARuntimeError: cudaErrorNoDevice: no CUDA-capable device is detected
DONE
RESULT {"h": 0.0095, "method": "cudss_hybrid", "dofs": 933993, "cells": 217815, "peak_rss_gb": 18.51, "gpu_peak_mib": 0, "mesh": 0.0008, "read_mesh_basis": 5.3417, "assembly": 6.8109, "linear_solve": 59.4931, "recovery_total": 7.2851, "solve_total": 79.6053, "recovery_proj_solve": 4.4779, "tip_uz": -0.0005, "nodal_vm": 0.005}
FAIL h=0.0075 m=cudss_hybrid rc=143: 
DONE
