# FEAgent env on Spark (2026-10-31)
Activate / run (always under the cap; user scopes work, no prlimit needed):
  cd ~/feagent && systemd-run --user --scope -q -p MemoryMax=20G ~/micromamba/envs/feagent/bin/python <script>
  (or: export MAMBA_ROOT_PREFIX=~/micromamba; ~/micromamba/bin/micromamba run -n feagent python ...)
Layout: old-style src/feagent + feacore/ + ace/ at repo root (branch spark-local-llm, tracks origin/ace-loop). No meshing-harness/ dir. `pip install -e . --no-deps` done (gmsh from conda).
Origin branches: main, mesh-harness, ace-harness, ace-loop, sculptured-team, improver-loop (HEAD), plus ace-cfd, ace-ui-bridge, benchmarks, cfd-openfoam, fea-*, gold-benchmarks, harness-fea-bcs, integrate-mesh-harness, merge-mesh-harness, physical-validation, single-agent-baseline, ui-* etc.
Versions: micromamba (23 MB) in ~/micromamba/bin; python 3.12; gmsh 4.15.2 (conda-forge), numpy 2.5.4, scipy 1.18.1, scikit-fem 12.0.2, pyamg 5.4.1, pyvista 0.49.2, vtk 9.7.1, mcp 2.3.0, meshio 5.3.5, h5py 3.16.0, anthropic 2.3.0, pytest 9.2.0, nvmath-python 1.0.0, cupy 14.2.0 (cupy-cuda13x). Extras also installed: pydantic, rapidfuzz, matplotlib, streamlit, plotly, altair, sympy, fastapi, uvicorn, httpx.
Sizes: conda env after gmsh/python 676 MB (15 s); final env 4.7 GB (CUDA libs via nvmath/cupy are most of it); pip cache 2.1 GB (~/.cache/pip, can be purged). pip step 3m47s.
free -g available: 30 GB before, 30 GB after (llm container untouched).
Checks: import gmsh, skfem, feacore, feagent, ace OK (gmsh 4.15.2); nvmath/cupy OK, getDeviceCount=1; scripts/gpu_check.py PASSED (3D Laplacian n=64000: cudss 0.70 s vs scipy 5.37 s; nafems_le10 22212 dofs cudss 0.54 s vs scipy 1.68 s; pardiso unavailable on ARM as expected); pytest -q -m "not slow" -x tests/test_solver_skfem.py: 51 passed in 0.9 s.
Failures: none. Benign nvmath warning about no multithreading lib in DirectSolverOptions. setup_gpu_box.sh not run (needs sudo apt); done by hand.
