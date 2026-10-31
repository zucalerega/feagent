"""Scaling benchmark: cantilever box, tet10 elasticity, feacore skfem backend.
Usage (from worktree root):  uv run python bench_gpu.py --h 0.02 --method cudss
Each invocation = one fresh process (clean peak RSS). Prints one JSON line 'RESULT {...}'.
methods: scipy | pardiso | cudss | cupycg (cupyx CG + Jacobi on GPU for the main solve; SciPy for L2 projection)
"""
import argparse, tempfile, json, os, resource, subprocess, threading, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h", type=float, required=True)
ap.add_argument("--method", default="cudss")
ap.add_argument("--mesh", default=None)
a = ap.parse_args()

T = {}
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB

gpu_peak = [0]; stop = [False]
def poll():
    while not stop[0]:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip()
            gpu_peak[0] = max(gpu_peak[0], int(o))
        except Exception: pass
        time.sleep(0.2)
threading.Thread(target=poll, daemon=True).start()

import gmsh
path = a.mesh or os.path.join(tempfile.gettempdir(), f"box_{a.h}.msh")
t = time.perf_counter()
if not os.path.exists(path):
    gmsh.initialize(["-v", "0"])
    L, W = 1.0, 0.2
    gmsh.model.occ.addBox(0, 0, 0, L, W, W); gmsh.model.occ.synchronize()
    vol = gmsh.model.getEntities(3)[0][1]
    for s in gmsh.model.getEntities(2):
        x = gmsh.model.occ.getCenterOfMass(*s)[0]
        if x < 1e-9: gmsh.model.addPhysicalGroup(2, [s[1]], name="fix")
        elif x > L - 1e-9: gmsh.model.addPhysicalGroup(2, [s[1]], name="load")
    gmsh.model.addPhysicalGroup(3, [vol], name="body")
    gmsh.option.setNumber("Mesh.MeshSizeMax", a.h); gmsh.option.setNumber("Mesh.MeshSizeMin", a.h)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(3); gmsh.model.mesh.setOrder(2)
    gmsh.write(path); gmsh.finalize()
T["mesh"] = time.perf_counter() - t

from feacore.case import Case, Analysis, Material, BC, Load, QoI
import feacore.solver.skfem_backend as sk

case = Case(mesh=path, analysis=Analysis(physics="elasticity", formulation="3d", order=2),
            materials={"m": Material(E=2e11, nu=0.3)}, regions=[],
            bcs=[BC(type="fixed", group="fix"), BC(type="traction", group="load", value=[0, 0, -1e6])],
            qois=[QoI(name="tip", type="point", field="uz", point=[1.0, 0.1, 0.1])])

orig_asm, orig_rec, orig_lin, orig_init = sk._assemble_elasticity, sk._recover_elasticity, sk.solve_linear, sk._Model.__init__
def timed(name, fn):
    def w(*aa, **kk):
        t = time.perf_counter(); r = fn(*aa, **kk); T[name] = T.get(name, 0) + time.perf_counter() - t; return r
    return w
sk._assemble_elasticity = timed("assembly", orig_asm)
sk._recover_elasticity = timed("recovery_total", orig_rec)  # stress at QPs + mass assembly + L2 projection solve
sk._Model.__init__ = timed("read_mesh_basis", orig_init)
rec_lin = [0.0]; n_calls = [0]

def lin(K, f, **kw):
    n_calls[0] += 1
    t = time.perf_counter()
    if a.method == "cupycg" and n_calls[0] == 1:
        import cupy as cp, cupyx.scipy.sparse as cs, cupyx.scipy.sparse.linalg as csl
        Kg = cs.csr_matrix(K); bg = cp.asarray(f)
        d = Kg.diagonal()
        M = csl.LinearOperator(Kg.shape, lambda v: v / d, dtype=cp.float64)
        its = [0]
        def cb(xk): its[0] += 1
        x, info = csl.cg(Kg, bg, rtol=1e-8, maxiter=int(os.environ.get("MAXIT", 20000)), M=M, callback=cb)
        x = cp.asnumpy(x)
        res = float(np.linalg.norm(K @ x - f) / np.linalg.norm(f))
        T["cg_iters"] = its[0]; T["cg_info"] = int(info); T["cg_relres"] = res
        T["linear_solve"] = time.perf_counter() - t
        from feacore.solver.linsolve import LinearSolveReport
        R = LinearSolveReport(method="cupy_cg", requested="cupy_cg", n=K.shape[0], nnz=K.nnz)
        R.residual = res; R.iterative = True; R.detail = "cupy cg jacobi"; R.tolerance = 1e-8
        return x, R
    kw = dict(kw); kw["method"] = "scipy" if a.method == "cupycg" else ("cudss" if a.method == "cudss_hybrid" else a.method)
    r = orig_lin(K, f, **kw)
    if n_calls[0] == 1: T["linear_solve"] = time.perf_counter() - t
    else: rec_lin[0] += time.perf_counter() - t
    return r
sk.solve_linear = lin

if a.method == "cudss_hybrid":   # cuDSS hybrid memory mode: factors in host RAM (feacore has no such option)
    import scipy.sparse as sp
    import feacore.solver.linsolve as ls
    from nvmath.sparse.advanced import (DirectSolverMatrixType, DirectSolverMatrixViewType, DirectSolverOptions,
                                        direct_solver, ExecutionCUDA, HybridMemoryModeOptions)
    def _hyb(K, f, report, symmetric):
        A = sp.csr_matrix(K, dtype=np.float64); A.sort_indices()
        b = np.asfortranarray(np.asarray(f, dtype=np.float64))
        opt = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.FULL)
        ex = ExecutionCUDA(hybrid_memory_mode_options=HybridMemoryModeOptions(hybrid_memory_mode=True))
        x = direct_solver(A, b, options=opt, execution=ex)
        report.detail = "cuDSS hybrid memory"
        return np.asarray(x).reshape(np.shape(f))
    ls._solve_cudss = _hyb
    a.method_eff = "cudss"

t0 = time.perf_counter()
res = sk.solve(case, linear_solver=("scipy" if a.method == "cupycg" else ("cudss" if a.method == "cudss_hybrid" else a.method)))
T["solve_total"] = time.perf_counter() - t0
T["recovery_proj_solve"] = rec_lin[0]
T["tip_uz"] = res.evaluate(case.qois[0])
t = time.perf_counter(); res.nodal("von_mises"); T["nodal_vm"] = time.perf_counter() - t
out = dict(h=a.h, method=a.method, dofs=int(res.info.n_dofs), cells=int(res.info.n_cells),
           peak_rss_gb=round(rss(), 2), gpu_peak_mib=gpu_peak[0],
           **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in T.items()})
stop[0] = True
print("RESULT", json.dumps(out))
