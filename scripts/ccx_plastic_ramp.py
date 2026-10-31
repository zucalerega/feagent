"""Elastoplastic load ramp of a gold / harness part with the opt-in CalculiX nonlinear path.

    python scripts/ccx_plastic_ramp.py SET_DIR OUT_DIR [--ratio 2.5] [--n 20] [--threads 8] [--tangent 1.5e9]

SET_DIR holds case.json (+ the mesh it names, or mesh.msh next to it). Steps:
1. linear ccx solve of the case as given; peak von Mises excluding singular corners (a node is singular when
   its stress is > 3x the median of its 16 nearest nodes and above the 99.5th percentile, as the video's
   build_ramp.py; the singular nodes and their 16 neighbours are excluded);
2. every load / prescribed displacement x k so that the linear reference peak = ratio x sy (S355 steel:
   E 210 GPa, nu 0.3, sy 355 MPa, linear hardening with tangent modulus --tangent);
3. nonlinear ccx run, ``increments`` fixed equal steps to lambda = 1 (on divergence: re-run adaptive);
4. OUT_DIR: inc_000..NNN.vtu + inc.pvd + summary.json (per-increment QoIs, plus the linear comparison and
   redistribution measures below). Extra .vtu fields: von_mises_linear (lambda x the linear field) and
   vm_ratio (plastic / linear von Mises).
Redistribution (per increment): peak von Mises off the singular set, plastic vs linear; volume with
von Mises >= 0.9 sy, plastic vs linear; volume of the still-elastic region where the plastic solution
carries > 5 % more von Mises than the linear one ("load shed onto the neighbours"), and its max ratio.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from feacore.case import Case
from feacore.solver import solve

E, NU, SY = 210e9, 0.3, 355e6


def singular_mask(X, vm, k=17):
    from scipy.spatial import cKDTree
    _, nb = cKDTree(X).query(X, k=min(k, len(X)))
    med = np.median(vm[nb[:, 1:]], axis=1)
    sing = (vm > 3.0 * np.maximum(med, 1e-30)) & (vm > np.percentile(vm, 99.5))
    excl = np.zeros(len(X), bool)
    excl[nb[sing].ravel()] = True
    return sing, excl


def nodal_volumes(model, measures):
    w = np.zeros(model.n_nodes)
    np.add.at(w, model.conn.ravel(), np.repeat(measures / model.conn.shape[1], model.conn.shape[1]))
    return w


def scaled_case(case: Case, k: float, n: int, tangent: float, adaptive=False) -> Case:
    c = copy.deepcopy(case)
    for b in c.bcs:
        if b.type in ("pressure", "force", "traction", "displacement") and b.value is not None:
            if isinstance(b.value, str) or (isinstance(b.value, list) and any(isinstance(v, str) for v in b.value)):
                raise ValueError(f"cannot scale expression-valued {b.type} BC")
            b.value = [k * v for v in b.value] if isinstance(b.value, list) else k * b.value
    for ld in c.loads:
        if ld.type == "body_force":
            ld.value = [k * v for v in ld.value]
        else:
            raise ValueError(f"cannot scale load {ld.type}")
    H = E * tangent / (E - tangent)                    # plastic modulus of the linear hardening
    for m in c.materials.values():
        m.E, m.nu = E, NU
        m.plasticity = {"model": "j2_isotropic", "yield_stress": SY, "hardening": [[0.0, SY], [1.0, SY + H]]}
    c.analysis.options = {**c.analysis.options, "nonlinear": True, "increments": n, "load_factor_max": 1.0,
                          "adaptive": adaptive}
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir"), ap.add_argument("out_dir")
    ap.add_argument("--ratio", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tangent", type=float, default=1.5e9)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    sd, out = Path(a.set_dir), Path(a.out_dir)
    case = Case.load(sd / "case.json")
    if not case.mesh_path().exists():
        case.mesh, case.base_dir = str((sd / "mesh.msh").resolve()), None
    for m in case.materials.values():
        m.E, m.nu = E, NU
    t0 = time.time()
    lin = solve(case, backend="ccx", threads=a.threads)
    X, vm_lin = lin.nodal("von_mises")
    sing, excl = singular_mask(X, vm_lin)
    peak_ref = float(vm_lin[~excl].max())
    k = a.ratio * SY / peak_ref
    t_lin = time.time() - t0
    nlcase = scaled_case(case, k, a.n, a.tangent)
    wd = a.workdir and Path(a.workdir)
    t0 = time.time()
    r = solve(nlcase, backend="ccx", threads=a.threads, workdir=wd)
    mode = "fixed"
    if not r.converged:
        first_try = {"lambda_reached": r.increments[-1].lam, "errors": r.errors[:3]}
        r = solve(scaled_case(case, k, a.n, a.tangent, adaptive=True), backend="ccx", threads=a.threads,
                  workdir=wd and wd.with_name(wd.name + "_adaptive"))
        mode = "adaptive (fixed increments diverged)"
    else:
        first_try = None
    t_nl = time.time() - t0
    vm_design = k * vm_lin
    vol = nodal_volumes(r.model, r.cell_measures())
    Vt = vol.sum()

    def fields(res, inc):
        lin_now = inc.lam * vm_design
        vm = res.nodal("von_mises")[1]
        return {"von_mises_linear": lin_now, "vm_ratio": vm / np.maximum(lin_now, 1e-30 * SY)}

    def redistribution(inc):
        r.use_increment(inc.index - 1)
        vm = r.nodal("von_mises")[1]
        lin_now = inc.lam * vm_design
        elastic = (inc.PEEQ <= 1e-9) & (lin_now > 0.2 * SY) & ~excl
        shed = elastic & (vm > 1.05 * lin_now)
        return {"lambda": inc.lam,
                "peak_vm_offsingular": float(vm[~excl].max()), "peak_vm_offsingular_linear": float(lin_now[~excl].max()),
                "peak_vm_all": float(vm.max()), "peak_vm_all_linear": float(lin_now.max()),
                "vol_frac_vm_ge_0.9sy": float(vol[vm >= 0.9 * SY].sum() / Vt),
                "vol_frac_vm_ge_0.9sy_linear": float(vol[lin_now >= 0.9 * SY].sum() / Vt),
                "vol_frac_plastic_nodal": float(vol[inc.PEEQ > 1e-9].sum() / Vt),
                "vol_frac_shed": float(vol[shed].sum() / Vt),
                "max_shed_ratio": float((vm[elastic] / lin_now[elastic]).max()) if elastic.any() else None}

    red = [redistribution(inc) for inc in r.increments]
    extra = {"set": sd.name, "mode": mode, "first_try": first_try, "load_scale_k": k, "ratio": a.ratio,
             "linear": {"peak_vm_offsingular_orig": peak_ref, "max_vm_orig": float(vm_lin.max()),
                        "p99_5_orig": float(np.percentile(vm_lin, 99.5)), "n_singular_nodes": int(sing.sum()),
                        "lambda_first_yield_linear_all": float(SY / (k * vm_lin.max())),
                        "lambda_first_yield_linear_offsingular": 1.0 / a.ratio},
             "material": {"E": E, "nu": NU, "sy": SY, "tangent": a.tangent},
             "wall_s": {"linear": t_lin, "nonlinear": t_nl}, "redistribution": red}
    s = r.write_series(out, extra=extra, point_data=fields)
    print(json.dumps({k2: s[k2] for k2 in ("set", "mode", "converged", "lambda_reached",
                                           "first_plastic_increment_lambda", "load_scale_k")}, default=float))


if __name__ == "__main__":
    main()
