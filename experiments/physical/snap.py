"""Crush a part between platens until it snaps (CPU only).

Displacement control: bottom face fixed, top face moved down along AXIS in steps (bonded platens).
Material: Formlabs Alpha Fast Resin, default post-cure (TDS rev. 2026-09-28): E 2230 MPa, UTS 46 MPa,
elongation at break 34 %; nu 0.35 assumed (not on the TDS).
Model (simplified, labelled as such): small-strain secant plasticity with element erosion.
Per element: elastic until the von Mises stress reaches the UTS, then the secant modulus E = UTS / eps_eq
(flow at the UTS), and the element breaks (stiffness ~0, permanently) at eps_eq >= 34 %.
Stops when the platen force falls below 35 % of its peak after the peak (the part has snapped).

usage: snap.py MESH.msh AXIS(x|y|z) OUT.npz
"""
import sys
import time

import meshio
import numpy as np
import pypardiso
import scipy.sparse as sp
from skfem import Basis, BilinearForm, ElementTetP0, ElementTetP1, ElementVector, MeshTet, asm, condense
from skfem.helpers import ddot, eye, sym_grad, trace

import json as _json
import os as _os
from pathlib import Path as _Path
_MJ = _os.environ.get("MATERIAL_JSON") or str(_Path(__file__).resolve().parent.parent / "material" / "alpha_fast_dogbone1.json")
_M = _json.load(open(_MJ))
E0, NU = _M["E"] * 1e6, _M["nu"]
_TAB = np.array(_M["plastic_table_true_stress_vs_plastic_strain"]); _S, _P = _TAB[:, 0] * 1e6, _TAB[:, 1]
SY = float(_S[0]); PEEQ_F = float(_M["failure_peeq"])
_SIG = np.linspace(0.0, _S[-1], 400); _EPS = _SIG / E0 + np.interp(_SIG, _S, _P, left=0.0)   # total strain vs stress
EF = float(_S[-1] / E0 + PEEQ_F)                                    # equivalent strain at failure
E_MIN = 1e-6 * E0
AXES = {"x": 0, "y": 1, "z": 2}


def load(path):
    m = meshio.read(path)
    tets = np.vstack([c.data for c in m.cells if c.type == "tetra"])
    names = {v[0]: k for k, v in m.field_data.items() if v[1] == 2}
    groups = {}
    for block, phys in zip(m.cells, m.cell_data["gmsh:physical"]):
        if block.type == "triangle":
            for tag in np.unique(phys):
                groups.setdefault(names.get(int(tag), str(tag)), []).append(block.data[phys == tag])
    used = np.unique(tets)
    remap = -np.ones(m.points.shape[0], dtype=int); remap[used] = np.arange(used.size)
    mesh = MeshTet(m.points[used].T, remap[tets].T)
    return mesh, {k: remap[np.vstack(v)] for k, v in groups.items()}


@BilinearForm
def stiffness(u, v, w):
    lam = w.E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = w.E / (2 * (1 + NU))
    e = sym_grad(u)
    return ddot(2 * mu * e + lam * eye(trace(e), e.shape[0]), sym_grad(v))


def group_nodes(tris):
    return np.unique(tris)


def strains(basis, u):
    g = basis.interpolate(u).grad[:, :, :, 0]               # P1: constant per element
    e = 0.5 * (g + g.transpose(1, 0, 2))
    dev = e - np.trace(e)[None, None, :] * np.eye(3)[:, :, None] / 3
    eps_eq = np.sqrt(2.0 / 3.0 * np.einsum("ijk,ijk->k", dev, dev))
    return e, eps_eq


def secant(eps_eq, E_prev, failed):
    sig = np.where(eps_eq <= _EPS[-1], np.interp(eps_eq, _EPS, _SIG), _S[-1])      # dogbone hardening, then flat
    E = np.where(eps_eq > 1e-12, sig / np.maximum(eps_eq, 1e-12), E0)
    newly = eps_eq >= EF
    failed = failed | newly
    E = np.minimum(E, E_prev)                                # monotonic: no healing
    E[failed] = E_MIN
    return E, failed


class Assembler:
    """Element matrices assembled once at E = 1; K(E) = sum_e E_e K_e by scaling (stiffness is linear in E)."""

    def __init__(self, basis, b0):
        ne = basis.mesh.t.shape[1]
        coo = stiffness.coo_data(basis, E=b0.interpolate(np.ones(ne)))
        self.rows, self.cols, self.shape = coo.indices[0], coo.indices[1], coo.shape
        nl = coo.local_shape[0] * coo.local_shape[1]
        test = np.random.default_rng(1).uniform(0.5, 2.0, ne)
        ref = stiffness.assemble(basis, E=b0.interpolate(test))
        for layout in ("elem_last", "elem_first"):
            self.layout, self.D = layout, coo.data
            if abs(self.matrix(test) - ref).max() < 1e-9 * abs(ref).max():
                return
        raise RuntimeError("could not match skfem's COO element layout")

    def scale(self, E):
        ne = E.size
        return np.repeat(E, self.D.size // ne) if self.layout == "elem_first" else np.tile(E, self.D.size // ne)

    def matrix(self, E):
        return sp.coo_matrix((self.D * self.scale(E), (self.rows, self.cols)), shape=self.shape).tocsr()


def solve_step(asmb, E, fixed, top_dofs, delta):
    K = asmb.matrix(E)
    x = np.zeros(K.shape[0]); x[top_dofs] = -delta
    A, b, xI, I = condense(K, np.zeros(K.shape[0]), x=x, D=fixed)
    x[I] = pypardiso.spsolve(A.tocsr(), b)
    return K, x


def von_mises(e, E):
    lam, mu = E * NU / ((1 + NU) * (1 - 2 * NU)), E / (2 * (1 + NU))
    s = 2 * mu[None, None] * e + lam[None, None] * np.trace(e)[None, None] * np.eye(3)[:, :, None]
    dev = s - np.trace(s)[None, None] * np.eye(3)[:, :, None] / 3
    return np.sqrt(1.5 * np.einsum("ijk,ijk->k", dev, dev))


def node_graph(mesh):
    t = mesh.t
    return t  # tets (4 x ne); the graph is rebuilt from intact tets each check


def separated(t, mesh, failed, bot, top):
    """True when no chain of intact elements connects a bottom node to a top node."""
    import scipy.sparse.csgraph as cg
    keep = t[:, ~failed]
    rows = np.repeat(keep[0], 3); cols = np.concatenate([keep[1], keep[2], keep[3]])
    n = mesh.p.shape[1]
    g = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    _, lab = cg.connected_components(g, directed=False)
    return np.intersect1d(lab[bot], lab[top]).size == 0


def main(msh, axis_name, out):
    axis = AXES[axis_name]
    mesh, groups = load(msh)
    basis = Basis(mesh, ElementVector(ElementTetP1())); b0 = basis.with_element(ElementTetP0())
    asmb = Assembler(basis, b0)
    bot, top = group_nodes(groups["bottom"]), group_nodes(groups["top"])
    nd = basis.nodal_dofs
    fixed = np.unique(np.concatenate([nd[:, bot].ravel(), nd[:, top].ravel()]))
    top_ax = nd[axis, top]
    E, failed = np.full(mesh.t.shape[1], E0), np.zeros(mesh.t.shape[1], bool)
    _, x = solve_step(asmb, E, fixed, top_ax, 1e-4)
    e, _ = strains(basis, x)
    vm = von_mises(e, E)
    d_yield = 1e-4 * SY / np.percentile(vm, 99.9)            # platen travel at first yield (non-singular)
    height = np.ptp(mesh.p[axis])
    hist, peak, t0 = [], 0.0, time.time()
    adj = node_graph(mesh)
    for k in range(1, 801):
        delta = d_yield * 0.05 * k                                     # small steps all the way
        for it in range(6):
            K, x = solve_step(asmb, E, fixed, top_ax, delta)
            e, eps_eq = strains(basis, x)
            E_new, failed_new = secant(eps_eq, E, failed)
            changed = np.count_nonzero(np.abs(E_new - E) > 1e-3 * E0)
            E, failed = E_new, failed_new
            if changed < 1e-4 * E.size:
                break
        force = -float((K @ x)[top_ax].sum())                # compressive platen force (N)
        peak = max(peak, force)
        vm = von_mises(e, E)
        hist.append(dict(delta=delta, force=force, failed=failed.copy(), yielded=(E < E0) & ~failed,
                         vm=vm.astype(np.float32), u=x[nd].T.astype(np.float32)))
        print(f"step {k:2d} delta {delta*1e3:7.3f} mm  F {force:9.1f} N  yielded {hist[-1]['yielded'].sum():6d}"
              f"  failed {failed.sum():6d}  iters {it+1}  {time.time()-t0:6.0f} s", flush=True)
        if (failed.any() and separated(adj, mesh, failed, bot, top)) or delta > 0.3 * height:
            break
    np.savez_compressed(out, points=mesh.p.T, tets=mesh.t.T, axis=axis,
                        delta=[h["delta"] for h in hist], force=[h["force"] for h in hist],
                        failed=np.array([h["failed"] for h in hist]), yielded=np.array([h["yielded"] for h in hist]),
                        vm=np.array([h["vm"] for h in hist]), u=np.array([h["u"] for h in hist]))
    print(f"peak force {peak:.1f} N; separated={separated(adj, mesh, failed, bot, top)}; wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
