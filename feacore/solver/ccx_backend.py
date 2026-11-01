"""CalculiX (ccx) backend: an independent second solver for cross-checking the fixed solver.

``solve(case, **options) -> CcxResult``. The case is translated to a CalculiX input deck
(``*NODE`` / ``*ELEMENT`` / ``*BOUNDARY`` / ``*DLOAD`` / ``*CLOAD`` / ...), ccx runs in a scratch
directory, and the ASCII ``.frd`` (U, RF, NT, nodal-averaged S / HFL) plus the ``.dat`` totals
(reaction forces, internal energy) are parsed back. See ``docs/solver_ccx.md`` for the mapping,
conventions and verification.

Nonlinear (elastoplastic, load-stepped) cases (``analysis.options.nonlinear``) are run by
``feacore.solver._ccx_nl`` and return a ``CcxNonlinearResult`` (docs/solver_ccx_nonlinear.md).

Options (all optional):
  workdir   directory for the ccx run (default: a fresh dir under $FEACORE_CCX_TMP or /tmp/ccx)
  keep      keep the scratch directory (default False; always kept when ccx fails)
  ccx       path of the ccx executable (default $FEACORE_CCX or ``ccx`` on PATH)
  threads   OpenMP threads for ccx (default $OMP_NUM_THREADS or 1)
  timeout   seconds before ccx is killed (default None)
"""

from __future__ import annotations

import math
import shutil
import time

import numpy as np

from feacore.case import (BC, Case, CaseError, ELASTICITY_FIELDS, HEAT_FIELDS, QoI, as_field,
                           hardening_table)
from feacore.solver._ccx_fem import Locator, _bary, interpolate, map_points, quadrature, shape
from feacore.solver._ccx_io import CcxError, fnum, id_lines, parse_dat_totals, parse_frd, run_ccx
from feacore.solver._ccx_mesh import FEModel, MeshError, build_model, read_gmsh  # noqa: F401
from feacore.solver.base import Result, SolveInfo

__all__ = ["solve", "CcxResult", "CcxError", "ELEMENT_TYPES", "AXI_CLOAD_FACTOR", "AXI_OUTPUT_FACTOR"]

# CalculiX element type per (formulation, order)
ELEMENT_TYPES = {
    ("plane_stress", 1): "CPS3", ("plane_stress", 2): "CPS6",
    ("plane_strain", 1): "CPE3", ("plane_strain", 2): "CPE6",
    ("axisymmetric", 1): "CAX3", ("axisymmetric", 2): "CAX6",
    ("3d", 1): "C3D4", ("3d", 2): "C3D10",
    ("2d", 1): "CPS3", ("2d", 2): "CPS6",            # heat, plane (thickness = analysis.thickness)
}

# CalculiX axisymmetric (CAX) convention, determined empirically against closed-form solutions
# (tests/test_solver_ccx.py::test_axisymmetric_units): ccx expands CAX elements into a 2-degree
# wedge. Concentrated loads (*CLOAD / *CFLUX) are input as full-360-degree totals, but the
# nodal reactions and totals it reports (RF, RFL, ELSE) belong to the 2-degree wedge, i.e.
# full-360 values / 180. The schema uses full-360 totals everywhere.
AXI_CLOAD_FACTOR = 1.0          # schema (full 360) -> *CLOAD / *CFLUX value
AXI_OUTPUT_FACTOR = 180.0       # ccx RF / RFL / ELSE (2-degree wedge) -> schema (full 360)

_VOIGT = ("sxx", "syy", "szz", "sxy", "syz", "sxz")

# Plane stress: internal CalculiX section thickness = PLANE_STRESS_RATIO x the shortest cell edge
# (override with analysis.options["ccx_plane_stress_ratio"]); see _Deck.__init__.
PLANE_STRESS_RATIO = 1e-3


def solve(case: Case, **options) -> "CcxResult":
    t0 = time.perf_counter()
    case.check()
    a = case.analysis
    if a.formulation == "axisymmetric" and a.order == 1:
        raise NotImplementedError(
            "ccx backend: axisymmetric order 1 (CAX3) is refused: CalculiX's 2-degree wedge "
            "expansion of CAX3 is ~2e-4 too stiff and its face loads / films are distributed "
            "inconsistently, so it fails the patch test (use analysis.order = 2)")
    gm = read_gmsh(case.mesh_path())
    model = build_model(gm, case.dim, a.order, axisymmetric=(a.formulation == "axisymmetric"))
    _check_curvature(model, a.formulation)
    deck = _Deck(case, model)
    inp = deck.build()
    if a.nonlinear:   # opt-in elastoplastic load stepping: docs/solver_ccx_nonlinear.md
        from feacore.solver._ccx_nl import solve_nonlinear
        return solve_nonlinear(case, model, deck, inp, options, t0)
    keep = bool(options.get("keep", False)) or options.get("workdir") is not None
    run = run_ccx(inp, job="job", workdir=options.get("workdir"), ccx=options.get("ccx"),
                  threads=options.get("threads"), timeout=options.get("timeout"))
    try:
        frd = parse_frd(run.frd)
        totals = parse_dat_totals(run.dat) if run.dat.exists() else {}
    except Exception as e:  # pragma: no cover - keep the directory for debugging
        raise CcxError(f"could not parse ccx results in {run.workdir}: {e}") from e
    res = CcxResult(case, model, deck, frd, totals, run)
    res.info.wall_time = time.perf_counter() - t0
    if not keep:
        shutil.rmtree(run.workdir, ignore_errors=True)
        res.workdir = None
    return res


# Quadratic cells whose interior edges are curved by more than this (midside offset / edge length)
# are refused for C3D10 and CAX6 (see _check_curvature).
CURVED_INTERIOR_TOL = 1e-3


def _check_curvature(model: FEModel, formulation: str) -> None:
    """CalculiX integrates C3D10 (4 points) and the CAX6 wedge with rules that are not exact on
    curved isoparametric cells: on meshes with strongly curved interior edges the patch test fails
    (6e-5 for C3D10, up to 4e-3 for CAX6), while CPS6 / CPE6 pass. gmsh keeps interior edges
    straight and curves only boundary edges (offset O(h^2 / R)), for which the error is a
    discretization-level difference that vanishes under refinement: accepted with a warning."""
    if model.order != 2 or not (model.dim == 3 or formulation == "axisymmetric"):
        return
    from feacore.solver._ccx_mesh import TET_EDGES, TRI_EDGES, _encode

    ed = TRI_EDGES if model.dim == 2 else TET_EDGES
    a = model.conn[:, ed[:, 0]].ravel()
    b = model.conn[:, ed[:, 1]].ravel()
    mid = model.conn[:, model.n_corners:].ravel()
    Xa, Xb, Xm = model.X[a], model.X[b], model.X[mid]
    rel = np.linalg.norm(Xm - 0.5 * (Xa + Xb), axis=1) / np.maximum(np.linalg.norm(Xb - Xa, axis=1), 1e-300)
    # boundary edges: edges of faces that belong to one cell only
    fl = model.faces_local[:, : model.dim]
    fc = np.sort(model.conn[:, fl].reshape(-1, model.dim), axis=1)
    uf, cnt = np.unique(fc, axis=0, return_counts=True)
    bf = uf[cnt == 1]
    pairs = [(0, 1)] if model.dim == 2 else [(0, 1), (1, 2), (0, 2)]
    be = np.unique(np.sort(np.vstack([bf[:, list(p)] for p in pairs]), axis=1), axis=0)
    base = model.n_nodes + 1
    on_b = np.isin(_encode(np.sort(np.c_[a, b], axis=1), base), _encode(be, base))
    worst_int = float(rel[~on_b].max()) if np.any(~on_b) else 0.0
    worst_b = float(rel[on_b].max()) if np.any(on_b) else 0.0
    kind = "C3D10" if model.dim == 3 else "CAX6"
    if worst_int > CURVED_INTERIOR_TOL:
        raise NotImplementedError(
            f"ccx backend: {kind} cells with curved interior edges (midside offset up to "
            f"{worst_int:.2g} x edge length) are refused: CalculiX's {kind} quadrature is not exact on "
            "curved isoparametric cells (patch test fails). Use a mesh with straight interior edges "
            "(gmsh default) or the skfem backend")
    if worst_b > 1e-6:
        model.warnings.append(
            f"{kind} cells with curved boundary edges (midside offset up to {worst_b:.2g} x edge length): "
            "CalculiX's quadrature is inexact on curved cells; expect discretization-level differences "
            "from skfem that vanish under refinement")


def axi_radial_factor(model: FEModel) -> np.ndarray:
    """Per-node factor c with (reported 2D radial displacement) = c * u_r for CAX elements.

    ccx expands CAX elements into a wedge spanning -1..+1 degree and reports, as the 2D radial
    displacement, the x-component of expanded nodes: u_r cos(1 deg) at nodes that only exist on
    the +-1 degree faces (CAX6 midside nodes, all CAX3 nodes) and the (1, 4, 1)/6 average over
    -1, 0, +1 degree, u_r (4 + 2 cos 1 deg) / 6, at CAX6 corner nodes. Measured to 2e-6 on exact
    linear fields (uniform thermal expansion)."""
    c1 = math.cos(math.radians(1.0))
    c = np.full(model.n_nodes, c1)
    if model.order == 2:
        c[np.unique(model.conn[:, :3])] = (4.0 + 2.0 * c1) / 6.0
    return c


# ============================================================================ deck
class _Deck:
    """Builds the CalculiX input deck and keeps everything needed to post-process."""

    def __init__(self, case: Case, model: FEModel):
        self.case, self.m = case, model
        a = case.analysis
        self.physics, self.form = a.physics, a.formulation
        self.heat = a.physics == "heat"
        self.axi = a.formulation == "axisymmetric"
        self.plane = a.formulation in ("plane_stress", "plane_strain", "2d")
        self.ncomp = 1 if self.heat else model.dim
        self.etype = ELEMENT_TYPES[(a.formulation, model.order)]
        self.nsets: dict[str, str] = {}       # group -> NSET name
        self.elsets: dict[str, str] = {}      # group -> ELSET name
        self.lines: list[str] = []
        self.fext = np.zeros((model.n_nodes, self.ncomp))     # consistent external nodal loads (schema units)
        self.dirichlet: dict[tuple[int, int], float] = {}    # (node, dof 0-based) -> value
        self.dirichlet_groups: list[str] = []
        self.thermal = False
        self.warnings: list[str] = []
        self.has_thermal_load = any(l.type == "temperature" for l in case.loads)
        # nonlinear (opt-in): every load / prescribed value is written x lf and ramped over the step
        self.nonlinear = a.nonlinear
        self.lf = float(a.options.get("load_factor_max", 1.0)) if self.nonlinear else 1.0
        # Plane stress: CalculiX expands CPS elements into a 3D slab of the section thickness whose
        # faces are free, so only a slab that is thin compared with the cells is in plane stress
        # (a thick slab carries transverse shear between cells and converges to a 3D plate
        # solution). Run ccx with a thin internal section t_int and scale the loads by t_int / t:
        # displacements, strains and stresses of plane stress do not depend on the thickness.
        self.t_section = case.analysis.thickness
        if self.form == "plane_stress":
            e = model.conn[:, [0, 1, 2]]
            lmin = float(np.min(np.linalg.norm(model.X[e] - model.X[np.roll(e, 1, axis=1)], axis=2)))
            ratio = float(case.analysis.options.get("ccx_plane_stress_ratio", PLANE_STRESS_RATIO))
            self.t_section = ratio * lmin

    # ----------------------------------------------------------- unit conversions
    @property
    def load_factor(self) -> float:
        """schema nodal load -> CalculiX *CLOAD / *CFLUX value."""
        if self.axi:
            return AXI_CLOAD_FACTOR
        return self.t_section / self.case.analysis.thickness

    @property
    def out_factor(self) -> float:
        """CalculiX RF / RFL / ELSE -> schema value (full 360 degrees for axisymmetric)."""
        if self.axi:
            return AXI_OUTPUT_FACTOR
        return self.case.analysis.thickness / self.t_section

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Integration weight of the physical body at points x (..., 3): thickness, 2 pi r or 1."""
        if self.axi:
            return 2.0 * math.pi * x[..., 0]
        if self.plane:
            return np.full(x.shape[:-1], self.case.analysis.thickness)
        return np.ones(x.shape[:-1])

    # ----------------------------------------------------------- names
    def nset(self, group: str) -> str:
        if group not in self.nsets:
            name = f"N{len(self.nsets) + 1}"
            self.nsets[group] = name
            nodes = self.m.nodes_of(group)
            self.lines += [f"*NSET, NSET={name}", id_lines(nodes + 1)]
        return self.nsets[group]

    def elset(self, group: str) -> str:
        if group in (None, "*"):
            return "EALL"
        if group not in self.elsets:
            name = f"E{len(self.elsets) + 1}"
            self.elsets[group] = name
            self.lines += [f"*ELSET, ELSET={name}", id_lines(self.m.elements_of(group) + 1)]
        return self.elsets[group]

    # ----------------------------------------------------------- quadrature on faces / cells
    def face_quad(self, elem: np.ndarray, face: np.ndarray, degree: int = 6):
        m = self.m
        sdim = m.dim - 1
        xi, w = quadrature(sdim, degree)
        N, dN = shape(sdim, m.order, xi)
        nodes = m.face_nodes(elem, face)                         # (nf, nfn)
        Xf = m.X[nodes]
        x, J = map_points(Xf, N, dN)                             # (nf, nq, 3), (nf, nq, 3, sdim)
        if sdim == 1:
            t = J[..., 0]
            n = np.stack([t[..., 1], -t[..., 0], np.zeros_like(t[..., 0])], axis=-1)
        else:
            n = np.cross(J[..., 0], J[..., 1])
        dA = np.linalg.norm(n, axis=-1)
        n = n / dA[..., None]
        # orient outward: away from the element centroid
        cen = m.X[m.conn[elem]].mean(axis=1)                     # (nf, 3)
        fc = Xf.mean(axis=1)
        sgn = np.sign(np.einsum("fi,fi->f", np.mean(n, axis=1), fc - cen))
        sgn[sgn == 0] = 1.0
        n = n * sgn[:, None, None]
        return nodes, N, x, w[None, :] * dA, n

    def cell_quad(self, elems: np.ndarray, degree: int = 6):
        m = self.m
        xi, w = quadrature(m.dim, degree)
        N, dN = shape(m.dim, m.order, xi)
        nodes = m.conn[elems]
        x, J = map_points(m.X[nodes][..., : m.dim], N, dN)
        detJ = np.abs(np.linalg.det(J))
        x3 = np.concatenate([x, np.zeros(x.shape[:-1] + (3 - m.dim,))], axis=-1) if m.dim < 3 else x
        return nodes, N, x3, w[None, :] * detJ

    def add_nodal(self, nodes, N, vals):
        """fext[nodes] += sum_q N_a vals_q  (vals (nf, nq, ncomp) already weighted)."""
        contrib = np.einsum("qa,fqc->fac", N, vals)
        np.add.at(self.fext, nodes.ravel(), contrib.reshape(-1, self.ncomp))

    # ----------------------------------------------------------- BC helpers
    def bc_nodes(self, bc: BC) -> np.ndarray:
        if bc.point is not None:
            p = np.zeros(3)
            p[: len(bc.point)] = bc.point
            d = np.linalg.norm(self.m.X - p, axis=1)
            i = int(np.argmin(d))
            h = (self.m.measures().mean()) ** (1.0 / self.m.dim)
            if d[i] > 0.5 * h:
                self.warnings.append(f"{bc.type} BC at point {bc.point}: nearest node is {d[i]:.3g} away "
                                     f"(cell size ~{h:.3g})")
            return np.array([i])
        return self.m.nodes_of(bc.group)

    def set_dirichlet(self, nodes, dof: int, values):
        values = np.broadcast_to(np.asarray(values, dtype=float), (len(nodes),))
        for n, v in zip(nodes.tolist(), values.tolist()):
            self.dirichlet[(n, dof)] = v

    # ----------------------------------------------------------- build
    def build(self) -> str:
        c, m = self.case, self.m
        L = self.lines
        L += ["*HEADING", f"feacore ccx backend: {self.physics} / {self.form} / {self.etype}"]
        L.append("*NODE, NSET=NALL")
        L += [f"{i + 1}, {fnum(x)}, {fnum(y)}, {fnum(z)}" for i, (x, y, z) in enumerate(m.X)]
        L.append(f"*ELEMENT, TYPE={self.etype}, ELSET=EALL")
        L += [f"{i + 1}, " + ", ".join(str(int(n) + 1) for n in row) for i, row in enumerate(m.conn)]
        self._check_groups()
        self._materials()
        step: list[str] = []
        if self.heat:
            step += self._heat_step()
        else:
            step += self._elastic_step()
        L += step
        return "\n".join(L) + "\n"

    def _check_groups(self):
        c, m = self.case, self.m
        refs = [(f"bcs ({b.type})", b.group) for b in c.bcs if b.group is not None]
        refs += [(f"loads ({l.type})", l.group) for l in c.loads]
        refs += [(f"regions", r.group) for r in c.regions]
        refs += [(f"qois ({q.name})", q.group) for q in c.qois if q.group is not None]
        missing = [(w, g) for w, g in refs if not m.has_group(g)]
        if missing:
            raise CaseError("groups not in the mesh: " + ", ".join(f"{g!r} ({w})" for w, g in missing)
                            + f"; mesh groups: {m.group_names()}")

    def _region_materials(self) -> np.ndarray:
        """Material key per element: first matching region in list order (as Case.material_for)."""
        c, m = self.case, self.m
        names = list(c.materials)
        mat = np.full(m.n_elems, -1, dtype=np.int64)
        if not c.regions:
            mat[:] = 0
        for r in c.regions:
            el = m.elements_of(r.group)
            sel = el[mat[el] < 0]
            mat[sel] = names.index(r.material)
        if np.any(mat < 0):
            raise CaseError(f"{int(np.sum(mat < 0))} cells are not covered by any region")
        return mat

    def _materials(self):
        c, m, L = self.case, self.m, self.lines
        names = list(c.materials)
        mat = self._region_materials()
        self.elem_mat = mat
        T_ref = c.analysis.T_ref
        for k, name in enumerate(names):
            el = np.nonzero(mat == k)[0]
            if len(el) == 0:
                continue
            mt = c.materials[name]
            es = f"M{k + 1}"
            L += [f"*ELSET, ELSET={es}", id_lines(el + 1)]
            L += [f"*MATERIAL, NAME=MAT{k + 1}"]
            if self.heat:
                L += ["*CONDUCTIVITY", fnum(mt.k)]
            else:
                L += ["*ELASTIC", f"{fnum(mt.E)}, {fnum(mt.nu)}"]
                if mt.alpha and self.has_thermal_load:
                    L += [f"*EXPANSION, ZERO={fnum(T_ref)}", fnum(mt.alpha)]
                if mt.plasticity is not None:
                    L += ["*PLASTIC"] + [f"{fnum(sv)}, {fnum(ep)}" for ep, sv in hardening_table(mt.plasticity)]
            L.append(f"*SOLID SECTION, ELSET={es}, MATERIAL=MAT{k + 1}")
            if self.plane:
                L.append(fnum(self.t_section))

    # ----------------------------------------------------------- elasticity
    def _elastic_step(self) -> list[str]:
        c, m = self.case, self.m
        dim = m.dim
        pre: list[str] = []          # model data before *STEP (initial conditions)
        S: list[str] = self._static_header()
        lf = self.lf
        dloads: list[str] = []
        X = m.X
        for b in c.bcs:
            if b.type in ("fixed", "displacement", "symmetry"):
                nodes = self.bc_nodes(b)
                if b.group is not None:
                    self.dirichlet_groups.append(b.group)
                if b.type == "fixed":
                    for d in range(dim):
                        self.set_dirichlet(nodes, d, 0.0)
                elif b.type == "displacement":
                    vals = b.value if isinstance(b.value, list) else [b.value] * len(b.components)
                    for d, v in zip(b.components, vals):
                        self.set_dirichlet(nodes, d, as_field(v, X[nodes, 0], X[nodes, 1], X[nodes, 2]))
                else:
                    d = self._symmetry_axis(b.group, nodes)
                    self.set_dirichlet(nodes, d, 0.0)
            elif b.type == "pressure":
                el, fc = m.facets_of(b.group)
                nodes, N, x, w, n = self.face_quad(el, fc)
                p = as_field(b.value, x[..., 0], x[..., 1], x[..., 2])
                trac = -p[..., None] * n[..., :dim]
                self.add_nodal(nodes, N, trac * (w * self.weight(x))[..., None])
                if isinstance(b.value, str) or self.etype == "CAX3":
                    # expressions, and CAX3 (ccx distributes *DLOAD on CAX3 faces inconsistently:
                    # right total, wrong nodal split) -> consistent nodal loads
                    self._cload_from(nodes, N, trac * (w * self.weight(x))[..., None])
                else:
                    dloads += [f"{e + 1}, P{f + 1}, {fnum(lf * b.value)}" for e, f in zip(el, fc)]
            elif b.type in ("traction", "force"):
                el, fc = m.facets_of(b.group)
                nodes, N, x, w, n = self.face_quad(el, fc)
                ww = w * self.weight(x)
                if b.type == "traction":
                    trac = np.stack([as_field(v, x[..., 0], x[..., 1], x[..., 2]) for v in b.value], axis=-1)
                else:
                    area = float(ww.sum())
                    trac = np.broadcast_to(np.asarray(b.value, float) / area, x.shape[:-1] + (dim,))
                vals = trac * ww[..., None]
                self.add_nodal(nodes, N, vals)
                self._cload_from(nodes, N, vals)
        for ld in c.loads:
            if ld.type == "body_force":
                el = m.elements_of(ld.group)
                nodes, N, x, w = self.cell_quad(el)
                ww = w * self.weight(x)
                bf = np.stack([as_field(v, x[..., 0], x[..., 1], x[..., 2]) for v in ld.value], axis=-1)
                self.add_nodal(nodes, N, bf * ww[..., None])
                if any(isinstance(v, str) for v in ld.value) or self.axi:
                    self._cload_from(nodes, N, bf * ww[..., None])
                else:
                    es = self.elset(ld.group)
                    dloads += [f"{es}, B{'XYZ'[d]}, {fnum(lf * v)}" for d, v in enumerate(ld.value) if v != 0]
            elif ld.type == "temperature":
                self.thermal = True
        if self.thermal:
            T = np.full(m.n_nodes, c.analysis.T_ref)
            for ld in c.loads:
                if ld.type == "temperature":
                    nodes = m.nodes_of(ld.group)
                    T[nodes] = as_field(ld.value, X[nodes, 0], X[nodes, 1], X[nodes, 2])
            self.T_nodal = T
            pre += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", f"NALL, {fnum(c.analysis.T_ref)}"]
            Tw = c.analysis.T_ref + lf * (T - c.analysis.T_ref)
            S += ["*TEMPERATURE"] + [f"{i + 1}, {fnum(t)}" for i, t in enumerate(Tw)]
        if not self.dirichlet:
            raise CaseError("no Dirichlet boundary conditions: the problem is singular")
        S.append("*BOUNDARY")
        S += [f"{n + 1}, {d + 1}, {d + 1}, {fnum(lf * v)}" for (n, d), v in sorted(self.dirichlet.items())]
        if dloads:
            S += ["*DLOAD"] + dloads
        cl = getattr(self, "_cloads", None)
        if cl is not None and np.any(cl):
            S.append("*CLOAD")
            for n in np.nonzero(np.any(cl != 0, axis=1))[0]:
                S += [f"{n + 1}, {d + 1}, {fnum(lf * cl[n, d] * self.load_factor)}" for d in range(dim)
                      if cl[n, d] != 0]
        S += ["*NODE FILE", "U, RF" + (", NT" if self.thermal else ""), "*EL FILE", "S"]
        if self.nonlinear:
            S[-1] = "S, PEEQ"
            S += ["*EL PRINT, ELSET=EALL", "PEEQ"] if self.case.analysis.options.get("ip_peeq", True) else []
        S += self._totals_requests("RF")
        S += ["*EL PRINT, ELSET=EALL, TOTALS=ONLY", "ELSE", "*END STEP"]
        return pre + S

    def _static_header(self) -> list[str]:
        """*STEP / *STATIC; nonlinear: fixed equal increments (DIRECT) over a unit step time, loads
        ramped linearly from 0 (ccx's default for *STATIC), so step time t = load factor / lf."""
        if not self.nonlinear:
            return ["*STEP", "*STATIC"]
        o = self.case.analysis.options
        n = int(o.get("increments", 20))
        step = "*STEP, INC=" + str(max(100, 20 * n)) + (", NLGEOM" if o.get("nlgeom") else "")
        dt = fnum(1.0 / n)
        if o.get("adaptive"):
            return [step, "*STATIC", f"{dt}, 1., {fnum(1e-4 / n)}, {dt}"]
        return [step, "*STATIC, DIRECT", f"{dt}, 1."]

    def _cload_from(self, nodes, N, vals):
        if getattr(self, "_cloads", None) is None:
            self._cloads = np.zeros((self.m.n_nodes, self.ncomp))
        contrib = np.einsum("qa,fqc->fac", N, vals)
        np.add.at(self._cloads, nodes.ravel(), contrib.reshape(-1, self.ncomp))

    def _totals_requests(self, what: str) -> list[str]:
        groups = list(dict.fromkeys(self.dirichlet_groups +
                                    [q.group for q in self.case.qois if q.type == "reaction" and q.group]))
        out = []
        for g in groups:
            out += [f"*NODE PRINT, NSET={self.nset(g)}, TOTALS=ONLY", what]
        return out

    def _symmetry_axis(self, group: str, nodes: np.ndarray) -> int:
        X = self.m.X[nodes, : self.m.dim]
        span = np.ptp(X, axis=0)
        size = max(float(np.ptp(self.m.X[:, : self.m.dim], axis=0).max()), 1e-300)
        flat = np.nonzero(span <= 1e-6 * size)[0]
        if len(flat) != 1:
            raise CaseError(f"symmetry BC on {group!r}: group is not a plane normal to a coordinate axis "
                            f"(coordinate spans {span})")
        return int(flat[0])

    # ----------------------------------------------------------- heat
    def _heat_step(self) -> list[str]:
        c, m = self.case, self.m
        X = m.X
        S: list[str] = ["*STEP", "*HEAT TRANSFER, STEADY STATE"]
        dfl: list[str] = []
        films: list[str] = []
        for b in c.bcs:
            if b.type == "temperature":
                nodes = self.bc_nodes(b)
                if b.group is not None:
                    self.dirichlet_groups.append(b.group)
                self.set_dirichlet(nodes, 0, as_field(b.value, X[nodes, 0], X[nodes, 1], X[nodes, 2]))
            elif b.type == "flux":
                el, fc = m.facets_of(b.group)
                nodes, N, x, w, n = self.face_quad(el, fc)
                q = as_field(b.value, x[..., 0], x[..., 1], x[..., 2])
                vals = (q * w * self.weight(x))[..., None]
                self.add_nodal(nodes, N, vals)
                if isinstance(b.value, str) or self.etype == "CAX3":   # see pressure above
                    self._cload_from(nodes, N, vals)
                else:
                    dfl += [f"{e + 1}, S{f + 1}, {fnum(b.value)}" for e, f in zip(el, fc)]
            elif b.type == "convection":
                el, fc = m.facets_of(b.group)
                films += [f"{e + 1}, F{f + 1}, {fnum(b.T_inf)}, {fnum(b.h)}" for e, f in zip(el, fc)]
        for ld in c.loads:
            if ld.type == "source":
                el = m.elements_of(ld.group)
                nodes, N, x, w = self.cell_quad(el)
                s = as_field(ld.value, x[..., 0], x[..., 1], x[..., 2])
                vals = (s * w * self.weight(x))[..., None]
                self.add_nodal(nodes, N, vals)
                if isinstance(ld.value, str) or self.axi:
                    self._cload_from(nodes, N, vals)
                else:
                    dfl.append(f"{self.elset(ld.group)}, BF, {fnum(ld.value)}")
        if self.dirichlet:
            S.append("*BOUNDARY")
            S += [f"{n + 1}, 11, 11, {fnum(v)}" for (n, d), v in sorted(self.dirichlet.items())]
        elif not films:
            raise CaseError("heat problem without temperature or convection BCs is singular")
        if dfl:
            S += ["*DFLUX"] + dfl
        if films:
            S += ["*FILM"] + films
        cl = getattr(self, "_cloads", None)
        if cl is not None and np.any(cl):
            S.append("*CFLUX")
            S += [f"{n + 1}, 11, {fnum(cl[n, 0] * self.load_factor)}" for n in np.nonzero(cl[:, 0])[0]]
        S += ["*NODE FILE", "NT, RFL", "*EL FILE", "HFL"]
        S += self._totals_requests("RFL")
        S += ["*END STEP"]
        return S


# ============================================================================ result
class CcxResult(Result):
    """CalculiX solution on the model nodes; probe/nodal/evaluate per ``feacore.case``."""

    def __init__(self, case: Case, model: FEModel, deck: _Deck, frd: dict, totals: dict, run):
        self.case, self.model, self.deck = case, model, deck
        self.totals = totals
        self.load_scale = 1.0     # applied loads = load_scale x deck.fext (nonlinear: increment's lambda)
        self.workdir = run.workdir
        self.ccx_stdout = run.stdout
        a = case.analysis
        nn = model.n_nodes
        self._fields: dict[str, np.ndarray] = {}
        self._locator: Locator | None = None

        def nodal_block(name, ncols):
            if name not in frd:
                return None
            b = frd[name]
            out = np.zeros((nn, ncols))
            idx = b.nodes - 1
            ok = (idx >= 0) & (idx < nn)
            out[idx[ok]] = b.values[ok][:, :ncols]
            return out

        if a.physics == "elasticity":
            U = nodal_block("DISP", 3)
            if U is None:
                raise CcxError("no DISP block in the .frd")
            Sg = nodal_block("STRESS", 6)
            if Sg is None:
                raise CcxError("no STRESS block in the .frd")
            if a.formulation == "axisymmetric":
                U[:, 0] /= axi_radial_factor(model)
            self.U = U
            self.RF = nodal_block("FORC", 3)
            self.S = self._map_stress(Sg)
        else:
            T = nodal_block("NDTEMP", 1)
            if T is None:
                raise CcxError("no NDTEMP block in the .frd")
            self.T = T[:, 0]
            self.Q = nodal_block("FLUX", 3)
            self.RFL = nodal_block("RFL", 1)
        n_cells = model.n_elems
        measure = float(deck.cell_quad(np.arange(model.n_elems), degree=2)[3].sum())
        warnings = list(model.warnings) + list(deck.warnings)
        warnings += [w for w in run.warnings if "OUTPUT" not in w][:10]
        ncomp = 1 if a.physics == "heat" else model.dim
        self.info = SolveInfo(
            backend="ccx", physics=a.physics, formulation=a.formulation, order=model.order,
            dim=model.dim, cell_type=model.cell_type, n_nodes=nn, n_cells=n_cells,
            n_dofs=nn * ncomp, measure=measure, h_rep=(measure / n_cells) ** (1.0 / model.dim),
            wall_time=run.wall_time, warnings=warnings)

    # ------------------------------------------------------------------ fields
    def _map_stress(self, Sg: np.ndarray) -> np.ndarray:
        """ccx SXX SYY SZZ SXY SYZ SZX -> schema (sxx, syy, szz, sxy, syz, sxz) + stt column."""
        f = self.case.analysis.formulation
        out = np.zeros((len(Sg), 7))
        out[:, 0], out[:, 1], out[:, 3] = Sg[:, 0], Sg[:, 1], Sg[:, 3]
        if f == "3d":
            out[:, 2], out[:, 4], out[:, 5] = Sg[:, 2], Sg[:, 4], Sg[:, 5]
        elif f == "plane_strain":
            out[:, 2] = Sg[:, 2]
        elif f == "axisymmetric":
            out[:, 6] = Sg[:, 2]                   # hoop stress (ccx SZZ for CAX elements)
            out[:, 2] = Sg[:, 2]                   # out-of-plane component = hoop (as skfem backend)
        return out

    def _tensor(self, s: np.ndarray) -> np.ndarray:
        """(n, 7) schema components -> full 3x3 tensors (axisymmetric: hoop in the zz slot)."""
        T = np.zeros((len(s), 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
        T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
        T[:, 0, 2] = T[:, 2, 0] = s[:, 5]
        if self.case.analysis.formulation == "axisymmetric":
            T[:, 2, 2] = s[:, 6]
        return T

    def _derive(self, field: str, base: np.ndarray) -> np.ndarray:
        """Field from interpolated base values: U (n, 3) / S (n, 7) / T (n,) / Q (n, 3)."""
        phys = self.case.analysis.physics
        if phys == "elasticity":
            if field not in ELASTICITY_FIELDS:
                raise CaseError(f"unknown elasticity field {field!r}")
            if field in ("ux", "uy", "uz"):
                return base["U"][:, "xyz".index(field[1])]
            if field == "u_mag":
                return np.linalg.norm(base["U"], axis=1)
            s = base["S"]
            if field in _VOIGT:
                return s[:, _VOIGT.index(field)]
            if field == "stt":
                return s[:, 6]
            T = self._tensor(s)
            if field == "von_mises":
                d = T - np.trace(T, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
                return np.sqrt(1.5 * np.einsum("nij,nij->n", d, d))
            ev = np.linalg.eigvalsh(T)[:, ::-1]                # s1 >= s2 >= s3
            if field == "tresca":
                return ev[:, 0] - ev[:, 2]
            return ev[:, {"s1": 0, "s2": 1, "s3": 2}[field]]
        if field not in HEAT_FIELDS:
            raise CaseError(f"unknown heat field {field!r}")
        if field == "T":
            return base["T"]
        if base.get("Q") is None:
            raise CcxError("no heat flux (FLUX) block in the .frd")
        q = base["Q"]
        if field == "q_mag":
            return np.linalg.norm(q[:, : self.model.dim], axis=1)
        return q[:, "xyz".index(field[1])]

    def _nodal_base(self) -> dict:
        if self.case.analysis.physics == "elasticity":
            return {"U": self.U, "S": self.S}
        return {"T": self.T, "Q": self.Q}

    def nodal(self, field: str):
        key = field
        if key not in self._fields:
            self._fields[key] = self._derive(field, self._nodal_base())
        return self.model.X[:, : self.model.dim].copy(), self._fields[key].copy()

    def probe(self, field: str, points) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        m = self.model
        if self._locator is None:
            self._locator = Locator(m.X, m.conn, m.dim, m.order)
        elem, xi, _ = self._locator.locate(pts[:, : m.dim])
        base = {k: (None if v is None else interpolate(m.conn, m.dim, m.order, v, elem, xi))
                for k, v in self._nodal_base().items()}
        return self._derive(field, base)

    # ------------------------------------------------------------------ QoIs
    def _group_nodes(self, group: str | None) -> np.ndarray:
        return self.model.nodes_of(group) if group not in (None, "*") else np.arange(self.model.n_nodes)

    def reaction(self, group: str, component: int = 0) -> float:
        """Total support reaction on ``group``: sum over its nodes of (RF - applied nodal load).
        CalculiX RF is the internal nodal force K u, so applied loads that act on constrained nodes
        are subtracted (they go straight into the support)."""
        d = self.deck
        heat = self.case.analysis.physics == "heat"
        comp = 0 if heat else int(component)
        nodes = self._group_nodes(group)
        fext = float(d.fext[nodes, comp].sum()) * self.load_scale
        key = ("heat generation" if heat else "force", d.nsets.get(group, "").upper())
        tot = self.totals.get(key)
        if tot is not None and len(tot) > comp:
            rf = float(tot[comp]) * d.out_factor
        else:
            arr = self.RFL if heat else self.RF
            if arr is None:
                raise CcxError("no reaction output available")
            rf = float(arr[nodes, comp].sum()) * d.out_factor
        return rf - fext

    def strain_energy(self) -> float:
        """0.5 u.K.u. Without thermal strain this is CalculiX's total internal energy (ELSE); with
        thermal loads ccx reports the energy of the mechanical strain only, so 0.5 u.K.u is
        integrated here from the CalculiX displacements (same quadrature as the stiffness)."""
        if self.deck.thermal:
            return self._energy_from_u()
        tot = self.totals.get(("internal energy", "EALL"))
        if tot is None:
            raise CcxError("no internal energy total in the .dat")
        return float(tot[0]) * self.deck.out_factor

    def _energy_from_u(self) -> float:
        m, d, c = self.model, self.deck, self.case
        dim = m.dim
        xi, w = quadrature(dim, 2 * m.order)
        N, dN = shape(dim, m.order, xi)                       # (nq, nen), (nq, nen, dim)
        Xe = m.X[m.conn][..., :dim]                           # (ne, nen, dim)
        J = np.einsum("qas,eai->eqis", dN, Xe)
        detJ = np.linalg.det(J)
        G = np.einsum("qas,eqsi->eqai", dN, np.linalg.inv(J))  # grad N (ne, nq, nen, dim)
        Ue = self.U[m.conn][..., :dim]                        # (ne, nen, dim)
        gu = np.einsum("eai,eqaj->eqij", Ue, G)               # du_i/dx_j
        eps = 0.5 * (gu + np.swapaxes(gu, -1, -2))
        x = np.einsum("qa,eai->eqi", N, Xe)
        names = list(c.materials)
        E = np.array([c.materials[n].E for n in names])[d.elem_mat][:, None]
        nu = np.array([c.materials[n].nu for n in names])[d.elem_mat][:, None]
        lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        tr = np.trace(eps, axis1=-2, axis2=-1)
        ee = np.einsum("eqij,eqij->eq", eps, eps)
        if self.case.analysis.formulation == "plane_stress":
            lam = 2 * lam * mu / (lam + 2 * mu)
        if self.case.analysis.formulation == "axisymmetric":
            ett = np.einsum("qa,ea->eq", N, Ue[..., 0]) / np.maximum(x[..., 0], 1e-300)
            tr, ee = tr + ett, ee + ett ** 2
        dens = 0.5 * (lam * tr ** 2 + 2 * mu * ee)
        x3 = np.concatenate([x, np.zeros(x.shape[:-1] + (3 - dim,))], axis=-1)
        return float(np.sum(dens * np.abs(detJ) * w[None, :] * d.weight(x3)))

    def integral(self, field: str, group: str | None, mean: bool = False) -> float:
        """Integral over the physical body (plane: x thickness, axisymmetric: x 2 pi r) of ``field``
        over a cell or facet group (None / "*" = whole domain); ``mean`` divides by the measure."""
        m, d = self.model, self.deck
        gdim = m.dim if group in (None, "*") else m.group_dim(group)
        if gdim == m.dim:
            el = m.elements_of(group)
            xi, w = quadrature(m.dim, 2 * m.order + 2)
            elq = np.repeat(el, len(w))
            xiq = np.tile(xi, (len(el), 1))
            _, _, x, ww = d.cell_quad(el, degree=2 * m.order + 2)
        elif gdim == m.dim - 1:
            el, fc = m.facets_of(group)
            nodes, N, x, ww, n = d.face_quad(el, fc, degree=2 * m.order + 2)
            elq, xiq = self._face_points_to_cells(el, fc, x)
        else:
            raise CaseError(f"integral over group {group!r} of dimension {gdim} is not defined")
        wq = (ww * d.weight(x)).ravel()
        base = {k: (None if v is None else interpolate(m.conn, m.dim, m.order, v, elq, xiq))
                for k, v in self._nodal_base().items()}
        vals = self._derive(field, base)
        total = float(np.dot(vals, wq))
        return total / float(wq.sum()) if mean else total

    def _face_points_to_cells(self, el, fc, x):
        """Reference coordinates (in the adjacent cell) of physical points on its faces."""
        m = self.model
        nq = x.shape[1]
        elq = np.repeat(el, nq)
        pts = x.reshape(-1, 3)[:, : m.dim]
        # straight-sided initial guess, then Newton on the isoparametric map of that cell
        lam = _bary(m.X[m.conn[elq, : m.dim + 1], : m.dim], pts)
        xi = lam[:, 1:]
        if m.order == 2:
            if self._locator is None:
                self._locator = Locator(m.X, m.conn, m.dim, m.order)
            xi = self._locator._newton(elq, pts, xi)
        return elq, xi

    def evaluate(self, qoi: QoI) -> float:
        t = qoi.type
        if t == "point":
            return float(self.probe(qoi.field, [qoi.point])[0])
        if t in ("max", "min", "max_abs"):
            _, v = self.nodal(qoi.field)
            v = v[self._group_nodes(qoi.group)]
            return float(v.max() if t == "max" else v.min() if t == "min" else np.abs(v).max())
        if t == "reaction":
            return self.reaction(qoi.group, qoi.component or 0)
        if t in ("integral", "mean"):
            return self.integral(qoi.field, qoi.group, mean=(t == "mean"))
        if t == "strain_energy":
            return self.strain_energy()
        raise CaseError(f"unknown QoI type {t!r}")
