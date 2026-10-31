"""Deterministic detectors. Each takes a ``SkfemResult`` (setup-only for the prechecks) and a
``Report`` and appends flags; ids match docs/failure-modes.md."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from feacore.diagnostics.flags import THRESHOLDS as T, Flag, Report
from feacore.diagnostics.zz import zz_estimate

_DIRICHLET = ("fixed", "displacement", "symmetry")
_LOADS = ("pressure", "traction", "force")
_MODE_NAMES = {2: ("tx", "ty", "rz"), 3: ("tx", "ty", "tz", "rx", "ry", "rz")}


# ======================================================================= setup (pre-solve)


def _rigid_modes(X: np.ndarray, comp: np.ndarray, dim: int) -> np.ndarray:
    """(n_dofs, n_modes) rigid-body modes evaluated at the dofs (coordinates centred, scaled)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc = Xc / max(float(np.abs(Xc).max()), 1e-300)
    x, y = Xc[0], Xc[1]
    z = Xc[2] if dim == 3 else 0 * x
    if dim == 2:
        rot = [np.stack([-y, x])]
    else:
        rot = [np.stack([0 * x, -z, y]), np.stack([z, 0 * x, -x]), np.stack([-y, x, 0 * x])]
    cols = [(comp == i).astype(float) for i in range(dim)]   # translations
    for r in rot:
        cols.append(r[comp, np.arange(len(comp))])
    return np.column_stack(cols)


def _cell_components(mesh) -> tuple[int, np.ndarray]:
    """Connected components of the mesh (cells sharing a vertex are connected)."""
    t = mesh.t
    nv = mesh.p.shape[1]
    nel = t.shape[1]
    rows = np.repeat(np.arange(nel), t.shape[0])
    A = coo_matrix((np.ones(rows.size), (rows, t.T.ravel())), shape=(nel, nv)).tocsr()
    n, lab = connected_components(A @ A.T, directed=False)
    return n, lab


def check_rigid_body(res, rep: Report) -> None:
    """S1: every connected body must have all rigid-body modes removed by its Dirichlet dofs."""
    dim = res.dim
    comp = res._component_of()
    n, lab = _cell_components(res.md.mesh)
    if n > 1:
        rep.add(Flag("disconnected_bodies", "setup", "info",
                     f"mesh has {n} disconnected bodies; each needs its own supports", value=n))
    constrained = np.zeros(res.basis.N, bool)
    constrained[res.D] = True
    for body in range(n):
        dofs = np.unique(res.basis.element_dofs[:, lab == body])
        R = _rigid_modes(res.basis.doflocs[:, dofs], comp[dofs], dim)
        Rc = R[constrained[dofs]]
        names = _MODE_NAMES[dim]
        if Rc.shape[0] == 0:
            free = list(names)
        else:
            _, s, Vt = np.linalg.svd(Rc, full_matrices=True)
            s = np.concatenate([s, np.zeros(len(names) - len(s))])
            null = Vt[s < 1e-8 * max(s.max(), 1e-300)]
            free = sorted({names[int(np.argmax(np.abs(v)))] for v in null})
        if free:
            where = res.basis.doflocs[:, dofs].mean(axis=1).tolist()
            rep.add(Flag("rigid_body_mode", "setup", "fail",
                         f"unconstrained rigid-body motion ({', '.join(free)})"
                         + (f" of body {body}" if n > 1 else "")
                         + ": the stiffness matrix is singular; add supports that stop these motions",
                         value=len(free), where=where))


def check_load(res, rep: Report) -> None:
    """S2: something must load the part (forces, body forces or prescribed displacements)."""
    if not np.any(res.f) and not np.any(res.xD):
        rep.add(Flag("no_load", "setup", "fail", "no applied load or prescribed displacement: "
                     "the solution is identically zero"))


def check_singular_bcs(res, rep: Report) -> None:
    """S3: constraints on points / edges (3D) give infinite stress at the constraint."""
    for b in res.case.bcs:
        if b.type not in _DIRICHLET:
            continue
        kind = "point" if b.point is not None else res.md.group(b.group).kind
        if kind in ("point", "vertices") or (kind == "edges" and res.dim == 3):
            rep.add(Flag("point_constraint", "setup", "warn",
                         f"{b.type} BC on a {kind[:-1] if kind.endswith('s') else kind} "
                         f"({b.group or b.point}): stress there is singular and grows without bound "
                         "under refinement; do not read peak stresses near it",
                         where=list(b.point) if b.point else None))


def check_element_order(res, rep: Report) -> None:
    """S4: linear simplices lock (volumetric for nu -> 0.5, shear in bending)."""
    if res.case.analysis.order != 1:
        return
    nu = float(res.cell_Enu[:, 1].max())
    if nu > T["locking_nu"]:
        rep.add(Flag("volumetric_locking", "setup", "warn",
                     f"linear elements with nu = {nu:.3g} lock volumetrically; use order 2",
                     value=nu, threshold=T["locking_nu"]))
    rep.add(Flag("linear_elements", "setup", "info",
                 "linear simplices are over-stiff in bending; displacements and stresses converge slowly "
                 "(check with verify.p_sensitivity)"))


def mesh_quality(mesh) -> np.ndarray:
    """Mean-ratio shape quality of the corner simplex (1 = equilateral, 0 = degenerate)."""
    P = mesh.p[:, mesh.t[: mesh.p.shape[0] + 1]]            # (dim, nv, nel)
    dim = P.shape[0]
    edges = [(i, j) for i in range(dim + 1) for j in range(i + 1, dim + 1)]
    l2 = sum(((P[:, i] - P[:, j]) ** 2).sum(axis=0) for i, j in edges)
    A = np.transpose(P[:, 1:] - P[:, :1], (2, 0, 1))
    vol = np.abs(np.linalg.det(A)) / (2 if dim == 2 else 6)
    if dim == 2:
        return 4 * np.sqrt(3) * vol / l2
    return 12 * (3 * vol) ** (2 / 3) / l2


def check_mesh(res, rep: Report) -> None:
    """M1/M2: inverted (folded curved) cells and poorly shaped cells."""
    q = mesh_quality(res.md.mesh)
    worst = int(np.argmin(q))
    at = res.md.mesh.p[:, res.md.mesh.t[:, worst]].mean(axis=1).tolist()
    rep.metrics["mesh_min_quality"] = float(q.min())
    if q.min() < T["quality_fail"]:
        rep.add(Flag("degenerate_element", "mesh", "fail", f"near-degenerate cell (quality {q.min():.2g})",
                     value=float(q.min()), threshold=T["quality_fail"], where=at))
    elif q.min() < T["quality_warn"]:
        rep.add(Flag("poor_element_quality", "mesh", "warn",
                     f"{int((q < T['quality_warn']).sum())} cells below quality {T['quality_warn']}",
                     value=float(q.min()), threshold=T["quality_warn"], where=at))
    det = np.asarray(res.basis.mapping.detDF(res.basis.X))
    folded = (det.min(axis=1) * det.max(axis=1)) <= T["jacobian_fail"]
    if folded.any():
        e = int(np.nonzero(folded)[0][0])
        rep.add(Flag("inverted_element", "mesh", "fail",
                     f"{int(folded.sum())} curved cells fold over (Jacobian changes sign)",
                     value=int(folded.sum()),
                     where=res.md.mesh.p[:, res.md.mesh.t[:, e]].mean(axis=1).tolist()))


PRECHECKS = (check_rigid_body, check_load, check_singular_bcs, check_element_order, check_mesh)


# ======================================================================= solution (post-solve)


def check_finite(res, rep: Report) -> bool:
    """N1"""
    ok = bool(np.all(np.isfinite(res.u)) and np.all(np.isfinite(res.S)))
    if not ok:
        rep.add(Flag("non_finite", "solve", "fail", "NaN / Inf in the solution"))
    return ok


def check_residual(res, rep: Report) -> None:
    """N2: the linear solve actually solved K u = f on the free dofs."""
    r = float(res.linear.residual)
    rep.metrics["linear_residual"] = r
    if not r <= T["residual_fail"]:
        rep.add(Flag("linear_residual", "solve", "fail",
                     f"relative residual {r:.2e}: ill-conditioned or not converged (iterative tolerance?)",
                     value=r, threshold=T["residual_fail"]))


def check_equilibrium(res, rep: Report) -> None:
    """N3: applied loads and support reactions balance (forces, per component)."""
    R = res.reaction_vector()
    worst = 0.0
    for idx in res.comp:
        app, rea = float(res.f[idx].sum()), float(R[idx][np.isin(idx, res.D)].sum())
        scale = float(np.abs(res.f[idx]).sum() + np.abs(R[idx]).sum())
        if scale > 0:
            worst = max(worst, abs(app + rea) / scale)
    rep.metrics["equilibrium_error"] = worst
    if worst > T["equilibrium_fail"]:
        rep.add(Flag("equilibrium", "solve", "fail",
                     f"loads and reactions do not balance (relative imbalance {worst:.2e})",
                     value=worst, threshold=T["equilibrium_fail"]))


def check_validity(res, rep: Report) -> None:
    """V1/V2: small-displacement, small-strain assumptions of linear static analysis."""
    _, um = res.nodal("u_mag")
    ratio = float(um.max() / res.md.bbox_diag)
    eps = res.eps_qp
    e_eq = np.sqrt(np.einsum("ij...,ij...->...", eps, eps))
    emax = float(e_eq.max())
    rep.metrics.update(max_disp=float(um.max()), disp_ratio=ratio, max_strain=emax)
    if ratio > T["disp_ratio_fail"]:
        rep.add(Flag("large_displacement", "validity", "fail",
                     f"max displacement is {ratio:.1%} of the part size: linear theory invalid "
                     "(near-mechanism, missing support, or unit mismatch)", value=ratio,
                     threshold=T["disp_ratio_fail"]))
    if emax > T["strain_fail"]:
        rep.add(Flag("large_strain", "validity", "fail",
                     f"max strain {emax:.2%}: unit mismatch (E vs loads) or invalid linear model",
                     value=emax, threshold=T["strain_fail"]))
    elif emax > T["strain_warn"]:
        rep.add(Flag("high_strain", "validity", "warn",
                     f"max strain {emax:.2%} exceeds typical elastic range", value=emax,
                     threshold=T["strain_warn"]))


def _singular_support(b, dim: int) -> bool:
    """Supports that create stress singularities at their edges: fully clamped ones. Symmetry and
    roller (single-component) conditions do not."""
    return b.type == "fixed" or (b.type == "displacement" and len(b.components or []) == dim)


def _facet_normals(mesh, facets: np.ndarray) -> np.ndarray:
    """Unit normals (n, dim) of facets from their corner vertices (orientation arbitrary)."""
    P = mesh.p[:, mesh.facets[:, facets]]                   # (dim, nv, n)
    if P.shape[0] == 2:
        t = P[:, 1] - P[:, 0]
        n = np.stack([t[1], -t[0]])
    else:
        n = np.cross((P[:, 1] - P[:, 0]).T, (P[:, 2] - P[:, 0]).T).T
    return (n / np.linalg.norm(n, axis=0)).T


def _group_edge_points(res, kinds: tuple[str, ...]) -> np.ndarray:
    """Points where a BC can create a stress singularity, (n, dim):

    * point BCs and Dirichlet BCs on vertices / edges,
    * the rim of a facet group: vertices it shares with other boundary facets, ignoring facets on
      symmetry / roller planes (the group continues across them in the full model). Clamped
      supports are singular along any rim; a load only where it stops part-way across a smooth
      surface (adjacent facet within 30 degrees of the group's facets)."""
    mesh = res.md.mesh
    bfac = mesh.boundary_facets()
    planes = [res.md.group(b.group).index for b in res.case.bcs
              if b.type in _DIRICHLET and b.group and not _singular_support(b, res.dim)
              and res.md.group(b.group).kind == "facets"]
    bfac = np.setdiff1d(bfac, np.concatenate(planes)) if planes else bfac
    out = []
    for b in res.case.bcs:
        if b.type not in kinds or (b.type in _DIRICHLET and not _singular_support(b, res.dim)):
            continue
        if b.point is not None:
            out.append(np.asarray(b.point[: res.dim], float)[None])
            continue
        g = res.md.group(b.group)
        if g.kind != "facets":
            verts = np.unique(mesh.edges[:, g.index]) if g.kind == "edges" else g.index
            out.append(mesh.p[:, verts].T)
            continue
        others = np.setdiff1d(bfac, g.index)
        rim = np.intersect1d(np.unique(mesh.facets[:, g.index]), np.unique(mesh.facets[:, others]))
        if rim.size and b.type in _LOADS:
            ng, no = _facet_normals(mesh, g.index), _facet_normals(mesh, others)
            keep = []
            for v in rim:
                a = ng[np.any(mesh.facets[:, g.index] == v, axis=0)]
                o = no[np.any(mesh.facets[:, others] == v, axis=0)]
                if np.abs(a @ o.T).max() > np.cos(np.radians(30)):
                    keep.append(v)
            rim = np.asarray(keep, dtype=int)
        if rim.size:
            out.append(mesh.p[:, rim].T)
    return np.vstack(out) if out else np.zeros((0, res.dim))


def check_discretization(res, rep: Report) -> dict:
    """D1: ZZ energy-norm error estimate (single mesh)."""
    zz = zz_estimate(res)
    rep.metrics.update(zz_eta_rel=zz["eta_rel"], zz_energy_error=zz["energy_error_est"],
                       zz_concentration_top1pct=zz["concentration_top1pct"])
    sev = "fail" if zz["eta_rel"] > T["zz_fail"] else "warn" if zz["eta_rel"] > T["zz_warn"] else None
    if sev:
        rep.add(Flag("under_resolved", "discretization", sev,
                     f"estimated energy-norm error {zz['eta_rel']:.1%} (ZZ); refine the mesh "
                     f"near {np.round(zz['worst_cell_at'], 4).tolist()}", value=zz["eta_rel"],
                     threshold=T["zz_warn"], where=zz["worst_cell_at"]))
    return zz


def check_peaks(res, rep: Report, zz: dict) -> None:
    """C1-C3: classify the von Mises peak: at a support / load edge (artificial singularity),
    unresolved geometric concentration, or resolved."""
    X, vm = res.nodal("von_mises")
    i = int(np.argmax(vm))
    peak, p99 = float(vm[i]), float(np.percentile(vm, 99))
    xp = X[i]
    cent = res.md.mesh.p[:, res.md.mesh.t].mean(axis=1).T
    near_cell = int(np.argmin(np.linalg.norm(cent - xp, axis=1)))
    h_loc = float((res.dx[near_cell].sum() / res.t) ** (1.0 / res.dim))
    rep.metrics.update(vm_max=peak, vm_p99=p99, vm_peak_at=xp.tolist(), vm_peak_ratio=peak / max(p99, 1e-300))

    def near(points):
        if not len(points):
            return False, np.inf
        d = float(np.min(np.linalg.norm(points - xp, axis=1)))
        return d <= T["peak_near_bc_h"] * h_loc, d

    at_support, _ = near(_group_edge_points(res, _DIRICHLET))
    at_load, _ = near(_group_edge_points(res, _LOADS))
    local = float(zz["local_rel"][near_cell])
    rep.metrics["peak_local_error"] = local
    if at_support:
        rep.add(Flag("peak_at_support", "singularity", "warn",
                     "von Mises peak sits at the edge of a support: likely a BC-induced singularity, "
                     "not a physical stress; report a stress away from the support", value=peak, where=xp.tolist()))
    elif at_load:
        rep.add(Flag("peak_at_load_edge", "singularity", "warn",
                     "von Mises peak sits at the edge of a loaded patch: likely a load-induced "
                     "singularity", value=peak, where=xp.tolist()))
    elif local > T["zz_local_hotspot"]:
        rep.add(Flag("unresolved_peak", "singularity", "warn",
                     f"peak stress cell has {local:.0%} local error: concentration (notch / re-entrant "
                     "corner) is under-resolved or singular; confirm with verify.refinement_study",
                     value=local, threshold=T["zz_local_hotspot"], where=xp.tolist()))


def check_overconstraint(res, rep: Report) -> None:
    """S5: supports fighting each other (reactions much larger than the applied load)."""
    R = res.reaction_vector()
    F = np.array([res.f[idx].sum() for idx in res.comp])
    groups = []
    for b in res.case.bcs:
        if b.type in _DIRICHLET and b.group is not None:
            d = res.group_dofs(b.group)
            groups.append(np.array([R[np.intersect1d(d, idx)].sum() for idx in res.comp]))
    if len(groups) < 2 or not np.any(F):
        return
    # per component: supports that share a load have same-sign reactions (ratio 1); supports
    # pushing against each other add opposite reactions on top of the load (ratio > 1)
    ratio = float(np.abs(np.array(groups)).sum() / np.abs(F).sum())
    rep.metrics["reaction_ratio"] = ratio
    if ratio > T["overconstraint_ratio"]:
        rep.add(Flag("over_constraint", "setup", "info",
                     f"support reactions sum to {ratio:.1f}x the applied load: supports react "
                     "against each other (over-constraint stiffens the model)", value=ratio,
                     threshold=T["overconstraint_ratio"]))


def check_design(res, rep: Report, design: dict | None) -> None:
    """F1/F2: physical failure of the design (not of the analysis), if limits are given."""
    if not design:
        return
    trusted = rep.metrics.get("vm_trusted", rep.metrics.get("vm_max"))
    if "yield" in design and trusted is not None:
        sf = design["yield"] / trusted if trusted > 0 else np.inf
        rep.metrics["safety_factor"] = float(sf)
        if sf < design.get("min_safety_factor", 1.0):
            rep.add(Flag("yield", "design", "warn",
                         f"von Mises {trusted:.4g} exceeds yield / SF (safety factor {sf:.2f})",
                         value=float(sf), threshold=design.get("min_safety_factor", 1.0)))
    if "max_disp" in design and rep.metrics.get("max_disp", 0) > design["max_disp"]:
        rep.add(Flag("deflection_limit", "design", "warn",
                     f"max displacement {rep.metrics['max_disp']:.4g} > limit {design['max_disp']:.4g}",
                     value=rep.metrics["max_disp"], threshold=design["max_disp"]))


def trusted_peak(res, rep: Report) -> None:
    """Max von Mises excluding nodes within peak_near_bc_h * h of support / load edges
    (BC-induced singularities would otherwise dominate any stress-based number)."""
    X, vm = res.nodal("von_mises")
    pts = np.vstack([_group_edge_points(res, _DIRICHLET), _group_edge_points(res, _LOADS)])
    keep = np.ones(len(vm), bool)
    if len(pts):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts).query(X)
        keep = d > T["peak_near_bc_h"] * res.info.h_rep
    rep.metrics["vm_trusted"] = float(vm[keep].max()) if keep.any() else float(vm.max())
