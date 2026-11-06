#!/usr/bin/env python
"""Mesh system `netgen`: Netgen (netgen-mesher, OCC geometry) meshes the STEP directly.

    build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear [--maxh H] [--grading G] [--curvsafety C]

Netgen generates the tet mesh and the second-order (tet10) curving (boundary edges onto the CAD
surface). The bottom/top platen faces are found from the OCC shape by geometry (a flat face whose
extent along AXIS is zero and which sits at the part's min / max along AXIS), never by index. The
mesh is then written through the gmsh API (named physical groups, MSH 4.1, and gmsh's own CalculiX
.inp writer, as the mesh harness does), assessed with the harness quality gates
(feagent.quality.assess_mesh) and rendered with feagent.render.render_quality.
Nonlinear purpose: tet10 with STRAIGHT edges (all mid-nodes at the corner midpoints), which CalculiX
accepts for plasticity. Run with the netgen-mesher package importable (see README).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

SYSTEM = "netgen"
THIN_FACTOR = 0.65  # Netgen overshoots a face maxh by ~25 % (the harness uses 0.8 for gmsh)
FACE_TOL = 1e-5  # m: flatness / position tolerance for the platen faces
# per (part stem, purpose): maxh [m]; chosen so linear ~165k dofs, nonlinear <= 40k nodes
DEFAULT_MAXH = {("abc_00000003", "linear"): 0.0016, ("abc_00000003", "nonlinear"): 0.0022,
                ("abc_00000065", "linear"): 0.0016, ("abc_00000065", "nonlinear"): 0.0030}
EXPECTED = {"abc_00000003": (1, 1), "abc_00000065": (1, 2)}
PAIRS = list(combinations(range(4), 2))  # tet edges as vertex-index pairs
GMSH_TET10_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]  # gmsh nodes 4..9
GMSH_TRI6_EDGES = [(0, 1), (1, 2), (0, 2)]  # gmsh nodes 3..5
TRI_PAIRS = [(0, 1), (0, 2), (1, 2)]


def platen_faces(shape, ax: int):
    """(bottom, top) lists of (face index, bbox) found by geometry from the OCC shape."""
    bb = shape.bounding_box
    lo, hi = bb[0][ax], bb[1][ax]
    bot, top = [], []
    for i, f in enumerate(shape.faces):
        b0, b1 = f.bounding_box
        if abs(b1[ax] - b0[ax]) > FACE_TOL:
            continue  # not a flat face normal to the axis
        c = 0.5 * (b0[ax] + b1[ax])
        if abs(c - lo) < FACE_TOL:
            bot.append((i, (np.array(tuple(b0), float), np.array(tuple(b1), float))))
        elif abs(c - hi) < FACE_TOL:
            top.append((i, (np.array(tuple(b0), float), np.array(tuple(b1), float))))
    return bot, top


def _assign_mids(corners_xyz: np.ndarray, mids_xyz: np.ndarray, pairs) -> np.ndarray:
    """For each element, which pair index each mid node (columns) lies on (nearest edge midpoint)."""
    pm = np.stack([0.5 * (corners_xyz[:, a] + corners_xyz[:, b]) for a, b in pairs], axis=1)  # (n,P,3)
    d = np.linalg.norm(mids_xyz[:, :, None, :] - pm[:, None, :, :], axis=3)  # (n,mid,pair)
    j = d.argmin(axis=2)
    assert (np.sort(j, axis=1) == np.arange(len(pairs))).all(), "ambiguous mid-node assignment"
    return j


def occ_to_harness(shape, hgeo) -> dict[int, int]:
    """OCC face index -> feagent (gmsh CAD) face tag, by matching the face bounding boxes."""
    out = {}
    for i, f in enumerate(shape.faces):
        b0, b1 = (np.array(tuple(x), float) for x in f.bounding_box)
        out[i] = min(hgeo.faces, key=lambda g: max(np.abs(np.array(g.bbox[0]) - b0).max(),
                                                    np.abs(np.array(g.bbox[1]) - b1).max())).tag
    return out


def netgen_mesh(step: str, ax: int, maxh: float, grading: float, curvsafety: float, hgeo,
                thin_layers: float, opt: int):
    from netgen.occ import OCCGeometry
    geo = OCCGeometry(step)
    bot, top = platen_faces(geo.shape, ax)
    thin = {int(t): float(h.value) for h in hgeo.hazards if h.kind == "thin_wall" and h.value
            for d, t in h.entities if d == 2}
    if thin_layers > 0:  # thin-wall faces: surface size = 0.8 * t / layers (the harness's rule)
        for i, tag in occ_to_harness(geo.shape, hgeo).items():
            if tag in thin:
                geo.SetFaceMeshsize(i, min(maxh, THIN_FACTOR * thin[tag] / thin_layers))
    kw = {"optsteps3d": opt} if opt else {}
    mesh = geo.GenerateMesh(maxh=maxh, grading=grading, curvaturesafety=curvsafety, **kw)
    mesh.SecondOrder()
    pts = np.array(mesh.Coordinates(), float)
    tets = mesh.Elements3D().NumPy()["nodes"].astype(np.int64) - 1
    e2 = mesh.Elements2D().NumPy()
    tris = e2["nodes"].astype(np.int64) - 1
    return pts, tets, tris, e2["index"].astype(np.int64), bot, top


_D = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)  # d(lambda_i)/d(xi,eta,zeta)
_REF = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
_REF10 = np.vstack([_REF] + [0.5 * (_REF[a] + _REF[b]) for a, b in GMSH_TET10_EDGES] + [_REF.mean(0)])


def min_jacobian_ratio(pts, tets10):
    """Per element: min over the 10 nodes and the centroid of detJ of the quadratic map / detJ of its
    straight-edged version (1 = straight; <= 0 tangled)."""
    x = pts[tets10]  # (n,10,3)
    c = pts[tets10[:, :4]]
    d0 = np.einsum("nj,nj->n", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))
    out = np.full(len(tets10), np.inf)
    for r in _REF10:
        lam = np.array([1 - r.sum(), r[0], r[1], r[2]])
        dN = np.zeros((10, 3))
        for i in range(4):
            dN[i] = (4 * lam[i] - 1) * _D[i]
        for m, (a, b) in enumerate(GMSH_TET10_EDGES):
            dN[4 + m] = 4 * (lam[a] * _D[b] + lam[b] * _D[a])
        jac = np.einsum("nak,ai->nik", x, dN)  # (n,3,3): columns d x_k / d xi_i ... det is orientation-safe
        out = np.minimum(out, np.linalg.det(jac) / d0)
    return out


def to_gmsh_tets(pts, tets, straight: bool, min_ratio: float = 0.2):
    """Reorder netgen tet10 to gmsh order (vertex pairs found geometrically), positive orientation."""
    c = pts[tets[:, :4]]
    j = _assign_mids(c, pts[tets[:, 4:]], PAIRS)
    nd = tets.copy()
    mid_of = np.empty((len(tets), 6), np.int64)  # pair index -> node id
    for m in range(6):
        mid_of[np.arange(len(tets)), j[:, m]] = tets[:, 4 + m]
    vol = np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))
    out = np.empty_like(nd)
    perm = np.tile(np.arange(4), (len(tets), 1))
    neg = vol < 0
    perm[neg, 1], perm[neg, 2] = 2, 1  # swap corners 1 and 2 of inverted tets
    v = np.take_along_axis(tets[:, :4], perm, axis=1)
    out[:, :4] = v
    for m, (a, b) in enumerate(GMSH_TET10_EDGES):
        pa, pb = perm[:, a], perm[:, b]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        k = _LUT[lo, hi]
        out[:, 4 + m] = mid_of[np.arange(len(tets)), k]
    new_pts = pts.copy()
    nstr = 0
    if not straight:  # curved boundary mids can tangle thin/curved cells: straighten those elements
        bad = np.flatnonzero(min_jacobian_ratio(pts, out) < min_ratio)
        nstr = len(bad)
        # a mid node shared with a good element must stay put, so move only nodes used by bad elements
        # alone and, for shared ones, accept the (still valid) curved position
        good = np.setdiff1d(np.arange(len(out)), bad)
        shared = np.isin(out[bad][:, 4:], out[good][:, 4:].ravel())
        for m, (a, b) in enumerate(GMSH_TET10_EDGES):
            sel = ~shared[:, m]
            new_pts[out[bad][sel, 4 + m]] = 0.5 * (pts[out[bad][sel, a]] + pts[out[bad][sel, b]])
    else:  # straight edges: every mid node at the corner midpoint (CalculiX-safe)
        for m, (a, b) in enumerate(GMSH_TET10_EDGES):
            new_pts[out[:, 4 + m]] = 0.5 * (pts[out[:, a]] + pts[out[:, b]])
    return out, new_pts, int(neg.sum()), nstr


_LUT = np.full((4, 4), -1)
for _k, (_a, _b) in enumerate(PAIRS):
    _LUT[_a, _b] = _k


def tri6_to_gmsh(pts, sel):
    """netgen tri6 rows -> gmsh tri6 order (mid nodes on edges 01, 12, 02)."""
    cc = pts[sel[:, :3]]
    j = _assign_mids(cc, pts[sel[:, 3:]], TRI_PAIRS)
    mid_of = np.empty((len(sel), 3), np.int64)
    for m in range(3):
        mid_of[np.arange(len(sel)), j[:, m]] = sel[:, 3 + m]
    g = np.empty_like(sel)
    g[:, :3] = sel[:, :3]
    g[:, 3], g[:, 4], g[:, 5] = mid_of[:, 0], mid_of[:, 2], mid_of[:, 1]
    return g


def group_tris(pts, tris, ax, planes_faces, plane_coord):
    """Surface tri6 (netgen) on the given faces: all 3 corners on the plane and centroid inside a face bbox.
    Returns {face index: tri6 array in gmsh order}."""
    out = {}
    c = pts[tris[:, :3]]
    flat = np.abs(c[:, :, ax] - plane_coord).max(axis=1) < FACE_TOL
    cen = c.mean(axis=1)
    for fi, (b0, b1) in planes_faces:
        inside = flat & (cen >= b0 - FACE_TOL).all(axis=1) & (cen <= b1 + FACE_TOL).all(axis=1)
        sel = tris[inside]
        if len(sel) == 0:
            out[fi] = sel
            continue
        g = tri6_to_gmsh(pts, sel)
        out[fi] = g
    return out


def write_ccx_clean(src: Path, dst: Path) -> None:
    """mesh.inp without the CPS6 surface elements (and the ELSETs of them): CalculiX treats any CPS6 as a
    plane-stress element and refuses a 3D model that contains them. NSETs and the `domain` ELSET stay."""
    lines = src.read_text().split("\n")
    out, drop, surf_sets = [], False, set()
    for ln in lines:
        if ln.startswith("*"):
            u = ln.upper().replace(" ", "")
            drop = False
            if u.startswith("*ELEMENT") and "TYPE=CPS6" in u:
                drop = True
                surf_sets.add(u.split("ELSET=")[1].split(",")[0])
            elif u.startswith("*ELSET") and u.split("ELSET=")[1].split(",")[0] in (
                    {"BOTTOM", "TOP"} | surf_sets):
                drop = True
        if not drop:
            out.append(ln)
    dst.write_text("\n".join(out))


def write_gmsh(out: Path, pts, tets, surf: dict[str, list[tuple[int, np.ndarray]]], other_faces=()):
    """MSH 4.1 + CalculiX .inp through the gmsh API. surf: group -> [(gmsh CAD face tag, tri6)]."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 4)
    try:
        gmsh.model.add("netgen")
        used = np.unique(np.concatenate([tets.ravel()] + [t.ravel() for g in surf.values() for _, t in g]))
        remap = np.full(len(pts), -1, np.int64)
        remap[used] = np.arange(1, len(used) + 1)
        gmsh.model.addDiscreteEntity(3, 1)
        gmsh.model.mesh.addNodes(3, 1, remap[used].tolist(), pts[used].ravel().tolist())
        gmsh.model.mesh.addElementsByType(1, 11, [], remap[tets].ravel().tolist())
        for name, lst in surf.items():
            tags = []
            for tag, t in lst:
                gmsh.model.addDiscreteEntity(2, tag)
                gmsh.model.mesh.addElementsByType(tag, 9, [], remap[t].ravel().tolist())
                tags.append(tag)
            gmsh.model.addPhysicalGroup(2, tags, name=name)
        for tag in other_faces:  # CAD faces without a group stay as empty entities (harness checks look them up)
            gmsh.model.addDiscreteEntity(2, int(tag))
        gmsh.model.addPhysicalGroup(3, [1], name="domain")
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(out / "mesh.msh"))
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
        gmsh.write(str(out / "mesh.inp"))
        inp = (out / "mesh.inp").read_text().split("\n")
        if inp and inp[0].lower().startswith("*heading"):  # gmsh writes the absolute mesh path here
            inp[1] = " netgen mesh (feagent experiments/mesh_systems/netgen)"
            (out / "mesh.inp").write_text("\n".join(inp))
        write_ccx_clean(out / "mesh.inp", out / "mesh_ccx.inp")
    finally:
        gmsh.finalize()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step")
    ap.add_argument("axis", choices=["x", "y", "z"])
    ap.add_argument("out_dir")
    ap.add_argument("--purpose", required=True, choices=["linear", "nonlinear"])
    ap.add_argument("--maxh", type=float, default=None, help="Netgen maxh [m] (default per part/purpose)")
    ap.add_argument("--grading", type=float, default=0.3, help="Netgen grading (smaller = smoother)")
    ap.add_argument("--thin-layers", type=float, default=0.0,
                    help="elements across thin walls (0 = off, the default: Netgen over-refines and tangles)")
    ap.add_argument("--opt", type=int, default=0, help="Netgen optsteps3d (0 = Netgen default)")
    ap.add_argument("--curvsafety", type=float, default=2.0, help="Netgen curvature safety (elements/rad ~)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.step).resolve().parent.parent.name.rsplit("_bc", 1)[0]
    if stem not in EXPECTED:
        stem = next((s for s in EXPECTED if s in str(a.step)), stem)
    ax = "xyz".index(a.axis)
    maxh = a.maxh or DEFAULT_MAXH.get((stem, a.purpose), 0.003)

    from feagent.geometry import analyze
    geo = analyze(a.step, tessellate=True)
    pts, tets, tris, tri_idx, bot, top = netgen_mesh(a.step, ax, maxh, a.grading, a.curvsafety, geo,
                                                     a.thin_layers, a.opt)
    exp = EXPECTED.get(stem)
    if exp and (len(bot), len(top)) != exp:
        raise SystemExit(f"face counts bottom/top {(len(bot), len(top))} != contract {exp}")
    if not bot or not top:
        raise SystemExit("no platen faces found")
    lo = min(b[1][0][ax] for b in bot)
    hi = max(b[1][0][ax] for b in top)
    gtets, gpts, n_inv, n_straightened = to_gmsh_tets(pts, tets, straight=a.purpose == "nonlinear")
    tris_by_idx = {int(k): tris[tri_idx == k] for k in np.unique(tri_idx)}
    sb = group_tris(gpts, tris, ax, bot, lo)
    st = group_tris(gpts, tris, ax, top, hi)

    # map OCC faces -> gmsh CAD tags (harness geometry), by plane position + bbox centre

    def gtag(b0, b1):
        c = 0.5 * (b0 + b1)
        best = min((f for f in geo.faces if f.kind == "plane"),
                   key=lambda f: np.linalg.norm(np.array(f.center) - c))
        assert np.linalg.norm(np.array(best.center) - c) < 1e-4, "no matching harness face"
        return best.tag, best

    bsurf, tsurf, gcad = [], [], {"bottom": [], "top": []}
    for name, sets, lst in (("bottom", bot, sb), ("top", top, st)):
        for fi, (b0, b1) in sets:
            tag, finfo = gtag(b0, b1)
            assert abs(abs(finfo.normal[ax]) - 1.0) < 1e-6
            if len(lst[fi]) == 0:
                raise SystemExit(f"{name} face {fi} got no surface elements")
            (bsurf if name == "bottom" else tsurf).append((tag, lst[fi]))
            gcad[name].append(tag)
    grouped = set(gcad["bottom"]) | set(gcad["top"])
    write_gmsh(out, gpts, gtets, {"bottom": bsurf, "top": tsurf},
               [f.tag for f in geo.faces if f.tag not in grouped])

    # all-faces copy of the mesh for the harness gates (thin-wall check reads every CAD face's triangles);
    # the delivered mesh.msh carries only the contract groups.
    from netgen.occ import OCCGeometry
    allsurf = {}
    for i, tag in occ_to_harness(OCCGeometry(a.step).shape, geo).items():
        sel = tris_by_idx.get(i + 1)
        if sel is not None and len(sel) and tag not in grouped:
            allsurf.setdefault(f"f{tag}", []).append((tag, tri6_to_gmsh(gpts, sel)))
    assess_dir = out / "_assess"
    assess_dir.mkdir(exist_ok=True)

    # harness quality gates
    from feagent.quality import assess_mesh
    from feagent.spec import QualitySpec
    from feagent.types import GroupResult, MeshResult
    groups = {n: GroupResult(name=n, dim=2, tags=t, elements=sum(len(x[1]) for x in s),
                             area=float(sum(geo.face(k).area for k in t)))
              for n, t, s in (("bottom", gcad["bottom"], bsurf), ("top", gcad["top"], tsurf))}
    groups["domain"] = GroupResult(name="domain", dim=3, tags=[1], elements=len(gtets), area=None)
    nn = len(np.unique(gtets))
    res = MeshResult(ok=True, spec_path="", out_dir=str(out), mesh_path=str(out / "mesh.msh"), exports={},
                     groups=groups, n_nodes=nn, n_elements=len(gtets), order=2, size_scale=1.0,
                     elapsed_s=0.0, errors=[], warnings=[], gates=[], geometry=geo)
    write_gmsh(assess_dir, gpts, gtets, {**{"bottom": bsurf, "top": tsurf}, **allsurf})
    q = assess_mesh(assess_dir / "mesh.msh", res, QualitySpec(), "fea", None)
    import shutil
    shutil.rmtree(assess_dir)
    build_s = round(time.perf_counter() - t0, 2)

    # picture with the harness renderer
    import gmsh
    from feagent.gmsh_session import gmsh_session
    from feagent.gmsh_util import element_qualities
    from feagent.meshdata import read_msh
    from feagent.render import render_quality
    md = read_msh(out / "mesh.msh")
    with gmsh_session():
        gmsh.open(str(out / "mesh.msh"))
        sicn = element_qualities(md.tet_tags, ["minSICN"])["minSICN"]
    render_quality(md, sicn, out / "mesh.png", clip_axis=a.axis)

    rep = q.to_dict()
    rep.update({"system": SYSTEM, "part": stem, "purpose": a.purpose, "n_nodes": nn,
                "n_elements": len(gtets), "order": 2, "dofs": 3 * nn, "build_s": build_s,
                "straight_edges": a.purpose == "nonlinear", "inverted_corner_swaps": n_inv, "tangled_elements_straightened": n_straightened,
                "netgen": {"maxh": maxh, "thin_layers": a.thin_layers, "optsteps3d": a.opt, "grading": a.grading, "curvaturesafety": a.curvsafety},
                "bottom_faces": gcad["bottom"], "top_faces": gcad["top"],
                "bcs": {"fea": {"fixed": ["bottom"], "loads": [{"group": "top", "force": [
                    *(-100.0 if k == ax else 0.0 for k in range(3))]}]}}})
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({"nodes": nn, "elements": len(gtets), "dofs": 3 * nn, "build_s": build_s,
                      "gates": [(g.get("id"), g.get("status")) for g in rep["gates"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
