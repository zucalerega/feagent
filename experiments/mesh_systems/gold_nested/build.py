#!/usr/bin/env python
"""gold_nested mesh system: one base tet mesh, uniformly split (gmsh ``mesh.refine``) into nested levels.

    build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear [--h0-frac F] [--levels N] [--select K]

Follows the gold-benchmark generator (branch gold-benchmarks, feacore/goldgen/mesh.py): level k = the base
mesh (Delaunay, global size F x bbox diagonal, curvature sizing) split k times, made quadratic (tet10, boundary
midside nodes on the CAD, interior edges straight). Faces are chosen by geometry (planar, normal along AXIS, at
the extreme position), carried through the refinement as physical groups bottom/top/domain.

Writes OUT_DIR/levels/L<k>/{mesh.msh,mesh.inp,report.json} for every level, and the level chosen for the
purpose copied to OUT_DIR/{mesh.msh,mesh.inp,report.json,mesh.png}, plus OUT_DIR/levels.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import gmsh
import numpy as np

SJ_MIN = 0.05
REPAIR_TARGET = 0.05       # after a split: repair only below this (optimizing swaps edges, so it breaks nesting)
BASE_TARGET = 0.3          # the base is repaired up to this once: every level inherits its cells
QUALITY_MIN = 0.002
LINEAR_MAX_DOFS = 400_000
NONLINEAR_MAX_NODES = 40_000
EXPECT = {"abc_00000003": (1, 1), "abc_00000065": (1, 2)}   # (bottom, top) face counts from the contract
DEFAULT_FRAC = {"linear": 0.125, "nonlinear": 0.125}


def _min_quality() -> float:
    """Worst mean-ratio quality of the linear corner tets."""
    tags, nodes = gmsh.model.mesh.getElementsByType(4)
    if len(tags) == 0:
        tags, nodes = gmsh.model.mesh.getElementsByType(11)
        nodes = np.asarray(nodes).reshape(len(tags), 10)[:, :4]
    else:
        nodes = np.asarray(nodes).reshape(len(tags), 4)
    ntags, X, _ = gmsh.model.mesh.getNodes()
    pos = np.empty(int(np.max(ntags)) + 1, dtype=np.int64)
    pos[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
    P = np.asarray(X).reshape(-1, 3)[pos[nodes.astype(np.int64)]]
    vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6
    l2 = sum(((P[:, i] - P[:, j]) ** 2).sum(1) for i in range(4) for j in range(i + 1, 4))
    return float(np.min(12 * (3 * vol) ** (2 / 3) / l2))


def _repair(target: float = REPAIR_TARGET) -> float:
    q = _min_quality()
    for _ in range(4):
        gmsh.model.mesh.optimize("", niter=3)
        gmsh.model.mesh.optimize("Relocate3D", niter=3)
        q = _min_quality()
        if q >= target:
            break
    return q


def _folded(threshold: float = SJ_MIN) -> np.ndarray:
    tags, _ = gmsh.model.mesh.getElementsByType(11)
    q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
    return np.asarray(tags)[q < threshold]


def _straighten(cells) -> None:
    edges = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]   # gmsh tet10 edge nodes
    for c in cells:
        _, nodes, _, _ = gmsh.model.mesh.getElement(int(c))
        X = {int(n): np.asarray(gmsh.model.mesh.getNode(int(n))[0]) for n in nodes[:4]}
        for a, b, m in edges:
            gmsh.model.mesh.setNode(int(nodes[m]), (0.5 * (X[int(nodes[a])] + X[int(nodes[b])])).tolist(), [])


def classify_faces(axis: int, tol_frac: float = 1e-6) -> tuple[list[int], list[int], dict]:
    """bottom/top by geometry: planar faces normal to the axis at the extreme positions."""
    b = gmsh.model.getBoundingBox(-1, -1)
    lo, hi = b[axis], b[3 + axis]
    diag = float(np.linalg.norm(np.array(b[3:]) - np.array(b[:3])))
    tol = tol_frac * diag * 100
    bottom, top = [], []
    for d, t in gmsh.model.getEntities(2):
        if gmsh.model.getType(d, t) != "Plane":
            continue
        fb = gmsh.model.getBoundingBox(d, t)
        if fb[3 + axis] - fb[axis] > tol:
            continue
        pos = fb[axis]
        (bottom if abs(pos - lo) < tol else top if abs(pos - hi) < tol else []).append(t)
    return bottom, top, {"axis_lo": lo, "axis_hi": hi, "diag": diag, "bbox": list(b)}


def _base(step, axis, h0_frac, curv, threads=1):
    """Open a session with the part, physical groups and the (unrepaired) base tet mesh. Returns face info."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", threads)
    gmsh.model.add("part")
    gmsh.model.occ.importShapes(str(step), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    bottom, top, info = classify_faces(axis)
    vol = [t for d, t in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(2, bottom, name="bottom")
    gmsh.model.addPhysicalGroup(2, top, name="top")
    gmsh.model.addPhysicalGroup(3, vol, name="domain")
    h0 = info["diag"] * h0_frac
    o = gmsh.option.setNumber
    o("Mesh.MeshSizeMax", h0); o("Mesh.MeshSizeMin", h0 / 10); o("Mesh.MeshSizeFromCurvature", curv)
    o("Mesh.Algorithm3D", 1); o("Mesh.HighOrderOptimize", 0)
    gmsh.model.mesh.generate(3)
    return bottom, top, vol, info


def predicted_nodes(step, axis, h0_frac, curv, level) -> int:
    """tet10 nodes of level k = vertices of the linear mesh split k+1 times (vertices + edges)."""
    try:
        _base(step, axis, h0_frac, curv)
        for _ in range(level + 1):
            gmsh.model.mesh.refine()
        return len(gmsh.model.mesh.getNodes()[0])
    finally:
        gmsh.finalize()


def fit_frac(step, axis, curv, level, max_nodes) -> float:
    """Smallest base size fraction whose level-`level` tet10 mesh stays within max_nodes (bisection in log h)."""
    lo, hi = 0.02, 0.5          # lo = fine (too many nodes), hi = coarse
    if predicted_nodes(step, axis, lo, curv, level) <= max_nodes:
        return lo
    for _ in range(9):
        mid = (lo * hi) ** 0.5
        if predicted_nodes(step, axis, mid, curv, level) <= max_nodes:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


def mesh_level(step: str, axis: int, k: int, h0_frac: float, out_msh: Path, out_inp: Path | None,
               threads: int = 1, curv: float = 8, ho: bool = True) -> dict:
    """Level k in a fresh session. Deterministic: Delaunay base, no multithreaded generator."""
    t0 = time.perf_counter()
    try:
        bottom, top, vol, info = _base(step, axis, h0_frac, curv, threads)
        o = gmsh.option.setNumber
        q0 = _min_quality()
        repaired = []
        if q0 < BASE_TARGET:
            repaired.append(0)
            _repair(BASE_TARGET)
        for i in range(k):
            gmsh.model.mesh.refine()
            if _min_quality() < REPAIR_TARGET:
                repaired.append(i + 1)
                _repair()
        q_lin = _min_quality()
        if q_lin < QUALITY_MIN:
            raise RuntimeError(f"L{k}: linear corner quality {q_lin:.2e}")
        n_tets_lin = len(gmsh.model.mesh.getElementsByType(4)[0])
        gmsh.model.mesh.setOrder(2)
        if ho:      # gmsh high-order optimizer (curved-opt): repairs the curved cells instead of straightening them
            o("Mesh.HighOrderOptimize", 2)
            gmsh.model.mesh.optimize("HighOrder")
        bad = _folded()
        n_bad = len(bad)
        for _ in range(5):
            if not len(bad):
                break
            _straighten(bad)              # those cells lose boundary curvature; interior stays straight
            bad = _folded()
        if len(bad):
            raise RuntimeError(f"L{k}: {len(bad)} cells with scaled Jacobian < {SJ_MIN}")
        tags, _ = gmsh.model.mesh.getElementsByType(11)
        sj = float(np.min(gmsh.model.mesh.getElementQualities(tags, "minSJ")))
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        out_msh.parent.mkdir(parents=True, exist_ok=True)
        o("Mesh.MshFileVersion", 4.1); o("Mesh.Binary", 0)
        gmsh.write(str(out_msh))
        if out_inp is not None:
            o("Mesh.SaveGroupsOfNodes", 1); o("Mesh.SaveGroupsOfElements", 1)
            gmsh.write(str(out_inp))
            clean_inp(out_inp)
        groups = {"bottom": bottom, "top": top, "domain": vol}
        areas = {n: float(sum(gmsh.model.occ.getMass(2, t) for t in groups[n])) for n in ("bottom", "top")}
        nel = {n: sum(len(gmsh.model.mesh.getElements(2, t)[1][0]) for t in groups[n]) for n in ("bottom", "top")}
    finally:
        gmsh.finalize()
    return {"level": k, "h0_frac": h0_frac, "curv": curv, "n_nodes": n_nodes, "n_elements": len(tags), "n_tets_linear": n_tets_lin,
            "order": 2, "dofs": 3 * n_nodes, "min_scaled_jacobian": sj, "min_linear_quality": q_lin,
            "base_quality": q0, "repaired_at_levels": repaired, "cells_straightened": n_bad,
            "faces": groups, "areas": areas, "face_elements": nel, "build_s": round(time.perf_counter() - t0, 2),
            "bbox": info["bbox"]}


def clean_inp(path: Path) -> None:
    """CalculiX cannot read gmsh's surface triangles (CPS6): drop them and the ELSETs `bottom`/`top`
    built from them. Left: C3D10 elements, ELSET domain, NSET bottom/top/domain."""
    out, skip = [], False
    for line in path.read_text().splitlines():
        if line.startswith("*"):
            head = line.upper().replace(" ", "")
            skip = "TYPE=CPS6" in head or head in ("*ELSET,ELSET=BOTTOM", "*ELSET,ELSET=TOP")
        if not skip:
            out.append(line)
    path.write_text("\n".join(out) + "\n")


def gates_report(step: str, msh: Path, meta: dict) -> dict:
    """Harness quality gates (feagent.quality.assess_mesh) on this mesh."""
    from feagent.geometry import analyze
    from feagent.quality import assess_mesh
    from feagent.spec import QualitySpec
    from feagent.types import GroupResult, MeshResult

    geo = analyze(str(step), tessellate=False)
    groups = {n: GroupResult(name=n, dim=2, tags=[int(t) for t in meta["faces"][n]],
                             elements=int(meta["face_elements"][n]), area=meta["areas"][n])
              for n in ("bottom", "top")}
    groups["domain"] = GroupResult(name="domain", dim=3, tags=[int(t) for t in meta["faces"]["domain"]],
                                   elements=meta["n_elements"], area=None)
    res = MeshResult(ok=True, spec_path="", out_dir=str(msh.parent), mesh_path=str(msh), exports={}, groups=groups,
                     n_nodes=meta["n_nodes"], n_elements=meta["n_elements"], order=2, size_scale=1.0,
                     elapsed_s=meta["build_s"], errors=[], warnings=[], gates=[], geometry=geo)
    rep = assess_mesh(msh, res, QualitySpec(), "fea", optimize=None)
    d = rep.to_dict()
    gates = d.get("gates", [])
    return {"gates": gates, "status": "fail" if any(g["status"] == "fail" for g in gates) else "pass"}


def render(msh: Path, png: Path) -> None:
    import os
    import subprocess
    import sys
    env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe",
               __GLX_VENDOR_LIBRARY_NAME="mesa", CUDA_VISIBLE_DEVICES="")
    subprocess.run([str(Path(sys.executable).parent / "feagent"), "render", str(msh), "--view", "quality", "-o", str(png)],
                   env=env, check=False, capture_output=True)


def choose(levels: list[dict], purpose: str) -> int:
    if purpose == "linear":     # finest level within the DOF budget
        ok = [m["level"] for m in levels if m["dofs"] <= LINEAR_MAX_DOFS]
        return max(ok) if ok else levels[0]["level"]
    ok = [m["level"] for m in levels if m["n_nodes"] <= NONLINEAR_MAX_NODES]   # coarsest that fits is L0
    return min(ok) if ok else levels[0]["level"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step"); ap.add_argument("axis", choices=["x", "y", "z"]); ap.add_argument("out")
    ap.add_argument("--purpose", choices=["linear", "nonlinear"], required=True)
    ap.add_argument("--h0-frac", type=float, default=None,
                    help="base size / bbox diagonal (default: fitted so the selected level meets the budget)")
    ap.add_argument("--curv", type=float, default=12, help="gmsh elements per 2 pi of curvature in the base")
    ap.add_argument("--levels", type=int, default=2, help="levels exported (L0..L<n-1>)")
    ap.add_argument("--select", type=int, default=None, help="override the level chosen for the purpose")
    ap.add_argument("--fit-level", type=int, default=None,
                    help="level whose size is fitted to the budget (default: the selected level, L1)")
    ap.add_argument("--no-ho", action="store_true", help="skip the gmsh high-order optimizer")
    ap.add_argument("--max-export-nodes", type=int, default=450_000)
    a = ap.parse_args(argv)
    axis = "xyz".index(a.axis)
    sel_level = 1 if a.select is None else a.select          # level that carries the purpose (L1)
    budget = LINEAR_MAX_DOFS // 3 if a.purpose == "linear" else NONLINEAR_MAX_NODES
    t_fit = time.perf_counter()
    fit_level = sel_level if a.fit_level is None else a.fit_level
    frac = a.h0_frac or fit_frac(a.step, axis, a.curv, fit_level, budget)
    fit_s = time.perf_counter() - t_fit
    print(f"base size fraction {frac} (fit {fit_s:.1f}s, level L{fit_level} <= {budget} nodes)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    part = next((k for k in EXPECT if k in str(a.step)), None)
    metas = []
    for k in range(a.levels):
        if metas and metas[-1]["n_nodes"] * 7 > a.max_export_nodes and k > sel_level:
            print(f"L{k} skipped: ~{metas[-1]['n_nodes'] * 7} nodes exceeds --max-export-nodes", flush=True)
            break
        d = out / "levels" / f"L{k}"
        m = mesh_level(a.step, axis, k, frac, d / "mesh.msh", d / "mesh.inp", curv=a.curv, ho=not (a.no_ho or a.purpose == "nonlinear"))   # nonlinear: no HO optimizer (keeps interior edges straight)
        if part:
            nb, nt = len(m["faces"]["bottom"]), len(m["faces"]["top"])
            assert (nb, nt) == EXPECT[part], f"face counts {(nb, nt)} != {EXPECT[part]}"
        assert m["faces"]["bottom"] and m["faces"]["top"]
        m["gates"] = gates_report(a.step, d / "mesh.msh", m) if m["n_nodes"] <= a.max_export_nodes else {"status": "skipped"}
        (d / "report.json").write_text(json.dumps(m, indent=1))
        metas.append(m)
        print(f"L{k}: nodes {m['n_nodes']} dofs {m['dofs']} tets {m['n_elements']} sj {m['min_scaled_jacobian']:.3f} "
              f"gates {m['gates']['status']} {m['build_s']}s", flush=True)
        if m["n_nodes"] > 130_000 and k + 1 < a.levels:
            pass
    sel = sel_level
    m = metas[sel]
    src = out / "levels" / f"L{sel}"
    for f in ("mesh.msh", "mesh.inp"):
        shutil.copy(src / f, out / f)
    rep = dict(m, system="gold_nested", part=part or Path(a.step).stem, purpose=a.purpose, level=sel,
               build_s=round(fit_s + sum(x["build_s"] for x in metas[: sel + 1]), 2),
               fit_s=round(fit_s, 2),
               build_s_this_level=m["build_s"])
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    (out / "levels.json").write_text(json.dumps([{k: v for k, v in x.items() if k not in ("faces", "gates")}
                                                 for x in metas], indent=1))
    render(out / "mesh.msh", out / "mesh.png")
    print(f"chosen L{sel} ({a.purpose}): nodes {m['n_nodes']} dofs {m['dofs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
