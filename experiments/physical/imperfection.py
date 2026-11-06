"""Geometric imperfection from linear buckling: CalculiX *BUCKLE on a contract mesh with the crush platen BCs,
then the mesh with its nodes moved by the first buckling mode, for a crush run (crush.py --mesh-dir OUT_DIR).

    python experiments/physical/imperfection.py --part abc_00000065 --mesh-dir DIR --out OUT_DIR
        [--platen sliding] [--amp-mm 0.1] [--mode 1] [--modes 4] [--threads 4]

Buckling load case: the bottom platen as in crush.py --platen; the top platen nodes carry a reference force
(1 kN total, spread evenly, along -axis) and, for --platen bonded, are held laterally; sliding / sliding_top
leave them free laterally. Buckling factors x 1 kN are the linear critical loads.
Imperfection: the mode normalised to a max nodal displacement of --amp-mm (default 0.1 mm, about 1 % of
abc_00000065's 10.8 mm arm and 1.2 % of its 8.1 mm neck wall). C3D10: corner nodes move by the mode,
mid-edge nodes by the mean of their two corners, so every edge keeps its original curvature (straight interior
edges stay straight; CalculiX refuses curved interior C3D10 edges). C3D4 / C3D8 (gmsh tetra / hexahedron,
run at order 1 as in crush.py): every node moves by the mode.
Writes OUT_DIR/mesh.msh (groups unchanged), mesh.inp, imperfection.json (factors, critical loads, amplitude,
where the mode is largest) and mode.vtu (the mode shape on the original mesh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crush import AX, PARTS, platen_bcs  # noqa: E402
from feacore.case import BC, Analysis, Case, Material  # noqa: E402

TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]    # feacore (CalculiX C3D10) mid-edge order
CELL_TYPES = ("tetra10", "tetra", "hexahedron")    # C3D10, C3D4, C3D8


def node_moves(model, U: np.ndarray, amp_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(mode scaled to a max nodal move of amp_m, node moves). Linear cells (C3D4, C3D8) move every node by the
    mode; C3D10 corner nodes move by the mode and mid-edge nodes by the mean of their two corners, so every edge
    keeps its shape (straight interior edges stay straight)."""
    if model.cell_type not in CELL_TYPES:
        raise SystemExit(f"cell type {model.cell_type}: one of {CELL_TYPES} expected")
    Us = U * (amp_m / np.linalg.norm(U, axis=1).max())
    dX = Us.copy()
    if model.cell_type == "tetra10":
        for j, (p, q) in enumerate(TET10_EDGES):
            dX[model.conn[:, 4 + j]] = 0.5 * (Us[model.conn[:, p]] + Us[model.conn[:, q]])
    return Us, dX


F_REF = 1000.0


def lumped_cload(model, group: str, force) -> list[str]:
    """*CLOAD lines spreading the total ``force`` over the corner nodes of the group's facets, by facet area
    (equal shares per facet corner). For linear cells, whose faces the backend cannot load (quad facets)."""
    el, fc = model.facets_of(group)
    nfc = model.n_face_corners
    fn = model.face_nodes(el, fc)[:, :nfc]
    P = model.X[fn]
    if nfc == 4:
        area = 0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1)
    else:
        area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    w = np.zeros(model.n_nodes)
    np.add.at(w, fn.ravel(), np.repeat(area / nfc, nfc))
    w /= w.sum()
    out = ["*CLOAD"]
    for n in np.nonzero(w)[0]:
        out += [f"{n + 1}, {d + 1}, {force[d] * w[n]:.12g}" for d in range(3) if force[d] != 0]
    return out


def buckle_deck(inp: str, n_modes: int) -> str:
    """The static deck with its step replaced by a *BUCKLE step (same *BOUNDARY and *CLOAD)."""
    i = inp.index("\n*STEP")
    keep, block = [], None
    for ln in inp[i + 1:].splitlines():
        if ln.startswith("*"):
            u = ln.upper()
            block = "BOUNDARY" if u.startswith("*BOUNDARY") else "CLOAD" if u.startswith("*CLOAD") else None
            if block:
                keep.append(ln)
        elif block:
            keep.append(ln)
    return inp[: i + 1] + "\n".join(["*STEP", "*BUCKLE", f"{n_modes}", *keep, "*NODE FILE", "U", "*END STEP"]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platen", choices=["bonded", "sliding", "sliding_top"], default="sliding")
    ap.add_argument("--amp-mm", type=float, default=0.1)
    ap.add_argument("--mode", type=int, default=1, help="which buckling mode (1 = lowest)")
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--material", default=str(ROOT / "experiments" / "material" / "alpha_fast_dogbone1.json"))
    a = ap.parse_args()
    import gmsh
    import meshio
    from scipy.spatial import cKDTree

    from feacore.solver._ccx_io import parse_frd_series, run_ccx
    from feacore.solver._ccx_mesh import build_model, read_gmsh
    from feacore.solver.ccx_backend import _Deck

    k = AX[PARTS[a.part]]
    src = Path(a.mesh_dir) / "mesh.msh"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gm = read_gmsh(src)
    order = 2 if any(ct == "tetra10" for ct, _ in gm.groups["domain"]) else 1      # as crush.py
    m = build_model(gm, 3, order)
    mat = json.loads(Path(a.material).read_text())
    bcs = platen_bcs(m, k, 1.0, a.platen)[0]
    bcs = [b for b in bcs if b.group != "top"]
    if a.platen == "bonded":
        bcs.append(BC("displacement", group="top", components=[c for c in range(3) if c != k], value=[0.0, 0.0]))
    f = [0.0, 0.0, 0.0]
    f[k] = -F_REF
    case = Case(mesh=str(src.resolve()), analysis=Analysis("elasticity", "3d", order=order),
                materials={"resin": Material(E=float(mat["E"]) * 1e6, nu=float(mat["nu"]))},
                bcs=bcs + ([BC("force", group="top", value=f)] if m.cell_type == "tetra10" else []))
    case.check()
    deck = _Deck(case, m)
    inp = deck.build()
    if m.cell_type != "tetra10":                # quad / linear facets: area-lumped nodal forces
        i = inp.index("*NODE FILE", inp.index("\n*STEP"))
        inp = inp[:i] + "\n".join(lumped_cload(m, "top", f)) + "\n" + inp[i:]
    run = run_ccx(buckle_deck(inp, a.modes), job="buckle", workdir=out / "ccx_buckle", threads=a.threads)
    factors = []
    for ln in run.dat.read_text().splitlines():
        p = ln.split()
        if len(p) == 2 and p[0].isdigit():
            try:
                factors.append(float(p[1]))
            except ValueError:
                pass
    modes = [b["DISP"].values[:, :3] for _, b in parse_frd_series(run.frd) if "DISP" in b]
    if len(modes) < a.mode:
        raise SystemExit(f"ccx returned {len(modes)} modes")
    U = np.zeros((m.n_nodes, 3))
    blk = [b for _, b in parse_frd_series(run.frd) if "DISP" in b][a.mode - 1]["DISP"]
    U[blk.nodes - 1] = blk.values[:, :3]
    nrm = np.linalg.norm(U, axis=1)
    U, dX = node_moves(m, U, a.amp_mm * 1e-3)
    imax = int(np.argmax(nrm))
    meshio.Mesh(m.X, [(m.cell_type, m.conn)], point_data={"mode": U}).write(str(out / "mode.vtu"))

    gmsh.initialize(["-v", "0"])
    try:
        gmsh.open(str(src))
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = xyz.reshape(-1, 3)
        d, idx = cKDTree(m.X).query(xyz)
        if d.max() > 1e-12:
            raise SystemExit("gmsh nodes do not match the model's nodes")
        new = xyz + dX[idx]
        for t, c in zip(tags.tolist(), new):
            gmsh.model.mesh.setNode(int(t), c.tolist(), [])
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(out / "mesh.msh"))
        gmsh.write(str(out / "mesh.inp"))
    finally:
        gmsh.finalize()
    lines, skip = [], False                     # CalculiX: drop gmsh's surface elements and their ELSETs
    for ln in (out / "mesh.inp").read_text().splitlines():
        if ln.startswith("*"):
            u = ln.upper().replace(" ", "")
            skip = ("*ELEMENT" in u and "TYPE=CPS" in u) or u in ("*ELSET,ELSET=BOTTOM", "*ELSET,ELSET=TOP")
        if not skip:
            lines.append(ln)
    (out / "mesh.inp").write_text("\n".join(lines) + "\n")
    m2 = build_model(read_gmsh(out / "mesh.msh"), 3, order)
    assert m2.n_nodes == m.n_nodes and np.allclose(m2.X, m.X + dX, atol=1e-12)
    info = {"source_mesh": str(src.resolve()), "part": a.part, "platen": a.platen, "reference_force_N": F_REF,
            "buckling_factors": factors[: a.modes], "critical_loads_N": [F_REF * x for x in factors[: a.modes]],
            "mode": a.mode, "amplitude_mm": a.amp_mm, "max_mode_node_m": m.X[imax].tolist(),
            "mode_direction_at_max": (U[imax] / np.linalg.norm(U[imax])).tolist(),
            "mean_lateral_move_top_mm": (U[m.nodes_of("top")].mean(axis=0) * 1e3).tolist()}
    (out / "imperfection.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
