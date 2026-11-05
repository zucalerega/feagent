"""Print-ready parts for the physical pull tests (Rigid 10K on an Instron), see physical_tests/README.md.

Builds them from STEP files already in the repo (metres) and writes <name>.step and <name>.stl in
millimetres to physical_tests/parts/, plus parts.png. STL carries no units; PreForm reads mm.

  abc_00000003_pull          gold part abc_00000003 (closed tube with two opposed side stubs). Both
                             stubs lengthened by 40 mm, same diameter and axis, for V-face wedge grips.
  abc_00000065_pull          gold part abc_00000065 (shaft clamp). A 45 x 6 x 16.2 mm grip tab under
                             the base (flush with the front and back faces), 3 mm fillets at its root.
  mounting_bracket_vertical  feacore mounting bracket, unchanged (pin pull through the load hole).

Only grip ends are added; everything the tests measure is the original geometry. The face tags of
the gold parts (f15, f17, f33) are the ones their gold statements use.

    python scripts/physical_parts.py [--out physical_tests/parts] [--no-render]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    "abc_00000003_pull": REPO / "benchmarks/gold/abc_00000003_bc0/public/geometry.step",
    "abc_00000065_pull": REPO / "benchmarks/gold/abc_00000065_bc0/public/geometry.step",
    "mounting_bracket_vertical": REPO / "benchmarks/mounting_bracket_vertical/public/geometry.step",
}
STUB_EXTENSION = 40.0                                   # mm added to each stub of abc_00000003
STUB_END_FACES = (15, 17)                               # planar stub ends at x = +-34.85 mm
TAB = {"face": 33, "thickness": 6.0, "length": 45.0, "fillet": 3.0}   # abc_00000065, base bottom y = -43.96
STL_OPTIONS = {"Mesh.MeshSizeFromCurvature": 48, "Mesh.MeshSizeMax": 5.0, "Mesh.MeshSizeMin": 0.2}  # chord <= 0.04 mm at r = 20 mm


def import_mm(path: Path) -> int:
    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
    if max(x1 - x0, y1 - y0, z1 - z0) < 1.0:            # repo STEP files are in metres
        gmsh.model.occ.dilate(gmsh.model.occ.getEntities(3), 0, 0, 0, 1e3, 1e3, 1e3)
        gmsh.model.occ.synchronize()
    (vol,) = gmsh.model.getEntities(3)
    return vol[1]


def extend_stubs(vol: int) -> None:
    tools = []
    for face in STUB_END_FACES:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, face)
        assert x1 - x0 < 1e-3, f"f{face} is not a planar end face normal to x"
        xc, yc, zc, r = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (y1 - y0) / 2
        tools.append((3, gmsh.model.occ.addCylinder(xc, yc, zc, np.sign(xc) * STUB_EXTENSION, 0, 0, r)))
    gmsh.model.occ.fuse([(3, vol)], tools)
    gmsh.model.occ.synchronize()


def add_base_tab(vol: int) -> None:
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, TAB["face"])
    assert y1 - y0 < 1e-3, f"f{TAB['face']} is not the planar base bottom"
    t, length = TAB["thickness"], TAB["length"]
    tab = gmsh.model.occ.addBox(-t / 2, y0 - length, z0, t, length + 0.05, z1 - z0)  # 0.05 mm overlap
    (fused,), _ = gmsh.model.occ.fuse([(3, vol)], [(3, tab)])
    gmsh.model.occ.synchronize()
    root = []
    for _, c in gmsh.model.getEntitiesInBoundingBox(-t / 2 - 0.05, y0 - 0.05, z0 - 0.1,
                                                    t / 2 + 0.05, y0 + 0.05, z1 + 0.1, 1):
        b = gmsh.model.getBoundingBox(1, c)
        if b[5] - b[2] > 1.0 and b[3] - b[0] < 1e-3:    # straight edges along z at x = +-t/2
            root.append(c)
    assert len(root) == 2, f"expected 2 tab root edges, got {root}"
    gmsh.model.occ.fillet([fused[1]], root, [TAB["fillet"]])
    gmsh.model.occ.synchronize()


def check_and_write(name: str, out: Path) -> dict:
    vols = gmsh.model.getEntities(3)
    assert len(vols) == 1, f"{name}: expected one solid, got {len(vols)}"
    volume = gmsh.model.occ.getMass(3, vols[0][1])
    gmsh.write(str(out / f"{name}.step"))
    for k, v in STL_OPTIONS.items():
        gmsh.option.setNumber(k, v)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(out / f"{name}.stl"))
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    _, nodes = gmsh.model.mesh.getElementsByType(2)
    tri = index[nodes.astype(np.int64)].reshape(-1, 3)
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = int((counts != 2).sum())
    assert open_edges == 0, f"{name}: surface mesh is not watertight ({open_edges} edges)"
    extent = xyz.max(0) - xyz.min(0)
    info = {"name": name, "extent_mm": extent.round(2).tolist(), "volume_cm3": round(volume / 1e3, 1),
            "triangles": len(tri), "xyz": xyz, "tri": tri}
    print(f"{name}: {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm, {volume / 1e3:.1f} cm3, "
          f"{len(tri)} triangles, watertight")
    return info


def render(parts: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(5 * len(parts), 9))
    for i, p in enumerate(parts):
        P = p["xyz"][p["tri"]]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
        lo, hi = p["xyz"].min(0), p["xyz"].max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        for j, (el, az) in enumerate([(25, -60), (20, 120)]):
            ax = fig.add_subplot(2, len(parts), j * len(parts) + i + 1, projection="3d")
            e, a = np.radians(el), np.radians(az)
            view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            shade = 0.35 + 0.65 * np.abs(n @ view)
            ax.add_collection3d(Poly3DCollection(P, facecolors=plt.cm.Blues(0.3 + 0.6 * shade), edgecolor="none"))
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
            ax.view_init(el, az)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
            if j == 0:
                ax.set_title(f"{p['name']}\n{' x '.join(f'{v:.1f}' for v in p['extent_mm'])} mm, {p['volume_cm3']} cm3")
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "physical_tests" / "parts")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    parts = []
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        for name, src in SOURCES.items():
            gmsh.model.add(name)
            vol = import_mm(src)
            if name == "abc_00000003_pull":
                extend_stubs(vol)
            elif name == "abc_00000065_pull":
                add_base_tab(vol)
            parts.append(check_and_write(name, args.out))
            gmsh.model.remove()
    finally:
        gmsh.finalize()
    if not args.no_render:
        render(parts, args.out / "parts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
