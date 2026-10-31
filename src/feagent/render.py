"""Offscreen pyvista renders with numeric captions (2x2 views, 1600x1200)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from .types import GeometryModel

if TYPE_CHECKING:
    from .meshdata import MeshData

_SIZE = (1600, 1200)
_VIEWS = ("iso", "+x", "+y", "+z")
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
            "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939", "#843c39"]
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class RenderResult:
    path: str
    caption: str

    def to_dict(self) -> dict:
        return {"path": self.path, "caption": self.caption}

    @classmethod
    def from_dict(cls, d: dict) -> RenderResult:
        return cls(**d)


def _plotter() -> pv.Plotter:
    return pv.Plotter(off_screen=True, shape=(2, 2), window_size=list(_SIZE))


def _view(pl: pv.Plotter, i: int, label: str) -> None:
    pl.subplot(i // 2, i % 2)
    pl.add_text(label, font_size=9, position="upper_left")


def _camera(pl: pv.Plotter, i: int) -> None:
    pl.subplot(i // 2, i % 2)
    if i == 0:
        pl.view_isometric()
    elif i == 1:
        pl.view_yz()  # camera at +x
    elif i == 2:
        pl.view_xz(negative=True)  # camera at +y
    else:
        pl.view_xy()  # camera at +z
    pl.reset_camera()


def _save(pl: pv.Plotter, out_png: str | Path) -> str:
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out))
    pl.close()
    return str(out)


def _surface(points: np.ndarray, tris: np.ndarray) -> pv.PolyData:
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).astype(np.int64).ravel()
    return pv.PolyData(np.asarray(points, float), faces)


def _grid(mesh: MeshData) -> pv.UnstructuredGrid:
    n = len(mesh.tets)
    cells = np.hstack([np.full((n, 1), 4), mesh.tets]).astype(np.int64).ravel()
    return pv.UnstructuredGrid(cells, np.full(n, pv.CellType.TETRA, np.uint8),
                               np.asarray(mesh.points, float))


def render_faces(model: GeometryModel, out_png: str | Path, highlight: list[int] | None = None,
                 max_labels: int = 60) -> RenderResult:
    t = model.tessellation
    if t is None:
        raise ValueError("model has no tessellation")
    hl = sorted(set(highlight or []))
    tags = sorted({int(x) for x in t.face_tag})
    pl = _plotter()
    ranked = sorted(model.faces, key=lambda f: -f.area)
    label_tags = {f.tag for f in ranked[:max_labels]} | set(hl)
    centers = {f.tag: f.center for f in model.faces}
    for i in range(4):
        _view(pl, i, _VIEWS[i])
        for k, tag in enumerate(tags):
            sel = t.face_tag == tag
            if not sel.any():
                continue
            mesh = _surface(t.points, t.triangles[sel])
            if hl:
                if tag in hl:
                    pl.add_mesh(mesh, color="red", show_edges=False)
                else:
                    pl.add_mesh(mesh, color="lightgrey", opacity=0.35)
            else:
                pl.add_mesh(mesh, color=_PALETTE[k % len(_PALETTE)])
        pts = [(tag, centers[tag]) for tag in sorted(label_tags) if tag in centers]
        if pts:
            pl.add_point_labels(np.array([c for _, c in pts], float), [f"f{tg}" for tg, _ in pts],
                                font_size=14, shape=None, show_points=False, always_visible=True)
        _camera(pl, i)
    cap = f"{len(model.faces)} faces; labels f<tag> shown for {len(label_tags)} (largest by area"
    cap += " plus highlighted)" if hl else ")"
    shown = ", ".join(f"f{f.tag} ({f.area:.4g} mm2)" for f in ranked[:min(max_labels, 10)])
    cap += f". Largest: {shown}"
    if hl:
        cap += f"; highlighted {','.join(f'f{h}' for h in hl)}"
    return RenderResult(_save(pl, out_png), cap)


def render_groups(mesh: MeshData, out_png: str | Path) -> RenderResult:
    pl = _plotter()
    names = sorted(mesh.tris)
    parts = []
    shell = _grid(mesh).extract_surface(algorithm="dataset_surface")
    for i in range(4):
        _view(pl, i, _VIEWS[i])
        pl.add_mesh(shell, color="lightgrey", opacity=0.25)
        for k, name in enumerate(names):
            tr = mesh.tris[name]
            if len(tr):
                pl.add_mesh(_surface(mesh.points, tr), color=_PALETTE[k % len(_PALETTE)],
                            label=name)
        if names:
            pl.add_legend(bcolor=None)
        _camera(pl, i)
    for name in names:
        nf = len({int(x) for x in mesh.tri_face.get(name, [])})
        parts.append(f"{name} ({nf} face{'s' if nf != 1 else ''}, {len(mesh.tris[name])} tris)")
    for name, ents in sorted(mesh.volume_groups.items()):
        parts.append(f"{name} (volume, {len(ents)} entit{'ies' if len(ents) != 1 else 'y'})")
    cap = f"{len(parts)} groups: " + ", ".join(parts)
    return RenderResult(_save(pl, out_png), cap)


def render_quality(mesh: MeshData, sicn_per_tet: np.ndarray, out_png: str | Path,
                   clip_axis: str = "z") -> RenderResult:
    s = np.asarray(sicn_per_tet, float)
    grid = _grid(mesh)
    grid.cell_data["SICN"] = s
    lo, hi = np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)
    mid = (lo + hi) / 2
    order = [clip_axis] + [a for a in "xyz" if a != clip_axis]
    worst = np.argsort(s)[:10]
    wc = mesh.points[mesh.tets[worst]].mean(axis=1) if len(worst) else np.zeros((0, 3))
    rad = float(np.linalg.norm(hi - lo)) * 0.015
    pl = _plotter()
    kw = dict(scalars="SICN", clim=[0, 1], cmap="RdYlGn", show_edges=True)
    for i in range(4):
        if i == 0:
            _view(pl, i, "iso")
            pl.add_mesh(grid, **kw)
        else:
            ax = order[i - 1]
            _view(pl, i, f"clip {ax}={mid[_AXES[ax]]:.4g}")
            n = [0.0, 0.0, 0.0]
            n[_AXES[ax]] = 1.0
            c = grid.clip(normal=n, origin=mid, crinkle=True)
            if c.n_cells:
                pl.add_mesh(c, **kw)
        for p in wc:
            pl.add_mesh(pv.Sphere(radius=rad, center=p), color="black")
        _camera(pl, i)
    cap = (f"SICN over {len(s)} tets: min {s.min():.3f}, mean {s.mean():.3f}, "
           f"max {s.max():.3f}; {int((s < 0.2).sum())} below 0.2; "
           f"worst {len(worst)} marked with spheres (min {s[worst[0]]:.3f}); "
           f"clip planes at bbox mid-planes (first: {clip_axis})") if len(s) else "no tets"
    return RenderResult(_save(pl, out_png), cap)


def render_field(mesh: MeshData, point_values: np.ndarray, name: str, out_png: str | Path,
                 warp: np.ndarray | None = None) -> RenderResult:
    v = np.asarray(point_values, float)
    pts = np.asarray(mesh.points, float)
    if warp is not None:
        pts = pts + np.asarray(warp, float)
    grid = _grid(mesh)
    grid.points = pts
    grid.point_data[name] = v
    surf = grid.extract_surface(algorithm="dataset_surface")
    pl = _plotter()
    for i in range(4):
        _view(pl, i, _VIEWS[i])
        pl.add_mesh(surf, scalars=name, cmap="viridis", show_edges=True)
        _camera(pl, i)
    cap = f"{name}: min {v.min():.4g}, max {v.max():.4g}, mean {v.mean():.4g} over {len(v)} nodes"
    if warp is not None:
        cap += f"; warped, max displacement {np.linalg.norm(warp, axis=1).max():.4g}"
    return RenderResult(_save(pl, out_png), cap)
