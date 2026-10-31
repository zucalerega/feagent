"""Offscreen pyvista renders with numeric captions (2x2 views, 1600x1200)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pyvista as pv

from .types import FaceInfo, GeometryModel, Tessellation
from .vec import bbox_diag

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


def _draw_views(labels: list[str] | tuple[str, ...],
                draw: Callable[[pv.Plotter, int], None]) -> pv.Plotter:
    """A 2x2 plotter: per subplot, its label, ``draw(pl, i)``, then the view camera."""
    pl = _plotter()
    for i, label in enumerate(labels):
        _view(pl, i, label)
        draw(pl, i)
        _camera(pl, i)
    return pl


def _save(pl: pv.Plotter, out_png: str | Path) -> str:
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out))
    pl.close()
    return str(out)


def _color(k: int) -> str:
    return _PALETTE[k % len(_PALETTE)]


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _surface(points: np.ndarray, tris: np.ndarray) -> pv.PolyData:
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).astype(np.int64).ravel()
    return pv.PolyData(np.asarray(points, float), faces)


def _grid(mesh: MeshData) -> pv.UnstructuredGrid:
    n = len(mesh.tets)
    cells = np.hstack([np.full((n, 1), 4), mesh.tets]).astype(np.int64).ravel()
    return pv.UnstructuredGrid(cells, np.full(n, pv.CellType.TETRA, np.uint8),
                               np.asarray(mesh.points, float))


def _shell(grid: pv.UnstructuredGrid) -> pv.PolyData:
    return grid.extract_surface(algorithm="dataset_surface")


# --- render_faces ---------------------------------------------------------

def render_faces(model: GeometryModel, out_png: str | Path, highlight: list[int] | None = None,
                 max_labels: int = 60) -> RenderResult:
    """CAD faces in 4 views, labelled f<tag> (largest faces + highlighted ones)."""
    t = model.tessellation
    if t is None:
        raise ValueError("model has no tessellation")
    hl = sorted(set(highlight or []))
    ranked = sorted(model.faces, key=lambda f: -f.area)
    label_tags = {f.tag for f in ranked[:max_labels]} | set(hl)
    layers = _face_layers(t, hl)
    labels = _face_labels(model, label_tags)

    def draw(pl: pv.Plotter, i: int) -> None:
        for mesh, style in layers:
            pl.add_mesh(mesh, **style)
        if labels:
            pl.add_point_labels(*labels, font_size=14, shape=None, show_points=False,
                                always_visible=True)

    pl = _draw_views(_VIEWS, draw)
    return RenderResult(_save(pl, out_png), _faces_caption(model, ranked, label_tags, hl,
                                                           max_labels))


def _face_layers(t: Tessellation, hl: list[int]) -> list[tuple[pv.PolyData, dict]]:
    """One surface per CAD face with its style: palette colours, or red-on-grey if highlighting."""
    layers = []
    for k, tag in enumerate(sorted({int(x) for x in t.face_tag})):
        if not hl:
            style: dict = {"color": _color(k)}
        elif tag in hl:
            style = {"color": "red", "show_edges": False}
        else:
            style = {"color": "lightgrey", "opacity": 0.35}
        layers.append((_surface(t.points, t.triangles[t.face_tag == tag]), style))
    return layers


def _face_labels(model: GeometryModel, tags: set[int]) -> tuple[np.ndarray, list[str]] | None:
    centers = {f.tag: f.center for f in model.faces}
    pts = [(tag, centers[tag]) for tag in sorted(tags) if tag in centers]
    if not pts:
        return None
    return np.array([c for _, c in pts], float), [f"f{tg}" for tg, _ in pts]


def _faces_caption(model: GeometryModel, ranked: list[FaceInfo], label_tags: set[int],
                   hl: list[int], max_labels: int) -> str:
    cap = f"{len(model.faces)} faces; labels f<tag> shown for {len(label_tags)} (largest by area"
    cap += " plus highlighted)" if hl else ")"
    shown = ", ".join(f"f{f.tag} ({f.area:.4g} mm2)" for f in ranked[:min(max_labels, 10)])
    cap += f". Largest: {shown}"
    if hl:
        cap += f"; highlighted {','.join(f'f{h}' for h in hl)}"
    return cap


# --- render_groups --------------------------------------------------------

def render_groups(mesh: MeshData, out_png: str | Path) -> RenderResult:
    """Surface groups in distinct colours over a translucent mesh shell, with legend."""
    names = sorted(mesh.tris)
    shell = _shell(_grid(mesh))
    surfaces = [(_surface(mesh.points, mesh.tris[n]), _color(k), n)
                for k, n in enumerate(names) if len(mesh.tris[n])]

    def draw(pl: pv.Plotter, i: int) -> None:
        pl.add_mesh(shell, color="lightgrey", opacity=0.25)
        for surf, color, name in surfaces:
            pl.add_mesh(surf, color=color, label=name)
        if names:
            pl.add_legend(bcolor=None)

    pl = _draw_views(_VIEWS, draw)
    return RenderResult(_save(pl, out_png), _groups_caption(mesh, names))


def _groups_caption(mesh: MeshData, names: list[str]) -> str:
    parts = []
    for name in names:
        nf = len({int(x) for x in mesh.tri_face.get(name, [])})
        parts.append(f"{name} ({nf} face{_plural(nf, '', 's')}, {len(mesh.tris[name])} tris)")
    for name, ents in sorted(mesh.volume_groups.items()):
        parts.append(f"{name} (volume, {len(ents)} entit{_plural(len(ents), 'y', 'ies')})")
    return f"{len(parts)} groups: " + ", ".join(parts)


# --- render_quality -------------------------------------------------------

def render_quality(mesh: MeshData, sicn_per_tet: np.ndarray, out_png: str | Path,
                   clip_axis: str = "z") -> RenderResult:
    """Per-tet SICN: iso view plus three mid-plane clips; worst 10 tets marked."""
    s = np.asarray(sicn_per_tet, float)
    grid = _grid(mesh)
    grid.cell_data["SICN"] = s
    lo, hi = np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)
    mid = (lo + hi) / 2
    axes = [clip_axis] + [a for a in "xyz" if a != clip_axis]
    worst = np.argsort(s)[:10]
    markers = _markers(mesh, worst, bbox_diag((lo, hi)) * 0.015)
    labels = ["iso"] + [f"clip {ax}={mid[_AXES[ax]]:.4g}" for ax in axes]

    def draw(pl: pv.Plotter, i: int) -> None:
        _add_sicn(pl, grid, mid, axes[i - 1] if i else None)
        for m in markers:
            pl.add_mesh(m, color="black")

    pl = _draw_views(labels, draw)
    return RenderResult(_save(pl, out_png), _quality_caption(s, worst, clip_axis))


def _markers(mesh: MeshData, tets: np.ndarray, radius: float) -> list[pv.PolyData]:
    centers = mesh.points[mesh.tets[tets]].mean(axis=1) if len(tets) else np.zeros((0, 3))
    return [pv.Sphere(radius=radius, center=c) for c in centers]


def _add_sicn(pl: pv.Plotter, grid: pv.UnstructuredGrid, mid: np.ndarray,
              clip_axis: str | None) -> None:
    """Whole grid (``clip_axis`` None) or its crinkle clip through ``mid`` normal to the axis."""
    kw = dict(scalars="SICN", clim=[0, 1], cmap="RdYlGn", show_edges=True)
    if clip_axis is None:
        pl.add_mesh(grid, **kw)
        return
    normal = [0.0, 0.0, 0.0]
    normal[_AXES[clip_axis]] = 1.0
    c = grid.clip(normal=normal, origin=mid, crinkle=True)
    if c.n_cells:
        pl.add_mesh(c, **kw)


def _quality_caption(s: np.ndarray, worst: np.ndarray, clip_axis: str) -> str:
    if not len(s):
        return "no tets"
    return (f"SICN over {len(s)} tets: min {s.min():.3f}, mean {s.mean():.3f}, "
            f"max {s.max():.3f}; {int((s < 0.2).sum())} below 0.2; "
            f"worst {len(worst)} marked with spheres (min {s[worst[0]]:.3f}); "
            f"clip planes at bbox mid-planes (first: {clip_axis})")


# --- render_field ---------------------------------------------------------

def render_field(mesh: MeshData, point_values: np.ndarray, name: str, out_png: str | Path,
                 warp: np.ndarray | None = None) -> RenderResult:
    """A nodal scalar field on the (optionally warped) mesh surface."""
    v = np.asarray(point_values, float)
    grid = _grid(mesh)
    if warp is not None:
        grid.points = np.asarray(mesh.points, float) + np.asarray(warp, float)
    grid.point_data[name] = v
    surf = _shell(grid)
    pl = _draw_views(_VIEWS, lambda pl, i: pl.add_mesh(surf, scalars=name, cmap="viridis",
                                                       show_edges=True))
    cap = f"{name}: min {v.min():.4g}, max {v.max():.4g}, mean {v.mean():.4g} over {len(v)} nodes"
    if warp is not None:
        cap += f"; warped, max displacement {np.linalg.norm(warp, axis=1).max():.4g}"
    return RenderResult(_save(pl, out_png), cap)
