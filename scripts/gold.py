"""Gold benchmarks (feacore.problems.gold): import generator archives, list, verify and render them.

    python scripts/gold.py import ARCHIVE.tgz|DIR [...] [--root benchmarks/gold] [--no-meshes]
    python scripts/gold.py list
    python scripts/gold.py verify [ID ...] [--level 0]
    python scripts/gold.py render [ID ...] [--level 1] [--runs-dir DIR] [--png DIR] [--no-store]

import  copies every BC set of a goldgen archive (out/<id>/public|private, out/_meshes/<part>/L<k>.msh,
        logs/) into the gold root after checking it: files present, ids consistent, nothing private in
        public/, reference case readable, references among the stated QoIs, every group a face of the
        part, and the STEP's face tags equal to the part's faces (what agents name groups by). Sets
        without a reference QoI are skipped (nothing to score). STEP copies of one part that differ
        only in their export timestamp are stored byte-identical, so git keeps one blob per part.
        index.json records every set and mesh (sha256, cells, h) and the archive it came from.
verify  re-solves sets on a stored mesh level with the fixed solver and compares every QoI with the
        generator's value on that level: the solver and the import reproduce the study.
render  re-solves sets on a stored mesh level and records each solve in the run store, where the
        UI's field viewer shows it (source "gold", agent "gold:L<level>"); --png also writes a PNG
        per set (von Mises on the deformed shape, supports and loads marked) and a contact sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SET_FILES = ("public/problem.json", "public/geometry.step", "private/reference.json", "private/study.json")
STEP_STAMP = re.compile(rb"FILE_NAME\('([^']*)','[^']*'")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unstamped(step: bytes) -> bytes:
    return STEP_STAMP.sub(rb"FILE_NAME('\1',''", step, count=1)


def _archive_name(src: Path) -> str:
    name = src.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _gold_root(a) -> Path:
    from feacore.problems.gold import gold_root

    return Path(a.root) if a.root else gold_root()


# --------------------------------------------------------------------------- import


def check_set(d: Path) -> list[str]:
    """What is wrong with one BC set directory of an archive (empty: fine)."""
    from feacore.case import Case, CaseError

    errs = [f"missing {f}" for f in SET_FILES if not (d / f).is_file()]
    if errs:
        return errs
    st = json.loads((d / "public" / "problem.json").read_text())
    ref = json.loads((d / "private" / "reference.json").read_text())
    if st.get("id") != d.name or ref.get("id") != d.name:
        errs.append(f"id mismatch: directory {d.name}, statement {st.get('id')}, reference {ref.get('id')}")
    extra = sorted(p.name for p in (d / "public").iterdir() if p.name not in ("problem.json", "geometry.step"))
    if extra:
        errs.append(f"unexpected public files {extra}")
    if {"references", "reference_case", "bcset"} & set(st):
        errs.append("the public statement carries private data")
    names = {q["name"] for q in st.get("qois", [])}
    errs += [f"reference {r['qoi']!r} is not a stated QoI" for r in ref.get("references", []) if r["qoi"] not in names]
    try:
        errs += Case.from_dict({**ref["reference_case"], "mesh": "MESH.msh"}).validate()
    except (CaseError, KeyError) as e:
        errs.append(f"reference case: {e}")
        return errs
    faces = {f"f{f['tag']}" for f in ref["part"]["faces"]}
    groups = {b["group"] for b in ref["reference_case"]["bcs"] if b.get("group")}
    groups |= {q["group"] for q in st.get("qois", []) if q.get("group")}
    errs += [f"group {g} is not a face of the part" for g in sorted(groups - faces)]
    return errs


def check_faces(problem) -> str | None:
    """None if the problem's geometry (its STEP, imported as agents import it) has the part's faces:
    same tags, areas and centroids; else what differs."""
    import gmsh

    from feacore.geometry.gmsh_tools import gmsh_session

    part = problem.part
    diag = float(np.linalg.norm(np.subtract(part["bbox"][3:], part["bbox"][:3])))
    with gmsh_session(problem.id):
        problem.build_geometry()
        got = {}
        for dim, ptag in gmsh.model.getPhysicalGroups(2):
            (tag,) = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
            got[gmsh.model.getPhysicalName(dim, ptag)] = (gmsh.model.occ.getMass(2, int(tag)),
                                                          gmsh.model.occ.getCenterOfMass(2, int(tag)))
    want = {f"f{f['tag']}": f for f in part["faces"]}
    if set(got) != set(want):
        return f"STEP faces {sorted(set(got) ^ set(want))} differ from the part's"
    for name, f in want.items():
        area, c = got[name]
        if abs(area - f["area"]) > 1e-6 * f["area"] or np.linalg.norm(np.subtract(c, f["centroid"])) > 1e-6 * diag:
            return f"STEP face {name} does not match the part (area {area:.4g} vs {f['area']:.4g} m^2)"
    return None


def _load_index(root: Path) -> dict:
    p = root / "index.json"
    index = json.loads(p.read_text()) if p.exists() else {}
    index.setdefault("sets", {})
    index.setdefault("parts", {})
    return index


def _write_index(root: Path, index: dict) -> None:
    index["sets"] = dict(sorted(index["sets"].items()))
    index["parts"] = dict(sorted(index["parts"].items()))
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n")


def _install_set(src: Path, root: Path, part: str, index: dict) -> str:
    """Copy one checked set into the root; "new", "updated" or "unchanged"."""
    dst = root / src.name
    files = {rel: (src / rel).read_bytes() for rel in SET_FILES}
    for other, row in index["sets"].items():            # one STEP blob per part (see the module doc)
        p = root / other / "public" / "geometry.step"
        if other != src.name and row.get("part") == part and p.is_file():
            canon = p.read_bytes()
            if canon != files["public/geometry.step"] and _unstamped(canon) == _unstamped(files["public/geometry.step"]):
                files["public/geometry.step"] = canon
            break
    old = {rel: (dst / rel).read_bytes() for rel in SET_FILES if (dst / rel).is_file()}
    if old == files:
        return "unchanged"
    status = "updated" if dst.exists() else "new"
    if dst.exists():
        shutil.rmtree(dst)
    for rel, data in files.items():
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / rel).write_bytes(data)
    return status


def _install_meshes(base: Path, root: Path, index: dict, archive: str) -> list[str]:
    msgs = []
    for side in sorted(base.rglob("_meshes/*/L*.json")):
        msh = side.with_suffix(".msh")
        if not msh.is_file():
            continue
        part, meta = side.parent.name, json.loads(side.read_text())
        dst = root / "_meshes" / part / msh.name
        sha = _sha256(msh)
        if dst.is_file() and _sha256(dst) != sha:
            msgs.append(f"  KEEP {part}/{msh.name}: differs from the installed mesh (the installed sets were solved on it)")
            continue
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(msh, dst)
            shutil.copyfile(side, dst.with_suffix(".json"))
            msgs.append(f"  mesh {part}/{msh.name}: {meta.get('cells')} cells, {msh.stat().st_size / 1e6:.1f} MB")
        entry = index["parts"].setdefault(part, {}).setdefault("meshes", {})
        entry[msh.stem] = {"file": f"_meshes/{part}/{msh.name}", "h": meta.get("h"), "cells": meta.get("cells"),
                           "scheme": meta.get("scheme"), "min_sj": meta.get("min_sj"),
                           "bytes": msh.stat().st_size, "sha256": sha, "archive": archive}
    return msgs


def cmd_import(a) -> int:
    from feacore.problems.gold import GoldProblem

    root = _gold_root(a)
    root.mkdir(parents=True, exist_ok=True)
    index = _load_index(root)
    n_bad = 0
    for src in map(Path, a.sources):
        archive = _archive_name(src)
        with tempfile.TemporaryDirectory() as tmp:
            if src.is_dir():
                base = src
            else:
                with tarfile.open(src) as tf:
                    tf.extractall(tmp, filter="data")      # refuses absolute paths, .., links out of tmp
                base = Path(tmp)
            summary = {}
            for log in base.rglob("*summary.jsonl"):
                for line in log.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        summary[row.get("id")] = row
            sets = sorted(p.parent.parent for p in base.rglob("private/reference.json"))
            print(f"{archive}: {len(sets)} BC sets")
            faces_ok: dict[str, str | None] = {}
            for d in sets:
                errs = check_set(d)
                p = None if errs else GoldProblem(d)
                if p is not None and not p.references():
                    errs.append("no reference QoI (nothing to score)")
                if not errs:
                    if p.family not in faces_ok:
                        faces_ok[p.family] = check_faces(p)
                    if faces_ok[p.family]:
                        errs.append(faces_ok[p.family])
                if errs:
                    n_bad += 1
                    print(f"  SKIP {d.name}: {'; '.join(errs)}")
                    continue
                status = _install_set(d, root, p.family, index)
                row = summary.get(p.id, {})
                index["sets"][p.id] = {"part": p.family, "archive": archive,
                                       "references": [r.qoi for r in p.references()],
                                       "singular": p.singular_qois(), "dofs": row.get("dofs"),
                                       "seconds": row.get("seconds")}
                index["parts"].setdefault(p.family, {}).update(source=p.part["source"], faces=len(p.part["faces"]))
                print(f"  {status:9s} {p.id}: references {', '.join(r.qoi for r in p.references())}")
            if not a.no_meshes:
                print("\n".join(_install_meshes(base, root, index, archive)))
    _write_index(root, index)
    print(f"index: {root / 'index.json'} ({len(index['sets'])} sets, {len(index['parts'])} parts)")
    return 1 if n_bad else 0


# --------------------------------------------------------------------------- list / verify


def _selected(a):
    from feacore.problems.gold import load_all

    problems, errors = load_all(_gold_root(a))
    for d, e in errors.items():
        print(f"warning: {d}: {e}", file=sys.stderr)
    if a.ids:
        want = set(a.ids)
        problems = [p for p in problems if p.id in want or p.family in want]
    return problems


def cmd_list(a) -> int:
    for p in _selected(a):
        levels = [k for k in range(4) if p.mesh_file(k)]
        refs = ", ".join(f"{r.qoi}={r.value:.4g} (tol {r.tolerance:.1%})" for r in p.references())
        print(f"{p.id}  [{len(p.part['faces'])} faces, meshes L{levels}]  scored: {p.scored_qoi()}\n"
              f"    references: {refs}\n    singular: {', '.join(p.singular_qois()) or '-'}")
    return 0


def _compare(p, level: int, vals: dict) -> list[tuple[str, float, float, bool]]:
    """(QoI, here, generator, agrees) for every QoI the generator computed on that level."""
    run = next((r for r in p.study()["runs"] if r["level"] == level), None)
    if run is None:
        return []
    gold = run["qois"]
    r_scale = max([abs(v) for k, v in gold.items() if k[:1] == "R" and k[2:3] == "_"] or [0.0])
    rows = []
    for k, g in gold.items():
        v = vals.get(k, float("nan"))
        scale = r_scale if k[:1] == "R" and k[2:3] == "_" else abs(g)
        rows.append((k, v, g, bool(abs(v - g) <= 1e-6 * scale + 1e-300)))
    return rows


def cmd_verify(a) -> int:
    from feacore.solver import solve

    n_bad = 0
    for p in _selected(a):
        face_err = check_faces(p)
        mesh = p.mesh_file(a.level)
        if mesh is None:
            print(f"{p.id}: no L{a.level} mesh imported{'; ' + face_err if face_err else ''}")
            n_bad += bool(face_err)
            continue
        t0 = time.perf_counter()
        res = solve(p.case(mesh), backend="skfem")
        rows = _compare(p, a.level, res.evaluate_all())
        bad = [r for r in rows if not r[3]]
        ok = not bad and not face_err and rows
        n_bad += not ok
        print(f"{p.id} L{a.level}: {'OK' if ok else 'MISMATCH'} ({res.info.n_dofs} dofs, "
              f"{time.perf_counter() - t0:.1f}s, {len(rows) - len(bad)}/{len(rows)} QoIs reproduce)")
        if face_err:
            print(f"    {face_err}")
        for k, v, g, _ in bad:
            print(f"    {k}: {v:.6e} here vs {g:.6e} in the study")
    return 1 if n_bad else 0


# --------------------------------------------------------------------------- render

# colours: the dataviz reference palette. Stress is a magnitude, so one sequential hue (blue,
# light = low -> dark = high); loads take categorical slot 2 (orange); supports and text use ink.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
LOAD, INK, INK2, MUTED, SURFACE = "#eb6834", "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def _bc_label(bc: dict) -> str:
    g = bc["group"]
    if bc["type"] == "fixed":
        return f"Fixed {g}"
    if bc["type"] == "displacement":
        return f"Roller {''.join('xyz'[k] for k in bc['components'])} = 0, {g}"
    if bc["type"] == "pressure":
        return f"Pressure {bc['value'] / 1e6:.3g} MPa, {g}"
    if bc["type"] == "traction":
        return f"Traction {np.linalg.norm(bc['value']) / 1e6:.3g} MPa, {g}"
    return f"Force {np.linalg.norm(bc['value']) / 1e3:.3g} kN, {g}"


def _facet_patches(mesh: Path, groups: list[str]) -> dict[str, np.ndarray]:
    """{group: (n, 3, 3) display triangles (corner coordinates)} of the mesh's facet groups; quadratic
    facets are split in four through their midside nodes, as the viewer does."""
    import meshio

    m = meshio.read(mesh)
    out = {}
    for g in groups:
        if g not in m.field_data:
            continue
        tag = m.field_data[g][0]
        tris = []
        for block, phys in zip(m.cells, m.cell_data["gmsh:physical"]):
            if block.type not in ("triangle", "triangle6"):
                continue
            c = block.data[np.asarray(phys) == tag]
            if block.type == "triangle6":       # meshio order: 3 = (0, 1), 4 = (1, 2), 5 = (2, 0)
                c = np.concatenate([c[:, [0, 3, 5]], c[:, [3, 1, 4]], c[:, [5, 4, 2]], c[:, [3, 4, 5]]])
            tris.append(m.points[c])
        if tris:
            out[g] = np.concatenate(tris)
    return out


def _nice_top(x: float) -> float:
    """Smallest 4 x step >= x with step in {1, 2, 2.5, 4, 5} x 10^k: colour-bar ticks at round values."""
    e = 10.0 ** np.floor(np.log10(max(x / 4, 1e-300)))
    return 4 * next(m * e for m in (1, 2, 2.5, 4, 5, 10) if m * e >= x / 4)


def _fmt_qoi(name: str, v: float) -> str:
    if name.startswith("vm"):
        return f"{v / 1e6:.4g} MPa"
    if name.startswith("u_"):
        return f"{v * 1e6:.4g} µm"
    if name == "strain_energy":
        return f"{v:.4g} J"
    return f"{v:.4g} N" if name[:1] == "R" else f"{v:.4g}"


def _view_dirs(patch_normals: list[np.ndarray]) -> list[np.ndarray]:
    """Two camera directions (from the part towards the camera): the first faces as many supports
    and loads as possible, the second the ones the first sees least (else the opposite side)."""
    c = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
    c /= np.linalg.norm(c, axis=1)[:, None]
    seen = np.array([[np.mean(n @ d > 0.2) for n in patch_normals] for d in c])   # (dirs, BCs) facing fraction
    tilt = 0.01 * (np.abs(c).sum(1) == 3)                  # ties: prefer a 3/4 (corner) view
    first = int(np.argmax(seen.sum(1) + tilt))
    weak = seen[first] < 0.5
    second = int(np.argmax(seen[:, weak].sum(1) + tilt)) if weak.any() else -1
    d2 = c[second] if second >= 0 and second != first else -c[first]
    return [c[first], d2]


def _font(size: int):
    from PIL import ImageFont

    try:                                   # matplotlib ships DejaVu Sans: the same sans on every box
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default(size=size)


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", RAMP[2:])    # from step 200: the part stays visible


def _render_views(surf, top: float, marks: list, letters: list, dirs: list, centre, diag: float):
    """The two 3D views side by side (numpy RGB image)."""
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1640, 820), border=False)
    edges = surf.extract_feature_edges(feature_angle=35, boundary_edges=False, manifold_edges=False,
                                       non_manifold_edges=False)
    for k, d in enumerate(dirs):
        pl.subplot(0, k)
        pl.set_background(SURFACE)
        pl.add_mesh(surf, scalars="vm", cmap=_cmap(), clim=[0, top], smooth_shading=True, split_sharp_edges=True,
                    ambient=0.45, diffuse=0.6, specular=0.0, show_scalar_bar=False)
        pl.add_mesh(edges, color=MUTED, line_width=1)
        for group in marks:
            for mesh_, kw in group:
                pl.add_mesh(mesh_, **kw)
        if letters:                                 # hidden with their face (no always_visible)
            pl.add_point_labels(np.asarray([a for a, _ in letters]), [t for _, t in letters], font_size=20,
                                bold=True, text_color=INK, shape_color="white", shape_opacity=0.9, show_points=False)
        pl.add_axes(color=INK, line_width=2, labels_off=False)
        up = (0, 0, 1) if abs(d[2]) < 0.95 else (0, 1, 0)
        pl.camera_position = [tuple(centre + 3 * diag * d), tuple(centre), up]
        pl.reset_camera()
        pl.camera.zoom(1.12)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def render_png(p, res, mesh: Path, out: Path, level: int) -> Path:
    """von Mises on the deformed boundary in two views; supports outlined in ink and loads in orange
    (with arrows), lettered; above, the title and the BCs in words; below, the reference QoIs on this
    mesh against the gold values."""
    import pyvista as pv
    from PIL import Image, ImageDraw
    from scipy.spatial import cKDTree

    from feacore.ui.viewdata import extract_view

    view = extract_view(res, mesh, fields=["von_mises", "ux", "uy", "uz"])
    P, T, U = np.asarray(view["points"], float), view["triangles"], np.asarray(view["disp"], float)
    vm = np.asarray(view["values"]["von_mises"]) / 1e6
    diag = float(np.linalg.norm(np.ptp(P, axis=0)))
    umax = float(np.linalg.norm(U, axis=1).max())
    scale = 0.05 * diag / umax if umax > 0 else 0.0
    X = P + scale * U
    surf = pv.PolyData(X, np.hstack([np.full((len(T), 1), 3), T]).ravel())
    surf["vm"] = vm
    normals = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)["Normals"]
    tree = cKDTree(P)
    top = _nice_top(float(np.percentile(vm, 99)))

    # supports and loads: patches lifted off the surface (no z-fighting), outlines, arrows, a letter each
    bcs = p.case("MESH.msh").bcs
    patches = _facet_patches(mesh, [b.group for b in bcs])
    marks, letters, legend, patch_normals = [], [], [], []
    for b in bcs:
        tri = patches.get(b.group)
        if tri is None:
            continue
        _, idx = tree.query(tri.reshape(-1, 3))
        V = X[idx] + 2e-3 * diag * normals[idx]
        patch = pv.PolyData(V, np.hstack([np.full((len(tri), 1), 3), np.arange(len(V)).reshape(-1, 3)]).ravel())
        patch = patch.clean()                          # shared vertices: only the patch rim is a boundary
        is_load = b.type in ("force", "pressure", "traction")
        col = LOAD if is_load else INK
        n_tri = normals[idx].reshape(-1, 3, 3).mean(1)
        n_tri /= np.linalg.norm(n_tri, axis=1)[:, None]
        patch_normals.append(n_tri)
        c = V.reshape(-1, 3, 3).mean(1)
        w = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        near = int(np.argmin(np.linalg.norm(c - (c * w[:, None]).sum(0) / w.sum(), axis=1)))   # on the face
        group = [(patch, dict(color=col, opacity=0.15, lighting=False)),
                 (patch.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                              non_manifold_edges=False), dict(color=col, line_width=4))]
        if is_load:
            L, arrows = 0.07 * diag, []
            for i in np.linspace(0, len(c) - 1, min(len(c), 10)).astype(int):
                d = np.asarray(b.value, float) if b.type in ("force", "traction") else -n_tri[i]
                d = d / np.linalg.norm(d)
                arrows.append(pv.Arrow(start=c[i] - d * L, direction=d, scale=L, tip_length=0.3, tip_radius=0.09,
                                       shaft_radius=0.03))
            group.append((pv.merge(arrows), dict(color=LOAD, lighting=False)))
        marks.append(group)
        letter = "ABCDEFGH"[len(letters)]
        letters.append((c[near] + 0.03 * diag * n_tri[near], letter))
        legend.append((letter, _bc_label(b.__dict__), col))
    dirs = _view_dirs(patch_normals) if patch_normals else [np.array([1.0, -1.0, 1.0]) / 3 ** 0.5] * 2
    img = _render_views(surf, top, marks, letters, dirs, 0.5 * (X.min(0) + X.max(0)), diag)

    # compose: header (title, BCs), the views and colour bar, footer (reference QoIs on this mesh vs gold)
    vals = res.evaluate_all()
    refs = p.references()
    W, head, row = img.shape[1] + 160, 118, 30
    foot = 84 + row * len(refs)
    page = Image.new("RGB", (W, head + img.shape[0] + foot), SURFACE)
    page.paste(Image.fromarray(img), (0, head))
    dr = ImageDraw.Draw(page)
    f_title, f_text, f_small = _font(30), _font(20), _font(17)
    bx, by, bw, bh = img.shape[1] + 30, head + 150, 26, img.shape[0] - 300
    cm = _cmap()
    for i in range(bh):
        dr.line((bx, by + i, bx + bw, by + i), fill=tuple(int(255 * v) for v in cm(1 - i / (bh - 1))[:3]))
    dr.text((bx - 24, by - 44), "von Mises [MPa]", fill=INK, font=f_small)
    for k in range(5):
        yk = by + bh - 1 - k * (bh - 1) / 4
        dr.line((bx + bw, yk, bx + bw + 6, yk), fill=INK2, width=2)
        dr.text((bx + bw + 10, yk - 11), f"{k * top / 4:.0f}" if top >= 4 else f"{k * top / 4:.2g}", fill=INK,
                font=f_small)
    dr.text((24, 14), f"{p.id}", fill=INK, font=f_title)
    dr.text((24 + dr.textlength(p.id, font=f_title) + 18, 24),
            f"von Mises on the deformed shape (x{scale:.0f}), L{level} reference mesh, {res.info.n_dofs:,} DOFs",
            fill=INK2, font=f_text)
    x = 24
    for letter, text, col in legend:
        dr.rectangle((x, 72, x + 16, 88), fill=col)
        dr.text((x + 24, 66), f"{letter}  {text}", fill=INK, font=f_text)
        x += 24 + dr.textlength(f"{letter}  {text}", font=f_text) + 36
    y = head + img.shape[0] + 8
    cols = (24, 250, 470, 760, 900)
    for cx, h in zip(cols, ("reference QoI", f"on L{level}", "gold (tolerance)", "difference")):
        dr.text((cx, y), h, fill=MUTED, font=f_small)
    for i, r in enumerate(refs):
        yy = y + 30 + i * row
        v = vals[r.qoi]
        cells = (r.qoi, _fmt_qoi(r.qoi, v), f"{_fmt_qoi(r.qoi, r.value)} (± {r.tolerance:.1%})", f"{v / r.value - 1:+.1%}")
        for cx, t in zip(cols, cells):
            dr.text((cx, yy), t, fill=INK, font=f_text)
    note = f"Colour range clipped at the 99th percentile; peak {vm.max():.0f} MPa"
    if "vm_max" not in {r.qoi for r in refs}:
        note += " (vm_max diverges with refinement, so it is not a reference)"
    dr.text((24, y + 40 + row * len(refs)), note + ". Supports outlined black, loads orange.", fill=INK2,
            font=f_small)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    return out


def contact_sheet(pngs: list[Path], out: Path, title: str, cols: int = 2, width: int = 1100) -> Path:
    from PIL import Image, ImageDraw

    tiles = [Image.open(p).convert("RGB") for p in pngs]
    tiles = [t.resize((width, round(width * t.height / t.width)), Image.LANCZOS) for t in tiles]
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    band = 70
    sheet = Image.new("RGB", (cols * width, band + sum(max(t.height for t in r) for r in rows)), SURFACE)
    y = band
    for r in rows:                        # tiles keep their aspect: the QoI tables differ in length
        for j, t in enumerate(r):
            sheet.paste(t, (j * width, y))
        y += max(t.height for t in r)
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), title, fill=INK, font=_font(28))
    sheet.save(out)
    return out


def cmd_render(a) -> int:
    from feacore.runstore import RunStore
    from feacore.solver import solve

    store = None if a.no_store else RunStore(a.runs_dir)
    pngs = []
    for p in _selected(a):
        mesh = p.mesh_file(a.level)
        if mesh is None:
            print(f"{p.id}: no L{a.level} mesh imported (import the archive with its meshes)")
            continue
        t0 = time.perf_counter()
        res = solve(p.case(mesh), backend="skfem")
        msg = f"{p.id} L{a.level}: {res.info.n_dofs} dofs, {time.perf_counter() - t0:.1f}s"
        if store is not None:
            h = next((r["h"] for r in p.study()["runs"] if r["level"] == a.level), None)
            rid = store.record_result(res, mesh_path=mesh, problem_id=p.id, agent_id=f"gold:L{a.level}",
                                      level=a.level, h_nominal=h, source="gold", copy_mesh=False,
                                      meta={"source": "gold", "part": p.family, "mesh_level": a.level})
            msg += f", run {rid}"
        if a.png:
            pngs.append(render_png(p, res, mesh, Path(a.png) / f"{p.id}_L{a.level}.png", a.level))
            msg += f", {pngs[-1]}"
        print(msg, flush=True)
    if len(pngs) > 1:
        sheet = contact_sheet(pngs, Path(a.png) / f"gold_L{a.level}.png",
                              f"Gold benchmark solutions on the L{a.level} reference meshes: von Mises on the "
                              f"deformed shape (each panel has its own scale)")
        print(sheet)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", help="gold root (default FEACORE_GOLD_DIR or benchmarks/gold)")
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("import", help="import goldgen archives (.tgz or extracted directories)")
    s.add_argument("sources", nargs="+")
    s.add_argument("--no-meshes", action="store_true", help="statements and references only")
    s.set_defaults(func=cmd_import)
    s = sub.add_parser("list", help="sets with their references (evaluator-only information)")
    s.add_argument("ids", nargs="*", help="set or part ids (default: all)")
    s.set_defaults(func=cmd_list)
    s = sub.add_parser("verify", help="re-solve on a stored level and compare with the study")
    s.add_argument("ids", nargs="*")
    s.add_argument("--level", type=int, default=0)
    s.set_defaults(func=cmd_verify)
    s = sub.add_parser("render", help="re-solve on a stored level; record runs for the UI, PNGs")
    s.add_argument("ids", nargs="*")
    s.add_argument("--level", type=int, default=1)
    s.add_argument("--runs-dir", help="run store root (default FEACORE_RUNS_DIR or ./runs)")
    s.add_argument("--no-store", action="store_true", help="do not record runs")
    s.add_argument("--png", help="directory for PNG renders")
    s.set_defaults(func=cmd_render)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
