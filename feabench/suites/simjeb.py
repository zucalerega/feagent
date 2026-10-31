"""SimJEB: Simulated Jet Engine Bracket Dataset (Whalen, Beyene & Mueller, Computer Graphics Forum 2021).

381 cleaned brackets from the GE / GrabCAD Jet Engine Bracket Challenge, each simulated in
OptiStruct under the challenge's four separate load cases (8000 lbf vertical, 8500 lbf
horizontal, 9500 lbf at 42 deg from vertical, 5000 lbf-in torsion; load through the 3/4"
clevis pin at interface 1, bolt interfaces 2-5 fixed; Ti-6Al-4V).  Per bracket the dataset
provides clean CAD (.stp), the OptiStruct model (.fem), a tet mesh (.vtk), a surface mesh
(.obj) and results (.csv); each type is a separate zip, plus a metadata file with summary
statistics and train/test splits.  Hosted on Harvard Dataverse, doi:10.7910/DVN/XFUWJG.

STATUS (2026-10-31): dataverse.harvard.edu is blocked by the network policy of the session
that wrote this module, so nothing here has been run against the real files.  Verified by
unit tests on synthetic inputs: the Nastran bulk-data parser, RBE2/RBE3 handling and the
.fem -> CalculiX conversion.  NOT verified: archive/file naming inside the zips and the
results .csv column layout -- ``read_results_csv`` detects columns heuristically and fails
loudly (listing the columns it saw) instead of guessing.  Run ``feabench simjeb fetch`` then
``python -m feabench.suites.simjeb inspect <dir>`` on first access and adjust ``CSV_HINTS``.

Licensing: per the SimJEB paper the data is under the Open Data Commons Attribution License,
while the CAD originates from the GrabCAD challenge (non-commercial use).  Keep downloaded
files out of git (``data/`` is git-ignored); commit only derived task specs if licensing allows.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SERVER = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/XFUWJG"
LBF = 4.4482216152605  # N
INCH = 25.4  # mm
LOAD_CASES_SPEC = {  # GE challenge statement (applied separately, through the pin at interface 1)
    "vertical": {"force_lbf": 8000.0, "description": "8000 lbf vertical, up"},
    "horizontal": {"force_lbf": 8500.0, "description": "8500 lbf horizontal, out (away from the mounting face)"},
    "diagonal": {"force_lbf": 9500.0, "description": "9500 lbf at 42 deg from vertical"},
    "torsion": {"moment_lbf_in": 5000.0, "description": "5000 lbf-in torsion"},
}

# ----------------------------------------------------------------------------- Dataverse download


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def list_files(server: str = SERVER, doi: str = DOI, version: str = ":latest-published") -> list[dict]:
    """File listing (id, name, size, md5) of the dataset via the Dataverse native API."""
    url = f"{server}/api/datasets/:persistentId/versions/{version}/files?persistentId={doi}"
    data = _get_json(url)["data"]
    out = []
    for f in data:
        df = f["dataFile"]
        tabular = "originalFileFormat" in df  # ingested tabular file: fetch the original bytes
        out.append({
            "id": df["id"],
            "name": df.get("originalFileName") if tabular and df.get("originalFileName") else df["filename"],
            "size": df.get("originalFileSize", df.get("filesize")) if tabular else df.get("filesize"),
            "md5": df.get("md5") or (df.get("checksum") or {}).get("value"),
            "checksum_type": (df.get("checksum") or {}).get("type", "MD5"),
            "tabular": tabular,
            "directory": f.get("directoryLabel", ""),
        })
    return out


def _download(server: str, f: dict, dest: Path) -> Path:
    url = f"{server}/api/access/datafile/{f['id']}" + ("?format=original" if f["tabular"] else "")
    target = dest / f["directory"] / f["name"] if f["directory"] else dest / f["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    h = hashlib.md5()
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as out:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if f.get("size"):
                print(f"\r  {f['name']}: {done / 1e6:.1f}/{f['size'] / 1e6:.1f} MB", end="", file=sys.stderr)
    print("", file=sys.stderr)
    if f.get("md5") and f.get("checksum_type", "MD5").upper() == "MD5" and h.hexdigest() != f["md5"]:
        part.unlink()
        raise IOError(f"MD5 mismatch for {f['name']}: got {h.hexdigest()}, expected {f['md5']}")
    part.replace(target)
    return target


def fetch(dest: str | Path, include: list[str] | None = None, dry_run: bool = False,
          server: str = SERVER, doi: str = DOI) -> list[dict]:
    """Download dataset files whose names match ``include`` glob patterns (default: all)."""
    dest = Path(dest)
    files = list_files(server, doi)
    sel = [f for f in files if not include or any(fnmatch.fnmatch(f["name"], p) for p in include)]
    total = sum(f.get("size") or 0 for f in sel)
    print(f"{len(sel)}/{len(files)} files selected, {total / 1e9:.2f} GB", file=sys.stderr)
    for f in sel:
        print(f"  {f['id']:>10}  {(f.get('size') or 0) / 1e6:10.1f} MB  {f['name']}", file=sys.stderr)
    if dry_run:
        return sel
    dest.mkdir(parents=True, exist_ok=True)
    for f in sel:
        target = dest / f["directory"] / f["name"] if f["directory"] else dest / f["name"]
        if target.exists() and f.get("size") and target.stat().st_size == f["size"]:
            continue
        _download(server, f, dest)
    (dest / "manifest.json").write_text(json.dumps({"server": server, "doi": doi, "files": files}, indent=1))
    return sel


def inspect(dest: str | Path) -> dict:
    """List the members of every downloaded zip (first thing to look at after a fetch)."""
    out = {}
    for z in sorted(Path(dest).rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            out[z.name] = {"n": len(names), "first": names[:10],
                           "extensions": sorted({Path(n).suffix for n in names})}
    return out


def extract(dest: str | Path, pattern: str = "*.zip") -> Path:
    dest = Path(dest)
    root = dest / "extracted"
    for z in sorted(dest.glob(pattern)):
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():  # guard against path traversal in member names
                p = (root / z.stem / m.filename).resolve()
                if not str(p).startswith(str((root / z.stem).resolve())):
                    raise IOError(f"unsafe path in {z.name}: {m.filename}")
            zf.extractall(root / z.stem)
    return root


# ----------------------------------------------------------------------------- Nastran / OptiStruct bulk data

_NASTRAN_EXP = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$")


def nastran_float(s: str) -> float:
    s = s.strip().replace("D", "E").replace("d", "e")
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        m = _NASTRAN_EXP.match(s)  # 1.5-3 -> 1.5e-3, 2.+5 -> 2e5
        if m:
            return float(f"{m.group(1)}e{m.group(2)}")
        raise


def _int(s: str) -> int | None:
    s = s.strip()
    return int(s) if s and re.fullmatch(r"[+-]?\d+", s) else None


def iter_bulk_cards(text: str):
    """Yield (NAME, [fields...]) for each bulk-data card; handles small/large/free field + continuations."""
    in_bulk = "BEGIN BULK" not in text.upper()
    name, fields = None, []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not in_bulk:
            if line.upper().startswith("BEGIN BULK"):
                in_bulk = True
            continue
        if not line.strip() or line.lstrip().startswith("$"):
            continue
        if line.upper().startswith("ENDDATA"):
            break
        line = line.split("$")[0]
        if "," in line[:80]:
            toks = [t.strip() for t in line.split(",")]
            head = toks[0]
            if head == "" or head.startswith(("+", "*")):
                fields += toks[1:]
                continue
            if name:
                yield name, fields
            name, fields = head.upper().rstrip("*"), toks[1:]
            continue
        head = line[:8].strip()
        if head.startswith("*") and name:  # large-field continuation
            fields += [line[8 + 16 * i: 24 + 16 * i].strip() for i in range(4)]
            continue
        if (head == "" or head.startswith("+")) and name:  # small-field continuation
            fields += [line[8 + 8 * i: 16 + 8 * i].strip() for i in range(8)]
            continue
        if name:
            yield name, fields
        if head.endswith("*"):
            name = head[:-1].upper()
            fields = [line[8 + 16 * i: 24 + 16 * i].strip() for i in range(4)]
        else:
            name = head.upper()
            fields = [line[8 + 8 * i: 16 + 8 * i].strip() for i in range(8)]
    if name:
        yield name, fields


def parse_case_control(text: str) -> list[dict]:
    """SUBCASE blocks (id, LABEL/SUBTITLE, LOAD, SPC) before BEGIN BULK."""
    head = text.split("BEGIN BULK")[0] if "BEGIN BULK" in text.upper() else ""
    subs, cur, glob = [], None, {}
    for ln in head.splitlines():
        s = ln.split("$")[0].strip()
        if not s:
            continue
        m = re.match(r"SUBCASE\s+(\d+)", s, re.I)
        if m:
            cur = {"id": int(m.group(1))}
            subs.append(cur)
            continue
        m = re.match(r"(LABEL|SUBTITLE|LOAD|SPC)\s*=\s*(.*)", s, re.I)
        if m:
            k, v = m.group(1).upper(), m.group(2).strip()
            (cur if cur is not None else glob)[k] = int(v) if v.isdigit() else v
    for sc in subs:
        for k, v in glob.items():
            sc.setdefault(k, v)
    if not subs and glob:
        subs.append({"id": 1, **glob})
    return subs


@dataclass
class FemModel:
    nodes: dict[int, np.ndarray] = field(default_factory=dict)
    tets: dict[int, list[int]] = field(default_factory=dict)  # eid -> node ids (4 or 10, Nastran order = ccx order)
    tet_pid: dict[int, int] = field(default_factory=dict)
    rbe2: list[dict] = field(default_factory=list)  # {eid, gn, cm, gm: [...]}
    rbe3: list[dict] = field(default_factory=list)  # {eid, ref, refc, groups: [(w, c, [g...])]}
    forces: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)  # sid -> [(grid, F)]
    moments: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    spcs: dict[int, list[tuple[int, str, float]]] = field(default_factory=dict)  # sid -> [(grid, dofs, value)]
    load_combos: dict[int, tuple[float, list[tuple[float, int]]]] = field(default_factory=dict)
    materials: dict[int, dict] = field(default_factory=dict)
    properties: dict[int, dict] = field(default_factory=dict)
    subcases: list[dict] = field(default_factory=list)
    unsupported: dict[str, int] = field(default_factory=dict)

    def loads_for(self, sid: int) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
        """Forces and moments of load set ``sid`` (expanding LOAD combinations)."""
        if sid in self.load_combos:
            s, parts = self.load_combos[sid]
            F, M = [], []
            for si, li in parts:
                f, m = self.loads_for(li)
                F += [(g, s * si * v) for g, v in f]
                M += [(g, s * si * v) for g, v in m]
            return F, M
        return list(self.forces.get(sid, [])), list(self.moments.get(sid, []))


def parse_fem(text: str) -> FemModel:
    m = FemModel(subcases=parse_case_control(text))
    for name, f in iter_bulk_cards(text):
        f = f + [""] * 20
        if name == "GRID":
            if _int(f[1]) not in (None, 0) or _int(f[5]) not in (None, 0):
                m.unsupported["GRID with CP/CD"] = m.unsupported.get("GRID with CP/CD", 0) + 1
            m.nodes[int(f[0])] = np.array([nastran_float(f[2]), nastran_float(f[3]), nastran_float(f[4])])
        elif name == "CTETRA":
            ids = [_int(x) for x in f[2:12]]
            nodes = [i for i in ids if i]
            m.tets[int(f[0])] = nodes[:10] if len(nodes) >= 10 else nodes[:4]
            m.tet_pid[int(f[0])] = int(f[1])
        elif name == "RBE2":
            gm, alpha = [], None
            for x in f[3:]:
                if not x:
                    continue
                if _int(x) is None:
                    alpha = nastran_float(x)
                    break
                gm.append(int(x))
            m.rbe2.append({"eid": int(f[0]), "gn": int(f[1]), "cm": f[2].strip(), "gm": gm, "alpha": alpha})
        elif name == "RBE3":
            groups, i, toks = [], 4, [x for x in f]
            while i < len(toks):
                x = toks[i].strip()
                if not x:
                    i += 1
                    continue
                if x.upper() in ("UM", "ALPHA", "STRESS"):
                    break
                if _int(x) is None:  # a weight starts a new group: WT, C, G...
                    w, c = nastran_float(x), toks[i + 1].strip()
                    i += 2
                    gs = []
                    while i < len(toks) and toks[i].strip() and _int(toks[i]) is not None:
                        gs.append(int(toks[i]))
                        i += 1
                    groups.append((w, c, gs))
                else:
                    i += 1
            m.rbe3.append({"eid": int(f[0]), "ref": int(f[2]), "refc": f[3].strip(), "groups": groups})
        elif name in ("FORCE", "MOMENT"):
            sid, g, cid = int(f[0]), int(f[1]), _int(f[2]) or 0
            if cid:
                m.unsupported[f"{name} with CID"] = m.unsupported.get(f"{name} with CID", 0) + 1
            vec = nastran_float(f[3]) * np.array([nastran_float(f[4]), nastran_float(f[5]), nastran_float(f[6])])
            (m.forces if name == "FORCE" else m.moments).setdefault(sid, []).append((g, vec))
        elif name == "SPC1":
            sid, c = int(f[0]), f[1].strip()
            ids, toks, k = [], [x for x in f[2:] if x], 0
            while k < len(toks):
                if k + 2 < len(toks) and toks[k + 1].upper() == "THRU":
                    ids += list(range(int(toks[k]), int(toks[k + 2]) + 1))
                    k += 3
                else:
                    if _int(toks[k]) is not None:
                        ids.append(int(toks[k]))
                    k += 1
            m.spcs.setdefault(sid, []).extend((g, c, 0.0) for g in ids)
        elif name == "SPC":
            sid = int(f[0])
            for k in (1, 4):
                if _int(f[k]):
                    m.spcs.setdefault(sid, []).append((int(f[k]), f[k + 1].strip(), nastran_float(f[k + 2])))
        elif name == "LOAD":
            sid, s = int(f[0]), nastran_float(f[1])
            parts, toks = [], [x for x in f[2:] if x]
            for k in range(0, len(toks) - 1, 2):
                parts.append((nastran_float(toks[k]), int(toks[k + 1])))
            m.load_combos[sid] = (s, parts)
        elif name == "MAT1":
            E, G, nu = nastran_float(f[1]), nastran_float(f[2]), nastran_float(f[3])
            if not nu and E and G:
                nu = E / (2 * G) - 1
            m.materials[int(f[0])] = {"E": E, "G": G, "nu": nu, "rho": nastran_float(f[4])}
        elif name == "PSOLID":
            m.properties[int(f[0])] = {"mid": int(f[1])}
        elif name in ("CTRIA3", "CTRIA6", "CQUAD4", "PSHELL", "CORD2R", "CORD2C", "PARAM", "EIGRL", "DESVAR"):
            m.unsupported[name] = m.unsupported.get(name, 0) + 1
    return m


# ----------------------------------------------------------------------------- RBE3 load distribution

def rbe3_nodal_forces(coords: dict[int, np.ndarray], ref_xyz: np.ndarray, groups: list,
                      force: np.ndarray, moment: np.ndarray) -> dict[int, np.ndarray]:
    """Equivalent nodal forces of an RBE3 with translational dependent DOFs (C = 123).

    f_i = w_i F / W + w_i (Phi^-1 M_cg) x r_i,  r_i = x_i - x_cg,  Phi = sum w_i (|r_i|^2 I - r_i r_i^T),
    M_cg = M + (x_ref - x_cg) x F.  This reproduces the resultant force and moment exactly and is
    the weighted least-squares distribution an RBE3 performs for a load at its reference node.
    """
    ids, w = [], []
    for wt, c, gs in groups:
        if c and set(c) - set("123"):
            raise NotImplementedError("RBE3 groups with rotational components are not supported")
        for g in gs:
            ids.append(g)
            w.append(wt)
    w = np.asarray(w, float)
    X = np.array([coords[g] for g in ids])
    W = w.sum()
    xcg = (w[:, None] * X).sum(0) / W
    r = X - xcg
    Mcg = np.asarray(moment, float) + np.cross(np.asarray(ref_xyz, float) - xcg, np.asarray(force, float))
    Phi = sum(wi * (ri @ ri * np.eye(3) - np.outer(ri, ri)) for wi, ri in zip(w, r))
    theta = np.linalg.lstsq(Phi, Mcg, rcond=None)[0]
    out: dict[int, np.ndarray] = {}
    for g, wi, ri in zip(ids, w, r):
        out[g] = out.get(g, 0.0) + wi * np.asarray(force, float) / W + wi * np.cross(theta, ri)
    return out


# ----------------------------------------------------------------------------- .fem -> CalculiX

def fem_to_ccx(model: FemModel, subcase: dict, path: str | Path) -> dict:
    """Write a ccx deck reproducing one OptiStruct subcase (linear static).

    * RBE2 whose independent node is fully constrained by the subcase SPC -> dependent nodes
      fixed in the RBE2's translational components; otherwise -> *RIGID BODY (loads on GN go to
      its reference node / rotation node).
    * RBE3 loaded at its reference node -> explicit nodal forces (``rbe3_nodal_forces``).
    """
    path = Path(path)
    spc = {}
    for g, c, v in model.spcs.get(subcase.get("SPC", -1), []):
        spc.setdefault(g, set()).update(c)
    forces, moments = model.loads_for(subcase.get("LOAD", -1))
    fixed_dependent, rigid = {}, []
    for r in model.rbe2:
        trans = set(r["cm"]) & set("123")
        if spc.get(r["gn"], set()) >= set("123456") or spc.get(r["gn"], set()) >= trans | set("456"):
            for g in r["gm"]:
                fixed_dependent.setdefault(g, set()).update(trans)
        else:
            rigid.append(r)
    nodal: dict[int, np.ndarray] = {}
    used_ref = set()
    for r in model.rbe3:
        F = sum((v for g, v in forces if g == r["ref"]), np.zeros(3))
        M = sum((v for g, v in moments if g == r["ref"]), np.zeros(3))
        if not np.any(F) and not np.any(M):
            continue
        used_ref.add(r["ref"])
        for g, v in rbe3_nodal_forces(model.nodes, model.nodes[r["ref"]], r["groups"], F, M).items():
            nodal[g] = nodal.get(g, 0.0) + v
    rigid_gn = {r["gn"] for r in rigid}
    for g, v in forces:
        if g not in used_ref and g not in rigid_gn:
            nodal[g] = nodal.get(g, 0.0) + v
    leftover_moments = [(g, v) for g, v in moments if g not in used_ref and g not in rigid_gn]
    if leftover_moments:
        raise NotImplementedError(f"moments on plain grids are not representable on solids: {leftover_moments[:3]}")

    tet_nodes = sorted({n for t in model.tets.values() for n in t})
    order = len(next(iter(model.tets.values())))
    out = ["*HEADING", f"SimJEB subcase {subcase.get('id')} {subcase.get('LABEL', '')}", "*NODE, NSET=NALL"]
    extra = sorted(rigid_gn - set(tet_nodes))
    for nid in tet_nodes + extra:
        x, y, z = model.nodes[nid]
        out.append(f"{nid}, {x:.12g}, {y:.12g}, {z:.12g}")
    out.append(f"*ELEMENT, TYPE={'C3D10' if order == 10 else 'C3D4'}, ELSET=EALL")
    out += [f"{eid}, " + ", ".join(map(str, conn)) for eid, conn in sorted(model.tets.items())]
    mats = list(model.materials.values())
    if len(mats) != 1:
        raise NotImplementedError(f"expected one MAT1, found {len(mats)}")
    mat = mats[0]
    out += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"{mat['E']:.12g}, {mat['nu']:.12g}"]
    if mat.get("rho"):
        out += ["*DENSITY", f"{mat['rho']:.12g}"]
    out.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT")
    for k, r in enumerate(rigid):
        out += [f"*NSET, NSET=RB{k}"] + [", ".join(map(str, r["gm"][i:i + 16])) for i in range(0, len(r["gm"]), 16)]
        out.append(f"*RIGID BODY, NSET=RB{k}, REF NODE={r['gn']}")
    out.append("*BOUNDARY")
    for g, dofs in sorted(fixed_dependent.items()):
        for d in sorted(dofs):
            out.append(f"{g}, {d}, {d}, 0.0")
    for g, c in sorted(spc.items()):
        if g in model.nodes and (g in set(tet_nodes) or g in rigid_gn):
            for d in sorted(set(c) & set("123")):
                out.append(f"{g}, {d}, {d}, 0.0")
    out += ["*STEP", "*STATIC", "*CLOAD"]
    for g, v in sorted(nodal.items()):
        for d in range(3):
            if v[d]:
                out.append(f"{g}, {d + 1}, {v[d]:.12g}")
    for g, v in forces:
        if g in rigid_gn:
            for d in range(3):
                if v[d]:
                    out.append(f"{g}, {d + 1}, {v[d]:.12g}")
    out += ["*NODE FILE", "U, RF", "*EL FILE", "S", "*END STEP"]
    path.write_text("\n".join(out) + "\n")
    return {"n_nodes": len(tet_nodes), "n_tets": len(model.tets), "fixed_nodes": len(fixed_dependent),
            "rigid_bodies": len(rigid), "loaded_nodes": len(nodal),
            "resultant": np.sum(list(nodal.values()), axis=0).tolist() if nodal else [0, 0, 0]}


# ----------------------------------------------------------------------------- interfaces -> task regions

def fit_cylinder(P: np.ndarray, n_dirs: int = 2000) -> dict:
    """Least-squares cylinder through points (axis by direction search + Kasa circle fit)."""
    P = np.asarray(P, float)
    c0 = P.mean(0)
    Q = P - c0
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - i / n_dirs)  # hemisphere
    th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    def circle(d):
        d = d / np.linalg.norm(d)
        u = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        x, y = Q @ u, Q @ v
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
        R = math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0))
        res = np.sqrt((x - sol[0]) ** 2 + (y - sol[1]) ** 2) - R
        return float(np.sqrt(np.mean(res ** 2))), d, c0 + sol[0] * u + sol[1] * v, R

    best = min((circle(d) for d in dirs), key=lambda t: t[0])
    from scipy.optimize import minimize

    def angles(d):
        return [math.acos(max(-1.0, min(1.0, d[2]))), math.atan2(d[1], d[0])]

    def vec(a):
        return np.array([math.sin(a[0]) * math.cos(a[1]), math.sin(a[0]) * math.sin(a[1]), math.cos(a[0])])

    opt = minimize(lambda a: circle(vec(a))[0], angles(best[1]), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 2000})
    refined = circle(vec(opt.x))
    rms, d, c, R = refined if refined[0] <= best[0] else best
    s = (P - c) @ d
    return {"point": c.tolist(), "axis": d.tolist(), "radius": R, "range": [float(s.min()), float(s.max())],
            "rms": rms}


def interface_regions(model: FemModel) -> dict:
    """Bolt (RBE2) and load (RBE3) interfaces as fitted cylinders + raw node coordinates."""
    out = {"bolts": [], "load": []}
    for r in model.rbe2:
        P = np.array([model.nodes[g] for g in r["gm"]])
        out["bolts"].append({"rbe2": r["eid"], "independent_node": model.nodes[r["gn"]].tolist(),
                             "cylinder": fit_cylinder(P), "points": P})
    for r in model.rbe3:
        P = np.array([model.nodes[g] for w, c, gs in r["groups"] for g in gs])
        out["load"].append({"rbe3": r["eid"], "reference_point": model.nodes[r["ref"]].tolist(), "points": P})
    return out


# ----------------------------------------------------------------------------- results csv

CSV_HINTS = {  # lower-case substrings used to recognise columns; adjust after inspecting real files
    "node": ["node", "nid", "grid", "id"],
    "x": ["x"], "y": ["y"], "z": ["z"],
    "disp": ["disp", "u_", "displacement"],
    "vm": ["vm", "mises", "von"],
    "cases": {"vertical": ["vert", "lc1", "case1", "_1"], "horizontal": ["horiz", "lc2", "case2", "_2"],
              "diagonal": ["diag", "lc3", "case3", "_3"], "torsion": ["tors", "lc4", "case4", "_4"]},
}


def read_results_csv(path: str | Path) -> dict[str, np.ndarray]:
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(x) if x.strip() else np.nan for x in r] for r in rows[1:] if r], float)
    return {h: data[:, i] for i, h in enumerate(header)}


def pick_columns(cols: dict[str, np.ndarray], load_case: str) -> dict[str, str]:
    """Map semantic fields to CSV columns for one load case, or raise with what was found."""
    names = {c.lower(): c for c in cols}
    case_keys = CSV_HINTS["cases"][load_case]
    found = {}
    for c_low, c in names.items():
        in_case = any(k in c_low for k in case_keys)
        if any(k in c_low for k in CSV_HINTS["vm"]) and in_case:
            found["vm"] = c
        if any(k in c_low for k in CSV_HINTS["disp"]) and in_case:
            for comp in ("x", "y", "z", "mag"):
                if c_low.endswith(comp) or f"_{comp}" in c_low:
                    found[f"disp_{comp}"] = c
    if not found:
        raise KeyError(f"cannot identify result columns for load case {load_case!r}; columns are "
                       f"{list(cols)}. Update feabench.suites.simjeb.CSV_HINTS after inspecting a file.")
    return found


def build_tasks(dest: str | Path, out: str | Path, ids: list[int] | None = None,
                load_cases: list[str] | None = None) -> list[Path]:  # pragma: no cover - needs real data
    """Create benchmark cases from downloaded + extracted SimJEB files.

    Requires, per bracket id: the .stp (geometry, agent-visible), the .fem (interfaces, loads,
    material) and the results .csv (reference fields).  Interfaces are exported to the task as
    fitted cylinders plus the interface node coordinates; the reference is the SimJEB nodal
    displacement field (relative-L2 QoI) plus max |U| and the 99.9th-percentile von Mises stress.
    """
    root = Path(dest) / "extracted"
    if not root.exists():
        raise FileNotFoundError(f"{root} missing: run `feabench simjeb fetch` and extract() first")
    stp = {int(re.findall(r"\d+", p.stem)[0]): p for p in root.rglob("*.stp") if re.findall(r"\d+", p.stem)}
    fem = {int(re.findall(r"\d+", p.stem)[0]): p for p in root.rglob("*.fem") if re.findall(r"\d+", p.stem)}
    csvs = {int(re.findall(r"\d+", p.stem)[0]): p for p in root.rglob("*.csv") if re.findall(r"\d+", p.stem)}
    made = []
    for bid in sorted(set(stp) & set(fem) & set(csvs)):
        if ids and bid not in ids:
            continue
        model = parse_fem(fem[bid].read_text(errors="replace"))
        inter = interface_regions(model)
        cols = read_results_csv(csvs[bid])
        for lc in load_cases or list(LOAD_CASES_SPEC):
            sub = next((s for s in model.subcases if lc[:4] in str(s.get("LABEL", "")).lower()), None)
            if sub is None:
                raise KeyError(f"bracket {bid}: no subcase label matching {lc!r} in {model.subcases}")
            colmap = pick_columns(cols, lc)
            case = Path(out) / f"{bid}_{lc}"
            (case / "task").mkdir(parents=True, exist_ok=True)
            (case / "reference").mkdir(parents=True, exist_ok=True)
            shutil.copy(stp[bid], case / "task" / "geometry.step")
            np.savez(case / "task" / "interfaces.npz",
                     **{f"bolt{i}": b["points"] for i, b in enumerate(inter["bolts"])},
                     **{f"load{i}": l["points"] for i, l in enumerate(inter["load"])})
            forces, moments = model.loads_for(sub["LOAD"])
            task = {"id": f"simjeb/{bid}_{lc}", "title": f"SimJEB bracket {bid}: {lc} load case",
                    "description": LOAD_CASES_SPEC[lc]["description"],
                    "geometry": {"file": "geometry.step", "format": "step"},
                    "materials": [{"name": "Ti-6Al-4V", **list(model.materials.values())[0]}],
                    "interfaces": {"bolts": [{k: v for k, v in b.items() if k != "points"} for b in inter["bolts"]],
                                   "load": [{k: v for k, v in l.items() if k != "points"} for l in inter["load"]]},
                    "loads": {"forces": [(g, v.tolist()) for g, v in forces],
                              "moments": [(g, v.tolist()) for g, v in moments]},
                    "analysis": {"type": "static"}}
            (case / "task" / "task.json").write_text(json.dumps(task, indent=1, default=float))
            node_col = next((c for c in cols if c.lower() in CSV_HINTS["node"]), None)
            ref_pts = np.array([model.nodes[int(n)] for n in cols[node_col]]) if node_col else None
            disp = [colmap.get(f"disp_{c}") for c in "xyz"]
            qois = []
            if ref_pts is not None and all(disp):
                U = np.c_[cols[disp[0]], cols[disp[1]], cols[disp[2]]]
                np.savez(case / "reference" / "field.npz", points=ref_pts, values=U)
                qois.append({"id": "disp_field", "kind": "field_error", "field": "U", "points": "field.npz",
                             "reference": 0.0, "tolerance": {"abs": 0.05}, "reference_type": "simjeb_optistruct"})
                qois.append({"id": "max_disp", "kind": "max", "field": "U", "component": "mag",
                             "reference": float(np.linalg.norm(U, axis=1).max()), "tolerance": {"rel": 0.03},
                             "reference_type": "simjeb_optistruct"})
            if "vm" in colmap:
                qois.append({"id": "vm_p999", "kind": "max", "field": "S", "component": "mises", "percentile": 99.9,
                             "reference": float(np.nanpercentile(cols[colmap["vm"]], 99.9)),
                             "tolerance": {"rel": 0.10}, "reference_type": "simjeb_optistruct"})
            (case / "reference" / "reference.json").write_text(json.dumps(
                {"id": task["id"], "qois": qois, "column_map": colmap}, indent=1))
            made.append(case)
    return made


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        print(json.dumps(inspect(sys.argv[2]), indent=1))
    else:
        print("usage: python -m feabench.suites.simjeb inspect <download-dir>")
