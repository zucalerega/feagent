"""code_aster validation tests (``astest/`` in https://gitlab.com/codeaster/src, GPL-3.0).

Each test variant ``<name>.export`` lists its command file (``.comm``, a Python script) and
mesh files.  Reference values live inside the command file in ``TEST_RESU`` /
``TEST_TABLE`` / ``TEST_FONCTION`` calls::

    _F(REFERENCE="ANALYTIQUE", GROUP_NO="A", NOM_CHAM="DEPL", NOM_CMP="DX",
       VALE_REFE=1.23e-4, VALE_CALC=1.2297e-4, PRECISION=0.01)

* ``VALE_REFE`` is the reference value, ``REFERENCE`` says where it comes from:
  ``ANALYTIQUE`` (closed form), ``SOURCE_EXTERNE`` (literature / other codes / experiment),
  ``AUTRE_ASTER`` (another code_aster model), ``NON_DEFINI``.
* ``VALE_CALC`` is what code_aster computes with the test's mesh (non-regression value).
* ``PRECISION`` is the accepted deviation of VALE_CALC from VALE_REFE (default 1e-3,
  relative unless ``CRITERE="ABSOLU"``).  code_aster's CI runs every test on every release,
  so |VALE_CALC - VALE_REFE| <= PRECISION has been demonstrated by upstream.

We never execute command files: they are parsed with :mod:`ast`, and constant expressions are
evaluated by a small whitelist evaluator (numbers, arithmetic, ``math`` functions, and
module-level constant assignments).  Anything else is recorded as an unresolved expression.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import warnings
from pathlib import Path

UPSTREAM = {
    "repo": "https://gitlab.com/codeaster/src.git",
    "commit": "7cb48262af47548277f4824ea47f1cc90848f057",
    "commit_date": "2026-10-21",
    "path": "astest",
    "license": "GPL-3.0-or-later",
}

INDEPENDENT = {"ANALYTIQUE", "SOURCE_EXTERNE"}
TEST_COMMANDS = {"TEST_RESU", "TEST_TABLE", "TEST_FONCTION"}
ENTRY_KEYS = [
    "REFERENCE", "VALE_REFE", "VALE_REFE_I", "VALE_REFE_C", "VALE_CALC", "VALE_CALC_I", "VALE_CALC_C",
    "PRECISION", "CRITERE", "ORDRE_GRANDEUR", "TOLE_MACHINE", "NOM_CHAM", "NOM_CMP", "NOM_VARI",
    "GROUP_NO", "NOEUD", "GROUP_MA", "MAILLE", "POINT", "SOUS_POINT", "NUME_ORDRE", "INST", "FREQ",
    "NUME_MODE", "PARA", "RESULTAT", "CHAM_GD", "TYPE_TEST", "NOM_PARA", "TABLE", "LEGENDE",
    "VALE_PARA", "FONCTION", "VALE_ABS", "FILTRE",
]
SOLVER_OPERATORS = {
    "MECA_STATIQUE", "STAT_NON_LINE", "MACRO_ELAS_MULT", "CALC_MODES", "MODE_ITER_SIMULT",
    "MODE_ITER_INV", "DYNA_VIBRA", "DYNA_NON_LINE", "DYNA_LINE", "THER_LINEAIRE", "THER_NON_LINE",
    "CALC_G", "CALC_G_XFEM", "POST_K1_K2_K3", "CALC_EUROPLEXUS", "MECA_NON_LINE", "CALC_ERREUR",
}

# ---------------------------------------------------------------------------- safe evaluation

_MATH = {k: getattr(math, k) for k in (
    "sqrt", "cos", "sin", "tan", "acos", "asin", "atan", "atan2", "exp", "log", "log10", "pow",
    "cosh", "sinh", "tanh", "radians", "degrees", "pi", "e", "fabs", "hypot")}
_MATH.update(abs=abs, min=min, max=max, float=float, int=int, pi=math.pi)
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
           ast.FloorDiv: lambda a, b: a // b}


class Unresolved(Exception):
    pass


def safe_eval(node: ast.AST, env: dict, depth: int = 0):
    """Evaluate a constant expression; raise Unresolved for anything else."""
    if depth > 50:
        raise Unresolved("too deep")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [safe_eval(e, env, depth + 1) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = safe_eval(node.operand, env, depth + 1)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = safe_eval(node.left, env, depth + 1), safe_eval(node.right, env, depth + 1)
        if not all(isinstance(x, (int, float)) for x in (a, b)):
            raise Unresolved("non-numeric binop")
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise Unresolved("large power")
        return _BINOPS[type(node.op)](a, b)
    if isinstance(node, ast.Name):
        if node.id in env:
            return safe_eval(env[node.id], env, depth + 1) if isinstance(env[node.id], ast.AST) else env[node.id]
        if node.id in _MATH:
            return _MATH[node.id]
        raise Unresolved(f"name {node.id}")
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("math", "numpy", "np"):
        if node.attr in _MATH:
            return _MATH[node.attr]
        raise Unresolved(f"attr {node.attr}")
    if isinstance(node, ast.Call) and not node.keywords:
        f = safe_eval(node.func, env, depth + 1)
        if f in _MATH.values():
            return f(*[safe_eval(a, env, depth + 1) for a in node.args])
    raise Unresolved(type(node).__name__)


def _value(node: ast.AST, env: dict):
    try:
        return safe_eval(node, env)
    except Exception:
        return {"expr": ast.unparse(node)[:200]}


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _module_constants(tree: ast.Module) -> dict:
    """Module-level `NAME = <expr>` assignments (later assignments win)."""
    env: dict = {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            env[st.targets[0].id] = st.value
    # keep only names that evaluate to constants
    out = {}
    for k, v in env.items():
        try:
            val = safe_eval(v, env)
            if isinstance(val, (int, float, str)) or (isinstance(val, list) and len(val) < 50):
                out[k] = val
        except Exception:
            pass
    return out


def _kw_dict(call: ast.Call, env: dict) -> dict:
    d = {}
    for kw in call.keywords:
        if kw.arg is None:
            d["**"] = ast.unparse(kw.value)[:120]
            continue
        if kw.arg in ENTRY_KEYS:
            if kw.arg == "FILTRE":
                d[kw.arg] = [_kw_dict(c, env) | {k.arg: _value(k.value, env) for k in c.keywords if k.arg}
                             for c in _as_F_list(kw.value)] or {"expr": ast.unparse(kw.value)[:120]}
            elif kw.arg in ("RESULTAT", "CHAM_GD", "TABLE", "FONCTION") and isinstance(kw.value, ast.Name):
                d[kw.arg] = kw.value.id
            else:
                d[kw.arg] = _value(kw.value, env)
    return d


def _as_F_list(node: ast.AST) -> list[ast.Call]:
    if isinstance(node, ast.Call) and _call_name(node) == "_F":
        return [node]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [e for e in node.elts if isinstance(e, ast.Call) and _call_name(e) == "_F"]
    return []


def parse_comm(text: str) -> dict:
    """Extract tests, models, materials, operators and header comments from a .comm file."""
    out: dict = {"entries": [], "unresolved_blocks": 0, "modelisations": [], "materials": [],
                 "operators": [], "header": _header(text), "parse_error": None}
    try:
        with warnings.catch_warnings():  # some command files contain invalid escape sequences
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError as exc:
        out["parse_error"] = f"{exc.msg} (line {exc.lineno})"
        return out
    env = _module_constants(tree)
    ops = set()
    in_loop = _nodes_inside_loops(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in SOLVER_OPERATORS:
            ops.add(name)
        if name == "AFFE_MODELE":
            for kw in node.keywords:
                if kw.arg == "AFFE":
                    for f in _as_F_list(kw.value):
                        for k in f.keywords:
                            if k.arg == "MODELISATION":
                                v = _value(k.value, env)
                                out["modelisations"] += v if isinstance(v, list) else [v]
        if name == "DEFI_MATERIAU":
            mat = {}
            for kw in node.keywords:
                if kw.arg in ("ELAS", "ELAS_ORTH", "ELAS_ISTR", "THER"):
                    for f in _as_F_list(kw.value):
                        mat[kw.arg] = {k.arg: _value(k.value, env) for k in f.keywords if k.arg}
            if mat:
                out["materials"].append(mat)
        if name in TEST_COMMANDS:
            entries = []
            if name == "TEST_TABLE":
                entries.append(_kw_dict(node, env))
            for kw in node.keywords:
                fl = _as_F_list(kw.value)
                if fl:
                    for f in fl:
                        e = _kw_dict(f, env)
                        e["_block"] = kw.arg
                        entries.append(e)
                elif kw.arg in ("RESU", "CHAM_NO", "CHAM_ELEM", "CARTE", "VALEUR", "OBJET", "GENE", "MAILLAGE"):
                    out["unresolved_blocks"] += 1
            for e in entries:
                e["_command"] = name
                e["_line"] = node.lineno
                if id(node) in in_loop:
                    e["_in_loop"] = True
                if any(k in e for k in ("VALE_REFE", "VALE_CALC", "VALE_REFE_I", "VALE_CALC_I",
                                        "VALE_REFE_C", "VALE_CALC_C")):
                    out["entries"].append(e)
    out["operators"] = sorted(ops)
    out["modelisations"] = sorted({m for m in out["modelisations"] if isinstance(m, str)})
    return out


def _nodes_inside_loops(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.FunctionDef, ast.ListComp)):
            for sub in ast.walk(node):
                ids.add(id(sub))
    return ids


def _header(text: str) -> list[str]:
    lines = text.splitlines()
    out, in_license = [], False
    for ln in lines[:80]:
        s = ln.strip()
        if not s.startswith("#"):
            if out:
                break
            continue
        body = s.lstrip("#").strip()
        if "Copyright" in body or "This file is part of code_aster" in body:
            in_license = True
            continue
        if in_license:
            if "along with code_aster" in body:
                in_license = False
            continue
        if not body or set(body) <= set("-=*_ ") or body.startswith(("coding", "!", "-*-")):
            continue
        out.append(body)
        if len(out) >= 15:
            break
    return out


# ---------------------------------------------------------------------------- reference typing

def classify(entry: dict) -> str:
    ref = entry.get("REFERENCE")
    if isinstance(ref, str):
        if ref in INDEPENDENT:
            return "independent"
        if ref == "AUTRE_ASTER":
            return "other_code_aster_model"
        return "undefined"
    return "regression_only"


def relative_gap(entry: dict) -> float | None:
    """|VALE_CALC - VALE_REFE| / |VALE_REFE| when both are scalars."""
    c, r = entry.get("VALE_CALC"), entry.get("VALE_REFE")
    if isinstance(c, (int, float)) and isinstance(r, (int, float)) and r != 0:
        return abs(c - r) / abs(r)
    return None


# ---------------------------------------------------------------------------- export + meshes

def parse_export(text: str) -> dict:
    files, params = [], {}
    for ln in text.splitlines():
        t = ln.split()
        if not t:
            continue
        if t[0] == "F" and len(t) >= 5:
            files.append({"type": t[1], "name": t[2], "mode": t[3], "unit": int(t[4]) if t[4].isdigit() else t[4]})
        elif t[0] == "P" and len(t) >= 2:
            params[t[1]] = " ".join(t[2:])
    return {"files": files, "params": params}


def mesh_groups(path: Path) -> dict:
    """Node/cell groups of a MED or Aster (.mail) mesh -> {group: {"n": count, "xyz": centroid, "bbox"}}."""
    if path.suffix in (".mmed", ".med"):
        return _med_groups(path)
    if path.suffix == ".mail":
        return _mail_groups(path)
    return {}


def _summarise(points) -> dict:
    import numpy as np

    p = np.asarray(points, dtype=float).reshape(-1, 3)
    d = {"n": int(len(p))}
    if len(p):
        d["xyz"] = [float(x) for x in p.mean(axis=0)]
        if len(p) > 1:
            d["bbox"] = [[float(x) for x in p.min(axis=0)], [float(x) for x in p.max(axis=0)]]
    return d


def _med_groups(path: Path) -> dict:
    try:
        return _med_groups_h5(path)
    except Exception:
        return _med_groups_meshio(path)


def _med_strings(ds) -> list[str]:
    """Group names in MED are fixed 80-character records, stored flat or as (n, 80)."""
    import numpy as np

    raw = np.asarray(ds[()])
    if raw.dtype.kind == "S" and raw.dtype.itemsize >= 80:
        chunks = [bytes(x) for x in raw.ravel()]
    else:
        b = raw.astype("S1").tobytes() if raw.dtype.kind == "S" else raw.astype(np.uint8).tobytes()
        chunks = [b[i:i + 80] for i in range(0, len(b) - len(b) % 80, 80)]
    return [c.split(b"\x00")[0].decode("latin-1").strip() for c in chunks if c.strip(b"\x00 ")]


def _med_groups_h5(path: Path) -> dict:
    """Direct HDF5 reader for MED 2.x/3.x/4.x node coordinates, families and groups."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        mname = next(iter(f["ENS_MAA"].keys()))
        mesh = f["ENS_MAA"][mname]
        base = mesh if "NOE" in mesh else mesh[next(k for k in mesh.keys() if "NOE" in mesh[k])]
        dim = int(mesh.attrs.get("ESP", mesh.attrs.get("DIM", 3)))
        coo = np.asarray(base["NOE"]["COO"][()], dtype=float)
        n = coo.size // dim
        pts = coo.reshape(dim, n).T
        if dim < 3:
            pts = np.c_[pts, np.zeros((n, 3 - dim))]
        node_fam = np.asarray(base["NOE"]["FAM"][()]) if "FAM" in base["NOE"] else np.zeros(n, int)
        fas = f["FAS"][mname] if "FAS" in f and mname in f["FAS"] else mesh["FAS"]
        fam_groups: dict[int, list[str]] = {}
        for kind in ("NOEUD", "ELEME"):
            if kind not in fas:
                continue
            for fname, fam in fas[kind].items():
                num = int(fam.attrs["NUM"])
                if "GRO" in fam and "NOM" in fam["GRO"]:
                    fam_groups[num] = _med_strings(fam["GRO"]["NOM"])
        groups: dict[str, set] = {}
        for fam_id in np.unique(node_fam):
            for g in fam_groups.get(int(fam_id), []):
                groups.setdefault(g, set()).update(np.where(node_fam == fam_id)[0].tolist())
        node_groups = set(groups)
        cells = {}
        if "MAI" in base:
            for tname, blk in base["MAI"].items():
                if "NOD" not in blk:
                    continue
                nod = np.asarray(blk["NOD"][()])
                ne = int(blk["NOD"].attrs.get("NBR", blk.attrs.get("NBR", 0))) or (len(blk["FAM"][()]) if "FAM" in blk else 0)
                if not ne:
                    continue
                conn = nod.reshape(len(nod) // ne, ne).T - 1
                cells[tname] = ne
                fam = np.asarray(blk["FAM"][()]) if "FAM" in blk else np.zeros(ne, int)
                for fam_id in np.unique(fam):
                    names = [g for g in fam_groups.get(int(fam_id), []) if g not in node_groups]
                    if names:
                        nodes = np.unique(conn[fam == fam_id])
                        for g in names:
                            groups.setdefault(g, set()).update(nodes.tolist())
    out = {g: _summarise(pts[sorted(ix)]) | {"kind": "node" if g in node_groups else "cell"}
           for g, ix in groups.items()}
    out["__mesh__"] = {"n_nodes": int(n), "cells": cells,
                       "bbox": [[float(x) for x in pts.min(0)], [float(x) for x in pts.max(0)]]}
    return out


def _med_groups_meshio(path: Path) -> dict:
    import meshio
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = meshio.read(path, file_format="med")
    pts = m.points if m.points.shape[1] == 3 else np.c_[m.points, np.zeros(len(m.points))]
    groups: dict[str, set] = {}
    pt = m.point_data.get("point_tags")
    if pt is not None:
        for fam, names in getattr(m, "point_tags", {}).items():
            idx = np.where(pt == fam)[0]
            for g in names:
                groups.setdefault(g, set()).update(idx.tolist())
    node_groups = set(groups)
    ct = m.cell_data.get("cell_tags")
    if ct is not None:
        for fam, names in getattr(m, "cell_tags", {}).items():
            for block, tags in zip(m.cells, ct):
                rows = np.where(tags == fam)[0]
                if len(rows):
                    nodes = np.unique(block.data[rows])
                    for g in names:
                        if g not in node_groups:
                            groups.setdefault(g, set()).update(nodes.tolist())
    out = {g: _summarise(pts[sorted(ix)]) | {"kind": "node" if g in node_groups else "cell"}
           for g, ix in groups.items()}
    out["__mesh__"] = {"n_nodes": int(len(pts)), "cells": {c.type: int(len(c.data)) for c in m.cells},
                       "bbox": [[float(x) for x in pts.min(0)], [float(x) for x in pts.max(0)]]}
    return out


def _mail_groups(path: Path) -> dict:
    """Minimal reader for the native ASCII Aster mesh format."""
    text = path.read_text(errors="replace").splitlines()
    coords: dict[str, list[float]] = {}
    cells: dict[str, list[str]] = {}
    gno: dict[str, list[str]] = {}
    gma: dict[str, list[str]] = {}
    i, n = 0, len(text)
    while i < n:
        t = text[i].split("%")[0].strip()
        key = t.split()[0].upper() if t else ""
        if key.startswith("COOR_"):
            dim = int(key[5])
            i += 1
            while i < n and text[i].strip().upper() != "FINSF":
                f = text[i].split("%")[0].split()
                if len(f) >= dim + 1:
                    xyz = [float(x.replace("D", "E")) for x in f[1:dim + 1]] + [0.0] * (3 - dim)
                    coords[f[0]] = xyz
                i += 1
        elif key in ("GROUP_NO", "GROUP_MA"):
            target = gno if key == "GROUP_NO" else gma
            i += 1
            name = None
            while i < n and text[i].strip().upper() != "FINSF":
                toks = text[i].split("%")[0].replace("NOM=", " ").split()
                for tok in toks:
                    if name is None:
                        name = tok
                        target[name] = []
                    else:
                        target[name].append(tok)
                i += 1
        elif key and key not in ("TITRE", "FIN") and re.match(r"^(SEG|TRIA|QUAD|TETRA|PENTA|HEXA|PYRAM|POI)\d+", key):
            i += 1
            while i < n and text[i].strip().upper() != "FINSF":
                f = text[i].split("%")[0].split()
                if f:
                    cells[f[0]] = f[1:]
                i += 1
        i += 1
    out = {}
    for g, names in gno.items():
        out[g] = _summarise([coords[x] for x in names if x in coords]) | {"kind": "node"}
    for g, names in gma.items():
        if g in out:
            continue
        nodes = {nd for c in names for nd in cells.get(c, [])}
        out[g] = _summarise([coords[x] for x in nodes if x in coords]) | {"kind": "cell"}
    if coords:
        import numpy as np

        p = np.array(list(coords.values()))
        out["__mesh__"] = {"n_nodes": len(p), "cells": len(cells),
                           "bbox": [p.min(0).tolist(), p.max(0).tolist()]}
    return out


# ---------------------------------------------------------------------------- manifest

def fetch(dest: str | Path, families: list[str] = ("sslv",), commit: str = UPSTREAM["commit"]) -> Path:
    dest = Path(dest)
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM["repo"], str(dest)],
                       check=True)
    pats = [f"/astest/{f}*" for f in families] + ["/LICENSE"]
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *pats], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    return dest / "astest"


def build_manifest(astest: str | Path, families: list[str] = ("sslv",), mesh: bool = True) -> dict:
    astest = Path(astest)
    tests = []
    for exp in sorted(astest.glob("*.export")):
        if not any(exp.stem.startswith(f) for f in families):
            continue
        ex = parse_export(exp.read_text(errors="replace"))
        files = ex["files"]
        comm_files = [f["name"] for f in files if f["type"] in ("comm",) or re.match(r"com\d", f["type"])]
        rec = {"id": exp.stem, "family": re.match(r"[a-z]+", exp.stem).group(0),
               "export": ex, "files": {}, "entries": [], "parse_errors": []}
        size = 0
        for f in files:
            p = astest / f["name"]
            if p.exists():
                rec["files"][f["name"]] = {"type": f["type"], "bytes": p.stat().st_size, "unit": f["unit"]}
                size += p.stat().st_size
            else:
                rec["files"][f["name"]] = {"type": f["type"], "missing": True}
        rec["total_bytes"] = size
        mods, mats, ops, header, unresolved = set(), [], set(), [], 0
        for cf in comm_files:
            p = astest / cf
            if not p.exists():
                continue
            parsed = parse_comm(p.read_text(errors="replace"))
            if parsed["parse_error"]:
                rec["parse_errors"].append(f"{cf}: {parsed['parse_error']}")
            for e in parsed["entries"]:
                e["_file"] = cf
                e["ref_class"] = classify(e)
                g = relative_gap(e)
                if g is not None:
                    e["calc_vs_refe_rel"] = g
                if e["ref_class"] != "regression_only":
                    e.setdefault("PRECISION_EFFECTIVE", e.get("PRECISION", 1.0e-3))
                rec["entries"].append(e)
            mods |= set(parsed["modelisations"])
            mats += parsed["materials"]
            ops |= set(parsed["operators"])
            header = header or parsed["header"]
            unresolved += parsed["unresolved_blocks"]
        rec.update(title=_title(header), header=header, modelisations=sorted(mods), materials=mats[:5],
                   operators=sorted(ops), unresolved_test_blocks=unresolved)
        if mesh:
            groups = {}
            for f in files:
                if f["type"] in ("mmed", "mail") and (astest / f["name"]).exists():
                    try:
                        groups.update(mesh_groups(astest / f["name"]))
                    except Exception as exc:  # pragma: no cover - malformed meshes
                        rec["parse_errors"].append(f"{f['name']}: mesh read failed: {exc}")
            meta = groups.pop("__mesh__", None)
            rec["mesh"] = meta
            for e in rec["entries"]:
                g = e.get("GROUP_NO") if isinstance(e.get("GROUP_NO"), str) else None
                g = g or (e.get("GROUP_MA") if isinstance(e.get("GROUP_MA"), str) else None)
                if g and g in groups:
                    e["location"] = groups[g]
        counts: dict[str, int] = {}
        for e in rec["entries"]:
            counts[e["ref_class"]] = counts.get(e["ref_class"], 0) + 1
        rec["ref_counts"] = counts
        tests.append(rec)
    summary = {
        "n_tests": len(tests),
        "n_with_independent_refs": sum(1 for t in tests if t["ref_counts"].get("independent")),
        "n_entries": sum(len(t["entries"]) for t in tests),
        "entries_by_class": {},
        "operators": {},
    }
    for t in tests:
        for k, v in t["ref_counts"].items():
            summary["entries_by_class"][k] = summary["entries_by_class"].get(k, 0) + v
        for op in t["operators"]:
            summary["operators"][op] = summary["operators"].get(op, 0) + 1
    return {"upstream": UPSTREAM, "families": list(families), "summary": summary, "tests": tests}


def _title(header: list[str]) -> str:
    for h in header:
        if h.upper().startswith(("CAS_TEST", "SSLV", "SDLV", "SSLS", "SSNV", "MODELISATION", "MAILLAGE")):
            continue
        if len(h) > 6:
            return h
    return header[0] if header else ""


def is_core_case(test: dict, max_bytes: int = 1_500_000) -> bool:
    """Vendoring rule: independent references, 3D solid modelling, no fracture / nonlinear dynamics."""
    if not test["ref_counts"].get("independent"):
        return False
    if not any(m.startswith("3D") for m in test["modelisations"]):
        return False
    if set(test["operators"]) & {"CALC_G", "CALC_G_XFEM", "POST_K1_K2_K3", "DYNA_NON_LINE"}:
        return False
    return test["total_bytes"] <= max_bytes


def vendor(manifest: dict, astest: str | Path, dest: str | Path, select=is_core_case) -> list[str]:
    """Copy all files of the selected tests into ``dest`` and mark them in the manifest."""
    import shutil

    astest, dest = Path(astest), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    chosen = []
    for t in manifest["tests"]:
        t["vendored"] = bool(select(t))
        if not t["vendored"]:
            continue
        for name, meta in t["files"].items():
            if not meta.get("missing"):
                shutil.copy(astest / name, dest / name)
        shutil.copy(astest / f"{t['id']}.export", dest / f"{t['id']}.export")
        chosen.append(t["id"])
    return chosen


def catalog_markdown(manifest: dict) -> str:
    """Human-browsable table of the vendored tests (what they are, what they reference, how well
    code_aster itself matches its references on the test mesh)."""
    rows = ["| Test | Title (from the command file) | Model | Operators | Independent refs (located) | "
            "code_aster vs ref, median |", "|---|---|---|---|---|---|"]
    for t in manifest["tests"]:
        if not t.get("vendored"):
            continue
        ind = [e for e in t["entries"] if e["ref_class"] == "independent"]
        gaps = sorted(e["calc_vs_refe_rel"] for e in ind if "calc_vs_refe_rel" in e)
        med = f"{gaps[len(gaps) // 2] * 100:.2g} %" if gaps else "-"
        located = sum(1 for e in ind if "location" in e)
        title = (t["title"] or "").replace("|", "/")[:70]
        rows.append(f"| {t['id']} | {title} | {', '.join(t['modelisations'])[:24]} | "
                    f"{', '.join(t['operators'])[:40]} | {len(ind)} ({located}) | {med} |")
    s = manifest["summary"]
    head = (f"# code_aster vendored tests\n\nGenerated from `manifest.json` (upstream commit "
            f"{manifest['upstream']['commit'][:10]}). {s['n_tests']} SSLV/SDLV variants parsed; "
            f"{s['n_with_independent_refs']} have independent references; the {len(rows) - 2} below are "
            f"vendored in `astest/`. 'code_aster vs ref' is the median |VALE_CALC - VALE_REFE| / |VALE_REFE| "
            f"over the test's independent references: how close code_aster itself gets on the test mesh.\n\n")
    return head + "\n".join(rows) + "\n"
