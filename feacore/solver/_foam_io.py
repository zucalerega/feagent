"""OpenFOAM case files for the OpenFOAM backend (private helper): dictionaries written with
``foamlib`` (based on the shipped ``simpleFoam/pitzDaily`` tutorial and the
``etc/templates/inflowOutflow`` template), plus readers for fields, polyMesh geometry, solver
logs and ``forces`` output.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    from foamlib import DimensionSet, FoamFieldFile, FoamFile
except ImportError as e:  # pragma: no cover - checked by the backend
    FoamFile = FoamFieldFile = DimensionSet = None
    FOAMLIB_ERROR = e
else:
    FOAMLIB_ERROR = None

LIB_FIELD = '"libfieldFunctionObjects.so"'
LIB_FORCES = '"libforces.so"'


def _w(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = FoamFile(path)
    for k, v in content.items():
        f[k] = v


def _t(s: str):
    """A multi-word OpenFOAM entry ('bounded Gauss upwind') as a foamlib tuple."""
    return tuple(s.split())


# --------------------------------------------------------------------------- system / constant


def write_control(case: Path, end: int, functions: dict | None = None) -> None:
    _w(case / "system" / "controlDict", {
        "application": "simpleFoam", "startFrom": "latestTime", "startTime": 0, "stopAt": "endTime",
        "endTime": int(end), "deltaT": 1, "writeControl": "timeStep", "writeInterval": int(end),
        "purgeWrite": 0, "writeFormat": "ascii", "writePrecision": 17, "writeCompression": False,
        "timeFormat": "general", "timePrecision": 8, "runTimeModifiable": False,
        "functions": functions or {},
    })


def write_schemes(case: Path, order: int, turbulent: bool) -> None:
    conv = "bounded Gauss linearUpwind grad(U)" if order == 2 else "bounded Gauss upwind"
    turb = "bounded Gauss limitedLinear 1" if order == 2 else "bounded Gauss upwind"
    div = {"default": "none", "div(phi,U)": _t(conv),
           "div((nuEff*dev2(T(grad(U)))))": _t("Gauss linear")}
    if turbulent:
        div.update({"div(phi,k)": _t(turb), "div(phi,omega)": _t(turb), "div(phi,epsilon)": _t(turb)})
    _w(case / "system" / "fvSchemes", {
        "ddtSchemes": {"default": "steadyState"},
        "gradSchemes": {"default": _t("Gauss linear")},
        "divSchemes": div,
        "laplacianSchemes": {"default": _t("Gauss linear corrected")},
        "interpolationSchemes": {"default": "linear"},
        "snGradSchemes": {"default": "corrected"},
        "wallDist": {"method": "meshWave"},
    })


def write_solution(case: Path, tol: float, turbulent: bool, ref: dict | None, relax_u: float,
                   relax_p: float | None, n_nonorth: int) -> None:
    lin = max(min(tol * 1e-2, 1e-8), 1e-15)
    fields = "(U|k|omega|epsilon)" if turbulent else "U"
    simple = {"nNonOrthogonalCorrectors": int(n_nonorth), "consistent": True,
              "residualControl": {"p": tol, "U": tol, **({'"(k|omega|epsilon)"': tol} if turbulent else {})}}
    if ref:
        simple.update(ref)
    relax: dict = {"equations": {"U": relax_u, '".*"': relax_u}}
    if relax_p is not None:
        relax["fields"] = {"p": relax_p}
    _w(case / "system" / "fvSolution", {
        "solvers": {
            "p": {"solver": "GAMG", "smoother": "GaussSeidel", "tolerance": lin, "relTol": 0.05},
            f'"{fields}"': {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": lin,
                            "relTol": 0.1},
        },
        "SIMPLE": simple,
        "relaxationFactors": relax,
    })


def write_transport(case: Path, nu: float, turbulence: str) -> None:
    _w(case / "constant" / "transportProperties",
       {"transportModel": "Newtonian", "nu": (DimensionSet(length=2, time=-1), float(nu))})
    if turbulence == "laminar":
        d = {"simulationType": "laminar"}
    else:
        d = {"simulationType": "RAS", "RAS": {"RASModel": turbulence, "turbulence": True, "printCoeffs": True}}
    _w(case / "constant" / "turbulenceProperties", d)


def set_patch_types(case: Path, types: dict) -> None:
    """Rewrite constant/polyMesh/boundary with geometric patch types {name: type}."""
    f = FoamFile(case / "constant" / "polyMesh" / "boundary")
    entries = f[None]
    out = []
    for name, d in entries:
        t = types.get(name, d.get("type", "patch"))
        e = {"type": t}
        if t == "wall":
            e["inGroups"] = ["wall"]
        e["nFaces"] = int(d["nFaces"])
        e["startFace"] = int(d["startFace"])
        out.append((name, e))
    f[None] = out


def write_field(case: Path, name: str, cls: str, dims: dict, internal, boundary: dict) -> None:
    path = case / "0" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    f = FoamFieldFile(path)
    f["FoamFile"] = {"version": 2.0, "format": "ascii", "class": cls, "object": name}
    f.dimensions = DimensionSet(**dims)
    f.internal_field = internal
    f.boundary_field = boundary


# --------------------------------------------------------------------------- readers


def read_list(path: Path) -> np.ndarray:
    return np.asarray(FoamFile(path)[None])


def read_poly(case: Path) -> dict:
    """points, owner, neighbour, patch table and boundary-face geometry (centres, area vectors)."""
    pm = case / "constant" / "polyMesh"
    points = np.asarray(FoamFile(pm / "points")[None], dtype=float)
    owner = np.asarray(FoamFile(pm / "owner")[None], dtype=np.int64)
    faces = FoamFile(pm / "faces")[None]
    patches = {name: (int(d["nFaces"]), int(d["startFace"]), d.get("type", "patch"))
               for name, d in FoamFile(pm / "boundary")[None]}
    nint = min((s for n, s, _ in patches.values()), default=len(owner))
    return {"points": points, "owner": owner, "faces": faces, "patches": patches, "n_internal": nint}


def face_geometry(points: np.ndarray, faces) -> tuple[np.ndarray, np.ndarray]:
    """OpenFOAM face centres and area vectors (triangle fan about the vertex average)."""
    n = len(faces)
    C = np.zeros((n, 3))
    S = np.zeros((n, 3))
    by_len: dict = {}
    for i, f in enumerate(faces):
        by_len.setdefault(len(f), []).append(i)
    for k, idx in by_len.items():
        idx = np.asarray(idx)
        F = np.asarray([faces[i] for i in idx], dtype=np.int64)
        X = points[F]                       # (m, k, 3)
        xm = X.mean(axis=1)
        if k == 3:
            S[idx] = 0.5 * np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
            C[idx] = xm
            continue
        sa = np.zeros((len(idx), 3))
        sc = np.zeros((len(idx), 3))
        sm = np.zeros(len(idx))
        for j in range(k):
            a, b = X[:, j], X[:, (j + 1) % k]
            c = a + b + xm
            s = 0.5 * np.cross(b - a, xm - a)
            m = np.linalg.norm(s, axis=1)
            sa += s
            sc += m[:, None] * c
            sm += m
        S[idx] = sa
        C[idx] = np.where(sm[:, None] > 0, sc / (3.0 * np.maximum(sm, 1e-300)[:, None]), xm)
    return C, S


def read_field(path: Path, n_cells: int):
    """(internal values (n,) or (n, k), boundaryField dict) of a volField file."""
    f = FoamFieldFile(path)
    internal = np.asarray(f.internal_field, dtype=float)
    if internal.ndim == 0 or (internal.ndim == 1 and len(internal) in (3, 9) and n_cells != len(internal)):
        internal = np.broadcast_to(internal, (n_cells,) + internal.shape).copy()
    return internal, f.boundary_field


def patch_values(bf: dict, name: str, owner_vals: np.ndarray) -> np.ndarray:
    """Face values of a patch: the written 'value' entry, else the owner cell values (zeroGradient)."""
    d = bf.get(name, {})
    if "value" in d:
        v = np.asarray(d["value"], dtype=float)
        if v.ndim == 0 or (v.ndim == 1 and owner_vals.ndim == 2 and len(v) == owner_vals.shape[1]
                           and len(owner_vals) != len(v)):
            v = np.broadcast_to(v, owner_vals.shape).copy()
        return v
    t = d.get("type", "zeroGradient")
    if t == "noSlip":
        return np.zeros_like(owner_vals)
    return owner_vals.copy()


# --------------------------------------------------------------------------- logs / outputs

_RES = re.compile(r"Solving for (\w+), Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+), "
                  r"No Iterations (\d+)")


def parse_log(text: str) -> dict:
    """Iterations, last initial residual per field (U = max over components, as residualControl
    does), convergence flag."""
    iters = 0
    cur: dict = {}
    last: dict = {}
    for line in text.splitlines():
        if line.startswith("Time = "):
            iters += 1
            if cur:
                last = cur
            cur = {}
            continue
        m = _RES.search(line)
        if m:
            fld, r0 = m.group(1), float(m.group(2))
            if fld in ("Ux", "Uy", "Uz"):
                cur["U"] = max(cur.get("U", 0.0), r0) if fld != "Ux" or "U" not in cur else cur["U"]
                cur.setdefault("_" + fld, r0)
            else:
                cur.setdefault(fld, r0)
    if cur:
        last = cur
    res = {k: v for k, v in last.items() if not k.startswith("_")}
    comps = [last[k] for k in ("_Ux", "_Uy", "_Uz") if k in last]
    if comps:
        res["U"] = max(comps)
    converged = "SIMPLE solution converged in" in text
    # (the banner says "Floating point exception trapping enabled": match residuals / crashes only)
    diverged = bool(re.search(r"residual = -?(nan|inf|NaN|Inf)|Floating point exception \(core|FOAM FATAL",
                              text))
    return {"iterations": iters, "residuals": res, "converged": converged, "diverged": diverged}


def read_forces(post: Path, name: str) -> np.ndarray | None:
    """(pressure (3,), viscous (3,)) of the last line of a ``forces`` function object output, for
    the ESI layouts: v2006+ ``force.dat`` (total, pressure, viscous columns) and older
    ``forces.dat`` (``((p) (v) (porous))`` groups)."""
    base = post / name
    if not base.is_dir():
        return None
    files = sorted(base.glob("*/force.dat")) + sorted(base.glob("*/forces.dat"))
    if not files:
        return None
    fpath = files[-1]
    lines = [ln for ln in fpath.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return None
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", lines[-1])]
    vals = nums[1:]
    if fpath.name == "force.dat" and len(vals) >= 9:      # total(3) pressure(3) viscous(3)
        return np.array([vals[3:6], vals[6:9]])
    if len(vals) >= 6:                                     # (pressure)(viscous)(porous)
        return np.array([vals[0:3], vals[3:6]])
    return None
