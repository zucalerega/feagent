"""Transient incompressible flow with OpenFOAM's ``pimpleFoam`` (``analysis.options["transient"]`` of a
physics "flow" case; reached through ``feacore.solver.solve(case, backend="openfoam")``).

The unsteady incompressible Navier-Stokes equations, laminar, formulations "2d" and "3d", on the same
mesh, patches, units and boundary conditions as the steady backend (feacore/solver/openfoam_backend.py),
whose helpers this module calls. docs/solver_transient.md ("Transient incompressible flow") has the
verification and the limits; in short:

* Time scheme ``backward`` (BDF2, second order; the first step is Euler) or ``euler`` (first order).
* Pressure-velocity coupling: PIMPLE with outer correctors. Every step repeats the momentum predictor and
  ``n_correctors`` pressure corrections until the initial residuals of U and p of an outer iteration are
  below ``outer_tol``, at most ``n_outer_correctors`` times. One outer iteration is the PISO algorithm,
  which linearises the convection with the flux of the previous step: its error is first order in the
  step whatever the time scheme (measured: docs/solver_transient.md). The outer loop removes it, so the
  only error in time is the time scheme's, at any Courant number at which the loop converges.
* Space: the steady backend's schemes (``linearUpwind`` convection for ``analysis.order`` 2, ``upwind``
  for 1, ``Gauss linear corrected`` diffusion, the skew-consistent schemes in 3D).
* Values in time (an expression with ``t``, a table) become OpenFOAM boundary conditions that vary in
  time: ``uniformFixedValue`` with a table where the value is the same on the whole patch,
  ``timeVaryingMappedFixedValue`` with one sample per face where it is not (``_timed_entry``).
* Every step is written (binary) and read back here while the solver runs: the QoIs are evaluated in
  numpy with the steady backend's formulas (OpenFOAM v1912's function objects crash), then the step is
  deleted unless it is one of the kept states.
* With ``max_co`` the solver adapts its step to that Courant number below ``dt``; it still lands on every
  multiple of ``dt``, and those are the steps of the result (``times``).

Options: ``analysis.options["transient"]`` (``t_end``, ``dt`` | ``n_steps``, ``scheme``, ``initial``,
``output_times`` | ``output_every``, ``max_co``) and, beside it in ``analysis.options`` or as keywords of
``solve``: ``n_outer_correctors`` (20), ``outer_tol`` (1e-6), ``n_correctors`` (2),
``n_non_orth_correctors`` (0), ``residual_tol`` (1e-8: the linear solvers), ``convection`` (None: by
``analysis.order``; or "linearUpwind", "linear", "limitedLinear", "upwind"), ``skew_correction``,
``timeout``, ``case_dir``, ``keep_case``. The steady backend's options apply to the steady solve of
``initial = {"steady": true}``.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from feacore.case import Case, CaseError, QoI, as_field, table_value, uses_time
from feacore.solver import _foam_io as fio
from feacore.solver import openfoam_backend as steady
from feacore.solver._foam_env import FoamNotFound, require_openfoam, run
from feacore.solver._foam_fv import CellPointInterpolator, FvMesh
from feacore.solver._foam_mesh import FRONT_BACK, convert
from feacore.solver._skfem_mesh import MeshError
from feacore.solver.base import TransientResult

OPTIONS = ("n_outer_correctors", "outer_tol", "n_correctors", "n_non_orth_correctors", "convection")
DEFAULTS = {"turbulence": "laminar", "n_outer_correctors": 20, "outer_tol": 1e-6, "n_correctors": 2,
            "n_non_orth_correctors": 0, "residual_tol": 1e-8, "convection": None, "timeout": None,
            "skew_correction": "auto"}
CONVECTION = {"linearUpwind": "Gauss linearUpwind grad(U)", "linear": "Gauss linear",
              "limitedLinear": "Gauss limitedLinear 1", "upwind": "Gauss upwind"}
DDT = {"backward": "backward", "euler": "Euler"}
MAX_SAMPLES = 20000          # samples in time of a boundary value given as an expression in t
SolverError = steady.SolverError


@dataclass
class TransientFlowInfo(steady.FlowSolveInfo):
    """``iterations`` = solver steps, ``converged`` = every step's outer loop converged, ``residuals`` =
    the initial residuals of the last outer iteration of the last step."""
    scheme: str = ""                 # "backward" | "euler"
    t_end: float = 0.0
    n_steps: int = 0                 # steps of the result (``times`` has n_steps + 1 values)
    dt: float = 0.0                  # the step of the result (the largest solver step)
    dt_min: float = 0.0              # the smallest solver step (= dt without max_co)
    solver_steps: int = 0            # steps pimpleFoam took (> n_steps when max_co cut the step)
    courant_max: float = 0.0         # the largest Courant number of any step
    courant_mean: float = 0.0        # the largest cell-mean Courant number of any step
    outer_iterations: int = 0        # the most outer iterations any step took
    continuity_max: float = 0.0      # the largest "sum local" continuity error of any step


# --------------------------------------------------------------------------- values in time


class _SetTime(ast.NodeTransformer):
    def __init__(self, t: float):
        self.t = float(t)

    def visit_Name(self, node):
        return ast.copy_location(ast.Constant(self.t), node) if node.id == "t" else node


def _freeze(value, t: float):
    """A value at the fixed time t: a table becomes a number, an expression that uses t the same
    expression with the number in its place; numbers and expressions in x, y, z stay."""
    if isinstance(value, dict) and "table" in value:
        return table_value(value, t)
    if isinstance(value, str) and uses_time(value):
        tree = _SetTime(t).visit(ast.parse(value, mode="eval"))
        return ast.unparse(ast.fix_missing_locations(tree))
    if isinstance(value, list):
        return [_freeze(v, t) for v in value]
    return value


def _frozen_case(case: Case, t: float) -> Case:
    """The steady case of the boundary values at time t (no transient options, no QoIs)."""
    out = Case.from_dict(case.to_dict(), base_dir=case.base_dir)
    for b in out.bcs:
        b.value = _freeze(b.value, t)
    out.analysis.options = {k: v for k, v in (out.analysis.options or {}).items() if k != "transient"}
    out.qois = []
    return out


def _velocity(bc, Cf: np.ndarray, Sf: np.ndarray, dim: int, t: float) -> np.ndarray:
    """The velocity a BC prescribes at face centres Cf (n, 3) at time t -> (n, 3). A scalar value is the
    speed along the inward normal."""
    z = Cf[:, 2] if dim == 3 else None
    v = bc.value
    if isinstance(v, list):
        comps = [as_field(c, Cf[:, 0], Cf[:, 1], z, t) for c in v]
        while len(comps) < 3:
            comps.append(np.zeros(len(Cf)))
        return np.column_stack(comps)
    s = as_field(v, Cf[:, 0], Cf[:, 1], z, t)
    nrm = Sf / np.linalg.norm(Sf, axis=1)[:, None]
    return -s[:, None] * nrm


def _sample_times(value, grid: np.ndarray, warnings: list, what: str) -> np.ndarray:
    """The times at which a value in time is sampled for OpenFOAM, which interpolates linearly between
    them. Tables: their own points (exact at every time). Expressions in t: every step of ``grid`` (exact
    at every step), at most MAX_SAMPLES."""
    parts = value if isinstance(value, list) else [value]
    t_end = float(grid[-1])
    ts = [np.array([0.0, t_end])]
    for part in parts:
        if isinstance(part, dict) and "table" in part:
            tab = np.asarray(part["table"], dtype=float)[:, 0]
            ts.append(tab[(tab > 0.0) & (tab < t_end)])
        elif isinstance(part, str) and uses_time(part):
            if len(grid) - 1 > MAX_SAMPLES:
                warnings.append(f"info: {what} is sampled at {MAX_SAMPLES} times, not at each of the "
                                f"{len(grid) - 1} steps (linear in between)")
                ts.append(np.linspace(0.0, t_end, MAX_SAMPLES + 1))
            else:
                ts.append(grid)
    return np.unique(np.concatenate(ts))


def _fmt_vectors(v: np.ndarray) -> str:
    v = np.atleast_2d(v)
    return f"{len(v)}\n(\n" + "\n".join("(" + " ".join(repr(float(c)) for c in row) + ")" for row in v) + "\n)\n"


def _fmt_scalars(v: np.ndarray) -> str:
    return f"{len(v)}\n(\n" + "\n".join(repr(float(c)) for c in v) + "\n)\n"


def _timed_entry(cdir: Path, patch: str, fld: str, ts: np.ndarray, vals: np.ndarray, Cf: np.ndarray) -> dict:
    """The boundaryField entry of a value in time. ``vals``: (n_times, n_faces, 3) for U or
    (n_times, n_faces) for p (kinematic), at the times ``ts``.

    * The same on every face: ``uniformFixedValue`` with ``uniformValue table`` (OpenFOAM's Function1,
      linear between the points, clamped outside).
    * Different from face to face: ``timeVaryingMappedFixedValue`` with ``mapMethod nearest``; the
      samples are files ``constant/boundaryData/<patch>/<time>/<field>`` at the face centres
      (``points``), without a header, as v1912 reads them."""
    vector = vals.ndim == 3
    scale = max(float(np.abs(vals).max()), 1e-300)
    spread = float((vals.max(axis=1) - vals.min(axis=1)).max())
    if spread <= 1e-12 * scale:
        mean = vals.mean(axis=1)
        table = [[float(t), [float(c) for c in v] if vector else float(v)] for t, v in zip(ts, mean)]
        first = [float(c) for c in mean[0]] if vector else float(mean[0])
        return {"type": "uniformFixedValue", "uniformValue": ("table", table), "value": first}
    bd = cdir / "constant" / "boundaryData" / patch
    bd.mkdir(parents=True, exist_ok=True)
    (bd / "points").write_text(_fmt_vectors(Cf))
    for t, v in zip(ts, vals):
        d = bd / f"{float(t):.15g}"
        d.mkdir(exist_ok=True)
        (d / fld).write_text(_fmt_vectors(v) if vector else _fmt_scalars(v))
    return {"type": "timeVaryingMappedFixedValue", "offset": [0.0, 0.0, 0.0] if vector else 0.0,
            "setAverage": False, "mapMethod": "nearest", "value": vals[0]}


# --------------------------------------------------------------------------- dictionaries


def _write_dict(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = fio.FoamFile(path)
    for k, v in content.items():
        f[k] = v


def _write_control(cdir: Path, t_end: float, dt: float, max_co: float | None) -> None:
    """Fixed step: every step is written. With ``max_co``: ``adjustableRunTime`` with the write interval
    ``dt``, so the solver cuts its step to the Courant number and still lands on every multiple of dt."""
    d = {"application": "pimpleFoam", "startFrom": "latestTime", "startTime": 0, "stopAt": "endTime",
         "endTime": float(t_end), "deltaT": float(dt),
         "writeControl": "timeStep" if max_co is None else "adjustableRunTime",
         "writeInterval": 1 if max_co is None else float(dt),
         "purgeWrite": 0, "writeFormat": "binary", "writePrecision": 17, "writeCompression": False,
         "timeFormat": "general", "timePrecision": 12, "runTimeModifiable": False,
         "adjustTimeStep": max_co is not None}
    if max_co is not None:
        d.update({"maxCo": float(max_co), "maxDeltaT": float(dt)})
    _write_dict(cdir / "system" / "controlDict", d)


def _write_schemes(cdir: Path, ddt: str, conv: str, skew: bool) -> None:
    grad: dict = {"default": fio._t("Gauss linear")}
    interp: dict = {"default": "linear"}
    if skew:                       # as the steady backend in 3D (docs/solver_openfoam.md, "3D meshes")
        grad = {"default": "pointCellsLeastSquares", "grad(p)": fio._t("Gauss linear")}
        interp["flux(HbyA)"] = fio._t("skewCorrected linear")
    _write_dict(cdir / "system" / "fvSchemes", {
        "ddtSchemes": {"default": ddt},
        "gradSchemes": grad,
        "divSchemes": {"default": "none", "div(phi,U)": fio._t(conv),
                       "div((nuEff*dev2(T(grad(U)))))": fio._t("Gauss linear")},
        "laplacianSchemes": {"default": fio._t("Gauss linear corrected")},
        "interpolationSchemes": interp,
        "snGradSchemes": {"default": "corrected"},
    })


def _write_solution(cdir: Path, lin: float, n_outer: int, outer_tol: float, n_corr: int, n_nonorth: int,
                    closed: bool) -> None:
    """Linear solvers to the absolute tolerance ``lin`` in the last corrector of the last outer iteration
    (``pFinal``, ``UFinal``), loosely before that; the outer loop stops at ``outer_tol``."""
    p = {"solver": "GAMG", "smoother": "GaussSeidel", "tolerance": lin, "relTol": 0.01}
    u = {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": lin, "relTol": 0.0}
    pimple: dict = {"nOuterCorrectors": int(n_outer), "nCorrectors": int(n_corr),
                    "nNonOrthogonalCorrectors": int(n_nonorth), "momentumPredictor": True}
    if n_outer > 1:
        pimple["residualControl"] = {'"(U|p)"': {"tolerance": float(outer_tol), "relTol": 0.0}}
    if closed:
        pimple.update({"pRefCell": 0, "pRefValue": 0.0})
    _write_dict(cdir / "system" / "fvSolution", {
        "solvers": {"p": p, "pFinal": {**p, "relTol": 0.0}, '"(U|UFinal)"': u},
        "PIMPLE": pimple,
        "relaxationFactors": {"equations": {'".*"': 1.0}},
    })


# --------------------------------------------------------------------------- states


class _Shared:
    """What every step of a run shares: the finite-volume mesh and the point interpolator."""

    def __init__(self, poly: dict, cdir: Path, fm, dim: int):
        nb = fio.read_list(cdir / "constant" / "polyMesh" / "neighbour").astype(np.int64)
        self.poly = poly
        self.mesh = FvMesh(poly["points"], poly["faces"], poly["owner"], nb, poly["patches"])
        if self.mesh.n_cells != fm.n_cells:
            raise SolverError(f"OpenFOAM mesh has {self.mesh.n_cells} cells, the input mesh {fm.n_cells}")
        self.dim = dim
        self.thickness = fm.thickness
        self._interp = None

    def interpolator(self):
        if self._interp is None:
            m = self.mesh
            tree = cKDTree(m.C[:, :self.dim])
            hloc = (np.abs(m.V) / (self.thickness if self.dim == 2 else 1.0)) ** (1.0 / self.dim)
            empty = [n for n, (_, _, t) in m.patches.items() if t == "empty"]
            self._interp = (CellPointInterpolator(m, self.poly["faces"], fit_dims=self.dim, skip_patches=empty),
                            tree, hloc)
        return self._interp


class _State(steady.OpenFOAMResult):
    """The flow at one time: the steady backend's result (probe, nodal, evaluate with its numpy
    post-processing), loaded from a time directory of the transient run. ``time`` is its time."""

    time: float = 0.0

    def _setup(self, sh: _Shared, dir_u: set, dir_p: set) -> None:
        """After a load: the patches whose value this run prescribes count as Dirichlet for the probes
        (the steady backend knows them by OpenFOAM type, and the mapped type is not in its list); a
        state without a flux file (the initial one) takes its boundary flux from the boundary velocity;
        the point interpolator is the shared one."""
        self._dirichlet = {"U": {n: self._Ub[n] for n in dir_u if n in self._Ub},
                           "p": {n: self.patch[n]["p"] for n in dir_p if n in self.patch}}
        self._interp, self._tree, self._hloc = sh.interpolator() if self._want_probe else (None, None, None)

    _want_probe = True

    def _load_fast(self, sh: _Shared, tdir: Path, need_grad: bool) -> None:
        """``OpenFOAMResult._load`` for a laminar step, without rebuilding the mesh: the same fields and
        patch values, so ``evaluate`` gives the same numbers (tests/test_transient_flow.py compares)."""
        mesh = sh.mesh
        self.mesh = mesh
        n = mesh.n_cells
        U, bU = fio.read_field(tdir / "U", n)
        p, bp = fio.read_field(tdir / "p", n)
        p = p.ravel()
        bphi = fio.FoamFieldFile(tdir / "phi").boundary_field if (tdir / "phi").exists() else {}
        self.C, self.V = mesh.C, mesh.V
        self.pshift = float(np.sum(p * self.V) / np.sum(self.V)) if self.closed else 0.0
        self.p_kin, self.U = p, U
        self.patch = {}
        Ub, pb, nu_b = {}, {}, {}
        for name, (nf, start, ptype) in mesh.patches.items():
            if nf == 0 or ptype == "empty":
                continue
            sl = mesh.patch_slice(name)
            own = mesh.owner[sl]
            uf = fio.patch_values(bU, name, U[own])
            if bU.get(name, {}).get("type") in ("slip", "symmetry", "symmetryPlane"):
                nrm = mesh.Sf[sl] / mesh.magSf[sl][:, None]
                uf = U[own] - nrm * np.einsum("ij,ij->i", U[own], nrm)[:, None]
            pf = fio.patch_values(bp, name, p[own])
            ph = bphi.get(name, {}).get("value")
            ph = np.zeros(nf) if ph is None else np.broadcast_to(np.asarray(ph, dtype=float), (nf,)).copy()
            Ub[name], pb[name], nu_b[name] = uf, pf, np.full(nf, self.nu)
            self.patch[name] = {"Cf": mesh.Cf[sl], "Sf": mesh.Sf[sl], "U": uf, "p": (pf - self.pshift) * self.rho,
                                "phi": ph}
        self._Ub, self._pb_kin, self._nu_b = Ub, pb, nu_b
        self._faces = sh.poly["faces"]
        self._nodes = {}
        if need_grad:
            self.gradU = mesh.grad(U, Ub)
        self.p = (p - self.pshift) * self.rho

    def boundary_flux(self) -> float:
        """The sum of the face flux over the whole boundary (m^3/s; 0 for an incompressible flow)."""
        return float(sum(d["phi"].sum() for d in self.patch.values()))


def _qoi_key(q: QoI) -> tuple:
    """What a QoI measures, without its name and its reduction in time."""
    pt = None if q.point is None else tuple(float(v) for v in q.point)
    pt2 = None if q.point2 is None else tuple(float(v) for v in q.point2)
    return (q.type, q.field, pt, q.group, q.component, pt2)


def _instant(q: QoI) -> QoI:
    return QoI(name=q.name, type=q.type, field=q.field, point=q.point, group=q.group, component=q.component,
               point2=q.point2)


# --------------------------------------------------------------------------- solve


def solve(case: Case, **options) -> "FoamTransientResult":
    t0 = time.perf_counter()
    case.check()
    a = case.analysis
    tr = a.transient
    if a.physics != "flow" or tr is None:
        raise CaseError("feacore.solver._foam_transient solves transient cases of physics 'flow' "
                        "(analysis.options.transient)")
    if fio.FOAMLIB_ERROR is not None:
        raise ImportError(f"the openfoam backend needs foamlib (pip install foamlib): {fio.FOAMLIB_ERROR}")
    mat = steady._fluid_material(case)
    fe = require_openfoam()
    if shutil.which("pimpleFoam", path=fe.env.get("PATH", "")) is None:
        raise FoamNotFound(f"{fe.describe()} has no pimpleFoam, which transient flow needs")
    opts = {**DEFAULTS, **{k: v for k, v in (a.options or {}).items() if k != "transient"}, **options}
    warnings: list[str] = []
    unknown = sorted(set(a.options or {}) - set(steady.KNOWN_OPTIONS) - set(OPTIONS) - {"transient"})
    if unknown:
        warnings.append(f"unknown analysis options ignored: {unknown}")
    if str(opts["turbulence"]) != "laminar":
        raise NotImplementedError(
            f"openfoam backend: transient flow is laminar only (turbulence = {opts['turbulence']!r}); the RANS "
            f"models are validated for steady flow first (docs/solver_transient.md)")
    scheme = tr.get("scheme") or "backward"
    conv = opts.get("convection") or ("linearUpwind" if int(a.order) == 2 else "upwind")
    if conv not in CONVECTION:
        raise CaseError(f"analysis.options.convection must be one of {tuple(CONVECTION)} (got {conv!r})")
    for key, low in (("n_outer_correctors", 1), ("n_correctors", 1), ("n_non_orth_correctors", 0)):
        v = opts[key]
        if not isinstance(v, int) or isinstance(v, bool) or v < low:
            raise CaseError(f"analysis.options.{key} must be an integer >= {low} (got {v!r})")
    for key in ("outer_tol", "residual_tol"):
        if not (isinstance(opts[key], (int, float)) and not isinstance(opts[key], bool) and opts[key] > 0):
            raise CaseError(f"analysis.options.{key} must be a number > 0 (got {opts[key]!r})")
    ini = tr.get("initial") or {}
    if ini.get("steady") and (set(ini) - {"steady"}):
        raise CaseError("analysis.options.transient.initial: give 'steady' or 'U' / 'p', not both")
    opts["skew_correction"] = steady._skew_correction(opts["skew_correction"], case.dim)
    if not Path(case.mesh_path()).exists():
        raise MeshError(f"mesh file not found: {str(case.mesh_path())!r}")

    keep = bool(opts.get("keep_case")) or opts.get("case_dir") is not None
    if opts.get("case_dir") is not None:
        cdir = Path(opts["case_dir"]).resolve()
        if cdir.exists() and any(cdir.iterdir()):
            raise ValueError(f"case_dir {str(cdir)!r} exists and is not empty")
        cdir.mkdir(parents=True, exist_ok=True)
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="feacore_foam_t_")
        cdir = Path(tmp.name)
    try:
        res = _run(case, fe, cdir, opts, mat, scheme, conv, warnings, t0, keep)
    except BaseException:
        if tmp is not None:
            tmp.cleanup()
        raise
    res._tmp = tmp
    if keep:
        if tmp is not None:
            tmp._finalizer.detach()  # type: ignore[attr-defined]
            res._tmp = None
        res.info.case_dir = str(cdir)
    return res


def _initial_fields(case: Case, ini: dict, sh: _Shared, rho: float):
    """(U (n, 3) or a 3-list, p kinematic (n,) or a number) of ``initial = {"U": [...], "p": ...}``:
    numbers, or expressions in x, y, z evaluated at the cell centres."""
    C = sh.mesh.C
    z = C[:, 2] if case.dim == 3 else None
    u = ini.get("U", [0.0] * case.dim)
    if not isinstance(u, list) or len(u) != case.dim:
        raise CaseError(f"analysis.options.transient.initial.U must be a list of {case.dim} components")
    pv = ini.get("p", 0.0)
    if isinstance(pv, (list, dict)) or isinstance(pv, bool):
        raise CaseError("analysis.options.transient.initial.p must be a number or an expression in x, y, z")
    if any(isinstance(c, str) for c in u):
        comps = [as_field(c, C[:, 0], C[:, 1], z) for c in u]
        while len(comps) < 3:
            comps.append(np.zeros(len(C)))
        U0 = np.column_stack(comps)
    else:
        U0 = [float(c) for c in u] + [0.0] * (3 - len(u))
    p0 = as_field(pv, C[:, 0], C[:, 1], z) / rho if isinstance(pv, str) else float(pv) / rho
    return U0, p0


def _steady_start(case0: Case, cdir: Path, opts: dict, sh: _Shared, warnings: list):
    """The steady solution of the values at t = 0: (U, p kinematic, path of its flux file). It is solved
    by the steady backend on the same mesh; the two OpenFOAM meshes must be the same cell for cell."""
    sdir = cdir / "steady_start"
    skw = {k: opts[k] for k in steady.KNOWN_OPTIONS
           if k in opts and k not in ("case_dir", "keep_case", "timeout") and opts[k] is not None}
    sres = steady.solve(case0, case_dir=str(sdir), timeout=opts.get("timeout"), **skw)
    for name in ("owner", "neighbour"):
        if (sdir / "constant" / "polyMesh" / name).read_bytes() != (cdir / "constant" / "polyMesh" / name).read_bytes():
            raise SolverError("initial steady state: the steady solve numbered the mesh differently")
    tdirs = sorted((d for d in sdir.iterdir() if d.is_dir() and steady._is_time(d.name) and float(d.name) > 0),
                   key=lambda d: float(d.name))
    n = sh.mesh.n_cells
    U, _ = fio.read_field(tdirs[-1] / "U", n)
    p, _ = fio.read_field(tdirs[-1] / "p", n)
    bad = [w for w in sres.info.warnings if "NOT CONVERGED" in w]
    if bad:
        warnings.append("initial state: the steady solve of the values at t = 0 did not converge, the run "
                        "starts from its last iterate (" + bad[0].replace("NOT CONVERGED: ", "") + ")")
    warnings.append(f"info: initial state = the steady solution of the values at t = 0 "
                    f"(simpleFoam, {sres.info.iterations} iterations, {sres.info.wall_time:.2f} s)")
    return U, p.ravel(), tdirs[-1] / "phi"


def _kept_steps(tr: dict, n: int, times: np.ndarray) -> np.ndarray:
    if tr.get("output_every") is not None:
        idx = list(range(0, n + 1, int(tr["output_every"]))) + [n]
    elif tr.get("output_times") is not None:
        idx = [int(np.argmin(np.abs(times - float(t)))) for t in tr["output_times"]] + [n]
    else:                                   # 10 states evenly spaced in time, the initial one and the last
        idx = list(np.round(np.linspace(0, n, 11)).astype(int))
    return np.unique(np.asarray(idx, dtype=np.int64))


def _run(case, fe, cdir: Path, opts, mat, scheme, conv, warnings, t0, keep) -> "FoamTransientResult":
    a = case.analysis
    tr = a.transient
    dim = case.dim
    rho, nu = float(mat.rho), float(mat.mu) / float(mat.rho)
    timeout = opts.get("timeout")
    times = a.time_steps()
    n_steps = len(times) - 1
    t_end, dt = float(times[-1]), float(times[-1]) / n_steps
    max_co = None if tr.get("max_co") is None else float(tr["max_co"])

    # ---- mesh and patches, as the steady backend
    fm = convert(case.mesh_path(), dim, float(a.thickness), cdir / "mesh.msh")
    warnings += fm.warnings
    fio.write_control(cdir, 1)              # ASCII, so that gmshToFoam writes a polyMesh foamlib reads
    run(fe, ["gmshToFoam", "mesh.msh"], cdir, timeout=timeout)
    (cdir / "system" / "controlDict").unlink()
    _write_control(cdir, t_end, dt, max_co)
    case0 = _frozen_case(case, 0.0)
    bcs0 = steady._patch_bcs(case0, fm, warnings)
    orig = {id(b0): b for b0, b in zip(case0.bcs, case.bcs)}
    bcs = {p: (orig[id(b)] if b is not None else None) for p, b in bcs0.items()}
    types = {p: (steady.PATCH_TYPE[bc.type] if bc is not None else "wall") for p, bc in bcs.items()}
    if dim == 2:
        types[FRONT_BACK] = "empty"
    fio.set_patch_types(cdir, types)
    (cdir / "mesh.msh").unlink()
    poly = fio.read_poly(cdir)
    sh = _Shared(poly, cdir, fm, dim)
    mesh = sh.mesh
    centres = {p: mesh.Cf[mesh.patch_slice(p)] for p in poly["patches"]}
    closed = not any(b.type == "pressure_outlet" for b in case.bcs)
    skew = opts["skew_correction"]

    # ---- boundary conditions: the steady backend's for the values at t = 0, then the values in time
    Ub, Pb, _, _, _ = steady._boundary_fields(case0, fm, bcs0, centres, rho, None)
    dir_u = {p for p, bc in bcs.items() if bc is None or bc.type in ("velocity_inlet", "wall")}
    dir_p = {p for p, bc in bcs.items() if bc is not None and bc.type == "pressure_outlet"}
    timed = []
    for p, bc in bcs.items():
        if bc is None or bc.value is None or not uses_time(bc.value):
            continue
        sl = mesh.patch_slice(p)
        Cf, Sf = mesh.Cf[sl], mesh.Sf[sl]
        ts = _sample_times(bc.value, times, warnings, f"the {bc.type} value on {bc.group!r}")
        if bc.type == "pressure_outlet":
            z = Cf[:, 2] if dim == 3 else None
            vals = np.array([as_field(bc.value, Cf[:, 0], Cf[:, 1], z, t) / rho for t in ts])
            Pb[p] = _timed_entry(cdir, p, "p", ts, vals, Cf)
            kind = Pb[p]["type"]
        else:
            vals = np.array([_velocity(bc, Cf, Sf, dim, t) for t in ts])
            Ub[p] = _timed_entry(cdir, p, "U", ts, vals, Cf)
            kind = Ub[p]["type"]
        timed.append(f"{bc.group!r} ({bc.type}): {kind}, {len(ts)} samples")
    if timed:
        warnings.append("info: boundary values in time: " + "; ".join(timed))

    # ---- initial state
    ini = tr.get("initial") or {}
    phi0 = None
    if ini.get("steady"):
        U0, p0, phi0 = _steady_start(case0, cdir, opts, sh, warnings)
    else:
        U0, p0 = _initial_fields(case, ini, sh, rho)
    Ucells = np.broadcast_to(np.asarray(U0, dtype=float), (mesh.n_cells, 3))
    for p, bc in bcs.items():               # the boundary value of the initial state where OpenFOAM
        if Ub[p].get("type") == "pressureInletOutletVelocity":    # extrapolates: the cell value
            Ub[p] = {**Ub[p], "value": Ucells[mesh.owner[mesh.patch_slice(p)]].copy()}
    fio.write_field(cdir, "U", "volVectorField", {"length": 1, "time": -1}, U0, Ub)
    fio.write_field(cdir, "p", "volScalarField", {"length": 2, "time": -2}, p0, Pb)
    if phi0 is not None:
        shutil.copyfile(phi0, cdir / "0" / "phi")
        shutil.rmtree(cdir / "steady_start", ignore_errors=True)

    # ---- dictionaries
    n_outer, n_corr = int(opts["n_outer_correctors"]), int(opts["n_correctors"])
    outer_tol, n_nonorth = float(opts["outer_tol"]), int(opts["n_non_orth_correctors"])
    lin = max(min(float(opts["residual_tol"]), 1e-6), 1e-15)
    _write_schemes(cdir, DDT[scheme], CONVECTION[conv], skew)
    _write_solution(cdir, lin, n_outer, outer_tol, n_corr, n_nonorth, closed)
    fio.write_transport(cdir, nu, "laminar")

    # ---- QoIs, evaluated at every step
    defs: dict = {}
    for q in case.qois:
        defs.setdefault(_qoi_key(q), _instant(q))
    need_grad = any(q.type == "force" for q in defs.values())
    want_probe = any(q.type in ("point", "point_difference") for q in defs.values())
    series = {k: np.full(n_steps + 1, np.nan) for k in defs}
    kept = _kept_steps(tr, n_steps, times)
    kept_set = set(int(i) for i in kept)
    dirs: dict = {}
    failed: dict = {}

    measure = fm.measure if dim == 2 else float(mesh.V.sum())
    info = TransientFlowInfo(
        backend="openfoam", physics="flow", formulation=a.formulation, order=int(a.order), dim=dim,
        cell_type=fm.cell_type, n_nodes=int(fm.n_nodes), n_cells=int(fm.n_cells),
        n_dofs=int(fm.n_cells * (dim + 1)), measure=float(measure),
        h_rep=float((measure / fm.n_cells) ** (1.0 / dim)), wall_time=0.0, warnings=warnings,
        residual_tol=float(opts["residual_tol"]), openfoam=fe.version, scheme=scheme, t_end=t_end,
        n_steps=n_steps, dt=dt)

    def new_state(tdir: Path, t: float, fast: bool) -> _State:
        st = _State(case, fm, cdir, rho, nu, closed, [])
        st.info = info
        st.time = float(t)
        st._want_probe = want_probe or not fast
        if fast:
            st._load_fast(sh, tdir, need_grad)
        else:
            st._load(poly, tdir)
        st._setup(sh, dir_u, dir_p)
        if not (tdir / "phi").exists():      # the initial state: boundary flux from the boundary velocity
            for name, d in st.patch.items():
                d["phi"] = np.einsum("ij,ij->i", d["U"], d["Sf"])
        return st

    diag = {"flux": np.full(n_steps + 1, np.nan)}

    def evaluate(i: int, tdir: Path) -> None:
        st = new_state(tdir, times[i], fast=True)
        diag["flux"][i] = st.boundary_flux()
        for key, q in defs.items():
            if key in failed:
                continue
            try:
                series[key][i] = float(st.evaluate_qoi(q))
            except (CaseError, MeshError, ValueError) as e:
                failed[key] = f"qoi {q.name!r} could not be evaluated: {e}"

    evaluate(0, cdir / "0")
    dirs[0] = cdir / "0"

    # ---- run: read each written step while the solver runs
    step = {"i": 0}

    def on_write(name: str) -> None:
        i = step["i"] + 1
        if i > n_steps:
            raise SolverError(f"pimpleFoam wrote more than the {n_steps} steps asked for (time {name})")
        tdir = cdir / name
        if abs(float(name) - times[i]) > 1e-3 * dt:
            raise SolverError(f"pimpleFoam wrote time {name}, step {i} is at {times[i]:.12g}")
        evaluate(i, tdir)
        prev = dirs.get(i - 1)
        if prev is not None and (i - 1) not in kept_set:
            shutil.rmtree(prev, ignore_errors=True)
            dirs.pop(i - 1)
        dirs[i] = tdir
        step["i"] = i

    t_solve = time.perf_counter()
    log = _run_pimple(fe, cdir, on_write, timeout, keep)
    t_solve = time.perf_counter() - t_solve
    if step["i"] != n_steps:
        raise SolverError(f"pimpleFoam stopped after {step['i']} of {n_steps} steps; last lines:\n{log.tail()}")

    # ---- result
    for msg in failed.values():
        warnings.append(msg)
    if closed:
        warnings.append("info: no pressure outlet (closed domain): p is defined up to a constant; it is "
                        "shifted to a zero volume mean")
    not_conv = int(np.sum(log.outer >= n_outer)) if n_outer > 1 else 0
    converged = not_conv == 0
    co_max = float(np.max(log.co_max)) if len(log.co_max) else 0.0
    co_mean = float(np.max(log.co_mean)) if len(log.co_mean) else 0.0
    dts = np.diff(np.asarray(log.times, dtype=float), prepend=0.0)
    dt_min = float(dts.min()) if len(dts) else dt
    if n_outer > 1 and not converged:
        warnings.append(f"NOT CONVERGED: in {not_conv} of {log.steps} steps the outer loop did not reach "
                        f"outer_tol {outer_tol:g} in {n_outer} iterations (Courant number up to {co_max:.2f}): "
                        f"reduce the step or raise n_outer_correctors")
    if n_outer == 1:
        warnings.append("info: n_outer_correctors = 1 is the PISO algorithm: the convecting flux is the "
                        "previous step's, an error of first order in the step (docs/solver_transient.md)")
        if co_max > 1.0:
            warnings.append(f"the Courant number reached {co_max:.2f} > 1 with n_outer_correctors = 1 (PISO): "
                            f"reduce the step or allow outer correctors")
    coupling = (f"PIMPLE, up to {n_outer} outer correctors to {outer_tol:g} (at most {int(log.outer.max())} used, "
                f"mean {float(log.outer.mean()):.1f})" if n_outer > 1 else "PISO")
    step_txt = f"step {dt:.6g} s" if max_co is None else (
        f"step adapted to Courant {max_co:g} below {dt:.6g} s ({log.steps} solver steps, smallest {dt_min:.3g} s)")
    warnings.append(f"info: transient flow, pimpleFoam: scheme {scheme} ({'second' if scheme == 'backward' else 'first'}"
                    f" order in time), {coupling}, {n_corr} pressure correctors, convection {conv}; {n_steps} steps "
                    f"to t = {t_end:g} s, {step_txt}")
    warnings.append(f"info: Courant number reached {co_max:.3g} (cell mean {co_mean:.3g}); continuity error "
                    f"(sum local) at most {float(np.max(log.cont)) if len(log.cont) else 0.0:.2e}")
    wall = time.perf_counter() - t0
    warnings.append(f"info: {fe.describe()}; pimpleFoam {log.steps} steps in {t_solve:.2f} s "
                    f"({1e3 * t_solve / max(log.steps, 1):.1f} ms per step, QoIs included); wall time {wall:.2f} s")
    info.wall_time = float(wall)
    info.iterations, info.converged = int(log.steps), bool(converged)
    info.residuals = {k: float(v) for k, v in fio.fold(log.last_res).items()}
    info.dt_min, info.solver_steps = dt_min, int(log.steps)
    info.courant_max, info.courant_mean = co_max, co_mean
    info.outer_iterations = int(log.outer.max())
    info.continuity_max = float(np.max(log.cont)) if len(log.cont) else 0.0
    res = FoamTransientResult(case, info, times, kept, series, defs, dirs, new_state)
    res.boundary_flux = diag["flux"]
    res.solver_times = np.asarray(log.times, dtype=float)
    res.courant = np.asarray(log.co_max, dtype=float)
    res.continuity = np.asarray(log.cont, dtype=float)
    res.outer = np.asarray(log.outer, dtype=np.int64)
    return res


class _Log:
    """The numbers of a pimpleFoam log, step by step (the log itself is not kept in memory)."""

    def __init__(self):
        self.times: list = []        # the time of every solver step
        self.co_max: list = []       # Courant number (max) the step started from
        self.co_mean: list = []
        self.cont: list = []         # continuity error "sum local" after the step's last corrector
        self.outer_n: list = []      # outer iterations of the step
        self.last_res: dict = {}
        self._tail: deque = deque(maxlen=40)
        self._co = (0.0, 0.0)
        self._cur: dict | None = None

    @property
    def steps(self) -> int:
        return len(self.times)

    @property
    def outer(self) -> np.ndarray:
        return np.asarray(self.outer_n, dtype=np.int64) if self.outer_n else np.zeros(1, dtype=np.int64)

    def tail(self) -> str:
        return "".join(self._tail).rstrip()

    def feed(self, line: str) -> str | None:
        """Returns the time name when a step is complete (its ``ExecutionTime`` line)."""
        self._tail.append(line)
        if line.startswith("Courant Number mean:"):
            w = line.split()
            try:
                self._co = (float(w[5]), float(w[3]))
            except (IndexError, ValueError):
                pass
        elif line.startswith("Time = "):
            self._cur = {"name": line[7:].strip(), "outer": 0, "res": {}, "cont": 0.0}
        elif self._cur is not None:
            cur = self._cur
            if line.startswith("PIMPLE: iteration"):
                cur["outer"] += 1
                cur["res"] = {}
            elif "Solving for" in line:
                m = fio._RES.search(line)
                if m:
                    cur["res"].setdefault(m.group(1), float(m.group(2)))
            elif line.startswith("time step continuity errors"):
                try:
                    cur["cont"] = abs(float(line.split("sum local =")[1].split(",")[0]))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("ExecutionTime"):
                self.times.append(float(cur["name"]))
                self.co_max.append(self._co[0])
                self.co_mean.append(self._co[1])
                self.cont.append(cur["cont"])
                self.outer_n.append(max(cur["outer"], 1))
                self.last_res = cur["res"]
                self._cur = None
                return cur["name"]
        return None


def _run_pimple(fe, cdir: Path, on_write, timeout, keep_log: bool) -> _Log:
    """Run pimpleFoam; after every step that wrote its time directory call ``on_write(time name)``. The
    solver is killed on a timeout or an error here, so no process is left behind."""
    env = dict(fe.env)
    env["PWD"] = str(cdir)
    log = _Log()
    t_start = time.monotonic()
    logf = open(cdir / "log.pimpleFoam", "w") if keep_log else None
    proc = subprocess.Popen(["pimpleFoam"], cwd=cdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if logf is not None:
                logf.write(line)
            name = log.feed(line)
            if name is not None and os.path.isdir(cdir / name):
                on_write(name)
            if timeout is not None and time.monotonic() - t_start > timeout:
                proc.kill()
                raise SolverError(f"pimpleFoam stopped by the {timeout} s timeout after {log.steps} steps "
                                  f"(t = {log.times[-1] if log.times else 0:g} s)")
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
        if logf is not None:
            logf.close()
    tail = log.tail()
    bad = [k for k, v in log.last_res.items() if not np.isfinite(v)]
    if code != 0 or "FOAM FATAL" in tail or bad:
        raise SolverError(f"pimpleFoam failed (exit {code}) after {log.steps} steps; last lines:\n{tail}")
    return log


# --------------------------------------------------------------------------- result


class FoamTransientResult(TransientResult):
    """A transient flow solution. ``times``: every step from 0 to t_end; ``kept``: the steps whose state
    is kept; ``series(qoi)``: the QoI at every step, for the QoIs of the case (whatever their ``time`` and
    ``window``); ``state(t)``: the kept state nearest to t, a steady-style result (probe, nodal,
    evaluate of the instantaneous value, write_vtu) with its ``time``.

    Diagnostics per result step: ``boundary_flux`` (the sum of the face flux over the boundary, m^3/s);
    per solver step: ``solver_times``, ``courant`` (max), ``continuity`` (OpenFOAM's "sum local"),
    ``outer`` (outer iterations)."""

    def __init__(self, case: Case, info: TransientFlowInfo, times, kept, series: dict, defs: dict, dirs: dict,
                 loader):
        self.case = case
        self.info = info
        self.times = np.asarray(times, dtype=float)
        self.kept = np.asarray(kept, dtype=np.int64)
        self._series = series
        self._defs = defs
        self._dirs = dirs
        self._loader = loader
        self._states: dict = {}
        self._tmp = None

    def series(self, qoi: QoI) -> np.ndarray:
        key = _qoi_key(qoi)
        if key not in self._series:
            raise CaseError(f"QoI {qoi.name!r} was not evaluated during the run: a series exists for the QoIs "
                            f"of the case ({sorted(q.name for q in self._defs.values())}); for another quantity "
                            f"evaluate the kept states (result.state(t).evaluate(qoi))")
        return self._series[key].copy()

    def state(self, t: float | None = None) -> _State:
        tk = self.times[self.kept]
        i = int(self.kept[-1] if t is None else self.kept[int(np.argmin(np.abs(tk - float(t))))])
        if i not in self._states:
            tdir = self._dirs.get(i)
            if tdir is None or not Path(tdir).is_dir():
                raise RuntimeError("the OpenFOAM case of this result was deleted (cleanup)")
            st = self._loader(tdir, self.times[i], False)
            self._states[i] = st
        return self._states[i]

    def evaluate(self, qoi: QoI) -> float:
        try:
            return super().evaluate(qoi)
        except ValueError as e:              # a reduction that the series cannot give (too few periods, ...)
            raise CaseError(f"QoI {qoi.name!r}: {e}") from None

    def evaluate_all(self) -> dict[str, float]:
        out = {}
        for q in self.case.qois:
            try:
                out[q.name] = float(self.evaluate(q))
            except (CaseError, ValueError) as e:
                out[q.name] = float("nan")
                msg = f"qoi {q.name!r} could not be evaluated: {e}"
                if msg not in self.info.warnings:
                    self.info.warnings.append(msg)
        return out

    def summary(self) -> dict:
        qois = self.evaluate_all()
        return {"info": asdict(self.info), "qois": qois, "meta": self.case.meta}

    def write_vtu(self, path, t: float | None = None):
        return self.state(t).write_vtu(path)

    def cleanup(self) -> None:
        """Delete the OpenFOAM case now (otherwise when the result is garbage collected). The states that
        were already loaded stay usable."""
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
