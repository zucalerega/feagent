"""OpenFOAM backend: steady incompressible flow (``feacore.solver.solve(case, backend="openfoam")``).

Physics "flow", formulations "2d" (one extruded cell layer of ``analysis.thickness``, ``empty``
front and back) and "3d". Solver: ``simpleFoam`` with SIMPLEC (``consistent yes``), laminar by
default or a RANS model (``analysis.options["turbulence"]``, e.g. ``"kOmegaSST"``). Convection:
``analysis.order`` 1 = ``bounded Gauss upwind``, 2 = ``bounded Gauss linearUpwind grad(U)``;
everything else second order (Gauss linear, corrected). Dictionaries follow the shipped
``simpleFoam/pitzDaily`` tutorial and the ``inflowOutflow`` template; they are written with
``foamlib``. The mesh goes through ``gmshToFoam``; forces come from OpenFOAM's ``forces``
function object (pressure + viscous), cell centres, volumes and gradients from
``writeCellCentres`` / ``writeCellVolumes`` / ``grad`` (``simpleFoam -postProcess``).

Units: SI as given by the case. OpenFOAM works with kinematic pressure p / rho; the backend
converts both ways, so case values and results are static pressure in Pa. 2D results (forces,
flow rates, integrals) are per ``analysis.thickness``.

Options (``analysis.options`` or keywords): ``turbulence`` ("laminar"), ``max_iterations`` (5000),
``residual_tol`` (1e-8: initial residuals of p and U, as ``residualControl``), ``relaxation_U``
(0.9), ``relaxation_p`` (none: SIMPLEC), ``n_non_orth_correctors`` (0), ``timeout`` (s),
``case_dir`` (run there instead of a temp dir and keep it), ``keep_case`` (keep the temp dir),
``turbulence_intensity`` (0.05), ``turbulence_length`` (0.07 * hydraulic size). See
docs/solver_openfoam.md.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from feacore.case import FLOW_FIELDS, Case, CaseError, QoI, as_field
from feacore.solver import _foam_io as fio
from feacore.solver._foam_env import require_openfoam, run
from feacore.solver._foam_fv import FvMesh
from feacore.solver._foam_mesh import FRONT_BACK, UNASSIGNED, FoamMesh, convert
from feacore.solver._skfem_mesh import MeshError
from feacore.solver.base import Result, SolveInfo

KNOWN_OPTIONS = ("turbulence", "max_iterations", "residual_tol", "relaxation_U", "relaxation_p",
                 "n_non_orth_correctors", "timeout", "case_dir", "keep_case", "turbulence_intensity",
                 "turbulence_length")
DEFAULTS = {"turbulence": "laminar", "max_iterations": 5000, "residual_tol": 1e-8, "relaxation_U": 0.9,
            "relaxation_p": None, "n_non_orth_correctors": 0, "timeout": None, "turbulence_intensity": 0.05}
RAS_MODELS = ("kOmegaSST", "kOmega", "kEpsilon", "realizableKE", "RNGkEpsilon", "SpalartAllmaras")
PATCH_TYPE = {"velocity_inlet": "patch", "pressure_outlet": "patch", "wall": "wall", "slip": "patch",
              "symmetry": "symmetry"}


class SolverError(RuntimeError):
    """The flow solve failed (diverged, crashed or produced no result)."""


@dataclass
class FlowSolveInfo(SolveInfo):
    iterations: int = 0
    converged: bool = False                 # residual_tol reached on every equation
    residuals: dict = field(default_factory=dict)   # last initial residual per equation
    residual_tol: float = 0.0
    openfoam: str = ""                      # OpenFOAM version used
    case_dir: str | None = None             # kept OpenFOAM case (keep_case / case_dir), else None


# --------------------------------------------------------------------------- setup helpers


def _vec3(v) -> list:
    v = list(v) + [0.0] * (3 - len(v))
    return v[:3]


def _patch_bcs(case: Case, fm: FoamMesh, warnings: list) -> dict:
    """{patch: BC or None (default wall)}; front/back is handled separately."""
    out: dict = {}
    for p, groups in fm.patches.items():
        hits = [b for b in case.bcs if b.group in groups]
        if len({b.type for b in hits}) > 1 or len(hits) > 1:
            warnings.append(f"patch {p!r} (groups {list(groups)}) is claimed by several BCs "
                            f"{[(b.type, b.group) for b in hits]}: the first one is used")
        out[p] = hits[0] if hits else None
    for b in case.bcs:
        if b.group not in fm.group_patches:
            fm.patches_of(b.group)          # raises a MeshError naming the problem
    nobc = sorted({g for p, bc in out.items() if bc is None for g in (fm.patches[p] or (UNASSIGNED,))})
    if nobc:
        warnings.append(f"info: boundary groups without a BC are no-slip walls: {nobc}")
    return out


def _value_at(value, X: np.ndarray, dim: int) -> np.ndarray:
    """Vector BC value (entries scalar or expression) at face centres X (n, 3) -> (n, 3)."""
    z = X[:, 2] if dim == 3 else None
    comps = [as_field(v, X[:, 0], X[:, 1], z) for v in value]
    while len(comps) < 3:
        comps.append(np.zeros(len(X)))
    return np.column_stack(comps)


def _is_expr(value) -> bool:
    if isinstance(value, str):
        return True
    return isinstance(value, list) and any(isinstance(v, str) for v in value)


def _boundary_fields(case: Case, fm: FoamMesh, bcs: dict, centres: dict, rho: float, turb: dict | None):
    """boundaryField dicts for U, p (and k, omega, nut)."""
    dim = case.dim
    U, P, K, W, NUT = {}, {}, {}, {}, {}
    zero = [0.0, 0.0, 0.0]
    for p, bc in bcs.items():
        t = bc.type if bc is not None else "wall"
        if t == "velocity_inlet":
            v = bc.value
            if isinstance(v, list):
                if _is_expr(v):
                    U[p] = {"type": "fixedValue", "value": _value_at(v, centres[p], dim)}
                else:
                    U[p] = {"type": "fixedValue", "value": _vec3([float(x) for x in v])}
            else:   # scalar (or expression) speed along the inward normal
                if isinstance(v, str):
                    X = centres[p]
                    ref = -as_field(v, X[:, 0], X[:, 1], X[:, 2] if dim == 3 else None)
                    U[p] = {"type": "surfaceNormalFixedValue", "refValue": ref, "value": np.zeros((len(X), 3))}
                else:
                    U[p] = {"type": "surfaceNormalFixedValue", "refValue": -float(v), "value": zero}
            P[p] = {"type": "zeroGradient"}
        elif t == "pressure_outlet":
            U[p] = {"type": "pressureInletOutletVelocity", "value": zero}
            if isinstance(bc.value, str):
                X = centres[p]
                P[p] = {"type": "fixedValue",
                        "value": as_field(bc.value, X[:, 0], X[:, 1], X[:, 2] if dim == 3 else None) / rho}
            else:
                P[p] = {"type": "fixedValue", "value": float(bc.value) / rho}
        elif t == "wall":
            if bc is not None and bc.value is not None:
                if _is_expr(bc.value):
                    U[p] = {"type": "fixedValue", "value": _value_at(bc.value, centres[p], dim)}
                else:
                    U[p] = {"type": "fixedValue", "value": _vec3([float(x) for x in bc.value])}
            else:
                U[p] = {"type": "noSlip"}
            P[p] = {"type": "zeroGradient"}
        elif t == "slip":
            U[p] = {"type": "slip"}
            P[p] = {"type": "zeroGradient"}
        elif t == "symmetry":
            U[p] = {"type": "symmetry"}
            P[p] = {"type": "symmetry"}
        else:  # pragma: no cover - validated by Case.check
            raise CaseError(f"unsupported flow BC {t!r}")
        if turb is not None:
            if t == "velocity_inlet":
                K[p] = {"type": "fixedValue", "value": turb["k"]}
                W[p] = {"type": "fixedValue", "value": turb["omega"]}
                NUT[p] = {"type": "calculated", "value": 0.0}
            elif t == "wall":
                K[p] = {"type": "kqRWallFunction", "value": turb["k"]}
                W[p] = {"type": "omegaWallFunction", "value": turb["omega"]}
                NUT[p] = {"type": "nutkWallFunction", "value": 0.0}
            elif t in ("slip",):
                K[p] = {"type": "zeroGradient"}
                W[p] = {"type": "zeroGradient"}
                NUT[p] = {"type": "calculated", "value": 0.0}
            elif t == "symmetry":
                K[p] = W[p] = NUT[p] = {"type": "symmetry"}
            else:   # pressure outlet
                K[p] = {"type": "inletOutlet", "inletValue": turb["k"], "value": turb["k"]}
                W[p] = {"type": "inletOutlet", "inletValue": turb["omega"], "value": turb["omega"]}
                NUT[p] = {"type": "calculated", "value": 0.0}
    if fm.dim == 2:
        for d in (U, P, K, W, NUT):
            d[FRONT_BACK] = {"type": "empty"}
    return U, P, K, W, NUT


# --------------------------------------------------------------------------- solve


def solve(case: Case, **options) -> "OpenFOAMResult":
    t0 = time.perf_counter()
    case.check()
    a = case.analysis
    if a.physics != "flow":
        raise CaseError(f"the openfoam backend solves physics 'flow' only (got {a.physics!r})")
    if fio.FOAMLIB_ERROR is not None:
        raise ImportError(f"the openfoam backend needs foamlib (pip install foamlib): {fio.FOAMLIB_ERROR}")
    if len(case.materials) != 1 or (case.regions and len({r.material for r in case.regions}) > 1):
        raise CaseError("flow cases take a single fluid material")
    fe = require_openfoam()
    opts = {**DEFAULTS, **(a.options or {}), **options}
    warnings: list[str] = []
    unknown = sorted(set(a.options or {}) - set(KNOWN_OPTIONS))
    if unknown:
        warnings.append(f"unknown analysis options ignored: {unknown}")
    mat = case.material_for(None)
    rho, mu = float(mat.rho), float(mat.mu)
    nu = mu / rho
    turbulence = str(opts["turbulence"])
    if turbulence != "laminar" and turbulence not in RAS_MODELS:
        raise CaseError(f"analysis.options.turbulence must be 'laminar' or one of {RAS_MODELS}")
    if turbulence not in ("laminar", "kOmegaSST", "kOmega"):
        raise CaseError(f"turbulence model {turbulence!r} is not wired up yet (use laminar or kOmegaSST)")
    tol = float(opts["residual_tol"])
    max_it = int(opts["max_iterations"])
    mesh_path = case.mesh_path()
    if not Path(mesh_path).exists():
        raise MeshError(f"mesh file not found: {str(mesh_path)!r}")

    keep = bool(opts.get("keep_case")) or opts.get("case_dir") is not None
    if opts.get("case_dir") is not None:
        cdir = Path(opts["case_dir"]).resolve()
        if cdir.exists() and any(cdir.iterdir()):
            raise ValueError(f"case_dir {str(cdir)!r} exists and is not empty")
        cdir.mkdir(parents=True, exist_ok=True)
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="feacore_foam_")
        cdir = Path(tmp.name)
    try:
        res = _run(case, fe, cdir, opts, rho, nu, turbulence, tol, max_it, warnings, t0)
    except BaseException:
        if tmp is not None and not keep:
            tmp.cleanup()
        raise
    res._tmp = None if keep else tmp
    if keep and tmp is not None:
        # detach the directory from TemporaryDirectory's finaliser
        tmp._finalizer.detach()  # type: ignore[attr-defined]
        res.info.case_dir = str(cdir)
    elif keep:
        res.info.case_dir = str(cdir)
    return res


def _run(case, fe, cdir: Path, opts, rho, nu, turbulence, tol, max_it, warnings, t0) -> "OpenFOAMResult":
    a = case.analysis
    dim = case.dim
    timeout = opts.get("timeout")
    fm = convert(case.mesh_path(), dim, float(a.thickness), cdir / "mesh.msh")
    warnings += fm.warnings
    fio.write_control(cdir, max_it)
    run(fe, ["gmshToFoam", "mesh.msh"], cdir, timeout=timeout)
    bcs = _patch_bcs(case, fm, warnings)
    types = {p: (PATCH_TYPE[bc.type] if bc is not None else "wall") for p, bc in bcs.items()}
    if dim == 2:
        types[FRONT_BACK] = "empty"
    fio.set_patch_types(cdir, types)
    (cdir / "mesh.msh").unlink()

    # boundary-face centres (expression BCs) from the polyMesh
    poly = fio.read_poly(cdir)
    centres = {}
    for p, (n, start, _) in poly["patches"].items():
        Cf, _ = fio.face_geometry(poly["points"], poly["faces"][start:start + n])
        centres[p] = Cf

    turbulent = turbulence != "laminar"
    turb = _turbulence_inlet(case, fm, opts, nu) if turbulent else None
    closed = not any(b.type == "pressure_outlet" for b in case.bcs)
    ref = {"pRefCell": 0, "pRefValue": 0.0} if closed else None
    fio.write_schemes(cdir, int(a.order), turbulent)
    fio.write_solution(cdir, tol, turbulent, ref, float(opts["relaxation_U"]),
                       None if opts.get("relaxation_p") is None else float(opts["relaxation_p"]),
                       int(opts["n_non_orth_correctors"]))
    fio.write_transport(cdir, nu, turbulence)
    U, P, K, W, NUT = _boundary_fields(case, fm, bcs, centres, rho, turb)
    p0 = 0.0
    fio.write_field(cdir, "U", "volVectorField", {"length": 1, "time": -1}, [0.0, 0.0, 0.0], U)
    fio.write_field(cdir, "p", "volScalarField", {"length": 2, "time": -2}, p0, P)
    if turbulent:
        fio.write_field(cdir, "k", "volScalarField", {"length": 2, "time": -2}, turb["k"], K)
        fio.write_field(cdir, "omega", "volScalarField", {"time": -1}, turb["omega"], W)
        fio.write_field(cdir, "nut", "volScalarField", {"length": 2, "time": -1}, 0.0, NUT)

    t_solve = time.perf_counter()
    proc = run(fe, ["simpleFoam"], cdir, timeout=timeout, check=False)
    log = proc.stdout + "\n" + proc.stderr
    st = fio.parse_log(log)
    if proc.returncode != 0 or st["diverged"]:
        tail = "\n".join(log.strip().splitlines()[-20:])
        raise SolverError(f"simpleFoam failed (exit {proc.returncode}, {st['iterations']} iterations"
                          f"{', diverged' if st['diverged'] else ''}); last lines:\n{tail}")
    t_solve = time.perf_counter() - t_solve
    times = sorted((d for d in cdir.iterdir() if d.is_dir() and _is_time(d.name) and float(d.name) > 0),
                   key=lambda d: float(d.name))
    if not times:
        raise SolverError("simpleFoam wrote no solution")
    tdir = times[-1]
    worst = max(st["residuals"].values()) if st["residuals"] else float("nan")
    converged = st["converged"] or (st["residuals"] and worst <= tol)
    if not converged:
        warnings.append(f"NOT CONVERGED: residual_tol {tol:g} not reached in {st['iterations']} iterations "
                        f"(final initial residuals {_fmt(st['residuals'])}); the iteration error may "
                        f"exceed the discretisation error: raise max_iterations or check the setup")

    res = OpenFOAMResult(case, fm, cdir, rho, nu, closed, warnings)
    res._load(poly, tdir)
    wall = time.perf_counter() - t0
    measure = fm.measure if dim == 2 else float(res.V.sum())
    info = FlowSolveInfo(
        backend="openfoam", physics="flow", formulation=a.formulation, order=int(a.order), dim=dim,
        cell_type=fm.cell_type, n_nodes=int(fm.n_nodes), n_cells=int(fm.n_cells),
        n_dofs=int(fm.n_cells * (dim + 1)), measure=float(measure),
        h_rep=float((measure / fm.n_cells) ** (1.0 / dim)), wall_time=float(wall), warnings=warnings,
        iterations=int(st["iterations"]), converged=bool(converged),
        residuals={k: float(v) for k, v in st["residuals"].items()}, residual_tol=tol,
        openfoam=fe.version)
    warnings.append(f"info: {fe.describe()}; simpleFoam {st['iterations']} iterations in {t_solve:.2f} s, "
                    f"final initial residuals {_fmt(st['residuals'])}")
    res.info = info
    return res


def _fmt(r: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:.2e}" for k, v in sorted(r.items())) + "}"


def _is_time(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _turbulence_inlet(case: Case, fm: FoamMesh, opts: dict, nu: float) -> dict:
    """k and omega from a turbulence intensity and a length scale (inlet speed from the BCs)."""
    speeds = []
    for b in case.bcs:
        if b.type == "velocity_inlet" and not _is_expr(b.value):
            speeds.append(float(np.linalg.norm(b.value)) if isinstance(b.value, list) else abs(float(b.value)))
    Uref = max(speeds) if speeds else 1.0
    I = float(opts.get("turbulence_intensity") or 0.05)
    size = float(np.min(np.ptp(fm.bbox, axis=0))) if fm.bbox is not None else 1.0
    L = float(opts.get("turbulence_length") or 0.07 * size)
    k = 1.5 * (Uref * I) ** 2
    omega = math.sqrt(k) / (0.09 ** 0.25 * L)
    return {"k": k, "omega": omega}


# --------------------------------------------------------------------------- result


class OpenFOAMResult(Result):
    """Cell-centred finite-volume solution. ``nodal`` gives cell centres and cell values; ``probe``
    reconstructs linearly from the nearest cell centre (cell value + Gauss gradient . offset)."""

    def __init__(self, case: Case, fm: FoamMesh, cdir: Path, rho: float, nu: float, closed: bool,
                 warnings: list):
        self.case = case
        self.fm = fm
        self.cdir = cdir
        self.rho = rho
        self.nu = nu
        self.closed = closed
        self.dim = case.dim
        self._warnings = warnings
        self._tree = None
        self._tmp = None
        self.forces: dict = {}

    # ---- load results
    def _load(self, poly: dict, tdir: Path) -> None:
        nb = fio.read_list(self.cdir / "constant" / "polyMesh" / "neighbour").astype(np.int64)
        mesh = FvMesh(poly["points"], poly["faces"], poly["owner"], nb, poly["patches"])
        self.mesh = mesh
        n = mesh.n_cells
        if n != self.fm.n_cells:
            raise SolverError(f"OpenFOAM mesh has {n} cells, the input mesh {self.fm.n_cells}")
        U, bU = fio.read_field(tdir / "U", n)
        p, bp = fio.read_field(tdir / "p", n)
        p = p.ravel()
        nut_b, nut = {}, None
        if (tdir / "nut").exists():
            nut, bnut = fio.read_field(tdir / "nut", n)
            nut = nut.ravel()
        bphi = fio.FoamFieldFile(tdir / "phi").boundary_field if (tdir / "phi").exists() else {}
        self.C, self.V = mesh.C, mesh.V
        # pressure level of a closed domain: zero volume mean (a mesh-independent definition)
        self.pshift = float(np.sum(p * self.V) / np.sum(self.V)) if self.closed else 0.0
        if self.closed:
            self._warnings.append("info: no pressure outlet (closed domain): p is defined up to a constant; "
                                  "it is shifted to a zero volume mean")
        self.p_kin = p
        self.U = U
        self.patch: dict = {}
        Ub, pb = {}, {}
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
            nu_b = np.full(nf, self.nu)
            if nut is not None:
                nutv = fio.patch_values(bnut, name, nut[own])
                nu_b = self.nu + np.broadcast_to(np.asarray(nutv, dtype=float), (nf,))
            Ub[name], pb[name], nut_b[name] = uf, pf, nu_b
            self.patch[name] = {"Cf": mesh.Cf[sl], "Sf": mesh.Sf[sl], "U": uf, "p": (pf - self.pshift) * self.rho,
                                "phi": ph}
        self._Ub, self._pb_kin, self._nu_b = Ub, pb, nut_b
        self.gradU = mesh.grad(U, Ub)                         # [c, i, j] = d U_j / d x_i
        self.gradp = mesh.grad(p, pb) * self.rho
        self.p = (p - self.pshift) * self.rho

    # ---- fields
    def _check_field(self, fld):
        if fld not in FLOW_FIELDS:
            raise CaseError(f"unknown field {fld!r} for flow; available: {FLOW_FIELDS}")

    def _cell_values(self, fld):
        self._check_field(fld)
        if fld == "p":
            return self.p
        if fld == "u_mag":
            return np.linalg.norm(self.U, axis=1)
        return self.U[:, "xyz".index(fld[1])]

    def _face_values(self, fld, d):
        if fld == "p":
            return d["p"]
        if fld == "u_mag":
            return np.linalg.norm(d["U"], axis=1)
        return d["U"][:, "xyz".index(fld[1])]

    def nodal(self, fld):
        return self.C[:, :self.dim].copy(), np.asarray(self._cell_values(fld), dtype=float).copy()

    def probe(self, fld, points):
        self._check_field(fld)
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        X = np.zeros((len(pts), 3))
        X[:, :min(3, pts.shape[1])] = pts[:, :3]
        if self.dim == 2:
            X[:, 2] = 0.5 * self.fm.thickness
        if self._tree is None:
            self._tree = cKDTree(self.C[:, :self.dim])
            self._hloc = (np.abs(self.V) / (self.fm.thickness if self.dim == 2 else 1.0)) ** (1.0 / self.dim)
        dist, c = self._tree.query(X[:, :self.dim])
        far = dist > 2.0 * self._hloc[c] + 1e-12
        if np.any(far):
            raise CaseError(f"probe point(s) outside the mesh: {pts[far][:3].tolist()}")
        dx = X - self.C[c]
        if self.dim == 2:
            dx[:, 2] = 0.0
        if fld == "p":
            return self.p[c] + np.einsum("ij,ij->i", self.gradp[c], dx)
        u = self.U[c] + np.einsum("ij,ijk->ik", dx, self.gradU[c])
        if fld == "u_mag":
            return np.linalg.norm(u, axis=1)
        return u[:, "xyz".index(fld[1])]

    # ---- groups
    def _group_faces(self, group):
        return [self.patch[p] for p in self.fm.patches_of(group) if p in self.patch]

    def _cells_of(self, group):
        if group in (None, "*"):
            return np.arange(len(self.V))
        if group in self.fm.cell_groups:
            return self._cell_map()[self.fm.cell_groups[group]]
        return None

    def _cell_map(self):
        """Input-mesh cell index -> OpenFOAM cell index (matched by centroids)."""
        if getattr(self, "_cmap", None) is None:
            d, idx = cKDTree(self.C[:, :self.dim]).query(self.fm.centroids[:, :self.dim])
            if len(np.unique(idx)) != len(idx):
                raise CaseError("cannot match the input cells to the OpenFOAM cells")
            self._cmap = idx
        return self._cmap

    def force(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        """(pressure, viscous) fluid force vectors on the body through ``group``."""
        if group not in self.forces:
            names = [p for p in self.fm.patches_of(group) if p in self.patch]
            self.forces[group] = self.mesh.forces(names, self._pb_kin, self.U, self.gradU / 1.0, self._Ub,
                                                  self._nu_b, self.rho, p_ref=self.pshift)
        return self.forces[group]

    def evaluate(self, qoi: QoI) -> float:
        t = qoi.type
        if t == "point_difference":
            return Result.evaluate_qoi(self, qoi)
        if t == "point":
            return float(self.probe(qoi.field, [qoi.point[: self.dim]])[0])
        if t in ("max", "min", "max_abs"):
            cells = self._cells_of(qoi.group)
            if cells is not None:
                v = self._cell_values(qoi.field)[cells]
            else:
                v = np.concatenate([self._face_values(qoi.field, d) for d in self._group_faces(qoi.group)])
            return float(v.max() if t == "max" else v.min() if t == "min" else np.abs(v).max())
        if t in ("integral", "mean"):
            self._check_field(qoi.field)
            cells = self._cells_of(qoi.group)
            if cells is not None:
                w = self.V[cells]
                v = self._cell_values(qoi.field)[cells]
            else:
                fl = self._group_faces(qoi.group)
                w = np.concatenate([np.linalg.norm(d["Sf"], axis=1) for d in fl])
                v = np.concatenate([self._face_values(qoi.field, d) for d in fl])
            integ = float(np.sum(v * w))
            return integ if t == "integral" else integ / float(np.sum(w))
        if t == "flow_rate":
            return float(sum(d["phi"].sum() for d in self._group_faces(qoi.group)))
        if t == "force":
            fp, fv = self.force(qoi.group)
            return float(fp[qoi.component] + fv[qoi.component])
        raise CaseError(f"QoI type {t!r} is not available for flow")

    def evaluate_all(self) -> dict[str, float]:
        out = {}
        for q in self.case.qois:
            try:
                out[q.name] = float(self.evaluate_qoi(q))
            except (CaseError, ValueError) as e:
                out[q.name] = float("nan")
                msg = f"qoi {q.name!r} could not be evaluated: {e}"
                if msg not in self.info.warnings:
                    self.info.warnings.append(msg)
        return out

    def summary(self) -> dict:
        qois = self.evaluate_all()
        return {"info": asdict(self.info), "qois": qois, "meta": self.case.meta}

    def write_vtu(self, path):
        """Cell centres with the cell values as a VTK point cloud."""
        import meshio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cells = [("vertex", np.arange(len(self.C)).reshape(-1, 1))]
        pdata = {f: self._cell_values(f) for f in FLOW_FIELDS}
        pdata["U"] = self.U
        meshio.Mesh(self.C, cells, point_data=pdata).write(str(path))
        return path

    def cleanup(self) -> None:
        """Delete the OpenFOAM case directory now (otherwise when the result is garbage collected)."""
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
