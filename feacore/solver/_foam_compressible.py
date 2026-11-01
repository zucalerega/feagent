"""OpenFOAM backend, compressible flow (physics "compressible"): ``rhoCentralFoam`` run to a steady state.

The fixed solver for supersonic flow of a perfect gas (``feacore.solver.solve(case, backend="openfoam")``
dispatches here). OpenFOAM's density-based ``rhoCentralFoam``: central-upwind fluxes (Kurganov &
Tadmor, option ``flux_scheme``), limited reconstruction of rho, U and T, explicit time
stepping at the Courant number ``max_co``; the dictionaries follow the shipped ``rhoCentralFoam``
tutorials (``forwardStep``, ``obliqueShock``), with the smooth vanAlbada limiter and Courant 0.2 by
default (see DEFAULTS). ``Material.mu`` = 0 is inviscid (Euler); mu > 0 adds
the laminar viscous terms (constant mu, Prandtl number option ``prandtl``).

Steady state: the run starts from the uniform freestream (the first ``freestream`` BC) and goes in
chunks of ``chunk`` flow times t_ref = L / |U_inf| (L: the mesh's extent along the freestream). After
each chunk the change of p, U and T over the last flow time (volume-weighted RMS, relative to p_inf,
|U_inf| and T_inf) is measured: at most ``steady_tol`` and the flow is steady. A limited central
scheme often settles into a small limit cycle instead (the limiter switching at a sonic point on a
blunt body keeps the field changing by ~1e-4 per flow time, locally), so the case's QoIs are also
evaluated after each chunk: when every QoI moved by at most ``qoi_tol`` of its natural scale (the
dynamic pressure q_inf for pressures, q_inf * L * depth for forces, T_inf, |U_inf|, ...) over the last
3 chunks, the QoIs are steady and the
run stops too (``info.steady`` = "qois", with a note on where the field still moves). Otherwise the
run continues, up to ``max_flow_times``.
``init_from`` (a kept case directory of a solve on the same domain) maps that solution onto this
mesh first (``mapFields -consistent``), which saves most of the transient in a mesh study.

Boundary conditions (``feacore.case``): freestream = ``inletOutlet`` on U, p and T (fixed where the
flow enters, extrapolated where it leaves: a supersonic freestream boundary); outlet = zero gradient
(supersonic outflow); wall = no slip, adiabatic; slip = inviscid wall; symmetry. Boundary groups
without a BC are slip walls for an inviscid gas and no-slip adiabatic walls for a viscous one.

Results: p is the absolute static pressure (Pa), T the static temperature (K), rho = p / (R T), mach =
|U| / sqrt(gamma R T). Forces integrate the gauge pressure p - p_inf (plus the viscous stress): the
force on a body that the domain cuts (no base) as if a base at p_inf closed it.

Options (``analysis.options`` or keywords): ``max_co`` (0.2), ``steady_tol`` (1e-6), ``qoi_tol`` (1e-4),
``max_flow_times`` (400), ``chunk`` (10 flow times), ``flux_scheme`` ("Kurganov" or "Tadmor"),
``limiter`` ("vanAlbada", "vanLeer" or "Minmod": the reconstruction of rho and T; U takes its V
version; vanAlbada at Courant 0.2 settles where vanLeer keeps a limit cycle, see DEFAULTS), ``prandtl`` (0.72), ``timeout`` (s), ``case_dir`` / ``keep_case`` (keep the OpenFOAM case),
``init_from``.
"""

from __future__ import annotations

import collections
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from feacore.case import COMPRESSIBLE_FIELDS, Case, CaseError, QoI
from feacore.solver import _foam_io as fio
from feacore.solver._foam_env import require_openfoam, run
from feacore.solver._foam_fv import DIRICHLET, FvMesh
from feacore.solver._foam_mesh import FRONT_BACK, UNASSIGNED, convert
from feacore.solver._skfem_mesh import MeshError
from feacore.solver.openfoam_backend import (FlowSolveInfo, OpenFOAMResult, SolverError, _fluid_material,
                                             _is_time, _vec3)

KNOWN_OPTIONS = ("max_co", "steady_tol", "qoi_tol", "max_flow_times", "chunk", "flux_scheme", "limiter", "prandtl",
                 "timeout", "case_dir", "keep_case", "init_from")
# vanAlbada at Courant 0.2: on the blunt body (supersonic_blunt_wedge_2d, h0, alpha 0) vanLeer settles into
# a limit cycle (a wave train in the shock layer behind the sonic line, the field changing by 4e-3 of p_inf
# per flow time, p_stag by 0.1 %) at any Courant number from 0.1 to 0.4, and vanAlbada does at 0.4;
# vanAlbada at 0.2 leaves 6e-4 per flow time and steady QoIs (2e-5).
DEFAULTS = {"max_co": 0.2, "steady_tol": 1e-6, "qoi_tol": 1e-4, "max_flow_times": 400, "chunk": 10,
            "flux_scheme": "Kurganov", "limiter": "vanAlbada", "prandtl": 0.72, "timeout": None}
QOI_WINDOW = 3        # chunks over which every QoI must hold still (qoi_tol)
LIMITERS = ("vanAlbada", "vanLeer", "Minmod")     # scalar names; U uses the V version (vanLeerV, ...)
PATCH_TYPE = {"freestream": "patch", "outlet": "patch", "wall": "wall", "slip": "patch", "symmetry": "symmetry"}
# OpenFOAM's universal gas constant RR = k / mu * 1e3 (J/(kmol K)) from its physicoChemical constants
# (etc/controlDict: k = 1.38065e-23, mu = 1.66054e-27): specie molWeight = RR / gas_constant gives back
# the case's gas constant to ~1e-6.
RR = 1.38065e-23 / 1.66054e-27
TAIL = 200            # solver output lines kept for error messages and log.rhoCentralFoam


@dataclass
class CompressibleSolveInfo(FlowSolveInfo):
    flow_times: float = 0.0       # simulated time / t_ref when the run stopped
    t_ref: float = 0.0            # one flow time L / |U_inf| (s)
    time_steps: int = 0
    history: list = field(default_factory=list)   # [(flow times, max relative change over one flow time)]
    qoi_history: list = field(default_factory=list)   # [(flow times, {qoi: value})] at the end of each chunk
    steady: str = ""              # "field" (the flow stopped changing), "qois" (only the QoIs did) or ""


# --------------------------------------------------------------------------- setup


def _gas(case: Case) -> tuple[float, float, float]:
    m = _fluid_material(case)
    return float(m.gamma), float(m.gas_constant), float(m.mu)


def _freestream(case: Case, warnings: list) -> tuple[np.ndarray, float, float]:
    free = [b for b in case.bcs if b.type == "freestream"]
    if not free:
        raise CaseError("a compressible case needs a freestream BC: it gives the initial field and the "
                        "reference pressure p_inf of the forces")
    b = free[0]
    U = np.array(_vec3([float(v) for v in b.value]))
    if any(_vec3([float(v) for v in o.value]) != list(U) or o.p_inf != b.p_inf or o.T_inf != b.T_inf
           for o in free[1:]):
        warnings.append(f"several freestream BCs with different states: {b.group!r}'s gives the initial "
                        "field and p_inf")
    if not np.linalg.norm(U) > 0:
        raise CaseError("the freestream velocity must not be zero")
    return U, float(b.p_inf), float(b.T_inf)


def _patch_bcs(case: Case, fm, viscous: bool, warnings: list) -> dict:
    """{patch: BC or None (default wall)}, as the incompressible path but with the compressible default."""
    out: dict = {}
    for p, groups in fm.patches.items():
        hits = [b for b in case.bcs if b.group in groups]
        if len(hits) > 1:
            warnings.append(f"patch {p!r} (groups {list(groups)}) is claimed by several BCs "
                            f"{[(b.type, b.group) for b in hits]}: the first one is used")
        out[p] = hits[0] if hits else None
    for b in case.bcs:
        if b.group not in fm.group_patches:
            fm.patches_of(b.group)          # raises a MeshError naming the problem
    nobc = sorted({g for p, bc in out.items() if bc is None for g in (fm.patches[p] or (UNASSIGNED,))})
    if nobc:
        kind = "no-slip adiabatic walls" if viscous else "slip walls (inviscid)"
        warnings.append(f"info: boundary groups without a BC are {kind}: {nobc}")
    if not viscous and any(bc is not None and bc.type == "wall" for bc in out.values()):
        warnings.append("an inviscid gas (mu = 0) at a no-slip wall: the wall carries no shear but its face "
                        "velocity is zero; use slip for an inviscid wall")
    return out


def _bc_type(bc, viscous: bool) -> str:
    return bc.type if bc is not None else ("wall" if viscous else "slip")


def _boundary_fields(bcs: dict, viscous: bool, U_inf, p_inf: float, T_inf: float, dim: int) -> tuple:
    U, P, T = {}, {}, {}
    for p, bc in bcs.items():
        t = _bc_type(bc, viscous)
        if t == "freestream":
            u = _vec3([float(v) for v in bc.value])
            U[p] = {"type": "inletOutlet", "inletValue": u, "value": u}
            P[p] = {"type": "inletOutlet", "inletValue": float(bc.p_inf), "value": float(bc.p_inf)}
            T[p] = {"type": "inletOutlet", "inletValue": float(bc.T_inf), "value": float(bc.T_inf)}
        elif t == "outlet":
            U[p] = P[p] = T[p] = {"type": "zeroGradient"}
        elif t == "wall":
            U[p] = {"type": "noSlip"}
            P[p] = T[p] = {"type": "zeroGradient"}
        elif t == "slip":
            U[p] = {"type": "slip"}
            P[p] = T[p] = {"type": "zeroGradient"}
        elif t == "symmetry":
            U[p] = P[p] = T[p] = {"type": "symmetry"}
        else:  # pragma: no cover - validated by Case.check
            raise CaseError(f"unsupported compressible BC {t!r}")
    if dim == 2:
        for d in (U, P, T):
            d[FRONT_BACK] = {"type": "empty"}
    return U, P, T


def _write_dicts(cdir: Path, gamma: float, R: float, mu: float, opts: dict) -> None:
    cp = gamma * R / (gamma - 1.0)
    fio._w(cdir / "constant" / "thermophysicalProperties", {
        "thermoType": {"type": "hePsiThermo", "mixture": "pureMixture", "transport": "const", "thermo": "hConst",
                       "equationOfState": "perfectGas", "specie": "specie", "energy": "sensibleInternalEnergy"},
        "mixture": {"specie": {"molWeight": RR / R},
                    "thermodynamics": {"Cp": cp, "Hf": 0.0},
                    "transport": {"mu": mu, "Pr": float(opts["prandtl"])}},
    })
    fio._w(cdir / "constant" / "turbulenceProperties", {"simulationType": "laminar"})
    flux = str(opts["flux_scheme"])
    if flux not in ("Kurganov", "Tadmor"):
        raise CaseError(f"analysis.options.flux_scheme must be 'Kurganov' or 'Tadmor' (got {flux!r})")
    lim = str(opts["limiter"])
    if lim not in LIMITERS:
        raise CaseError(f"analysis.options.limiter must be one of {LIMITERS} (got {lim!r})")
    fio._w(cdir / "system" / "fvSchemes", {
        "fluxScheme": flux,
        "ddtSchemes": {"default": "Euler"},
        "gradSchemes": {"default": fio._t("Gauss linear")},
        "divSchemes": {"default": "none", "div(tauMC)": fio._t("Gauss linear")},
        "laplacianSchemes": {"default": fio._t("Gauss linear corrected")},
        "interpolationSchemes": {"default": "linear", "reconstruct(rho)": lim, "reconstruct(U)": lim + "V",
                                 "reconstruct(T)": lim},
        "snGradSchemes": {"default": "corrected"},
    })
    fio._w(cdir / "system" / "fvSolution", {
        "solvers": {
            '"(rho|rhoU|rhoE)"': {"solver": "diagonal"},
            "U": {"solver": "smoothSolver", "smoother": "GaussSeidel", "nSweeps": 2, "tolerance": 1e-9,
                  "relTol": 0.01},
            "e": {"solver": "smoothSolver", "smoother": "GaussSeidel", "nSweeps": 2, "tolerance": 1e-10,
                  "relTol": 0.0},
        },
    })


def _write_control(cdir: Path, end: float, dt0: float, write: float, max_co: float) -> None:
    fio._w(cdir / "system" / "controlDict", {
        "application": "rhoCentralFoam", "startFrom": "latestTime", "startTime": 0, "stopAt": "endTime",
        "endTime": float(end), "deltaT": float(dt0), "writeControl": "adjustable", "writeInterval": float(write),
        "purgeWrite": 2, "writeFormat": "ascii", "writePrecision": 15, "writeCompression": False,
        "timeFormat": "general", "timePrecision": 15, "runTimeModifiable": False, "adjustTimeStep": True,
        "maxCo": float(max_co), "maxDeltaT": 1.0, "functions": {},
    })


def _times(cdir: Path) -> list[Path]:
    return sorted((d for d in cdir.iterdir() if d.is_dir() and _is_time(d.name) and float(d.name) > 0),
                  key=lambda d: float(d.name))


_NAN = re.compile(r"\bnan\b|-nan\b|\binf\b", re.I)


def _run_solver(fe, cdir: Path, timeout: float | None) -> tuple[int, int, list[str], str | None]:
    """rhoCentralFoam with its output streamed: (exit code, time steps, last lines, stop reason).
    Only the tail is kept (a fine mesh runs 10^5 steps of ~8 lines). Stops on NaN / inf or a timeout."""
    import signal

    env = dict(fe.env)
    env["PWD"] = str(cdir)
    sig = int(signal.SIGUSR1)
    cmd = ["rhoCentralFoam", "-opt-switch", f"stopAtWriteNowSignal={sig}"]
    tail: collections.deque = collections.deque(maxlen=TAIL)
    steps, reason, t0 = 0, None, time.monotonic()
    with subprocess.Popen(cmd, cwd=cdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          bufsize=1) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            tail.append(line)
            if line.startswith("Time = "):
                steps += 1
            if reason is None:
                if _NAN.search(line) and ("Courant" in line or "deltaT" in line or "residual" in line):
                    reason = "diverged"
                elif timeout is not None and time.monotonic() - t0 > timeout:
                    reason = "timeout"
                if reason is not None:
                    try:
                        proc.send_signal(sig if reason == "timeout" else signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        code = proc.wait()
    lines = list(tail)
    with open(cdir / "log.rhoCentralFoam", "a") as f:
        f.write("".join(lines))
    return code, steps, lines, reason


def _read(tdir: Path, n: int) -> dict:
    out = {}
    for name in ("p", "T", "U"):
        v, bf = fio.read_field(tdir / name, n)
        out[name] = (v.ravel() if name != "U" else v, bf)
    return out


def _qoi_scales(case: Case, T_inf: float, speed: float, mach: float, rho_inf: float, L: float,
                depth: float) -> dict:
    """The natural scale of every QoI, against which qoi_tol measures its drift: the dynamic pressure
    q_inf = rho_inf |U_inf|^2 / 2 for pressures (the scale of Cp), q_inf * L * depth for forces,
    the freestream values for the rest."""
    q_inf = 0.5 * rho_inf * speed ** 2
    field = {"p": q_inf, "T": T_inf, "rho": rho_inf, "mach": mach, "ux": speed, "uy": speed, "uz": speed,
             "u_mag": speed}
    out = {}
    for q in case.qois:
        if q.type == "force":
            out[q.name] = q_inf * L * depth
        elif q.type == "flow_rate":
            out[q.name] = speed * L * depth
        elif q.type == "integral":
            out[q.name] = field.get(q.field, 1.0) * L ** case.dim
        else:
            out[q.name] = field.get(q.field, 1.0)
    return out


def _qois_steady(samples: list[dict], scales: dict, tol: float) -> tuple[bool, float]:
    """Every QoI's spread (max - min) over the samples, relative to its scale: (all <= tol, worst)."""
    worst = 0.0
    for name, sc in scales.items():
        v = [s[name] for s in samples if name in s]
        if len(v) < len(samples) or not all(math.isfinite(x) for x in v):
            return False, float("inf")
        worst = max(worst, (max(v) - min(v)) / sc)
    return worst <= tol, worst


def _where(a: dict, b: dict, C: np.ndarray, scale: dict, dim: int) -> str:
    """Where the flow still changes most (for the limit-cycle note)."""
    d = np.abs(b["p"][0] - a["p"][0]) / scale["p"]
    i = int(np.argmax(d))
    return "(" + ", ".join(f"{x:.4g}" for x in C[i, :dim]) + ")"


def _change(a: dict, b: dict, V: np.ndarray, scale: dict) -> dict:
    """Volume-weighted RMS change of p, U and T between two solutions, relative to the freestream."""
    w = V / V.sum()
    out = {}
    for k in ("p", "U", "T"):
        d = b[k][0] - a[k][0]
        d2 = np.sum(d * d, axis=1) if d.ndim == 2 else d * d
        out[k] = float(math.sqrt(np.sum(w * d2)) / scale[k])
    return out


# --------------------------------------------------------------------------- solve


def solve(case: Case, **options) -> "CompressibleResult":
    t0 = time.perf_counter()
    case.check()
    a = case.analysis
    if fio.FOAMLIB_ERROR is not None:
        raise ImportError(f"the openfoam backend needs foamlib (pip install foamlib): {fio.FOAMLIB_ERROR}")
    fe = require_openfoam()
    opts = {**DEFAULTS, **(a.options or {}), **options}
    warnings: list[str] = []
    unknown = sorted(set(a.options or {}) - set(KNOWN_OPTIONS))
    if unknown:
        warnings.append(f"unknown analysis options ignored: {unknown}")
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
        tmp = tempfile.TemporaryDirectory(prefix="feacore_rhocentral_")
        cdir = Path(tmp.name)
    try:
        res = _run(case, fe, cdir, opts, warnings, t0)
    except BaseException:
        if tmp is not None and not keep:
            tmp.cleanup()
        raise
    res._tmp = None if keep else tmp
    if keep and tmp is not None:
        tmp._finalizer.detach()  # type: ignore[attr-defined]
    if keep:
        res.info.case_dir = str(cdir)
    return res


def _run(case: Case, fe, cdir: Path, opts: dict, warnings: list, t0: float) -> "CompressibleResult":
    a = case.analysis
    dim = case.dim
    timeout = opts.get("timeout")
    gamma, R, mu = _gas(case)
    viscous = mu > 0
    U_inf, p_inf, T_inf = _freestream(case, warnings)
    speed = float(np.linalg.norm(U_inf))
    a_inf = math.sqrt(gamma * R * T_inf)
    if speed / a_inf <= 1.0:
        warnings.append(f"the freestream is subsonic (Mach {speed / a_inf:.3f}): the freestream and outlet BCs "
                        "are written for supersonic flow, where every characteristic leaves through the outlet")

    fm = convert(case.mesh_path(), dim, float(a.thickness), cdir / "mesh.msh")
    warnings += fm.warnings
    _write_control(cdir, 1.0, 1e-9, 1.0, float(opts["max_co"]))     # gmshToFoam needs a controlDict
    run(fe, ["gmshToFoam", "mesh.msh"], cdir, timeout=timeout)
    bcs = _patch_bcs(case, fm, viscous, warnings)
    types = {p: PATCH_TYPE[_bc_type(bc, viscous)] for p, bc in bcs.items()}
    if dim == 2:
        types[FRONT_BACK] = "empty"
    fio.set_patch_types(cdir, types)
    (cdir / "mesh.msh").unlink()
    _write_dicts(cdir, gamma, R, mu, opts)
    Ub, Pb, Tb = _boundary_fields(bcs, viscous, U_inf, p_inf, T_inf, dim)
    fio.write_field(cdir, "U", "volVectorField", {"length": 1, "time": -1}, list(U_inf), Ub)
    fio.write_field(cdir, "p", "volScalarField", {"mass": 1, "length": -1, "time": -2}, p_inf, Pb)
    fio.write_field(cdir, "T", "volScalarField", {"temperature": 1}, T_inf, Tb)

    poly = fio.read_poly(cdir)
    nb = fio.read_list(cdir / "constant" / "polyMesh" / "neighbour").astype(np.int64)
    mesh = FvMesh(poly["points"], poly["faces"], poly["owner"], nb, poly["patches"])
    if mesh.n_cells != fm.n_cells:
        raise SolverError(f"OpenFOAM mesh has {mesh.n_cells} cells, the input mesh {fm.n_cells}")
    if opts.get("init_from"):
        src = Path(opts["init_from"]).resolve()
        run(fe, ["mapFields", str(src), "-consistent", "-sourceTime", "latestTime"], cdir, timeout=timeout)
        warnings.append(f"info: initial field mapped from {src} (mapFields -consistent)")

    # one flow time: the mesh's extent along the freestream / |U_inf|
    e = U_inf / speed
    L = float(np.ptp(mesh.C @ e)) or float(np.max(np.ptp(mesh.C, axis=0)))
    t_ref = L / speed
    h_min = float(np.min(np.abs(mesh.V) / (fm.thickness if dim == 2 else 1.0)) ** (1.0 / dim))
    dt0 = 0.1 * float(opts["max_co"]) * h_min / (speed + a_inf)
    chunk = max(1.0, float(opts["chunk"]))
    max_ft = max(chunk, float(opts["max_flow_times"]))
    tol, qtol = float(opts["steady_tol"]), float(opts["qoi_tol"])
    scale = {"p": p_inf, "U": speed, "T": T_inf}
    depth = float(fm.thickness) if dim == 2 else L
    qscales = _qoi_scales(case, T_inf, speed, speed / a_inf, p_inf / (R * T_inf), L, depth)

    t_solve = time.perf_counter()
    steps, end, change, reason, history, qhist = 0, 0.0, {}, None, [], []
    converged, steady, spread, where = False, "", float("nan"), ""
    while True:
        end = min(end + chunk * t_ref, max_ft * t_ref)
        _write_control(cdir, end, dt0, t_ref, float(opts["max_co"]))
        left = None if timeout is None else timeout - (time.perf_counter() - t0)
        if left is not None and left <= 0:
            reason = "timeout"
            break
        code, n, tail, reason = _run_solver(fe, cdir, left)
        steps += n
        if code != 0 or reason == "diverged":
            last = "".join(tail[-25:])
            raise SolverError(f"rhoCentralFoam failed (exit {code}{', diverged' if reason == 'diverged' else ''}, "
                              f"{steps} time steps); last lines:\n{last}")
        times = _times(cdir)
        for old in times[:-2]:                 # purgeWrite works within one run only
            shutil.rmtree(old)
        times = times[-2:]
        if len(times) >= 2 and abs(float(times[-1].name) - float(times[-2].name) - t_ref) <= 1e-6 * t_ref:
            ft = round(float(times[-1].name) / t_ref, 3)
            ra, rb = _read(times[-2], mesh.n_cells), _read(times[-1], mesh.n_cells)
            change = _change(ra, rb, mesh.V, scale)
            history.append((ft, max(change.values())))
            if max(change.values()) <= tol:
                converged, steady = True, "field"
                break
            if case.qois:          # a limit cycle (e.g. limiter chatter at a sonic point) can keep the field
                snap = CompressibleResult(case, fm, cdir, gamma, R, mu, p_inf, [])   # moving while the QoIs hold
                snap._load(poly, times[-1], mesh)
                qhist.append((ft, snap.evaluate_all()))
                if len(qhist) >= QOI_WINDOW:
                    ok, spread = _qois_steady([q for _, q in qhist[-QOI_WINDOW:]], qscales, qtol)
                    if ok:
                        converged, steady = True, "qois"
                        where = _where(ra, rb, mesh.C, scale, dim)
                        break
        if reason == "timeout" or end >= max_ft * t_ref * (1 - 1e-12):
            break
    t_solve = time.perf_counter() - t_solve
    times = _times(cdir)
    if not times:
        raise SolverError("rhoCentralFoam wrote no solution")
    tdir = times[-1]
    flow_times = float(tdir.name) / t_ref
    if reason == "timeout":
        warnings.append(f"NOT CONVERGED: stopped by the {timeout} s timeout after {flow_times:.1f} flow times")
    elif steady == "qois":
        warnings.append(f"info: steady QoIs: over the last {QOI_WINDOW} chunks ({QOI_WINDOW * chunk:g} flow times) "
                        f"every QoI moved by at most {spread:.1e} of its scale (qoi_tol {qtol:g}); the flow field "
                        f"itself still changes by {_fmt(change)} per flow time, most at {where} (a small limit "
                        "cycle, typically the limiter at a sonic point)")
    elif not converged:
        warnings.append(f"NOT CONVERGED: the flow still changes by {_fmt(change)} per flow time after "
                        f"{flow_times:.1f} flow times (steady_tol {tol:g}, max_flow_times {max_ft:g})")

    res = CompressibleResult(case, fm, cdir, gamma, R, mu, p_inf, warnings)
    res._load(poly, tdir, mesh)
    wall = time.perf_counter() - t0
    measure = fm.measure if dim == 2 else float(res.V.sum())
    res.info = CompressibleSolveInfo(
        backend="openfoam", physics="compressible", formulation=a.formulation, order=int(a.order), dim=dim,
        cell_type=fm.cell_type, n_nodes=int(fm.n_nodes), n_cells=int(fm.n_cells),
        n_dofs=int(fm.n_cells * (dim + 2)), measure=float(measure),
        h_rep=float((measure / fm.n_cells) ** (1.0 / dim)), wall_time=float(wall), warnings=warnings,
        iterations=int(steps), converged=bool(converged), residuals=change, residual_tol=tol,
        openfoam=fe.version, flow_times=flow_times, t_ref=t_ref, time_steps=int(steps), history=history,
        qoi_history=qhist, steady=steady)
    warnings.append(f"info: {fe.describe()}; rhoCentralFoam ({opts['flux_scheme']}, {opts['limiter']}, "
                    f"max_co {opts['max_co']}) "
                    f"{steps} time steps, {flow_times:.1f} flow times of {t_ref:.4g} s in {t_solve:.1f} s; "
                    f"change over the last flow time {_fmt(change)}")
    return res


def _fmt(r: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:.2e}" for k, v in sorted(r.items())) + "}"


# --------------------------------------------------------------------------- result


class CompressibleResult(OpenFOAMResult):
    """Cell-centred solution of a compressible solve: p (absolute, Pa), T (K), U; rho and mach derived."""

    def __init__(self, case: Case, fm, cdir: Path, gamma: float, R: float, mu: float, p_inf: float,
                 warnings: list):
        super().__init__(case, fm, cdir, rho=1.0, nu=mu, closed=False, warnings=warnings)
        self.gamma, self.R, self.mu, self.p_inf = gamma, R, mu, p_inf

    def _load(self, poly: dict, tdir: Path, mesh: FvMesh | None = None) -> None:
        if mesh is None:
            nb = fio.read_list(self.cdir / "constant" / "polyMesh" / "neighbour").astype(np.int64)
            mesh = FvMesh(poly["points"], poly["faces"], poly["owner"], nb, poly["patches"])
        self.mesh = mesh
        n = mesh.n_cells
        f = _read(tdir, n)
        (p, bp), (T, bT), (U, bU) = f["p"], f["T"], f["U"]
        self.C, self.V = mesh.C, mesh.V
        self.p, self.T, self.U = p, T, U
        self.pshift = 0.0
        self.patch: dict = {}
        Ub, pb, Tb, mub = {}, {}, {}, {}
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
            tf = fio.patch_values(bT, name, T[own])
            Ub[name], pb[name], Tb[name], mub[name] = uf, pf, tf, np.full(nf, self.mu)
            self.patch[name] = {"Cf": mesh.Cf[sl], "Sf": mesh.Sf[sl], "U": uf, "p": pf, "T": tf,
                                "phi": np.einsum("ij,ij->i", uf, mesh.Sf[sl])}
        self._Ub, self._pb, self._Tb, self._mu_b = Ub, pb, Tb, mub
        self._faces = poly["faces"]
        dirichlet = {k: {nm: v for nm, v in vals.items() if bf.get(nm, {}).get("type") in DIRICHLET}
                     for k, vals, bf in (("U", Ub, bU), ("p", pb, bp), ("T", Tb, bT))}
        self._dirichlet = dirichlet
        self._interp = None
        self._nodes: dict = {}
        self.gradU = mesh.grad(U, Ub)

    # ---- fields
    def _check_field(self, fld):
        if fld not in COMPRESSIBLE_FIELDS:
            raise CaseError(f"unknown field {fld!r} for compressible flow; available: {COMPRESSIBLE_FIELDS}")

    def _derived(self, fld, p, T, U):
        if fld == "p":
            return p
        if fld == "T":
            return T
        if fld == "rho":
            return p / (self.R * T)
        if fld == "mach":
            return np.linalg.norm(U, axis=1) / np.sqrt(self.gamma * self.R * T)
        if fld == "u_mag":
            return np.linalg.norm(U, axis=1)
        return U[:, "xyz".index(fld[1])]

    def _cell_values(self, fld):
        self._check_field(fld)
        return self._derived(fld, self.p, self.T, self.U)

    def _face_values(self, fld, d):
        self._check_field(fld)
        return self._derived(fld, d["p"], d["T"], d["U"])

    def probe(self, fld, points):
        """Continuous point values: p, T and U interpolated (``CellPointInterpolator``), rho and mach
        derived from them at the point."""
        self._check_field(fld)
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        X = np.zeros((len(pts), 3))
        X[:, :min(3, pts.shape[1])] = pts[:, :3]
        if self.dim == 2:
            X[:, 2] = 0.5 * self.fm.thickness
        if self._interp is None:
            from scipy.spatial import cKDTree

            from feacore.solver._foam_fv import CellPointInterpolator
            self._tree = cKDTree(self.C[:, :self.dim])
            self._hloc = (np.abs(self.V) / (self.fm.thickness if self.dim == 2 else 1.0)) ** (1.0 / self.dim)
            empty = [nm for nm, (_, _, t) in self.mesh.patches.items() if t == "empty"]
            self._interp = CellPointInterpolator(self.mesh, self._faces, fit_dims=self.dim, skip_patches=empty)
        for key, vals in (("p", self.p[:, None]), ("T", self.T[:, None]), ("U", self.U)):
            if key not in self._nodes:
                self._nodes[key] = self._interp.node_values(vals, self._dirichlet[key])
        try:
            tet, lam = self._interp.locate(X, self._tree, self._hloc, self.dim)
        except ValueError as e:
            raise CaseError(str(e)) from None
        at = {k: np.einsum("ni,nik->nk", lam, self._nodes[k][self._interp.tets[tet]]) for k in ("p", "T", "U")}
        return self._derived(fld, at["p"][:, 0], at["T"][:, 0], at["U"])

    def force(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        """(pressure, viscous) force of the gas on the body through ``group``; the pressure part with the
        gauge pressure p - p_inf."""
        if group not in self.forces:
            names = [p for p in self.fm.patches_of(group) if p in self.patch]
            self.forces[group] = self.mesh.forces(names, self._pb, self.U, self.gradU, self._Ub, self._mu_b,
                                                  1.0, p_ref=self.p_inf)
        return self.forces[group]

    def evaluate(self, qoi: QoI) -> float:
        if qoi.type in ("integral", "mean", "max", "min", "max_abs"):
            self._check_field(qoi.field)
        return super().evaluate(qoi)

    def write_vtu(self, path):
        """Cell centres with the cell values as a VTK point cloud."""
        import meshio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cells = [("vertex", np.arange(len(self.C)).reshape(-1, 1))]
        pdata = {f: self._cell_values(f) for f in COMPRESSIBLE_FIELDS}
        pdata["U"] = self.U
        meshio.Mesh(self.C, cells, point_data=pdata).write(str(path))
        return path
