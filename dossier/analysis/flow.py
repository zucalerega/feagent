"""The flow analysis of a load case: the pressure drop of the fluid volume of a component.

    run(project, ctx, geo, lc, workdir, opts) -> pipeline.Analysis        analysis type "flow"

The fluid volume is `component.flow` (model FlowDomain): a STEP file of the wetted volume, the inlet and the
outlet as face queries, and the fluid (`rho`, `mu`, optional `roughness` of the wall). The load case gives the
mass flow, the pressures and the fluid temperature. The result is `pressure_drop.total@part` in Pa.

What the engine can do, and why it stops there
    The solver is the repository's OpenFOAM backend (`feacore.solver`, simpleFoam, steady, incompressible).
    The repository validates it for laminar flow. For turbulent flow this module validates the k-omega SST
    model with wall functions before it uses it (the numbers are below), and the validation holds for one
    kind of mesh only: structured hexahedra with the plain Gauss schemes. On the meshes an arbitrary CAD
    volume gets (tetrahedra), the same model was wrong by 15 to 26 % and did not converge. Thus the engine
    analyses one kind of fluid volume: a STEPPED BORE, that is a body of revolution whose faces are planes
    across the axis and cylinders about the axis (straight bores, counterbores, grooves, steps). It reads the
    radius profile from the STEP file, checks it against the volume of the solid, and makes the mesh itself.
    For every other shape, and outside the validated range, `run` gives a record with the status
    "not_supported" and the exact reason. It never gives a number that the validation does not cover.

    not_supported when: OpenFOAM or foamlib is not installed (detected at each run); the component has no
    flow domain; the load case has no mass flow; the fluid volume is not a stepped bore; the inlet and the
    outlet have different diameters; the Reynolds number is between laminar and the validated turbulent
    range, or above it.

The model
    domain      the bore, with a straight pipe added upstream (42 diameters for turbulent flow, 14 for laminar
                flow) and downstream (12 diameters)
    inlet       a velocity profile (power law for turbulent flow, parabola for laminar flow) scaled to the
                mass flow, far upstream. The inlet length makes the flow developed at the part.
    outlet      a constant static pressure
    walls       no slip, hydraulically smooth; turbulent flow: wall functions (first cell in the log layer)
    turbulence  k-omega SST (Re >= 1e5), laminar (Re < 2000)
    mesh        structured hexahedra: a square core, a transition and concentric rings ("butterfly"); the wall
                nodes are on a circle that gives the polygon the area of the true circle
    result      the pressure drop as a valve flow test measures it: static pressure 2 diameters upstream of
                the inlet face less static pressure 6 diameters downstream of the outlet face, less the
                friction loss of these two straight lengths (from the developed pressure gradient upstream)

Validation of the turbulent path (tests/dossier/test_dossier_flow.py, marked slow; OpenFOAM v1912)
    Fully developed flow in a smooth pipe of 60 diameters, friction factor from the pressure gradient between
    42 and 57 diameters, against the Colebrook formula for a smooth wall (Prandtl's law):

        Re          mesh (cells per quadrant)   y+ of the wall cells   error of the friction factor
        1.0e5       4, 5, 7                     169, 114, 85           -3.0 %, -4.1 %, -4.6 %
        1.384e6     5, 7, 10                    1240, 940, 630         -2.6 %, -2.9 %, -3.2 %
        (runs on the O-grid of the benchmark pipe_turbulent_3d: y+ 66 to 1525, -2.8 % to -4.7 %)

    Sudden expansion (diameter ratio 0.7, Re 1.384e6, meshes 5 and 7): the loss coefficient is +7.0 % and
    +6.8 % above the Borda-Carnot relation (1 - beta^2)^2; the residuals stall near 3e-3 behind the step.
    Thus the model gives a friction that is 3 to 5 % low and a separation loss that is about 7 % high. The
    stated model uncertainty is MODEL_UNCERTAINTY (10 %). docs/solver_transient.md, "Turbulent flow (RANS)".

    What did not validate (not used): tetrahedra with the skew-consistent schemes -26 %, tetrahedra with the
    plain schemes -15 % (the residuals did not decrease, the pressure field oscillated), triangular prisms
    with the skew-consistent schemes +13 to +17 % (the extended least-squares gradient reads the wall value in
    cells that touch the wall with a point, which gives too much turbulence there).

The uncertainty of the result (ResultValue.uncertainty) is the sum of three parts, and the note of the result
gives each: the mesh study (Grid Convergence Index, safety factor 3), the model (MODEL_UNCERTAINTY), and the
wall roughness when the flow domain gives one (the analysis has smooth walls; the Colebrook formula gives the
increase of the friction loss). The status "converged" needs a mesh part under policy.max_uncertainty and a
solver that converged on every mesh: the residual tolerance (1e-5), or residuals that stalled under
RESIDUAL_STALL_MAX while the pressure drop changed less than QOI_TOL in the last QOI_WINDOW iterations (steady
SIMPLE does not reach 1e-5 behind a step: the separated shear layer keeps the residuals near 1e-4).

Not verified: swirl or a distorted profile from a pump at the inlet, cavitation, compressible flow, rough
walls in the solver, bores that are not circular, and any bend. `opts.quick` gives one coarse mesh: a first
look.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dossier import units
from dossier.model import (AnalysisRecord, Artifact, BCItem, Check, LoadCase, MeshLevel, Origin, ResultValue)
from dossier.project import make_id
from dossier.records import Journal

ENGINE = "engine:flow"
RESULT_KEY = "pressure_drop.total@part"

# ---- the validated range of the turbulent path (see the module docstring and the slow tests)
VALIDATED_RE = (1.0e5, 1.4e6)       # Reynolds numbers of the pipe validation
VALIDATED_YPLUS = (60.0, 1600.0)    # y+ of the wall cells that the validation covers
MODEL_UNCERTAINTY = 0.10            # stated uncertainty of the turbulence model (measured: -4.6 % to +7 %)
RE_LAMINAR = 2000.0                 # under this Reynolds number the flow is laminar
VALIDATION = {
    "pipe": {"reference": "Colebrook formula for a smooth wall", "error": (-0.047, -0.026)},
    "expansion": {"reference": "Borda-Carnot relation, diameter ratio 0.7", "error": (0.068, 0.070)},
}

# ---- the domain, in diameters of the inlet
INLET_LENGTH = {"turbulent": 42.0, "laminar": 14.0}  # laminar: the window (8 D) starts 6 D after the inlet
OUTLET_LENGTH = 12.0
TAP_UP, TAP_DOWN = 2.0, 6.0         # pressure taps: upstream of the inlet face, downstream of the outlet face
TAP_DOWN_ALT = (4.0, 8.0)           # other downstream taps, for the check of the tap position
DEVELOPED_WINDOW = 6.0              # the developed pressure gradient is measured over this length upstream

# ---- acceptance limits of the checks
MASS_BALANCE_TOL = 1e-4
MASS_FLOW_TOL = 2e-3
MOMENTUM_TOL = 0.02
DEVELOPED_TOL = 0.02
TAP_TOL = 0.05
YPLUS_AREA_MIN = 0.90
RESIDUAL_STALL_MAX = 3e-3           # a stalled run counts as converged only under this residual (the validation's
                                    # sudden expansion stalls near 3e-3) ...
QOI_WINDOW, QOI_TOL = 500, 1e-3     # ... and when, over the snapshots of the last QOI_WINDOW iterations, both the
                                    # spread (largest - smallest) of the pressure drop and the change that a straight
                                    # line through them gives over the window are under QOI_TOL of the pressure drop
HISTORY_INTERVAL = 30               # the solver writes the fields every this many iterations: not a divisor of
                                    # round periods (100, 200), so an oscillation of such a period is not aliased away
QOI_MIN_SNAPSHOTS = 10              # snapshots in the window, the final state included
# ---- wall roughness: the flow domain gives a surface finish Ra; the friction formula needs the sand-grain ks
KS_PER_RA = 5.9                     # ks = 5.9 Ra (Adams et al. 2012, Proc. IMechE C 226, machined pipe surfaces)
KS_PER_RA_RANGE = (2.0, 10.0)       # the ratios that correlations give for machined and drawn surfaces
TAP_NEGLIGIBLE = 0.005              # a tap-position spread under this is not added to the uncertainty


class FlowDomainError(ValueError):
    """The fluid volume is not one that this engine can analyse. The text is the reason, in STE."""


# =========================================================================== availability


def _openfoam():
    """The OpenFOAM environment of the backend, or raises with the reason."""
    from feacore.solver import _foam_io as fio
    from feacore.solver._foam_env import require_openfoam
    fe = require_openfoam()
    if fio.FOAMLIB_ERROR is not None:
        raise ImportError(str(fio.FOAMLIB_ERROR))
    return fe


def availability() -> tuple[bool, str, str]:
    """(usable, reason in STE, description of the solver). Looked up at each call."""
    try:
        fe = _openfoam()
    except ImportError:
        return False, ("The Python package foamlib is not installed. The flow analysis needs it to write the "
                       "OpenFOAM files. Install it with: uv pip install --python .venv/bin/python foamlib."), ""
    except Exception:  # noqa: BLE001  FoamNotFound and what a broken installation raises
        return False, ("OpenFOAM is not installed. The flow analysis needs the OpenFOAM ESI version v1912 or "
                       "newer. Install it with: apt-get install -y openfoam. Then install the Python package "
                       "foamlib with: uv pip install --python .venv/bin/python foamlib."), ""
    return True, "", f"OpenFOAM {fe.version} simpleFoam (feacore openfoam backend)"


# =========================================================================== correlations (mesh design, checks)


def colebrook(reynolds: float, relative_roughness: float = 0.0) -> float:
    """The Darcy friction factor of turbulent pipe flow (Colebrook; a smooth wall gives Prandtl's law)."""
    x = 8.0
    for _ in range(200):
        new = -2.0 * math.log10(relative_roughness / 3.7 + 2.51 * x / reynolds)
        if abs(new - x) < 1e-13:
            break
        x = new
    return 1.0 / new**2


# =========================================================================== the fluid volume


@dataclass
class Bore:
    """A stepped bore of revolution in the CAD frame, in metres."""

    origin: np.ndarray              # the centre of the inlet face
    axis: np.ndarray                # unit vector from the inlet to the outlet
    xs: list[float]                 # positions along the axis: 0 (inlet face), the steps, the length (outlet face)
    radii: list[float]              # the radius of each segment between two positions
    volume: float                   # of the solid in the CAD file
    inlet_faces: list[int] = field(default_factory=list)
    outlet_faces: list[int] = field(default_factory=list)
    n_faces: int = 0
    source: str = ""

    @property
    def length(self) -> float:
        return self.xs[-1]

    @property
    def diameter(self) -> float:
        return 2.0 * self.radii[0]

    def profile_volume(self) -> float:
        return sum(math.pi * r * r * (b - a) for r, a, b in zip(self.radii, self.xs[:-1], self.xs[1:]))


def _mm(x: float) -> str:
    return units.fmt(units.Q(x, "m"), "mm")


def _cad_scale(path: Path, length_unit: str) -> float:
    """Metres per unit of the numbers that the CAD kernel gives for this file."""
    try:
        from dossier.geometry import UNIT_M, step_length_unit
        header, _ = step_length_unit(Path(path))
        if header:
            return UNIT_M[length_unit] * 1e-3 / header
    except Exception:  # noqa: BLE001  the geometry engine is not there, or the file has no header unit
        pass
    return 1.0


def read_bore(path: Path, inlet, outlet, length_unit: str = "m") -> Bore:
    """Read a STEP file as a stepped bore. `inlet` and `outlet` are FeatureDef (a query or face tags).
    Raises FlowDomainError when the solid is not a stepped bore."""
    from feagent.errors import FeagentError
    from feagent.geometry import analyze
    from feagent.select import select_faces
    path = Path(path)
    name = path.name
    if not path.is_file():
        raise FlowDomainError(f"The CAD file of the flow domain, {name}, does not exist.")
    try:
        model = analyze(str(path))
    except FeagentError as e:
        raise FlowDomainError(f"The CAD kernel cannot read the file {name}: {e}") from e
    if len(model.volumes) != 1:
        raise FlowDomainError(f"The file {name} has {len(model.volumes)} solids. The flow domain must be one solid.")
    k = _cad_scale(path, length_unit)
    faces = {f.tag: f for f in model.faces}

    def pick(fd, what: str) -> list[int]:
        if fd.select is not None:
            try:
                tags = select_faces(model, fd.select)
            except FeagentError as e:
                raise FlowDomainError(f"The query of the {what} of the flow domain is not valid: {e}") from e
        else:
            tags = sorted(set(int(t) for t in fd.faces or []))
            missing = [t for t in tags if t not in faces]
            if missing:
                raise FlowDomainError(f"The {what} of the flow domain names the face f{missing[0]}. The file "
                                      f"{name} has no such face.")
        if not tags:
            raise FlowDomainError(f"The query of the {what} of the flow domain selects no face of {name}.")
        if fd.expect is not None and len(tags) != fd.expect:
            raise FlowDomainError(f"The query of the {what} of the flow domain selects {len(tags)} faces of "
                                  f"{name}. The flow domain expects {fd.expect}.")
        for t in tags:
            if faces[t].kind != "plane" or faces[t].normal is None:
                raise FlowDomainError(f"The {what} of the flow domain must be a plane face. The face f{t} of "
                                      f"{name} is a {faces[t].kind}.")
        return tags

    def plane(tags: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
        area = sum(faces[t].area for t in tags)
        c = sum(faces[t].area * np.asarray(faces[t].center, float) for t in tags) / area
        n = np.asarray(faces[tags[0]].normal, float)
        return c * k, n / np.linalg.norm(n), area * k * k

    t_in, t_out = pick(inlet, "inlet"), pick(outlet, "outlet")
    if set(t_in) & set(t_out):
        raise FlowDomainError("The inlet and the outlet of the flow domain select the same face.")
    c_in, n_in, a_in = plane(t_in)
    c_out, n_out, a_out = plane(t_out)
    axis = -n_in
    length = float(np.dot(c_out - c_in, axis))
    diag = float(model.diag) * k
    tol = 1e-6 * diag
    off = np.linalg.norm((c_out - c_in) - length * axis)
    if float(np.dot(n_out, axis)) < 1.0 - 1e-9 or length <= tol or off > 1e-5 * diag:
        raise FlowDomainError("The inlet face and the outlet face are not opposite on one axis. The flow analysis "
                              "is for a straight bore: a bend is not in its validated range.")
    segs: list[tuple[float, float, float]] = []            # (x start, x end, radius)
    for t, f in faces.items():
        if t in t_in or t in t_out:
            continue
        if f.kind == "plane" and f.normal is not None:
            if abs(abs(float(np.dot(np.asarray(f.normal, float), axis))) - 1.0) > 1e-9:
                raise FlowDomainError(f"The face f{t} of {name} is a plane that is not across the axis of the "
                                      f"bore. The flow analysis is for a stepped bore of revolution.")
            continue
        if f.kind != "cylinder" or f.axis is None or f.radius is None or f.axis_point is None:
            raise FlowDomainError(f"The face f{t} of {name} is a {f.kind}. The flow analysis is for a stepped "
                                  f"bore of revolution: its faces are planes across the axis and cylinders about "
                                  f"the axis.")
        d = np.asarray(f.axis_point, float) * k - c_in
        if abs(abs(float(np.dot(np.asarray(f.axis, float), axis))) - 1.0) > 1e-9 \
                or np.linalg.norm(d - np.dot(d, axis) * axis) > 1e-5 * diag:
            raise FlowDomainError(f"The cylinder face f{t} of {name} is not about the axis of the bore. The flow "
                                  f"analysis is for a stepped bore of revolution.")
        if f.concave:
            raise FlowDomainError(f"The cylinder face f{t} of {name} is the surface of a body in the flow. The "
                                  f"flow analysis is for a bore without a body in it.")
        r = float(f.radius) * k
        span = math.radians(float(f.angle_span)) if f.angle_span else 2.0 * math.pi     # feagent gives degrees
        half = 0.5 * float(f.area) * k * k / (r * span)
        xc = float(np.dot(np.asarray(f.center, float) * k - c_in, axis))
        segs.append((xc - half, xc + half, r))
    if not segs:
        raise FlowDomainError(f"The file {name} has no cylinder face about the axis of the bore.")
    merged: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for a, b, r in segs:                                    # a cylinder in two or more faces is one segment
        merged[(round(a / tol), round(b / tol), round(r / tol))] = (a, b, r)
    segs = sorted(merged.values())
    xs, radii = [0.0], []
    for a, b, r in segs:
        if abs(a - xs[-1]) > 1e-5 * diag:
            raise FlowDomainError(f"The cylinder faces of {name} do not give one radius at each position of the "
                                  f"axis. The flow analysis is for a stepped bore without a body in it.")
        xs.append(b)
        radii.append(r)
    if abs(xs[-1] - length) > 1e-5 * diag:
        raise FlowDomainError(f"The cylinder faces of {name} do not go from the inlet face to the outlet face.")
    xs[-1] = length
    bore = Bore(origin=c_in, axis=axis, xs=xs, radii=radii, volume=float(model.volumes[0].volume) * k**3,
                inlet_faces=t_in, outlet_faces=t_out, n_faces=len(faces), source=name)
    if abs(bore.profile_volume() / bore.volume - 1.0) > 1e-4:
        raise FlowDomainError(f"The radius profile of {name} gives a volume of {bore.profile_volume() * 1e9:.1f} "
                              f"mm3 and the solid has {bore.volume * 1e9:.1f} mm3. The solid is not a stepped bore "
                              f"of revolution.")
    for area, r, what in ((a_in, radii[0], "inlet"), (a_out, radii[-1], "outlet")):
        if abs(area / (math.pi * r * r) - 1.0) > 1e-4:
            raise FlowDomainError(f"The {what} face of the flow domain is not the full section of the bore.")
    return bore


# =========================================================================== the mesh


@dataclass
class HexMesh:
    points: np.ndarray              # (n, 3) in the CAD frame
    hexes: np.ndarray               # (n, 8) gmsh node order, zero-based
    faces: dict                     # "inlet" | "outlet" | "wall" -> (n, 4) node rows
    stations: np.ndarray            # the axial positions of the cell layers' faces (local x), n_layers + 1
    m: int                          # cells per quadrant of the section
    h_wall: float                   # thickness of the wall cells of the bore, m
    h_axial_min: float
    n_section: int                  # cells of the smallest section

    @property
    def n_cells(self) -> int:
        return len(self.hexes)

    @property
    def n_nodes(self) -> int:
        return len(self.points)


def _axial_stations(breaks: list[float], steps: list[tuple[float, float]], core: tuple[float, float],
                    h_core: float, h_far: float, growth: float, min_cells: dict[int, int]) -> np.ndarray:
    """Stations with a node at every break. The size is h_step at a step and grows away from it by
    `growth` per cell; it is at most h_core in the core range and grows to h_far outside it."""
    def size(x: np.ndarray) -> np.ndarray:
        dist = np.maximum(np.maximum(core[0] - x, x - core[1]), 0.0)
        h = np.minimum(h_far, h_core + (growth - 1.0) * dist)
        for xs, hs in steps:
            h = np.minimum(h, hs + (growth - 1.0) * np.abs(x - xs))
        return h

    out = [breaks[0]]
    for i, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
        x = np.linspace(a, b, 2001)
        w = 1.0 / size(x)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(x))])
        n = max(int(round(cum[-1])), min_cells.get(i, 1), 1)
        out += list(np.interp(np.arange(1, n + 1) * cum[-1] / n, cum, x))
        out[-1] = b
    return np.asarray(out)


def build_mesh(bore: Bore, m: int, upstream: float, downstream: float, extra_breaks: list[float] = (),
               y_wall_min: float = 0.0, growth: float = 1.25) -> HexMesh:
    """Structured hexahedra for the bore with straight pipes of `upstream` and `downstream` metres added.
    `m` is the number of cells per quadrant of the section; every size is in proportion to 1 / m.
    `y_wall_min` is the smallest thickness that a wall cell can have (wall functions)."""
    r0 = min(bore.radii)
    d0 = 2.0 * r0
    n_circ = 4 * m
    fac = math.sqrt(2.0 * math.pi / (n_circ * math.sin(2.0 * math.pi / n_circ)))   # polygon area = circle area
    h_wall = max(r0 / (2.0 * m), y_wall_min)
    a_sq, r_t = 0.4 * r0, 0.7 * r0
    n_t = max(1, int(round(0.275 * m)))
    n_min_band = max(2, math.ceil(0.6 * m))

    def rings(depth: float, n_min: int) -> int:
        n = max(n_min, int(round(depth / h_wall)))
        return max(1, min(n, int(depth / y_wall_min))) if y_wall_min > 0 else n

    levels = sorted({round(r, 12) for r in bore.radii})                # the distinct radii, smallest first
    bands = [(r0, rings(r0 - r_t, 1))]
    h_wall = (r0 - r_t) / bands[0][1]                        # the thickness the wall cells of the bore get
    for lo, hi in zip(levels[:-1], levels[1:]):
        bands.append((hi, rings(hi - lo, n_min_band)))

    # ---- the section: square nodes, then circles of 4 m nodes
    g = np.linspace(-a_sq, a_sq, m + 1)
    yy, zz = np.meshgrid(g, g, indexing="ij")
    sq = np.column_stack([yy.ravel(), zz.ravel()])
    def sid(i, j):
        return i * (m + 1) + j
    per = ([sid(m, p) for p in range(m)] + [sid(m - q, m) for q in range(m)]
           + [sid(0, m - q) for q in range(m)] + [sid(q, 0) for q in range(m)])
    theta = -math.pi / 4 + 2.0 * math.pi * np.arange(n_circ) / n_circ
    unit = np.column_stack([np.cos(theta), np.sin(theta)])
    circles, circle_r = [], []                               # node rings and the true radius of each
    for layer in range(1, n_t + 1):
        s = layer / n_t
        circles.append((1.0 - s) * sq[per] + s * r_t * fac * unit)
        circle_r.append(r_t * s)
    prev = r_t
    for r_out, n in bands:
        for q in range(1, n + 1):
            r = prev + (r_out - prev) * q / n
            circles.append(r * fac * unit)
            circle_r.append(r)
        prev = r_out
    n_sq = len(sq)
    sec = np.vstack([sq] + circles)
    n_sec = len(sec)
    core_cells = [[sid(i, j), sid(i + 1, j), sid(i + 1, j + 1), sid(i, j + 1)] for i in range(m) for j in range(m)]
    ring_cells = []                                          # per circle layer: (n_circ, 4) section nodes
    for layer in range(len(circles)):
        inner = np.asarray(per) if layer == 0 else n_sq + (layer - 1) * n_circ + np.arange(n_circ)
        outer = n_sq + layer * n_circ + np.arange(n_circ)
        ring_cells.append(np.column_stack([inner, outer, np.roll(outer, -1), np.roll(inner, -1)]))
    n_always = n_t                                           # the transition layers exist in every section

    # ---- the axis: local x from -upstream to length + downstream
    xs = [-upstream] + list(bore.xs) + [bore.length + downstream]
    rs = [bore.radii[0]] + list(bore.radii) + [bore.radii[-1]]
    brk = sorted(set(xs) | {x for x in extra_breaks if -upstream < x < bore.length + downstream})
    brk = [b for i, b in enumerate(brk) if i == 0 or b - brk[i - 1] > 1e-9 * d0]
    step_x = [x for i, x in enumerate(bore.xs[1:-1], start=1) if abs(bore.radii[i] - bore.radii[i - 1]) > 1e-12]
    n_min_seg = max(3, math.ceil(0.8 * m))
    steps, min_cells = [], {}
    for x in step_x:
        near = [b - a for a, b in zip(bore.xs[:-1], bore.xs[1:])
                if (abs(a - x) < 1e-12 or abs(b - x) < 1e-12)
                and any(abs(a - s) < 1e-12 for s in step_x) and any(abs(b - s) < 1e-12 for s in step_x)]
        steps.append((x, min([h_wall] + [seg / n_min_seg for seg in near])))
    for i, (a, b) in enumerate(zip(brk[:-1], brk[1:])):
        if any(abs(a - s) < 1e-12 for s in step_x) and any(abs(b - s) < 1e-12 for s in step_x):
            min_cells[i] = n_min_seg
    h_core, h_far = 0.5 * d0 * 5.0 / m, 1.0 * d0 * 5.0 / m
    st = _axial_stations(brk, steps, (-1.0 * d0, bore.length + 2.0 * d0), h_core, h_far, growth, min_cells)
    n_x = len(st) - 1
    mid = 0.5 * (st[:-1] + st[1:])
    seg = np.clip(np.searchsorted(np.asarray(xs), mid, side="right") - 1, 0, len(rs) - 1)
    r_layer = np.asarray(rs)[seg]                            # the wall radius of each cell layer
    top = np.array([max(i for i, r in enumerate(circle_r) if r <= rl * (1 + 1e-9)) for rl in r_layer])

    # ---- hexahedra, boundary faces
    hexes, inlet, outlet, wall = [], [], [], []
    core = np.asarray(core_cells)
    for i in range(n_x):
        cells = np.vstack([core] + ring_cells[: top[i] + 1])
        hexes.append(np.column_stack([cells + i * n_sec, cells + (i + 1) * n_sec]))
        if i == 0:
            inlet.append(cells)
        if i == n_x - 1:
            outlet.append(cells + n_x * n_sec)
        outer = n_sq + top[i] * n_circ + np.arange(n_circ)
        wall.append(np.column_stack([outer + i * n_sec, np.roll(outer, -1) + i * n_sec,
                                     np.roll(outer, -1) + (i + 1) * n_sec, outer + (i + 1) * n_sec]))
        if i + 1 < n_x and top[i + 1] != top[i]:             # a step: the ring cells of one side only
            lo, hi = sorted((top[i], top[i + 1]))
            wall.append(np.vstack(ring_cells[lo + 1: hi + 1]) + (i + 1) * n_sec)
    hexes = np.vstack(hexes)
    faces = {"inlet": np.vstack(inlet), "outlet": np.vstack(outlet), "wall": np.vstack(wall)}
    # ---- nodes in the CAD frame, unused nodes removed
    e1 = bore.axis / np.linalg.norm(bore.axis)
    trial = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(trial, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    pts = (bore.origin[None, None, :] + st[:, None, None] * e1[None, None, :]
           + sec[None, :, 0, None] * e2[None, None, :] + sec[None, :, 1, None] * e3[None, None, :]).reshape(-1, 3)
    used = np.unique(hexes)
    renum = np.full(len(pts), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    assert n_always <= int(top.min()) + 1
    return HexMesh(points=pts[used], hexes=renum[hexes], faces={k: renum[v] for k, v in faces.items()},
                   stations=st, m=m, h_wall=h_wall, h_axial_min=float(np.diff(st).min()),
                   n_section=len(core_cells) + n_circ * (int(top.min()) + 1))


def write_msh(path: Path, mesh: HexMesh) -> Path:
    """gmsh 2.2 ASCII with the groups inlet, outlet, wall (faces) and fluid (cells)."""
    names = ["inlet", "outlet", "wall"]
    with open(path, "w") as f:
        f.write(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n{len(names) + 1}\n")
        for i, n in enumerate(names, start=1):
            f.write(f'2 {i} "{n}"\n')
        f.write(f'3 {len(names) + 1} "fluid"\n$EndPhysicalNames\n$Nodes\n{mesh.n_nodes}\n')
        np.savetxt(f, np.column_stack([np.arange(1, mesh.n_nodes + 1), mesh.points]), fmt="%d %.15g %.15g %.15g")
        n_el = sum(len(v) for v in mesh.faces.values()) + mesh.n_cells
        f.write(f"$EndNodes\n$Elements\n{n_el}\n")
        eid = 1
        for i, n in enumerate(names, start=1):
            rows = mesh.faces[n]
            head = np.column_stack([np.arange(eid, eid + len(rows)), np.full(len(rows), 3), np.full(len(rows), 2),
                                    np.full(len(rows), i), np.full(len(rows), i)])
            np.savetxt(f, np.column_stack([head, rows + 1]), fmt="%d")
            eid += len(rows)
        tag = len(names) + 1
        head = np.column_stack([np.arange(eid, eid + mesh.n_cells), np.full(mesh.n_cells, 5),
                                np.full(mesh.n_cells, 2), np.full(mesh.n_cells, tag), np.full(mesh.n_cells, tag)])
        np.savetxt(f, np.column_stack([head, mesh.hexes + 1]), fmt="%d")
        f.write("$EndElements\n")
    return Path(path)


# =========================================================================== one solve


@dataclass
class Setup:
    """What one solve needs besides the mesh."""

    mass_flow: float                # kg/s
    rho: float
    mu: float
    regime: str                     # "turbulent" | "laminar"
    upstream: float                 # length of the pipe added upstream, m
    downstream: float
    max_iterations: int = 1500
    residual_tol: float = 1e-5
    timeout: float | None = None


@dataclass
class Level:
    """The solution on one mesh, reduced to what the result and the checks need."""

    m: int
    cells: int
    nodes: int
    h_wall: float
    x: np.ndarray                   # centres of the cell layers (local x: 0 at the inlet face), m
    p: np.ndarray                   # area mean of the static pressure of each layer, Pa
    p_total: np.ndarray             # mass-flow mean of the total pressure of each layer, Pa
    gradient: float                 # developed pressure gradient -dp/dx upstream of the part, Pa/m
    gradient_parts: tuple[float, float]     # the same in the two halves of the window
    dp: float                       # the pressure drop of the part, Pa
    dp_taps: dict[float, float]     # the same with other downstream taps (in diameters)
    dp_total_faces: float           # loss of total pressure from the inlet face to the outlet face, Pa
    mass_in: float                  # kg/s through the inlet
    mass_balance: float             # (in - out) / in
    momentum_residual: float        # axial momentum balance of the domain / wall force
    y_plus: dict                    # p05, p50, p95 and the area fraction in the validated range (part and taps)
    iterations: int
    converged: bool
    residuals: dict
    qoi_change: dict | None         # change of the pressure drop over the last QOI_WINDOW iterations (not converged)
    warnings: list[str]
    solve_s: float
    quality: dict
    log: str = ""


def iteration_window(iterations, values, end: int, window: int = QOI_WINDOW) -> dict | None:
    """The iterative stability of a QoI from its snapshots (iteration numbers and values, the final state at
    ``end`` included): over the snapshots in the last ``window`` iterations, the spread (largest - smallest) and
    the change that a least-squares straight line through them gives from the start of the window to the end of
    the run, both relative to the final value. None when the window is not covered: the run is shorter than the
    window, or it has fewer than QOI_MIN_SNAPSHOTS snapshots in it (two snapshots cannot tell an oscillation
    whose period divides their distance from a steady value)."""
    it = np.asarray(iterations, dtype=float)
    v = np.asarray(values, dtype=float)
    sel = (it >= end - window) & np.isfinite(v)
    if end < window or sel.sum() < QOI_MIN_SNAPSHOTS or it[sel].min() > end - window + HISTORY_INTERVAL:
        return None
    it, v = it[sel], v[sel]
    final = float(v[np.argmax(it)])
    if final == 0.0:
        return None
    slope = float(np.polyfit(it - it.mean(), v, 1)[0])
    spread = float((v.max() - v.min()) / abs(final))
    trend = float(abs(slope) * (end - it.min()) / abs(final))
    return {"iterations": int(end - it.min()), "snapshots": int(sel.sum()), "spread": spread, "trend": trend,
            "change": max(spread, trend)}


def stalled_but_steady(residuals: dict, qoi_change: dict | None) -> bool:
    """A run that did not reach the residual tolerance counts as converged when its residuals are under
    RESIDUAL_STALL_MAX and the pressure drop is steady over the window (``iteration_window``: spread and trend
    under QOI_TOL)."""
    return bool(residuals and max(residuals.values()) <= RESIDUAL_STALL_MAX and qoi_change
                and qoi_change["spread"] <= QOI_TOL and qoi_change["trend"] <= QOI_TOL)


Level.solved = property(lambda lv: bool(lv.converged or stalled_but_steady(lv.residuals, lv.qoi_change)),
                        doc="Iteratively converged: the residual tolerance, or stalled residuals with a pressure "
                            "drop that is steady over the last QOI_WINDOW iterations.")


def _interp(x: np.ndarray, v: np.ndarray, at: float) -> float:
    return float(np.interp(at, x, v))


def solve_level(bore: Bore, mesh: HexMesh, setup: Setup, mesh_path: Path, log_path: Path | None = None) -> Level:
    """Solve one mesh with the OpenFOAM backend and reduce the solution. Raises what the backend raises."""
    from feacore.case import BC, Analysis, Case, Material, QoI, Region
    from feacore.geometry import fv_quality
    from feacore.solver import solve

    d = bore.diameter
    r_in = bore.radii[0]
    area = math.pi * r_in * r_in
    v_mean = setup.mass_flow / (setup.rho * area)
    re = setup.rho * v_mean * d / setup.mu
    e = bore.axis
    n_circ = 4 * mesh.m
    r_mesh = r_in * math.sqrt(2.0 * math.pi / (n_circ * math.sin(2.0 * math.pi / n_circ)))
    # ---- the inlet profile, scaled so that the faces of this mesh carry the mass flow
    quad = mesh.points[mesh.faces["inlet"]]
    # area centroid of each face, as the solver evaluates the inlet profile (not the mean of the corners)
    t1 = 0.5 * np.linalg.norm(np.cross(quad[:, 1] - quad[:, 0], quad[:, 2] - quad[:, 0]), axis=1)
    t2 = 0.5 * np.linalg.norm(np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 0]), axis=1)
    af = t1 + t2
    cf = ((t1[:, None] * (quad[:, 0] + quad[:, 1] + quad[:, 2]) + t2[:, None] * (quad[:, 0] + quad[:, 2] + quad[:, 3]))
          / (3.0 * af[:, None]))
    c0 = bore.origin - setup.upstream * e
    rad = np.linalg.norm((cf - c0) - np.outer((cf - c0) @ e, e), axis=1)
    r_mesh, c0f, ef = float(r_mesh), [float(v) for v in c0], [float(v) for v in e]
    if setup.regime == "turbulent":
        expo = float(math.sqrt(colebrook(re)))               # power law u ~ (1 - r/R)^(1/n), n = 1 / sqrt(f)
        shape = np.maximum(1.0 - rad / r_mesh, 1e-9) ** expo
        radial = f"maximum(1 - sqrt(maximum(D2 - S*S, 0))/{r_mesh!r}, 1e-9)**{expo!r}"
    else:
        shape = np.maximum(1.0 - (rad / r_mesh) ** 2, 0.0)
        radial = f"maximum(1 - maximum(D2 - S*S, 0)/{r_mesh * r_mesh!r}, 0)"
    u_max = float(setup.mass_flow / (setup.rho * float(np.sum(shape * af))))
    dx = [f"(x - {c0f[0]!r})", f"(y - {c0f[1]!r})", f"(z - {c0f[2]!r})"]
    d2 = " + ".join(f"{t}**2" for t in dx)
    s = " + ".join(f"{t}*{ef[i]!r}" for i, t in enumerate(dx))
    radial = radial.replace("D2", f"({d2})").replace("S*S", f"({s})**2")
    velocity = [f"{u_max * ef[i]!r}*{radial}" if abs(ef[i]) > 1e-12 else 0.0 for i in range(3)]
    options = {"max_iterations": int(setup.max_iterations), "residual_tol": float(setup.residual_tol),
               "history_interval": HISTORY_INTERVAL}
    if setup.timeout:
        options["timeout"] = float(setup.timeout)
    if setup.regime == "turbulent":
        intensity = 0.16 * re ** -0.125
        options.update(turbulence="kOmegaSST", skew_correction=False, turbulence_intensity=intensity,
                       turbulence_velocity=v_mean, turbulence_length=0.07 * d)
    else:
        options.update(turbulence="laminar")
    case = Case(mesh=str(mesh_path),
                analysis=Analysis(physics="flow", formulation="3d", order=2, options=options),
                materials={"fluid": Material(rho=setup.rho, mu=setup.mu)},
                bcs=[BC("velocity_inlet", group="inlet", value=velocity),
                     BC("pressure_outlet", group="outlet", value=0.0), BC("wall", group="wall")],
                regions=[Region("fluid", "fluid")],
                qois=[QoI("q_in", "flow_rate", group="inlet"), QoI("q_out", "flow_rate", group="outlet"),
                      QoI("f_x", "force", group="wall", component=0), QoI("f_y", "force", group="wall", component=1),
                      QoI("f_z", "force", group="wall", component=2)])
    t0 = time.perf_counter()
    res = solve(case, backend="openfoam")
    solve_s = time.perf_counter() - t0
    try:
        q = res.evaluate_all()
        info = res.info
        rho = setup.rho
        # ---- means of each cell layer
        xc = (res.C - bore.origin) @ e
        layer = np.clip(np.searchsorted(mesh.stations, xc) - 1, 0, len(mesh.stations) - 2)
        n_l = len(mesh.stations) - 1
        thick = np.diff(mesh.stations)
        a_cell = res.V / thick[layer]
        u_ax = res.U @ e
        p_t = res.p + 0.5 * rho * np.einsum("ij,ij->i", res.U, res.U)
        a_sum = np.bincount(layer, a_cell, n_l)
        flux = np.bincount(layer, a_cell * u_ax, n_l)
        p_mean = np.bincount(layer, a_cell * res.p, n_l) / a_sum
        pt_mean = np.bincount(layer, a_cell * u_ax * p_t, n_l) / flux
        x_mid = 0.5 * (mesh.stations[:-1] + mesh.stations[1:])
        # ---- the developed gradient upstream, the taps, the pressure drop
        x_up = -TAP_UP * d
        lo = x_up - DEVELOPED_WINDOW * d

        def slope(a: float, b: float, pm=None) -> float:
            pm = p_mean if pm is None else pm
            sel = (x_mid >= a) & (x_mid <= b)
            if sel.sum() < 2:
                return float("nan")
            w = np.sqrt(thick[sel])
            m_ = np.column_stack([np.ones(sel.sum()), x_mid[sel]]) * w[:, None]
            return -float(np.linalg.lstsq(m_, pm[sel] * w, rcond=None)[0][1])

        grad = slope(lo, x_up)
        half = 0.5 * (lo + x_up)
        parts = (slope(lo, half), slope(half, x_up))

        def drop(tap: float, pm=None) -> float:
            pm = p_mean if pm is None else pm
            g = grad if pm is p_mean else slope(lo, x_up, pm)
            x_dn = bore.length + tap * d
            return _interp(x_mid, pm, x_up) - _interp(x_mid, pm, x_dn) - g * ((0.0 - x_up) + (x_dn - bore.length))

        taps = {t: drop(t) for t in (*TAP_DOWN_ALT, TAP_DOWN) if t * d < setup.downstream - 0.5 * d}
        # ---- iterative convergence of the result: the pressure drop at every snapshot of the last QOI_WINDOW
        # iterations (iteration_window: spread and straight-line trend)
        qoi_change = None
        if not info.converged:
            its, dps = [], []
            for t in res.history_times():
                if info.iterations - QOI_WINDOW <= t < info.iterations:
                    r0 = res.at_time(t)
                    its.append(float(t))
                    dps.append(drop(TAP_DOWN, np.bincount(layer, a_cell * r0.p, n_l) / a_sum))
            its.append(float(info.iterations))
            dps.append(taps.get(TAP_DOWN, float("nan")))
            qoi_change = iteration_window(its, dps, int(info.iterations))
        dp_faces = _interp(x_mid, pt_mean, 0.0) - _interp(x_mid, pt_mean, bore.length)
        # ---- balances
        mass_in = -rho * q["q_in"]
        balance = (q["q_in"] + q["q_out"]) / q["q_in"] if q["q_in"] else float("nan")
        mom = 0.0
        for name in ("inlet", "outlet"):
            pt = res.patch[name]
            mom += float(np.sum(-pt["p"] * (pt["Sf"] @ e) - rho * (pt["U"] @ e) * pt["phi"]))
        f_wall = float(np.dot([q["f_x"], q["f_y"], q["f_z"]], e))
        mom_res = (mom - f_wall) / abs(f_wall) if f_wall else float("nan")
        # ---- y+ of the wall cells (k-based, as the wall function has it; from the wall shear when there is no k)
        y_plus: dict = {}
        if setup.regime == "turbulent":
            ypr = res.y_plus("wall")                         # the backend's y+ (from k, as the wall functions)
            yp = ypr["y_plus"]
            y_plus["basis"] = ypr["basis"]
            xw = (ypr["Cf"] - bore.origin) @ e
            zone = (xw >= x_up) & (xw <= bore.length + TAP_DOWN * d)
            aw = ypr["area"][zone]
            order = np.argsort(yp[zone])
            cum = np.cumsum(aw[order]) / aw.sum()
            ys = yp[zone][order]
            y_plus.update({f"p{int(pc * 100):02d}": float(ys[min(np.searchsorted(cum, pc), len(ys) - 1)])
                           for pc in (0.05, 0.5, 0.95)})
            ok = (yp[zone] >= VALIDATED_YPLUS[0]) & (yp[zone] <= VALIDATED_YPLUS[1])
            y_plus["in_range"] = float(aw[ok].sum() / aw.sum())
            y_plus["below"] = float(aw[yp[zone] < VALIDATED_YPLUS[0]].sum() / aw.sum())
        log = ""
        if log_path is not None:
            src = Path(res.cdir) / "log.simpleFoam"
            if src.is_file():
                shutil.copyfile(src, log_path)
                log = str(log_path)
        try:
            quality = fv_quality.from_msh(str(mesh_path), 3)
        except Exception as ex:  # noqa: BLE001
            quality = {"error": str(ex)}
        return Level(m=mesh.m, cells=int(info.n_cells), nodes=mesh.n_nodes, h_wall=mesh.h_wall, x=x_mid, p=p_mean,
                     p_total=pt_mean, gradient=grad, gradient_parts=parts, dp=taps.get(TAP_DOWN, float("nan")),
                     dp_taps=taps, dp_total_faces=dp_faces, mass_in=mass_in, mass_balance=float(balance),
                     momentum_residual=float(mom_res), y_plus=y_plus, iterations=int(info.iterations),
                     converged=bool(info.converged), residuals={k: float(v) for k, v in info.residuals.items()},
                     qoi_change=qoi_change,
                     warnings=[w for w in info.warnings if not w.startswith("info")], solve_s=solve_s,
                     quality=quality, log=log)
    finally:
        res.cleanup()


def mesh_uncertainty(sizes: list[float], values: list[float]) -> dict:
    """The discretization uncertainty of the value on the finest mesh (relative), from the values on two or
    three meshes (coarse to fine). Grid Convergence Index with a safety factor of 3. `estimated` is True when
    the uncertainty is an error estimate of a sequence that converges (two meshes, monotone, oscillating with
    smaller changes, changes under the noise level); False for one mesh and for a diverging or indeterminate
    sequence, whose uncertainty is only the scatter of the values."""
    out = {"uncertainty": None, "extrapolated": None, "order": None, "method": "one mesh: no estimate",
           "estimated": False}
    if len(values) < 2 or not all(math.isfinite(v) for v in values) or values[-1] == 0.0:
        return out
    f1, f2 = values[-1], values[-2]
    r = sizes[-2] / sizes[-1]
    two = 3.0 * abs(f2 - f1) / (r - 1.0) / abs(f1)          # first order assumed: the order is not known
    if len(values) == 2:
        return {**out, "uncertainty": two, "method": "two meshes, first order assumed, safety factor 3",
                "estimated": math.isfinite(two)}
    from feacore.verification.gci import gci3
    g = gci3(sizes[-3:], values[-3:], p_theory=2.0)
    if g.convergence == "monotone" and g.gci_fine_conservative is not None:
        return {"uncertainty": float(g.gci_fine_conservative), "extrapolated": float(g.phi_ext), "order": float(g.p),
                "method": "three meshes, monotone, observed order, safety factor 3", "estimated": True}
    if g.convergence in ("exact", "converged"):
        return {**out, "uncertainty": float(g.gci_fine or 0.0), "method": "three meshes, changes under the noise level",
                "estimated": True}
    spread = 3.0 * max(abs(f2 - f1), abs(values[-3] - f2)) / abs(f1)
    return {**out, "uncertainty": max(spread, two),
            "method": f"three meshes, {g.convergence.replace('_', ' ')}: three times the largest change",
            "estimated": g.convergence == "oscillatory"}


def level_plan(quick: bool, levels: int) -> list[int]:
    """Cells per quadrant of each mesh, coarse to fine: a ratio of about 1.4. Quick: the coarsest mesh only (the
    second one is over the node cap of a quick run). Four cells per quadrant put y+ over the validated range at
    the example's Reynolds number."""
    return [5] if quick else [5, 7, 10, 14][: max(1, min(int(levels), 4))]


# =========================================================================== the engine


def _rel(project, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(project.out).resolve()))
    except ValueError:
        return str(path)


def _record(lc: LoadCase, status: str, message: str = "", **kw) -> AnalysisRecord:
    an_id = "AN-" + (lc.id[3:] if lc.id.startswith("LC-") else lc.id)
    return AnalysisRecord(id=an_id, title=lc.title, load_case=lc.id, analysis_type="flow", status=status,
                          message=message, origin=Origin(by=ENGINE, stage="analyze"), **kw)


def _pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f} %"


def _pa(x: float) -> str:
    return units.display(units.Q(float(x), "Pa"))


def _fluid(project, ctx, comp) -> dict:
    names = ctx.names()
    out = {}
    for key, unit in (("rho", "kg/m^3"), ("mu", "Pa*s"), ("roughness", "m")):
        v = comp.flow.fluid.get(key)
        if v is None:
            continue
        q = v.quantity(names)
        out[key] = float(units.si(units.to(q, unit)))
        out[key + "_source"] = v.expr if isinstance(v.expr, str) else units.display(q)
    return out


def run(project, ctx, geo, lc: LoadCase, workdir: Path, opts):
    """Analyse a flow load case. Returns pipeline.Analysis(record, journal)."""
    from dossier.pipeline import Analysis
    j = Journal("idealize", ENGINE)
    t_start = time.perf_counter()
    comp = project.components[lc.component]
    log = getattr(opts, "log", None) or (lambda s: None)

    def stop(message: str, **kw):
        return Analysis(_record(lc, "not_supported", message, **kw), j)

    # ---- what the load case and the component give
    if comp.flow is None:
        return stop(f"The component {comp.id} has no flow domain. Add a flow block to its component.yaml: the "
                    f"STEP file of the wetted volume, the inlet face, the outlet face and the fluid.")
    flows = [x for x in lc.loads if x.kind == "mass_flow" and x.si is not None]
    if not flows:
        return stop(f"The load case {lc.id} has no mass flow. A flow analysis needs a mass flow.")
    mdot = abs(float(flows[0].si) * flows[0].factor)
    try:
        fluid = _fluid(project, ctx, comp)
    except (units.UnitsError, ValueError) as e:
        return stop(f"The fluid of the flow domain has a value that the analysis cannot read: {e}")
    if "rho" not in fluid or "mu" not in fluid or fluid["rho"] <= 0 or fluid["mu"] <= 0:
        return stop("The flow domain gives no density or no viscosity. Add rho and mu to the fluid of the flow "
                    "block.")
    temps = [x for x in lc.loads if x.kind == "fluid_temperature" and x.interface == flows[0].interface
             and x.si is not None]
    pressures = [x for x in lc.loads if x.kind == "pressure" and x.si is not None]
    cad = project.component_dir(comp.id) / comp.flow.cad.file
    try:
        bore = read_bore(cad, comp.flow.inlet, comp.flow.outlet, comp.flow.cad.length_unit)
    except FlowDomainError as e:
        return stop(f"{e} The turbulent flow path is validated for a stepped bore with a mesh of hexahedra only.")
    d = bore.diameter
    if abs(bore.radii[-1] / bore.radii[0] - 1.0) > 1e-3:
        return stop(f"The inlet of the flow domain has a diameter of {_mm(d)} and the outlet has "
                    f"{_mm(2 * bore.radii[-1])}. The analysis defines the pressure drop for equal diameters only.")
    rho, mu = fluid["rho"], fluid["mu"]
    v_mean = mdot / (rho * math.pi * d * d / 4.0)
    re = rho * v_mean * d / mu
    q_dyn = 0.5 * rho * v_mean * v_mean
    material = {"rho": rho, "mu": mu, "mass_flow": mdot, "velocity": v_mean, "reynolds": re,
                "dynamic_pressure": q_dyn, "diameter": d, "length": bore.length,
                "profile": [{"from": a, "to": b, "radius": r} for r, a, b in zip(bore.radii, bore.xs[:-1], bore.xs[1:])]}
    if temps:
        material["fluid_temperature"] = float(temps[0].si)
    if "roughness" in fluid:
        material["roughness"] = fluid["roughness"]
    if re < RE_LAMINAR:
        regime = "laminar"
    elif VALIDATED_RE[0] * 0.999 <= re <= VALIDATED_RE[1] * 1.001:
        regime = "turbulent"
    else:
        where = ("between laminar flow and the validated turbulent range" if re < VALIDATED_RE[0]
                 else "above the validated turbulent range")
        return stop(f"The Reynolds number of the flow is {re:.3g}, which is {where}. The flow analysis is validated "
                    f"for a Reynolds number under {RE_LAMINAR:g} (laminar) and from {VALIDATED_RE[0]:g} to "
                    f"{VALIDATED_RE[1]:g} (k-omega SST with wall functions).", material=material)
    usable, reason, solver = availability()
    if not usable:
        return stop(reason, material=material)

    # ---- the model: decisions and assumptions
    sub = lc.id
    eff = flows[0].effect
    up, down = INLET_LENGTH[regime] * d, OUTLET_LENGTH * d
    n_steps = sum(1 for a, b in zip(bore.radii[:-1], bore.radii[1:]) if abs(a - b) > 1e-12)
    j.assume(sub, "flow-domain", "The fluid volume of the part",
             statement=f"The file {bore.source} is the volume of the fluid in the part in this phase. It is a bore "
                       f"with a diameter of {_mm(d)}, a length of {_mm(bore.length)} and {n_steps} steps.",
             rationale=(f"The flow domain of the component gives this description: \"{comp.flow.description.strip()}\""
                        if comp.flow.description.strip() else
                        "The flow domain of the component gives the file. It gives no description of the volume."),
             impact="A different open area or more steps change the pressure drop.",
             verify_by="Compare the file with the drawings of the part and of the parts in it.",
             links=[("about", comp.id)])
    if regime == "turbulent":
        j.decide(sub, "turbulence", "Turbulence model and wall treatment",
                 choice="The analysis uses the k-omega SST turbulence model with wall functions on smooth walls.",
                 rationale=f"The Reynolds number is {re:.3g}, thus the flow is turbulent. This module validates the "
                           f"model on developed pipe flow at Reynolds numbers from {VALIDATED_RE[0]:g} to "
                           f"{VALIDATED_RE[1]:g}. The friction factor is {_pct(-VALIDATION['pipe']['error'][1])} to "
                           f"{_pct(-VALIDATION['pipe']['error'][0])} lower than the Colebrook formula for a smooth wall. The loss "
                           f"coefficient of a sudden expansion is {_pct(VALIDATION['expansion']['error'][1])} higher "
                           f"than the Borda-Carnot value. "
                           f"The stated model uncertainty is {_pct(MODEL_UNCERTAINTY, 0)}.",
                 alternatives=[("Resolve the wall layer without wall functions.",
                                "The wall cells must then be 1000 times thinner. This path is not validated here."),
                               ("Use a hand calculation only.",
                                "A hand calculation has no result for the steps of this bore.")],
                 basis="calculation", links=[("uses", eff)])
    else:
        j.decide(sub, "turbulence", "Laminar flow",
                 choice="The analysis has no turbulence model.",
                 rationale=f"The Reynolds number is {re:.3g}, which is under {RE_LAMINAR:g}. The flow is laminar. The "
                           f"repository validates the solver for laminar flow in a pipe.",
                 alternatives=[("Use a turbulence model.", "A turbulence model is wrong for laminar flow.")],
                 basis="calculation", links=[("uses", eff)])
    j.decide(sub, "mesh", "Mesh of hexahedra made from the radius profile",
             choice="The analysis makes a structured mesh of hexahedra from the radius profile of the bore.",
             rationale="The turbulence model gave the correct friction factor on hexahedra only. On tetrahedra the "
                       "error was 15 % to 26 % and the solver did not converge. The profile of the mesh has the "
                       "volume of the solid in the CAD file.",
             alternatives=[("Mesh the CAD file with tetrahedra.",
                            "The validation failed on tetrahedra, thus the result is not reliable.")],
             basis="calculation")
    profile_word = "power law" if regime == "turbulent" else "parabola"
    j.decide(sub, "inlet", "Inlet condition: developed flow from an inlet length",
             choice=f"The analysis adds a straight pipe of {INLET_LENGTH[regime]:g} diameters upstream of the inlet "
                    f"face. A velocity profile ({profile_word}) with the mass flow of {flows[0].display} enters this pipe.",
             rationale=("The inlet length makes the flow developed at the part. In the validation the pressure "
                        "gradient was constant to 1 % after 35 diameters. A developed flow is correct when the "
                        "duct upstream of the part is long and straight."
                        if regime == "turbulent" else
                        "The parabola is the exact profile of developed laminar flow. A developed flow is correct "
                        "when the duct upstream of the part is long and straight."),
             alternatives=[("Put a constant velocity on the inlet face of the part.",
                            "A constant profile has a larger wall friction and is not the flow from a long duct.")],
             basis="engineering_judgement", needs_review=True, links=[("uses", eff)])
    j.decide(sub, "outlet", "Outlet condition: constant static pressure downstream",
             choice=f"The analysis adds a straight pipe of {OUTLET_LENGTH:g} diameters downstream of the outlet face. "
                    f"The static pressure is constant on the end of this pipe.",
             rationale="The flow is incompressible, thus only pressure differences have an effect. The added length "
                       "keeps the outlet condition away from the steps of the part.",
             alternatives=[("Put the constant pressure on the outlet face of the part.",
                            "The pressure is not constant across the section near a step.")],
             basis="engineering_judgement")
    reqs = sorted({c.requirement for c in lc.criteria})
    j.decide(sub, "pressure-drop", "Definition of the pressure drop",
             choice=f"The pressure drop is the static pressure {TAP_UP:g} diameters upstream of the inlet face less the "
                    f"static pressure {TAP_DOWN:g} diameters downstream of the outlet face. "
                    f"From this difference, the engine subtracts the friction loss of these two lengths.",
             rationale="This is the pressure drop that a flow test of the part measures. The flow is developed at "
                       "the two positions, thus the static pressure difference is the loss of total pressure. The "
                       "result includes the loss downstream of the part that the steps cause. The friction loss "
                       "of the two lengths comes from the developed pressure gradient upstream of the part.",
             alternatives=[("Use the static pressure difference between the inlet face and the outlet face.",
                            "The velocity profile at the outlet face is not developed, thus this difference is not "
                            "the full loss."),
                           ("Use the loss of total pressure between the inlet face and the outlet face.",
                            "This value does not include the loss downstream of the part. A check compares it "
                            "with the result.")],
             basis="engineering_judgement", needs_review=True, links=[("verifies", r) for r in reqs])
    t_text = f" The load case gives a fluid temperature of {temps[0].display}." if temps else ""
    p_text = (f" The lowest pressure of the load case is {_pa(min(float(x.si) for x in pressures))}." if pressures
              else "")
    j.assume(sub, "incompressible", "Incompressible flow with constant properties",
             statement=f"The fluid has a constant density of {rho:g} kg/m3 and a constant viscosity of {mu:g} Pa s. "
                       f"The flow is steady and has no cavitation.",
             rationale=f"The flow domain gives the density ({fluid['rho_source']}) and the viscosity "
                       f"({fluid['mu_source']}).{t_text} The velocity is {v_mean:.3g} m/s, which is much less than the "
                       f"speed of sound in a liquid.{p_text}",
             impact="A gas or a liquid near its vapour pressure needs a different analysis.",
             verify_by="Compare the lowest static pressure with the vapour pressure of the fluid.",
             links=[("uses", eff)])
    rough_part = 0.0
    if regime == "turbulent":
        if fluid.get("roughness", 0.0) > 0.0:
            rr = fluid["roughness"] / d
            f_s, f_r = colebrook(re), colebrook(re, rr)
            rough_part = f_r / f_s - 1.0
            j.assume(sub, "smooth-walls", "Hydraulically smooth walls",
                     statement="The walls of the bore are hydraulically smooth in the analysis.",
                     rationale=f"The wall functions of the solver are for a smooth wall. The flow domain gives a "
                               f"roughness of {units.fmt(units.Q(fluid['roughness'], 'm'), 'um')}. The Colebrook "
                               f"formula gives a friction factor of {f_r:.4f} with this roughness and {f_s:.4f} "
                               f"for a smooth wall.",
                     impact=f"The friction loss can be {_pct(rough_part, 0)} larger. The uncertainty of the result "
                            f"includes this difference for the friction part.",
                     verify_by="Measure the pressure drop in a flow test.", conservative=False)
        else:
            j.assume(sub, "smooth-walls", "Hydraulically smooth walls",
                     statement="The walls of the bore are hydraulically smooth in the analysis.",
                     rationale="The wall functions of the solver are for a smooth wall. The flow domain gives no "
                               "roughness of the wall.",
                     impact="A rough wall has a larger friction loss. Give the roughness in the fluid of the flow "
                            "block, and the uncertainty of the result includes it.",
                     verify_by="Compare the roughness on the drawing with the thickness of the viscous layer.",
                     conservative=False)

    # ---- the meshes and the solves
    ms = level_plan(bool(opts.quick), int(opts.levels))
    y_min = 0.0
    if regime == "turbulent":
        u_tau = v_mean * math.sqrt(colebrook(re) / 8.0)
        y_min = 2.0 * 1.25 * VALIDATED_YPLUS[0] * (mu / rho) / u_tau     # thickness for y+ = 1.25 x the lower limit
    setup = Setup(mass_flow=mdot, rho=rho, mu=mu, regime=regime, upstream=up, downstream=down,
                  max_iterations=600 if opts.quick else 2000, residual_tol=1e-5,
                  timeout=120.0 if opts.quick else 1500.0)
    extra = [-TAP_UP * d, -(TAP_UP + DEVELOPED_WINDOW) * d, 0.0, bore.length, bore.length + TAP_DOWN * d]
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    meshes = []
    for lev, m in enumerate(ms):
        mesh = build_mesh(bore, m, up, down, extra_breaks=extra, y_wall_min=y_min)
        path = write_msh(workdir / f"mesh_L{lev}.msh", mesh)
        if meshes and mesh.n_nodes > int(opts.node_cap):     # the node cap of the run: no finer mesh
            path.unlink()
            break
        meshes.append((lev, mesh, path))
        log(f"  flow mesh L{lev}: {mesh.n_cells} hexahedra, wall cell {mesh.h_wall * 1e3:.3g} mm")

    def one(item):
        lev, mesh, path = item
        return solve_level(bore, mesh, setup, path, workdir / f"solver_L{lev}.log")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:      # at most two solver processes
            levels = list(pool.map(one, sorted(meshes, key=lambda it: -it[1].n_cells)))
        levels.sort(key=lambda lv: lv.cells)
    except Exception as e:  # noqa: BLE001  the solver failed: say so, with what it said
        text = " ".join(str(e).split())[:400]
        return Analysis(_record(lc, "failed", f"The flow solver did not finish: {type(e).__name__}: {text}",
                                solver=solver, material=material), j)
    j.stage = "solve"

    # ---- the result with its uncertainty
    an = _record(lc, "done", solver=solver + (", k-omega SST" if regime == "turbulent" else ", laminar"),
                 material=material)
    values = [lv.dp for lv in levels]
    sizes = [1.0 / lv.m for lv in levels]
    est = mesh_uncertainty(sizes, values)
    value = values[-1]
    u_mesh = est["uncertainty"]
    friction_share = min(1.0, max(0.0, levels[-1].gradient * bore.length / value)) if value > 0 else 1.0
    u_model = MODEL_UNCERTAINTY if regime == "turbulent" else 0.0
    u_rough = rough_part * friction_share
    solved = all(lv.solved for lv in levels)
    max_u = float(project.policy.max_uncertainty)
    if u_mesh is None:
        status, total = "not_converged", None
    else:
        total = u_mesh + u_model + u_rough
        status = "converged" if (est["estimated"] and u_mesh <= max_u and solved and value > 0) else "not_converged"
    parts = [f"The mesh study gives {_pct(u_mesh)} ({est['method']})." if u_mesh is not None
             else "One mesh gives no estimate of the mesh error."]
    if regime == "turbulent":
        parts.append(f"The turbulence model adds {_pct(u_model, 0)}.")
        if u_rough > 0:
            parts.append(f"The wall roughness adds {_pct(u_rough)}: the friction part is "
                         f"{_pct(friction_share, 0)} of the pressure drop.")
    if total is not None:
        parts.append(f"The uncertainty of the result is the sum, {_pct(total)}.")
    if opts.quick:
        parts.append("The meshes are coarse. This result is a first look.")
    an.results[RESULT_KEY] = ResultValue(
        key=RESULT_KEY, value=float(value), unit="Pa", uncertainty=total, status=status,
        error_estimated=bool(est["estimated"] and total is not None and solved),
        per_level=[float(v) for v in values], extrapolated=est["extrapolated"], order=est["order"], method="fe",
        near="the bore from the inlet face to the outlet face", note=" ".join(parts))
    j.decide(sub, "uncertainty", "Uncertainty of the pressure drop",
             choice="The uncertainty of the result is the sum of the mesh part, the model part and the roughness part.",
             rationale=" ".join(parts[:-1] if opts.quick else parts)
                       + (" In the validation, the model gave a friction loss that is 3 % to 5 % too low. "
                          "It gave a loss at a sudden expansion that is 7 % too high. The model part covers the two errors."
                          if regime == "turbulent" else ""),
             alternatives=[("Give the mesh part only.",
                            "The error of the turbulence model is larger than the mesh error, thus the margin must include it.")],
             basis="calculation", stage="solve")

    # ---- records of the model
    fine = levels[-1]
    an.mesh = [MeshLevel(level=lev, h=float(mesh.h_wall), nodes=int(mesh.n_nodes), cells=int(mesh.n_cells),
                         dofs=int(mesh.n_cells * (6 if regime == "turbulent" else 4)), order=2,
                         min_quality=None, h_rep=float(d / (4.0 * mesh.m) * math.pi / 2.0), path=_rel(project, path))
               for lev, mesh, path in meshes]
    an.bcs = [
        BCItem(type="velocity_inlet", value=mdot, effect=eff, interface=flows[0].interface,
               decision=make_id("DEC", sub, "inlet"),
               description=f"A developed velocity profile with a mass flow of {flows[0].display} enters "
                           f"{INLET_LENGTH[regime]:g} diameters upstream of the inlet face."),
        BCItem(type="pressure_outlet", value=0.0, decision=make_id("DEC", sub, "outlet"),
               description=f"The static pressure is constant {OUTLET_LENGTH:g} diameters downstream of the outlet face."),
        BCItem(type="wall", decision=make_id("DEC", sub, "turbulence"),
               description="Every other face is a smooth wall with no slip."),
    ]
    an.statement = (
        f"The analysis solves the steady incompressible flow of {flows[0].display} through the bore of the part. "
        f"The Reynolds number is {re:.3g} and the mean velocity is {v_mean:.3g} m/s. "
        + ("The turbulence model is k-omega SST with wall functions. " if regime == "turbulent"
           else "The flow is laminar. ")
        + f"The mesh study has {len(levels)} meshes of hexahedra, and the finest mesh has {fine.cells} cells. "
        + f"The pressure drop is {_pa(value)}, which is {value / q_dyn:.3f} times the dynamic pressure."
        + ("\n\nThe meshes are coarse. This result is a first look." if opts.quick else ""))
    an.checks = _checks(an.id, levels, setup, bore, regime, re, u_mesh, max_u, est, geo, flows[0], d)
    for lev, mesh, path in meshes:
        an.artifacts.append(Artifact(path=_rel(project, path), kind="mesh",
                                     caption=f"Mesh {lev} of the fluid volume: {mesh.n_cells} hexahedra."))
    for lev, lv in enumerate(levels):
        if lv.log:
            an.artifacts.append(Artifact(path=_rel(project, Path(lv.log)), kind="log",
                                         caption=f"Solver log of mesh {lev}: {lv.iterations} iterations."))
    data = workdir / "pressure_along_axis.json"
    data.write_text(json.dumps({
        "unit": {"x": "m", "p": "Pa"}, "x_zero": "the inlet face", "diameter": d, "length": bore.length,
        "taps": {"upstream": -TAP_UP * d, "downstream": bore.length + TAP_DOWN * d},
        "levels": [{"cells_per_quadrant": lv.m, "cells": lv.cells, "x": [float(v) for v in lv.x],
                    "p_static": [float(v) for v in lv.p], "p_total": [float(v) for v in lv.p_total],
                    "pressure_drop": lv.dp, "developed_gradient": lv.gradient,
                    "loss_of_total_pressure_between_faces": lv.dp_total_faces, "y_plus": lv.y_plus,
                    "iterations": lv.iterations, "residuals": lv.residuals, "solve_s": round(lv.solve_s, 1)}
                   for lv in levels]}, indent=1) + "\n")
    an.artifacts.append(Artifact(path=_rel(project, data), kind="data",
                                 caption="Static and total pressure along the axis on each mesh."))
    if getattr(opts, "pictures", False):
        pic = _picture(workdir / "pressure_along_axis.png", bore, levels, d)
        if pic is not None:
            an.artifacts.append(Artifact(path=_rel(project, pic), kind="picture",
                                         caption="Static pressure along the axis of the bore. The lines show the "
                                                 "two pressure taps and the faces of the part."))
    an.runtime_s = round(time.perf_counter() - t_start, 1)
    return Analysis(an, j)


def _checks(an_id: str, levels: list[Level], setup: Setup, bore: Bore, regime: str, re: float,
            u_mesh: float | None, max_u: float, est: dict, geo: dict | None, flow_item, d: float) -> list[Check]:
    fine = levels[-1]
    out: list[Check] = []

    def add(key: str, kind: str, statement: str, passed, value=None, limit=None):
        out.append(Check(id=make_id("CHK", an_id, key), kind=kind, statement=statement, passed=passed,
                         value=None if value is None or not math.isfinite(value) else float(value), limit=limit))

    worst = max(abs(lv.mass_balance) for lv in levels)
    add("mass-balance", "mass_balance",
        f"The mass flow out of the domain differs from the mass flow into it by {worst:.1e} of the mass flow. "
        f"The limit is {MASS_BALANCE_TOL:.0e}.", worst <= MASS_BALANCE_TOL, worst, MASS_BALANCE_TOL)
    dev = max(abs(lv.mass_in / setup.mass_flow - 1.0) for lv in levels)
    add("mass-flow", "mass_balance",
        f"The mass flow in the analysis is {fine.mass_in:.4g} kg/s. The load case gives {setup.mass_flow:.4g} kg/s. "
        f"The largest difference on a mesh is {_pct(dev, 2)}.", dev <= MASS_FLOW_TOL, dev, MASS_FLOW_TOL)
    mom = max(abs(lv.momentum_residual) for lv in levels)
    add("momentum-balance", "equilibrium",
        f"The pressure forces and the momentum flows on the inlet and the outlet balance the force on the walls "
        f"to {_pct(mom, 2)} of the wall force. The limit is {_pct(MOMENTUM_TOL, 0)}.",
        mom <= MOMENTUM_TOL if math.isfinite(mom) else None, mom, MOMENTUM_TOL)
    bad = [i for i, lv in enumerate(levels) if not lv.solved]
    stalled = [i for i, lv in enumerate(levels) if lv.solved and not lv.converged]
    res_worst = max((max(lv.residuals.values()) for lv in levels if lv.residuals), default=float("nan"))
    if bad:
        text = (f"The solver did not reach the residual tolerance of {setup.residual_tol:g} on mesh "
                f"{', '.join(str(i) for i in bad)}. The largest residual at the end is {res_worst:.1e}.")
        changes = [lv.qoi_change["change"] for lv in levels if lv.qoi_change]
        if changes:
            text += (f" In the last {QOI_WINDOW} iterations the pressure drop changed by up to {_pct(max(changes), 2)}. "
                     f"The limit is {_pct(QOI_TOL, 1)}.")
    elif stalled:
        changes = [levels[i].qoi_change["change"] for i in stalled]
        text = (f"The residuals did not decrease any more on mesh {', '.join(str(i) for i in stalled)}, at {res_worst:.1e} or "
                f"less (limit {RESIDUAL_STALL_MAX:.0e}). In the last {QOI_WINDOW} iterations the pressure drop changed "
                f"by {_pct(max(changes), 3)} or less (limit {_pct(QOI_TOL, 1)}). Thus the iteration error of the "
                f"result is small."
                + (f" The other meshes reached the residual tolerance of {setup.residual_tol:g}."
                   if len(stalled) < len(levels) else ""))
    else:
        text = (f"The solver reached the residual tolerance of {setup.residual_tol:g} on each mesh, after "
                f"{', '.join(str(lv.iterations) for lv in levels)} iterations.")
    limit = RESIDUAL_STALL_MAX if stalled and not bad else setup.residual_tol
    add("residuals", "residuals", text, not bad, res_worst, limit)
    if regime == "turbulent":
        yp = fine.y_plus
        inside = min(lv.y_plus.get("in_range", 0.0) for lv in levels)
        below = max(lv.y_plus.get("below", 0.0) for lv in levels)
        add("y-plus", "y_plus",
            f"On the finest mesh, y+ of the wall cells is {yp.get('p05', float('nan')):.0f} to "
            f"{yp.get('p95', float('nan')):.0f} (5 % to 95 % of the wall area, median {yp.get('p50', float('nan')):.0f}). "
            f"The validated range is {VALIDATED_YPLUS[0]:g} to {VALIDATED_YPLUS[1]:g}. On each mesh at least "
            f"{_pct(inside, 0)} of the wall area between the pressure taps is in this range, and {_pct(below, 0)} is "
            f"below it.", inside >= YPLUS_AREA_MIN, inside, YPLUS_AREA_MIN)
        add("validated-range", "model_validation",
            f"The Reynolds number is {re:.3g}. The validation of the turbulence model covers {VALIDATED_RE[0]:g} to "
            f"{VALIDATED_RE[1]:g}, with an error of the friction factor of {_pct(VALIDATION['pipe']['error'][0])} to "
            f"{_pct(VALIDATION['pipe']['error'][1])}.", True, re, VALIDATED_RE[1])
    g0, g1 = fine.gradient_parts
    diff = abs(g1 / g0 - 1.0) if g0 else float("nan")
    add("developed-inflow", "developed_flow",
        f"The pressure gradient upstream of the part changes by {_pct(diff, 2)} between the two halves of the last "
        f"{DEVELOPED_WINDOW:g} diameters. The limit for developed flow is {_pct(DEVELOPED_TOL, 0)}.",
        diff <= DEVELOPED_TOL if math.isfinite(diff) else None, diff, DEVELOPED_TOL)
    alt = [v for t, v in fine.dp_taps.items() if t != TAP_DOWN and math.isfinite(v)]
    if alt and fine.dp:
        spread = max(abs(v / fine.dp - 1.0) for v in alt)
        add("tap-position", "sensitivity",
            f"The pressure drop changes by {_pct(spread, 2)} when the downstream pressure tap moves to "
            f"{' or '.join(f'{t:g}' for t in TAP_DOWN_ALT)} diameters. The limit is {_pct(TAP_TOL, 0)}.",
            spread <= TAP_TOL, spread, TAP_TOL)
    if fine.dp:
        ratio = fine.dp_total_faces / fine.dp
        add("total-pressure", "definition",
            f"The loss of total pressure from the inlet face to the outlet face is {_pa(fine.dp_total_faces)}, which "
            f"is {ratio:.3f} times the pressure drop. The difference is the loss downstream of the part and the "
            f"change of the velocity profile.", None, ratio, None)
    q = fine.quality or {}
    no, sk = (q.get("non_orthogonality") or {}).get("max"), (q.get("skewness") or {}).get("max")
    if no is not None and sk is not None:
        from feacore.geometry import fv_quality
        ok, _ = fv_quality.passes(q)
        add("mesh-quality", "mesh_quality",
            f"The finest mesh has no inverted cell. Its largest non-orthogonality is {no:.1f} degrees (limit "
            f"{fv_quality.NON_ORTHO_MAX:g}) and its largest skewness is {sk:.2f} (limit {fv_quality.SKEW_MAX:g}).",
            bool(ok), float(no), fv_quality.NON_ORTHO_MAX)
    if u_mesh is not None:
        vals = ", ".join(_pa(lv.dp) for lv in levels)
        add("mesh-study", "error_estimate",
            f"The pressure drop on the meshes is {vals}, coarse to fine. The mesh uncertainty of the finest value "
            f"is {_pct(u_mesh)}. The policy permits {_pct(max_u, 0)}."
            + ("" if u_mesh <= max_u else " The uncertainty is more than this limit, and the status and the uncertainty of the result show it. "
               "This check is information only."),
            True if u_mesh <= max_u else None, u_mesh, max_u)     # CONTRACTS.md: convergence is not a failed check
    else:
        add("mesh-study", "error_estimate", "The analysis has one mesh. It gives no estimate of the mesh error.",
            None)
    volume = bore.profile_volume() / bore.volume - 1.0
    add("geometry", "geometry",
        f"The radius profile of the mesh has the volume of the solid in {bore.source} to {abs(volume):.1e}.",
        abs(volume) <= 1e-4, abs(volume), 1e-4)
    match = _matches_part(geo, flow_item, d)
    if match is not None:
        ok, feature, dia = match
        add("part-geometry", "geometry",
            (f"The inlet of the fluid volume has a diameter of {_mm(d)}. The feature {feature} of the part has the "
             f"same diameter." if ok else
             f"The inlet of the fluid volume has a diameter of {_mm(d)}. No wetted cylinder of the interface "
             f"{flow_item.interface} has this diameter: the nearest is {feature} with {_mm(dia)}."), ok, d, dia)
    return out


def _matches_part(geo: dict | None, item, d: float):
    """(same, feature id, its diameter): does a wetted cylinder of the interface with the mass flow have the
    diameter of the inlet of the fluid volume? None when the geometry stage has nothing to compare."""
    feats = (geo or {}).get("features") or {}
    best = None
    for fid in item.features or []:
        dia = (feats.get(fid) or {}).get("diameter")
        if dia:
            if best is None or abs(dia - d) < abs(best[1] - d):
                best = (fid, float(dia))
    if best is None:
        return None
    return abs(best[1] / d - 1.0) <= 1e-3, best[0], best[1]


def _picture(path: Path, bore: Bore, levels: list[Level], d: float) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    lo, hi = -(TAP_UP + 2) * d, bore.length + (TAP_DOWN + 2) * d
    for i, lv in enumerate(levels):
        sel = (lv.x >= lo) & (lv.x <= hi)
        ax.plot(lv.x[sel] * 1e3, lv.p[sel] * 1e-3, lw=1.2, label=f"mesh {i}: {lv.cells} cells")
    for x, name in ((-TAP_UP * d, "tap"), (0.0, "inlet face"), (bore.length, "outlet face"),
                    (bore.length + TAP_DOWN * d, "tap")):
        ax.axvline(x * 1e3, color="0.6", lw=0.8, ls="--")
        ax.annotate(name, (x * 1e3, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("position along the axis from the inlet face, mm")
    ax.set_ylabel("static pressure, kPa (outlet end = 0)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =========================================================================== validation cases (the slow tests)


def straight_bore(diameter: float, length: float) -> Bore:
    """A straight pipe along x from the origin (validation and tests)."""
    return Bore(origin=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]), xs=[0.0, float(length)], radii=[diameter / 2.0],
                volume=math.pi * diameter**2 / 4.0 * length, source="pipe")


def developed_pipe(reynolds: float, m: int, workdir: Path, diameter: float = 0.025, rho: float = 1141.0,
                   mu: float = 1.95e-4, length_d: float = 60.0, window: tuple[float, float] = (42.0, 57.0),
                   max_iterations: int = 2000) -> dict:
    """Developed turbulent flow in a smooth pipe with the mesh and the settings of the engine: the friction
    factor from the pressure gradient in `window` (diameters from the inlet) against the Colebrook formula."""
    d = diameter
    bore = straight_bore(d, 1.0 * d)                         # the "part" is 1 diameter of the pipe
    v = reynolds * mu / (rho * d)
    mdot = rho * v * math.pi * d * d / 4.0
    up, down = window[1] * d, (length_d - window[1] - 1.0) * d
    setup = Setup(mass_flow=mdot, rho=rho, mu=mu, regime="turbulent", upstream=up, downstream=down,
                  max_iterations=max_iterations)
    u_tau = v * math.sqrt(colebrook(reynolds) / 8.0)
    y_min = 2.0 * 1.25 * VALIDATED_YPLUS[0] * (mu / rho) / u_tau
    mesh = build_mesh(bore, m, up, down, extra_breaks=[-(window[1] - window[0]) * d, 0.0], y_wall_min=y_min)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = write_msh(workdir / f"pipe_re{reynolds:.3g}_m{m}.msh", mesh)
    lv = solve_level(bore, mesh, setup, path)
    sel = (lv.x >= -(window[1] - window[0]) * d) & (lv.x <= 0.0)
    w = np.sqrt(np.gradient(lv.x)[sel])
    a = np.column_stack([np.ones(sel.sum()), lv.x[sel]]) * w[:, None]
    grad = -float(np.linalg.lstsq(a, lv.p[sel] * w, rcond=None)[0][1])
    f = grad * d / (0.5 * rho * v * v)
    f_ref = colebrook(reynolds)
    return {"reynolds": reynolds, "m": m, "cells": lv.cells, "friction_factor": f, "colebrook": f_ref,
            "error": f / f_ref - 1.0, "y_plus": lv.y_plus, "iterations": lv.iterations, "converged": lv.converged,
            "residuals": lv.residuals, "mass_balance": lv.mass_balance, "momentum_residual": lv.momentum_residual,
            "solve_s": lv.solve_s}
