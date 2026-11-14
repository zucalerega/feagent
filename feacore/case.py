"""Solver-agnostic description of one FE analysis (a "case").

This module is the contract between whatever produces a mesh and a setup (agents,
the mesh harness, the reference builders in ``feacore.problems``) and the fixed
solver backends in ``feacore.solver``. Everything here round-trips through JSON.

Conventions
-----------
* Mesh: a gmsh ``.msh`` file (v2.2 or v4.1, ASCII or binary) with *named* physical
  groups. Every cell must belong to a physical group of the mesh dimension (gmsh
  drops unassigned elements when any physical group exists). BCs, loads, regions and
  QoIs refer to groups by name. Linear (tri3/tet4) or quadratic (tri6/tet10) cells;
  quadratic cells are treated as curved (isoparametric) geometry. ``analysis.order = 1`` on a
  quadratic mesh drops the midside nodes (straight-sided); order 2 on a linear mesh is straight-sided P2.
* Axisymmetric: u_r = 0 is imposed automatically on nodes on the axis (x = 0).
* 2D meshes lie in the xy-plane. Axisymmetric: x = r >= 0, y = z, axis at x = 0.
* Units are whatever the case uses consistently (the problems use SI: m, N, Pa, K).
* Plane formulations: forces, reactions and energies are for the given
  ``analysis.thickness``. Axisymmetric: they are totals over the full 360 degrees.
* Expressions (strings) are evaluated in x, y, z (axisymmetric: x=r, y=z) with a
  restricted set of numpy functions; see ``eval_expr``.

Boundary conditions (``BC.type``)
---------------------------------
``group`` is normally a facet group (edges in 2D, faces in 3D). Dirichlet types (fixed,
displacement, temperature) may also use lower-dimensional groups (points in 2D; curves or
points in 3D), constraining every node on them. They may also carry ``where``: an expression in
x, y, z (same syntax as values) that keeps only the group's nodes where it is true (nonzero). For
example ``{"type": "displacement", "group": "outer", "components": [2], "value": 0,
"where": "abs(z) < 1e-6"}`` constrains the line where the outer face meets z = 0, so no edge group
is needed. Selecting no node is an error.
elasticity:
  fixed         all displacement components = 0 on ``group`` (or the node nearest ``point``)
  displacement  prescribed ``components`` (0=x, 1=y, 2=z) = ``value`` (scalar for all listed
                components, or one entry per component; entries may be expressions)
  symmetry      zero normal displacement on a planar, axis-aligned facet group
  pressure      traction t = -value * n (n = outward normal): positive pushes into the body
  traction      traction vector ``value`` (force / area), global axes
  force         total force vector ``value`` spread uniformly (per area) over ``group``
heat (also used for Poisson problems, k = diffusion coefficient):
  temperature   Dirichlet T = value
  flux          heat flux *into* the body per area: k dT/dn = value
  convection    k dT/dn = h (T_inf - T)

Loads (``Load.type``), applied on cell group ``group`` ("*" = whole domain)
------------------------------------------------------------------------
elasticity: body_force (vector, force / volume), temperature (T field for thermal strain
            alpha * (T - analysis.T_ref))
heat:       source (volumetric heat source / Poisson right-hand side)

The ``value`` of a temperature load is a number, an expression in x, y, z, or a temperature field
from a file: ``{"field": "<path to an .npz file>"}``. This is how the temperatures of a
heat-conduction solve enter an elasticity case (one-way coupling). The file holds ``x`` (n, dim) node
coordinates and ``T`` (n,) temperatures; other arrays in it are ignored. A path that is not absolute
is relative to the folder of the case file (``base_dir``), as the mesh path is. The solver maps the
values to its own nodes by nearest point. A node of the load's cell group with no field point within
``FIELD_MATCH_TOL`` (1e-6) of the bounding-box diagonal is a ``CaseError``: the field does not belong
to this mesh. The solver then interpolates the nodal values with its shape functions, so between the
nodes the temperature is the FE interpolant, not the exact field. A heat solve on the same mesh and
order has the same nodes: ``write_temperature_field(path, *heat_result.nodal("T"))``. A field at the
nodes of a quadratic solve also serves an order-1 analysis of the same mesh (its nodes are the
corners); the reverse is refused. skfem backend only: the ccx backend refuses a field file.

flow (incompressible, steady by default; OpenFOAM backend):
  velocity_inlet  velocity ``value``: a vector [ux, uy(, uz)] (entries may be expressions) or a scalar
                  speed along the inward normal
  pressure_outlet static pressure ``value`` (Pa, physical, not kinematic; usually 0)
  wall            no-slip; optional ``value`` = wall velocity vector (moving wall / lid)
  slip            zero normal velocity, zero shear
  symmetry        symmetry plane
  2D flow meshes lie in the xy-plane; the backend extrudes one cell layer of ``analysis.thickness``,
  so forces and flow rates are per that thickness. ``analysis.order`` selects the convection scheme
  (1 = upwind, 2 = second order). Options: ``turbulence`` ("laminar" default, or a RANS model name),
  ``wall_treatment`` ("wall_functions" default with a RANS model, or "resolved": the mesh resolves the
  viscous sublayer, y+ about 1),
  ``max_iterations``, ``residual_tol`` (keep it tight, e.g. 1e-8, so the iteration error stays far
  below the discretization error that the GCI measures).

compressible (a perfect gas, ``Material.gamma`` and ``Material.gas_constant``; ``mu`` = 0 is inviscid;
OpenFOAM backend, rhoCentralFoam run to a steady state; supersonic external flow):
  freestream      velocity ``value`` [ux, uy(, uz)] (m/s), static pressure ``p_inf`` (Pa) and static
                  temperature ``T_inf`` (K): fixed where the flow enters, extrapolated where it leaves
                  (a supersonic freestream boundary). It also gives the initial field and the
                  reference pressure of the force QoI.
  outlet          supersonic outflow: everything extrapolated (no value)
  wall            no slip, adiabatic (viscous gas only; an inviscid gas needs slip)
  slip            inviscid wall: zero normal velocity
  symmetry        symmetry plane
  total_inlet     subsonic inflow from a reservoir: total pressure ``p0`` (Pa) and total temperature
                  ``T0`` (K), the flow enters along the inward normal
  pressure_outlet static back pressure ``value`` (Pa) where the flow leaves subsonic; where it leaves
                  supersonic everything is extrapolated
  Fields: ux, uy, uz, u_mag, p (absolute static pressure, Pa), T (K), rho (kg/m^3), mach. Options:
  ``max_co`` (Courant number), ``steady_tol``, ``max_flow_times``, ``timeout``; see
  feacore/solver/_foam_compressible.py.

Nonlinear (elastoplastic) analysis (opt-in; CalculiX backend only)
-------------------------------------------------------------------
``Material.plasticity`` (optional): ``{"model": "j2_isotropic", "yield_stress": sy,
"hardening": [[plastic_strain, stress], ...]}``: von Mises (J2) plasticity with isotropic
hardening. ``hardening`` is the flow stress vs equivalent plastic strain, piecewise linear, starting
at ``[0, yield_stress]`` (prepended when the first point has plastic_strain > 0) and constant beyond
the last point; omit it for elastic-perfectly plastic.
``analysis.options`` (all optional except ``nonlinear``): ``nonlinear: true`` turns the analysis into a
load-stepped static one; ``increments`` (default 20) fixed, equal increments; ``load_factor_max``
(default 1.0) the multiplier of every load and prescribed displacement at the end of the ramp;
``nlgeom`` (default false) geometric nonlinearity (large displacements); ``adaptive`` (default false)
lets the solver cut an increment back when it does not converge (increments are then not equal).
``failure`` (optional, ccx): progressive failure by element erosion,
``{"criterion": "peeq", "limit": 0.043, ...}``: the ramp runs in restart blocks and after every block
the cells whose PEEQ (integration-point mean, or max with ``"aggregate": "max"``) reaches ``limit`` are
removed; loading continues from the current state. ``"criterion": "peeq_tension"`` erodes at ``limit`` only
where the cell is in tension (compressive cells at ``compression_factor`` x limit, default 5);
``"principal"`` erodes where the max principal stress reaches ``stress_limit``. Displacement-driven loading only; see
``docs/solver_ccx_nonlinear.md`` for the block / increment options and the method's limits.
A material with ``plasticity`` requires ``nonlinear: true``. The skfem backend refuses nonlinear cases.

Time-dependent and modal analyses (opt-in; see docs/solver_transient.md)
-------------------------------------------------------------------------
Time in values. In a transient analysis a value may depend on the time ``t`` (seconds, from 0):
an expression may use ``t`` beside x, y, z (``"90 + 203 * exp(-t / 5)"``), and a scalar value may be
a table ``{"table": [[t0, v0], [t1, v1], ...]}`` (linear between the points, constant before the
first and after the last). This holds for the ``value`` of a BC or Load (each component of a vector
separately) and for the ``h`` and ``T_inf`` of a convection BC. A steady analysis refuses both.

``analysis.options["transient"]`` makes the analysis time-dependent:
  ``t_end``            end time (s); the analysis starts at t = 0
  ``dt`` | ``n_steps``  the fixed time step (s), or the number of steps (give one of them)
  ``scheme``           heat: "crank_nicolson" (default, second order), "backward_euler" (first order,
                       damps the modes a step cannot resolve), "bdf2"; elasticity (dynamics): "newmark"
                       (average acceleration, default), "hht" (with ``alpha`` in [-1/3, 0]); flow:
                       "backward" (default, second order) or "euler"
  ``initial``          the state at t = 0. heat: ``{"T": number | expression in x, y, z |
                       {"field": "<.npz with x and T>"}}``; elasticity: ``{"u": [...], "v": [...]}``
                       (displacement and velocity, default at rest); flow: ``{"U": [...], "p": number}``
                       or ``{"steady": true}`` (start from the steady solution of the t = 0 values)
  ``output_times`` | ``output_every``  the steps whose full state is kept (default: 10 states evenly
                       spaced in time, and the last). QoIs are evaluated at every step.
  ``max_co``           flow only: adapt the step to this Courant number, never above ``dt``
Materials: transient heat needs ``rho`` and ``cp`` (the heat capacity per volume is rho * cp);
dynamics and modal analysis need ``rho``. ``analysis.options["damping"] = {"rayleigh": [a, b]}``
gives the damping matrix C = a M + b K of a dynamic analysis.

``analysis.options["modal"] = {"n_modes": n, "shift": f}`` asks for the n natural frequencies nearest
above ``f`` (Hz, default 0) of an elasticity case, with their mode shapes. Loads are ignored; the
Dirichlet BCs act as zero. A model with rigid-body modes is solved too: they are reported, not counted.

A QoI of a transient analysis is a time series. ``QoI.time`` reduces it to one number: omitted = the
value at ``t_end``; a number = the value at that time (linear between steps); "max", "min", "max_abs",
"mean", "rms" = over the steps; "frequency" = the dominant frequency of the series (Hz). ``QoI.window``
= [t0, t1] restricts the reduction to that span (to leave out a start-up transient).

Quantities of interest (``QoI.type``)
-------------------------------------
  point          ``field`` at ``point`` (stresses use the solver's recovered, continuous field)
  point_difference  ``field(point) - field(point2)``, e.g. a pressure drop between two probes
  max/min/max_abs extreme of the recovered nodal ``field`` over ``group`` (None = whole mesh)
  reaction       support force on the body: (K u - f) summed over the group's constrained DOFs,
                 ``component`` (heat: heat flow into the body through the group)
  integral/mean  integral of ``field`` over cell or facet ``group`` including thickness (plane) or
                 2*pi*r (axisymmetric); mean = integral / equally weighted measure
  strain_energy  total strain energy 0.5 u.K.u (elasticity only)
  force          flow: total fluid force (pressure + viscous) on the body through ``group``, ``component``;
                 compressible: with the gauge pressure p - p_inf of the freestream (as if a base at p_inf
                 closed a body that the domain cuts)
  flow_rate      flow: volumetric flow rate through ``group``, positive out of the domain
  mass_flow      flow, compressible: mass flow rate through ``group`` (kg/s), positive out of the domain
  natural_frequency  modal: the frequency of mode ``mode`` (1 = lowest elastic mode), Hz
  effective_mass     modal: the share of the total mass that mode ``mode`` moves along ``component``
flow only, on a wall group, with max / min / mean: the wall fields ``y_plus`` and ``wall_shear`` (Pa).
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

SCHEMA_VERSION = 1

PHYSICS = ("elasticity", "heat", "flow", "compressible")
FORMULATIONS = {
    "elasticity": ("plane_stress", "plane_strain", "axisymmetric", "3d"),
    "heat": ("2d", "axisymmetric", "3d"),
    "flow": ("2d", "3d"),
    "compressible": ("2d", "3d"),
}
# compressible: a perfect gas (Material.gamma, Material.gas_constant; mu = 0 is inviscid, Euler).
# freestream: value = velocity [ux, uy(, uz)] in m/s, p_inf = static pressure (Pa), T_inf = static
# temperature (K); fixed where the flow enters, extrapolated where it leaves (supersonic freestream).
# outlet: everything extrapolated (supersonic outflow). wall: no slip, adiabatic. slip: inviscid wall.
BC_TYPES = {
    "elasticity": ("fixed", "displacement", "symmetry", "pressure", "traction", "force"),
    "heat": ("temperature", "flux", "convection"),
    "flow": ("velocity_inlet", "pressure_outlet", "wall", "slip", "symmetry"),
    "compressible": ("freestream", "outlet", "wall", "slip", "symmetry", "total_inlet", "pressure_outlet"),
}
LOAD_TYPES = {
    "elasticity": ("body_force", "temperature"),
    "heat": ("source",),
    "flow": (),
    "compressible": (),
}
QOI_TYPES = ("point", "point_difference", "max", "min", "max_abs", "reaction", "integral", "mean",
             "strain_energy", "force", "flow_rate", "mass_flow", "natural_frequency", "effective_mass")
QOI_TIME = ("max", "min", "max_abs", "mean", "rms", "frequency")   # QoI.time, besides a number
TRANSIENT_SCHEMES = {"heat": ("crank_nicolson", "backward_euler", "bdf2"), "elasticity": ("newmark", "hht"),
                     "flow": ("backward", "euler")}
TRANSIENT_OPTIONS = ("t_end", "dt", "n_steps", "scheme", "alpha", "initial", "output_times", "output_every", "max_co")
MODAL_OPTIONS = ("n_modes", "shift")
WALL_FIELDS = ("y_plus", "wall_shear")    # flow: on a wall group, with max / min / mean

# Fields a backend must be able to probe. In 2D, uz, syz and sxz are 0; szz is the out-of-plane
# stress (0 in plane stress). Axisymmetric: sxx = s_rr, syy = s_zz (axial), sxy = s_rz, and the
# hoop stress is both stt and szz (out-of-plane). stt is 0 outside axisymmetric.
ELASTICITY_FIELDS = (
    "ux", "uy", "uz", "u_mag",
    "sxx", "syy", "szz", "sxy", "syz", "sxz", "stt",
    "von_mises", "tresca", "s1", "s2", "s3",
)
HEAT_FIELDS = ("T", "qx", "qy", "qz", "q_mag")  # q = -k grad T
FLOW_FIELDS = ("ux", "uy", "uz", "u_mag", "p")    # p = static pressure (Pa)
COMPRESSIBLE_FIELDS = FLOW_FIELDS + ("T", "rho", "mach")   # p absolute static pressure, T static temperature
DISPLACEMENT_LIKE = ("ux", "uy", "uz", "u_mag", "T")
STRESS_LIKE = tuple(f for f in ELASTICITY_FIELDS if f not in DISPLACEMENT_LIKE) + ("qx", "qy", "qz", "q_mag")

Value = Union[float, str, list]

# A temperature load from a file: a field point counts as a node of the solver when it is within this
# fraction of the mesh's bounding-box diagonal (see "Loads" in the module docstring).
FIELD_MATCH_TOL = 1e-6


class CaseError(ValueError):
    """Raised for an invalid case description."""


# --------------------------------------------------------------------------- schema


@dataclass
class Material:
    E: float = 0.0       # Young's modulus
    nu: float = 0.0      # Poisson's ratio
    alpha: float = 0.0   # coefficient of thermal expansion
    k: float = 1.0       # conductivity (heat) / diffusion coefficient (Poisson)
    plasticity: dict | None = None   # optional J2 plasticity (nonlinear analyses only), see module doc
    rho: float = 1.0     # density (flow; transient heat; dynamics and modal analysis)
    cp: float = 0.0      # specific heat capacity (transient heat), J/(kg K)
    mu: float = 0.0      # dynamic viscosity (flow); kinematic viscosity = mu / rho
    gamma: float | None = None         # compressible: ratio of specific heats cp / cv
    gas_constant: float | None = None  # compressible: specific gas constant R = cp - cv (J/(kg K))


@dataclass
class Region:
    group: str           # cell physical group ("*" = all cells)
    material: str        # key into Case.materials


@dataclass
class BC:
    type: str
    group: str | None = None            # facet group (Dirichlet: any lower-dim group)
    point: list[float] | None = None    # alternative to group: node nearest to this point
    components: list[int] | None = None
    value: Value | None = None
    where: str | None = None            # Dirichlet node filter, e.g. "abs(z) < 1e-6"
    h: Value | dict | None = None       # convection coefficient (transient: also an expression in t or a table)
    T_inf: Value | dict | None = None   # convection ambient temperature (transient: as h); freestream static temperature
    p_inf: float | None = None          # freestream static pressure (compressible)
    p0: float | None = None             # total_inlet: total pressure (compressible)
    T0: float | None = None             # total_inlet: total temperature (compressible)


@dataclass
class Load:
    type: str
    value: Value | dict                 # temperature: also {"field": "<.npz with x and T>"}, see module doc
    group: str = "*"


@dataclass
class QoI:
    name: str
    type: str
    field: str | None = None
    point: list[float] | None = None
    group: str | None = None
    component: int | None = None
    point2: list[float] | None = None   # point_difference: value = field(point) - field(point2)
    time: float | str | None = None     # transient: None = at t_end, a time, or one of QOI_TIME
    window: list[float] | None = None   # transient: [t0, t1], the span that `time` reduces over
    mode: int | None = None             # modal: the mode number (1 = lowest elastic mode)


@dataclass
class Analysis:
    physics: str = "elasticity"
    formulation: str = "3d"
    order: int = 2                 # polynomial order of the FE interpolation (1 or 2)
    thickness: float = 1.0         # plane stress / plane strain thickness
    T_ref: float = 0.0             # stress-free reference temperature
    options: dict = field(default_factory=dict)  # backend options, e.g. {"stress_recovery": "l2"}

    @property
    def nonlinear(self) -> bool:
        """True for a load-stepped nonlinear analysis (``options["nonlinear"]``)."""
        return bool((self.options or {}).get("nonlinear", False))

    @property
    def transient(self) -> dict | None:
        """The ``options["transient"]`` dict of a time-dependent analysis, else None."""
        t = (self.options or {}).get("transient")
        return t if t else None

    @property
    def modal(self) -> dict | None:
        """The ``options["modal"]`` dict of a modal analysis, else None."""
        m = (self.options or {}).get("modal")
        return m if m else None

    def time_steps(self) -> np.ndarray:
        """The times of a transient analysis, from 0 to t_end (n + 1 values), for a fixed step."""
        tr = self.transient or {}
        t_end = float(tr["t_end"])
        n = int(tr["n_steps"]) if tr.get("n_steps") is not None else max(1, int(round(t_end / float(tr["dt"]))))
        return np.linspace(0.0, t_end, n + 1)


@dataclass
class Case:
    mesh: str
    analysis: Analysis
    materials: dict[str, Material]
    bcs: list[BC]
    loads: list[Load] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    qois: list[QoI] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    base_dir: str | None = field(default=None, compare=False, repr=False)  # not serialised

    # ---- paths
    def mesh_path(self) -> Path:
        p = Path(self.mesh)
        if not p.is_absolute() and self.base_dir is not None:
            p = Path(self.base_dir) / p
        return p

    def file_path(self, name: str | Path) -> Path:
        """A file the case refers to (a temperature field): a path that is not absolute is relative to
        ``base_dir``, the folder of the case file, as the mesh path is."""
        p = Path(name)
        if not p.is_absolute() and self.base_dir is not None:
            p = Path(self.base_dir) / p
        return p

    @property
    def dim(self) -> int:
        """Geometric dimension implied by the formulation."""
        return 3 if self.analysis.formulation == "3d" else 2

    def material_for(self, group: str | None = None) -> Material:
        """Material of a cell group (single-material cases: the only material)."""
        if not self.regions:
            return next(iter(self.materials.values()))
        for r in self.regions:
            if r.group == group or r.group == "*":
                return self.materials[r.material]
        raise CaseError(f"no material region for group {group!r}")

    # ---- serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("base_dir", None)
        for m in d["materials"].values():      # linear / incompressible cases serialise exactly as before
            for k in ("plasticity", "gamma", "gas_constant"):
                if m.get(k) is None:
                    m.pop(k, None)
            if not m.get("cp"):
                m.pop("cp", None)
        for b in d["bcs"]:
            for k in ("p_inf", "p0", "T0"):
                if b.get(k) is None:
                    b.pop(k, None)
        for q in d["qois"]:
            for k in ("time", "window", "mode"):
                if q.get(k) is None:
                    q.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | Path | None = None) -> "Case":
        d = dict(d)
        version = d.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise CaseError(f"unsupported schema_version {version} (expected {SCHEMA_VERSION})")
        try:
            return cls(
                mesh=d["mesh"],
                analysis=Analysis(**d.get("analysis", {})),
                materials={k: Material(**v) for k, v in d["materials"].items()},
                bcs=[BC(**b) for b in d.get("bcs", [])],
                loads=[Load(**l) for l in d.get("loads", [])],
                regions=[Region(**r) for r in d.get("regions", [])],
                qois=[QoI(**q) for q in d.get("qois", [])],
                meta=d.get("meta", {}),
                schema_version=version,
                base_dir=str(base_dir) if base_dir is not None else None,
            )
        except (KeyError, TypeError) as e:
            raise CaseError(f"malformed case: {e}") from e

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Case":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)

    # ---- validation (structure only; group names are checked by the backend against the mesh)
    def validate(self) -> list[str]:
        errs: list[str] = []
        a = self.analysis
        if a.physics not in PHYSICS:
            return [f"analysis.physics must be one of {PHYSICS}, got {a.physics!r}"]
        if a.formulation not in FORMULATIONS[a.physics]:
            errs.append(f"analysis.formulation for {a.physics} must be one of {FORMULATIONS[a.physics]}")
        if a.order not in (1, 2):
            errs.append("analysis.order must be 1 or 2")
        if a.thickness <= 0:
            errs.append("analysis.thickness must be > 0")
        if not self.materials:
            errs.append("at least one material is required")
        for name, m in self.materials.items():
            if a.physics == "elasticity" and (m.E <= 0 or not (-1.0 < m.nu < 0.5)):
                errs.append(f"material {name!r}: need E > 0 and -1 < nu < 0.5")
            if a.physics == "heat" and m.k <= 0:
                errs.append(f"material {name!r}: need k > 0")
            if a.physics == "flow" and (m.rho <= 0 or m.mu <= 0):
                errs.append(f"material {name!r}: need rho > 0 and mu > 0")
            if a.physics == "compressible" and not (m.gamma is not None and m.gamma > 1.0
                                                    and m.gas_constant is not None and m.gas_constant > 0
                                                    and m.mu >= 0):
                errs.append(f"material {name!r}: a compressible gas needs gamma > 1, gas_constant > 0 and mu >= 0 "
                            f"(mu = 0: inviscid)")
        errs += _validate_nonlinear(self)
        errs += _validate_time(self)
        for r in self.regions:
            if r.material not in self.materials:
                errs.append(f"region {r.group!r} references unknown material {r.material!r}")
        ncomp = self.dim
        for i, b in enumerate(self.bcs):
            tag = f"bcs[{i}] ({b.type})"
            if b.type not in BC_TYPES[a.physics]:
                errs.append(f"{tag}: type must be one of {BC_TYPES[a.physics]}")
                continue
            if (b.group is None) == (b.point is None):
                errs.append(f"{tag}: give exactly one of group / point")
            if b.point is not None and b.type not in ("fixed", "displacement", "temperature"):
                errs.append(f"{tag}: point BCs only for fixed / displacement / temperature")
            if b.where is not None:
                if b.type not in ("fixed", "displacement", "temperature") or b.group is None:
                    errs.append(f"{tag}: where only for fixed / displacement / temperature on a group")
                else:
                    try:
                        _compile(b.where)
                    except (CaseError, SyntaxError) as e:
                        errs.append(f"{tag}: bad where expression: {e}")
            if b.type == "displacement":
                if not b.components or any(c not in range(ncomp) for c in b.components):
                    errs.append(f"{tag}: components must be a non-empty subset of {list(range(ncomp))}")
                if b.value is None:
                    errs.append(f"{tag}: value required")
                elif isinstance(b.value, list) and b.components and len(b.value) != len(b.components):
                    errs.append(f"{tag}: value list must match components")
            if b.type in ("traction", "force"):
                if not isinstance(b.value, list) or len(b.value) != ncomp:
                    errs.append(f"{tag}: value must be a list of {ncomp} components")
            if b.type in ("pressure", "temperature", "flux") and (b.value is None or isinstance(b.value, list)):
                errs.append(f"{tag}: value must be a scalar or expression")
            if b.type == "convection" and (b.h is None or b.T_inf is None):
                errs.append(f"{tag}: h and T_inf required")
            if b.type == "velocity_inlet" and (b.value is None or (isinstance(b.value, list) and len(b.value) != ncomp)):
                errs.append(f"{tag}: value must be a scalar normal speed or a list of {ncomp} components")
            if b.type == "pressure_outlet" and (b.value is None or isinstance(b.value, list)):
                errs.append(f"{tag}: value must be a scalar or expression")
            if b.type == "wall" and b.value is not None and (not isinstance(b.value, list) or len(b.value) != ncomp):
                errs.append(f"{tag}: wall velocity must be a list of {ncomp} components")
            if a.physics == "compressible" and b.type == "wall" and b.value is not None:
                errs.append(f"{tag}: compressible walls are stationary (no value)")
            if b.type == "freestream":
                if not isinstance(b.value, list) or len(b.value) != ncomp or any(
                        isinstance(v, (str, bool)) or not isinstance(v, (int, float)) for v in b.value):
                    errs.append(f"{tag}: value must be the freestream velocity, a list of {ncomp} numbers (m/s)")
                if b.p_inf is None or not b.p_inf > 0 or b.T_inf is None or not b.T_inf > 0:
                    errs.append(f"{tag}: p_inf > 0 (static pressure, Pa) and T_inf > 0 (static temperature, K) required")
            if b.type == "outlet" and b.value is not None:
                errs.append(f"{tag}: an outlet takes no value (supersonic outflow: everything is extrapolated)")
            if b.type == "total_inlet" and not (b.p0 is not None and b.p0 > 0 and b.T0 is not None and b.T0 > 0):
                errs.append(f"{tag}: p0 > 0 (total pressure, Pa) and T0 > 0 (total temperature, K) required")
        for i, l in enumerate(self.loads):
            tag = f"loads[{i}] ({l.type})"
            if l.type not in LOAD_TYPES[a.physics]:
                errs.append(f"{tag}: type must be one of {LOAD_TYPES[a.physics]}")
            elif l.type == "body_force" and (not isinstance(l.value, list) or len(l.value) != ncomp):
                errs.append(f"{tag}: value must be a list of {ncomp} components")
            elif l.type == "temperature" and self.materials and all(m.alpha == 0 for m in self.materials.values()):
                errs.append(f"{tag}: thermal load but every material has alpha = 0")
            if isinstance(l.value, dict) and "table" not in l.value and l.type in LOAD_TYPES[a.physics] \
                    and l.type != "body_force":      # a time table is checked with the time-dependent values
                if l.type != "temperature":
                    errs.append(f"{tag}: value must be a number or an expression (a field file is for a "
                                f"temperature load only)")
                elif set(l.value) != {"field"} or not isinstance(l.value["field"], str) or not l.value["field"].strip():
                    errs.append(f'{tag}: a field value must be {{"field": "<path to an .npz file with x (n, dim) '
                                f'and T (n,)>"}}, got {l.value!r}')
        fields = {"elasticity": ELASTICITY_FIELDS, "heat": HEAT_FIELDS, "flow": FLOW_FIELDS,
                  "compressible": COMPRESSIBLE_FIELDS}[a.physics]
        names = set()
        for i, q in enumerate(self.qois):
            tag = f"qois[{i}] ({q.name})"
            if q.name in names:
                errs.append(f"{tag}: duplicate QoI name")
            names.add(q.name)
            if q.type not in QOI_TYPES:
                errs.append(f"{tag}: type must be one of {QOI_TYPES}")
                continue
            if q.type in ("point", "point_difference", "max", "min", "max_abs", "integral", "mean") \
                    and q.field not in fields:
                if not (a.physics == "flow" and q.field in WALL_FIELDS and q.type in ("max", "min", "mean")
                        and q.group is not None):
                    errs.append(f"{tag}: field must be one of {fields}"
                                + (f" (or, on a wall group with max / min / mean, one of {WALL_FIELDS})"
                                   if a.physics == "flow" else ""))
            if q.type == "point_difference" and (q.point is None or q.point2 is None
                                                 or len(q.point) not in (2, 3) or len(q.point2) not in (2, 3)):
                errs.append(f"{tag}: point_difference needs point and point2 ([x, y(, z)])")
            if q.type == "point" and (q.point is None or len(q.point) not in (2, 3)):
                errs.append(f"{tag}: point [x, y(, z)] required")
            if q.type == "reaction" and (q.group is None or q.component not in range(ncomp)):
                errs.append(f"{tag}: reaction needs group and component in {list(range(ncomp))}")
            if q.type == "strain_energy" and a.physics != "elasticity":
                errs.append(f"{tag}: strain_energy is elasticity-only")
            if q.type in ("force", "flow_rate", "mass_flow") and a.physics not in ("flow", "compressible"):
                errs.append(f"{tag}: {q.type} is flow-only")
            if q.type == "mass_flow" and q.group is None:
                errs.append(f"{tag}: mass_flow needs group")
            if q.type in ("natural_frequency", "effective_mass"):
                if a.modal is None:
                    errs.append(f"{tag}: {q.type} needs a modal analysis (analysis.options.modal)")
                if not isinstance(q.mode, int) or isinstance(q.mode, bool) or q.mode < 1:
                    errs.append(f"{tag}: {q.type} needs mode, an integer >= 1")
                if q.type == "effective_mass" and q.component not in range(ncomp):
                    errs.append(f"{tag}: effective_mass needs component in {list(range(ncomp))}")
            elif a.modal is not None:
                errs.append(f"{tag}: a modal analysis gives natural_frequency and effective_mass only")
            if q.time is not None or q.window is not None:
                if a.transient is None:
                    errs.append(f"{tag}: time / window need a transient analysis (analysis.options.transient)")
                elif isinstance(q.time, str) and q.time not in QOI_TIME:
                    errs.append(f"{tag}: time must be a number or one of {QOI_TIME}")
                elif isinstance(q.time, bool) or not isinstance(q.time, (int, float, str, type(None))):
                    errs.append(f"{tag}: time must be a number or one of {QOI_TIME}")
                if q.window is not None and not (isinstance(q.window, list) and len(q.window) == 2
                                                 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                         for v in q.window) and q.window[0] < q.window[1]):
                    errs.append(f"{tag}: window must be [t0, t1] with t0 < t1")
            if q.type == "force" and (q.group is None or q.component not in range(ncomp)):
                errs.append(f"{tag}: force needs group and component in {list(range(ncomp))}")
            if q.type == "flow_rate" and q.group is None:
                errs.append(f"{tag}: flow_rate needs group")
        return errs

    def check(self) -> "Case":
        errs = self.validate()
        if errs:
            raise CaseError("invalid case:\n  " + "\n  ".join(errs))
        return self


# --------------------------------------------------------------------------- nonlinear

PLASTICITY_MODELS = ("j2_isotropic",)
NONLINEAR_OPTIONS = ("nonlinear", "increments", "load_factor_max", "nlgeom", "adaptive", "failure")
FAILURE_CRITERIA = ("peeq", "peeq_tension", "principal")


def hardening_table(p: dict) -> list[tuple[float, float]]:
    """(plastic strain, flow stress) points of a ``Material.plasticity`` dict, starting at (0, sy)."""
    sy = float(p["yield_stress"])
    pts = [(float(e), float(s)) for e, s in (p.get("hardening") or [])]
    if not pts or pts[0][0] > 0:
        pts.insert(0, (0.0, sy))
    return pts


def _validate_nonlinear(case: "Case") -> list[str]:
    a, errs = case.analysis, []
    opts = a.options or {}
    plastic = {n: m.plasticity for n, m in case.materials.items() if m.plasticity is not None}
    if plastic and not a.nonlinear:
        errs.append(f"materials {sorted(plastic)} have plasticity: set analysis.options.nonlinear = true")
    if a.nonlinear:
        if a.physics != "elasticity":
            errs.append("analysis.options.nonlinear is only defined for elasticity")
        n = opts.get("increments", 20)
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            errs.append("analysis.options.increments must be an integer >= 1")
        lf = opts.get("load_factor_max", 1.0)
        if not isinstance(lf, (int, float)) or isinstance(lf, bool) or not lf > 0:
            errs.append("analysis.options.load_factor_max must be a number > 0")
    fail = opts.get("failure")
    if fail is not None:
        if not a.nonlinear or not plastic:
            errs.append("analysis.options.failure needs nonlinear = true and a material with plasticity")
        if not isinstance(fail, dict) or fail.get("criterion") not in FAILURE_CRITERIA:
            errs.append(f"analysis.options.failure.criterion must be one of {FAILURE_CRITERIA}")
        elif not isinstance(fail.get("limit"), (int, float)) or not fail["limit"] > 0:
            errs.append("analysis.options.failure.limit must be a number > 0")
        elif fail.get("aggregate", "mean") not in ("mean", "max"):
            errs.append("analysis.options.failure.aggregate must be 'mean' or 'max'")
        elif fail["criterion"] == "principal" and not (isinstance(fail.get("stress_limit"), (int, float))
                                                       and fail["stress_limit"] > 0):
            errs.append("analysis.options.failure.stress_limit (Pa) must be a number > 0 for criterion 'principal'")
    for name, p in plastic.items():
        tag = f"material {name!r}: plasticity"
        if not isinstance(p, dict) or p.get("model") not in PLASTICITY_MODELS:
            errs.append(f"{tag}: model must be one of {PLASTICITY_MODELS}")
            continue
        sy = p.get("yield_stress")
        if not isinstance(sy, (int, float)) or not sy > 0:
            errs.append(f"{tag}: yield_stress must be > 0")
            continue
        try:
            pts = hardening_table(p)
        except (TypeError, ValueError):
            errs.append(f"{tag}: hardening must be a list of [plastic_strain, stress] pairs")
            continue
        e, s = np.array(pts).T
        if np.any(np.diff(e) <= 0) or e[0] < 0:
            errs.append(f"{tag}: hardening plastic strains must start at >= 0 and increase")
        if abs(s[0] - sy) > 1e-9 * sy or np.any(np.diff(s) < 0):
            errs.append(f"{tag}: hardening must start at yield_stress and not soften")
    return errs


# --------------------------------------------------------------------------- time-dependent and modal


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_time(case: "Case") -> list[str]:
    """``options["transient"]``, ``options["modal"]``, ``options["damping"]``, and values that depend on
    time (module doc, "Time-dependent and modal analyses")."""
    a, errs = case.analysis, []
    opts = a.options or {}
    tr, mo = opts.get("transient"), opts.get("modal")
    for key, val in (("transient", tr), ("modal", mo)):
        if val is not None and not isinstance(val, dict):
            errs.append(f"analysis.options.{key} must be a dict")
    tr = tr if isinstance(tr, dict) and tr else None
    mo = mo if isinstance(mo, dict) and mo else None
    if tr is not None:
        tag = "analysis.options.transient"
        if a.physics not in TRANSIENT_SCHEMES:
            errs.append(f"{tag}: a transient analysis is defined for {tuple(TRANSIENT_SCHEMES)}, not {a.physics!r}")
        if a.nonlinear:
            errs.append(f"{tag}: a transient analysis cannot also be nonlinear")
        unknown = sorted(set(tr) - set(TRANSIENT_OPTIONS))
        if unknown:
            errs.append(f"{tag}: unknown keys {unknown}; keys: {TRANSIENT_OPTIONS}")
        t_end = tr.get("t_end")
        if not _num(t_end) or not t_end > 0:
            errs.append(f"{tag}.t_end must be a number > 0 (s)")
            t_end = None
        dt, n = tr.get("dt"), tr.get("n_steps")
        if (dt is None) == (n is None):
            errs.append(f"{tag}: give exactly one of dt (s) and n_steps")
        elif dt is not None and (not _num(dt) or not dt > 0 or (t_end is not None and dt > t_end)):
            errs.append(f"{tag}.dt must be a number > 0 and at most t_end")
        elif n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
            errs.append(f"{tag}.n_steps must be an integer >= 1")
        schemes = TRANSIENT_SCHEMES.get(a.physics, ())
        if tr.get("scheme") is not None and tr["scheme"] not in schemes:
            errs.append(f"{tag}.scheme for {a.physics} must be one of {schemes}")
        if tr.get("alpha") is not None and (tr.get("scheme") != "hht" or not _num(tr["alpha"])
                                            or not -1.0 / 3.0 - 1e-12 <= tr["alpha"] <= 0.0):
            errs.append(f"{tag}.alpha is for scheme 'hht' and must be in [-1/3, 0]")
        ini = tr.get("initial")
        if ini is not None:
            keys = {"heat": ("T",), "elasticity": ("u", "v"), "flow": ("U", "p", "steady")}.get(a.physics, ())
            if not isinstance(ini, dict) or set(ini) - set(keys):
                errs.append(f"{tag}.initial for {a.physics} must be a dict with keys from {keys}")
            elif any(uses_time(v) for v in ini.values() if not isinstance(v, bool)):
                errs.append(f"{tag}.initial is the state at t = 0: it cannot depend on t")
        ot, oe = tr.get("output_times"), tr.get("output_every")
        if ot is not None and oe is not None:
            errs.append(f"{tag}: give output_times or output_every, not both")
        if ot is not None and not (isinstance(ot, list) and ot and all(_num(v) and v >= 0 for v in ot)
                                   and (t_end is None or max(ot) <= t_end * (1 + 1e-12))):
            errs.append(f"{tag}.output_times must be a list of times in [0, t_end]")
        if oe is not None and (not isinstance(oe, int) or isinstance(oe, bool) or oe < 1):
            errs.append(f"{tag}.output_every must be an integer >= 1")
        if tr.get("max_co") is not None and (a.physics != "flow" or not _num(tr["max_co"]) or not tr["max_co"] > 0):
            errs.append(f"{tag}.max_co is for flow and must be a number > 0")
        for name, m in case.materials.items():
            if a.physics == "heat" and not (m.rho > 0 and m.cp > 0):
                errs.append(f"material {name!r}: transient heat needs rho > 0 (kg/m^3) and cp > 0 (J/(kg K))")
            if a.physics == "elasticity" and not m.rho > 0:
                errs.append(f"material {name!r}: a dynamic analysis needs rho > 0 (kg/m^3)")
    if mo is not None:
        tag = "analysis.options.modal"
        if a.physics != "elasticity":
            errs.append(f"{tag}: a modal analysis is defined for elasticity only")
        if tr is not None or a.nonlinear:
            errs.append(f"{tag}: a modal analysis cannot also be transient or nonlinear")
        unknown = sorted(set(mo) - set(MODAL_OPTIONS))
        if unknown:
            errs.append(f"{tag}: unknown keys {unknown}; keys: {MODAL_OPTIONS}")
        n = mo.get("n_modes")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            errs.append(f"{tag}.n_modes must be an integer >= 1")
        if mo.get("shift") is not None and (not _num(mo["shift"]) or mo["shift"] < 0):
            errs.append(f"{tag}.shift must be a frequency >= 0 (Hz)")
        for name, m in case.materials.items():
            if not m.rho > 0:
                errs.append(f"material {name!r}: a modal analysis needs rho > 0 (kg/m^3)")
    damp = opts.get("damping")
    if damp is not None:
        ray = damp.get("rayleigh") if isinstance(damp, dict) else None
        if not (isinstance(damp, dict) and set(damp) == {"rayleigh"} and isinstance(ray, list) and len(ray) == 2
                and all(_num(v) and v >= 0 for v in ray)):
            errs.append('analysis.options.damping must be {"rayleigh": [a, b]} with a, b >= 0 (C = a M + b K)')
        elif not (tr is not None and a.physics == "elasticity"):
            errs.append("analysis.options.damping is for a transient elasticity (dynamic) analysis")
    # values that depend on time
    timed = [(f"bcs[{i}] ({b.type}) {k}", v) for i, b in enumerate(case.bcs)
             for k, v in (("value", b.value), ("h", b.h), ("T_inf", b.T_inf)) if v is not None]
    timed += [(f"loads[{i}] ({l.type}) value", l.value) for i, l in enumerate(case.loads)]
    for tag, v in timed:
        for part in (v if isinstance(v, list) else [v]):
            if isinstance(part, dict) and "table" in part:
                err = _table_error(part)
                if err:
                    errs.append(f"{tag}: {err}")
        try:
            dep = uses_time(v)
        except (CaseError, SyntaxError):
            continue                  # a bad expression is reported where it is used
        if dep and tr is None:
            errs.append(f"{tag} depends on the time t: that needs a transient analysis (analysis.options.transient)")
    if tr is None:
        for i, b in enumerate(case.bcs):
            if b.type == "convection" and not (_num(b.h) and _num(b.T_inf)) and b.h is not None and b.T_inf is not None:
                errs.append(f"bcs[{i}] (convection): h and T_inf must be numbers in a steady analysis")
    return errs


def _table_error(value: dict) -> str | None:
    tab = value.get("table")
    if set(value) != {"table"} or not isinstance(tab, list) or not tab:
        return 'a table must be {"table": [[t0, v0], [t1, v1], ...]}'
    if not all(isinstance(r, (list, tuple)) and len(r) == 2 and _num(r[0]) and _num(r[1]) for r in tab):
        return "every table row must be [time, value], two numbers"
    ts = [r[0] for r in tab]
    if any(b <= a_ for a_, b in zip(ts, ts[1:])):
        return "the table times must increase"
    return None


def uses_time(value) -> bool:
    """True when a value (a number, an expression, a table, or a list of these) depends on the time t."""
    if isinstance(value, dict):
        return "table" in value
    if isinstance(value, str):
        return "t" in _compile(value).co_names
    if isinstance(value, (list, tuple)):
        return any(uses_time(v) for v in value)
    return False


def table_value(value: dict, t: float) -> float:
    """A table at time t: linear between its points, constant before the first and after the last."""
    tab = np.asarray(value["table"], dtype=float)
    return float(np.interp(float(t), tab[:, 0], tab[:, 1]))


# --------------------------------------------------------------------------- expressions

_FUNCS = {
    n: getattr(np, n)
    for n in (
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh",
        "exp", "log", "log10", "sqrt", "abs", "minimum", "maximum", "where", "hypot", "sign",
    )
}
_CONSTS = {"pi": np.pi, "e": np.e}
_VARS = ("x", "y", "z", "t")      # t: the time of a transient analysis (seconds)
_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
)


def _compile(expr: str):
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise CaseError(f"disallowed syntax in expression {expr!r}: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _FUNCS and node.id not in _CONSTS and node.id not in _VARS:
            raise CaseError(f"unknown name {node.id!r} in expression {expr!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise CaseError(f"disallowed call in expression {expr!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise CaseError(f"only numeric constants allowed in expression {expr!r}")
    return compile(tree, "<expr>", "eval")


def eval_expr(expr: str, x, y, z=None, t: float | None = None) -> np.ndarray:
    """Evaluate a restricted arithmetic expression in x, y, z (arrays of equal shape) and, in a transient
    analysis, the time t (a number). An expression that uses t without a time is a ``CaseError``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.zeros_like(x) if z is None else np.asarray(z, dtype=float)
    env = {"x": x, "y": y, "z": z, **_FUNCS, **_CONSTS}
    code = _compile(expr)
    if "t" in code.co_names:
        if t is None:
            raise CaseError(f"expression {expr!r} uses the time t, which only a transient analysis has")
        env["t"] = np.float64(t)
    # errstate: a numpy floating-point warning (e.g. 1/x at r = 0) would otherwise go through the
    # warnings machinery, which needs __import__ and fails under the empty builtins below.
    with np.errstate(all="ignore"):
        out = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - AST-whitelisted above
    return np.broadcast_to(np.asarray(out, dtype=float), x.shape).copy()


def as_field(value: float | str | dict, x, y, z=None, t: float | None = None) -> np.ndarray:
    """Scalar, expression or time table -> array of values at the given coordinates (and time)."""
    if isinstance(value, str):
        return eval_expr(value, x, y, z, t)
    if isinstance(value, dict) and "table" in value:
        if t is None:
            raise CaseError(f"a table ({value!r}) gives a value in time, which only a transient analysis has")
        return np.full(np.shape(x), table_value(value, t))
    if isinstance(value, dict):
        raise CaseError(f"a field file ({value!r}) is not a number or an expression: only a temperature load "
                        f"of the skfem backend takes one (it maps the field to its nodes)")
    return np.full(np.shape(x), float(value))


# --------------------------------------------------------------------------- temperature field files


def write_temperature_field(path: str | Path, x, T) -> Path:
    """Write a temperature field for ``Load(type="temperature", value={"field": path})``: an .npz file
    with ``x`` (n, dim) node coordinates and ``T`` (n,) temperatures. For a heat result of the same
    mesh and order: ``write_temperature_field(path, *result.nodal("T"))``."""
    x, T = np.asarray(x, dtype=float), np.asarray(T, dtype=float).ravel()
    if x.ndim != 2 or x.shape[1] not in (2, 3) or len(x) != len(T) or len(T) == 0:
        raise CaseError(f"a temperature field needs x (n, 2 or 3) and T (n,), got x {x.shape} and T {T.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:          # an open file: np.savez would add ".npz" to another suffix
        np.savez(f, x=x, T=T)
    return path


def read_temperature_field(path: str | Path, dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read a temperature field file: (x (n, dim), T (n,)). Every problem is a ``CaseError`` that names
    the file. With ``dim`` = 2, a file with three coordinates is accepted when its third one is constant
    (a 2D mesh lies in the xy-plane) and comes back with two."""
    path = Path(path)
    if not path.is_file():
        raise CaseError(f"temperature field file not found: {str(path)!r}")
    try:
        with np.load(path, allow_pickle=False) as z:
            names = list(getattr(z, "files", []))
            if "x" not in names or "T" not in names:
                raise CaseError(f"temperature field {str(path)!r}: an .npz file with the arrays x (n, dim) and "
                                f"T (n,) is required; it has {names or 'no named arrays'}")
            x, T = np.asarray(z["x"], dtype=float), np.asarray(z["T"], dtype=float)
    except CaseError:
        raise
    except Exception as e:  # noqa: BLE001  numpy raises several kinds for a file that is not an .npz
        raise CaseError(f"cannot read temperature field {str(path)!r}: {type(e).__name__}: {e}") from e
    if T.ndim == 2 and T.shape[1] == 1:
        T = T[:, 0]
    if x.ndim != 2 or x.shape[1] not in (2, 3) or T.ndim != 1 or len(x) != len(T) or len(T) == 0:
        raise CaseError(f"temperature field {str(path)!r}: need x (n, 2 or 3) and T (n,), got x {x.shape} and "
                        f"T {T.shape}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(T))):
        raise CaseError(f"temperature field {str(path)!r}: x or T holds values that are not finite")
    if dim is not None and x.shape[1] != dim:
        span = float(np.ptp(x[:, :2], axis=0).max())
        if dim == 2 and float(np.ptp(x[:, 2])) <= 1e-9 * max(span, 1e-300):
            x = x[:, :2]
        else:
            raise CaseError(f"temperature field {str(path)!r}: its points have {x.shape[1]} coordinates but the "
                            f"analysis is {dim}D")
    return np.ascontiguousarray(x), T
