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
  ``max_iterations``, ``residual_tol`` (keep it tight, e.g. 1e-8, so the iteration error stays far
  below the discretization error that the GCI measures).

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
  force          flow: total fluid force (pressure + viscous) on the body through ``group``, ``component``
  flow_rate      flow: volumetric flow rate through ``group``, positive out of the domain
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

SCHEMA_VERSION = 1

PHYSICS = ("elasticity", "heat", "flow")
FORMULATIONS = {
    "elasticity": ("plane_stress", "plane_strain", "axisymmetric", "3d"),
    "heat": ("2d", "axisymmetric", "3d"),
    "flow": ("2d", "3d"),
}
BC_TYPES = {
    "elasticity": ("fixed", "displacement", "symmetry", "pressure", "traction", "force"),
    "heat": ("temperature", "flux", "convection"),
    "flow": ("velocity_inlet", "pressure_outlet", "wall", "slip", "symmetry"),
}
LOAD_TYPES = {
    "elasticity": ("body_force", "temperature"),
    "heat": ("source",),
    "flow": (),
}
QOI_TYPES = ("point", "point_difference", "max", "min", "max_abs", "reaction", "integral", "mean",
             "strain_energy", "force", "flow_rate")

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
DISPLACEMENT_LIKE = ("ux", "uy", "uz", "u_mag", "T")
STRESS_LIKE = tuple(f for f in ELASTICITY_FIELDS if f not in DISPLACEMENT_LIKE) + ("qx", "qy", "qz", "q_mag")

Value = Union[float, str, list]


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
    rho: float = 1.0     # density (flow)
    mu: float = 0.0      # dynamic viscosity (flow); kinematic viscosity = mu / rho


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
    h: float | None = None              # convection coefficient
    T_inf: float | None = None          # convection ambient temperature


@dataclass
class Load:
    type: str
    value: Value
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
        for m in d["materials"].values():      # linear cases serialise exactly as before
            if m.get("plasticity") is None:
                m.pop("plasticity", None)
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
        errs += _validate_nonlinear(self)
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
        for i, l in enumerate(self.loads):
            tag = f"loads[{i}] ({l.type})"
            if l.type not in LOAD_TYPES[a.physics]:
                errs.append(f"{tag}: type must be one of {LOAD_TYPES[a.physics]}")
            elif l.type == "body_force" and (not isinstance(l.value, list) or len(l.value) != ncomp):
                errs.append(f"{tag}: value must be a list of {ncomp} components")
            elif l.type == "temperature" and self.materials and all(m.alpha == 0 for m in self.materials.values()):
                errs.append(f"{tag}: thermal load but every material has alpha = 0")
        fields = {"elasticity": ELASTICITY_FIELDS, "heat": HEAT_FIELDS, "flow": FLOW_FIELDS}[a.physics]
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
                errs.append(f"{tag}: field must be one of {fields}")
            if q.type == "point_difference" and (q.point is None or q.point2 is None
                                                 or len(q.point) not in (2, 3) or len(q.point2) not in (2, 3)):
                errs.append(f"{tag}: point_difference needs point and point2 ([x, y(, z)])")
            if q.type == "point" and (q.point is None or len(q.point) not in (2, 3)):
                errs.append(f"{tag}: point [x, y(, z)] required")
            if q.type == "reaction" and (q.group is None or q.component not in range(ncomp)):
                errs.append(f"{tag}: reaction needs group and component in {list(range(ncomp))}")
            if q.type == "strain_energy" and a.physics != "elasticity":
                errs.append(f"{tag}: strain_energy is elasticity-only")
            if q.type in ("force", "flow_rate") and a.physics != "flow":
                errs.append(f"{tag}: {q.type} is flow-only")
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


# --------------------------------------------------------------------------- expressions

_FUNCS = {
    n: getattr(np, n)
    for n in (
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh",
        "exp", "log", "log10", "sqrt", "abs", "minimum", "maximum", "where", "hypot", "sign",
    )
}
_CONSTS = {"pi": np.pi, "e": np.e}
_VARS = ("x", "y", "z")
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


def eval_expr(expr: str, x, y, z=None) -> np.ndarray:
    """Evaluate a restricted arithmetic expression in x, y, z (arrays of equal shape)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.zeros_like(x) if z is None else np.asarray(z, dtype=float)
    env = {"x": x, "y": y, "z": z, **_FUNCS, **_CONSTS}
    # errstate: a numpy floating-point warning (e.g. 1/x at r = 0) would otherwise go through the
    # warnings machinery, which needs __import__ and fails under the empty builtins below.
    with np.errstate(all="ignore"):
        out = eval(_compile(expr), {"__builtins__": {}}, env)  # noqa: S307 - AST-whitelisted above
    return np.broadcast_to(np.asarray(out, dtype=float), x.shape).copy()


def as_field(value: float | str, x, y, z=None) -> np.ndarray:
    """Scalar or expression -> array of values at the given coordinates."""
    if isinstance(value, str):
        return eval_expr(value, x, y, z)
    return np.full(np.shape(x), float(value))
