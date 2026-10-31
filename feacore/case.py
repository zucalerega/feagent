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
  quadratic cells are treated as curved (isoparametric) geometry.
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
points in 3D), constraining every node on them.
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

Quantities of interest (``QoI.type``)
-------------------------------------
  point          ``field`` at ``point`` (stresses use the solver's recovered, continuous field)
  max/min/max_abs extreme of the recovered nodal ``field`` over ``group`` (None = whole mesh)
  reaction       total reaction force ``component`` on Dirichlet group ``group``
  integral/mean  integral / average of ``field`` over cell or facet ``group``
  strain_energy  total strain energy 0.5 u.K.u (elasticity only)
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

SCHEMA_VERSION = 1

PHYSICS = ("elasticity", "heat")
FORMULATIONS = {
    "elasticity": ("plane_stress", "plane_strain", "axisymmetric", "3d"),
    "heat": ("2d", "axisymmetric", "3d"),
}
BC_TYPES = {
    "elasticity": ("fixed", "displacement", "symmetry", "pressure", "traction", "force"),
    "heat": ("temperature", "flux", "convection"),
}
LOAD_TYPES = {
    "elasticity": ("body_force", "temperature"),
    "heat": ("source",),
}
QOI_TYPES = ("point", "max", "min", "max_abs", "reaction", "integral", "mean", "strain_energy")

# Fields a backend must be able to probe. In 2D, z-components are 0 (plane strain szz is
# the out-of-plane stress). Axisymmetric: sxx = s_rr, syy = s_zz, sxy = s_rz, stt = hoop.
ELASTICITY_FIELDS = (
    "ux", "uy", "uz", "u_mag",
    "sxx", "syy", "szz", "sxy", "syz", "sxz", "stt",
    "von_mises", "tresca", "s1", "s2", "s3",
)
HEAT_FIELDS = ("T", "qx", "qy", "qz", "q_mag")  # q = -k grad T
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


@dataclass
class Analysis:
    physics: str = "elasticity"
    formulation: str = "3d"
    order: int = 2                 # polynomial order of the FE interpolation (1 or 2)
    thickness: float = 1.0         # plane stress / plane strain thickness
    T_ref: float = 0.0             # stress-free reference temperature
    options: dict = field(default_factory=dict)  # backend options, e.g. {"stress_recovery": "l2"}


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
        for i, l in enumerate(self.loads):
            tag = f"loads[{i}] ({l.type})"
            if l.type not in LOAD_TYPES[a.physics]:
                errs.append(f"{tag}: type must be one of {LOAD_TYPES[a.physics]}")
            elif l.type == "body_force" and (not isinstance(l.value, list) or len(l.value) != ncomp):
                errs.append(f"{tag}: value must be a list of {ncomp} components")
            elif l.type == "temperature" and self.materials and all(m.alpha == 0 for m in self.materials.values()):
                errs.append(f"{tag}: thermal load but every material has alpha = 0")
        fields = ELASTICITY_FIELDS if a.physics == "elasticity" else HEAT_FIELDS
        names = set()
        for i, q in enumerate(self.qois):
            tag = f"qois[{i}] ({q.name})"
            if q.name in names:
                errs.append(f"{tag}: duplicate QoI name")
            names.add(q.name)
            if q.type not in QOI_TYPES:
                errs.append(f"{tag}: type must be one of {QOI_TYPES}")
                continue
            if q.type in ("point", "max", "min", "max_abs", "integral", "mean") and q.field not in fields:
                errs.append(f"{tag}: field must be one of {fields}")
            if q.type == "point" and (q.point is None or len(q.point) not in (2, 3)):
                errs.append(f"{tag}: point [x, y(, z)] required")
            if q.type == "reaction" and (q.group is None or q.component not in range(ncomp)):
                errs.append(f"{tag}: reaction needs group and component in {list(range(ncomp))}")
            if q.type == "strain_energy" and a.physics != "elasticity":
                errs.append(f"{tag}: strain_energy is elasticity-only")
        return errs

    def check(self) -> "Case":
        errs = self.validate()
        if errs:
            raise CaseError("invalid case:\n  " + "\n  ".join(errs))
        return self


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
    out = eval(_compile(expr), {"__builtins__": {}}, env)  # noqa: S307 - AST-whitelisted above
    return np.broadcast_to(np.asarray(out, dtype=float), x.shape).copy()


def as_field(value: float | str, x, y, z=None) -> np.ndarray:
    """Scalar or expression -> array of values at the given coordinates."""
    if isinstance(value, str):
        return eval_expr(value, x, y, z)
    return np.full(np.shape(x), float(value))
