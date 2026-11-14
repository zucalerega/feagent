"""Idealization: the loads and supports of a load case as boundary conditions of the fixed solver.

    ideal = Idealizer(project, geo, lc, journal, thermal=False)   # the rules; raises NotSupported
    real = ideal.realize(MeshGeom(mesh_path))                     # the numbers on one mesh
    real.bcs, real.loads                                          # feacore BC and Load objects
    ideal.record_items(real)                                      # the BCItems of the analysis record

Rules (each rule is one Decision of the load case; the BCItems point to it)

  pressure      a uniform pressure normal to every face of the features of its role.
  force         the total force as a uniform traction over the union of its seat faces (the force is split
                between the faces by area, so the traction is the same on all of them). The resultant acts
                at the area centroid c of the faces. A point of application `at` away from c adds the
                moment (at - c) x F, applied like a moment.
  moment        the traction field t(x) = w x (x - c), with w = J^-1 M and
                J = integral over the faces of (|r|^2 I - r r^T) dA, r = x - c.
                Its net force is zero because c is the centroid, and its moment about c is J w = M. On a
                flat face with the normal n this is the classical result: the component of M along n is
                a torque (shear traction in proportion to the distance from c, with the polar moment),
                the components in the plane are bending (normal traction that is linear in the in-plane
                coordinates, with the second moments of area and the product of inertia: J holds them).
                On faces that are not in one plane the same field is still statically exact, but it is
                not a classical stress distribution: the Decision then asks for a review.
  acceleration  a body force of density times the acceleration vector on the whole part. The vector of
                the effect is the direction of the inertial load on the part, as gravity is.
  support       the seat faces of every interface in `lc.supports` are held.
                Static case: fixed in all directions.
                Thermo-structural case: the prescribed displacement u = alpha (T_face - T_ref) (x - c),
                with T_face the area mean of the temperature on the held faces (thermal solution on the
                same mesh) and c their area centroid. For one flat face that is zero displacement normal
                to the face and the free thermal contraction in its plane. A fully fixed face cannot
                contract and gives a false stress of about E alpha dT / (1 - nu).
                With several held interfaces the engine uses ONE centroid and ONE mean temperature for
                all of them (the neighbours are one body at one temperature). A check fails when the mean
                temperatures of the interfaces differ by more than the policy's thermal_delta.
  free part     no support: the loads must balance. The engine integrates every load on the mesh. The
                net force and the net moment (about the centre of volume) must be below BALANCE_TOL (1 %)
                of the largest load, else the analysis fails (Unbalanced). The scale of the force is the
                largest resultant of one load, but not less than 1 % of the largest integral of |traction|
                (so that a closed pressure boundary, whose resultant is zero, still has a scale). The
                scale of the moment is the largest moment of one load about the centre of volume, but not
                less than the force scale times half the bounding-box diagonal.
                Rigid-body motion is stopped at three nodes far apart (3-2-1: three, two and one
                displacement components). The residual of the balance (mesh error of curved faces, or
                load values that balance to less than 1 %) is taken out with a rigid-body acceleration
                field b(x) = a + w x (x - c_v) on the volume (inertia relief), so the three nodes carry
                no force. A check compares their reactions with the largest load.

A load that the engine cannot idealize makes the load case `not_supported` with the reason (NotSupported):
a load without features, a load kind without a rule (displacement), a held interface with a joint that
is not a fixed joint, a missing material property. Temperature effects are not mechanical loads: in a
thermo-structural case they enter through the temperature field of the thermal load case.

Geometry numbers (areas, centroids, the tensor J, the volume) come from the mesh of each level, with the
solver's own quadrature (order 4 on quadratic triangles and tetrahedra). So the net force of a moment
traction is zero on every mesh, and the checks integrate exactly what the solver applies: the resultants
are computed from the expression strings that go to the solver, with feacore's expression evaluator.

Not verified / limits
  * The force and moment tractions are statically equivalent to the loads. The stress next to the seat
    faces depends on the real load path (bolts, gasket) and is approximate (an Assumption says so).
  * The thermal support takes the neighbour at the temperature and expansion of the held face. A
    neighbour with another expansion gives a stress that is not in the results (an Assumption says so).
  * A moment on faces that are not in one plane is supported, but only flat faces were verified against
    beam theory in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dossier import units
from dossier.model import BCItem, LoadCase, LoadItem
from dossier.records import Journal

HELD_JOINTS = ("", "bolted_flange", "welded", "bonded", "clamped", "threaded")
THERMAL_KINDS = ("fluid_temperature", "temperature", "heat_flux")
NO_STRUCTURAL_KINDS = ("mass_flow", "signal")
BALANCE_TOL = 0.01          # free part: net load / largest load
MOMENT_TOL = 0.01           # moment traction: |integrated moment - load| / |load|
REACTION_TOL = 1e-3         # free part: largest restraint reaction / largest load
QUAD_ORDER = 4              # the solver's quadrature order for quadratic elements (2 x order)


class NotSupported(Exception):
    """The engine has no model for this load case. The message is the exact reason (STE)."""


class Unbalanced(Exception):
    """The loads of a free part do not balance. The message gives the numbers (STE)."""


# --------------------------------------------------------------------------- text helpers


def fmt(value, unit: str) -> str:
    """An SI number or vector for people: fmt(15.75e6, "Pa") -> "15.75 MPa"."""
    v = np.asarray(value, dtype=float)
    return units.display(units.from_si(v.tolist() if v.ndim else float(v), unit))


def names(items, limit: int = 6) -> str:
    items = list(items)
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" and {len(items) - limit} more"


def _num(v: float) -> str:
    return f"({float(v)!r})"


def rigid_exprs(a, w, c) -> list[str]:
    """The field a + w x (x - c) as three expressions in x, y, z for the solver."""
    a, w, c = (np.asarray(v, dtype=float) for v in (a, w, c))
    rx, ry, rz = f"(x - {_num(c[0])})", f"(y - {_num(c[1])})", f"(z - {_num(c[2])})"
    return [f"{_num(a[0])} + {_num(w[1])}*{rz} - {_num(w[2])}*{ry}",
            f"{_num(a[1])} + {_num(w[2])}*{rx} - {_num(w[0])}*{rz}",
            f"{_num(a[2])} + {_num(w[0])}*{ry} - {_num(w[1])}*{rx}"]


def eval_exprs(exprs: list[str], x: np.ndarray) -> np.ndarray:
    """Three solver expressions at points (n, 3) -> (n, 3), with the solver's own evaluator."""
    from feacore.case import as_field
    return np.column_stack([as_field(e, x[:, 0], x[:, 1], x[:, 2]) for e in exprs])


def inertia_tensor(x: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """J = sum of w (|r|^2 I - r r^T), r = x - c: the second moments of a surface or a volume about c."""
    r = x - c
    return np.einsum("q,q->", w, np.einsum("qi,qi->q", r, r)) * np.eye(3) - np.einsum("q,qi,qj->ij", w, r, r)


# --------------------------------------------------------------------------- the mesh as the solver sees it


@dataclass
class SurfQuad:
    """Quadrature of a set of boundary faces: points (m, 3), weights (m,), outward unit normals (m, 3)."""

    x: np.ndarray
    w: np.ndarray
    n: np.ndarray
    basis: Any = field(repr=False, default=None)

    @property
    def area(self) -> float:
        return float(self.w.sum())

    @property
    def centroid(self) -> np.ndarray:
        return (self.w @ self.x) / self.w.sum()

    def inertia(self, c=None) -> np.ndarray:
        return inertia_tensor(self.x, self.w, self.centroid if c is None else np.asarray(c, float))

    def mean_normal(self) -> np.ndarray:
        return (self.w @ self.n) / self.w.sum()

    def planar(self, tol: float = 1e-6) -> bool:
        """True when the faces lie in one plane (same normal everywhere, all points in the plane)."""
        nb = self.mean_normal()
        if abs(np.linalg.norm(nb) - 1.0) > tol:
            return False
        size = max(float(np.ptp(self.x, axis=0).max()), 1e-300)
        return bool(np.abs((self.x - self.centroid) @ nb).max() <= tol * size)

    def mean(self, nodal: np.ndarray) -> float:
        """Area mean of a nodal field (n_nodes,)."""
        v = np.asarray(self.basis.interpolate(np.asarray(nodal, dtype=float))).ravel()
        return float(self.w @ v / self.w.sum())


class MeshGeom:
    """One mesh level as the fixed solver reads it: the node set of the quadratic solution, the named
    face groups with their quadrature, and the volume. `x` is the node set of every solve on this mesh,
    in the solver's order (the fields files use it)."""

    def __init__(self, path: str | Path):
        from skfem import Basis, ElementTetP2

        from feacore.solver._skfem_mesh import read_mesh
        self.path = Path(path)
        self.md = read_mesh(self.path, 3, 2)
        self.mesh = self.md.mesh
        self.elem = ElementTetP2()
        self.basis = Basis(self.mesh, self.elem, intorder=QUAD_ORDER)
        self.x = np.ascontiguousarray(self.basis.doflocs.T)
        self.diag = float(self.md.bbox_diag)
        self.center = 0.5 * (self.md.bbox[0] + self.md.bbox[1])
        self._quad: dict[tuple, SurfQuad] = {}
        self._nodes: dict[tuple, np.ndarray] = {}
        self._volume = None
        self._node_h = None

    # ---- groups
    def has(self, name: str) -> bool:
        g = self.md.groups.get(name)
        return g is not None and g.kind == "facets" and len(g.index) > 0

    def missing(self, groups) -> list[str]:
        return [g for g in groups if not self.has(g)]

    def facets(self, groups) -> np.ndarray:
        idx = [self.md.groups[g].index for g in groups]
        return np.unique(np.concatenate(idx)) if idx else np.zeros(0, dtype=np.int64)

    def nodes(self, groups) -> np.ndarray:
        """Indices (into `x`) of the nodes on the faces of the groups."""
        key = tuple(sorted(groups))
        if key not in self._nodes:
            f = self.facets(key)
            self._nodes[key] = (np.unique(np.asarray(self.basis.get_dofs(facets=f).all())) if len(f)
                                else np.zeros(0, dtype=np.int64))
        return self._nodes[key]

    def quad(self, groups) -> SurfQuad:
        key = tuple(sorted(groups))
        if key not in self._quad:
            from skfem import FacetBasis
            fb = FacetBasis(self.mesh, self.elem, facets=self.facets(key), intorder=QUAD_ORDER)
            x = np.asarray(fb.global_coordinates()).reshape(3, -1).T
            self._quad[key] = SurfQuad(x, np.asarray(fb.dx).ravel(), np.asarray(fb.normals).reshape(3, -1).T, fb)
        return self._quad[key]

    # ---- volume
    def volume(self) -> tuple[float, np.ndarray, np.ndarray]:
        """(volume, centre of volume, J about it) of the mesh."""
        if self._volume is None:
            x = np.asarray(self.basis.global_coordinates()).reshape(3, -1).T
            w = np.asarray(self.basis.dx).ravel()
            v = float(w.sum())
            c = (w @ x) / v
            self._volume = (v, c, inertia_tensor(x, w, c))
        return self._volume

    def node_h(self) -> np.ndarray:
        """The local element size at every node: the largest mean edge length of the cells at the node."""
        if self._node_h is None:
            p = self.mesh.p[:, self.mesh.t[:4]]                      # (3, 4, cells): corner vertices
            pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            h = np.mean([np.linalg.norm(p[:, i] - p[:, j], axis=0) for i, j in pairs], axis=0)
            out = np.zeros(len(self.x))
            dofs = self.basis.element_dofs
            for k in range(dofs.shape[0]):
                np.maximum.at(out, dofs[k], h)
            self._node_h = out
        return self._node_h

    def min_quality(self) -> float:
        from feacore.diagnostics.checks import mesh_quality
        return float(mesh_quality(self.mesh).min())


# --------------------------------------------------------------------------- the rules


@dataclass
class _Load:
    item: LoadItem
    kind: str                       # pressure | force | moment | acceleration
    value: Any                      # SI, with the factor of the load item
    groups: list[str]               # face groups of the mesh
    at: np.ndarray | None = None


@dataclass
class _Held:
    interface: str
    features: list[str]
    groups: list[str]
    effect: str | None = None


@dataclass
class Applied:
    """One load as it is applied on one mesh, with its resultant on that mesh."""

    kind: str                       # pressure | force | moment | acceleration | relief
    effect: str | None
    force: np.ndarray               # resultant force, N
    moment: np.ndarray              # resultant moment about the origin, N m
    size: float                     # integral of |traction| dA or |body force| dV, N
    info: dict = field(default_factory=dict)


@dataclass
class Realized:
    """The idealization on one mesh."""

    geom: MeshGeom
    bcs: list = field(default_factory=list)             # feacore.case.BC
    loads: list = field(default_factory=list)           # feacore.case.Load
    applied: list[Applied] = field(default_factory=list)
    held_groups: list[str] = field(default_factory=list)
    restraints: list[dict] = field(default_factory=list)   # free part: {"point", "node", "components"}
    support: dict = field(default_factory=dict)         # thermo support: centroid, mean temperature, ...
    balance: dict = field(default_factory=dict)         # free part: residual and scales
    checks: dict = field(default_factory=dict)          # key -> {"kind", "value", "limit", "statement", "passed"}

    @property
    def force(self) -> np.ndarray:
        return np.sum([a.force for a in self.applied], axis=0) if self.applied else np.zeros(3)

    @property
    def moment(self) -> np.ndarray:
        return np.sum([a.moment for a in self.applied], axis=0) if self.applied else np.zeros(3)

    def force_scale(self) -> float:
        """The largest load: the largest resultant of one load, at least 1 % of the largest |traction| integral."""
        loads = [a for a in self.applied if a.kind != "relief"]
        if not loads:
            return 0.0
        return max(max(float(np.linalg.norm(a.force)) for a in loads), 0.01 * max(a.size for a in loads))

    def moment_scale(self, about: np.ndarray) -> float:
        loads = [a for a in self.applied if a.kind != "relief"]
        if not loads:
            return 0.0
        m = max(float(np.linalg.norm(a.moment - np.cross(about, a.force))) for a in loads)
        return max(m, self.force_scale() * 0.5 * self.geom.diag)


class Idealizer:
    """The rules of one load case. The constructor reads the load case, writes the decisions and the
    assumptions to `journal`, and raises NotSupported when a load or a support has no rule."""

    def __init__(self, project, geo: dict, lc: LoadCase, journal: Journal, thermal: bool = False,
                 thermal_analysis: str | None = None):
        self.project, self.geo, self.lc, self.j = project, geo, lc, journal
        self.thermal = thermal
        self.thermal_analysis = thermal_analysis
        self.comp = project.components.get(lc.component) or project.subject
        mat = project.material_of(self.comp.id)
        self.material = mat
        self.E = float(units.si(mat.E.quantity()))
        self.nu = float(mat.nu)
        self.rho = float(units.si(mat.rho.quantity()))
        self.alpha = float(units.si(mat.alpha.quantity())) if mat.alpha is not None else None
        self.T_ref = float(units.si(units.to(self.comp.reference_temperature.quantity(), "K")))
        self.loads: list[_Load] = []
        self.held: list[_Held] = []
        self.thermal_effects: list[str] = []      # temperature effects of the load case (not mechanical loads)
        self.other_effects: list[str] = []        # effects with no structural action (mass flow)
        self.decisions: dict[str, str] = {}       # rule -> decision id
        self._parse()
        self._decide()

    # ------------------------------------------------------------------ reading the load case
    def _faces(self, features: list[str], interface: str | None, role: str) -> list[str]:
        feats = self.geo.get("features", {}) or {}
        out: list[str] = []
        for f in features:
            if f == "*":
                tags = ((self.geo.get("interfaces", {}) or {}).get(interface or "", {}) or {}).get(role, [])
                out += [f"f{t}" for t in tags]
            elif f in feats and feats[f].get("faces"):
                out += [f"f{t}" for t in feats[f]["faces"]]
            else:
                out.append(f)                  # by contract every feature id is a face group of the mesh
        return list(dict.fromkeys(out))

    def _parse(self) -> None:
        lc, problems = self.lc, []
        for it in lc.loads:
            k = it.kind
            if k == "support":
                continue
            if k in THERMAL_KINDS:
                self.thermal_effects.append(it.effect)
                continue
            if k in NO_STRUCTURAL_KINDS:
                self.other_effects.append(it.effect)
                continue
            if k not in ("pressure", "force", "moment", "acceleration"):
                problems.append(f"The structural engine has no rule for a load of the kind {k} (effect {it.effect}).")
                continue
            if it.si is None:
                problems.append(f"The load {it.effect} ({k}) has no value.")
                continue
            value = np.asarray(it.si, dtype=float) * float(it.factor)
            if (k == "pressure") != (value.ndim == 0) or (value.ndim == 1 and value.shape != (3,)):
                problems.append(f"The load {it.effect} ({k}) must be "
                                f"{'one number' if k == 'pressure' else 'a vector with three components'}.")
                continue
            groups: list[str] = []
            if k != "acceleration":
                groups = self._faces(list(it.features), it.interface, it.role)
                if not groups:
                    problems.append(f"The load {it.effect} ({k}) has no features: the interface {it.interface} "
                                    f"gives no feature for the role {it.role or '(none)'}.")
                    continue
            at = np.asarray(it.at_si, dtype=float) if it.at_si is not None and k == "force" else None
            self.loads.append(_Load(it, k, float(value) if k == "pressure" else value, groups, at))
        by_id = {i.id: i for i in self.comp.interfaces}
        support_items = {it.interface: it for it in lc.loads if it.kind == "support" and it.interface}
        for iid in dict.fromkeys([*lc.supports, *support_items]):
            itf = by_id.get(iid)
            if itf is None:
                problems.append(f"The support {iid} is not an interface of the component {self.comp.id}.")
                continue
            if itf.joint not in HELD_JOINTS:
                problems.append(f"The structural engine has no support model for the joint type {itf.joint} of the "
                                f"interface {iid}. It has a model for these joint types: "
                                f"{', '.join(j for j in HELD_JOINTS if j)}.")
                continue
            item = support_items.get(iid)
            feats = list(item.features) if item is not None and item.features else list(itf.features.get("seat", []))
            role = (item.role if item is not None and item.role else "seat")
            groups = self._faces(feats, iid, role)
            if not groups:
                problems.append(f"The support {iid} has no seat features.")
                continue
            self.held.append(_Held(iid, feats, groups, item.effect if item is not None else None))
        if self.thermal and self.alpha is None:
            problems.append(f"The material {self.material.id} has no coefficient of thermal expansion (alpha).")
        if not self.loads and not self.thermal:
            problems.append("The load case has no mechanical load.")
        if problems:
            raise NotSupported(" ".join(problems))

    @property
    def free(self) -> bool:
        return not self.held

    def groups(self) -> list[str]:
        """Every face group the idealization needs in the mesh."""
        out = [g for ld in self.loads for g in ld.groups] + [g for h in self.held for g in h.groups]
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------ decisions and assumptions
    def _decide(self) -> None:
        lc, j = self.lc, self.j
        by_kind: dict[str, list[_Load]] = {}
        for ld in self.loads:
            by_kind.setdefault(ld.kind, []).append(ld)

        def effects(kind):
            return [ld.item.effect for ld in by_kind.get(kind, [])]

        if "pressure" in by_kind:
            d = j.decide(
                subject=lc.id, key="idealize-pressure", title="Pressure as a surface pressure",
                choice="The engine applies each pressure as a uniform pressure normal to every face of its features.",
                rationale="A fluid at rest pushes normal to a wall with the same pressure at every point. "
                          "The system context gives one pressure value for each interface.",
                alternatives=[("Apply a pressure field from a flow analysis.",
                               "The load case gives one value for each interface and no field.")],
                basis="rule", links=effects("pressure"), by="rule:structural.pressure")
            self.decisions["pressure"] = d.id
        if "force" in by_kind:
            offset = any(ld.at is not None for ld in by_kind["force"])
            d = j.decide(
                subject=lc.id, key="idealize-force", title="Force as a uniform traction on the seat faces",
                choice="The engine spreads each force as a uniform traction over the faces of its seat features. "
                       "The resultant acts at the area centroid of these faces."
                       + (" A force with a point of application away from the centroid also gets the moment of "
                          "the offset." if offset else ""),
                rationale="The uniform traction has the same resultant as the force. Away from the seat faces the "
                          "stress depends on the resultant only (principle of Saint-Venant).",
                alternatives=[("Apply the force at the bolt holes.",
                               "The load case does not give the bolt forces. The stress near the seat faces is "
                               "approximate in the two models."),
                              ("Apply the force at one point.", "A point force gives a stress without limit at the point.")],
                basis="rule", links=effects("force"), by="rule:structural.force")
            self.decisions["force"] = d.id
        if "moment" in by_kind or any(ld.at is not None for ld in by_kind.get("force", [])):
            d = j.decide(
                subject=lc.id, key="idealize-moment", title="Moment as a linear traction field on the seat faces",
                choice="The engine applies each moment as the traction field t = w x (x - c) on the faces of its seat "
                       "features. The point c is the area centroid of the faces. The vector w is the moment divided "
                       "by the tensor of the second moments of area of the faces.",
                rationale="On a flat face this field is the linear normal stress of beam theory for the moment "
                          "components in the plane, with the product of inertia. For the component along the normal "
                          "it is a shear stress in proportion to the distance from the centroid. The net force of the "
                          "field is zero. The engine integrates the field on each mesh and checks its moment.",
                alternatives=[("Apply the moment as two opposite point forces.",
                               "Point forces give a stress without limit at the points."),
                              ("Connect the face to a rigid body and load the body.",
                               "The fixed solver has no rigid body elements, and a rigid face is stiffer than the "
                               "real joint.")],
                basis="calculation",
                links=[*effects("moment"), *[ld.item.effect for ld in by_kind.get("force", []) if ld.at is not None]],
                by="rule:structural.moment")
            self.decisions["moment"] = d.id
        if "acceleration" in by_kind:
            d = j.decide(
                subject=lc.id, key="idealize-acceleration", title="Acceleration as a body force",
                choice="The engine applies each acceleration as a body force on the full part. The body force is the "
                       "density times the acceleration vector.",
                rationale="The system context gives the vector as the direction of the inertial load on the part, as "
                          f"for gravity. The density of {self.material.id} is {fmt(self.rho, 'kg/m^3')}.",
                alternatives=[("Make a dynamic analysis with the vibration spectrum.",
                               "The system context gives quasi-static accelerations.")],
                basis="rule", links=effects("acceleration"), by="rule:structural.acceleration")
            self.decisions["acceleration"] = d.id
        if self.loads and any(ld.kind in ("force", "moment") for ld in self.loads):
            j.assume(
                subject=lc.id, key="load-path", title="Forces and moments spread over the full seat faces",
                statement="Each force and each moment acts as a smooth traction on the full seat face of its interface.",
                rationale="The traction has the correct resultant. The real joint puts the load in at the bolts and "
                          "the gasket, and the system context does not give these local forces.",
                impact="The stress next to a loaded seat face is approximate. The stress at a distance of more than "
                       "the width of the seat face does not change.",
                verify_by="Compare the governing stress location with the loaded seat faces.",
                conservative=None, links=[ld.item.effect for ld in self.loads if ld.kind in ("force", "moment")],
                by="rule:structural.load_path")
        held = [h.interface for h in self.held]
        per_kelvin = self.E * (self.alpha or 0.0) / (1.0 - self.nu)
        if self.held and not self.thermal:
            d = j.decide(
                subject=lc.id, key="idealize-support", title="Held faces as fixed supports",
                choice=f"The engine holds the seat faces of {names(held)} in all directions.",
                rationale="The system context says that these connections hold the part. The engine takes each "
                          "neighbour as rigid, which is the usual model of a bolted joint on a stiff neighbour.",
                alternatives=[("Model the neighbour and the bolts.", "The project has no geometry of the neighbour."),
                              ("Hold only the direction normal to the face.",
                               "The bolts and the friction also stop the motion in the plane of the face.")],
                basis="engineering_judgement", needs_review=True, links=[*held, *[h.effect for h in self.held if h.effect]],
                by="rule:structural.support")
            self.decisions["support"] = d.id
            j.assume(
                subject=lc.id, key="rigid-neighbour", title="The neighbour of a held face is rigid",
                statement=f"The neighbour of {names(held)} does not deform and does not move.",
                rationale="The system context gives a support at these interfaces and no stiffness of the neighbour.",
                impact="A flexible neighbour lets the held face deform. The stress near the held face then becomes "
                       "lower and the displacement of the part becomes larger.",
                verify_by="Compare the stiffness of the neighbour flange with the stiffness of the part at the held face.",
                conservative=None, links=held, by="rule:structural.support")
        if self.held and self.thermal:
            d = j.decide(
                subject=lc.id, key="idealize-support",
                title="Held faces follow the thermal contraction of the neighbour",
                choice=f"The engine prescribes the displacement u = alpha (T_face - T_ref) (x - c) on the seat faces of "
                       f"{names(held)}. T_face is the mean temperature of the held faces from the thermal analysis. "
                       "The point c is the area centroid of the held faces. For one flat face this is zero "
                       "displacement normal to the face and the free thermal contraction in its plane.",
                rationale="The neighbour cools and contracts together with the part. A face that is fully fixed cannot "
                          f"contract. For this material that gives a false stress of approximately "
                          f"{fmt(per_kelvin, 'Pa')} for each kelvin of temperature change (E alpha / (1 - nu)).",
                alternatives=[("Hold the face fully fixed.",
                               f"The false thermal stress is approximately {fmt(per_kelvin, 'Pa')} for each kelvin."),
                              ("Let the face slide in its plane and stop the rigid-body motion with point restraints.",
                               "Point restraints give a stress without limit at the points, and a bolted joint does "
                               "not slide freely.")],
                basis="engineering_judgement", needs_review=True, links=[*held, *[h.effect for h in self.held if h.effect]],
                by="rule:structural.thermal_support")
            self.decisions["support"] = d.id
            j.assume(
                subject=lc.id, key="neighbour-expansion",
                title="The neighbour has the expansion and the temperature of the held face",
                statement=f"The neighbour of {names(held)} has the same coefficient of thermal expansion and the same "
                          "temperature as the held face. The neighbour is rigid for all other loads."
                          + (" All held interfaces are on one neighbour body with one temperature." if len(held) > 1 else ""),
                rationale="The system context gives no material and no temperature of the neighbour. Two flanges in "
                          "contact have almost the same temperature at the joint.",
                impact=f"If the neighbour contracts more or less than the held face, the held face gets a thermal "
                       f"stress of approximately {fmt(per_kelvin, 'Pa')} for each kelvin of difference. "
                       "This stress is not in the results.",
                verify_by="Compare the material and the temperature of the neighbour flange with those of the held face.",
                conservative=False, links=held, by="rule:structural.thermal_support")
        if not self.held:
            d = j.decide(
                subject=lc.id, key="idealize-free", title="Free part with a rigid-body restraint at three nodes",
                choice="The engine holds three nodes in six displacement components to stop the rigid-body motion "
                       "(three, two and one components). It balances the small load residual of each mesh with a "
                       "rigid-body acceleration field on the volume.",
                rationale="The load case has no support, and the engine checks that its loads balance. Six components "
                          "at three nodes far apart stop the six rigid-body motions and no deformation. With the "
                          "acceleration field the three nodes carry no force.",
                alternatives=[("Hold one face.", "A held face of a free part gives a false stress at the face."),
                              ("Hold the part with soft springs.",
                               "Springs change the stress, and their stiffness has no physical basis.")],
                basis="rule", links=[ld.item.effect for ld in self.loads], by="rule:structural.free_part")
            self.decisions["free"] = d.id
        if self.thermal:
            src = self.thermal_analysis or "the thermal analysis"
            d = j.decide(
                subject=lc.id, key="idealize-temperature", title="Temperature field from the thermal analysis",
                choice=f"The engine takes the temperature of {src} on the same mesh and applies the thermal strain "
                       f"alpha (T - T_ref). The reference temperature T_ref is {fmt(self.T_ref, 'K')}.",
                rationale="The same mesh gives the temperature at each node without interpolation between meshes. "
                          "The component is free of stress at its reference temperature.",
                alternatives=[("Apply one uniform temperature.",
                               "The part has temperature gradients, and the gradients cause thermal stress.")],
                basis="rule", links=[*(([self.thermal_analysis]) if self.thermal_analysis else []), *self.thermal_effects],
                by="rule:structural.temperature")
            self.decisions["temperature"] = d.id
            j.assume(
                subject=lc.id, key="properties-temperature",
                title="The material properties do not change with the temperature",
                statement=f"E, nu and alpha of {self.material.id} are constant in the temperature range of the load case.",
                rationale="The material record gives one value for each property. The value of alpha is a mean value "
                          "for the service range.",
                impact="If E or alpha changes with the temperature, the thermal stress changes in the same proportion.",
                verify_by="Compare the properties at the lowest temperature of the load case with the material record.",
                conservative=None, links=[self.material.id], by="rule:structural.temperature")
        j.assume(
            subject=lc.id, key="linear-elastic", title="Linear elastic material and small displacements",
            statement="The material is linear elastic, and the displacements are small against the dimensions of the part.",
            rationale="The fixed solver is linear. The requirements give stress limits at or below the yield strength.",
            impact="Where the stress is above the yield strength, the real stress is lower and the real strain is "
                   "higher than the result. A linear result above the yield strength is not a real stress.",
            verify_by="Compare the largest von Mises stress outside singular locations with the yield strength.",
            conservative=None, links=[self.material.id], by="rule:structural.linear")

    # ------------------------------------------------------------------ the numbers on one mesh
    def realize(self, mg: MeshGeom, T: np.ndarray | None = None, field_path: str | None = None) -> Realized:
        """The boundary conditions and loads on the mesh `mg`. `T` is the temperature at the nodes of
        `mg` (K) and `field_path` the file the solver reads it from, for a thermo-structural case."""
        from feacore.case import BC, Load

        missing = mg.missing(self.groups())
        if missing:
            raise NotSupported(f"The mesh has no face group for {names(missing)}. The geometry stage must give "
                               "each feature of a load or a support as a group of the mesh.")
        real = Realized(mg)
        vol, c_v, J_v = mg.volume()
        for ld in self.loads:
            eff = ld.item.effect
            if ld.kind == "pressure":
                p = float(ld.value)
                q = mg.quad(ld.groups)
                real.bcs += [BC(type="pressure", group=g, value=p) for g in ld.groups]
                real.applied.append(Applied("pressure", eff, -p * (q.w @ q.n), -p * (q.w @ np.cross(q.x, q.n)),
                                            abs(p) * q.area, {"area": q.area}))
            elif ld.kind == "force":
                F = np.asarray(ld.value, dtype=float)
                q = mg.quad(ld.groups)
                c = q.centroid
                for g in ld.groups:
                    real.bcs.append(BC(type="force", group=g, value=(F * mg.quad([g]).area / q.area).tolist()))
                real.applied.append(Applied("force", eff, F.copy(), np.cross(c, F), float(np.linalg.norm(F)),
                                            {"area": q.area, "centroid": c}))
                if ld.at is not None and np.linalg.norm(ld.at - c) > 1e-6 * mg.diag:
                    self._moment(real, ld, np.cross(ld.at - c, F), q, offset=True)
            elif ld.kind == "moment":
                self._moment(real, ld, np.asarray(ld.value, dtype=float), mg.quad(ld.groups))
            elif ld.kind == "acceleration":
                b = self.rho * np.asarray(ld.value, dtype=float)
                real.loads.append(Load(type="body_force", value=b.tolist(), group="*"))
                real.applied.append(Applied("acceleration", eff, b * vol, np.cross(c_v, b * vol),
                                            float(np.linalg.norm(b)) * vol, {"volume": vol}))
        if self.thermal:
            if field_path is None or T is None:
                raise ValueError("a thermo-structural case needs the temperature at the nodes and its file")
            real.loads.append(Load(type="temperature", value={"field": str(field_path)}, group="*"))
        if self.held:
            self._support(real, T)
        else:
            self._free(real, vol, c_v, J_v)
        return real

    def _moment(self, real: Realized, ld: _Load, M: np.ndarray, q: SurfQuad, offset: bool = False) -> None:
        from feacore.case import BC
        c = q.centroid
        J = q.inertia(c)
        if np.linalg.cond(J) > 1e12:
            raise NotSupported(f"The faces of the load {ld.item.effect} have no area moment in one direction, so they "
                               "cannot carry its moment.")
        w = np.linalg.solve(J, M)
        exprs = rigid_exprs(np.zeros(3), w, c)
        real.bcs += [BC(type="traction", group=g, value=list(exprs)) for g in ld.groups]
        t = eval_exprs(exprs, q.x)                       # what the solver evaluates at its quadrature points
        force = q.w @ t
        moment_c = q.w @ np.cross(q.x - c, t)
        size = float(q.w @ np.linalg.norm(t, axis=1))
        err = float(np.linalg.norm(moment_c - M) / max(np.linalg.norm(M), 1e-300))
        ferr = float(np.linalg.norm(force) / max(size, 1e-300))
        eff = ld.item.effect
        real.applied.append(Applied("moment", eff, force, moment_c + np.cross(c, force), size,
                                    {"centroid": c, "omega": w, "exprs": exprs, "moment": M, "offset": offset,
                                     "planar": q.planar(), "peak_traction": float(np.linalg.norm(t, axis=1).max())}))
        what = "the offset of the force" if offset else "the moment"
        real.checks[f"moment-{eff}{'-offset' if offset else ''}"] = {
            "kind": "load_resultant", "value": max(err, ferr), "limit": MOMENT_TOL,
            "passed": bool(err <= MOMENT_TOL and ferr <= MOMENT_TOL),
            "statement": f"The traction field for {what} of {eff} gives a moment of {fmt(moment_c, 'N*m')} and a net "
                         f"force of {fmt(force, 'N')} on the mesh. The load is {fmt(M, 'N*m')}. "
                         f"The largest traction is {fmt(np.linalg.norm(t, axis=1).max(), 'Pa')}."}

    def _support(self, real: Realized, T: np.ndarray | None) -> None:
        from feacore.case import BC
        mg = real.geom
        groups = list(dict.fromkeys(g for h in self.held for g in h.groups))
        real.held_groups = groups
        if not self.thermal:
            real.bcs += [BC(type="fixed", group=g) for g in groups]
            return
        q = mg.quad(groups)
        c = q.centroid
        T_face = q.mean(T)
        k = self.alpha * (T_face - self.T_ref)
        exprs = [f"{_num(k)}*({v} - {_num(ci)})" for v, ci in zip("xyz", c)]        # u = k (x - c)
        real.bcs += [BC(type="displacement", group=g, components=[0, 1, 2], value=list(exprs)) for g in groups]
        per = {h.interface: mg.quad(h.groups).mean(T) for h in self.held}
        nodes = mg.nodes(groups)
        real.support = {"centroid": c, "T_face": T_face, "strain": k, "exprs": exprs, "per_interface": per,
                        "T_min": float(T[nodes].min()), "T_max": float(T[nodes].max()), "planar": q.planar()}
        delta = float(units.si(units.to(self.project.policy.thermal_delta.quantity(), "K")))
        if len(per) > 1:
            spread = max(per.values()) - min(per.values())
            real.checks["support-temperature"] = {
                "kind": "support_temperature", "value": spread, "limit": delta, "passed": bool(spread <= delta),
                "statement": f"The mean temperatures of the held interfaces differ by {fmt(spread, 'K')}. The support "
                             f"model takes one temperature for all held faces. The limit of the policy is {fmt(delta, 'K')}."}

    def _free(self, real: Realized, vol: float, c_v: np.ndarray, J_v: np.ndarray) -> None:
        from feacore.case import BC, Load
        mg = real.geom
        F, M_c = real.force, real.moment - np.cross(c_v, real.force)
        fs, ms = real.force_scale(), real.moment_scale(c_v)
        rel_f = float(np.linalg.norm(F) / fs) if fs > 0 else 0.0
        rel_m = float(np.linalg.norm(M_c) / ms) if ms > 0 else 0.0
        real.balance = {"force": F, "moment": M_c, "force_scale": fs, "moment_scale": ms, "rel_force": rel_f,
                        "rel_moment": rel_m, "about": c_v}
        statement = (f"The applied loads of the free part give a net force of {fmt(F, 'N')} and a net moment of "
                     f"{fmt(M_c, 'N*m')} about the centre of volume. The largest load is {fmt(fs, 'N')}. "
                     f"The residual is {100 * max(rel_f, rel_m):.2g} % of the largest load.")
        real.checks["balance"] = {"kind": "load_balance", "value": max(rel_f, rel_m), "limit": BALANCE_TOL,
                                  "passed": bool(max(rel_f, rel_m) <= BALANCE_TOL), "statement": statement}
        if max(rel_f, rel_m) > BALANCE_TOL:
            err = Unbalanced("The loads of a free part must balance. " + statement
                             + f" The limit is {100 * BALANCE_TOL:g} %. Add the missing load or a support to the phase.")
            err.check = real.checks["balance"]
            raise err
        if self.loads and (np.linalg.norm(F) > 0 or np.linalg.norm(M_c) > 0):
            a, w = -F / vol, -np.linalg.solve(J_v, M_c)
            exprs = rigid_exprs(a, w, c_v)
            real.loads.append(Load(type="body_force", value=list(exprs), group="*"))
            real.applied.append(Applied("relief", None, -F, -(M_c + np.cross(c_v, F)), 0.0,
                                        {"exprs": exprs, "acceleration": a, "omega": w}))
        for point, node, comps in restraint_points(mg.x):
            real.bcs.append(BC(type="displacement", point=[float(v) for v in point], components=list(comps), value=0.0))
            real.restraints.append({"point": point, "node": node, "components": list(comps)})

    # ------------------------------------------------------------------ the record
    def record_items(self, real: Realized) -> list[BCItem]:
        """The supports and loads of the analysis record, with the numbers of the mesh `real`."""
        out: list[BCItem] = []
        applied = {(a.kind, a.effect, bool(a.info.get("offset"))): a for a in real.applied}

        def faces_and_features(features, groups):
            return [f for f in features if f != "*"], [g for g in groups if g.startswith("f") and g[1:].isdigit()]

        for ld in self.loads:
            it = ld.item
            fe, fa = faces_and_features(list(it.features), ld.groups)
            where = f"the features {names(fe)}" if fe else f"the faces {names(fa)}"
            if ld.kind == "pressure":
                a = applied[("pressure", it.effect, False)]
                out.append(BCItem(type="pressure", features=fe, faces=fa, value=float(ld.value), effect=it.effect,
                                  interface=it.interface, decision=self.decisions.get("pressure"),
                                  description=f"A pressure of {fmt(ld.value, 'Pa')} acts on {where}. "
                                              f"The area is {fmt(a.info['area'], 'm^2')}."))
            elif ld.kind == "force":
                a = applied[("force", it.effect, False)]
                out.append(BCItem(type="force", features=fe, faces=fa, value=np.asarray(ld.value).tolist(),
                                  effect=it.effect, interface=it.interface, decision=self.decisions.get("force"),
                                  description=f"A force of {fmt(ld.value, 'N')} acts as a uniform traction on {where}. "
                                              f"The area is {fmt(a.info['area'], 'm^2')}."))
                off = applied.get(("moment", it.effect, True))
                if off is not None:
                    out.append(self._moment_item(it, fe, fa, off, where, offset=True))
            elif ld.kind == "moment":
                out.append(self._moment_item(it, fe, fa, applied[("moment", it.effect, False)], where))
            elif ld.kind == "acceleration":
                b = self.rho * np.asarray(ld.value, dtype=float)
                out.append(BCItem(type="body_force", value=b.tolist(), effect=it.effect, interface=it.interface,
                                  decision=self.decisions.get("acceleration"),
                                  description=f"An acceleration of {fmt(ld.value, 'm/s^2')} acts on the full part as a "
                                              f"body force of density times acceleration."))
        if self.thermal:
            out.append(BCItem(type="temperature", value={"T_ref": self.T_ref, "alpha": self.alpha,
                                                         "source": self.thermal_analysis},
                              decision=self.decisions.get("temperature"),
                              description=f"The temperature field of {self.thermal_analysis or 'the thermal analysis'} "
                                          f"gives the thermal strain. The reference temperature is {fmt(self.T_ref, 'K')}."))
        for h in self.held:
            fe, fa = faces_and_features(h.features, h.groups)
            if not self.thermal:
                out.append(BCItem(type="fixed", features=fe, faces=fa, value=0.0, effect=h.effect, interface=h.interface,
                                  decision=self.decisions.get("support"),
                                  description=f"The seat faces of {h.interface} are fixed in all directions."))
            else:
                s = real.support
                out.append(BCItem(type="displacement", features=fe, faces=fa, effect=h.effect, interface=h.interface,
                                  decision=self.decisions.get("support"),
                                  value={"components": [0, 1, 2], "expressions": s["exprs"],
                                         "centroid": np.asarray(s["centroid"]).tolist(), "mean_temperature": s["T_face"],
                                         "thermal_strain": s["strain"]},
                                  description=f"The seat faces of {h.interface} follow the free thermal contraction about "
                                              f"the centroid of the held faces. The mean temperature of the held faces is "
                                              f"{fmt(s['T_face'], 'K')}, and the thermal strain is {s['strain']:.3e}."))
        if not self.held:
            pts = [{"point": np.asarray(r["point"]).tolist(), "components": r["components"]} for r in real.restraints]
            out.append(BCItem(type="displacement", value={"points": pts}, decision=self.decisions.get("free"),
                              description="Three nodes stop the rigid-body motion with three, two and one displacement "
                                          "components. The first node is at "
                                          f"{fmt(real.restraints[0]['point'], 'm')}."))
            rel = next((a for a in real.applied if a.kind == "relief"), None)
            if rel is not None:
                b = real.balance
                out.append(BCItem(type="body_force", value={"expressions": rel.info["exprs"]},
                                  decision=self.decisions.get("free"),
                                  description="A rigid-body acceleration field balances the load residual of the mesh. "
                                              f"The residual force is {fmt(b['force'], 'N')}, which is "
                                              f"{100 * max(b['rel_force'], b['rel_moment']):.2g} % of the largest load."))
        return out

    def _moment_item(self, it: LoadItem, fe, fa, a: Applied, where: str, offset: bool = False) -> BCItem:
        M = a.info["moment"]
        what = "The offset of the force gives a moment" if offset else "A moment"
        return BCItem(type="traction", features=fe, faces=fa, effect=it.effect, interface=it.interface,
                      decision=self.decisions.get("moment"),
                      value={"moment": np.asarray(M).tolist(), "centroid": np.asarray(a.info["centroid"]).tolist(),
                             "omega": np.asarray(a.info["omega"]).tolist(), "expressions": a.info["exprs"]},
                      description=f"{what} of {fmt(M, 'N*m')}. It acts as a linear traction field on {where}. "
                                  f"The largest traction is {fmt(a.info['peak_traction'], 'Pa')}.")


def restraint_points(x: np.ndarray) -> list[tuple[np.ndarray, int, tuple[int, ...]]]:
    """A 3-2-1 restraint on nodes `x` (n, 3): [(point, node index, components)].

    Node 1 and node 2 are the two ends of the part along its longest bounding-box axis a. Node 1 is
    held in x, y and z. Node 2 is held in the two axes other than a, which stops the two rotations
    about axes across the line 1-2. Node 3 is the node farthest from the line 1-2; it is held in the
    global component along which a rotation about the line 1-2 moves it most."""
    x = np.asarray(x, dtype=float)
    a = int(np.argmax(np.ptp(x, axis=0)))
    i1, i2 = int(np.argmin(x[:, a])), int(np.argmax(x[:, a]))
    d = x[i2] - x[i1]
    dn = d / np.linalg.norm(d)
    r = x - x[i1]
    perp = r - np.outer(r @ dn, dn)
    i3 = int(np.argmax(np.einsum("ij,ij->i", perp, perp)))
    k = int(np.argmax(np.abs(np.cross(d, x[i3] - x[i1]))))
    others = tuple(i for i in range(3) if i != a)
    out = [(x[i1].copy(), i1, (0, 1, 2)), (x[i2].copy(), i2, others), (x[i3].copy(), i3, (k,))]
    # the six constraints must stop the six rigid-body motions: rank of the constraint matrix
    rows = []
    c = x.mean(axis=0)
    scale = max(float(np.linalg.norm(np.ptp(x, axis=0))), 1e-300)
    for p, _, comps in out:
        y = (p - c) / scale
        for comp in comps:
            e = np.zeros(3)
            e[comp] = 1.0
            rows.append(np.concatenate([e, np.cross(y, e)]))       # u_comp = e . (t + w x y) = e.t + (y x e).w
    if np.linalg.matrix_rank(np.array(rows), tol=1e-8) < 6:
        raise NotSupported("The engine found no three nodes that stop the rigid-body motion of the part.")
    return out
