"""Shared helpers for the incompressible-flow (CFD) benchmark problems (``feacore.problems.cfd``).

* ``FlowProblem``: a ``Problem`` with physics "flow". Reference meshes are linear (finite volumes
  use straight-sided cells, so ``geom_order`` is ignored) and block-structured: every block is a
  transfinite, recombined (quad / hex) patch whose cell counts scale with 1/h, so a mesh sequence is
  geometrically similar (graded edges keep their first-to-last cell-size ratio fixed). 2D meshes
  lie in the xy-plane with counter-clockwise cells; the backend extrudes them one layer of
  ``analysis.thickness`` (the problems use thickness 1, so forces and flow rates are per metre of
  depth).
* ``TransfiniteBlocks``: small builder for those block meshes (shared points / lines, one call per
  block, edge node counts and gradings derived from a refinement factor).
* ``num`` formats a float for the restricted expression language with full precision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from feacore.case import BC, Analysis, Case, Material, QoI, Region
from feacore.geometry.gmsh_tools import add_group, ensure_domain_group, gmsh_session
from feacore.problems.base import Problem, Reference

# Citations used by several problems.
WHITE = "F.M. White, Viscous Fluid Flow, 3rd ed., McGraw-Hill (2006)"
SCHAEFER_TUREK = (
    "M. Schaefer & S. Turek, 'Benchmark computations of laminar flow around a cylinder', in E.H. "
    "Hirschel (ed.), Flow Simulation with High-Performance Computers II, Notes on Numerical Fluid "
    "Mechanics 52, Vieweg (1996) 547-566")
FEATFLOW_2D1 = ("https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/"
                "dfg_benchmark1_re20.html")
GHIA = ("U. Ghia, K.N. Ghia & C.T. Shin, 'High-Re solutions for incompressible flow using the Navier-Stokes "
        "equations and a multigrid method', J. Comput. Phys. 48 (1982) 387-411")
GHIA_COPY = "https://github.com/pypr/pysph/blob/master/pysph/examples/ghia_cavity_data.py"


def num(v: float) -> str:
    """Float literal for expressions (full double precision, never locale dependent)."""
    return repr(float(v))


def refinement_factor(h0: float, h: float) -> float:
    """k = h0 / h, the factor every cell count of the h0 mesh is multiplied by."""
    if h <= 0:
        raise ValueError(f"mesh size must be > 0, got {h}")
    return h0 / h


def count(n0: int, k: float) -> int:
    """Cells on an edge that has n0 cells at k = 1 (at least one)."""
    return max(1, int(round(n0 * k)))


def progression(n: int, ratio: float) -> float:
    """gmsh "Progression" coefficient for n cells whose last/first cell-size ratio is ``ratio``.
    Keeping the ratio fixed while n grows keeps the graded edge geometrically similar."""
    return 1.0 if n <= 1 or ratio == 1.0 else ratio ** (1.0 / (n - 1))


# --------------------------------------------------------------------------- block meshes


class TransfiniteBlocks:
    """Build an OCC face set from 4-sided blocks that share points and edges, then make every
    block a transfinite, recombined patch.

    Points are deduplicated by coordinates, straight edges by their end points, so two blocks that
    list the same edge share it (conformal mesh). Cell counts and gradings are set afterwards with
    ``set_edge`` on the (signed) curve tags; ``coords`` maps point tags to coordinates."""

    def __init__(self, tol: float = 1e-12):
        import gmsh

        self.occ = gmsh.model.occ
        self.tol = tol
        self.points: dict[tuple, int] = {}
        self.coords: dict[int, tuple[float, float, float]] = {}
        self.lines: dict[tuple[int, int], int] = {}
        self.arcs: dict[tuple[int, int], int] = {}
        self.blocks: list[int] = []
        self._centres: list[int] = []

    def point(self, x: float, y: float, z: float = 0.0) -> int:
        key = (round(x / self.tol), round(y / self.tol), round(z / self.tol))
        if key not in self.points:
            self.points[key] = self.occ.addPoint(x, y, z)
            self.coords[self.points[key]] = (x, y, z)
        return self.points[key]

    def line(self, a: tuple, b: tuple) -> int:
        """Signed tag of the straight edge from point a to point b."""
        pa, pb = self.point(*a), self.point(*b)
        if (pb, pa) in self.lines:
            return -self.lines[(pb, pa)]
        if (pa, pb) not in self.lines:
            self.lines[(pa, pb)] = self.occ.addLine(pa, pb)
        return self.lines[(pa, pb)]

    def arc(self, a: tuple, centre: tuple, b: tuple) -> int:
        """Signed tag of the circular arc (< 180 deg) from a to b about centre."""
        pa, pb = self.point(*a), self.point(*b)
        if (pb, pa) in self.arcs:
            return -self.arcs[(pb, pa)]
        if (pa, pb) not in self.arcs:
            c = self.occ.addPoint(*centre)
            self._centres.append(c)
            self.arcs[(pa, pb)] = self.occ.addCircleArc(pa, c, pb)
        return self.arcs[(pa, pb)]

    def block(self, curves: list[int]) -> int:
        """Plane face bounded by 4 signed curve tags forming a closed loop."""
        s = self.occ.addPlaneSurface([self.occ.addCurveLoop(curves)])
        self.blocks.append(s)
        return s

    def finish(self) -> None:
        self.occ.remove([(0, c) for c in self._centres])
        self.occ.synchronize()


def set_edge(curve: int, n: int, ratio: float = 1.0) -> int:
    """n cells on |curve|; cell size grows by ``ratio`` (last / first) along the signed direction.
    Returns the unsigned curve tag."""
    import gmsh

    tag = abs(int(curve))
    coef = progression(n, ratio)
    if curve < 0:
        coef = 1.0 / coef
    gmsh.model.mesh.setTransfiniteCurve(tag, n + 1, "Progression", coef)
    return tag


def transfinite_all(dim: int = 2) -> None:
    """Every surface (and volume) transfinite + recombined; call after all edges are set."""
    import gmsh

    for _, s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    if dim == 3:
        for _, v in gmsh.model.getEntities(3):
            gmsh.model.mesh.setTransfiniteVolume(v)
            gmsh.model.mesh.setRecombine(3, v)


def orient_ccw_2d() -> int:
    """Reverse the elements of every surface whose cells are clockwise in the xy-plane (so all
    2D cells have a +z normal). One element decides per surface (a surface mesh is consistently
    oriented). Returns the number of surfaces reversed."""
    import gmsh

    flipped = []
    for _, s in gmsh.model.getEntities(2):
        types, _, nodes = gmsh.model.mesh.getElements(2, s)
        if not len(types):
            continue
        props = gmsh.model.mesh.getElementProperties(types[0])
        n_nodes, n_corners = props[3], props[5]
        corners = [int(v) for v in nodes[0][:n_nodes][:n_corners]]
        xy = np.array([gmsh.model.mesh.getNode(v)[0][:2] for v in corners])
        area = sum(xy[i, 0] * xy[(i + 1) % n_corners, 1] - xy[(i + 1) % n_corners, 0] * xy[i, 1]
                   for i in range(n_corners))
        if area < 0:
            flipped.append((2, s))
    if flipped:
        gmsh.model.mesh.reverse(flipped)
    return len(flipped)


# --------------------------------------------------------------------------- problem base


class FlowProblem(Problem):
    """Base class of the CFD problems: physics "flow", linear block-structured reference meshes."""

    physics = "flow"
    formulation = "2d"
    dim = 2
    units = "SI (m, s, kg, Pa, N)"
    expected_behavior = "smooth"
    material_name = "fluid"
    thickness = 1.0                 # 2D extrusion depth: forces / flow rates are per metre of depth

    # solver controls written into analysis.options (iteration error << discretization error)
    max_iterations = 20000
    residual_tol = 1e-8

    def material(self) -> Material:
        p = self.params
        return Material(rho=p["rho"], mu=p["mu"])

    def options(self) -> dict:
        return {"turbulence": "laminar", "max_iterations": self.max_iterations,
                "residual_tol": self.residual_tol}

    def make_case(self, mesh_path, order: int, bcs: list[BC]) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="flow", formulation=self.formulation, order=order,
                              thickness=self.thickness, options=self.options()),
            materials={self.material_name: self.material()},
            bcs=bcs,
            regions=[Region(group="fluid", material=self.material_name)],
            qois=self.qois(),
            meta={"problem": self.id, "setup": "reference"},
        )

    # Finite volumes: straight-sided linear cells, so geom_order is ignored (a second-order gmsh
    # mesh would only add midside nodes the solver drops). Cells are made counter-clockwise.
    def mesh(self, path: str | Path, h: float, geom_order: int = 1, refine_steps: int = 0) -> Path:
        import gmsh

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gmsh_session(self.id):
            self.build_geometry()
            ensure_domain_group(self.dim)
            self.apply_sizing(h)
            gmsh.model.mesh.generate(self.dim)
            for _ in range(refine_steps):
                gmsh.model.mesh.refine()
            if self.dim == 2:
                orient_ccw_2d()
            gmsh.write(str(path))
        return path

    def apply_sizing(self, h: float) -> None:  # every CFD problem sets its own block counts
        raise NotImplementedError

    def build_cad(self) -> None:
        """Agent-facing CAD (one face / solid, no meshing blocks). Default: ``build_geometry``."""
        self.build_geometry()

    def export_step(self, path: str | Path) -> Path:
        import gmsh

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gmsh_session(self.id):
            self.build_cad()
            gmsh.write(str(path))
        return path


def boundary_curves_where(predicate) -> list[int]:
    """Tags of curves whose bounding box (xmin, ymin, zmin, xmax, ymax, zmax) satisfies predicate."""
    import gmsh

    return [t for _, t in gmsh.model.getEntities(1) if predicate(*gmsh.model.getBoundingBox(1, t))]


def near(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def group(dim: int, tags, name: str) -> int:
    return add_group(dim, tags, name)


__all__ = [
    "WHITE", "SCHAEFER_TUREK", "FEATFLOW_2D1", "GHIA", "GHIA_COPY", "num", "refinement_factor", "count",
    "progression", "TransfiniteBlocks", "set_edge", "transfinite_all", "orient_ccw_2d", "FlowProblem",
    "boundary_curves_where", "near", "group", "BC", "QoI", "Reference",
]
