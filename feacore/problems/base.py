"""Benchmark / test-part problem definition.

A ``Problem`` packages one analysis task:
  * an agent-facing statement (``description`` + ``statement()``, CAD via ``export_step``),
  * a *reference setup* (``build_geometry`` + ``apply_sizing`` + ``case``) that the fixed
    solver runs to validate itself and the problem,
  * reference values (``references``) and expected convergence behaviour, kept private
    from agents.

Subclasses set the class attributes and implement ``description``, ``build_geometry``,
``qois`` and ``case``. Register instances with ``feacore.problems.registry.register``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from feacore.case import Case, QoI, SCHEMA_VERSION
from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session, set_uniform_size


@dataclass
class Reference:
    qoi: str                 # QoI name the reference applies to
    value: float
    kind: str                # analytical | published | handbook | manufactured | converged_numerical | published_numerical
    source: str              # citation: document + equation / table / page; URL if public
    tolerance: float = 0.02  # relative tolerance for "agrees with reference"
    notes: str = ""


class Problem(ABC):
    id: str = ""
    title: str = ""
    family: str = ""                 # parametric family; variants share it
    tags: tuple[str, ...] = ()       # e.g. ("2d", "plane_stress", "stress_concentration", "nafems")
    dim: int = 2                     # mesh dimension
    physics: str = "elasticity"
    formulation: str = "plane_stress"
    units: str = "SI (m, N, Pa, K)"
    expected_behavior: str = "smooth"  # smooth | stress_concentration | singular | thermal | ...
    default_params: dict = {}

    def __init__(self, id: str | None = None, **params):
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown params {sorted(unknown)}")
        self.params = {**self.default_params, **params}
        if id is not None:
            self.id = id
        if not self.id:
            raise ValueError(f"{type(self).__name__} needs an id")

    # ------------------------------------------------------------- to implement
    @abstractmethod
    def description(self) -> str:
        """Agent-facing problem statement: geometry, material, loads, supports and the
        QoIs to report, in words, with units. Never include reference answers."""

    @abstractmethod
    def build_geometry(self) -> None:
        """Create the OCC geometry in the current gmsh model, synchronize, and add *named*
        physical groups for every boundary / region the case references."""

    @abstractmethod
    def qois(self) -> list[QoI]:
        """Quantities agents must report (also evaluated in validation)."""

    @abstractmethod
    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        """Reference analysis setup on a mesh produced by ``mesh``; must include ``self.qois()``."""

    # ------------------------------------------------------------- optional overrides
    def references(self) -> list[Reference]:
        return []

    def expectations(self) -> dict[str, str]:
        """Per-QoI expected convergence: "converge" (default) or "diverge" (singular QoI)."""
        return {}

    def expected_order(self, qoi: QoI, order: int) -> float | None:
        """Theoretical convergence order in h for a QoI; None -> default from field type."""
        return None

    def exact(self, field: str, points: np.ndarray) -> np.ndarray | None:
        """Exact field values at points (analytical problems only)."""
        return None

    def apply_sizing(self, h: float) -> None:
        """Mesh-size controls for characteristic size h. Default uniform. Overrides must scale
        every size with h so a sequence of h gives geometrically similar refinement."""
        set_uniform_size(h)

    def study_sizes(self, order: int = 2) -> list[float]:
        """Mesh sizes (coarse -> fine) for the reference convergence study."""
        h0 = self.params.get("h0")
        if h0 is None:
            raise NotImplementedError(f"{self.id}: set params['h0'] or override study_sizes")
        r = 2.0 if self.dim == 2 else 1.5
        return [h0, h0 / r, h0 / r**2]

    # ------------------------------------------------------------- provided
    def mesh(self, path: str | Path, h: float, geom_order: int = 2, refine_steps: int = 0) -> Path:
        """Reference mesh at characteristic size h (validation only; agents mesh themselves).
        ``refine_steps`` > 0 splits every element that many times after meshing (nested
        refinement, ratio 2 per step; new boundary nodes are snapped to the CAD geometry)."""
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
            if geom_order >= 2:
                gmsh.model.mesh.setOrder(2)
            gmsh.write(str(path))
        return path

    def export_step(self, path: str | Path) -> Path:
        import gmsh

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gmsh_session(self.id):
            self.build_geometry()
            gmsh.write(str(path))
        return path

    def statement(self) -> dict:
        """Public, agent-facing problem description (no reference values)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "family": self.family,
            "tags": list(self.tags),
            "dim": self.dim,
            "physics": self.physics,
            "formulation": self.formulation,
            "units": self.units,
            "params": self.params,
            "description": self.description(),
            "qois": [asdict(q) for q in self.qois()],
            "geometry_file": "geometry.step",
        }

    def private(self, order: int = 2) -> dict:
        """Evaluator-only data: references, expectations and the reference setup."""
        return {
            "id": self.id,
            "expected_behavior": self.expected_behavior,
            "expectations": self.expectations(),
            "references": [asdict(r) for r in self.references()],
            "reference_case": self.case("MESH.msh", order=order).to_dict(),
            "study_sizes": self.study_sizes(order),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}>"
