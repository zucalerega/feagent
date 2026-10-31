"""Backend-independent result interface. Every solver backend returns a ``Result``."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from feacore.case import Case, QoI


@dataclass
class SolveInfo:
    backend: str             # "skfem" | "ccx"
    physics: str
    formulation: str
    order: int               # FE interpolation order actually used
    dim: int                 # mesh dimension (2 or 3)
    cell_type: str           # mesh cell type, e.g. "triangle6", "tetra10"
    n_nodes: int             # mesh nodes
    n_cells: int             # mesh cells of dimension ``dim``
    n_dofs: int              # unknowns before Dirichlet elimination
    measure: float           # area (2D) / volume (3D) of the mesh, no thickness / 2*pi factors
    h_rep: float             # representative cell size (measure / n_cells) ** (1 / dim)
    wall_time: float         # seconds for assembly + solve + recovery
    warnings: list[str] = field(default_factory=list)


class Result(ABC):
    """Solution of one case. Field names are listed in ``feacore.case``."""

    info: SolveInfo
    case: Case

    @abstractmethod
    def probe(self, field: str, points) -> np.ndarray:
        """Values of ``field`` at points (n, dim) or (n, 3). Stresses come from the recovered
        continuous field. Points marginally outside the discrete domain (curved boundaries)
        are evaluated in the nearest cell."""

    @abstractmethod
    def nodal(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """(coordinates (n, dim), values (n,)) of ``field`` at the nodes of the solution space."""

    @abstractmethod
    def evaluate(self, qoi: QoI) -> float:
        """Evaluate one quantity of interest (see the QoI table in ``feacore.case``)."""

    def evaluate_all(self) -> dict[str, float]:
        return {q.name: float(self.evaluate(q)) for q in self.case.qois}

    def write_vtu(self, path: str | Path) -> Path:  # optional for backends
        raise NotImplementedError

    def summary(self) -> dict:
        qois = self.evaluate_all()  # first: QoI evaluation may append to info.warnings
        return {"info": asdict(self.info), "qois": qois, "meta": self.case.meta}

    def save_summary(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))
        return path
