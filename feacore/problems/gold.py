"""Gold benchmarks: third-party CAD parts (ABC dataset) with generated supports and loads, and
reference values certified by a nested-mesh convergence study.

They come from the gold-benchmark generator (``feacore.goldgen``, branch ``gold-benchmarks``) and
are data, not code. ``python scripts/gold.py import ARCHIVE`` copies a generator archive into
``gold_root()`` (``FEACORE_GOLD_DIR``, default ``benchmarks/gold``), one directory per BC set in
the benchmark-store layout:

  <id>/public/problem.json      statement for agents: supports and loads in words, faces named f<tag>
  <id>/public/geometry.step     CAD rescaled to metres; its OCC face tags are the f<tag> names
  <id>/private/reference.json   references (certified QoIs), BC set, reference case, part faces
  <id>/private/study.json       the generator's study: every QoI on every mesh level, GCI per QoI
  _meshes/<part>/L<k>.msh       the study's nested tet10 meshes, shared by a part's BC sets
                                (not in git; index.json lists them with their sha256)

A QoI becomes a reference only if it converged with a fine-grid GCI within 1-3 % (tolerance
max(family tolerance, 2 x GCI)); diverging QoIs (peak stresses at clamp edges) are expected to
diverge. Every set found is registered as a ``GoldProblem`` when the registry discovers this module.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import warnings
from pathlib import Path

from feacore.case import Case, QoI
from feacore.problems.base import Problem, Reference
from feacore.problems.registry import register
from feacore.problems.store import DEFAULT_ROOT

# preference for the single scored QoI of a set: global and smooth first, reactions last
SCORE_ORDER = ("strain_energy", "u_mean_", "u_max", "vm_mean", "R")


def gold_root() -> Path:
    return Path(os.environ.get("FEACORE_GOLD_DIR", DEFAULT_ROOT / "gold"))


class GoldProblem(Problem):
    """One BC set of a gold part, backed by its directory under ``gold_root()``."""

    dim = 3
    physics = "elasticity"
    formulation = "3d"
    units = "SI (m, N, Pa)"
    stored = True            # data-backed: feacore.problems.store leaves its files alone

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self._statement = json.loads((self.dir / "public" / "problem.json").read_text())
        super().__init__(id=self._statement["id"])
        self.title = self._statement.get("title", self.id)
        self.family = self._statement.get("family", "")
        self.tags = tuple(self._statement.get("tags", ("gold",)))
        self._private_data: dict | None = None
        self._study_data: dict | None = None

    # ------------------------------------------------------------- stored data
    @property
    def step_path(self) -> Path:
        return self.dir / "public" / self._statement.get("geometry_file", "geometry.step")

    @property
    def part(self) -> dict:
        """Part description from the generator: id, source file, scale, bbox, faces (tag, kind,
        area, centroid, normal / axis)."""
        return self._private()["part"]

    def _private(self) -> dict:
        if self._private_data is None:
            self._private_data = json.loads((self.dir / "private" / "reference.json").read_text())
        return self._private_data

    def study(self) -> dict:
        """The generator's convergence study: {"runs": [per level], "qois": {name: GCI entry}}."""
        if self._study_data is None:
            p = self.dir / "private" / "study.json"
            self._study_data = json.loads(p.read_text()) if p.exists() else {"runs": [], "qois": {}}
        return self._study_data

    def mesh_file(self, level: int) -> Path | None:
        """The generator's reference mesh of level ``level`` (0 = coarsest), if imported."""
        p = self.dir.parent / "_meshes" / self.family / f"L{level}.msh"
        return p if p.exists() else None

    def singular_qois(self) -> list[str]:
        return [k for k, v in self.study()["qois"].items() if v.get("kind") == "singular"]

    def scored_qoi(self) -> str | None:
        """The reference QoI to score a run on when only one is used (e.g. the ACE loop)."""
        def kind(name: str) -> str:
            if name.startswith("u_mean_"):
                return "u_mean_"
            return "R" if name[:1] == "R" and name[2:3] == "_" else name

        names = [r.qoi for r in self.references()]
        for k in SCORE_ORDER:
            hit = [n for n in names if kind(n) == k]
            if hit:
                return hit[0]
        return names[0] if names else None

    # ------------------------------------------------------------- Problem API
    @property
    def expected_behavior(self) -> str:
        return "singular" if self.singular_qois() else "smooth"

    def description(self) -> str:
        return self._statement["description"]

    def qois(self) -> list[QoI]:
        return [QoI(**q) for q in self._statement["qois"]]

    def references(self) -> list[Reference]:
        return [Reference(**r) for r in self._private()["references"]]

    def expectations(self) -> dict[str, str]:
        return {q: "diverge" for q in self.singular_qois()}

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        d = copy.deepcopy(self._private()["reference_case"])
        d["mesh"] = str(mesh_path)
        d["analysis"]["order"] = order
        return Case.from_dict(d)

    def build_geometry(self) -> None:
        """The STEP as is (already in metres), every boundary face a group ``f<tag>``, the solid ``solid``."""
        import gmsh

        from feacore.geometry.gmsh_tools import add_group

        ents = gmsh.model.occ.importShapes(str(self.step_path), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        vols = [t for d, t in ents if d == 3]
        for _, t in gmsh.model.getBoundary([(3, v) for v in vols], oriented=False):
            add_group(2, [t], f"f{t}")
        add_group(3, vols, "solid")

    def study_sizes(self, order: int = 2) -> list[float]:
        """The generator's nominal sizes, coarse to fine."""
        hs = [r["h"] for r in self.study()["runs"]]
        return sorted(set(hs), reverse=True) or [0.1 / 8, 0.1 / 16, 0.1 / 32]

    def export_step(self, path: str | Path) -> Path:
        path = Path(path)
        if path.resolve() != self.step_path.resolve():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.step_path, path)
        return path

    def statement(self) -> dict:
        return copy.deepcopy(self._statement)

    def private(self, order: int = 2) -> dict:
        d = copy.deepcopy(self._private())
        d.setdefault("expected_behavior", self.expected_behavior)
        d.setdefault("expectations", self.expectations())
        d.setdefault("study_sizes", self.study_sizes(order))
        if order != d["reference_case"]["analysis"]["order"]:
            d["reference_case"] = self.case("MESH.msh", order=order).to_dict()
        return d


def load_all(root: str | Path | None = None) -> tuple[list[GoldProblem], dict[str, str]]:
    """Every gold set under ``root`` (default ``gold_root()``), and {directory: error} for sets that
    could not be read."""
    root = Path(root) if root is not None else gold_root()
    found, errors = [], {}
    for p in sorted(root.glob("*/public/problem.json")):
        try:
            found.append(GoldProblem(p.parent.parent))
        except Exception as e:  # noqa: BLE001 - one unreadable set must not hide the others
            errors[str(p.parent.parent)] = f"{type(e).__name__}: {e}"
    return found, errors


_problems, _errors = load_all()
for _p in _problems:
    register(_p)
for _dir, _err in _errors.items():
    warnings.warn(f"gold set {_dir} skipped: {_err}", stacklevel=1)
