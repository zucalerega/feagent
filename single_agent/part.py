"""A part as the analyst agent sees it: the STEP geometry and nothing else.

`BlindPart` is a feacore Problem built from a STEP file alone. Its faces are the named groups
`f<tag>` (OCC face tags, as for the gold benchmarks) and the solid is `solid`; extra named groups
(unions of faces the agent names itself, e.g. "bolt_pads") can be added per mesh. It has no
statement, no reference setup and no reference values, so nothing that reads it (the geometry
description, the solve feedback, the renderer) can hand the agent the supports or loads.

It is registered in the feacore registry under a neutral label, because the ace helpers it reuses
(ace.geometry_info, ace.solve_feedback) look problems up by id. The label is not the benchmark id.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from feacore.case import Case, QoI
from feacore.problems.base import Problem


def label_for(problem_id: str) -> str:
    """Neutral, stable label for a benchmark: the agent never sees the benchmark id."""
    return "part-" + hashlib.sha256(problem_id.encode()).hexdigest()[:8]


class BlindPart(Problem):
    dim = 3
    physics = "elasticity"
    formulation = "3d"
    units = "SI (m, N, Pa)"
    tags = ("blind",)

    def __init__(self, step_path: str | Path, label: str):
        super().__init__(id=label)
        self.step_path = Path(step_path)
        self.title = label
        self.groups: dict[str, list[int]] = {}   # extra named groups: name -> face tags
        self.agent_qois: list[QoI] = []          # the agent's own QoIs (what the solve feedback evaluates)

    def description(self) -> str:
        return ""     # deliberately empty: no statement to infer supports or loads from

    def qois(self) -> list[QoI]:
        return list(self.agent_qois)

    def case(self, mesh_path, order: int = 2) -> Case:
        raise NotImplementedError("a blind part has no reference setup")

    def build_geometry(self) -> None:
        """The STEP as is (metres), every boundary face a group f<tag>, the solid `solid`, plus the
        extra groups in self.groups."""
        import gmsh

        from feacore.geometry.gmsh_tools import add_group

        ents = gmsh.model.occ.importShapes(str(self.step_path), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        vols = [t for d, t in ents if d == 3]
        faces = sorted({abs(t) for _, t in gmsh.model.getBoundary([(3, v) for v in vols], oriented=False)})
        for t in faces:
            add_group(2, [t], f"f{t}")
        for name, tags in self.groups.items():
            add_group(2, tags, name)
        add_group(3, vols, "solid")

    def face_tags(self) -> list[int]:
        import gmsh

        from feacore.geometry.gmsh_tools import gmsh_session
        with gmsh_session(self.id + "-faces"):
            ents = gmsh.model.occ.importShapes(str(self.step_path), highestDimOnly=True)
            gmsh.model.occ.synchronize()
            vols = [t for d, t in ents if d == 3]
            return sorted({abs(t) for _, t in gmsh.model.getBoundary([(3, v) for v in vols], oriented=False)})


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(part: BlindPart) -> BlindPart:
    """Register the part in the feacore registry (in this process) under its label. An existing BlindPart
    with the same label and the same STEP file is reused; one with the same label and the same STEP content
    (another workspace of the same problem) is replaced, since everything cached by label is geometry."""
    from feacore.problems import registry
    registry.discover()
    old = registry._REGISTRY.get(part.id)
    if old is not None:
        if isinstance(old, BlindPart) and old.step_path.resolve() == part.step_path.resolve():
            return old
        if isinstance(old, BlindPart) and _digest(old.step_path) == _digest(part.step_path):
            registry._REGISTRY[part.id] = part
            return part
        raise ValueError(f"label {part.id!r} is already registered for another problem")
    return registry.register(part)
