"""Opt-in glue to the feagent mesh harness (ACE_HARNESS=1, ace.config.HARNESS).

For a 3D problem the agent's inspect_geometry result gains `harness`: the feagent geometry digest
of the problem's built geometry (every CAD face labelled with the named groups it belongs to), plus
its features (holes, fillets) and hazards (thin walls, small fillets/faces, short edges) with
locations and groups, i.e. where local refinement pays. Each mesh result gains `quality`: the
feagent mesh gate table, measured on the written .msh against the CAD. Nothing here writes a mesh
or changes a node count or a score. 2D and axisymmetric problems are left alone (the harness
analyses 3D OCC solids only). feagent is imported lazily: ACE imports without it.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import gmsh

from ace import config, fea
from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
from feacore.problems.base import Problem

MAX_ITEMS = 10    # features / hazards listed in the inspect_geometry result (the digest has the counts)
MAX_GROUPS = 40   # named surface groups listed in the digest
GROUPS_NOTE = "(not listed; NAMED SURFACE GROUPS above gives every group's faces)"

TOOL_NOTES = {
    "inspect_geometry": " Also returns `harness`: a geometry digest (CAD faces with kind, area, centre, "
                        "normal/axis, radius, and the named groups each face belongs to), `features` "
                        "(holes, fillets: radius, axis, groups) and `hazards` (thin walls, small fillets "
                        "or faces, short edges: value, threshold, location, groups), i.e. where local "
                        "refinement pays.",
    "mesh": " Also returns `quality`: the mesh gates (inverted elements, minimum SICN and gamma, volume and "
            "group-area fidelity vs the CAD, connectivity, elements across thin walls), each with "
            "status, value, threshold and a one-line message.",
}


def applies(problem: Problem) -> bool:
    """The harness is on and can analyse this problem (3D solids only)."""
    return config.HARNESS and problem.dim == 3


def tool_specs(tools: list[dict]) -> list[dict]:
    """A copy of the agent's tool schemas whose descriptions mention the extra result fields."""
    return [dict(t, description=t["description"] + TOOL_NOTES.get(t["name"], "")) for t in tools]


# --------------------------------------------------------------------------- geometry


@lru_cache(maxsize=8)
def _analysis(problem_id: str):
    """(feagent GeometryModel, {surface group name: CAD face tags}) of the problem's built geometry,
    from the same session pattern as fea.geometry_summary; ref_size = bbox diagonal / 40."""
    from feagent.geometry import analyze_current_model

    problem = fea.get_problem(problem_id)
    with gmsh_session(problem.id):
        problem.build_geometry()
        ensure_domain_group(problem.dim)
        bbox = fea._bbox_of(gmsh.model.getEntities(problem.dim))
        geo = analyze_current_model(ref_size=math.dist(bbox[:3], bbox[3:]) / 40, source=problem.id)
        groups = {gmsh.model.getPhysicalName(d, t): [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, t)]
                  for d, t in gmsh.model.getPhysicalGroups(2)}
    return geo, groups


def inspect(problem: Problem) -> dict:
    """The `harness` field of inspect_geometry; {"error": ...} if anything in the harness fails."""
    try:
        geo, groups = _analysis(problem.id)
        labels = _face_labels(groups)
        return {"digest": _digest(geo, groups, labels), "features": _features(geo, labels),
                "hazards": _hazards(geo, labels)}
    except Exception as e:  # noqa: BLE001  the agent still gets the plain summary; the run goes on
        return {"error": f"harness geometry analysis failed: {type(e).__name__}: {e}"}


def _face_labels(groups: dict[str, list[int]]) -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {}
    for name, tags in groups.items():
        for t in tags:
            labels.setdefault(t, []).append(name)
    return labels


def _groups_of(faces, labels: dict[int, list[str]]) -> list[str]:
    return sorted({n for f in faces for n in labels.get(int(f), [])})


def _digest(geo, groups: dict[str, list[int]], labels: dict[int, list[str]]) -> str:
    """feagent's summary digest, adapted: a NAMED SURFACE GROUPS section after the header, a groups
    column on the FACES table, and no feagent-only hints (spec suggestions, detail=full, units: ACE
    problems are built in code)."""
    from feagent.geometry import digest_text

    head, *rest = digest_text(geo, detail="summary").rstrip("\n").split("\n\n")
    fix = {"FACES": lambda s: _label_faces(s, labels), "HAZARDS": _drop_suggestions}
    rest = [fix.get(s.split(" ", 1)[0], lambda s: s)(s).replace("(use detail=full)", GROUPS_NOTE) for s in rest]
    return "\n\n".join([head.replace(f"  units={geo.units}", "", 1), _named_groups(groups), *rest]) + "\n"


def _named_groups(groups: dict[str, list[int]]) -> str:
    lines = ["NAMED SURFACE GROUPS (names for refinements and BCs)"]
    lines += [f"- {n}: faces {sorted(t)}" for n, t in list(groups.items())[:MAX_GROUPS]]
    more = len(groups) - MAX_GROUPS
    return "\n".join(lines + ([f"… {more} more (inspect_geometry `groups` lists them all)"] if more > 0 else []))


def _label_faces(section: str, labels: dict[int, list[str]]) -> str:
    lines = section.split("\n")
    lines[1] += "  groups"
    for i, line in enumerate(lines[2:], 2):
        tag = line.split(maxsplit=1)[0]
        if tag.isdigit():
            lines[i] = f"{line}  {','.join(labels.get(int(tag), ['-']))}"
    return "\n".join(lines)


def _drop_suggestions(section: str) -> str:
    return "\n".join(line.split(" → ", 1)[0] for line in section.split("\n"))


def _num(x, zero: float = 0.0):
    """4 significant digits (lists element-wise), |x| <= zero -> 0 (kernel noise); None stays None."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_num(v, zero) for v in x]
    return 0.0 if abs(x) <= zero else float(f"{float(x):.4g}")


def _features(geo, labels: dict[int, list[str]]) -> list[dict]:
    z = 1e-6 * geo.diag
    return [{"kind": ft.kind, "radius": _num(ft.params["radius"]), "axis": _num(ft.params["axis"], 1e-9),
             "axis_point": _num(ft.params["axis_point"], z), "length": _num(ft.params["depth"]),
             "faces": list(ft.faces), "groups": _groups_of(ft.faces, labels)}
            for ft in geo.features[:MAX_ITEMS]]


def _hazards(geo, labels: dict[int, list[str]]) -> list[dict]:
    edge_faces = {e.tag: e.faces for e in geo.edges}
    out = []
    for hz in geo.hazards[:MAX_ITEMS]:
        # Faces the hazard is on: its faces, or the faces meeting at its edges.
        faces = sorted({f for d, t in hz.entities for f in {2: [t], 1: edge_faces.get(t, [])}.get(d, [])})
        out.append({"kind": hz.kind, "value": _num(hz.value), "threshold": _num(hz.threshold),
                    "location": _num(hz.location, 1e-6 * geo.diag), "faces": faces, "groups": _groups_of(faces, labels),
                    "message": hz.message})
    return out


# --------------------------------------------------------------------------- mesh quality


def quality(problem: Problem, mesh_path: str | Path) -> list[dict] | dict:
    """The `quality` field of a mesh result: feagent's mesh gates on the written .msh (read only).
    Volumes and areas of order-2 meshes are integrated with the curved-element Jacobian."""
    try:
        from feagent.quality import assess_mesh
        from feagent.spec import QualitySpec

        result = _mesh_result(problem, mesh_path)
        with _quiet_stderr():   # feagent's own gmsh sessions log every .msh read at Info level
            rep = assess_mesh(mesh_path, result, QualitySpec(), analysis="fea")
        return [{"id": g.id, "status": g.status, "value": _num(g.value), "threshold": _num(g.threshold),
                 "message": g.message} for g in rep.gates]
    except Exception as e:  # noqa: BLE001  the mesh itself is fine; only the measurement failed
        return {"error": f"harness quality check failed: {type(e).__name__}: {e}"}


@contextmanager
def _quiet_stderr():
    """Send fd 2 to /dev/null. feagent's gmsh_session opens its own sessions (gmsh options reset on
    initialize, so General.Verbosity can't be set from here) and routes gmsh output to stderr."""
    sys.stderr.flush()
    saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(null)


def _mesh_result(problem: Problem, mesh_path: str | Path):
    """The minimal feagent MeshResult assess_mesh needs: the CAD model and the surface groups."""
    from feagent.types import GroupResult, MeshResult

    geo, groups = _analysis(problem.id)
    return MeshResult(ok=True, spec_path="", out_dir=str(Path(mesh_path).parent), mesh_path=str(mesh_path),
                      exports={}, groups={n: GroupResult(name=n, dim=2, tags=t, elements=0, area=None)
                                          for n, t in groups.items()},
                      n_nodes=0, n_elements=0, order=0, size_scale=1.0, elapsed_s=0.0, errors=[],
                      warnings=[], gates=[], geometry=geo)
