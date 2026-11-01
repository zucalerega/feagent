"""Planted failures: synthetic agent runs with a known root cause, for tests and for evaluating the
diagnoser. Each run is real (a real mesh and a real solve); only the agent's choices are scripted:
the problem's reference setup on a sensible mesh, with exactly one fault injected.

    python -m ace.planted <out_dir>        # write all planted runs as labelled run records

Every run record carries a label: {"true_root_cause": <category>, "true_stage": <stage>}.
Notes on substitutions (this harness cannot express everything literally):
  - point LOADS do not exist here, so the singularity case puts a point CONSTRAINT at the scored
    point (same physics: an infinite stress there that a refinement study exposes). Its root
    cause is that boundary-condition choice, so its true stage is "bcs";
  - every problem has a single load, so "gamed by dropping a load" leaves the model unloaded:
    it solves to zero stress, which looks perfectly converged.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from ace import fea
from ace.checks import reference_setup
from ace.config import TASKS

# A sensible mesh request per problem (quadratic, refined at the scored point, within the cap).
GOOD_MESH = {
    "lame_cylinder_plane_strain": {"h": 0.08, "order": 2, "refinements": [{"group": "bore", "size": 0.025, "radius": 0.03}]},  # 80/120 nodes, 1.9%
    "nafems_le1": {"h": 0.6, "order": 2, "refinements": [{"point": [2.0, 0.0], "size": 0.08, "radius": 0.6}]},  # 148/250 nodes, 1.4%
    "nafems_le10": {"h": 0.6, "order": 2, "refinements": [{"point": [2.0, 0.0, 0.3], "size": 0.25, "radius": 1.0}]},
    "nafems_le11_axisym": {"h": 0.1, "order": 2, "refinements": [{"point": [1.0, 0.0], "size": 0.025, "radius": 0.2}]},  # 0.3%
    "nafems_le11_3d": {"h": 0.2, "order": 2, "refinements": [{"point": [0.0, 1.0, 0.0], "size": 0.05, "radius": 0.4}]},  # 0.8%
    "cooks_membrane": {"h": 4.0, "order": 2, "refinements": [{"point": [48.0, 52.0], "size": 1.0, "radius": 8.0}]},  # 0.06%
    # flow (linear finite volumes meshed on the CAD, ace.flow_mesh; nodes / cap, error with the
    # reference setup, linearUpwind, OpenFOAM v1912)
    "poiseuille_channel": {"h": 0.0008, "quads": True, "refinements": []},                 # 1026/1200 nodes, 0.60%
    "lid_driven_cavity_re100": {"h": 0.025, "quads": True, "refinements": []},             # 1913/2500 nodes, 0.19%
    "taylor_couette_2d": {"h": 0.0008, "quads": True, "refinements": []},                  # 1849/2500 nodes, 0.64%
    # a boundary layer at the cylinder: 0.34 % at 3229 nodes (refinement alone, 2 mm: 0.60 % at 6345 nodes)
    "cylinder_channel_dfg_2d1": {"h": 0.025, "quads": True,                               # 3229/12000 nodes, 0.34%
                                 "refinements": [{"group": "cylinder", "size": 0.004, "radius": 0.12}],
                                 "boundary_layer": {"groups": ["cylinder", "walls"], "first": 0.001, "ratio": 1.3,
                                                    "layers": 4}},
    # the step's corner and the bubble behind it refined (uniform at the floor: ~3 % off at 10.9k nodes)
    "backward_facing_step_2d": {"h": 0.0025, "quads": True,                              # 5310/12000 nodes, 1.6%
                                "refinements": [{"point": [0.0, 0.01], "size": 0.0003, "radius": 0.02},
                                                {"box": [-0.005, 0.0, -1.0, 0.06, 0.02, 1.0], "size": 0.0006}]},
    # swept hexahedra (tets need ~5x the cells: uniform tets at h = 1 mm are 2.4 % off at 4078 nodes)
    "pipe_poiseuille_3d": {"h": 0.001, "quads": True, "refinements": [],                  # 1309/6000 nodes, 0.32%
                           "sweep": {"source": "inlet", "layers": 10}},
    "square_duct_3d": {"h": 0.0007, "quads": True, "refinements": [],                     # 3168/6000 nodes, 1.25%
                       "sweep": {"source": "inlet", "layers": 10, "ratio": 6}},
    # the tube wall (2.5 mm) bends: it needs small elements everywhere around the tube, not at a point
    "abc_00000003_pull": {"h": 0.0022, "order": 2, "refinements": []},  # 40187/60000 nodes, -2.3%
    "abc_00000065_compression": {"h": 0.004, "order": 2, "refinements": []},  # 9506/40000 nodes, ~1.5% from h = 1.6 mm
}


def _drop(setup, group):
    s = copy.deepcopy(setup)
    s["bcs"] = [b for b in s["bcs"] if b.get("group") != group]
    return s


def _move_load(setup, src, dst):
    s = copy.deepcopy(setup)
    for b in s["bcs"]:
        if b.get("group") == src and b["type"] in ("pressure", "traction", "force"):
            b["group"] = dst
    return s


def _scale_E(setup, f):
    s = copy.deepcopy(setup)
    s["material"] = dict(s["material"], E=s["material"]["E"] * f)
    return s


def _add(setup, bc):
    s = copy.deepcopy(setup)
    s["bcs"] = s["bcs"] + [bc]
    return s


def _replace(setup, group, bc):
    return _add(_drop(setup, group), bc)


# name -> (problem, mesh request (None = GOOD_MESH), setup transform, true_root_cause, true_stage)
PLANTED = {
    "control_lame": ("lame_cylinder_plane_strain", None, lambda s: s, "none", None),
    "control_le1": ("nafems_le1", None, lambda s: s, "none", None),
    "missing_constraint": ("lame_cylinder_plane_strain", None, lambda s: _drop(s, "sym_x0"),
                           "missing_constraint", "bcs"),
    "load_on_wrong_face": ("nafems_le1", None, lambda s: _move_load(s, "edge_BC", "edge_DA"),
                           "wrong_load_location", "bcs"),
    "unit_error": ("lame_cylinder_plane_strain", None, lambda s: _scale_E(s, 1e-6),      # E in MPa given as Pa
                   "units", "bcs"),
    "mesh_too_coarse": ("nafems_le1", {"h": 1.2, "order": 1, "refinements": []}, lambda s: s,
                        "mesh_too_coarse", "mesh"),
    "singular_peak": ("lame_cylinder_plane_strain", None,
                      lambda s: _replace(s, "sym_y0", {"type": "fixed", "point": [0.1, 0.0]}),
                      "singularity", "bcs"),           # the point constraint creates the singularity
    "gamed_extra_support": ("nafems_le1", None, lambda s: _add(s, {"type": "fixed", "group": "edge_BC"}),
                            "gaming", "bcs"),
    "gamed_dropped_load": ("nafems_le10", None, lambda s: _drop(s, "face_upper"), "gaming", "bcs"),
}


def make_run(name: str, workdir: Path) -> dict:
    """A run record (same shape as ace.agent.run_agent's) for one planted case."""
    pid, req, transform, cause, stage = PLANTED[name]
    problem = fea.get_problem(pid)
    req = copy.deepcopy(req or GOOD_MESH[pid])
    workdir = Path(workdir)
    m = fea.mesh_from_request(problem, workdir / "mesh_0.msh", req)
    m.update(request=req, index=0)
    setup = transform(reference_setup(problem))
    res = fea.run_solve(problem, m, setup, TASKS[pid]["qoi"])
    trace = [
        {"type": "tool_call", "id": "p0", "name": "mesh", "input": {**req, "using": []}, "turn": 0},
        {"type": "tool_result", "id": "p0", "name": "mesh", "turn": 0, "is_error": not m["ok"],
         "result": {k: v for k, v in m.items() if k not in ("path", "request", "index", "sha256", "gmsh")}},
        {"type": "tool_call", "id": "p1", "name": "solve", "input": {**setup, "using": []}, "turn": 1},
        {"type": "tool_result", "id": "p1", "name": "solve", "turn": 1, "is_error": not res["ok"],
         "result": {k: v for k, v in res.items() if k != "info"}},
    ]
    return {
        "problem_id": pid, "run_id": f"planted-{name}", "generation": 0, "stop": "submitted", "turns": 2,
        "cited": [], "submitted": 0 if res["ok"] else None, "trace": trace, "meshes": [m],
        "solves": [{"mesh": m, "mesh_index": 0, "setup": setup, "result": res}], "usage": [],
        "label": {"true_root_cause": cause, "true_stage": stage},
    }


def write_all(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    paths = []
    for name in PLANTED:
        run = make_run(name, out_dir / "work" / name)
        p = out_dir / f"{run['run_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(run, indent=1, default=str))
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_all(Path(sys.argv[1])):
        print(p)
