"""The automesher refined uniformly: gmsh with all its default sizes (exactly ace.field_error.automesher)
but with Mesh.MeshSizeFactor = f, so every element is f x the default size. f = 1 is the scored
automesher itself (checked against ace/field_refs/<problem>_auto.json). Scored with the same error
the runs were scored with (ace.field_error.errors_of_result: max(field energy-norm error, RMS probe
error) against the cached reference). Writes automesher_sweep.json next to this file only; reads the
cached references in ace/field_refs/ (nothing there is rebuilt: they all exist).

    .venv/bin/python talk_plots/act1_agentic_works/automesher_sweep.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gmsh  # noqa: E402
import meshio  # noqa: E402

from ace import fea  # noqa: E402
from ace.checks import reference_setup  # noqa: E402
from ace.config import TASKS  # noqa: E402
from ace.field_error import _solve, errors_of_result  # noqa: E402
from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session  # noqa: E402

PROBLEMS = ["cooks_membrane", "nafems_le1", "nafems_le11_axisym", "nafems_le11_3d"]
FACTORS = {2: [2.0, 1.5, 1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.22, 0.16],
           3: [1.5, 1.0, 0.8, 0.65, 0.5, 0.4]}


def one(pid: str, f: float) -> dict:
    p = fea.get_problem(pid)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "auto.msh"
        with gmsh_session(p.id):
            p.build_geometry()
            ensure_domain_group(p.dim)
            gmsh.option.setNumber("Mesh.MeshSizeFactor", f)
            gmsh.model.mesh.generate(p.dim)
            gmsh.model.mesh.setOrder(2)
            gmsh.write(str(path))
        res = _solve(p, {"path": str(path), "order": 2}, reference_setup(p), TASKS[pid]["qoi"])
        e = errors_of_result(pid, res)
        return {"factor": f, "nodes": len(meshio.read(path).points), "dofs": int(res.info.n_dofs),
                "error": e["error"], "field": e["field"], "points": e["points"]}


def main() -> None:
    out = {}
    for pid in PROBLEMS:
        dim = fea.get_problem(pid).dim
        rows = []
        for f in FACTORS[dim]:
            r = one(pid, f)
            rows.append(r)
            print(pid, r, flush=True)
        cached = json.loads((ROOT / "ace" / "field_refs" / f"{pid}_auto.json").read_text())
        r1 = next(r for r in rows if r["factor"] == 1.0)
        out[pid] = {"rows": rows, "cached_auto": cached,
                    "factor1_matches_cache": abs(r1["error"] - cached["error"]) < 1e-9 * cached["error"]
                    and r1["dofs"] == cached["dofs"]}
        print(pid, "factor 1 reproduces cached automesher:", out[pid]["factor1_matches_cache"], flush=True)
    (Path(__file__).with_name("automesher_sweep.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
