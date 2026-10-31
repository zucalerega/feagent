"""Print the agent prompt for one exported benchmark (default NAFEMS LE10).

    python scripts/nafems_task.py [--problem nafems_le10] [--root benchmarks]

The prompt holds the NAFEMS statement (loads and supports in words), the files the agent may use,
the named groups of the provided mesh, and the case-JSON format to reply with. It never contains
the reference value. Score the reply with scripts/compare_nafems.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMPT = """\
[SYSTEM]
You are an expert FEA configuration agent. Set up a static structural analysis of the part below for
the fixed solver. Apply exactly the loads and supports the problem statement describes, no more and
no less. Achieve the requested accuracy at the lowest computational cost (fewest degrees of freedom)
without stress blow-ups, numerical divergence or unconstrained rigid-body modes.

[PROBLEM: {id}]
{description}

[FILES]
- CAD (source of truth): {public}/geometry.step
- Ready-made mesh: {public}/mesh.msh (quadratic tetrahedra, {n_cells} cells). You may use it as is, or
  mesh the STEP yourself with the mesh harness; your mesh must carry named physical groups.

[NAMED GROUPS IN THE PROVIDED MESH]
{groups}

[SOLVER CAPABILITIES]
- physics "elasticity", formulation "{formulation}", element order 1 or 2
- BC types: fixed (all components 0), displacement (listed components = value; 0=x 1=y 2=z),
  symmetry (zero normal displacement on a planar, axis-aligned face), pressure (value > 0 pushes into
  the body), traction (force/area vector), force (total force vector spread over the group)
- Dirichlet BCs may use face, edge or point groups; loads use face groups.

[INSTRUCTIONS]
1. Map every load and support in the statement to a group and a BC type.
2. If the statement cannot be applied with these groups and BC types, output only:
   INVALID_CONFIG: <reason>
3. Otherwise output one JSON object in a ```json fenced block, in this exact format:
{schema}
   "mesh" is relative to {public}/. Units SI (m, N, Pa). Report the requested quantity as a QoI.
"""

SCHEMA = {
    "mesh": "mesh.msh",
    "analysis": {"physics": "elasticity", "formulation": "3d", "order": 2},
    "materials": {"steel": {"E": "<Pa>", "nu": "<->"}},
    "regions": [{"group": "<cell group>", "material": "steel"}],
    "bcs": [{"type": "<bc type>", "group": "<group name>", "components": "[..] (displacement only)",
             "value": "<number or list>"}],
    "qois": [{"name": "<as named in the statement>", "type": "point", "field": "<e.g. syy>",
              "point": ["x", "y", "z"]}],
}


def build(problem_id: str, root: Path) -> str:
    from feacore.solver._skfem_mesh import read_mesh

    public = root / problem_id / "public"
    statement = json.loads((public / "problem.json").read_text())
    md = read_mesh(public / "mesh.msh", statement["dim"], 2)
    groups = "\n".join(f"- {g.name}: {g.kind} ({len(g.index)})" for g in md.groups.values())
    return PROMPT.format(id=problem_id, description=statement["description"], public=public.relative_to(ROOT)
                         if public.is_relative_to(ROOT) else public, n_cells=md.mesh.t.shape[1],
                         groups=groups, formulation=statement["formulation"],
                         schema="   " + json.dumps(SCHEMA, indent=2).replace("\n", "\n   "))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problem", default="nafems_le10")
    ap.add_argument("--root", default=str(ROOT / "benchmarks"))
    args = ap.parse_args(argv)
    print(build(args.problem, Path(args.root)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
