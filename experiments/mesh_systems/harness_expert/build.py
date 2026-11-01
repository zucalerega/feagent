#!/usr/bin/env python
"""harness_expert: the FEAgent mesh harness driven like an expert.

usage: build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear

The expert workflow, frozen into a spec per (part, purpose): `feagent inspect` digest ->
mesh.yaml with groups by (normal + box) query with `expect`, thin-wall layers, local refinement at
fillets/rounds/holes and the platen faces -> `feagent check` (mesh + gates) -> report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# Per part: platen queries (normal + box on the face centre; metres) and the physics-driven sizes.
# Sizes in metres. `thin` = size.thin_wall_layers; refine entries are (selector, size, distance).
PARTS = {
    "abc_00000003": {
        "axis": "z",
        "bottom": ({"normal": [0, 0, -1], "box": [[-1, -1, -1e-3], [1, 1, 1e-3]]}, 1),
        "top": ({"normal": [0, 0, 1], "box": [[-1, -1, 0.0610], [1, 1, 0.0615]]}, 1),
        "linear": {"global": 0.006, "layers": 2, "min": 0.0004,
                   "refine": [("bottom", 0.0020, 0.006), ("top", 0.0020, 0.006),
                              ({"kind": "torus"}, 0.0012, 0.005),
                              ({"kind": "cone"}, 0.0005, 0.012),
                              ({"feature": "round"}, 0.0015, 0.005)]},
        "nonlinear": {"global": 0.008, "layers": 1, "min_el": 1, "min": 0.0006, "order": 2,
                      "refine": [("bottom", 0.0030, 0.006), ("top", 0.0030, 0.006),
                                 ({"kind": "torus"}, 0.0020, 0.005),
                                 ({"kind": "cone"}, 0.0012, 0.012),
                                 ({"feature": "round"}, 0.0022, 0.005)]},
    },
    "abc_00000065": {
        "axis": "y",
        "bottom": ({"normal": [0, -1, 0], "box": [[-1, -0.0445, -1], [1, -0.0435, 1]]}, 1),
        "top": ({"normal": [0, 1, 0], "box": [[-1, 0.0333, -1], [1, 0.0343, 1]]}, 2),
        "linear": {"global": 0.006, "layers": 2, "min": 0.0004,
                   "refine": [("bottom", 0.0025, 0.005), ("top", 0.0025, 0.005),
                              ({"feature": "fillet"}, 0.0015, 0.005),
                              ({"feature": "hole"}, 0.0015, 0.004)]},
        "nonlinear": {"global": 0.012, "layers": 1, "min_el": 1, "curv": 8, "min": 0.0008, "order": 2,
                      "refine": [("bottom", 0.006, 0.005), ("top", 0.006, 0.005),
                                 ({"feature": "fillet"}, 0.0040, 0.005),
                                 ({"feature": "hole"}, 0.0016, 0.0035)]},
    },
}


def make_spec(step: Path, part: str, purpose: str, out: Path) -> dict:
    cfg, knobs = PARTS[part], PARTS[part][purpose]
    groups = {}
    for name in ("bottom", "top"):
        sel, n = cfg[name]
        groups[name] = {"select": sel, "expect": n}
    refine = []
    for sel, size, dist in knobs["refine"]:
        entry = {"group": sel} if isinstance(sel, str) else {"select": sel}
        refine.append({**entry, "size": size, "distance": dist})
    return {
        "name": f"{part}_{purpose}", "geometry": str(step.resolve()), "analysis": "fea",
        "groups": groups,
        "size": {"global": knobs["global"], "min": knobs["min"], "curvature": knobs.get("curv", 12),
                 "thin_wall_layers": knobs["layers"]},
        "refine": refine,
        "quality": {"thin_wall_min_elements": knobs.get("min_el", 2)},
        "element": {"order": knobs.get("order", 2), "optimize": "netgen"},
        "export": ["msh", "inp"], "output": ".",
        "bcs": {"fea": {"fixed": ["bottom"], "loads": [{"group": "top", "force": [0, 0, 0]}],
                        "E": 1879.8e6, "nu": 0.35}},
    }


def strip_surface_elements(inp: Path) -> None:
    """CalculiX treats gmsh's CPS6/CPS3 surface elements as 2D elements and rejects the 3D deck: drop those
    blocks and the ELSETs made of them (`bottom`, `top`); NSETs and the `domain` ELSET stay."""
    out, skip = [], False
    for line in inp.read_text().splitlines():
        if line.startswith("*"):
            u = line.upper().replace(" ", "")
            skip = ("*ELEMENT" in u and "TYPE=CPS" in u) or u in ("*ELSET,ELSET=BOTTOM", "*ELSET,ELSET=TOP")
        if not skip:
            out.append(line)
    inp.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step"), ap.add_argument("axis"), ap.add_argument("out")
    ap.add_argument("--purpose", choices=["linear", "nonlinear"], required=True)
    a = ap.parse_args()
    step, out = Path(a.step), Path(a.out)
    part = step.parent.parent.name.replace("_bc0", "")
    out.mkdir(parents=True, exist_ok=True)
    spec = make_spec(step, part, a.purpose, out)
    force = 100.0 * (1 if a.axis in "xyz" else 0)
    sign = -1.0  # compressive on the top platen: push in -axis direction? top is at +axis -> -axis
    vec = [0.0, 0.0, 0.0]
    vec["xyz".index(a.axis)] = sign * force
    spec["bcs"]["fea"]["loads"][0]["force"] = vec
    (out / "mesh.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    from feagent.check import run_check
    t0 = time.time()
    rep = run_check(out / "mesh.yaml")
    build_s = time.time() - t0
    strip_surface_elements(out / "mesh.inp")
    rj = json.loads((out / "report.json").read_text())
    m = rj.get("mesh", {})
    order = spec["element"]["order"]
    n_nodes = m.get("n_nodes")
    rj["contract"] = {"system": "harness_expert", "part": part, "purpose": a.purpose,
                      "n_nodes": n_nodes, "n_elements": m.get("n_elements"), "order": order,
                      "dofs": 3 * n_nodes if n_nodes else None, "build_s": round(build_s, 1)}
    (out / "report.json").write_text(json.dumps(rj, indent=1))
    (out / "mesh.png").write_bytes((out / "quality.png").read_bytes())
    print(json.dumps(rj["contract"]), rep.status)
    for g in rep.gates:
        print(f"  {g.id:18s} {g.status:5s} {g.message}")
    return 0 if rep.status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
