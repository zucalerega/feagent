"""Mesh-system layout helpers shared by run_all.py, linear_ramp.py and compare.py.

    python experiments/physical/meshsys.py ROOT            # (re)build ROOT/_canon/<system>/<part>/<purpose> symlinks
    python experiments/physical/meshsys.py ROOT --list     # print which meshes exist

ROOT is the mesh-systems output root (env MESHSYS_OUT); layout: experiments/mesh_systems/layout.json."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAYOUT = REPO / "experiments" / "mesh_systems" / "layout.json"
PURPOSES = ("linear", "nonlinear")


def layout() -> dict:
    return json.loads(LAYOUT.read_text())


def systems() -> list[str]:
    return list(layout()["systems"])


def parts() -> list[str]:
    return list(layout()["parts"])


def axis(part: str) -> str:
    return layout()["parts"][part]["axis"]


def label(system: str) -> str:
    return layout()["systems"][system].get("label", system)


def mesh_dir(root: Path, system: str, part: str, purpose: str) -> Path:
    return Path(root) / layout()["systems"][system]["dirs"][part][purpose]


def ready(d: Path) -> bool:
    """A system's mesh dir is usable once it has mesh.msh and report.json (report.json is written last)."""
    return (Path(d) / "mesh.msh").exists() and (Path(d) / "report.json").exists()


def canon(root: Path) -> list[Path]:
    """ROOT/_canon/<system>/<part>/<purpose> -> the system's own dir (relative symlinks)."""
    root = Path(root)
    made = []
    for s in systems():
        for p in parts():
            for u in PURPOSES:
                src = mesh_dir(root, s, p, u)
                link = root / "_canon" / s / p / u
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    if link.is_symlink():
                        link.unlink()
                    else:
                        continue
                if src.exists():
                    link.symlink_to(os.path.relpath(src, link.parent))
                    made.append(link)
    return made


def report(d: Path) -> dict:
    """Common fields of a system's report.json: n_nodes, n_elements, dofs, order, build_s, gates {id: status}."""
    r = json.loads((Path(d) / "report.json").read_text())
    c = {**r, **r.get("contract", {})}
    gl = r.get("gates", [])
    if isinstance(gl, dict):
        gl = gl.get("gates", [])
    gl = list(gl) + list(r.get("mesh_gates", []))
    gates = {}
    for g in gl:
        if isinstance(g, dict) and "id" in g:
            st = g.get("status")
            if gates.get(g["id"]) != "fail":                   # worst status wins on duplicates
                gates[g["id"]] = st if gates.get(g["id"]) != "warn" or st == "fail" else "warn"
    return {"n_nodes": c.get("n_nodes"), "n_elements": c.get("n_elements", c.get("n_tets")), "dofs": c.get("dofs"),
            "order": c.get("order"), "build_s": c.get("build_s"), "element": c.get("element") or c.get("mesh_kind"),
            "gates": gates, "fail": sorted(k for k, v in gates.items() if v == "fail"),
            "warn": sorted(k for k, v in gates.items() if v == "warn")}


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["MESHSYS_OUT"])
    if "--list" in sys.argv:
        for s in systems():
            for p in parts():
                print(s, p, *(f"{u}:{'ok' if ready(mesh_dir(root, s, p, u)) else '--'}" for u in PURPOSES))
    else:
        for l in canon(root):
            print(l, "->", os.readlink(l))
