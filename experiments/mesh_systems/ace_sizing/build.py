#!/usr/bin/env python
"""ace_sizing: ACE-style mesh (global h + refinements at bottom/top and stress hotspots).

build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear [--h H] [--no-hotspots]
"""
from __future__ import annotations

import argparse, json, os, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from acemesh import geometry_model, make_mesh  # noqa: E402
from sanity import hotspots, sanity  # noqa: E402

# h picked by the 1 %-of-next-finer rule on the linear sanity solve (see README), per part
DEFAULT_H = {"abc_00000003": 0.003, "abc_00000065": 0.0035}
HOT_FRAC, HOT_RADIUS_H, CONTACT_FRAC, CONTACT_RADIUS_H = 1 / 3, 4.0, 0.5, 3.0
NL_NODE_CAP = 38000


def refinements(h, spots, frac=CONTACT_FRAC):
    r = [{"group": g, "size": h * frac, "radius": h * CONTACT_RADIUS_H} for g in ("bottom", "top")]
    r += [{"point": p, "size": h * HOT_FRAC, "radius": h * HOT_RADIUS_H} for p, _ in spots]
    return r


def find_hotspots(step, axis, h0, k=2):
    """Coarse pass (ACE agent's first look): contact refinement only, solve, take the k worst von Mises nodes."""
    with tempfile.TemporaryDirectory() as td:
        make_mesh(step, axis, Path(td) / "c.msh", 2 * h0, 2, refinements(2 * h0, []))
        _, res = sanity(Path(td) / "c.msh", axis)
        return hotspots(res, k, min_sep=6 * h0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step"); ap.add_argument("axis", choices="xyz"); ap.add_argument("out")
    ap.add_argument("--purpose", choices=["linear", "nonlinear"], required=True)
    ap.add_argument("--h", type=float); ap.add_argument("--no-hotspots", action="store_true")
    ap.add_argument("--ho-opt", action="store_true", help="gmsh HighOrder optimise after curving (can abort gmsh)")
    a = ap.parse_args(argv)
    t0 = time.time()
    part = Path(a.step).resolve().parents[1].name.removesuffix("_bc0")
    h = a.h or DEFAULT_H[part]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spots = [] if a.no_hotspots else find_hotspots(a.step, a.axis, h)
    from feagent.quality import assess_mesh
    from feagent.spec import QualitySpec
    from feagent.types import GroupResult, MeshResult
    geo = geometry_model(a.step, a.axis)

    def attempt(h, frac):
        s = make_mesh(a.step, a.axis, out / "mesh.msh", h, 2, refinements(h, spots, frac),
                      write_inp=out / "mesh.inp", high_order_opt=a.ho_opt)
        groups = {n: GroupResult(name=n, dim=2, tags=g["tags"], elements=g["elements"], area=g["area"])
                  for n, g in s["groups"].items()}
        groups["domain"] = GroupResult(name="domain", dim=3, tags=[1], elements=s["elements"], area=None)
        mr = MeshResult(ok=True, spec_path="", out_dir=str(out), mesh_path=str(out / "mesh.msh"), exports={},
                        groups=groups, n_nodes=s["nodes"], n_elements=s["elements"], order=2, size_scale=1.0,
                        elapsed_s=0.0, errors=[], warnings=[], gates=[], geometry=geo)
        return s, assess_mesh(out / "mesh.msh", mr, QualitySpec(), "fea")

    # The ACE agent's loop: mesh, read the gates, remesh at a slightly different h / contact size until
    # nothing is inverted and the quality gate passes (curved tet10 near thin walls is sensitive to h).
    best, tried = None, []
    for k, (scale, frac) in enumerate((sc, fr) for sc in (1, .93, .86, .8, .74, .68, .62) for fr in (CONTACT_FRAC, .7)):
        hk = h * scale
        s, rep = attempt(hk, frac)
        if a.purpose == "nonlinear" and s["nodes"] > NL_NODE_CAP:
            continue
        sicn_ok = next(g.status for g in rep.gates if g.id == "sicn") != "fail"
        tried.append({"h": hk, "contact_frac": frac, "sicn_min": rep.sicn.min, "inverted": rep.inverted})
        key = (rep.inverted == 0, sicn_ok, rep.sicn.min)
        if best is None or key > best[0]:
            best = (key, hk, frac)
        if rep.inverted == 0 and sicn_ok:
            break
    if tried and best[1:] != (hk, frac):
        s, rep = attempt(best[1], best[2])
    h, frac = best[1], best[2]
    build_s = time.time() - t0
    d = rep.to_dict()
    d.update(system="ace_sizing", part=part, purpose=a.purpose, n_nodes=s["nodes"], n_elements=s["elements"],
             order=2, dofs=3 * s["nodes"], build_s=round(build_s, 2),
             knobs={"h": h, "floor": s["floor"], "contact_frac": frac, "refinements": refinements(h, spots, frac), "attempts": tried},
             hotspots=[{"point": p, "vm_Pa": v} for p, v in spots])
    d["face_counts"] = {n: len(g["tags"]) for n, g in s["groups"].items()}
    (out / "report.json").write_text(json.dumps(d, indent=1, default=float))
    print(json.dumps({k: d[k] for k in ("part", "purpose", "n_nodes", "n_elements", "dofs", "build_s", "face_counts")}))
    if a.purpose == "linear":
        o, _ = sanity(out / "mesh.msh", a.axis)
        (out / "sanity.json").write_text(json.dumps(o, indent=1)); print(json.dumps(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
