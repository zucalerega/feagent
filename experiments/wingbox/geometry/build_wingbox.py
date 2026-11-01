"""Build the stage-1 wingbox (mm) with gmsh's OpenCASCADE kernel.

Writes one STEP per part to parts/<part>.step and wingbox.step with all 10 solids (each part its
own solid, touching face to face; the mesh harness fragments them conformally on load).
Run from the repo root:  CUDA_VISIBLE_DEVICES="" uv run python experiments/wingbox/geometry/build_wingbox.py
"""
from __future__ import annotations

import json
from pathlib import Path

import gmsh

HERE = Path(__file__).resolve().parent
SPAN, WIDTH, DEPTH = 2000.0, 600.0, 150.0
T_SKIN, T_SPAR, T_RIB = 20.0, 15.0, 10.0
RIB_Y = [0.0, 400.0, 800.0, 1200.0, 1600.0, 1990.0]   # rib start y; each rib is T_RIB thick

# part -> (x0, y0, z0, dx, dy, dz)
PARTS = {
    "upper_skin": (0.0, 0.0, DEPTH - T_SKIN, WIDTH, SPAN, T_SKIN),
    "lower_skin": (0.0, 0.0, 0.0, WIDTH, SPAN, T_SKIN),
    "front_spar": (0.0, 0.0, T_SKIN, T_SPAR, SPAN, DEPTH - 2 * T_SKIN),
    "rear_spar": (WIDTH - T_SPAR, 0.0, T_SKIN, T_SPAR, SPAN, DEPTH - 2 * T_SKIN),
    **{f"rib_{i}": (T_SPAR, y, T_SKIN, WIDTH - 2 * T_SPAR, T_RIB, DEPTH - 2 * T_SKIN)
       for i, y in enumerate(RIB_Y)},
}


def hand_volume() -> dict[str, float]:
    return {k: dx * dy * dz for k, (_, _, _, dx, dy, dz) in PARTS.items()}


def write_step(names: list[str], path: Path) -> float:
    gmsh.clear()
    gmsh.model.add(path.stem)
    for n in names:
        gmsh.model.occ.addBox(*PARTS[n])
    gmsh.model.occ.synchronize()
    vol = sum(gmsh.model.occ.getMass(3, t) for _, t in gmsh.model.getEntities(3))
    gmsh.write(str(path))
    return vol


def main() -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    try:
        occ = {n: write_step([n], HERE / "parts" / f"{n}.step") for n in PARTS}
        total = write_step(list(PARTS), HERE / "wingbox.step")
    finally:
        gmsh.finalize()
    hand = hand_volume()
    summary = {"parts": {n: {"hand_mm3": hand[n], "occ_mm3": occ[n]} for n in PARTS},
               "total_hand_mm3": sum(hand.values()), "total_occ_mm3": total,
               "rel_error": total / sum(hand.values()) - 1.0}
    (HERE / "volumes.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
