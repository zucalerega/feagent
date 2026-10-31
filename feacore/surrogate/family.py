"""Training family for the neural surrogate: a 2D plane-stress plate with a hole or an edge notch.

This family exists only to generate surrogate training / test data. It is deliberately NOT
registered in ``feacore.problems.registry`` (it is not a benchmark for agents).

Geometry: rectangle ``W x H`` with either a circular hole (centre ``cx, cy``, radius ``r``) or a
semicircular notch of radius ``r`` centred at ``(cx, H)`` on the top edge.

Boundary-condition templates (``bc``):
  clamped  left edge fixed, uniform traction ``traction * (cos a, sin a)`` on the right edge
  biaxial  rollers (symmetry) on the left and bottom edges, traction ``traction`` (x) on the
           right edge and ``biax_ratio * traction`` (y) on the top edge

Sampling (``sample_params``): the in-distribution ranges in ``ID_RANGES`` define the training
distribution; ``OOD_CATEGORIES`` describe deliberately out-of-distribution test cases (larger
holes, thin ligaments, notches, steep load angles, extreme Poisson ratios, other aspect ratios,
other mesh sizes, another BC template). Absolute scales (W, E, traction) are sampled over wide
ranges on purpose: the surrogate works in nondimensional variables, so they must not matter.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.geometry.gmsh_tools import add_group, in_box
from feacore.problems.base import Problem

# Mesh size relative to W used for the training data (Problem.mesh at a fixed relative size).
DEFAULT_H_REL = 0.06

# In-distribution parameter ranges (relative to W where a length).
ID_RANGES = {
    "W": (0.05, 0.5),            # m, log-uniform (irrelevant after nondimensionalisation)
    "aspect": (0.5, 1.0),        # H / W
    "r_rel": (0.06, 0.20),       # r / W
    "ligament_rel": (0.08, None),  # min distance hole edge -> plate edge, / W
    "cx_rel": (0.25, 0.75),      # hole centre x / W
    "angle_deg": (-30.0, 30.0),  # traction direction on the right edge (clamped template)
    "traction": (1e6, 1e8),      # Pa, log-uniform
    "E": (50e9, 250e9),          # Pa, log-uniform
    "nu": (0.2, 0.4),
    "thickness": (1e-3, 2e-2),   # m (irrelevant for traction loads in plane stress)
}

OOD_CATEGORIES = {
    "big_hole": "hole radius r/W in [0.24, 0.30] (training max 0.20)",
    "thin_ligament": "hole edge 0.02-0.05 W from the plate edge (training >= 0.08 W)",
    "notch": "semicircular edge notch instead of an interior hole (unseen topology)",
    "steep_load": "traction angle |a| in [50, 90] deg (training |a| <= 30 deg)",
    "extreme_nu": "Poisson ratio in [0.0, 0.1] or [0.45, 0.49] (training [0.2, 0.4])",
    "tall_plate": "aspect ratio H/W in [1.3, 2.0] (training [0.5, 1.0])",
    "fine_mesh": "mesh size 0.5x the training size",
    "biaxial_bc": "rollers on left/bottom + biaxial traction (unseen BC template)",
}


class SurrogatePlate(Problem):
    id = "surrogate_plate"
    title = "Plate with a hole or an edge notch (surrogate training family)"
    family = "surrogate_plate"
    tags = ("2d", "plane_stress", "stress_concentration", "surrogate_training")
    dim = 2
    physics = "elasticity"
    formulation = "plane_stress"
    expected_behavior = "stress_concentration"
    default_params = {
        "W": 0.1, "H": 0.075, "kind": "hole", "r": 0.012, "cx": 0.05, "cy": 0.0375,
        "bc": "clamped", "traction": 10e6, "angle_deg": 0.0, "biax_ratio": 0.5,
        "E": 200e9, "nu": 0.3, "thickness": 0.005, "h0": 0.006,
    }

    def __init__(self, id: str | None = None, **params):
        super().__init__(id=id, **params)
        p = self.params
        if p["kind"] not in ("hole", "notch", "none"):
            raise ValueError(f"kind must be hole | notch | none, got {p['kind']!r}")
        if p["bc"] not in ("clamped", "biaxial"):
            raise ValueError(f"bc must be clamped | biaxial, got {p['bc']!r}")

    @property
    def cutout_group(self) -> str | None:
        return {"hole": "hole", "notch": "notch"}.get(self.params["kind"])

    # ------------------------------------------------------------------ Problem API
    def description(self) -> str:
        p = self.params
        geo = {
            "hole": f"a circular hole of radius {p['r']:.4g} m centred at ({p['cx']:.4g}, {p['cy']:.4g}) m",
            "notch": f"a semicircular notch of radius {p['r']:.4g} m centred at ({p['cx']:.4g}, {p['H']:.4g}) m "
                     f"on the top edge",
            "none": "no cut-out",
        }[p["kind"]]
        if p["bc"] == "clamped":
            a = math.radians(p["angle_deg"])
            bcs = (f"The left edge (x = 0) is fully fixed. The right edge (x = {p['W']:.4g} m) carries a uniform "
                   f"traction ({p['traction'] * math.cos(a):.4g}, {p['traction'] * math.sin(a):.4g}) Pa.")
        else:
            bcs = (f"The left and bottom edges are symmetry planes (rollers). The right edge carries a uniform "
                   f"x-traction of {p['traction']:.4g} Pa and the top edge a y-traction of "
                   f"{p['traction'] * p['biax_ratio']:.4g} Pa.")
        return (f"Plane-stress plate {p['W']:.4g} m x {p['H']:.4g} m, thickness {p['thickness']:.4g} m, with {geo}. "
                f"Linear elastic material E = {p['E']:.4g} Pa, nu = {p['nu']:.4g}. {bcs} Report the maximum von Mises "
                f"stress and the displacement of the midpoint of the right edge.")

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        occ = gmsh.model.occ
        W, H, r = float(p["W"]), float(p["H"]), float(p["r"])
        rect = occ.addRectangle(0.0, 0.0, 0.0, W, H)
        if p["kind"] == "hole":
            disk = occ.addDisk(float(p["cx"]), float(p["cy"]), 0.0, r, r)
            occ.cut([(2, rect)], [(2, disk)])
        elif p["kind"] == "notch":
            disk = occ.addDisk(float(p["cx"]), H, 0.0, r, r)
            occ.cut([(2, rect)], [(2, disk)])
        occ.synchronize()
        tol = 1e-4 * W  # OCC pads bounding boxes by ~1e-7
        left = in_box(1, xmax=0.0, tol=tol)
        right = in_box(1, xmin=W, tol=tol)
        bottom = in_box(1, ymax=0.0, tol=tol)
        top = in_box(1, ymin=H, tol=tol)
        straight = set(left) | set(right) | set(bottom) | set(top)
        add_group(1, left, "left")
        add_group(1, right, "right")
        add_group(1, bottom, "bottom")
        add_group(1, top, "top")
        cut = [t for _, t in gmsh.model.getEntities(1) if t not in straight]
        if self.cutout_group and cut:
            add_group(1, cut, self.cutout_group)
        add_group(2, [t for _, t in gmsh.model.getEntities(2)], "plate")

    def qois(self) -> list[QoI]:
        p = self.params
        q = [QoI("vm_max", "max", field="von_mises")]
        if self.cutout_group:
            q.append(QoI(f"vm_max_{self.cutout_group}", "max", field="von_mises", group=self.cutout_group))
        tip = [float(p["W"]), 0.5 * float(p["H"])]
        q += [QoI("ux_tip", "point", field="ux", point=tip), QoI("uy_tip", "point", field="uy", point=tip)]
        return q

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        p = self.params
        t = float(p["traction"])
        if p["bc"] == "clamped":
            a = math.radians(float(p["angle_deg"]))
            bcs = [BC("fixed", group="left"),
                   BC("traction", group="right", value=[t * math.cos(a), t * math.sin(a)])]
        else:
            bcs = [BC("symmetry", group="left"), BC("symmetry", group="bottom"),
                   BC("traction", group="right", value=[t, 0.0]),
                   BC("traction", group="top", value=[0.0, t * float(p["biax_ratio"])])]
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="plane_stress", order=order,
                              thickness=float(p["thickness"])),
            materials={"material": Material(E=float(p["E"]), nu=float(p["nu"]))},
            bcs=bcs,
            qois=self.qois(),
            meta={"problem": self.id, "family": self.family, "params": dict(p)},
        )


# ---------------------------------------------------------------------------- sampling


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _hole_centre(rng, W, H, r, lig_lo, lig_hi=None, cx_rel=(0.25, 0.75)):
    """Hole centre with every ligament >= lig_lo * W (and, if lig_hi, the smallest one <= lig_hi * W)."""
    for _ in range(200):
        cx = rng.uniform(*cx_rel) * W
        cy = rng.uniform(0.0, H)
        ligs = (cx - r, W - cx - r, cy - r, H - cy - r)
        m = min(ligs) / W
        if m >= lig_lo and (lig_hi is None or m <= lig_hi):
            return cx, cy
    return None


def sample_params(rng: np.random.Generator, category: str = "id", h_rel: float = DEFAULT_H_REL) -> tuple[dict, float]:
    """Random family parameters for ``category`` ("id" or a key of ``OOD_CATEGORIES``).
    Returns (params, mesh size h in m); h = h_rel * max(W, H) (fine_mesh: half of that)."""
    R = ID_RANGES
    for _ in range(1000):
        W = _loguniform(rng, *R["W"])
        aspect = rng.uniform(*R["aspect"])
        kind, bc = "hole", "clamped"
        r_rel = rng.uniform(*R["r_rel"])
        lig_lo, lig_hi = R["ligament_rel"][0], None
        angle = rng.uniform(*R["angle_deg"])
        nu = rng.uniform(*R["nu"])
        h = h_rel
        biax = 0.5
        if category == "big_hole":
            aspect = rng.uniform(0.85, 1.0)
            r_rel = rng.uniform(0.24, 0.30)
            lig_lo = 0.06
        elif category == "thin_ligament":
            lig_lo, lig_hi = 0.02, 0.05
        elif category == "notch":
            kind = "notch"
        elif category == "steep_load":
            angle = rng.uniform(50.0, 90.0) * rng.choice([-1.0, 1.0])
        elif category == "extreme_nu":
            nu = rng.uniform(0.0, 0.1) if rng.random() < 0.5 else rng.uniform(0.45, 0.49)
        elif category == "tall_plate":
            aspect = rng.uniform(1.3, 2.0)
        elif category == "fine_mesh":
            h = 0.5 * h_rel
        elif category == "biaxial_bc":
            bc = "biaxial"
            biax = rng.uniform(-0.5, 1.0)
        elif category != "id":
            raise ValueError(f"unknown category {category!r}; use 'id' or one of {sorted(OOD_CATEGORIES)}")
        H = aspect * W
        r = r_rel * W
        if kind == "hole":
            c = _hole_centre(rng, W, H, r, lig_lo, lig_hi, R["cx_rel"])
            if c is None:
                continue
            cx, cy = c
        else:
            cx = rng.uniform(*R["cx_rel"]) * W
            cy = H
            if r > 0.6 * H:
                continue
        params = {
            "W": W, "H": H, "kind": kind, "r": r, "cx": cx, "cy": cy, "bc": bc,
            "traction": _loguniform(rng, *R["traction"]), "angle_deg": float(angle), "biax_ratio": float(biax),
            "E": _loguniform(rng, *R["E"]), "nu": float(nu), "thickness": _loguniform(rng, *R["thickness"]),
            "h0": h * max(W, H),
        }
        return params, h * max(W, H)
    raise RuntimeError(f"could not sample parameters for category {category!r}")
