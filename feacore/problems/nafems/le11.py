"""NAFEMS LE11: solid cylinder / taper / sphere under a linear temperature field.

Definition (NAFEMS TNSB Rev. 3, test LE11, sheet dated 15-6-90/2), verified against:
  * the NAFEMS LE11 data sheet reproduced in the FeenoX repository, doc/nafems-le11-problem.png
    (https://github.com/seamplex/feenox): profile dimensions, T = (x^2 + y^2)^(1/2) + z,
    supports, E = 210 GPa, nu = 0.3, alpha = 2.3e-4 /C, output sigma_zz at A, target -105 MPa
    ("refined axisymmetric");
  * FeenoX example examples/nafems-le11.geo / .fee (sigma_z(A) = -105.04 MPa);
  * FEATool tutorial "Temperature Loading of a Tapered Cylinder" (r in [0.7071, 1.4], height 1.79):
    https://www.featool.com/doc/Structural_Mechanics_07_temperature_loading1

Profile in the (r, z) half-plane, axis r = 0 (sheet dimensions):
  A = (1, 0), B = (1.4, 0)                        bottom face z = 0
  B -> D: arc of radius 1.4 centred at the origin, D at z = 0.7, r = sqrt(1.4^2 - 0.7^2)
  D -> G: straight taper to G = (1.0, 1.39)        (0.7 + 0.345 + 0.345)
  G -> I: cylinder r = 1.0 up to I = (1.0, 1.79)   (+ 0.4)
  I -> H: top face z = 1.79, H = (0.7071, 1.79)
  H -> E: inner cylinder r = 0.7071 down to E = (0.7071, 0.7071) (45 degrees on the unit circle)
  E -> A: arc of radius 1 centred at the origin
FeenoX places D at z = 0.7071 (total height 1.7971); the sheet's dimension chain gives 0.7 / 1.79.
"""

from __future__ import annotations

import math
from pathlib import Path

from feacore.case import BC, Analysis, Case, Load, Material, QoI
from feacore.geometry.gmsh_tools import distance_threshold_field, use_min_field
from feacore.problems.base import Problem, Reference
from feacore.problems.nafems._common import (
    FinerP1Studies,
    LE11_SHEET, STEEL, TNSB, entities_where, group, near, point_tag,
)
from feacore.problems.registry import register

ALPHA = 2.3e-4                   # 1/degC
R_IN = math.sqrt(0.5)            # 0.7071: inner cylinder radius, E at 45 degrees on r = 1
Z_D, Z_G, Z_TOP = 0.7, 1.39, 1.79
R_D = math.sqrt(1.4**2 - Z_D**2)
PROFILE = {  # (r, z)
    "A": (1.0, 0.0), "B": (1.4, 0.0), "D": (R_D, Z_D), "G": (1.0, Z_G),
    "I": (1.0, Z_TOP), "H": (R_IN, Z_TOP), "E": (R_IN, R_IN),
}

GEOMETRY_TEXT = (
    "The meridian section (r = radial distance from the z axis, z = axial coordinate, metres) is "
    "bounded by: the bottom face z = 0 from A = (r 1.0, z 0) to B = (r 1.4, z 0); an outer spherical "
    "surface of radius 1.4 centred at the origin from B up to D = (r sqrt(1.4^2 - 0.7^2) = 1.2124, "
    "z 0.7); an outer conical taper (straight in the section) from D to G = (r 1.0, z 1.39); an outer "
    "cylinder r = 1.0 from G to I = (r 1.0, z 1.79); the top face z = 1.79 from I to H = (r 0.70711, "
    "z 1.79); an inner cylinder r = 0.70711 from H down to E = (r 0.70711, z 0.70711), the 45-degree "
    "point of the unit circle; and an inner spherical surface of radius 1.0 centred at the origin "
    "from E back to A. The body is hollow (it does not reach the axis)."
)
COMMON_TEXT = (
    "Material: isotropic linear elastic, E = 210 GPa, Poisson's ratio 0.3, coefficient of thermal "
    "expansion 2.3e-4 per degC.\n"
    "Load: a prescribed temperature field (no heat-transfer analysis needed), linear in the radial "
    "and axial directions: T [degC] = sqrt(x^2 + y^2) + z with the axis along z, i.e. T = r + z. "
    "The stress-free reference temperature is 0 degC. No mechanical loads.\n"
)


def _profile_curves(occ, plane: str) -> int:
    """Add the LE11 profile as an OCC plane surface. plane="xy": (x, y) = (r, z) (axisymmetric);
    plane="xz": (x, z) = (r, z) at y = 0 (for revolution about z). Returns the surface tag."""
    def P(r, z):
        return occ.addPoint(r, z, 0.0) if plane == "xy" else occ.addPoint(r, 0.0, z)

    o = P(0.0, 0.0)
    p = {k: P(*v) for k, v in PROFILE.items()}
    curves = [
        occ.addLine(p["A"], p["B"]),
        occ.addCircleArc(p["B"], o, p["D"]),
        occ.addLine(p["D"], p["G"]),
        occ.addLine(p["G"], p["I"]),
        occ.addLine(p["I"], p["H"]),
        occ.addLine(p["H"], p["E"]),
        occ.addCircleArc(p["E"], o, p["A"]),
    ]
    loop = occ.addCurveLoop(curves)
    surf = occ.addPlaneSurface([loop])
    occ.remove([(0, o)])
    return surf


def _is_inner(r: float, z: float, tol: float = 1e-6) -> bool:
    """(r, z) on the inner boundary (unit sphere below E, or the inner cylinder r = R_IN)."""
    return abs(math.hypot(r, z) - 1.0) < tol and z <= R_IN + tol or abs(r - R_IN) < tol


def _mid_value(dim: int, tag: int):
    import gmsh

    lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
    return gmsh.model.getValue(dim, tag, [0.5 * (a + b) for a, b in zip(lo, hi)])


class _LE11Base(Problem):
    family = "nafems_le11"
    physics = "elasticity"
    expected_behavior = "thermal"

    def expected_order(self, qoi: QoI, order: int) -> float | None:
        # A sits on the corner where the axially supported bottom face meets the free inner sphere.
        # The axial stress has a steep, non-smooth variation into that corner: the surface value at A
        # converges like ~h^1 with P2 (observed p = 0.9-1.1 on graded and uniform meshes) while
        # points 0.02 m above A converge at the full rate. P1 stresses are O(h) anyway.
        if qoi.name == "szz_A":
            return 1.0
        return None

    def references(self) -> list[Reference]:
        return [Reference(
            qoi="szz_A", value=-105e6, kind="published", tolerance=0.02,
            source=(f"{TNSB}, test LE11, output direct stress sigma_zz at A, target -105 MPa (refined "
                    f"axisymmetric); data sheet reproduced at {LE11_SHEET}; FeenoX "
                    "examples/nafems-le11.fee reports -105.04 MPa"),
            notes="Axial stress at A = (r 1, z 0), the inner edge of the bottom face.",
        )]


class NafemsLE11Axisym(FinerP1Studies, _LE11Base):
    id = "nafems_le11_axisym"
    title = "NAFEMS LE11: cylinder/taper/sphere, thermal load (axisymmetric)"
    tags = ("nafems", "2d", "axisymmetric", "thermal_stress", "curved_boundary")
    dim = 2
    formulation = "axisymmetric"
    default_params = {"h0": 0.1}

    def description(self) -> str:
        return (
            "NAFEMS benchmark LE11, solid cylinder/taper/sphere under a temperature load (linear "
            "elastic, static, axisymmetric about the z axis).\n"
            "Geometry: a hollow solid of revolution. " + GEOMETRY_TEXT + " The STEP file holds the "
            "meridian section in the xy-plane with x = r and y = z (axis of revolution: x = 0).\n"
            + COMMON_TEXT +
            "Supports: bottom face (z = 0, edge AB in the section): zero axial (z) displacement. Top "
            "face (z = 1.79, edge HI): zero axial displacement. Radial expansion is free everywhere.\n"
            "Report: szz_A = axial direct stress sigma_zz at point A (r = 1.0, z = 0), in Pa. (In an "
            "axisymmetric model with x = r, y = z this is the yy stress component.)"
        )

    def build_geometry(self) -> None:
        import gmsh

        occ = gmsh.model.occ
        _profile_curves(occ, "xy")
        occ.synchronize()
        bottom = entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y0, 0.0) and near(y1, 0.0))
        top = entities_where(1, lambda x0, y0, z0, x1, y1, z1: near(y0, Z_TOP) and near(y1, Z_TOP))
        inner, outer = [], []
        for _, t in gmsh.model.getEntities(1):
            if t in bottom or t in top:
                continue
            x, y, _z = _mid_value(1, t)
            (inner if _is_inner(x, y) else outer).append(t)
        group(1, bottom, "edge_AB_bottom")
        group(1, top, "edge_HI_top")
        group(1, inner, "inner")
        group(1, outer, "outer")
        group(2, [t for _, t in gmsh.model.getEntities(2)], "solid")

    def apply_sizing(self, h: float) -> None:
        a = point_tag(*PROFILE["A"])
        f = distance_threshold_field(points=[a], size_min=h / 4, size_max=h, dist_min=0.05, dist_max=0.5)
        use_min_field([f], h_max=h)

    def qois(self) -> list[QoI]:
        # axisymmetric: syy is the axial stress s_zz
        return [QoI(name="szz_A", type="point", field="syy", point=[1.0, 0.0])]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="axisymmetric", order=order, T_ref=0.0),
            materials={"steel": Material(**STEEL, alpha=ALPHA)},
            bcs=[
                BC(type="displacement", group="edge_AB_bottom", components=[1], value=0.0),
                BC(type="displacement", group="edge_HI_top", components=[1], value=0.0),
            ],
            loads=[Load(type="temperature", value="x + y")],  # T = r + z (x = r, y = z)
            qois=self.qois(),
            meta={"problem": self.id, "units": "SI (m, N, Pa, degC)"},
        )


class NafemsLE11Solid(_LE11Base):
    id = "nafems_le11_3d"
    title = "NAFEMS LE11: cylinder/taper/sphere, thermal load (3D quarter model)"
    tags = ("nafems", "3d", "solid", "thermal_stress", "curved_boundary", "symmetry")
    dim = 3
    formulation = "3d"
    default_params = {"h0": 0.2}

    def description(self) -> str:
        return (
            "NAFEMS benchmark LE11, solid cylinder/taper/sphere under a temperature load (3D linear "
            "elastic solid, static).\n"
            "Geometry: a hollow solid of revolution about the z axis; the STEP file holds the quarter "
            "x >= 0, y >= 0 (faces on the planes x = 0 and y = 0 are symmetry planes of the full "
            "body). " + GEOMETRY_TEXT + "\n"
            + COMMON_TEXT +
            "Supports: symmetry on the xz-plane (face on y = 0): zero y-displacement. Symmetry on the "
            "yz-plane (face on x = 0): zero x-displacement. Bottom face (z = 0): zero z-displacement. "
            "Top face (z = 1.79): zero z-displacement. All curved faces are free.\n"
            "Report: szz_A = direct stress sigma_zz at point A = (0, 1.0, 0) (inner edge of the bottom "
            "face, on the y axis), in Pa."
        )

    def build_geometry(self) -> None:
        import gmsh

        occ = gmsh.model.occ
        surf = _profile_curves(occ, "xz")
        occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, math.pi / 2)
        occ.synchronize()
        if len(gmsh.model.getEntities(3)) != 1:
            raise RuntimeError("LE11 geometry: expected a single volume")
        bottom = entities_where(2, lambda x0, y0, z0, x1, y1, z1: near(z0, 0.0) and near(z1, 0.0))
        top = entities_where(2, lambda x0, y0, z0, x1, y1, z1: near(z0, Z_TOP) and near(z1, Z_TOP))
        sym_xz = entities_where(2, lambda x0, y0, z0, x1, y1, z1: near(y0, 0.0) and near(y1, 0.0))
        sym_yz = entities_where(2, lambda x0, y0, z0, x1, y1, z1: near(x0, 0.0) and near(x1, 0.0))
        planar = set(bottom + top + sym_xz + sym_yz)
        inner, outer = [], []
        for _, t in gmsh.model.getEntities(2):
            if t in planar:
                continue
            x, y, z = _mid_value(2, t)
            (inner if _is_inner(math.hypot(x, y), z) else outer).append(t)
        group(2, bottom, "face_bottom")
        group(2, top, "face_top")
        group(2, sym_xz, "face_sym_xz")
        group(2, sym_yz, "face_sym_yz")
        group(2, inner, "inner")
        group(2, outer, "outer")
        group(3, [t for _, t in gmsh.model.getEntities(3)], "solid")

    def apply_sizing(self, h: float) -> None:
        a = point_tag(0.0, 1.0, 0.0)
        f = distance_threshold_field(points=[a], size_min=h / 6, size_max=h, dist_min=0.05, dist_max=0.6)
        use_min_field([f], h_max=h)

    def qois(self) -> list[QoI]:
        return [QoI(name="szz_A", type="point", field="szz", point=[0.0, 1.0, 0.0])]

    def case(self, mesh_path: str | Path, order: int = 2) -> Case:
        return Case(
            mesh=str(mesh_path),
            analysis=Analysis(physics="elasticity", formulation="3d", order=order, T_ref=0.0),
            materials={"steel": Material(**STEEL, alpha=ALPHA)},
            bcs=[
                BC(type="symmetry", group="face_sym_xz"),
                BC(type="symmetry", group="face_sym_yz"),
                BC(type="displacement", group="face_bottom", components=[2], value=0.0),
                BC(type="displacement", group="face_top", components=[2], value=0.0),
            ],
            loads=[Load(type="temperature", value="sqrt(x**2 + y**2) + z")],
            qois=self.qois(),
            meta={"problem": self.id, "units": "SI (m, N, Pa, degC)"},
        )


register(NafemsLE11Axisym())
register(NafemsLE11Solid())
