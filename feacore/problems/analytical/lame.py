"""Lamé problems: thick-walled cylinder and thick sphere under internal pressure.

Exact solutions (a = bore radius, b = outer radius, p = internal pressure, E, nu):

Cylinder, K = p a^2 / (b^2 - a^2)  (Lamé 1852; Timoshenko & Goodier, Ch. 4, "Stress distribution
symmetrical about an axis" and "Displacements for symmetrical stress distributions"; Barber,
Elasticity, Ch. 8, the thick-walled cylinder):
    s_rr = K (1 - b^2/r^2),  s_tt = K (1 + b^2/r^2)
    plane strain (eps_z = 0):  s_zz = 2 nu K,
        u_r = (1 + nu) K / E [ (1 - 2 nu) r + b^2 / r ]
    open ends (s_zz = 0):       u_r = K / E [ (1 - nu) r + (1 + nu) b^2 / r ],  u_z = -2 nu K z / E
    von Mises (plane strain) = K sqrt(3 b^4/r^4 + (1 - 2 nu)^2), max at r = a
    von Mises (open ends)    = K sqrt(3 b^4/r^4 + 1),            max at r = a

Sphere, K = p a^3 / (b^3 - a^3)  (Lamé; Timoshenko & Goodier, "symmetrical stress distribution in a
sphere"; Barber, Elasticity, spherically symmetric problems):
    s_rr = K (1 - b^3/r^3),  s_tt = s_pp = K (1 + b^3 / (2 r^3))
    u_r  = K / E [ (1 - 2 nu) r + (1 + nu) b^3 / (2 r^2) ]
    von Mises = |s_tt - s_rr| = 1.5 K b^3 / r^3, max at r = a

All formulas were re-derived with sympy (strain-displacement + Hooke + equilibrium) before use.
Strain energies follow from Clapeyron's theorem, U = 1/2 * (work of the bore pressure), because the
symmetry planes / constrained ends do no work: U = 1/2 p u_r(a) A_bore.
"""

from __future__ import annotations

import math

import numpy as np

from feacore.case import BC, QoI
from feacore.geometry.gmsh_tools import add_group
from feacore.problems.analytical._common import BARBER, STEEL, TG, ExactProblem, quarter_annulus
from feacore.problems.base import Reference
from feacore.problems.registry import register

_CYL_SRC = (f"Lamé thick-walled cylinder: {TG}, Ch. 4 'Stress distribution symmetrical about an axis' "
            f"(stresses) and 'Displacements for symmetrical stress distributions' (u_r); {BARBER}, Ch. 8")
_SPH_SRC = (f"Lamé thick spherical shell: {TG}, 'Symmetrical stress distribution in a sphere'; "
            f"{BARBER}, spherically symmetric problems")


class _LameCylinder(ExactProblem):
    family = "lame_cylinder"
    expected_behavior = "smooth"

    def _K(self) -> float:
        p = self.params
        return p["p"] * p["a"] ** 2 / (p["b"] ** 2 - p["a"] ** 2)

    def _radial(self, r: np.ndarray):
        p = self.params
        a, b, E, nu = p["a"], p["b"], p["E"], p["nu"]
        K = self._K()
        srr = K * (1 - b**2 / r**2)
        stt = K * (1 + b**2 / r**2)
        if self.axial_constrained:
            szz = np.full_like(r, 2 * nu * K)
            ur = (1 + nu) * K / E * ((1 - 2 * nu) * r + b**2 / r)
        else:
            szz = np.zeros_like(r)
            ur = K / E * ((1 - nu) * r + (1 + nu) * b**2 / r)
        return ur, srr, stt, szz

    def ur_exact(self, r: float) -> float:
        return float(self._radial(np.array([r]))[0][0])

    def vm_max_exact(self) -> float:
        p = self.params
        a, b, nu = p["a"], p["b"], p["nu"]
        c = (1 - 2 * nu) if self.axial_constrained else 1.0
        return self._K() * math.sqrt(3 * b**4 / a**4 + c**2)


class LameCylinderPlaneStrain(_LameCylinder):
    id = "lame_cylinder_plane_strain"
    title = "Thick-walled cylinder under internal pressure (plane strain, quarter model)"
    tags = ("2d", "plane_strain", "analytical", "pressure_vessel", "curved_boundary", "symmetry")
    dim = 2
    physics = "elasticity"
    formulation = "plane_strain"
    axial_constrained = True
    default_params = {"a": 0.1, "b": 0.2, "p": 100e6, **STEEL, "thickness": 1.0, "h0": 0.025}

    def description(self) -> str:
        p = self.params
        return (
            "Long thick-walled steel cylinder (pipe) under internal pressure, analysed in plane strain "
            "(axial strain zero: the ends are restrained axially). By symmetry only the quarter cross-section "
            "in the first quadrant is modelled (STEP: a planar face in the xy-plane).\n"
            f"Geometry: quarter annulus centred on the origin, bore radius a = {p['a']} m, outer radius "
            f"b = {p['b']} m, x >= 0 and y >= 0. Edges: straight edge on y = 0 from (a, 0) to (b, 0); outer "
            "arc r = b; straight edge on x = 0 from (0, b) to (0, a); bore arc r = a.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}. Out-of-plane thickness "
            f"{p['thickness']} m (report energies per this thickness).\n"
            f"Loads: uniform internal pressure p = {p['p']:.4g} Pa ({p['p'] / 1e6:g} MPa) on the bore arc. "
            "The outer arc is traction free.\n"
            "Supports: symmetry on x = 0 (u_x = 0) and on y = 0 (u_y = 0); nothing else.\n"
            "Report (SI units):\n"
            "  ur_bore      radial displacement at the bore, u_x at (a, 0) [m]\n"
            "  hoop_bore    hoop stress at the bore, sigma_yy at (a, 0) [Pa]\n"
            "  szz_bore     out-of-plane (axial) stress sigma_zz at (a, 0) [Pa]\n"
            "  vm_max       maximum von Mises stress in the section (including sigma_zz) [Pa]\n"
            "  strain_energy total strain energy of the modelled quarter, for the given thickness [J]"
        )

    def build_geometry(self) -> None:
        g = quarter_annulus(self.params["a"], self.params["b"])
        add_group(1, [g["inner"]], "bore")
        add_group(1, [g["outer"]], "outer")
        add_group(1, [g["x0"]], "sym_x0")
        add_group(1, [g["y0"]], "sym_y0")
        add_group(2, [g["surface"]], "domain")

    def qois(self) -> list[QoI]:
        a = self.params["a"]
        return [
            QoI("ur_bore", "point", field="ux", point=[a, 0.0]),
            QoI("hoop_bore", "point", field="syy", point=[a, 0.0]),
            QoI("szz_bore", "point", field="szz", point=[a, 0.0]),
            QoI("vm_max", "max", field="von_mises"),
            QoI("strain_energy", "strain_energy"),
        ]

    def case(self, mesh_path, order: int = 2):
        p = self.params
        bcs = [BC("symmetry", group="sym_x0"), BC("symmetry", group="sym_y0"),
               BC("pressure", group="bore", value=p["p"])]
        return self.make_case(mesh_path, order, bcs, thickness=p["thickness"])

    def _primary(self, x, y, z):
        r = np.hypot(x, y)
        ur, srr, stt, szz = self._radial(r)
        c, s = x / r, y / r
        return {"ux": ur * c, "uy": ur * s,
                "sxx": srr * c**2 + stt * s**2, "syy": srr * s**2 + stt * c**2,
                "sxy": (srr - stt) * s * c, "szz": szz}

    def references(self) -> list[Reference]:
        p = self.params
        a = p["a"]
        U = 0.5 * p["p"] * self.ur_exact(a) * (math.pi * a / 2) * p["thickness"]
        return [
            self.point_reference("ur_bore", _CYL_SRC + "; u_r(a), plane strain", 0.01),
            self.point_reference("hoop_bore", _CYL_SRC + "; s_tt(a) = K (1 + b^2/a^2)", 0.02),
            self.point_reference("szz_bore", _CYL_SRC + "; plane strain s_zz = nu (s_rr + s_tt) = 2 nu K", 0.02),
            Reference("vm_max", self.vm_max_exact(), "analytical",
                      _CYL_SRC + "; von Mises K sqrt(3 b^4/a^4 + (1-2nu)^2) at r = a", 0.02),
            Reference("strain_energy", U, "analytical",
                      "Clapeyron: U = 1/2 p u_r(a) (pi a / 2) t, with u_r(a) from " + _CYL_SRC, 0.01),
        ]


class LameCylinderAxisym(_LameCylinder):
    """Axisymmetric slice a <= r <= b, 0 <= z <= H. ``axial="constrained"``: u_z = 0 on both
    ends (identical to plane strain); ``axial="free"``: u_z = 0 on z = 0 only, the top end is
    traction free (open-ended cylinder, s_zz = 0, uniform axial contraction)."""

    family = "lame_cylinder"
    dim = 2
    physics = "elasticity"
    formulation = "axisymmetric"
    default_params = {"a": 0.1, "b": 0.2, "H": 0.05, "p": 100e6, **STEEL, "axial": "constrained", "h0": 0.025}

    @property
    def axial_constrained(self) -> bool:
        return self.params["axial"] == "constrained"

    @property
    def title(self):  # type: ignore[override]
        ends = "axially constrained ends" if self.axial_constrained else "open ends"
        return f"Thick-walled cylinder under internal pressure (axisymmetric slice, {ends})"

    @property
    def tags(self):  # type: ignore[override]
        return ("2d", "axisymmetric", "analytical", "pressure_vessel",
                "plane_strain_equivalent" if self.axial_constrained else "open_ended")

    def description(self) -> str:
        p = self.params
        if self.axial_constrained:
            ends = ("Both end faces z = 0 and z = H are restrained axially (u_z = 0) and free radially: the "
                    "slice represents a long cylinder whose axial strain is zero (plane strain).")
        else:
            ends = ("Open-ended cylinder: the end face z = 0 is a symmetry plane (u_z = 0, free radially); "
                    "the end face z = H is traction free. There is no axial load (no end caps).")
        q = [
            "  ur_bore      radial displacement at the bore, u_r at (r, z) = (a, H/2) [m]",
            "  hoop_bore    hoop stress at the bore, sigma_theta at (a, H/2) [Pa]",
        ]
        if self.axial_constrained:
            q.append("  szz_bore     axial stress sigma_zz at (a, H/2) [Pa]")
        else:
            q.append("  uz_top       axial displacement u_z at (b, H) [m]")
        q += ["  vm_max       maximum von Mises stress [Pa]",
              "  strain_energy total strain energy of the modelled slice, full 360 degrees [J]"]
        return (
            "Thick-walled steel cylinder under internal pressure, axisymmetric model. Coordinates: "
            "x = r (radius, axis of revolution at x = 0), y = z (axial). A slice of axial height H of the wall "
            "is modelled (STEP: a planar rectangle in the xy-plane = the r-z half-plane).\n"
            f"Geometry: rectangle a <= r <= b, 0 <= z <= H with bore radius a = {p['a']} m, outer radius "
            f"b = {p['b']} m, H = {p['H']} m. Edges: bore r = a, outer surface r = b, ends z = 0 and z = H.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}.\n"
            f"Loads: uniform internal pressure p = {p['p']:.4g} Pa ({p['p'] / 1e6:g} MPa) on the bore r = a. "
            "Outer surface r = b traction free.\n"
            f"Supports: {ends}\n"
            "Report (SI units):\n" + "\n".join(q)
        )

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        occ = gmsh.model.occ
        s = occ.addRectangle(p["a"], 0, 0, p["b"] - p["a"], p["H"])
        occ.synchronize()
        tags = {}
        for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            x, y, _ = occ.getCenterOfMass(1, c)
            if abs(x - p["a"]) < 1e-9:
                tags["bore"] = c
            elif abs(x - p["b"]) < 1e-9:
                tags["outer"] = c
            elif abs(y) < 1e-9:
                tags["bottom"] = c
            else:
                tags["top"] = c
        for name in ("bore", "outer", "bottom", "top"):
            add_group(1, [tags[name]], name)
        add_group(2, [s], "domain")

    def qois(self) -> list[QoI]:
        p = self.params
        a, b, H = p["a"], p["b"], p["H"]
        q = [QoI("ur_bore", "point", field="ux", point=[a, H / 2]),
             QoI("hoop_bore", "point", field="stt", point=[a, H / 2])]
        if self.axial_constrained:
            q.append(QoI("szz_bore", "point", field="syy", point=[a, H / 2]))
        else:
            q.append(QoI("uz_top", "point", field="uy", point=[b, H]))
        q += [QoI("vm_max", "max", field="von_mises"), QoI("strain_energy", "strain_energy")]
        return q

    def case(self, mesh_path, order: int = 2):
        p = self.params
        bcs = [BC("symmetry", group="bottom"), BC("pressure", group="bore", value=p["p"])]
        if self.axial_constrained:
            bcs.insert(1, BC("symmetry", group="top"))
        return self.make_case(mesh_path, order, bcs)

    def _primary(self, x, y, z):
        p = self.params
        ur, srr, stt, szz = self._radial(x)
        uz = np.zeros_like(x) if self.axial_constrained else -2 * p["nu"] * self._K() * y / p["E"]
        return {"ux": ur, "uy": uz, "sxx": srr, "syy": szz, "sxy": np.zeros_like(x), "stt": stt}

    def references(self) -> list[Reference]:
        p = self.params
        a, H = p["a"], p["H"]
        U = 0.5 * p["p"] * self.ur_exact(a) * (2 * math.pi * a * H)
        tag = "constrained ends (= plane strain)" if self.axial_constrained else "open ends (s_zz = 0)"
        refs = [
            self.point_reference("ur_bore", f"{_CYL_SRC}; u_r(a), {tag}", 0.01),
            self.point_reference("hoop_bore", f"{_CYL_SRC}; s_tt(a) = K (1 + b^2/a^2)", 0.02),
        ]
        if self.axial_constrained:
            refs.append(self.point_reference("szz_bore", f"{_CYL_SRC}; s_zz = 2 nu K (eps_z = 0)", 0.02))
        else:
            refs.append(self.point_reference("uz_top", f"{_CYL_SRC}; open ends: u_z = -2 nu K z / E", 0.01))
        c = "(1-2nu)^2" if self.axial_constrained else "1"
        refs += [
            Reference("vm_max", self.vm_max_exact(), "analytical",
                      f"{_CYL_SRC}; von Mises K sqrt(3 b^4/a^4 + {c}) at r = a ({tag})", 0.02),
            Reference("strain_energy", U, "analytical",
                      f"Clapeyron: U = 1/2 p u_r(a) (2 pi a H), {tag}; u_r(a) from {_CYL_SRC}", 0.01),
        ]
        return refs


class LameSphereAxisym(ExactProblem):
    id = "lame_sphere_axisym"
    title = "Thick sphere under internal pressure (axisymmetric, quarter meridian section)"
    family = "lame_sphere"
    tags = ("2d", "axisymmetric", "analytical", "pressure_vessel", "curved_boundary", "symmetry")
    dim = 2
    physics = "elasticity"
    formulation = "axisymmetric"
    expected_behavior = "smooth"
    default_params = {"a": 0.1, "b": 0.2, "p": 100e6, **STEEL, "h0": 0.025}

    def _K(self) -> float:
        p = self.params
        return p["p"] * p["a"] ** 3 / (p["b"] ** 3 - p["a"] ** 3)

    def _radial(self, rho):
        p = self.params
        b, E, nu = p["b"], p["E"], p["nu"]
        K = self._K()
        ur = K / E * ((1 - 2 * nu) * rho + (1 + nu) * b**3 / (2 * rho**2))
        return ur, K * (1 - b**3 / rho**3), K * (1 + b**3 / (2 * rho**3))

    def description(self) -> str:
        p = self.params
        return (
            "Thick-walled steel sphere (spherical pressure vessel) under internal pressure, axisymmetric "
            "model. Coordinates: x = r (distance from the axis of revolution, axis at x = 0), y = z (along "
            "the axis). By symmetry about the equatorial plane only the quarter meridian section in x >= 0, "
            "y >= 0 is modelled (STEP: a planar face in the xy-plane).\n"
            f"Geometry: quarter annulus centred on the origin, inner radius a = {p['a']} m, outer radius "
            f"b = {p['b']} m. Edges: equator y = 0 from (a, 0) to (b, 0); outer arc of radius b; edge on the "
            "axis x = 0 from (0, b) to (0, a); inner arc of radius a.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}.\n"
            f"Loads: uniform internal pressure p = {p['p']:.4g} Pa ({p['p'] / 1e6:g} MPa) on the inner arc. "
            "The outer surface is traction free.\n"
            "Supports: equatorial symmetry plane y = 0 (u_z = 0); the edge x = 0 lies on the axis of "
            "revolution (u_r = 0 there).\n"
            "Report (SI units):\n"
            "  ur_bore       radial displacement at the bore on the equator, u_r at (a, 0) [m]\n"
            "  uz_pole       axial displacement at the inner pole, u_z at (0, a) [m]\n"
            "  hoop_bore     hoop (circumferential) stress sigma_theta at (a, 0) [Pa]\n"
            "  vm_max        maximum von Mises stress [Pa]\n"
            "  strain_energy total strain energy of the modelled half sphere (full 360 degrees) [J]"
        )

    def build_geometry(self) -> None:
        g = quarter_annulus(self.params["a"], self.params["b"])
        add_group(1, [g["inner"]], "bore")
        add_group(1, [g["outer"]], "outer")
        add_group(1, [g["x0"]], "axis")
        add_group(1, [g["y0"]], "equator")
        add_group(2, [g["surface"]], "domain")

    def qois(self) -> list[QoI]:
        a = self.params["a"]
        return [
            QoI("ur_bore", "point", field="ux", point=[a, 0.0]),
            QoI("uz_pole", "point", field="uy", point=[0.0, a]),
            QoI("hoop_bore", "point", field="stt", point=[a, 0.0]),
            QoI("vm_max", "max", field="von_mises"),
            QoI("strain_energy", "strain_energy"),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [BC("symmetry", group="axis"), BC("symmetry", group="equator"),
               BC("pressure", group="bore", value=self.params["p"])]
        return self.make_case(mesh_path, order, bcs)

    def _primary(self, x, y, z):
        rho = np.hypot(x, y)
        ur, srr, stt = self._radial(rho)
        c, s = x / rho, y / rho
        # meridional plane: (r, z) components of the spherical field; hoop = s_tt
        return {"ux": ur * c, "uy": ur * s,
                "sxx": srr * c**2 + stt * s**2, "syy": srr * s**2 + stt * c**2,
                "sxy": (srr - stt) * s * c, "stt": stt}

    def references(self) -> list[Reference]:
        p = self.params
        a = p["a"]
        ura = float(self._radial(np.array([a]))[0][0])
        return [
            self.point_reference("ur_bore", _SPH_SRC + "; u_r(a)", 0.01),
            self.point_reference("uz_pole", _SPH_SRC + "; u_z at the pole = u_r(a)", 0.01),
            self.point_reference("hoop_bore", _SPH_SRC + "; s_tt(a) = K (1 + b^3/(2a^3))", 0.02),
            Reference("vm_max", 1.5 * self._K() * p["b"] ** 3 / a**3, "analytical",
                      _SPH_SRC + "; von Mises = 1.5 K b^3/a^3 at r = a", 0.02),
            Reference("strain_energy", 0.5 * p["p"] * ura * 2 * math.pi * a**2, "analytical",
                      "Clapeyron: U = 1/2 p u_r(a) (2 pi a^2) for the half sphere; " + _SPH_SRC, 0.01),
        ]


class LameSphere3D(LameSphereAxisym):
    id = "lame_sphere_3d"
    title = "Thick sphere under internal pressure (3D octant)"
    tags = ("3d", "analytical", "pressure_vessel", "curved_boundary", "symmetry")
    dim = 3
    formulation = "3d"
    default_params = {"a": 0.1, "b": 0.2, "p": 100e6, **STEEL, "h0": 0.04}

    def description(self) -> str:
        p = self.params
        return (
            "Thick-walled steel sphere (spherical pressure vessel) under internal pressure, 3D. By symmetry "
            "only the octant x >= 0, y >= 0, z >= 0 is modelled (STEP: a solid octant of a spherical shell).\n"
            f"Geometry: spherical shell centred on the origin, inner radius a = {p['a']} m, outer radius "
            f"b = {p['b']} m, cut by the three coordinate planes. Faces: inner sphere r = a, outer sphere "
            "r = b, and the three planar faces on x = 0, y = 0 and z = 0.\n"
            f"Material: linear elastic, E = {p['E']:.4g} Pa, nu = {p['nu']}.\n"
            f"Loads: uniform internal pressure p = {p['p']:.4g} Pa ({p['p'] / 1e6:g} MPa) on the inner "
            "spherical face. The outer face is traction free.\n"
            "Supports: symmetry on x = 0 (u_x = 0), y = 0 (u_y = 0) and z = 0 (u_z = 0).\n"
            "Report (SI units):\n"
            "  ur_bore       radial displacement at the bore, u_x at (a, 0, 0) [m]\n"
            "  hoop_bore     hoop stress at the bore, sigma_yy at (a, 0, 0) [Pa]\n"
            "  vm_max        maximum von Mises stress [Pa]\n"
            "  strain_energy total strain energy of the modelled octant [J]"
        )

    def build_geometry(self) -> None:
        import gmsh

        p = self.params
        a, b = p["a"], p["b"]
        occ = gmsh.model.occ
        outer = occ.addSphere(0, 0, 0, b, angle1=0.0, angle2=math.pi / 2, angle3=math.pi / 2)
        inner = occ.addSphere(0, 0, 0, a, angle1=0.0, angle2=math.pi / 2, angle3=math.pi / 2)
        out, _ = occ.cut([(3, outer)], [(3, inner)])
        occ.synchronize()
        vol = [t for d, t in out if d == 3]
        groups = {"bore": [], "outer": [], "sym_x0": [], "sym_y0": [], "sym_z0": []}
        tol = 1e-6 * b
        for _, f in gmsh.model.getBoundary([(3, v) for v in vol], oriented=False):
            cx, cy, cz = occ.getCenterOfMass(2, f)  # (bounding boxes are padded by OCC)
            if abs(cx) < tol:
                groups["sym_x0"].append(f)
            elif abs(cy) < tol:
                groups["sym_y0"].append(f)
            elif abs(cz) < tol:
                groups["sym_z0"].append(f)
            else:  # spherical faces: centroid of an octant of a sphere of radius R is at R sqrt(3)/2
                rc = math.sqrt(cx * cx + cy * cy + cz * cz)
                groups["bore" if rc < 0.5 * (a + b) * math.sqrt(3) / 2 else "outer"].append(f)
        for name, tags in groups.items():
            add_group(2, tags, name)
        add_group(3, vol, "domain")

    def qois(self) -> list[QoI]:
        a = self.params["a"]
        return [
            QoI("ur_bore", "point", field="ux", point=[a, 0.0, 0.0]),
            QoI("hoop_bore", "point", field="syy", point=[a, 0.0, 0.0]),
            QoI("vm_max", "max", field="von_mises"),
            QoI("strain_energy", "strain_energy"),
        ]

    def case(self, mesh_path, order: int = 2):
        bcs = [BC("symmetry", group="sym_x0"), BC("symmetry", group="sym_y0"), BC("symmetry", group="sym_z0"),
               BC("pressure", group="bore", value=self.params["p"])]
        return self.make_case(mesh_path, order, bcs)

    def _primary(self, x, y, z):
        rho = np.sqrt(x * x + y * y + z * z)
        ur, srr, stt = self._radial(rho)
        n = np.stack([x, y, z]) / rho
        out = {}
        for i, c in enumerate("xyz"):
            out[f"u{c}"] = ur * n[i]
        # s = stt I + (srr - stt) n n^T
        comps = {"sxx": (0, 0), "syy": (1, 1), "szz": (2, 2), "sxy": (0, 1), "syz": (1, 2), "sxz": (0, 2)}
        for k, (i, j) in comps.items():
            out[k] = (srr - stt) * n[i] * n[j] + (stt if i == j else 0.0)
        return out

    def references(self) -> list[Reference]:
        p = self.params
        a = p["a"]
        ura = float(self._radial(np.array([a]))[0][0])
        return [
            self.point_reference("ur_bore", _SPH_SRC + "; u_r(a)", 0.01),
            self.point_reference("hoop_bore", _SPH_SRC + "; s_tt(a) = K (1 + b^3/(2a^3))", 0.02),
            Reference("vm_max", 1.5 * self._K() * p["b"] ** 3 / a**3, "analytical",
                      _SPH_SRC + "; von Mises = 1.5 K b^3/a^3 at r = a", 0.02),
            Reference("strain_energy", 0.5 * p["p"] * ura * math.pi * a**2 / 2, "analytical",
                      "Clapeyron: U = 1/2 p u_r(a) (pi a^2 / 2) for the octant; " + _SPH_SRC, 0.01),
        ]

    def study_sizes(self, order: int = 2) -> list[float]:
        h0 = self.params["h0"]
        return [h0, h0 / 1.5, h0 / 2.25]


register(LameCylinderPlaneStrain())
register(LameCylinderAxisym(id="lame_cylinder_axisym", axial="constrained"))
register(LameCylinderAxisym(id="lame_cylinder_axisym_open", axial="free"))
register(LameSphereAxisym())
register(LameSphere3D())

__all__ = ["LameCylinderPlaneStrain", "LameCylinderAxisym", "LameSphereAxisym", "LameSphere3D"]
