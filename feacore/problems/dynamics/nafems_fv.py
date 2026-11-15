"""NAFEMS free-vibration benchmarks that a plane, axisymmetric or solid model can represent.

FV32  cantilevered tapered membrane (plane stress)
FV41  free cylinder, axisymmetric vibration (axisymmetric, one rigid-body mode)
FV52  simply supported "solid" square plate (3D, three rigid-body modes)

Definitions and reference frequencies: NAFEMS TNSB Rev. 3 (1990) and NAFEMS R0015 (1987), which are not
public. The values used here were read from search-engine extracts of these public reproductions (their
hosts are not reachable from the build session, so the pages themselves were not opened):

* Abaqus Benchmarks Guide, "FV32: Cantilevered tapered membrane", "FV41: Free cylinder: axisymmetric
  vibration", "FV52: Simply supported 'solid' square plate"
  (https://abaqus-docs.mit.edu/2017/English/SIMACAEBMKRefMap/simabmk-c-fv32.htm, ...-fv41.htm, ...-fv52.htm):
  the NAFEMS reference rows, plate thickness 0.05 m (FV32), wall thickness 0.4 m (FV41), E = 200 GPa,
  nu = 0.3, rho = 8000 kg/m^3, ux = uy = 0 along the y-axis (FV32), uz = 0 along the four edges on the
  plane z = -0.5 m (FV52).
* Altair OptiStruct verification OS-V 0440 (FV32: the taper runs from 5.0 m to 1.0 m), OS-V 0450 (FV41:
  unsupported, the first mode is a rigid-body mode) and OS-V 0455 (FV52: 10 m x 10 m x 1 m, three rigid
  modes) (https://help.altair.com/hwsolvers/os/topics/solvers/os/cantilevered_tapered_membrane_fv32_r.htm).
* Ansys Verification Manual, Benchmark C5 (FV52: modes 1-3 rigid, modes 4-10 = 45.897, 109.44, 109.44,
  167.89, 193.59, 206.19, 206.19 Hz; supports on the bottom edges only, not at the mid-plane).

NOT confirmed by an extract and taken from the author's memory of the NAFEMS sheets: the length 10 m of
FV32, and the mean radius 2.0 m and length 10 m of FV41. With these the fixed solver reproduces all six
(FV32) and all five (FV41) published frequencies to 0.05 %, which wrong dimensions would not do.

FV52 has no mesh-converged answer for a solid model: a displacement constraint along a LINE of a 3D
solid has zero capacity (the displacement under a line load is logarithmically singular), so the
supports soften as the mesh is refined and the bending frequencies drift down without limit, slowly
(P2 tetrahedra: -2.5 % at h = 2 m, -4.2 % at h = 1 m, -5.6 % at h = 0.5 m against the published 45.897 Hz).
The published values are those of a thick-plate theory, which the coarse meshes of the benchmark
reproduce to a few per cent. Only the in-plane mode (193.59 Hz), which does not load the supports, is
a converging quantity; the bending modes are declared singular (``expectations`` = "diverge"). The
well-posed counterpart with an exact 3D reference is ``ss_thick_plate_modes`` (``exact_modes.py``).
"""

from __future__ import annotations

from feacore.case import BC
from feacore.geometry.gmsh_tools import add_group, in_box
from feacore.problems.base import Reference
from feacore.problems.dynamics._common import ABAQUS_BMK, NAFEMS_FV_STEEL, R0015, TNSB, ModalProblem
from feacore.problems.registry import register

_MATERIAL = "Material: isotropic linear elastic, E = 200 GPa, Poisson's ratio 0.3, density 8000 kg/m^3.\n"


def _source(test: str, extra: str) -> str:
    return (f"{TNSB}, test {test} (also {R0015}); values as reproduced in the {ABAQUS_BMK} "
            f"(https://abaqus-docs.mit.edu/2017/English/SIMACAEBMKRefMap/simabmk-c-{test.lower()}.htm) and {extra}")


class NafemsFV32(ModalProblem):
    id = "nafems_fv32"
    title = "NAFEMS FV32: cantilevered tapered membrane, natural frequencies (plane stress)"
    family = "nafems_fv32"
    tags = ("nafems", "2d", "plane_stress", "modal", "dynamics")
    dim = 2
    formulation = "plane_stress"
    n_modes = 6
    default_params = {**NAFEMS_FV_STEEL, "thickness": 0.05, "h0": 1.0}
    FREQUENCIES = (44.623, 130.03, 162.70, 246.05, 379.90, 391.44)

    def description(self) -> str:
        return (
            "NAFEMS benchmark FV32, cantilevered tapered membrane: free vibration (natural frequencies), plane "
            "stress, in-plane motion only.\n"
            "Geometry (metres; the STEP file holds the planar face in the plane z = 0): a trapezoidal plate, "
            "10.0 long in x, symmetric about the x-axis. The edge on x = 0 is 5.0 high (y from -2.5 to 2.5), the "
            "edge on x = 10.0 is 1.0 high (y from -0.5 to 0.5); the two long edges are straight. Thickness 0.05.\n"
            + _MATERIAL +
            "Supports: the edge on x = 0 is fully fixed (u_x = u_y = 0). Everything else is free. There is no "
            "load: this is a modal analysis.\n"
            "Report the six lowest natural frequencies in Hz, in ascending order: f1, f2, f3, f4, f5, f6."
        )

    def build_geometry(self) -> None:
        import gmsh

        occ = gmsh.model.occ
        pts = [occ.addPoint(0, -2.5, 0), occ.addPoint(10, -0.5, 0), occ.addPoint(10, 0.5, 0), occ.addPoint(0, 2.5, 0)]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        s = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.synchronize()
        add_group(1, in_box(1, xmax=0.0), "root")
        add_group(1, in_box(1, xmin=10.0), "tip")
        add_group(2, [s], "membrane")

    def bcs(self):
        return [BC(type="fixed", group="root")]

    def references(self) -> list[Reference]:
        src = _source("FV32", "Altair OptiStruct OS-V 0440")
        return [Reference(f"f{k}", f, "published", src, tolerance=0.005,
                          notes="Five significant digits; the fixed solver converges 0.01-0.04 % below.")
                for k, f in enumerate(self.FREQUENCIES, start=1)]


class NafemsFV41(ModalProblem):
    id = "nafems_fv41"
    title = "NAFEMS FV41: free cylinder, axisymmetric vibration (natural frequencies)"
    family = "nafems_fv41"
    tags = ("nafems", "2d", "axisymmetric", "modal", "dynamics", "rigid_body_mode")
    dim = 2
    formulation = "axisymmetric"
    n_modes = 5
    default_params = {**NAFEMS_FV_STEEL, "h0": 0.2}
    FREQUENCIES = (243.53, 377.41, 394.11, 397.72, 405.28)

    def description(self) -> str:
        return (
            "NAFEMS benchmark FV41, free cylinder: axisymmetric free vibration (natural frequencies).\n"
            "Geometry (metres; the STEP file holds the meridian section in the xy-plane with x = r and y = z, "
            "the axis of revolution is x = 0): a hollow circular cylinder with inner radius 1.8, outer radius "
            "2.2 (wall thickness 0.4) and length 10.0 (z from 0 to 10.0).\n"
            + _MATERIAL +
            "Supports: none, the cylinder is free. Only axisymmetric motion (u_r, u_z) is considered. The axial "
            "translation is a rigid-body mode at 0 Hz; do not count it.\n"
            "Report the five lowest natural frequencies of the ELASTIC axisymmetric modes in Hz, in ascending "
            "order: f1, f2, f3, f4, f5."
        )

    def build_geometry(self) -> None:
        import gmsh

        s = gmsh.model.occ.addRectangle(1.8, 0.0, 0.0, 0.4, 10.0)
        gmsh.model.occ.synchronize()
        add_group(1, in_box(1, xmax=1.8), "inner")
        add_group(1, in_box(1, xmin=2.2), "outer")
        add_group(2, [s], "wall")

    def references(self) -> list[Reference]:
        src = _source("FV41", "Altair OptiStruct OS-V 0450 (NAFEMS modes 2-6; mode 1 is the rigid-body mode)")
        return [Reference(f"f{k}", f, "published", src, tolerance=0.005,
                          notes="Five significant digits; the fixed solver converges within 0.03 %.")
                for k, f in enumerate(self.FREQUENCIES, start=1)]


class NafemsFV52(ModalProblem):
    id = "nafems_fv52"
    title = "NAFEMS FV52: simply supported 'solid' square plate, natural frequencies (3D)"
    family = "nafems_fv52"
    tags = ("nafems", "3d", "modal", "dynamics", "rigid_body_mode", "singular")
    dim = 3
    formulation = "3d"
    n_modes = 7
    expected_behavior = "singular"
    default_params = {**NAFEMS_FV_STEEL, "h0": 2.0}
    FREQUENCIES = (45.897, 109.44, 109.44, 167.89, 193.59, 206.19, 206.19)
    IN_PLANE = 5                     # f5 (NAFEMS mode 8) does not load the supports

    def description(self) -> str:
        return (
            "NAFEMS benchmark FV52, simply supported 'solid' square plate: free vibration (natural "
            "frequencies) of a thick plate modelled as a 3D solid.\n"
            "Geometry (metres): a square plate 10.0 x 10.0 in plan (x and y from 0 to 10.0) and 1.0 thick (z "
            "from -0.5 to 0.5).\n"
            + _MATERIAL +
            "Supports: u_z = 0 along the four edges of the bottom face (the lines where the plane z = -0.5 "
            "meets the four side faces). Nothing else is restrained, so the plate keeps three rigid-body modes "
            "at 0 Hz (two in-plane translations and the rotation about z); do not count them.\n"
            "Report the seven lowest natural frequencies of the ELASTIC modes in Hz, in ascending order: "
            "f1, f2, f3, f4, f5, f6, f7."
        )

    def build_geometry(self) -> None:
        import gmsh

        b = gmsh.model.occ.addBox(0.0, 0.0, -0.5, 10.0, 10.0, 1.0)
        gmsh.model.occ.synchronize()
        add_group(1, in_box(1, zmax=-0.5), "support_edges")
        add_group(2, in_box(2, zmax=-0.5), "bottom")
        add_group(2, in_box(2, zmin=0.5), "top")
        add_group(3, [b], "plate")

    def bcs(self):
        return [BC(type="displacement", group="support_edges", components=[2], value=0.0)]

    def study_sizes(self, order: int = 2) -> list[float]:
        return [2.0, 1.5, 1.0]

    def expectations(self) -> dict[str, str]:
        return {f"f{k}": "diverge" for k in range(1, self.n_modes + 1) if k != self.IN_PLANE}

    def references(self) -> list[Reference]:
        src = _source("FV52", "Ansys Verification Manual, Benchmark C5 (NAFEMS modes 4-10; modes 1-3 are rigid)")
        out = []
        for k, f in enumerate(self.FREQUENCIES, start=1):
            if k == self.IN_PLANE:
                out.append(Reference(f"f{k}", f, "published", src, tolerance=0.005,
                                     notes="In-plane mode: independent of the supports, converges (0.001 %)."))
            else:
                out.append(Reference(f"f{k}", f, "published", src, tolerance=0.05,
                                     notes="Bending mode on line supports: a thick-plate value. A solid model has "
                                           "no mesh-converged answer (the line constraint softens with refinement); "
                                           "the tolerance holds for h >= 1 m only."))
        return out


for _p in (NafemsFV32(), NafemsFV41(), NafemsFV52()):
    register(_p)

__all__ = ["NafemsFV32", "NafemsFV41", "NafemsFV52"]
