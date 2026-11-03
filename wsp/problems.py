"""The student's problem pool: a TEST split of the repo's own benchmarks and a TRAIN split of new variants.

TEST is every registered 2D / axisymmetric elasticity benchmark in feacore (the problems the team's
results are reported on), and the student never trains on any of them. TRAIN is new parametric
variants of the same part families (other notch depths and radii, fillet ratios, hole sizes ...),
registered here at import and nowhere else. Four TEST families have no TRAIN variant at all (Cook's
membrane, NAFEMS LE1, NAFEMS LE11 and the re-entrant L-bracket), so the test set measures both
in-family transfer (a new notch radius) and out-of-family transfer (a new kind of part).

Every problem gets an ace.config.TASKS entry (scored QoI and node cap) because ace.field_error builds
its references and the automesher baseline from it. Field-mode scoring does not use the QoI itself.
"""

from __future__ import annotations

from ace import config
from feacore.problems import registry

MM = 1e-3
NODE_CAP_2D = 20000

# --------------------------------------------------------------------------- test: the repo's benchmarks
TEST_OUT_OF_FAMILY = ["cooks_membrane", "nafems_le1", "nafems_le11_axisym", "lbracket_reentrant"]
TEST_IN_FAMILY = [
    "notched_bar_t5_r2", "notched_bar_t5_r4", "notched_bar_t10_r2", "notched_bar_t10_r4",
    "stepped_shaft_axi_Dd1p5_rd0p05", "stepped_shaft_axi_Dd1p5_rd0p1", "stepped_shaft_axi_Dd1p5_rd0p2",
    "stepped_shaft_axi_Dd2_rd0p05", "stepped_shaft_axi_Dd2_rd0p1", "stepped_shaft_axi_Dd2_rd0p2",
    "stepped_shaft_axi_Dd2_sharp",
    "plate_hole_finite_width_d01", "plate_hole_finite_width_d03", "plate_hole_finite_width_d05",
    "clevis_lug_2d_D20", "clevis_lug_2d_D28",
    "pressure_vessel_axi_equal", "pressure_vessel_axi_thin_head",
    "kirsch_exact_traction", "rotating_disk", "timoshenko_cantilever",
    "lame_cylinder_axisym", "lame_cylinder_axisym_open", "lame_cylinder_plane_strain", "lame_sphere_axisym",
]
# Left out of scoring: the automesher's own field error is below the scoring floor (0.5 %, the
# reference's accuracy) or barely above it, so g = e_auto / max(e, 0.5 %) cannot reach 2 and no mesh
# can score near 1x: smooth analytical fields that the default mesh already resolves. Measured with
# python -m wsp.refs: rotating_disk 0.15 %, lame_cylinder_axisym(_open) 0.08 %, train_lame_cyl_ps_b150 0.60 %.
DEGENERATE = {"rotating_disk", "lame_cylinder_axisym", "lame_cylinder_axisym_open", "train_lame_cyl_ps_b150"}
TEST_IN_FAMILY = [p for p in TEST_IN_FAMILY if p not in DEGENERATE]
TEST = TEST_OUT_OF_FAMILY + TEST_IN_FAMILY


# --------------------------------------------------------------------------- train: new variants
def _tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _train_variants() -> list:
    """(id, problem) of every training variant. Parameters avoid the TEST values and stay inside each
    family's valid range (U-notch depth > radius; fillet radius <= the shoulder height (D - d) / 2)."""
    from feacore.problems.analytical.cantilever import TimoshenkoCantilever
    from feacore.problems.analytical.kirsch import KirschExactTraction
    from feacore.problems.analytical.lame import LameCylinderPlaneStrain, LameSphereAxisym
    from feacore.problems.analytical.plate_hole import PlateHoleFiniteWidth
    from feacore.problems.parts.clevis_lug import ClevisLug2D
    from feacore.problems.parts.notched_bar import NotchedBar
    from feacore.problems.parts.pressure_vessel import PressureVesselAxi
    from feacore.problems.parts.stepped_shaft import SteppedShaftAxi

    out = []
    for t, r in ((4, 1), (6, 1.5), (6, 3), (8, 2), (8, 5), (12, 2), (12, 4), (15, 3)):
        out.append(NotchedBar(id=f"train_notched_bar_t{_tag(t)}_r{_tag(r)}", notch_depth=t * MM, notch_radius=r * MM))
    for Dd, rd in ((1.2, 0.05), (1.2, 0.08), (1.3, 0.1), (1.8, 0.03), (1.8, 0.12), (2.5, 0.08), (2.5, 0.25),
                   (3.0, 0.1)):
        out.append(SteppedShaftAxi(id=f"train_stepped_shaft_axi_Dd{_tag(Dd)}_rd{_tag(rd)}", D_over_d=Dd, r_over_d=rd))
    for Dd in (1.5, 3.0):           # sharp shoulders: re-entrant corners (the singular trap)
        out.append(SteppedShaftAxi(id=f"train_stepped_shaft_axi_Dd{_tag(Dd)}_sharp", D_over_d=Dd, r_over_d=0.0))
    for dW in (0.15, 0.2, 0.25, 0.4, 0.45, 0.6):
        # h0 sets the reference meshes (ace.field_error: h0 / 9 for the fine one): scale it with the
        # hole like the registered variants (0.0015 / 0.004 / 0.006 m at d/W = 0.1 / 0.3 / 0.5)
        out.append(PlateHoleFiniteWidth(id=f"train_plate_hole_d{_tag(dW)}", d_over_W=dW, h0=0.0005 + 0.012 * dW))
    for D in (16, 24, 31):
        out.append(ClevisLug2D(id=f"train_clevis_lug_2d_D{D}", hole_diameter=D * MM))
    for wall, head in ((0.02, 0.015), (0.02, 0.03), (0.015, 0.015)):
        out.append(PressureVesselAxi(id=f"train_pressure_vessel_w{_tag(wall * 1e3)}_h{_tag(head * 1e3)}",
                                     wall_thickness=wall, head_thickness=head))
    for a in (0.03, 0.07):
        out.append(KirschExactTraction(id=f"train_kirsch_a{_tag(a * 1e3)}", a=a, h0=0.04 * a / 0.05))
    for a, b in ((0.1, 0.15), (0.1, 0.3)):
        out.append(LameCylinderPlaneStrain(id=f"train_lame_cyl_ps_b{_tag(b * 1e3)}", a=a, b=b,
                                           h0=0.0125 * (b - a) / 0.1))
    out.append(LameSphereAxisym(id="train_lame_sphere_axi_b300", a=0.1, b=0.3, h0=0.025))
    for L, D in ((1.0, 0.1), (1.0, 0.3)):
        out.append(TimoshenkoCantilever(id=f"train_cantilever_D{_tag(D * 1e3)}", L=L, D=D, h0=0.05 * D / 0.2))
    return out


_REGISTERED: list[str] = []


def register_train() -> list[str]:
    """Register the training variants (once) and return their ids."""
    if not _REGISTERED:
        known = {p.id for p in registry.all_problems()}
        for p in _train_variants():
            if p.id not in known:
                registry.register(p)
            _REGISTERED.append(p.id)
    return list(_REGISTERED)


def scored_qoi(problem) -> str:
    """The QoI ace.field_error solves with (any of the problem's QoIs works: field mode scores the
    whole stress field). The first QoI with a reference, else the first QoI."""
    names = [q.name for q in problem.qois()]
    try:
        with_ref = {r.qoi for r in problem.references()}
    except Exception:  # noqa: BLE001  a variant's handbook fit may not cover its parameters
        with_ref = set()
    return next((n for n in names if n in with_ref), names[0])


def ensure_tasks(ids: list[str]) -> None:
    """Give every problem an ace.config.TASKS entry (the dict ace.checks / ace.field_error read)."""
    for pid in ids:
        if pid not in config.TASKS:
            p = registry.get(pid)
            config.TASKS[pid] = {"qoi": scored_qoi(p), "node_cap": NODE_CAP_2D}


def train_ids() -> list[str]:
    ids = [p for p in register_train() if p not in DEGENERATE]
    ensure_tasks(ids)
    return ids


def test_ids() -> list[str]:
    ensure_tasks(TEST)
    return list(TEST)


def family(pid: str) -> str:
    """The part family of a problem (train variants map onto their test family)."""
    register_train()
    return registry.get(pid).family or pid


# Register on import so `registry.get("train_...")` works in every process (workers re-import this).
register_train()
