"""SYNTHETIC records for the tests of the dossier report. NOT RESULTS OF ANY ANALYSIS.

    make_records(project)      write a complete set of stage files into a COPY of the example project

The report has to be developed and tested before the engines exist, and its tests must not run solvers.
This module writes what the pipeline would write: the real context stage, then hand-made geometry data,
findings (the six planted defects of the example and a few more), thirteen load cases, hand calculations
with steps and LaTeX, analysis records with supports, loads, three mesh levels, results in every status
and checks that pass and fail, compliance items with every verdict, more than fifty decisions and
assumptions across the stages, records proposed by an agent, a simple system diagram, placeholder
pictures and reviews in every state.

Every number here is invented. The numbers are physically plausible (the hand calculations use the real
formulas with the real inputs of the example), so that the report reads like a real one, but no solver
made them. Every record says so in its origin (`engine:synthetic`, `rule:synthetic...`) and the load case
and analysis titles end with "(synthetic)". Use it on a temporary copy only (the `workproject` fixture of
conftest.py); never run it on dossier/examples/mov_valve.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml

from dossier import context as context_mod
from dossier import units
from dossier.model import (AnalysisRecord, Artifact, BCItem, CalcInput, CalcOutput, CalcStep, Check, Comparison,
                           ComplianceItem, CriterionRef, CycleUse, HandCalcResult, Link, LoadCase, MeshLevel,
                           Origin, RequirementVerdict, ResultValue)
from dossier.project import Project, digest_of, make_id
from dossier.records import Journal

SYN = "engine:synthetic"
MPA = 1.0e6
MESH = [  # level, h [m], nodes, cells, min quality, representative size [m]
    (1, 0.0040, 21480, 12950, 0.21, 0.00352),
    (2, 0.0028, 54310, 34870, 0.24, 0.00251),
    (3, 0.0020, 118900, 79640, 0.31, 0.00179),
]
PHASES = ["PH-PROOF", "PH-BURST", "PH-LEAK", "PH-TRANSPORT", "PH-CHILL", "PH-PRESTART", "PH-FIRE", "PH-SHUTDOWN"]
COLD = ["CHILL", "PRESTART", "FIRE", "SHUTDOWN"]
MECHANICAL = ("pressure", "force", "moment", "acceleration", "displacement")
THERMAL = ("fluid_temperature", "temperature", "heat_flux")


def make_records(project: Project, reviews: bool = True, pictures: bool = True) -> None:
    """Write synthetic stage files for every stage into `project` (a temporary copy of the example)."""
    if "examples" in project.root.parts and "dossier" in project.root.parts:
        raise ValueError("make_records writes synthetic records: use a temporary copy of the example project")
    ctx = context_mod.run_stage(project)
    geo, jg = _geometry(project, pictures)
    project.write_stage("geometry", geometry=geo, findings=jg.findings, decisions=jg.decisions,
                        assumptions=jg.assumptions)
    jv = _validate(project, ctx)
    project.write_stage("validate", findings=jv.findings, decisions=jv.decisions, assumptions=jv.assumptions)
    (project.out / "system.svg").write_text(_diagram(project, ctx))
    load_cases, jp = _load_cases(project, ctx)
    hand_calcs = _hand_calcs(project, ctx, geo, load_cases)
    for lc in load_cases:
        lc.hand_calcs = [h.id for h in hand_calcs if h.load_case == lc.id]
        keys = [project.requirement(c.requirement).criteria[c.criterion].key() for c in lc.criteria]
        keys += [c.result_key for h in hand_calcs if h.load_case == lc.id for c in h.comparisons]
        lc.result_keys = list(dict.fromkeys(keys))
    project.write_stage("plan", load_cases=load_cases, hand_calcs=hand_calcs, findings=jp.findings,
                        decisions=jp.decisions, assumptions=jp.assumptions)
    analyses = _analyses(project, ctx, load_cases, hand_calcs, pictures)
    for lc_id, (rec, j) in analyses.items():
        project.write_stage("analyze", project.analysis_dir(lc_id) / "analysis.json",
                            options={"quick": False, "levels": 3, "node_cap": 120000}, analyses=[rec],
                            findings=j.findings, decisions=j.decisions, assumptions=j.assumptions)
    items, verdicts, ja = _assess(project, ctx, {k: v[0] for k, v in analyses.items()}, hand_calcs)
    project.write_stage("assess", compliance=items, verdicts=verdicts, findings=ja.findings,
                        decisions=ja.decisions, assumptions=ja.assumptions)
    jagent = _agent()
    (project.root / "agent").mkdir(exist_ok=True)
    project.write_stage("agent", project.root / "agent" / "analyst.json", findings=jagent.findings,
                        decisions=jagent.decisions, assumptions=jagent.assumptions)
    if reviews:
        _reviews(project)


# =========================================================================== geometry


_FEATURES = {   # id: (kind, faces, radius, axis, normal, centroid, area, length)
    "FT-INLET-BORE": ("cylinder", [12], 0.0125, [1, 0, 0], None, [-0.0505, 0, 0], 3.69e-3, 0.047),
    "FT-OUTLET-BORE": ("cylinder", [13], 0.0125, [1, 0, 0], None, [0.0505, 0, 0], 3.69e-3, 0.047),
    "FT-SEAT-IN-WALL": ("cylinder", [14], 0.016, [1, 0, 0], None, [-0.0245, 0, 0], 5.03e-4, 0.005),
    "FT-SEAT-OUT-WALL": ("cylinder", [15], 0.016, [1, 0, 0], None, [0.0245, 0, 0], 5.03e-4, 0.005),
    "FT-SEAT-IN-SHOULDER": ("plane", [16], None, None, [1, 0, 0], [-0.027, 0, 0], 3.13e-4, None),
    "FT-SEAT-OUT-SHOULDER": ("plane", [17], None, None, [-1, 0, 0], [0.027, 0, 0], 3.13e-4, None),
    "FT-CAVITY-WALL": ("cylinder", [18], 0.022, [0, 0, 1], None, [0, 0, 0.006], 6.13e-3, 0.056),
    "FT-CAVITY-FLOOR": ("mixed", [19, 20], None, None, [0, 0, 1], [0, 0, -0.0221], 1.55e-3, None),
    "FT-CAVITY-ROOF": ("plane", [21], None, None, [0, 0, -1], [0, 0, 0.034], 1.32e-3, None),
    "FT-STEM-BORE": ("cylinder", [22], 0.008, [0, 0, 1], None, [0, 0, 0.079], 4.52e-3, 0.090),
    "FT-INLET-FLANGE-FACE": ("plane", [1], None, None, [-1, 0, 0], [-0.075, 0, 0], 5.67e-3, None),
    "FT-OUTLET-FLANGE-FACE": ("plane", [2], None, None, [1, 0, 0], [0.075, 0, 0], 5.67e-3, None),
    "FT-INLET-BOLT-HOLES": ("cylinder", [23, 24, 25, 26, 27, 28], 0.0033, [1, 0, 0], None, [-0.069, 0, 0], 1.49e-3, 0.012),
    "FT-OUTLET-BOLT-HOLES": ("cylinder", [29, 30, 31, 32, 33, 34], 0.0033, [1, 0, 0], None, [0.069, 0, 0], 1.49e-3, 0.012),
    "FT-PAD-TOP": ("plane", [3], None, None, [0, 0, 1], [0, 0, 0.124], 4.18e-3, None),
    "FT-PAD-HOLES": ("cylinder", [35, 36, 37, 38], 0.00275, [0, 0, 1], None, [0, 0, 0.120], 5.53e-4, 0.008),
    "FT-NECK": ("cylinder", [39], 0.012, [0, 0, 1], None, [0, 0, 0.080], 5.28e-3, 0.070),
    "FT-NECK-ROOT-FILLET": ("torus", [40], 0.003, [0, 0, 1], None, [0, 0, 0.043], 3.9e-4, None),
}


def _geometry(project: Project, pictures: bool) -> tuple[dict, Journal]:
    comp = project.subject
    cad = project.cad_path()
    features, faces = {}, {}
    for fid, (kind, tags, radius, axis, normal, centroid, area, length) in _FEATURES.items():
        features[fid] = {"faces": tags, "area": area, "centroid": centroid, "kind": kind, "radius": radius,
                         "diameter": 2 * radius if radius else None, "axis": axis, "normal": normal,
                         "length": length, "description": comp.features[fid].description}
        for t in tags:
            faces[str(t)] = {"kind": kind if kind != "mixed" else ("plane" if t == 19 else "torus"),
                             "area": area / len(tags), "centroid": centroid, "normal": normal, "radius": radius,
                             "axis": axis, "features": [fid], "roles": []}
    used = {t for f in features.values() for t in f["faces"]}
    external = [t for t in range(1, 48) if t not in used]
    for t in external:
        faces[str(t)] = {"kind": "cylinder" if t % 2 else "plane", "area": 1.1e-3, "centroid": [0.0, 0.0, 0.01 * (t % 7)],
                         "normal": None, "radius": None, "axis": None, "features": [], "roles": ["INT-ENV:external"]}
    interfaces = {}
    for it in comp.interfaces:
        roles = {}
        for role, feats in it.features.items():
            tags = external if feats == ["*"] else sorted({t for f in feats for t in features[f]["faces"]})
            roles[role] = tags
            for t in tags:
                tag = f"{it.id}:{role}"
                if tag not in faces[str(t)]["roles"]:
                    faces[str(t)]["roles"].append(tag)
        interfaces[it.id] = roles
    scls = {}
    for s in comp.scls:
        length = math.dist(s.start, s.end)
        scls[s.id] = {"start": list(s.start), "end": list(s.end), "length": length, "ok": True, "message": ""}
    scls["SCL-NECK-ROOT"]["message"] = "The end point is 0.02 mm from the surface. The line was moved to the surface."
    pics = []
    if pictures:
        (project.out / "geometry").mkdir(parents=True, exist_ok=True)
        _picture(project.out / "geometry" / "overview.png", "Valve body, section in the x-z plane", None)
        _picture(project.out / "geometry" / "interfaces.png", "Interfaces and features of the valve body", "features")
        pics = [{"path": "geometry/overview.png", "caption": "Section of the valve body in the x-z plane (synthetic picture)."},
                {"path": "geometry/interfaces.png", "caption": "The interfaces of the valve body with their features (synthetic picture)."}]
    geo = {
        "component": comp.id,
        "cad": {"file": comp.cad.file, "sha256": hashlib.sha256(cad.read_bytes()).hexdigest(), "volume": 2.862e-4,
                "mass": 2.862e-4 * 7990.0, "bbox": [[-0.075, -0.045, -0.030], [0.075, 0.045, 0.124]], "n_faces": 47},
        "features": features, "interfaces": interfaces, "faces": faces,
        "edges": {"reentrant": [
            {"tag": 31, "faces": [15, 17], "angle_deg": 270.0, "length": 0.1005, "midpoint": [0.027, 0.016, 0.0]},
            {"tag": 29, "faces": [14, 16], "angle_deg": 270.0, "length": 0.1005, "midpoint": [-0.027, 0.016, 0.0]},
            {"tag": 44, "faces": [18, 21], "angle_deg": 270.0, "length": 0.1382, "midpoint": [0.022, 0.0, 0.034]},
            {"tag": 47, "faces": [21, 22], "angle_deg": 270.0, "length": 0.0503, "midpoint": [0.008, 0.0, 0.034]}]},
        "scls": scls, "pictures": pics, "problems": [],
    }
    j = Journal("geometry", "engine:synthetic.geometry")
    j.decide("CMP-MOV-BODY", "length-unit", "Length unit of the CAD file", "Read the CAD file in metres.",
             "The component file gives the length unit m. The largest dimension of the part is then 154 mm. "
             "The drawing gives the same dimension.", basis="input", links=[("source", "doc:MOV-100-001 rev A")])
    j.decide("INT-ENV", "external-faces", "External faces of the outer surfaces",
             "Use the 15 faces that no other interface uses as the external faces of INT-ENV.",
             "The interface INT-ENV gives the sign * for its external role. The rule gives every face that has no other role.",
             alternatives=[("List the external faces by tag.", "A list of tags does not survive a new export of the CAD file.")],
             by="rule:synthetic.geometry.external")
    j.assume("CMP-MOV-BODY", "nominal-geometry", "Nominal dimensions",
             "The CAD file gives the nominal dimensions of the part.",
             "The drawing MOV-100-001 rev A gives no tolerance model for the analysis.",
             "A wall at its smallest thickness has a higher stress. The margins decrease by about the tolerance ratio of the wall.",
             verify_by="Compare the wall thicknesses of the drawing with their tolerances.", conservative=False,
             links=[("source", "doc:MOV-100-001 rev A")])
    j.find("geometry.scl_surface", "minor", "model", "SCL-NECK-ROOT", "end-off-surface",
           "The stress classification line ends near the surface, not on it",
           "The end point of SCL-NECK-ROOT is 0.02 mm from the outer surface of the neck.",
           "The component file gives the end point [0, -0.012, 0.052] m. The nearest point of the face FT-NECK is 0.02 mm away. "
           "The geometry stage moved the end point to the surface.",
           suggestion="Correct the end point in the component file.", values={"distance": 2.0e-5},
           links=[("about", "FT-NECK")])
    return geo, j


def _picture(path: Path, title: str, mode: str | None) -> None:
    """A schematic section of the valve body. `mode`: None (plain), "features", or a field name (colour)."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    fig = Figure(figsize=(6.0, 4.4), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
    solid = [(-75, -45, 12, 90), (63, -45, 12, 90), (-63, -20, 35, 40), (28, -20, 35, 40), (-28, -30, 56, 72),
             (-12, 42, 24, 70), (-38, 112, 76, 12)]
    void = [(-75, -12.5, 150, 25), (-22, -22, 44, 56), (-8, 34, 16, 90), (-27, -16, 5, 32), (22, -16, 5, 32)]
    if mode and mode != "features":
        x, z = np.meshgrid(np.linspace(-80, 80, 320), np.linspace(-50, 130, 360))
        if mode == "temperature":
            field = 90 + 172 * np.clip((z - 30) / 94, 0, 1) ** 1.6
            cmap, label = "viridis", "temperature, K"
        else:
            field = 30 + 110 * np.exp(-((x - 12) ** 2 + (z - 43) ** 2) / 500) + 60 * np.exp(-((x - 27) ** 2 + (z + 14) ** 2) / 90)
            cmap, label = "magma", "von Mises stress, MPa"
        mask = np.zeros_like(x, dtype=bool)
        for (x0, z0, w, h) in solid:
            mask |= (x >= x0) & (x <= x0 + w) & (z >= z0) & (z <= z0 + h)
        for (x0, z0, w, h) in void:
            mask &= ~((x >= x0) & (x <= x0 + w) & (z >= z0) & (z <= z0 + h))
        im = ax.imshow(np.ma.masked_where(~mask, field), extent=(-80, 80, -50, 130), origin="lower", cmap=cmap)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        for (x0, z0, w, h) in solid:
            ax.add_patch(Rectangle((x0, z0), w, h, facecolor="#b9c2c9", edgecolor="none"))
        for (x0, z0, w, h) in void:
            ax.add_patch(Rectangle((x0, z0), w, h, facecolor="white", edgecolor="none"))
        if mode == "features":
            for (x0, z0, w, h, c, name) in [(-76.5, -45, 3, 90, "#2a78d6", "INT-INLET"), (73.5, -45, 3, 90, "#eb6834", "INT-OUTLET"),
                                             (-38, 122.5, 76, 3, "#1baf7a", "INT-ACT"), (-8, 34, 16, 90, "#6250d6", "INT-INTERNALS")]:
                ax.add_patch(Rectangle((x0, z0), w, h, facecolor=c, edgecolor="none", alpha=0.85))
                ax.text(x0 + w / 2, z0 + h + 3 if name != "INT-INTERNALS" else 20, name, fontsize=7, ha="center", color=c)
    ax.set_xlim(-90, 100)
    ax.set_ylim(-52, 140)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.02, 0.94, title, fontsize=10, weight="bold")
    fig.text(0.98, 0.94, "SYNTHETIC PICTURE", fontsize=8, color="#b4491c", ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")


def _diagram(project: Project, ctx) -> str:
    """The system diagram: the real renderer when it is installed, else a simple grid of boxes with the same
    conventions (one group per element and per connection, `id` = the record id, `data-*` attributes)."""
    try:
        from dossier.system.diagram import render
        return render(project, ctx)
    except Exception:  # noqa: BLE001  the renderer of another engine is missing or fails: draw a stand-in
        pass
    s = project.system
    groups: dict[str, list] = {}
    for e in s.elements:
        groups.setdefault(e.group or "System", []).append(e)
    pos, parts = {}, []
    w, h, gx, gy = 150, 46, 40, 44
    for r, (gname, els) in enumerate(groups.items()):
        y = 50 + r * (h + gy + 20)
        parts.append(f'<text x="8" y="{y - 12}" font-size="11" fill="currentColor" opacity="0.7">{gname}</text>')
        for c, e in enumerate(els):
            pos[e.id] = (20 + c * (w + gx), y)
    lines = []
    for c in s.connections:
        (x1, y1), (x2, y2) = pos[c.from_], pos[c.to]
        n = len({x.id for x in c.effects})
        lines.append(f'<g id="{c.id}" class="connection" data-kind="connection" data-from="{c.from_}" data-to="{c.to}" '
                     f'data-medium="{c.medium}" data-effects="{n}"><title>{c.name}</title>'
                     f'<line x1="{x1 + w / 2}" y1="{y1 + h / 2}" x2="{x2 + w / 2}" y2="{y2 + h / 2}" stroke="currentColor" '
                     f'stroke-width="{2.5 if n else 1.2}" opacity="{0.8 if n else 0.4}"/></g>')
    boxes = []
    for e in s.elements:
        x, y = pos[e.id]
        subject = e.component == project.subject.id
        boxes.append(f'<g id="{e.id}" class="element{" subject" if subject else ""}" data-kind="element" '
                     f'data-group="{e.group}"><title>{e.name}</title>'
                     f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{"#2a78d6" if subject else "#8a9299"}" '
                     f'fill-opacity="{0.25 if subject else 0.14}" stroke="currentColor" stroke-width="{2 if subject else 1}"/>'
                     f'<text x="{x + w / 2}" y="{y + 20}" font-size="11" text-anchor="middle" fill="currentColor">{e.name[:24]}</text>'
                     f'<text x="{x + w / 2}" y="{y + 35}" font-size="9" text-anchor="middle" fill="currentColor" opacity="0.7">{e.id}</text></g>')
    width = 20 + max(len(v) for v in groups.values()) * (w + gx)
    height = 50 + len(groups) * (h + gy + 20)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="sans-serif" role="img" aria-label="System diagram (synthetic stand-in)">'
            + "".join(parts) + "".join(lines) + "".join(boxes) + "</svg>")


# =========================================================================== validate


def _validate(project: Project, ctx) -> Journal:
    j = Journal("validate", "engine:synthetic.validate")
    p = ctx.params
    j.find("req.meop_envelope", "error", "consistency", "REQ-MOV-001", "meop-envelope",
           "The MEOP is lower than the largest operating pressure",
           "The requirement gives an MEOP of 10.5 MPa. The system context gives 10.72 MPa at the inlet during shutdown. "
           "The requirement value is too low.",
           "The largest pressure of the operation phases is the effect EF-IN-P-SHUTDOWN in the phase PH-SHUTDOWN. "
           "Its value is p_max_inlet = p_pump_max + dp_surge = 10.72 MPa. The closure surge dp_surge is 0.32 MPa.",
           suggestion="Increase the MEOP to 10.72 MPa or more. As an alternative, decrease the closure surge.",
           values={"meop": p["meop"].si, "largest_pressure": p["p_max_inlet"].si},
           links=[("uses", "EF-IN-P-SHUTDOWN"), ("uses", "p_max_inlet"), ("uses", "meop"), ("uses", "PH-SHUTDOWN")])
    j.find("req.factor_policy", "major", "policy", "REQ-MOV-004", "yield-factor",
           "The yield factor is lower than the factor of the policy",
           "The requirement gives a yield factor of 1.1. The policy asks for 1.25.",
           "The parameter yield_factor of REQ-MOV-004 has the role yield_factor. The policy factor 'yield' is 1.25 for limit loads.",
           suggestion="Increase the yield factor of the requirement to 1.25, or record a waiver of the policy.",
           values={"yield_factor": 1.1, "policy": 1.25}, links=[("uses", "yield_factor"), ("uses", "POLICY")])
    j.find("req.temperature_range", "error", "consistency", "REQ-MOV-005", "min-temperature",
           "The fluid temperature range does not include the LOX temperature",
           "The requirement gives a lowest fluid temperature of 233 K. The system context gives LOX at 90 K.",
           "The effects EF-IN-T-CHILL, EF-IN-T-PRESTART, EF-IN-T-FIRE and EF-IN-T-SHUTDOWN have a fluid temperature of t_lox = 90 K. "
           "The parameter t_fluid_min has the role min_temperature.",
           suggestion="Decrease the lowest fluid temperature of the requirement to 90 K or less.",
           values={"t_fluid_min": 233.0, "lowest_fluid_temperature": 90.0},
           links=[("uses", "t_fluid_min"), ("uses", "t_lox"), ("uses", "EF-IN-T-FIRE")])
    j.find("req.param_check", "major", "consistency", "REQ-MOV-006", "design-cycles",
           "The design number of cycles is lower than the pressure cycles of the engine life",
           "The requirement gives 200 design cycles. The system context gives 267 pressure cycles.",
           "The check of the parameter design_cycles is: design_cycles >= n_pressure_cycles. "
           "The context gives n_pressure_cycles = n_runs + n_proof + n_checkout = 65 + 2 + 200 = 267.",
           suggestion="Increase the design number of cycles to 267 or more.",
           values={"design_cycles": 200.0, "n_pressure_cycles": 267.0},
           links=[("uses", "design_cycles"), ("uses", "n_pressure_cycles")])
    j.find("req.verifiable", "major", "quality", "REQ-MOV-009", "not-verifiable",
           "The requirement has no criterion that analysis can verify",
           "The requirement uses the words 'sufficiently stiff'. It gives no quantity and no limit.",
           "The requirement has no criterion and no parameter. The word 'sufficiently' has no measure. "
           "The requirement REQ-MOV-010 gives a roundness limit for one phase only.",
           suggestion="Give a limit for the deformation of each seat pocket and the phases it applies to.",
           links=[("about", "REQ-MOV-010")])
    j.find("req.coverage", "major", "coverage", "PH-TRANSPORT", "uncovered",
           "No requirement applies to the transport phase",
           "The phase PH-TRANSPORT puts loads on the valve body. No requirement applies to this phase.",
           "The phase has the category handling. The effects EF-ACT-F-TRANSPORT, EF-ACT-M-TRANSPORT and EF-ENV-A-TRANSPORT act on the "
           "valve body. The strength requirement REQ-MOV-004 applies to the category operation only.",
           suggestion="Add a strength requirement for the handling phases, or include the category handling in REQ-MOV-004.",
           links=[("uses", "EF-ACT-F-TRANSPORT"), ("uses", "EF-ACT-M-TRANSPORT"), ("uses", "EF-ENV-A-TRANSPORT"),
                  ("about", "REQ-MOV-004")])
    j.find("sys.assumed_parameter", "minor", "context", "h_stagnant", "assumed",
           "A heat transfer coefficient has no source",
           "The parameter h_stagnant has the basis 'assumed' and no source document.",
           "The thermal load cases of PH-PRESTART and PH-SHUTDOWN use h_stagnant = 400 W/(m²·K) on the wetted faces.",
           suggestion="Give a source for the value, or give its range.", values={"h_stagnant": 400.0},
           links=[("uses", "EF-IN-T-PRESTART"), ("uses", "EF-IN-T-SHUTDOWN")])
    j.find("sys.element_description", "info", "quality", "EL-CHAMBER", "no-description",
           "An element has no description",
           "The element EL-CHAMBER has no description.",
           "The element has no connection to the valve body. The missing description has no effect on the analysis.")
    # ---- decisions and assumptions of the context and of the validation
    for name, what in [("dp_valve_est", "The pressure at the valve outlet during firing changes. The change is small in comparison with 10.4 MPa."),
                       ("h_stagnant", "The temperature field of the closed valve changes. The thermal stress changes with it."),
                       ("h_chill", "The temperature field during chill-down changes. The thermal stress changes with it."),
                       ("h_purge", "The temperature of the outlet nozzle changes when the valve is closed."),
                       ("h_amb", "The temperature of the neck and of the actuator mounting face changes.")]:
        j.assume(name, "assumed", f"Assumed value of {name}",
                 f"The value {p[name].display} of the parameter {name} is correct.",
                 "The system context gives the value with the basis 'assumed'. No source document gives it.",
                 what, verify_by="Compare the value with a correlation, a test or the data of the supplier.",
                 conservative=None, stage="context", by="rule:synthetic.context.assumed")
    j.decide("PH-LEAK", "covered", "Coverage of the leak check",
             "Treat the phase PH-LEAK as covered by the requirement REQ-MOV-002.",
             "The loads of PH-LEAK are 0.667 times the loads of PH-PROOF, with the same supports. "
             "A positive margin of the proof test is then a positive margin of the leak check.",
             alternatives=[("Report the phase as not covered.", "The proof test has the same load pattern at a higher level.")],
             basis="calculation", by="rule:synthetic.req.coverage", links=[("uses", "PH-PROOF"), ("about", "REQ-MOV-002")])
    for rid in ("REQ-MOV-001", "REQ-MOV-005"):
        j.decide(rid, "verification", f"Verification method of {rid}",
                 f"Do not verify {rid} by analysis.",
                 "The requirement states a number and has no criterion. Its verification method is review. "
                 "The validation compares the number with the system context.",
                 by="rule:synthetic.req.verification", basis="input")
    return j


# =========================================================================== plan


def _load_cases(project: Project, ctx) -> tuple[list[LoadCase], Journal]:
    comp = project.subject
    j = Journal("plan", "engine:synthetic.loadcases")
    origin = Origin(by=SYN, stage="plan")
    cases: list[LoadCase] = []

    def phase_loads(ph: str, kinds: tuple[str, ...]):
        items = ctx.loads(project, None, ph)
        return [x for x in items if x.kind in kinds], [x.interface for x in items if x.kind == "support" and x.interface]

    def crit(*pairs):
        return [CriterionRef(requirement=r, criterion=i) for r, i in pairs]

    loads, sup = phase_loads("PH-PROOF", MECHANICAL)
    cases.append(LoadCase(id="LC-PROOF", title="Proof pressure test (synthetic)", analysis_type="static_structural",
                          component=comp.id, phase="PH-PROOF", loads=loads, supports=sup, criteria=crit(("REQ-MOV-002", 0)),
                          statement="The valve body has the proof pressure of 15.75 MPa at ambient temperature. "
                                    "The blind covers pull on the flanges and on the mounting face. No interface holds the part.",
                          origin=origin))
    for name, scale, crits, text in [("BURST", 2.5 / 1.5, crit(("REQ-MOV-003", 0)), "the burst pressure of 26.25 MPa"),
                                     ("LEAK", 1.0 / 1.5, [], "the MEOP of 10.5 MPa")]:
        loads, sup = phase_loads(f"PH-{name}", MECHANICAL)
        cases.append(LoadCase(id=f"LC-{name}", title=f"{project.phase('PH-' + name).name} (synthetic)",
                              analysis_type="static_structural", component=comp.id, phase=f"PH-{name}", loads=loads,
                              supports=sup, criteria=crits, scaled_from="LC-PROOF", scale=scale, status="scaled",
                              statement=f"The valve body has {text} at ambient temperature. "
                                        f"Every load is {scale:.3f} times the load of LC-PROOF. The results come from LC-PROOF by scale.",
                              origin=origin))
        j.decide(f"LC-{name}", "scaled", f"Scale of the load case LC-{name}",
                 f"Scale the results of LC-PROOF by {scale:.3f} for LC-{name}. Do not solve LC-{name}.",
                 f"Every load of PH-{name} is {scale:.3f} times the load of PH-PROOF. The supports are the same. "
                 "A linear static analysis then gives the same result by scale.",
                 alternatives=[(f"Solve LC-{name} with its own mesh study.", "The result is the same and the solve takes time."),
                               ("Use a nonlinear analysis.", "The criterion compares an elastic section stress with a strength.")],
                 basis="calculation", by="rule:synthetic.loadcases.proportional", links=[("uses", "LC-PROOF")])
    j.decide("LC-PROOF", "select", "Load case of the proof test",
             "Solve the phase PH-PROOF as the static load case LC-PROOF.",
             "The requirement REQ-MOV-002 applies to PH-PROOF. The temperature is ambient, so a thermal analysis is not necessary. "
             "The load cases LC-BURST and LC-LEAK come from it by scale.",
             by="rule:synthetic.loadcases.phase", links=[("about", "REQ-MOV-002")])
    for name in COLD:
        ph = f"PH-{name}"
        tl, sup = phase_loads(ph, THERMAL)
        ml, _ = phase_loads(ph, MECHANICAL)
        cases.append(LoadCase(id=f"LC-TH-{name}", title=f"Temperature field: {project.phase(ph).name} (synthetic)",
                              analysis_type="thermal_steady", component=comp.id, phase=ph, loads=tl, supports=[],
                              criteria=crit(("REQ-MOV-007", 0)) if name == "FIRE" else [],
                              statement=f"The steady temperature field of the valve body in the phase {ph}. "
                                        "The fluid cools the wetted faces. The ambient air heats the outer surfaces.",
                              origin=origin))
        crits = crit(("REQ-MOV-004", 0), ("REQ-MOV-004", 1)) + (crit(("REQ-MOV-010", 0)) if name == "PRESTART" else [])
        cases.append(LoadCase(id=f"LC-{name}", title=f"{project.phase(ph).name} (synthetic)", analysis_type="thermo_structural",
                              component=comp.id, phase=ph, loads=ml, supports=sup, criteria=crits, uses=[f"LC-TH-{name}"],
                              statement=f"The limit loads of the phase {ph} act together with the temperature field of LC-TH-{name}. "
                                        "The pump discharge flange holds the inlet flange.",
                              origin=origin))
        j.decide(f"LC-{name}", "thermal", f"Thermal load case for {ph}",
                 f"Add the thermal load case LC-TH-{name} and use its temperature field in LC-{name}.",
                 "The fluid temperature is 90 K. The reference temperature of the part is 293.15 K. "
                 "The difference is more than the policy limit of 5 K.",
                 alternatives=[("Analyse the mechanical loads only.", "The thermal stress is not small in comparison with the pressure stress.")],
                 by="rule:synthetic.loadcases.thermal", links=[("uses", f"LC-TH-{name}"), ("uses", "POLICY")])
    cases.append(LoadCase(id="LC-FATIGUE", title="Fatigue of the duty cycles (synthetic)", analysis_type="fatigue",
                          component=comp.id, criteria=crit(("REQ-MOV-006", 0)),
                          uses=["LC-PROOF", "LC-LEAK", "LC-CHILL", "LC-PRESTART", "LC-FIRE", "LC-SHUTDOWN"],
                          cycles=[CycleUse(duty_cycle="DC-RUN", count=65, states=["LC-CHILL", "LC-PRESTART", "LC-FIRE", "LC-SHUTDOWN"]),
                                  CycleUse(duty_cycle="DC-PROOF", count=2, states=["LC-PROOF"]),
                                  CycleUse(duty_cycle="DC-LEAK", count=200, states=["LC-LEAK"])],
                          statement="The cumulative fatigue damage of the three duty cycles of the system context. "
                                    "Each duty cycle starts and ends in the unloaded state.",
                          origin=origin))
    fl = [x for x in ctx.loads(project, None, "PH-FIRE") if x.kind == "mass_flow"]
    cases.append(LoadCase(id="LC-FLOW-FIRE", title="Flow through the open valve (synthetic)", analysis_type="flow",
                          component=comp.id, phase="PH-FIRE", loads=fl, criteria=crit(("REQ-MOV-008", 0)),
                          statement="LOX flows through the open valve at the rated mass flow of 5.3 kg/s. "
                                    "The result is the pressure drop between the inlet and the outlet.",
                          origin=origin))
    j.decide("LC-FATIGUE", "states", "States of the duty cycles",
             "Use the load cases LC-CHILL, LC-PRESTART, LC-FIRE and LC-SHUTDOWN as the states of the duty cycle DC-RUN.",
             "The duty cycle DC-RUN has the sequence PH-CHILL, PH-PRESTART, PH-FIRE, PH-SHUTDOWN. Each phase has one load case. "
             "The duty cycles DC-PROOF and DC-LEAK have one state each.",
             by="rule:synthetic.loadcases.cycles", links=[("uses", "DC-RUN"), ("uses", "DC-PROOF"), ("uses", "DC-LEAK")])
    j.decide("LC-FLOW-FIRE", "select", "Load case of the flow analysis",
             "Analyse the flow in the phase PH-FIRE as the load case LC-FLOW-FIRE.",
             "The requirement REQ-MOV-008 gives a limit for the pressure drop at the rated mass flow. "
             "The phase PH-FIRE has the mass flow effect EF-IN-FLOW-FIRE.",
             by="rule:synthetic.loadcases.phase", links=[("about", "REQ-MOV-008"), ("uses", "EF-IN-FLOW-FIRE")])
    j.decide("PH-TRANSPORT", "no-load-case", "No load case for the transport phase",
             "Do not analyse the phase PH-TRANSPORT.",
             "No requirement applies to PH-TRANSPORT. Without a criterion, an analysis gives no margin. "
             "The finding FND-PH-TRANSPORT-UNCOVERED reports the gap.",
             alternatives=[("Analyse the phase with the criteria of REQ-MOV-004.",
                            "REQ-MOV-004 applies to operation phases only. The tool does not change the scope of a requirement.")],
             needs_review=True, by="rule:synthetic.loadcases.coverage", links=[("about", "FND-PH-TRANSPORT-UNCOVERED")])
    j.decide("LC-FIRE", "lateral-direction", "Direction of the lateral acceleration",
             "Apply the lateral acceleration in the +y direction only.",
             "The lateral direction that bends the neck about the x axis is the worst case for the neck root. "
             "The responsible engineer selected this direction.",
             alternatives=[("Analyse four lateral directions.",
                            "The body is almost symmetric about the x-z plane. The +y direction and the -y direction give the same stress.")],
             basis="engineering_judgement", needs_review=True, by="human:A. Example", links=[("uses", "EF-ENV-A-FIRE")])
    j.assume("LC-PROOF", "fixture", "The test fixture puts no load on the valve",
             "The test fixture puts no load on the valve body.",
             "The description of the phase PH-PROOF says that a fixture holds the valve without load.",
             "A fixture that clamps the flanges adds bending stress to the nozzles.",
             verify_by="Examine the test procedure MOV-ATP-001 rev A.", conservative=None,
             links=[("source", "doc:MOV-ATP-001 rev A")])
    j.assume("DC-RUN", "unloaded-state", "The part returns to the unloaded state",
             "The valve body returns to the unloaded state at ambient temperature after each engine run.",
             "The system context gives the duty cycle DC-RUN from ambient conditions back to ambient conditions.",
             "A residual stress after the first run changes the mean stress of the later cycles.",
             conservative=None, by="rule:synthetic.loadcases.cycles")
    j.assume("LC-FATIGUE", "sequence", "Order of the states in a duty cycle",
             "The order of the states in a duty cycle has no effect on the damage.",
             "The stress-life method with a linear damage rule does not use the order of the cycles.",
             "The damage of a real sequence can be higher or lower than the linear rule gives.",
             verify_by="Compare the result with a test of the duty cycle.", conservative=None)
    return cases, j


def _hand_calcs(project: Project, ctx, geo: dict, load_cases: list[LoadCase]) -> list[HandCalcResult]:
    comp = project.subject
    specs = {h.id: h for h in comp.hand_calcs}
    by_id = {lc.id: lc for lc in load_cases}
    out: list[HandCalcResult] = []
    origin = Origin(by="engine:synthetic.handcalc", stage="plan")

    def inputs_of(spec, lc):
        fns = ctx.functions(lc.loads, geo)
        names = ctx.names()
        vals, rows = {}, []
        for name, v in spec.inputs.items():
            q = v.quantity(names, fns)
            vals[name] = units.si(q)
            src = f"= {v.expr}" if not v.is_literal else "The component file gives the value."
            rows.append(CalcInput(name=name, si=float(vals[name]), unit=units.si_unit(q), display=units.display(q), source=src))
        return vals, rows

    def output(name, si, unit, desc):
        return CalcOutput(name=name, si=si, unit=unit, display=units.display(units.from_si(si, unit)), description=desc)

    def record(spec, lc, model, reference, rows, outs, steps, assumptions, warnings, fe: dict):
        comps = []
        vals = {o.name: o.si for o in outs}
        for c in spec.compare:
            fe_v = fe.get(c.with_)
            hand = vals[c.output]
            ratio = (fe_v / hand) if fe_v is not None and c.band is not None else None
            tol = float(units.si(c.tolerance.quantity())) if c.tolerance is not None else None
            agree = None
            if fe_v is not None:
                agree = (c.band[0] <= ratio <= c.band[1]) if c.band is not None else abs(fe_v - hand) <= tol
            comps.append(Comparison(output=c.output, result_key=c.with_, hand_value=hand, fe_value=fe_v, ratio=ratio,
                                    band=c.band or (None if tol is not None else project.policy.handcalc_band),
                                    tolerance_si=tol, agree=agree, why=c.why))
        out.append(HandCalcResult(id=make_id("HCR", spec.id[3:], lc.id[3:]), title=f"{spec.title}, {lc.id} (synthetic)",
                                  spec=spec.id, model=model, load_case=lc.id, reference=reference, inputs=rows,
                                  outputs=outs, steps=steps, assumptions=assumptions, warnings=warnings, comparisons=comps,
                                  links=[Link(rel="uses", to=x.effect) for x in lc.loads if x.kind == "pressure"][:1],
                                  origin=origin))

    def mpa(x):
        return f"{x / MPA:.2f} MPa"

    scale = {"LC-PROOF": 1.0, "LC-BURST": 2.5 / 1.5, "LC-LEAK": 1.0 / 1.5}
    for lc_id, s in scale.items():
        lc = by_id[lc_id]
        for sid, key, fe_proof in [("HC-SHELL-HOOP", "max_principal.membrane@SCL-SHELL", 63.9 * MPA),
                                   ("HC-NOZZLE-HOOP", "max_principal.membrane@SCL-NOZZLE-OUT", 24.1 * MPA)]:
            spec = specs[sid]
            v, rows = inputs_of(spec, lc)
            p_, ri, ro = v["p"], v["r_i"], v["r_o"]
            mean = p_ * ri / (ro - ri)
            inner = p_ * (ro**2 + ri**2) / (ro**2 - ri**2)
            outer = 2 * p_ * ri**2 / (ro**2 - ri**2)
            record(spec, lc, spec.model, "Lamé equations of a thick-walled cylinder with internal pressure.",
                   rows,
                   [output("hoop_mean", mean, "Pa", "Mean hoop stress through the wall."),
                    output("hoop_inner", inner, "Pa", "Hoop stress at the inner surface."),
                    output("hoop_outer", outer, "Pa", "Hoop stress at the outer surface."),
                    output("radial_inner", -p_, "Pa", "Radial stress at the inner surface.")],
                   [CalcStep(text="The mean hoop stress follows from the equilibrium of a half cylinder.",
                             latex=r"\sigma_{h,mean} = \frac{p \, r_i}{r_o - r_i}", result=mpa(mean)),
                    CalcStep(text="The Lamé equation gives the hoop stress at the inner surface.",
                             latex=r"\sigma_{h,i} = p \, \frac{r_o^2 + r_i^2}{r_o^2 - r_i^2}", result=mpa(inner)),
                    CalcStep(text="The Lamé equation gives the hoop stress at the outer surface.",
                             latex=r"\sigma_{h,o} = \frac{2 \, p \, r_i^2}{r_o^2 - r_i^2}", result=mpa(outer)),
                    CalcStep(text="The radial stress at the inner surface is the negative pressure.",
                             latex=r"\sigma_{r,i} = -p", result=mpa(-p_))],
                   ["The cylinder is long and has open ends.", "The material is linear elastic and isotropic.",
                    "The pressure is uniform on the inner surface."],
                   [], {key: fe_proof * s})
        spec = specs["HC-FLOOR-PLATE"]
        v, rows = inputs_of(spec, lc)
        p_, a, t = v["p"], v["a"], v["t"]
        nu = project.material_of().nu
        rows.append(CalcInput(name="nu", si=nu, unit="", display=f"{nu:g}", source="The material MAT-316L gives the value."))
        sc = 3 * (1 + nu) * p_ * a**2 / (8 * t**2)
        se = 3 * p_ * a**2 / (4 * t**2)
        ss = 3 * (3 + nu) * p_ * a**2 / (8 * t**2)
        record(spec, lc, spec.model, "Thin circular plate with uniform pressure, clamped edge and simply supported edge.",
               rows,
               [output("sigma_center_clamped", sc, "Pa", "Bending stress at the centre of a clamped plate."),
                output("sigma_edge_clamped", se, "Pa", "Bending stress at the edge of a clamped plate."),
                output("sigma_center_simple", ss, "Pa", "Bending stress at the centre of a simply supported plate.")],
               [CalcStep(text="The bending stress at the centre of a clamped plate is:",
                         latex=r"\sigma_{c} = \frac{3 \, (1 + \nu) \, p \, a^2}{8 \, t^2}", result=mpa(sc)),
                CalcStep(text="The bending stress at the edge of a clamped plate is:",
                         latex=r"\sigma_{e} = \frac{3 \, p \, a^2}{4 \, t^2}", result=mpa(se)),
                CalcStep(text="The bending stress at the centre of a simply supported plate is:",
                         latex=r"\sigma_{s} = \frac{3 \, (3 + \nu) \, p \, a^2}{8 \, t^2}", result=mpa(ss))],
               ["The plate is thin and flat.", "The deflection is small in comparison with the thickness."],
               [f"The ratio t / a is {t / a:.2f}. The thin-plate formula is correct for a ratio below 0.2."],
               {"max_principal.membrane_bending@SCL-FLOOR": 112.6 * MPA * s})
    # ---- neck bending in steady firing
    lc, spec = by_id["LC-FIRE"], specs["HC-NECK-BENDING"]
    v, rows = inputs_of(spec, lc)
    m_, f_, ro, ri, re = v["M"], v["F"], v["r_o"], v["r_i"], v["r_eval"]
    area = math.pi * (ro**2 - ri**2)
    inertia = math.pi / 4 * (ro**4 - ri**4)
    sig = f_ / area + m_ * re / inertia
    record(spec, lc, spec.model, "Beam theory: axial force and bending moment on a tube section.", rows,
           [output("area", area, "m^2", "Area of the tube section."),
            output("inertia", inertia, "m^4", "Second moment of area of the tube section."),
            output("sigma_axial", f_ / area, "Pa", "Axial stress of the stem pressure force."),
            output("sigma_bending", m_ * re / inertia, "Pa", "Bending stress at the evaluation radius."),
            output("sigma_tension", sig, "Pa", "Axial stress on the tension side at the evaluation radius.")],
           [CalcStep(text="The area of the tube section is:", latex=r"A = \pi \, (r_o^2 - r_i^2)", result=f"{area * 1e6:.1f} mm²"),
            CalcStep(text="The second moment of area of the tube section is:", latex=r"I = \frac{\pi}{4} \, (r_o^4 - r_i^4)",
                     result=f"{inertia * 1e12:.0f} mm⁴"),
            CalcStep(text="The axial stress on the tension side is the sum of the two parts.",
                     latex=r"\sigma = \frac{F}{A} + \frac{M \, r_{eval}}{I}", result=mpa(sig))],
           ["The neck is a straight tube with a constant section.", "Plane sections stay plane.",
            "The temperature change along the neck causes no axial stress."],
           [], {"szz.membrane@SCL-NECK-ROOT": 114.7 * MPA})
    # ---- temperature of the actuator mounting face
    lc, spec = by_id["LC-TH-FIRE"], specs["HC-ACT-TEMP"]
    v, rows = inputs_of(spec, lc)
    perim, ac = 2 * math.pi * v["r_o"], math.pi * (v["r_o"]**2 - v["r_i"]**2)
    m = math.sqrt(v["h"] * perim / (v["k"] * ac))
    ml = m * v["L"]
    g = v["h"] * v["A_tip"] / (m * v["k"] * ac)
    theta = (v["T_base"] - v["T_inf"]) / (math.cosh(ml) + g * math.sinh(ml))
    t_tip = v["T_inf"] + theta
    record(spec, lc, spec.model, "Fin of constant section with a convective tip (heat transfer textbooks).", rows,
           [output("m", m, "1/m", "Fin parameter."), output("T_tip", t_tip, "K", "Temperature of the mounting pad.")],
           [CalcStep(text="The fin parameter of the neck is:", latex=r"m = \sqrt{\frac{h \, P}{k \, A_c}}", result=f"{m:.2f} 1/m"),
            CalcStep(text="The mounting pad loses heat as a lumped tip with the conductance ratio:",
                     latex=r"g = \frac{h \, A_{tip}}{m \, k \, A_c}", result=f"{g:.3f}"),
            CalcStep(text="The temperature of the tip follows from the fin equation.",
                     latex=r"T_{tip} = T_\infty + \frac{T_{base} - T_\infty}{\cosh(m L) + g \, \sinh(m L)}", result=f"{t_tip:.1f} K")],
           ["The temperature is uniform in each section of the neck.", "The base of the neck is at the LOX temperature.",
            "The heat transfer coefficient is constant along the neck."],
           [], {"temperature.min@INT-ACT": 252.6})
    # ---- pressure drop
    lc, spec = by_id["LC-FLOW-FIRE"], specs["HC-PRESSURE-DROP"]
    v, rows = inputs_of(spec, lc)
    vel = v["mdot"] / (v["rho"] * math.pi * v["d"]**2 / 4)
    re_ = v["rho"] * vel * v["d"] / v["mu"]
    fr = 1.0 / (-1.8 * math.log10((v["roughness"] / v["d"] / 3.7) ** 1.11 + 6.9 / re_)) ** 2
    dyn = 0.5 * v["rho"] * vel**2
    dp_f, dp_k = fr * v["L"] / v["d"] * dyn, v["K"] * dyn
    record(spec, lc, spec.model, "Darcy-Weisbach equation with the Haaland friction factor and a loss coefficient.", rows,
           [output("velocity", vel, "m/s", "Mean velocity in the flow bore."),
            output("reynolds", re_, "", "Reynolds number."), output("friction_factor", fr, "", "Darcy friction factor."),
            output("dp_friction", dp_f, "Pa", "Pressure drop of the wall friction."),
            output("dp_minor", dp_k, "Pa", "Pressure drop of the steps."),
            output("dp_total", dp_f + dp_k, "Pa", "Pressure drop of the open valve.")],
           [CalcStep(text="The mean velocity in the flow bore is:", latex=r"v = \frac{4 \, \dot{m}}{\rho \, \pi \, d^2}", result=f"{vel:.2f} m/s"),
            CalcStep(text="The Reynolds number is:", latex=r"Re = \frac{\rho \, v \, d}{\mu}", result=f"{re_:.0f}"),
            CalcStep(text="The Haaland equation gives the friction factor.",
                     latex=r"\frac{1}{\sqrt{f}} = -1.8 \, \log_{10}\left[\left(\frac{\epsilon / d}{3.7}\right)^{1.11} + \frac{6.9}{Re}\right]",
                     result=f"{fr:.4f}"),
            CalcStep(text="The pressure drop is the sum of the friction loss and the minor loss.",
                     latex=r"\Delta p = \left(f \, \frac{L}{d} + K\right) \frac{\rho \, v^2}{2}", result=f"{(dp_f + dp_k) / 1e3:.1f} kPa")],
           ["The flow is turbulent and fully developed.", "The loss coefficient K includes the steps at the two seat pockets and the ball."],
           [], {})
    return out


# =========================================================================== analyses


def _rv(key, value, unit, unc=None, status="converged", ratios=(0.955, 0.987), near="", loc=None, method="fe", note=""):
    levels = [] if value is None else [value * r for r in ratios] + [value]
    extra = {}
    if status == "converged" and unc and len(levels) == 3:
        extra = {"extrapolated": value + (value - levels[1]) * 0.9, "order": 1.9}
    return ResultValue(key=key, value=value, unit=unit, uncertainty=unc, status=status, location=loc, near=near,
                       per_level=levels, method=method, note=note, **extra)


def _mesh(dofs_per_node: int, name: str) -> list[MeshLevel]:
    return [MeshLevel(level=k, h=h, nodes=n, cells=c, dofs=dofs_per_node * n, order=2, min_quality=q, h_rep=hr,
                      path=f"meshes/{name}_L{k}.msh") for k, h, n, c, q, hr in MESH]


def _std_checks(lc_id: str, err: float, ok_err: bool = True) -> list[Check]:
    return [
        Check(id=f"CHK-{lc_id[3:]}-EQUILIBRIUM", kind="equilibrium", passed=True, value=0.0002, limit=0.01,
              statement="The reactions balance the applied loads. The largest difference is 0.02 % of the largest load."),
        Check(id=f"CHK-{lc_id[3:]}-MESH-QUALITY", kind="mesh_quality", passed=True, value=0.31, limit=0.10,
              statement="The smallest element quality of the finest mesh is 0.31. The limit is 0.10."),
        Check(id=f"CHK-{lc_id[3:]}-ERROR-ESTIMATE", kind="error_estimate", passed=ok_err, value=err, limit=0.05,
              statement=f"The largest discretization uncertainty of the key results is {err * 100:.1f} %. The limit is 5 %."),
    ]


def _bc_value(x):
    return x.si


def _analyses(project: Project, ctx, load_cases: list[LoadCase], hand_calcs: list[HandCalcResult],
              pictures: bool) -> dict[str, tuple[AnalysisRecord, Journal]]:
    by_id = {lc.id: lc for lc in load_cases}
    mat = project.material_of()
    material = {"E": units.si(mat.E.quantity()), "nu": mat.nu, "rho": units.si(mat.rho.quantity()),
                "alpha": units.si(mat.alpha.quantity()), "k": units.si(mat.k.quantity())}
    out: dict[str, tuple[AnalysisRecord, Journal]] = {}
    scl_edge = "the edge between FT-SEAT-OUT-SHOULDER and FT-SEAT-OUT-WALL"

    def bcs_of(lc: LoadCase, fixed_decision: str | None) -> list[BCItem]:
        items = []
        for it in lc.supports:
            feats = next(i for i in project.subject.interfaces if i.id == it).features.get("seat", [])
            items.append(BCItem(type="fixed", features=feats, effect="EF-IN-HOLD", interface=it, decision=fixed_decision,
                                description="The face is fixed in all directions."))
        for x in lc.loads:
            kind = {"fluid_temperature": "convection"}.get(x.kind, x.kind)
            value = {"h": x.h_si, "T": x.si} if x.kind == "fluid_temperature" else _bc_value(x)
            desc = {"pressure": "The pressure acts on the wetted faces of the interface.",
                    "force": "The force acts as a uniform traction on the face.",
                    "moment": "The moment acts as a linear traction on the face.",
                    "acceleration": "The acceleration acts on the mass of the body.",
                    "fluid_temperature": "The fluid exchanges heat with the faces by convection."}.get(x.kind, "")
            items.append(BCItem(type=kind, features=[f for f in x.features if f != "*"],
                                faces=["f%d" % t for t in ([4, 5, 6, 7, 8] if x.features == ["*"] else [])],
                                value=value, effect=x.effect, interface=x.interface,
                                decision=(f"DEC-{lc.id}-{x.kind.upper()}" if (lc.id, x.kind) in decided else None),
                                description=desc))
        return items

    decided: set[tuple[str, str]] = set()

    def journal(lc_id: str) -> Journal:
        return Journal("idealize", "engine:synthetic.structural")

    def artifacts(lc_id: str, field: str | None, caption: str) -> list[Artifact]:
        arts = [Artifact(path=f"analyses/{lc_id}/fields_L{k}.npz", kind="field", caption=f"Fields of mesh level {k}.") for k in (2, 3)]
        arts.append(Artifact(path=f"analyses/{lc_id}/solve.log", kind="log", caption="Log of the solver."))
        if pictures and field:
            _picture(project.analysis_dir(lc_id) / f"{field}.png", caption, field)
            arts.insert(0, Artifact(path=f"analyses/{lc_id}/{field}.png", kind="picture", caption=caption + " (synthetic picture)."))
        return arts

    def hc_checks(lc_id: str) -> list[Check]:
        checks = []
        for h in hand_calcs:
            if h.load_case != lc_id:
                continue
            for c in h.comparisons:
                if c.fe_value is None:
                    continue
                if c.band is not None and c.ratio is not None:
                    text = (f"The analysis gives {c.fe_value / MPA:.1f} MPa. The hand calculation gives {c.hand_value / MPA:.1f} MPa. "
                            f"The ratio {c.ratio:.2f} is {'in' if c.agree else 'not in'} the band {c.band[0]:g} to {c.band[1]:g}.")
                else:
                    text = (f"The analysis gives {c.fe_value:.1f} K. The hand calculation gives {c.hand_value:.1f} K. "
                            f"The difference is {'in' if c.agree else 'not in'} the tolerance of {c.tolerance_si:g} K.")
                checks.append(Check(id=f"CHK-{lc_id[3:]}-{h.spec}", kind="hand_calc", statement=text, passed=c.agree,
                                    value=c.ratio if c.ratio is not None else abs(c.fe_value - c.hand_value),
                                    limit=c.band[1] if c.band is not None else c.tolerance_si, link=h.id))
        return checks

    # ------------------------------------------------------------------ proof (solved) and its scaled cases
    proof = {
        "von_mises.membrane_bending@scls": (121.4, 0.021, "SCL-FLOOR"),
        "von_mises.membrane_bending@SCL-SHELL": (71.2, 0.012, ""), "von_mises.membrane_bending@SCL-NOZZLE-OUT": (38.9, 0.009, ""),
        "von_mises.membrane_bending@SCL-FLOOR": (121.4, 0.021, ""), "von_mises.membrane_bending@SCL-ROOF": (96.3, 0.018, ""),
        "von_mises.membrane_bending@SCL-NECK-ROOT": (31.0, 0.007, ""),
        "von_mises.membrane@scls": (61.8, 0.008, "SCL-SHELL"),
        "max_principal.membrane@SCL-SHELL": (63.9, 0.008, ""), "max_principal.membrane@SCL-NOZZLE-OUT": (24.1, 0.009, ""),
        "max_principal.membrane_bending@SCL-FLOOR": (112.6, 0.020, ""),
    }
    for lc_id, s, method in [("LC-PROOF", 1.0, "fe"), ("LC-BURST", 2.5 / 1.5, "scaled"), ("LC-LEAK", 1.0 / 1.5, "scaled")]:
        lc = by_id[lc_id]
        j = journal(lc_id)
        res = {k: _rv(k, v * MPA * s, "Pa", u, near=near, method=method) for k, (v, u, near) in proof.items()}
        res["von_mises.peak@part"] = _rv("von_mises.peak@part", 212.0 * MPA * s, "Pa", None, "singular", (0.769, 0.882),
                                         near=scl_edge, loc=[0.027, 0.016, 0.0], method=method,
                                         note="The value increases with each mesh level. The edge has no radius in the CAD file.")
        res["displacement.peak@part"] = _rv("displacement.peak@part", 21.3e-6 * s, "m", 0.010, near="FT-CAVITY-FLOOR",
                                            loc=[0.0, 0.0, -0.022], method=method, ratios=(0.981, 0.994))
        if lc_id == "LC-PROOF":
            decided.update({(lc_id, "force")})
            j.decide(lc_id, "support", "Support of the free valve body",
                     "Hold the body with a statically determinate support at six degrees of freedom on the inlet flange face.",
                     "The loads of the proof test are in equilibrium. A statically determinate support then has zero reactions "
                     "and adds no stress.",
                     alternatives=[("Fix the inlet flange face.", "A fixed face stops the radial growth of the flange and adds stress that is not real."),
                                   ("Use inertia relief.", "The solver backend has no inertia relief.")],
                     basis="engineering_judgement", links=[("about", "ASM-LC-PROOF-FIXTURE")])
            j.decide(lc_id, "force", "Force of the blind covers",
                     "Apply each cover force as a uniform traction on its face.",
                     "The gasket and the bolts spread the force on the face. The stress classification lines are far from the flanges.",
                     alternatives=[("Apply the force on the bolt holes.", "The section stresses do not change. The mesh at the holes is coarse.")])
            j.assume(lc_id, "linear", "Linear elastic material",
                     "The material stays linear elastic at the proof pressure.",
                     "The section stresses are below the yield strength. Local yield at an edge does not change the section stresses.",
                     "With general yield of a wall section, the stress distribution changes and the linear results are not correct.",
                     verify_by="Make sure that the membrane plus bending stress at each stress classification line is below 170 MPa.",
                     conservative=True)
            bcs = [BCItem(type="displacement", features=["FT-INLET-FLANGE-FACE"], value=[0.0, 0.0, 0.0], interface="INT-INLET",
                          decision="DEC-LC-PROOF-SUPPORT",
                          description="Three points of the face hold six degrees of freedom. The reactions are zero.")] + bcs_of(lc, None)
            rec = AnalysisRecord(
                id="AN-PROOF", title=lc.title, load_case=lc_id, analysis_type=lc.analysis_type, status="done",
                statement="A linear static analysis of the valve body with the proof pressure on every wetted face. "
                          "The cover forces act on the two flange faces and on the mounting face. "
                          "Three meshes of quadratic tetrahedra give the convergence of each result.",
                solver="synthetic (no solver ran)", material=material, bcs=bcs, mesh=_mesh(3, "body"), results=res,
                checks=_std_checks(lc_id, 0.021) + hc_checks(lc_id)
                + [Check(id="CHK-PROOF-ENERGY", kind="energy", passed=None,
                         statement="The check of the strain energy between mesh levels did not run. The solver gave no energy value.")],
                artifacts=artifacts(lc_id, "von_mises", "Von Mises stress of LC-PROOF on the finest mesh"), runtime_s=212.4,
                origin=Origin(by=SYN, stage="analyze"))
        else:
            rec = AnalysisRecord(
                id=f"AN-{lc_id[3:]}", title=lc.title, load_case=lc_id, analysis_type=lc.analysis_type, status="done",
                statement=f"The results of AN-PROOF multiplied by {s:.3f}. No solve ran for this load case.",
                solver="scaled from AN-PROOF", material=material, bcs=[], mesh=_mesh(3, "body"), results=res,
                checks=hc_checks(lc_id), artifacts=[], runtime_s=0.1, links=[Link(rel="derives_from", to="AN-PROOF")],
                origin=Origin(by="engine:synthetic.scaled", stage="analyze"))
        out[lc_id] = (rec, j)

    # ------------------------------------------------------------------ thermal
    thermal = {"CHILL": (254.1, 90.9, 264.0, 55.2), "PRESTART": (253.4, 90.6, 263.2, 56.9), "FIRE": (252.6, 90.3, 262.1, 58.4),
               "SHUTDOWN": (253.0, 90.5, 262.6, 57.8)}
    for name, (t_act, t_min, t_max, q) in thermal.items():
        lc_id = f"LC-TH-{name}"
        lc = by_id[lc_id]
        j = Journal("idealize", "engine:synthetic.thermal")
        res = {"temperature.min@INT-ACT": _rv("temperature.min@INT-ACT", t_act, "K", 0.005, ratios=(1.009, 1.003), near="FT-PAD-TOP",
                                              loc=[0.0, 0.0, 0.124]),
               "temperature.min@part": _rv("temperature.min@part", t_min, "K", 0.001, ratios=(1.002, 1.0005), near="FT-CAVITY-FLOOR"),
               "temperature.max@part": _rv("temperature.max@part", t_max, "K", 0.004, ratios=(1.006, 1.002), near="FT-PAD-TOP"),
               "heat_flow.total@INT-ENV": _rv("heat_flow.total@INT-ENV", q, "W", 0.015, ratios=(0.97, 0.991))}
        if name == "FIRE":
            decided.add((lc_id, "fluid_temperature"))
            j.decide(lc_id, "fluid_temperature", "Convection on the wetted faces",
                     "Apply convection with the heat transfer coefficient and the fluid temperature of each effect on its wetted faces.",
                     "The effects EF-IN-T-FIRE and EF-OUT-T-FIRE give the LOX temperature of 90 K and the coefficient h_lox. "
                     "The effect EF-ENV-T gives the ambient air on the outer surfaces.",
                     links=[("uses", "EF-IN-T-FIRE"), ("uses", "EF-OUT-T-FIRE"), ("uses", "EF-ENV-T")])
            j.decide(lc_id, "flange-faces", "Thermal condition of the flange faces",
                     "Set the two flange faces and the mounting face to adiabatic.",
                     "The pump flange and the feed line are at the LOX temperature too. The heat flow through the flange faces is small. "
                     "The context gives no temperature for the actuator.",
                     alternatives=[("Set the flange faces to 90 K.", "The context gives no temperature for the mating parts."),
                                   ("Add the actuator as a heat source.", "The context gives no heat flow of the actuator.")],
                     basis="engineering_judgement", needs_review=True)
            j.assume(lc_id, "steady", "Steady temperature field",
                     "The temperature field reaches the steady state during the firing.",
                     "The firing takes 400 s. The wall of the shell reaches the LOX temperature in about 10 s.",
                     "The neck needs about 1600 s to reach the steady state. During a firing of 400 s the mounting face is warmer than "
                     "the analysis gives.",
                     verify_by="Do a transient thermal analysis of the neck.", conservative=True)
            j.assume(lc_id, "no-radiation", "No radiation and no frost",
                     "Radiation and frost have no effect on the heat transfer to the ambient air.",
                     "The parameter h_amb includes natural convection with frost as one assumed value.",
                     "A frost layer insulates the neck. The mounting face is then colder than the analysis gives.",
                     verify_by="Compare the mounting face temperature with a chill-down test.", conservative=False)
        rec = AnalysisRecord(
            id=f"AN-TH-{name}", title=lc.title, load_case=lc_id, analysis_type="thermal_steady", status="done",
            statement="A steady heat conduction analysis of the valve body. Convection acts on the wetted faces and on the outer "
                      "surfaces. The flange faces are adiabatic.",
            solver="synthetic (no solver ran)", material=material, bcs=bcs_of(lc, None), mesh=_mesh(1, "body"), results=res,
            checks=[Check(id=f"CHK-TH-{name}-HEAT-BALANCE", kind="equilibrium", passed=True, value=0.0004, limit=0.01,
                          statement="The heat flows of all faces balance. The largest difference is 0.04 % of the heat flow."),
                    Check(id=f"CHK-TH-{name}-ERROR-ESTIMATE", kind="error_estimate", passed=True, value=0.015, limit=0.05,
                          statement="The largest discretization uncertainty of the key results is 1.5 %. The limit is 5 %.")]
            + hc_checks(lc_id),
            artifacts=artifacts(lc_id, "temperature" if name == "FIRE" else None, f"Temperature field of {lc_id} on the finest mesh"),
            runtime_s=41.7, origin=Origin(by="engine:synthetic.thermal", stage="analyze"))
        out[lc_id] = (rec, j)

    # ------------------------------------------------------------------ thermo-structural
    fillet = ("FT-NECK-ROOT-FILLET", [0.0, -0.0128, 0.0425])
    struct = {
        "CHILL": {"peak": (96.4, 0.024, "converged", (0.952, 0.985), fillet), "mb": (None, None, "not_run", "", None),
                  "disp": 0.29e-3, "err": 0.024},
        "PRESTART": {"peak": (139.0, 0.091, "not_converged", (0.806, 0.917), ("FT-CAVITY-FLOOR", [0.021, 0.0, -0.021])),
                     "mb": (97.5, 0.016, "converged", "SCL-FLOOR", None), "disp": 0.30e-3, "err": 0.091},
        "FIRE": {"peak": (171.6, None, "singular", (0.747, 0.862), (scl_edge, [0.027, 0.016, 0.0])),
                 "mb": (118.7, 0.014, "converged", "SCL-NECK-ROOT", None), "disp": 0.33e-3, "err": 0.032},
        "SHUTDOWN": {"peak": (141.0, 0.041, "converged", (0.921, 0.969), fillet), "mb": (109.3, 0.017, "converged", "SCL-FLOOR", None),
                     "disp": 0.31e-3, "err": 0.041},
    }
    for name, d in struct.items():
        lc_id = f"LC-{name}"
        lc = by_id[lc_id]
        j = journal(lc_id)
        pv, pu, ps, pr, (pnear, ploc) = d["peak"]
        res = {"von_mises.peak@part": _rv("von_mises.peak@part", pv * MPA, "Pa", pu, ps, pr, near=pnear, loc=ploc,
                                          note={"singular": "The value increases with each mesh level. The edge has no radius in the CAD file.",
                                                "not_converged": "The uncertainty is more than the policy limit of 5 %."}.get(ps, ""))}
        mv, mu, ms, mnear, _ = d["mb"]
        res["von_mises.membrane_bending@scls"] = _rv(
            "von_mises.membrane_bending@scls", mv * MPA if mv else None, "Pa", mu, ms, near=mnear,
            note="" if mv else "The evaluation of the stress classification lines did not run for this load case. The log gives the cause.")
        res["displacement.peak@part"] = _rv("displacement.peak@part", d["disp"], "m", 0.006, near="FT-PAD-TOP", ratios=(0.99, 0.997))
        if name == "PRESTART":
            res["radial_displacement.range@FT-SEAT-OUT-WALL"] = _rv("radial_displacement.range@FT-SEAT-OUT-WALL", 26.4e-6, "m", 0.020,
                                                                   ratios=(0.951, 0.985), near="FT-SEAT-OUT-WALL", loc=[0.0245, 0.0, 0.016])
        if name == "FIRE":
            res["szz.membrane@SCL-NECK-ROOT"] = _rv("szz.membrane@SCL-NECK-ROOT", 114.7 * MPA, "Pa", 0.013, near="SCL-NECK-ROOT")
            res["von_mises.peak@FT-NECK-ROOT-FILLET"] = _rv("von_mises.peak@FT-NECK-ROOT-FILLET", 148.3 * MPA, "Pa", 0.032,
                                                            ratios=(0.939, 0.978), near=fillet[0], loc=fillet[1])
        decided.add((lc_id, "pressure") if name == "FIRE" else (lc_id, "none"))
        j.decide(lc_id, "support", "Support at the inlet flange",
                 "Fix the inlet flange face in all directions.",
                 "The pump discharge flange holds the valve. The pump housing is much stiffer than the valve body.",
                 alternatives=[("Fix only the bolt holes.", "The gasket also carries compression. The stress in the neck is not sensitive to this choice."),
                               ("Model the pump flange as an elastic body.", "The project has no model of the pump housing.")],
                 basis="engineering_judgement", needs_review=(name == "FIRE"), links=[("uses", "EF-IN-HOLD")])
        if name == "FIRE":
            j.decide(lc_id, "pressure", "Pressure on the wetted faces",
                     "Apply the pressure of each effect as a uniform pressure on the wetted faces of its interface.",
                     "The interface INT-INLET lists the wetted faces upstream of the downstream seat. The interface INT-OUTLET lists the others.",
                     links=[("uses", "EF-IN-P-FIRE"), ("uses", "EF-OUT-P-FIRE"), ("uses", "EF-STEM-P-FIRE")])
            j.decide(lc_id, "outlet-load", "Force and moment of the feed line",
                     "Apply the force as a uniform traction and the moment as a linear traction on the outlet flange face.",
                     "The two tractions have the same resultant as the effects EF-OUT-F-FIRE and EF-OUT-M-FIRE. "
                     "The stress near the face is not a result of interest.",
                     alternatives=[("Use a rigid coupling of the face to a point.", "A rigid coupling stiffens the flange too much.")],
                     links=[("uses", "EF-OUT-F-FIRE"), ("uses", "EF-OUT-M-FIRE")])
            j.decide(lc_id, "temperature-field", "Temperature field of the structural analysis",
                     "Use the temperature field of AN-TH-FIRE on the same mesh.",
                     "The thermal analysis and the structural analysis use the same nodes. No interpolation is necessary.",
                     links=[("uses", "LC-TH-FIRE")])
            j.assume(lc_id, "rigid-pump", "Rigid pump flange",
                     "The pump discharge flange is rigid.",
                     "The description of the interface says that the pump housing is much stiffer than the valve body.",
                     "An elastic flange decreases the thermal stress near the inlet flange. The stress in the neck does not change.",
                     conservative=True, links=[("uses", "IF-PUMP-MOV")])
            j.assume(lc_id, "properties-temperature", "Elastic properties at room temperature",
                     "The elastic modulus and the coefficient of thermal expansion at 293 K apply at 90 K.",
                     "The material file gives one set of properties. Its coefficient of thermal expansion is a mean value between 90 K and 293 K.",
                     "The elastic modulus of the material is about 7 % higher at 90 K. The thermal stress is then about 7 % higher than "
                     "the analysis gives.",
                     verify_by="Compare the properties with the cryogenic data of the material specification.", conservative=False,
                     links=[("uses", "MAT-316L")])
            j.assume(lc_id, "strength-temperature", "Strength at room temperature",
                     "The yield strength at 293 K applies to the cold body.",
                     "The strength of the material increases at low temperature. The material file says so.",
                     "The margins are larger than the analysis gives.", conservative=True, links=[("uses", "MAT-316L")])
            j.decide(lc_id, "peak-zone", "Zone without peak evaluation at the fixed face",
                     "Leave out the zone within 6 mm of the fixed face from the peak evaluation.",
                     "A fixed face stops the thermal contraction of the flange. The stress next to the face is an effect of the support "
                     "and is not a result.",
                     alternatives=[("Include the zone.", "The peak stress is then the stress of the support and has no meaning."),
                                   ("Leave out the full inlet flange.", "The neck of the flange carries real stress.")],
                     basis="engineering_judgement", needs_review=True, stage="solve")
            j.decide(lc_id, "singular", "Singular peak at the seat pocket edge",
                     "Classify the peak stress at the edge of the downstream seat pocket as singular.",
                     "The peak value increases by 15 % and by 16 % between the mesh levels. The edge has no radius in the CAD file.",
                     alternatives=[("Report the value of the finest mesh as the result.", "The value depends on the mesh size and has no limit.")],
                     basis="calculation", stage="solve", by="rule:synthetic.results.singular")
            j.assume(lc_id, "small-displacement", "Small displacements",
                     "The displacements are small in comparison with the wall thickness.",
                     "The largest displacement is 0.33 mm. Most of it is thermal contraction, which causes no strain.",
                     "With large displacements, the linear analysis is not correct.",
                     verify_by="Compare the largest displacement of each analysis with the wall thickness of 4 mm.",
                     conservative=None, stage="solve")
        if name == "PRESTART":
            j.decide(lc_id, "seat-force", "Force of the downstream seat ring",
                     "Apply the seat force as a uniform pressure on the shoulder FT-SEAT-OUT-SHOULDER.",
                     "The seat ring is soft in comparison with the body and spreads the force on the shoulder.",
                     alternatives=[("Apply the force on a line at the mean sealing diameter.", "A line load causes a stress peak that is not real.")],
                     links=[("uses", "EF-SEAT-F-PRESTART")])
            j.decide(lc_id, "not-converged", "Convergence of the peak stress",
                     "Classify the peak stress of LC-PRESTART as not converged.",
                     "The discretization uncertainty is 9.1 %. The policy limit is 5 %.",
                     basis="calculation", stage="solve", by="rule:synthetic.results.convergence", links=[("uses", "POLICY")])
        checks = _std_checks(lc_id, d["err"], d["err"] <= 0.05) + hc_checks(lc_id)
        rec = AnalysisRecord(
            id=f"AN-{name}", title=lc.title, load_case=lc_id, analysis_type="thermo_structural", status="done",
            statement=f"A linear static analysis with the limit loads of {lc.phase} and the temperature field of AN-TH-{name}. "
                      "The inlet flange face is fixed. The peak evaluation leaves out the zone within 6 mm of the fixed face.",
            solver="synthetic (no solver ran)", material=material,
            bcs=bcs_of(lc, f"DEC-{lc_id}-SUPPORT") + [BCItem(type="temperature_field", value={"from": f"AN-TH-{name}"},
                                                              description=f"The temperature of each node comes from AN-TH-{name}.",
                                                              decision="DEC-LC-FIRE-TEMPERATURE-FIELD" if name == "FIRE" else None)],
            mesh=_mesh(3, "body"), results=res, checks=checks,
            artifacts=artifacts(lc_id, "von_mises" if name in ("FIRE", "PRESTART") else None, f"Von Mises stress of {lc_id} on the finest mesh"),
            runtime_s=230.0 + 7 * len(name), origin=Origin(by=SYN, stage="analyze"))
        out[lc_id] = (rec, j)

    # ------------------------------------------------------------------ mesh decisions (once, with the first solved case)
    jm = out["LC-PROOF"][1]
    jm.decide("CMP-MOV-BODY", "mesh-levels", "Mesh levels",
              "Use three meshes of quadratic tetrahedra with the sizes 4.0 mm, 2.8 mm and 2.0 mm.",
              "The node limit is 120000. A size ratio of 1.4 between the levels gives a clear convergence trend.",
              alternatives=[("Use linear tetrahedra.", "Linear tetrahedra are too stiff in bending."),
                            ("Use two levels.", "Two levels give no observed order of convergence.")],
              stage="mesh", by="engine:synthetic.meshing")
    jm.decide("CMP-MOV-BODY", "mesh-refinement", "Local refinement of the mesh",
              "Refine the mesh at the neck root fillet, at the corner radius of the cavity floor and at the four re-entrant edges.",
              "The stress changes quickly at these locations. The fillet radius is 3 mm and the local size is 0.75 mm on the finest mesh.",
              stage="mesh", by="engine:synthetic.meshing", links=[("uses", "FT-NECK-ROOT-FILLET"), ("uses", "FT-CAVITY-FLOOR")])

    # ------------------------------------------------------------------ fatigue
    lc = by_id["LC-FATIGUE"]
    j = Journal("idealize", "engine:synthetic.fatigue")
    j.decide("LC-FATIGUE", "method", "Fatigue method",
             "Use the stress-life method with the Goodman correction and the linear damage rule.",
             "The material file gives Basquin constants and the Goodman correction. The stresses are below the yield strength in "
             "most of the body.",
             alternatives=[("Use the strain-life method.", "The material file has no strain-life constants.")])
    j.assume("LC-FATIGUE", "basquin", "Stress-life constants of the material file",
             "The Basquin constants of the material file apply to the forged body.",
             "The material file gives these constants for the stress-life method.",
             "The material file marks the constants as illustrative. With design data, the damage can be much higher.",
             verify_by="Replace the constants with the design data of the material specification.", conservative=False,
             links=[("uses", "MAT-316L")])
    rec = AnalysisRecord(
        id="AN-FATIGUE", title=lc.title, load_case="LC-FATIGUE", analysis_type="fatigue", status="done",
        statement="The stress fields of the load case states give the stress range at each node for each duty cycle. "
                  "The stress-life curve gives the damage of one cycle. The linear rule adds the damage of 267 cycles.",
        solver="synthetic (no solver ran)", material={"sigma_f": 1.0e9, "b": -0.10, "mean_stress": "goodman"},
        mesh=_mesh(3, "body")[1:],
        results={"damage.total@part": _rv("damage.total@part", 0.0312, "", None, "estimate", (0.61, 0.84), near=scl_edge,
                                          loc=[0.027, 0.016, 0.0], method="derived",
                                          note="The largest damage is at a singular stress peak. The value is an upper bound on this mesh only."),
                 "damage.total@FT-NECK-ROOT-FILLET": _rv("damage.total@FT-NECK-ROOT-FILLET", 0.0041, "", 0.31, "not_converged",
                                                         (0.71, 0.90), near="FT-NECK-ROOT-FILLET", method="derived",
                                                         note="A stress uncertainty of 3 % gives a damage uncertainty of about 30 %.")},
        checks=[Check(id="CHK-FATIGUE-STATES", kind="inputs", passed=True,
                      statement="Each state of each duty cycle has a stress field on the two finest meshes."),
                Check(id="CHK-FATIGUE-ERROR-ESTIMATE", kind="error_estimate", passed=False, value=0.31, limit=0.05,
                      statement="The largest discretization uncertainty of the key results is 31 %. The limit is 5 %.")],
        artifacts=[Artifact(path="analyses/LC-FATIGUE/damage.npz", kind="data", caption="Damage at each node of the finest mesh.")],
        runtime_s=12.9, origin=Origin(by="engine:synthetic.fatigue", stage="analyze"))
    out["LC-FATIGUE"] = (rec, j)

    # ------------------------------------------------------------------ flow: not supported
    lc = by_id["LC-FLOW-FIRE"]
    rec = AnalysisRecord(
        id="AN-FLOW-FIRE", title=lc.title, load_case="LC-FLOW-FIRE", analysis_type="flow", status="not_supported",
        message="The flow engine needs OpenFOAM. OpenFOAM is not installed on this computer.",
        results={"pressure_drop.total@part": ResultValue(key="pressure_drop.total@part", value=None, unit="Pa", status="not_run")},
        origin=Origin(by="engine:pipeline", stage="analyze"))
    out["LC-FLOW-FIRE"] = (rec, Journal("solve", "engine:pipeline"))
    return out


# =========================================================================== assess


def _assess(project: Project, ctx, analyses: dict[str, AnalysisRecord], hand_calcs: list[HandCalcResult]):
    j = Journal("assess", "engine:synthetic.compliance")
    mat = project.material_of()
    sy, su = units.si(mat.yield_strength.quantity()), units.si(mat.ultimate_strength.quantity())
    items: list[ComplianceItem] = []
    policy_caveat = "FND-REQ-MOV-004-YIELD-FACTOR"

    def add(req_id, idx, lc_id, factor, factor_source, limit, limit_source, verdict=None, reasons=(), caveats=(),
            value=None, unc=None, statement=None, analysis=True):
        req = project.requirement(req_id)
        c = req.criteria[idx]
        an = analyses[lc_id]
        r = an.results.get(c.key())
        if value is None and r is not None:
            value, unc = r.value, r.uncertainty
        unit = (r.unit if r is not None else "") or ""
        cons = margin = None
        if value is not None:
            u = unc or 0.0
            cons = value * (1 + u) if c.comparator == "<=" else value * (1 - u)
            margin = limit / (factor * cons) - 1 if c.comparator == "<=" else (cons / factor) / limit - 1
        if verdict is None:
            verdict = "not_assessed" if margin is None else ("pass" if margin >= 0 else "fail")
        ms = f"{margin:+.2f}" if margin is not None else "not available"
        text = statement or {
            "pass": f"The criterion {c.id} of {req_id} is met in {lc_id}. The margin of safety is {ms}.",
            "fail": f"The criterion {c.id} of {req_id} is not met in {lc_id}. The margin of safety is {ms}.",
            "inconclusive": f"The analysis does not show compliance with the criterion {c.id} of {req_id} in {lc_id}. "
                            f"The margin of the available value is {ms}.",
            "not_assessed": f"The criterion {c.id} of {req_id} has no result in {lc_id}.",
        }[verdict]
        items.append(ComplianceItem(
            id=make_id("CMPL", req_id, c.id or str(idx), lc_id), title=f"{req.title}: {c.id} in {lc_id}",
            requirement=req_id, criterion=idx, load_case=lc_id, analysis=an.id if analysis else None, result_key=c.key(),
            value=value, unit=unit, uncertainty=unc, conservative_value=cons, factor=factor, factor_source=factor_source,
            limit=limit, limit_source=limit_source, comparator=c.comparator, margin=margin, verdict=verdict, statement=text,
            reasons=list(reasons), caveats=list(caveats), origin=Origin(by="engine:synthetic.compliance", stage="assess")))
        return items[-1]

    yield_src, ult_src = "The yield strength of the material MAT-316L.", "The ultimate strength of the material MAT-316L."
    add("REQ-MOV-002", 0, "LC-PROOF", 1.0, "The criterion gives the factor 1.0. The proof pressure includes the proof factor.",
        sy, yield_src, reasons=["Every result of the criterion is converged.", "The three hand calculations agree with the analysis."],
        caveats=["FND-REQ-MOV-001-MEOP-ENVELOPE"])
    add("REQ-MOV-003", 0, "LC-BURST", 1.0, "The criterion gives the factor 1.0. The burst pressure includes the burst factor.",
        su, ult_src, reasons=["The results come from LC-PROOF by the scale 1.667.", "Every result of the criterion is converged."],
        caveats=["FND-REQ-MOV-001-MEOP-ENVELOPE"])
    fy = "The parameter yield_factor of the requirement REQ-MOV-004 gives 1.1."
    fu = "The parameter ultimate_factor of the requirement REQ-MOV-004 gives 1.4."
    add("REQ-MOV-004", 0, "LC-CHILL", 1.1, fy, sy, yield_src, reasons=["The peak stress is converged."], caveats=[policy_caveat])
    add("REQ-MOV-004", 0, "LC-PRESTART", 1.1, fy, sy, yield_src, verdict="inconclusive", caveats=[policy_caveat],
        reasons=["The peak stress is not converged. Its uncertainty is 9.1 % and the policy limit is 5 %."])
    add("REQ-MOV-004", 0, "LC-FIRE", 1.1, fy, sy, yield_src, verdict="inconclusive", caveats=[policy_caveat],
        reasons=["The peak stress is singular. The value increases with each mesh level.",
                 "The hand calculation HCR-NECK-BENDING-FIRE does not agree with the analysis."])
    add("REQ-MOV-004", 0, "LC-SHUTDOWN", 1.1, fy, sy, yield_src, reasons=["The peak stress is converged."],
        caveats=[policy_caveat, "FND-REQ-MOV-001-MEOP-ENVELOPE"])
    add("REQ-MOV-004", 1, "LC-CHILL", 1.4, fu, su, ult_src,
        reasons=["The analysis AN-CHILL has no value for the result von_mises.membrane_bending@scls."])
    add("REQ-MOV-004", 1, "LC-PRESTART", 1.4, fu, su, ult_src, reasons=["The section stress is converged."])
    add("REQ-MOV-004", 1, "LC-FIRE", 1.4, fu, su, ult_src, verdict="inconclusive",
        reasons=["The hand calculation HCR-NECK-BENDING-FIRE does not agree with the analysis."])
    add("REQ-MOV-004", 1, "LC-SHUTDOWN", 1.4, fu, su, ult_src, reasons=["The section stress is converged."])
    add("REQ-MOV-006", 0, "LC-FATIGUE", 4.0, "The parameter life_factor of the requirement REQ-MOV-006 gives 4.",
        1.0, "The criterion gives the damage limit 1.0.", verdict="inconclusive", caveats=["FND-REQ-MOV-006-DESIGN-CYCLES"],
        reasons=["The damage is an estimate. The largest damage is at a singular stress peak."])
    add("REQ-MOV-007", 0, "LC-TH-FIRE", 1.0, "The criterion gives no factor.", 233.0,
        "The parameter t_act_min of the requirement REQ-MOV-007 gives 233 K.",
        reasons=["The temperature is converged.", "The hand calculation HCR-ACT-TEMP-TH-FIRE agrees with the analysis."])
    hc = next(h for h in hand_calcs if h.spec == "HC-PRESSURE-DROP")
    dp = next(o.si for o in hc.outputs if o.name == "dp_total")
    add("REQ-MOV-008", 0, "LC-FLOW-FIRE", 1.0, "The criterion gives no factor.", 60.0e3,
        "The parameter dp_max of the requirement REQ-MOV-008 gives 60 kPa.", verdict="inconclusive", value=dp, analysis=False,
        reasons=["The flow analysis is not available. The value is the estimate of the hand calculation " + hc.id + "."])
    items[-1].links.append(Link(rel="evidence", to=hc.id, note="estimate"))
    items[-1].unit = "Pa"
    add("REQ-MOV-010", 0, "LC-PRESTART", 1.0, "The criterion gives no factor.", 20.0e-6,
        "The parameter roundness_max of the requirement REQ-MOV-010 gives 20 µm.",
        reasons=["The result is converged. Its uncertainty is 2.0 %.", "The out-of-roundness of 26.4 µm is more than the limit of 20 µm."])

    verdicts = []
    order = {"fail": 0, "inconclusive": 1, "not_assessed": 2, "pass": 3}
    for req in project.reqs():
        mine = [i for i in items if i.requirement == req.id]
        if not mine:
            why = {"REQ-MOV-009": "The requirement has no criterion. The finding FND-REQ-MOV-009-NOT-VERIFIABLE gives the cause."}.get(
                req.id, "The verification method of the requirement is review. The analysis gives no verdict.")
            verdicts.append(RequirementVerdict(requirement=req.id, verdict="not_assessed", statement=why))
            continue
        worst = min(mine, key=lambda i: order[i.verdict])
        with_margin = [i for i in mine if i.margin is not None]
        gov = min(with_margin, key=lambda i: i.margin) if with_margin else None
        n = {v: sum(1 for i in mine if i.verdict == v) for v in order}
        text = {"pass": f"Every criterion of {req.id} is met. The smallest margin of safety is {gov.margin:+.2f}." if gov else "",
                "fail": f"The requirement {req.id} is not met. The smallest margin of safety is {gov.margin:+.2f}." if gov else "",
                "inconclusive": f"The analysis does not show compliance with {req.id}. {n['inconclusive']} of {len(mine)} items are inconclusive.",
                "not_assessed": f"The requirement {req.id} has no verdict."}[worst.verdict]
        verdicts.append(RequirementVerdict(requirement=req.id, verdict=worst.verdict, margin=gov.margin if gov else None,
                                           governing=gov.id if gov else None, statement=text, items=[i.id for i in mine]))
    j.decide("REQ-MOV-004", "factor", "Factor of safety against yield",
             "Use the yield factor 1.1 of the requirement in the margins.",
             "The requirement states the factor. The finding FND-REQ-MOV-004-YIELD-FACTOR reports that the policy asks for 1.25. "
             "Each compliance item shows the finding as a caveat.",
             alternatives=[("Use the policy factor 1.25.", "The tool does not change a requirement. A person decides the correction.")],
             needs_review=True, links=[("about", policy_caveat), ("uses", "yield_factor")])
    j.decide("REQ-MOV-008", "estimate", "Estimate of the pressure drop",
             "Use the hand calculation as an estimate of the pressure drop. Give the verdict inconclusive.",
             "The flow analysis is not available on this computer. The hand calculation gives 14.3 kPa. "
             "An estimate cannot show compliance.",
             alternatives=[("Give no value.", "The estimate shows the engineer how far the value is from the limit.")],
             links=[("uses", hc.id)])
    j.decide("REQ-MOV-006", "upper-bound", "Damage at a singular stress peak",
             "Report the damage at the seat pocket edge as an estimate. Give the verdict inconclusive.",
             "The largest damage is at an edge where the stress is singular. The damage there has no converged value.",
             alternatives=[("Use the damage at the neck root fillet only.", "The edge of the seat pocket is a real location of high stress.")],
             needs_review=True)
    j.assume("MAT-316L", "minimum-strength", "Strength of the forged body",
             "The yield strength of 170 MPa and the ultimate strength of 485 MPa are the minimum values of the forged body.",
             "The material file gives these values for the specification ASTM A182 F316L.",
             "A body with a lower strength has smaller margins.",
             verify_by="Compare the values with the material certificate of the forging.", conservative=None)
    return items, verdicts, j


# =========================================================================== agent proposals, reviews


def _agent() -> Journal:
    j = Journal("idealize", "agent:analyst")
    j.decide("LC-FIRE", "submodel", "Proposal: edge radius at the downstream seat pocket",
             "Add a submodel of the downstream seat pocket with an edge radius of 0.3 mm.",
             "A real edge has a radius. With a radius, the peak stress has a limit, and the analysis can give a verdict for REQ-MOV-004.",
             alternatives=[("Keep the sharp edge and use the section stress only.", "The requirement gives a criterion for the peak stress.")],
             basis="engineering_judgement", needs_review=True, links=[("about", "CMPL-REQ-MOV-004-YIELD-LC-FIRE")])
    j.assume("LC-PRESTART", "seat-ring-contact", "Even contact of the seat ring",
             "The seat ring touches the shoulder on its full area.",
             "The seat ring is a soft polymer part and follows the shape of the shoulder.",
             "A ring that touches on a narrow band puts a higher local load on the pocket. The out-of-roundness can increase.",
             verify_by="Examine the contact pattern of the seat ring after the proof test.", conservative=False,
             links=[("about", "DEC-LC-PRESTART-SEAT-FORCE")])
    j.find("agent.handcalc_validity", "minor", "model", "HC-FLOOR-PLATE", "thick-plate",
           "The floor is too thick for the thin-plate formula",
           "The hand calculation HC-FLOOR-PLATE uses a thin-plate formula. The floor has a thickness ratio of 0.36.",
           "The thin-plate formula is correct for a thickness ratio below 0.2. Shear deformation is not small in a thick plate. "
           "The accepted band of the comparison is wide for this reason.",
           suggestion="Use a thick-plate formula, or keep the wide band and record the reason.", values={"t_over_a": 0.36})
    return j


def _reviews(project: Project) -> None:
    """Reviews in every state. Three reviews are stale: their digest is not the digest of the record."""
    recs = project.records()
    plan = {
        "accepted": ["DEC-LC-BURST-SCALED", "DEC-LC-LEAK-SCALED", "DEC-LC-PROOF-SELECT", "DEC-LC-PROOF-SUPPORT", "DEC-LC-PROOF-FORCE",
                     "DEC-CMP-MOV-BODY-LENGTH-UNIT", "DEC-INT-ENV-EXTERNAL-FACES", "DEC-LC-FIRE-THERMAL", "DEC-LC-CHILL-THERMAL",
                     "DEC-CMP-MOV-BODY-MESH-LEVELS", "ASM-LC-PROOF-LINEAR", "ASM-LC-FIRE-STRENGTH-TEMPERATURE",
                     "ASM-H-AMB-ASSUMED", "FND-REQ-MOV-001-MEOP-ENVELOPE", "FND-REQ-MOV-005-MIN-TEMPERATURE", "DEC-LC-FIRE-LATERAL-DIRECTION"],
        "rejected": ["DEC-LC-TH-FIRE-FLANGE-FACES", "ASM-LC-FATIGUE-BASQUIN"],
        "changes_requested": ["DEC-LC-FIRE-PEAK-ZONE", "ASM-CMP-MOV-BODY-NOMINAL-GEOMETRY", "FND-PH-TRANSPORT-UNCOVERED"],
    }
    comments = {
        "DEC-LC-TH-FIRE-FLANGE-FACES": "The actuator is a heat source. Use the heat flow of ACT-SPEC-004.",
        "ASM-LC-FATIGUE-BASQUIN": "Illustrative constants are not acceptable for a life verdict.",
        "DEC-LC-FIRE-PEAK-ZONE": "Give the reason for 6 mm. Show the stress along the nozzle.",
        "ASM-CMP-MOV-BODY-NOMINAL-GEOMETRY": "Add a run with the minimum wall thickness.",
        "FND-PH-TRANSPORT-UNCOVERED": "Agreed. The specification owner must add the requirement.",
        "DEC-LC-BURST-SCALED": "Checked. The loads are proportional.",
        "FND-REQ-MOV-001-MEOP-ENVELOPE": "Agreed. Change request CR-0412 is open.",
    }
    body: dict[str, dict] = {}
    for state, ids in plan.items():
        for i, rid in enumerate(ids):
            if rid not in recs:
                raise KeyError(f"fixture review of a record that does not exist: {rid}")
            body[rid] = {"state": state, "by": "A. Example" if i % 3 else "B. Reviewer", "date": "2026-11-12",
                         "comment": comments.get(rid, ""), "digest": digest_of(recs[rid])}
    for rid in ("DEC-LC-FIRE-SUPPORT", "ASM-LC-TH-FIRE-STEADY", "FND-REQ-MOV-004-YIELD-FACTOR"):
        body[rid] = {"state": "accepted", "by": "A. Example", "date": "2026-11-03", "comment": "Accepted in the first review.",
                     "digest": "0000000000000000"}
    project.reviews_path.write_text("# SYNTHETIC reviews for the report tests.\n"
                                    + yaml.safe_dump({"reviews": dict(sorted(body.items()))}, sort_keys=False, allow_unicode=True))
