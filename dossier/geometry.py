"""The geometry stage: the CAD of the component, measured, and its named features as CAD faces.

    geo, journal = dossier.geometry.build(project, options)

`geo` is the dict of CONTRACTS.md ("geometry"); every later stage reads it. Lengths are in metres. Keys that
this module adds to the contract are marked (+).

    component   the id of the component under verification
    cad         file, sha256 (of the file as given), volume, mass (volume x material density), bbox, n_faces;
                (+) n_edges, n_solids, area, center_of_mass, diagonal, density, length_unit, scale,
                header_unit, si_file (the converted copy, relative to project.out, or None)
    features    FT-ID -> faces (sorted OCC face tags), area, centroid (area-weighted), kind (the kind of its
                faces, "mixed" when they differ), radius / diameter / axis (when every face has the same),
                normal (planes with one outward normal), length (cylinders: the extent along the axis),
                description; (+) axis_point (a point on the axis, when the faces have one axis line)
    interfaces  INT-ID -> wetted / seat / bolts / external -> face tags, "*" resolved (an interface with a
                `body` role also has that key)
    faces       "tag" -> kind, area, centroid, normal (planes, outward), radius, axis, features, roles
                ("INT-ID:role"); (+) axis_point, concave (a hole or fillet surface), fillet, length,
                thickness (an estimate of the wall behind the face, from rays on a tessellation), bbox,
                neighbors
    edges       reentrant: tag, faces, angle_deg (the largest material angle along the edge), length,
                midpoint; (+) angle_min_deg, kind, points (a polyline along the edge, for distance checks);
                (+) count (the edges between two faces)
    scls        SCL-ID -> start, end, length, ok, message; (+) start_face, end_face, start_distance,
                end_distance, angle_deg (the angle between the line and the surface normal at each end)
    pictures    path (relative to project.out) and caption, when options.pictures
    problems    what the stage could not resolve, one sentence each

How things are measured, and what the limits are
    * Faces and feature queries: meshing-harness `feagent.geometry.analyze` and `feagent.select` (the
      OpenCASCADE kernel through gmsh). A feature is `select` (a geometric query) or `faces` (tags).
    * Length unit. `component.cad.length_unit` is the unit of the numbers in the CAD file. The kernel reads
      the numbers of a STEP file as they are, under a header unit that this module reads itself (the kernel's
      own unit conversion on import does not survive a write, so it is not used). A file whose numbers are
      metres under a millimetre header (what gmsh writes, the example) is read as it is. Other files are
      scaled once by the factor (length_unit / header unit) in a separate process (a uniform scaling of the
      solid, exact for the analytic surfaces) and written to `<out>/geometry/<component>_si.step`; the copy
      is checked face by face against the source (kind, area x factor^2, centre x factor, volume). The
      process is separate to keep the kernel's global state clean. When the header and `length_unit`
      disagree, the decision about the length unit asks for a review. STEP files with no or several length
      units, IGES files and BREP files not in metres raise GeometryError.
    * Exactly one solid. A file with no solid or with several raises GeometryError.
    * Material angle of an edge: `ace.geometry_info` (the angle of the material wedge between the two
      faces), sampled at 9 points along the edge because it changes along the edge where two curved faces
      meet. Re-entrant: the largest sample is over 190 deg (ace.geometry_info.REENTRANT_DEG). An angle that
      peaks between the samples can be missed.
    * Stress classification lines: both ends within 1e-4 of the bounding-box diagonal of a face (exact
      kernel distance), 41 points between the ends inside the solid, and the line between the ends at least
      5e-6 of the diagonal away from every face (exact kernel distance, so a thin void is found). The first
      and last 2e-4 of the diagonal of the line are not checked.
    * Pictures are drawn with matplotlib from a surface tessellation (no OpenGL): polygons are sorted by
      depth, so a picture can show small ordering errors. They are a check of which faces a feature has,
      not a drawing.

Not verified: CAD with more than one solid, non-STEP files, STEP files with a unit other than millimetre in
the header (the unit logic is tested with rewritten headers of the example, not with files from a CAD
system), and freeform (B-spline) faces, for which features have no radius or axis.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dossier import units
from dossier.records import Journal

UNIT_M = {"m": 1.0, "mm": 1e-3, "in": 0.0254}        # metres per unit of Cad.length_unit
# the units the kernel's STEP reader can convert to (its names), in metres
KERNEL_UNITS = {"MM": 1e-3, "M": 1.0, "CM": 1e-2, "UM": 1e-6, "KM": 1e3, "INCH": 0.0254, "FT": 0.3048,
                "MI": 1609.344, "MIL": 2.54e-5, "UIN": 2.54e-8}
CONTRACT_ROLES = ("wetted", "seat", "bolts", "external")
SURFACE_TOL = 1e-4          # an SCL end is on the surface within this fraction of the bounding-box diagonal
SCL_SAMPLES = 41            # points between the ends of an SCL that must be inside the solid
SCL_OBLIQUE_DEG = 20.0      # an SCL further than this from the surface normal at an end gets a finding
ANGLE_FRACTIONS = (0.02, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.98)   # where an edge's angle is sampled
EDGE_POINTS = 49            # points of the polyline given for each re-entrant edge
SIZE_LIMITS = (1e-3, 50.0)  # a bounding-box diagonal outside this range (m) is not usual for one component
LIST_MAX = 20               # face tags listed in one sentence
ROLE_COLOURS = {"wetted": (0.00, 0.45, 0.74), "seat": (0.85, 0.33, 0.10), "bolts": (0.47, 0.67, 0.19),
                "external": (0.93, 0.69, 0.13), "body": (0.49, 0.18, 0.56)}
PALETTE = [(0.00, 0.45, 0.74), (0.85, 0.33, 0.10), (0.47, 0.67, 0.19), (0.49, 0.18, 0.56), (0.93, 0.69, 0.13),
           (0.30, 0.75, 0.93), (0.64, 0.08, 0.18), (0.25, 0.25, 0.25)]


class GeometryError(RuntimeError):
    """The CAD of the component cannot be read or measured. The message says what to change."""


# =========================================================================== small helpers


def _num(x, sig: int = 12) -> float:
    """A float without last-bit noise and without -0.0."""
    return float(f"{float(x):.{sig}g}") + 0.0


def _vec(v, sig: int = 12) -> list[float]:
    return [_num(c, sig) for c in v]


def _unit3(v) -> list[float]:
    """A direction: unit length, tiny components set to 0."""
    a = np.asarray(v, float)
    a = a / (np.linalg.norm(a) or 1.0)
    return [float(round(c, 9)) + 0.0 for c in a]


def _len(x: float) -> str:
    return units.display(units.from_si(float(x), "m"))


def _area(x: float) -> str:
    return units.display(units.from_si(float(x), "m^2"))


def _faces_text(tags) -> str:
    """"f1, f2 and f3" (at most LIST_MAX tags, then "and N more")."""
    names = [f"f{t}" for t in tags]
    if len(names) > LIST_MAX:
        return ", ".join(names[:LIST_MAX]) + f" and {len(names) - LIST_MAX} more"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _quiet():
    """Silence gmsh's C-level output (single_agent.harness.quiet)."""
    from single_agent.harness import quiet
    return quiet()


# =========================================================================== the length unit of the CAD file

_SI_PREFIX = {"$": 1.0, ".KILO.": 1e3, ".HECTO.": 1e2, ".DECA.": 1e1, ".DECI.": 1e-1, ".CENTI.": 1e-2,
              ".MILLI.": 1e-3, ".MICRO.": 1e-6, ".NANO.": 1e-9}
_UNIT_NAME = {1.0: "m", 1e-1: "dm", 1e-2: "cm", 1e-3: "mm", 1e-6: "um", 1e3: "km", 0.0254: "in", 0.3048: "ft"}


def _entity(text: str, ident: str) -> str | None:
    m = re.search(rf"#{ident}\s*=\s*(.*?);", text, re.S)
    return m.group(1) if m else None


def _unit_factor(text: str, ident: str, depth: int = 0) -> float | None:
    """Metres per unit of the STEP length-unit entity #ident; None when it is not a length unit this
    module can read."""
    body = _entity(text, ident)
    if body is None or "LENGTH_UNIT" not in body:
        return None
    m = re.search(r"SI_UNIT\s*\(\s*(\$|\.[A-Z]+\.)\s*,\s*\.METRE\.\s*\)", body)
    if m:
        return _SI_PREFIX.get(m.group(1))
    m = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'[^']*'\s*,\s*#(\d+)\s*\)", body)
    if m and depth < 4:
        ref = _entity(text, m.group(1)) or ""
        mm = re.search(r"LENGTH_MEASURE\s*\(\s*([-+0-9.Ee]+)\s*\)\s*,\s*#(\d+)", ref)
        if mm:
            base = _unit_factor(text, mm.group(2), depth + 1)
            if base is not None:
                return float(mm.group(1)) * base
    return None


def _unit_name(factor: float) -> str:
    for f, name in _UNIT_NAME.items():
        if abs(factor / f - 1.0) < 1e-9:
            return name
    return f"{factor:g} m"


def step_length_unit(path: Path) -> tuple[float | None, str]:
    """The length unit that the header of a STEP file declares: (metres per unit, name), or
    (None, the reason) when the file declares none, several, or one this module cannot read."""
    text = Path(path).read_text(errors="replace")
    ids: set[str] = set()
    for m in re.finditer(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*\(\s*\(([^()]*)\)", text):
        ids |= set(re.findall(r"#(\d+)", m.group(1)))
    factors: list[float] = []
    unread = 0
    for i in sorted(ids, key=int):
        body = _entity(text, i) or ""
        if "LENGTH_UNIT" not in body:
            continue
        f = _unit_factor(text, i)
        if f is None:
            unread += 1
        elif not any(abs(f / g - 1.0) < 1e-9 for g in factors):
            factors.append(f)
    if unread:
        return None, "the header has a length unit that dossier cannot read"
    if not factors:
        return None, "the header declares no length unit"
    if len(factors) > 1:
        return None, "the header declares more than one length unit (" + ", ".join(_unit_name(f) for f in factors) + ")"
    return factors[0], _unit_name(factors[0])


@dataclass(frozen=True)
class SiCad:
    """The CAD file of a component and the file the kernel reads in metres."""

    source: Path                # the file the component names
    path: Path                  # read with the kernel default (target millimetre), its coordinates are metres
    sha256: str                 # of `source`
    length_unit: str            # component.cad.length_unit
    scale: float                # metres per unit of the numbers in `source`
    header_unit: str            # the length unit in the header of the STEP file ("" for a format without one)
    kernel_unit: str            # "MM" (the file is read as it is) or "scaled" (a copy scaled to metres)
    converted: bool             # `path` is a converted copy of `source`


def _kernel_table(path: Path) -> dict:
    """{(dim, tag): (measure, centre, type)} of the faces and solids of a CAD file, read with the kernel
    default (target unit millimetre, set explicitly so that a stale process-wide setting cannot scale it)."""
    import gmsh

    from feacore.geometry.gmsh_tools import gmsh_session
    out = {}
    with _quiet(), gmsh_session("dossier-units"):
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        for dim in (2, 3):
            for _, t in gmsh.model.getEntities(dim):
                out[(dim, t)] = (float(gmsh.model.occ.getMass(dim, t)),
                                 np.asarray(gmsh.model.occ.getCenterOfMass(dim, t), float), gmsh.model.getType(dim, t))
    return out


def _length_unit_id(text: str) -> str:
    """The id of the one length-unit entity of a STEP file (step_length_unit has checked that there is one)."""
    ids: set[str] = set()
    for m in re.finditer(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*\(\s*\(([^()]*)\)", text):
        ids |= set(re.findall(r"#(\d+)", m.group(1)))
    found = [i for i in sorted(ids, key=int) if "LENGTH_UNIT" in (_entity(text, i) or "")]
    if len(found) != 1:
        raise GeometryError("the header of the STEP file does not have exactly one length unit entity")
    return found[0]


def _convert(source: Path, dst: Path, k: float) -> None:
    """Write `dst`: `source` with the header length unit replaced, so that the kernel (default target
    millimetre) gives coordinates that are `k` times those it gives for `source`, that is metres. Only the
    header changes (the numbers and the surfaces are not touched, so the copy is exact). The copy is then
    checked against the source face by face (kind, area x k^2, centre x k, volume x k^3)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    text = Path(source).read_text(errors="replace")
    h, _ = step_length_unit(source)
    uid = _length_unit_id(text)
    ent = re.search(rf"#{uid}\s*=\s*(.*?);", text, re.S)
    top = max(int(i) for i in re.findall(r"^#(\d+)\s*=", text, re.M))
    a_, b_, c_ = top + 1, top + 2, top + 3
    # the new header unit is h x k metres per number: the kernel then gives k times what it gives for `source`
    factor = f"{float(h * k):.16E}"            # a STEP real always has a decimal point
    extra = (f"#{a_} = LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE({factor}), #{b_});\n"
             f"#{b_} = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );\n"
             f"#{c_} = DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n")
    # a conversion-based unit, as the STEP files of CAD systems declare an inch
    unit = f"( CONVERSION_BASED_UNIT('SCALED', #{a_}) LENGTH_UNIT() NAMED_UNIT(#{c_}) )"
    new = text[:ent.start(1)] + unit + text[ent.end(1):]
    end = new.rfind("ENDSEC;")                 # the end of the data section
    new = new[:end] + extra + new[end:]
    tmp.write_text(new)
    a, b = _kernel_table(source), _kernel_table(tmp)
    problems = []
    if sorted(a) != sorted(b):
        problems.append(f"the copy has {len(b)} faces and solids, the source has {len(a)}")
    else:
        size = max((np.abs(c).max() for _, c, _ in b.values()), default=1.0) or 1.0
        for key, (measure, centre, kind) in a.items():
            mb, cb, kb = b[key]
            want = measure * k ** key[0]
            if kind != kb or abs(mb - want) > 1e-6 * abs(want) or np.abs(cb - centre * k).max() > 1e-6 * size:
                problems.append(f"{'face' if key[0] == 2 else 'solid'} {key[1]} differs")
    if problems:
        tmp.unlink(missing_ok=True)
        raise GeometryError(f"{source.name}: the copy in metres does not agree with the source ({'; '.join(problems[:5])})")
    tmp.replace(dst)


def si_cad(project, component_id: str | None = None) -> SiCad:
    """The CAD of a component as a file whose coordinates the kernel reads in metres: the file itself, or a
    converted copy in `<out>/geometry/` (made once per source file). Raises GeometryError."""
    comp = project.components[component_id or project.file.subject]
    src = project.cad_path(comp.id)
    if not src.is_file():
        raise GeometryError(f"component {comp.id}: the CAD file {src} does not exist")
    sha = sha256_of(src)
    unit = comp.cad.length_unit
    scale = UNIT_M[unit]
    ext = src.suffix.lower()
    if ext in (".step", ".stp"):
        h_scale, h_name = step_length_unit(src)
        if h_scale is None:
            raise GeometryError(f"{src.name}: {h_name}. dossier needs a STEP file with one length unit in its header.")
        k = scale * 1e-3 / h_scale      # metres per unit of the numbers the kernel gives by default
        kernel = "MM" if abs(k - 1.0) < 1e-12 else "scaled"
    elif ext == ".brep":
        if unit != "m":
            raise GeometryError(f"{src.name}: a BREP file has no length unit and dossier cannot scale it. Give the "
                                f"CAD as a STEP file, or a BREP file in metres.")
        h_name, kernel, k = "", "MM", 1.0
    else:
        raise GeometryError(f"{src.name}: dossier reads STEP (.step, .stp) and BREP (.brep) files")
    if kernel == "MM":
        return SiCad(src, src, sha, unit, scale, h_name, kernel, False)
    dst = project.out / "geometry" / f"{comp.id}_si.step"
    meta = dst.with_suffix(".json")
    want = {"source_sha256": sha, "length_unit": unit, "kernel_unit": kernel, "factor": k}
    current = None
    if dst.is_file() and meta.is_file():
        try:
            current = json.loads(meta.read_text())
        except ValueError:
            current = None
    if not current or any(current.get(key) != v for key, v in want.items()) or current.get("sha256") != sha256_of(dst):
        _convert(src, dst, k)
        meta.write_text(json.dumps({**want, "sha256": sha256_of(dst)}, indent=1) + "\n")
    return SiCad(src, dst, sha, unit, scale, h_name, kernel, True)


# =========================================================================== the CAD model, the features, the part

_MODELS: dict[tuple[str, str], object] = {}        # (sha256 of the source, length unit) -> feagent GeometryModel


def model_of(project, component_id: str | None = None):
    """The feagent GeometryModel of a component's CAD, in metres (cached per process and CAD file)."""
    cad = si_cad(project, component_id)
    key = (cad.sha256, cad.length_unit)
    if key not in _MODELS:
        from feagent.errors import FeagentError
        from feagent.geometry import analyze
        try:
            with _quiet():
                model = analyze(str(cad.path))
        except FeagentError as e:
            raise GeometryError(f"{cad.source.name}: {e}") from e
        if len(model.volumes) != 1:
            raise GeometryError(f"{cad.source.name}: the CAD file holds {len(model.volumes)} solids. dossier verifies "
                                f"one component that is one solid. Export the component as one solid.")
        _MODELS[key] = model
    return _MODELS[key]


def feature_faces(project, component_id: str | None = None) -> tuple[dict[str, list[int]], dict[str, dict]]:
    """Every feature of a component as sorted face tags, and what went wrong for the features that did not
    resolve: {feature id: {"kind": "query" | "tags", "message": text, ...}}."""
    from feagent.errors import QueryError
    from feagent.select import select_faces
    comp = project.components[component_id or project.file.subject]
    model = model_of(project, comp.id)
    known = {f.tag for f in model.faces}
    faces: dict[str, list[int]] = {}
    errors: dict[str, dict] = {}
    for fid, fd in comp.features.items():
        if fd.select is not None:
            try:
                faces[fid] = select_faces(model, fd.select)
            except QueryError as e:
                faces[fid] = []
                errors[fid] = {"kind": "query", "message": str(e)}
        else:
            tags = sorted(set(int(t) for t in fd.faces or []))
            missing = [t for t in tags if t not in known]
            faces[fid] = [t for t in tags if t in known]
            if missing:
                errors[fid] = {"kind": "tags", "message": f"no face with tag {', '.join(map(str, missing))}",
                               "missing": missing}
    return faces, errors


def _blind_part():
    from single_agent.part import BlindPart

    class ComponentPart(BlindPart):
        """The CAD of a dossier component as a feacore Problem: groups f<tag>, the named groups in
        `groups` (features, interface roles) and `solid`. The CAD is in metres."""

        tags = ("dossier",)

        def build_geometry(self) -> None:
            import gmsh
            # explicit: a target unit left by an earlier STEP import of this process would scale the part
            gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
            super().build_geometry()

    return ComponentPart


_PART_CLASS = None


def part_for(project, component_id: str | None = None, groups: dict[str, list[int]] | None = None):
    """The registered feacore Problem of a component (see dossier.meshing.part_for). `groups` are the named
    face groups its geometry gets besides f<tag> and solid; default: every feature that has faces."""
    global _PART_CLASS
    from feacore.problems import registry
    comp = project.components[component_id or project.file.subject]
    cad = si_cad(project, comp.id)
    if groups is None:
        groups = {fid: tags for fid, tags in feature_faces(project, comp.id)[0].items()}
    groups = {name: sorted(set(int(t) for t in tags)) for name, tags in groups.items() if tags}
    reserved = [n for n in groups if n == "solid" or re.fullmatch(r"f\d+", n)]
    if reserved:
        raise GeometryError(f"the group name(s) {reserved} are reserved for the faces and the solid of the mesh")
    if _PART_CLASS is None:
        _PART_CLASS = _blind_part()
    label = f"dossier-{comp.id}-{cad.sha256[:10]}"
    registry.discover()
    part = registry._REGISTRY.get(label)
    if part is None:
        part = registry.register(_PART_CLASS(cad.path, label))
    elif not isinstance(part, _PART_CLASS):
        raise GeometryError(f"the problem id {label!r} is already in use in the feacore registry")
    part.step_path = Path(cad.path)        # the same CAD content (the label holds its hash); the path can differ
    part.title = f"{comp.name} ({comp.id})"
    part.groups = groups
    return part


# =========================================================================== measures


def _face_record(f, fillets: set[int]) -> dict:
    plane = f.kind == "plane" and f.normal is not None
    return {
        "kind": f.kind,
        "area": _num(f.area),
        "centroid": _vec(f.center),
        "normal": _unit3(f.normal) if plane else None,
        "radius": _num(f.radius) if f.radius is not None else None,
        "axis": _unit3(f.axis) if f.axis is not None else None,
        "axis_point": _vec(f.axis_point) if f.axis_point is not None else None,
        "concave": f.concave,
        "fillet": f.tag in fillets,
        "length": None,
        "thickness": _num(f.thickness, 4) if f.thickness is not None else None,
        "bbox": [_vec(f.bbox[0], 9), _vec(f.bbox[1], 9)],
        "neighbors": [int(t) for t in f.neighbors],
        "features": [],
        "roles": [],
    }


def _same(values, rtol: float) -> bool:
    values = list(values)
    return all(v is not None for v in values) and max(values) - min(values) <= rtol * max(abs(v) for v in values)


def _measure_feature(tags: list[int], faces: dict[str, dict], diag: float) -> dict:
    """Area, centroid, kind and the shape facts that every face of the feature shares."""
    recs = [faces[str(t)] for t in tags]
    out = {"faces": list(tags), "area": 0.0, "centroid": None, "kind": None, "radius": None, "diameter": None,
           "axis": None, "normal": None, "length": None, "axis_point": None}
    if not recs:
        return out
    area = sum(r["area"] for r in recs)
    out["area"] = _num(area)
    out["centroid"] = _vec(sum(r["area"] * np.asarray(r["centroid"]) for r in recs) / area)
    kinds = {r["kind"] for r in recs}
    out["kind"] = kinds.pop() if len(kinds) == 1 else "mixed"
    if out["kind"] in ("cylinder", "torus", "sphere") and _same([r["radius"] for r in recs], 1e-3):
        out["radius"] = _num(float(np.mean([r["radius"] for r in recs])), 9)
        out["diameter"] = _num(2.0 * out["radius"], 9)
    axes = [r["axis"] for r in recs]
    if all(a is not None for a in axes) and all(abs(float(np.dot(axes[0], a))) > 1.0 - 1e-6 for a in axes):
        out["axis"] = list(axes[0])
        pts = [r["axis_point"] for r in recs]
        if all(p is not None for p in pts):
            a0, p0 = np.asarray(axes[0]), np.asarray(pts[0])
            off = [np.linalg.norm((np.asarray(p) - p0) - np.dot(np.asarray(p) - p0, a0) * a0) for p in pts]
            if max(off) <= 1e-6 * diag + 1e-3 * (out["radius"] or 0.0):
                out["axis_point"] = list(pts[0])
    normals = [r["normal"] for r in recs]
    if out["kind"] == "plane" and all(n is not None for n in normals) \
            and all(float(np.dot(normals[0], n)) > 1.0 - 1e-6 for n in normals):
        out["normal"] = list(normals[0])
    if out["kind"] == "cylinder" and out["axis"] is not None:
        spans = [r.get("_span") for r in recs]
        if all(s is not None for s in spans):
            if out["axis_point"] is not None:       # one cylinder in several faces: the extent of all of them
                out["length"] = _num(max(s[1] for s in spans) - min(s[0] for s in spans), 9)
            elif _same([s[1] - s[0] for s in spans], 1e-3):      # equal holes side by side: the length of one
                out["length"] = _num(float(np.mean([s[1] - s[0] for s in spans])), 9)
    return out


def _kernel_facts(part, model, scls, diag: float) -> dict:
    """What needs the CAD kernel itself, in one gmsh session: the edges between two faces with the material
    angle along them, the extent of each cylinder along its axis, and the checks of the stress
    classification lines."""
    import gmsh

    from ace.geometry_info import _Describer, _unit
    from feacore.geometry.gmsh_tools import gmsh_session
    tol = SURFACE_TOL * diag
    out: dict = {"edges": [], "spans": {}, "scls": {}, "unmeasured_edges": [], "n_edges": 0}
    with _quiet(), gmsh_session(part.id + "-geometry"):
        part.build_geometry()
        gmsh.model.occ.synchronize()
        d = _Describer(part)
        vols = list(d.domain)
        bnd = sorted({abs(t) for _, t in gmsh.model.getBoundary([(3, v) for v in vols], combined=True, oriented=False)})
        bset = set(bnd)
        com = np.zeros(3)
        for v in vols:
            com += gmsh.model.occ.getMass(3, v) * np.asarray(gmsh.model.occ.getCenterOfMass(3, v))
        out["volume"] = float(sum(gmsh.model.occ.getMass(3, v) for v in vols))
        out["center_of_mass"] = com / out["volume"]

        # ---- edges between two faces: the material angle at several points
        edge_tags = sorted({abs(t) for _, t in gmsh.model.getBoundary([(2, b) for b in bnd], combined=False,
                                                                       oriented=False)})
        for e in edge_tags:
            up = sorted(int(b) for b in gmsh.model.getAdjacencies(1, e)[0] if int(b) in bset)
            length = float(gmsh.model.occ.getMass(1, e))
            if len(up) != 2 or length < 1e-9 * diag:
                continue                        # the seam of a closed face, or a degenerate edge
            out["n_edges"] += 1
            lo, hi = gmsh.model.getParametrizationBounds(1, e)
            angles = []
            for fr in ANGLE_FRACTIONS:
                t = lo[0] + fr * (hi[0] - lo[0])
                try:
                    at = np.asarray(gmsh.model.getValue(1, e, [t]))
                    a = d.interior_angle(e, up, at, _unit(gmsh.model.getDerivative(1, e, [t])))
                except Exception:  # noqa: BLE001  gmsh raises plain Exception; an angle not measured is reported
                    a = None
                if a is not None:
                    angles.append(float(a))
            if not angles:
                out["unmeasured_edges"].append(int(e))
                continue
            ts = np.linspace(lo[0], hi[0], EDGE_POINTS)
            pts = np.asarray(gmsh.model.getValue(1, e, ts.tolist())).reshape(-1, 3)
            out["edges"].append({"tag": int(e), "faces": up, "angle_deg": round(max(angles), 1),
                                 "angle_min_deg": round(min(angles), 1), "length": _num(length, 9),
                                 "midpoint": _vec(gmsh.model.getValue(1, e, [0.5 * (lo[0] + hi[0])]), 9),
                                 "kind": gmsh.model.getType(1, e).lower(),
                                 "points": [_vec(p, 9) for p in pts]})

        # ---- extent of every cylinder along its axis, from its boundary curves (exact: the surface is ruled)
        for f in model.faces:
            if f.kind != "cylinder" or f.axis is None:
                continue
            ext = []
            for e in f.edges:
                try:
                    lo, hi = gmsh.model.getParametrizationBounds(1, e)
                    p = np.asarray(gmsh.model.getValue(1, e, np.linspace(lo[0], hi[0], 257).tolist())).reshape(-1, 3)
                except Exception:  # noqa: BLE001
                    ext = []
                    break
                ext.append(p @ np.asarray(f.axis))
            if ext:
                t = np.concatenate(ext)
                out["spans"][f.tag] = (float(t.min()), float(t.max()))

        # ---- stress classification lines (last: the probe points and lines are added to the kernel model)
        occ = gmsh.model.occ

        def inside(x) -> bool:
            return any(gmsh.model.isInside(3, v, [float(c) for c in x]) for v in vols)

        for scl in scls:
            a, b = np.asarray(scl.start, float), np.asarray(scl.end, float)
            length = float(np.linalg.norm(b - a))
            rec = {"start": _vec(a), "end": _vec(b), "length": _num(length), "ok": True, "message": "",
                   "start_face": None, "end_face": None, "start_distance": None, "end_distance": None,
                   "angle_deg": [None, None]}
            msgs: list[str] = []
            if length <= 10.0 * tol:
                msgs.append(f"The line is {_len(length)} long. The check needs a line of more than {_len(10 * tol)}.")
            else:
                u = (b - a) / length
                for i, (name, x) in enumerate((("start", a), ("end", b))):
                    pt = occ.addPoint(*[float(c) for c in x])
                    dist, face = min((float(occ.getDistance(0, pt, 2, f)[0]), f) for f in bnd)
                    rec[f"{name}_face"], rec[f"{name}_distance"] = int(face), _num(dist, 6)
                    if dist > tol:
                        msgs.append(f"The {name} point is {_len(dist)} from the surface of the part. "
                                    f"The tolerance is {_len(tol)}.")
                    else:
                        try:
                            _, uv = gmsh.model.getClosestPoint(2, face, [float(c) for c in x])
                            n = _unit(gmsh.model.getNormal(face, list(uv)))
                            rec["angle_deg"][i] = round(math.degrees(math.acos(min(1.0, abs(float(n @ u))))), 1)
                        except Exception:  # noqa: BLE001  the angle is extra information
                            pass
                p1, p2 = a + 2.0 * tol * u, b - 2.0 * tol * u
                outside = sum(not inside(p1 + s * (p2 - p1)) for s in np.linspace(0.0, 1.0, SCL_SAMPLES))
                rec["outside_fraction"] = round(outside / SCL_SAMPLES, 3)
                if outside:
                    msgs.append("The line leaves the material between its ends.")
                else:
                    line = occ.addLine(occ.addPoint(*[float(c) for c in p1]), occ.addPoint(*[float(c) for c in p2]))
                    clearance = min(float(occ.getDistance(1, line, 2, f)[0]) for f in bnd)
                    rec["clearance"] = _num(clearance, 6)
                    if clearance <= 0.05 * tol:
                        msgs.append("The line touches the surface of the part between its ends.")
            if msgs:
                rec["ok"], rec["message"] = False, " ".join(msgs)
            out["scls"][scl.id] = rec
    return out


# =========================================================================== the stage


def build(project, opts) -> tuple[dict, Journal]:
    """The geometry stage (see the module docstring). Raises GeometryError when the CAD cannot be read."""
    comp = project.subject
    j = Journal("geometry", "engine:geometry")
    log = getattr(opts, "log", None) or (lambda s: None)
    problems: list[str] = []

    cad = si_cad(project, comp.id)
    model = model_of(project, comp.id)
    diag = float(model.diag)
    lo, hi = np.asarray(model.bbox[0], float), np.asarray(model.bbox[1], float)
    step = 10.0 ** math.floor(math.log10(diag * 1e-5))          # the kernel adds its tolerance to a bounding box
    bbox = [[_num(round(c / step) * step, 9) for c in lo], [_num(round(c / step) * step, 9) for c in hi]]
    extent = [bbox[1][i] - bbox[0][i] for i in range(3)]

    # ---- faces
    fillets = {t for ft in model.features if ft.kind == "fillet" for t in ft.faces}
    fillets |= {f.tag for f in model.faces if f.kind == "torus" and f.concave}
    faces = {str(f.tag): _face_record(f, fillets) for f in sorted(model.faces, key=lambda f: f.tag)}
    all_tags = sorted(int(t) for t in faces)
    for h in model.hazards:
        if h.kind == "analysis_incomplete":
            problems.append(f"The analysis of the CAD model is not complete. The kernel gives this message: \"{h.message}\"")

    # ---- features: resolve, then measure (the kernel session below gives the cylinder lengths)
    feat_faces, feat_errors = feature_faces(project, comp.id)
    part = part_for(project, comp.id, groups=feat_faces)
    facts = _kernel_facts(part, model, comp.scls, diag)
    for tag, span in facts["spans"].items():
        faces[str(tag)]["_span"] = span
        faces[str(tag)]["length"] = _num(span[1] - span[0], 9)
    features: dict[str, dict] = {}
    for fid, fd in comp.features.items():
        tags = feat_faces[fid]
        features[fid] = {**_measure_feature(tags, faces, diag), "description": fd.description}
        for t in tags:
            faces[str(t)]["features"].append(fid)
        _feature_findings(j, problems, model, fid, fd, tags, feat_errors.get(fid))
    for rec in faces.values():
        rec.pop("_span", None)

    # ---- interfaces: roles to faces, "*" resolved
    interfaces, star = _resolve_interfaces(j, comp, feat_faces, faces, all_tags)
    _interface_findings(j, problems, comp, feat_faces, interfaces, star)

    # ---- edges
    reentrant = _reentrant_edges(j, comp, facts, problems)

    # ---- stress classification lines
    scls = facts["scls"]
    for scl in comp.scls:
        _scl_findings(j, problems, scl, scls[scl.id])

    # ---- the CAD as a whole
    volume = float(model.volumes[0].volume)
    material = project.material_of(comp.id)
    density = float(units.si(material.rho.quantity()))
    geo = {
        "component": comp.id,
        "cad": {
            "file": comp.cad.file, "sha256": cad.sha256, "volume": _num(volume), "mass": _num(volume * density, 9),
            "bbox": bbox, "n_faces": len(faces),
            "n_edges": facts["n_edges"], "n_solids": len(model.volumes),
            "area": _num(sum(r["area"] for r in faces.values())),
            "center_of_mass": _vec(facts["center_of_mass"], 9), "diagonal": _num(diag, 9), "density": density,
            "length_unit": cad.length_unit, "scale": cad.scale, "header_unit": cad.header_unit,
            "si_file": str(cad.path.relative_to(project.out)) if cad.converted else None,
        },
        "features": features,
        "interfaces": interfaces,
        "faces": faces,
        "edges": {"reentrant": reentrant, "count": facts["n_edges"]},
        "scls": scls,
        "pictures": [],
        "problems": problems,
    }
    _unit_records(j, problems, comp, cad, extent, diag, volume, volume * density, material)
    j.assume(
        subject=comp.id, key="nominal-geometry", title="Nominal geometry",
        statement="The CAD model gives the nominal dimensions of the part. The analysis uses these dimensions.",
        rationale="The project has no data about the manufacturing tolerances of the part.",
        impact="A wall that is thinner than the nominal wall has a higher stress. The margins of safety are then "
               "smaller than the report shows.",
        verify_by="Compare the smallest wall thickness that the drawing permits with the CAD model.",
        conservative=False, links=[("source", f"doc:{comp.drawing}")] if comp.drawing else [])

    if getattr(opts, "pictures", False):
        try:
            geo["pictures"] = _pictures(project, comp, part, geo)
        except Exception as e:  # noqa: BLE001  a picture is an aid: the reason goes into the stage file
            problems.append(f"The geometry stage did not make the pictures. The error is: \"{type(e).__name__}: {e}\"")
    log(f"geometry: {len(faces)} faces, {len(features)} features, {len(reentrant)} re-entrant edges, "
        f"{len(j.findings)} findings")
    return geo, j


# --------------------------------------------------------------------------- records of the stage


def _feature_findings(j: Journal, problems: list[str], model, fid: str, fd, tags: list[int], error: dict | None) -> None:
    selection = fd.select if fd.select is not None else {"faces": fd.faces}
    if error and error["kind"] == "query":
        f = j.find(rule="geometry.feature_query", severity="error", category="model", subject=fid, key="query",
                   title=f"Selection of the feature {fid}",
                   statement=f"The geometry stage cannot read the selection of the feature {fid}.",
                   rationale=f"The query language gives this message: \"{error['message']}\"",
                   suggestion="Correct the keys of the selection in the component file.",
                   values={"selection": selection})
        problems.append(f.statement)
        return
    if error and error["kind"] == "tags":
        missing = error["missing"]
        f = j.find(rule="geometry.feature_faces", severity="error", category="model", subject=fid, key="face-tags",
                   title=f"Face tags of the feature {fid}",
                   statement=f"The feature {fid} lists {_faces_text(missing)}. The CAD model has no face with "
                             f"{'this tag' if len(missing) == 1 else 'these tags'}.",
                   rationale=f"The CAD model has {len(model.faces)} faces. A new export of the CAD can change the "
                             f"face tags.",
                   suggestion="Select the faces of the feature with a geometric query.",
                   values={"selection": selection, "missing": missing})
        problems.append(f.statement)
    if not tags:
        explanation = ""
        if fd.select is not None:
            try:
                from feagent.select import explain_miss
                explanation = explain_miss(model, fd.select, fd.expect)
            except Exception as e:  # noqa: BLE001  the explanation is extra information
                explanation = f"{type(e).__name__}: {e}"
        f = j.find(rule="geometry.feature_empty", severity="error", category="model", subject=fid, key="no-faces",
                   title=f"The feature {fid} has no face",
                   statement=f"The selection of the feature {fid} gives no face.",
                   rationale="No face of the CAD model agrees with all the keys of the selection. The values of this "
                             "finding show the selection and the nearest faces.",
                   suggestion="Correct the selection of the feature in the component file.",
                   values={"selection": selection, "expect": fd.expect, "explanation": explanation})
        problems.append(f.statement)
        return
    if fd.expect is not None and len(tags) != fd.expect:
        f = j.find(rule="geometry.feature_count", severity="error", category="model", subject=fid, key="face-count",
                   title=f"Number of faces of the feature {fid}",
                   statement=f"The selection of the feature {fid} gives {_count(len(tags), 'face')}. The component "
                             f"file expects {_count(fd.expect, 'face')}.",
                   rationale=f"The selection gives {_faces_text(tags)}. The values of this finding show the selection.",
                   suggestion="Make the selection more specific, or correct the expected number of faces.",
                   values={"selection": selection, "faces": tags, "expect": fd.expect})
        problems.append(f.statement)


def _resolve_interfaces(j: Journal, comp, feat_faces: dict, faces: dict, all_tags: list[int]):
    """Role -> face tags per interface. "*" is every face that no other role of any interface uses."""
    used: set[int] = set()
    for it in comp.interfaces:
        for feats in it.features.values():
            for fid in feats:
                if fid != "*":
                    used |= set(feat_faces.get(fid, []))
    star = sorted(set(all_tags) - used)
    interfaces: dict[str, dict] = {}
    for it in comp.interfaces:
        roles = {r: [] for r in CONTRACT_ROLES}
        for role, feats in it.features.items():
            tags: set[int] = set()
            for fid in feats:
                tags |= set(star) if fid == "*" else set(feat_faces.get(fid, []))
            roles[role] = sorted(tags)
            for t in roles[role]:
                faces[str(t)]["roles"].append(f"{it.id}:{role}")
            if "*" in feats:
                listed = (f"The faces are {_faces_text(star)}." if len(star) <= LIST_MAX
                          else f"The first faces are {_faces_text(star[:LIST_MAX])}. The file geometry.json lists "
                               f"all of them.")
                j.decide(
                    subject=it.id, key=f"{role}-faces", title=f"Faces of the selection * in the interface {it.id}",
                    choice=f"The selection * gives {_count(len(star), 'face')} to the role {role} of the interface "
                           f"{it.id}. {listed}" if star else
                           f"The selection * gives no face to the role {role} of the interface {it.id}.",
                    rationale=f"The component file gives the selection * for this role. The selection takes each face "
                              f"that has no other role in an interface. The CAD model has {_count(len(all_tags), 'face')}, "
                              f"and other roles use {len(used)} of them.",
                    alternatives=[("Name each feature of this role in the component file.",
                                   "The selection * cannot miss a face. A list of names can.")],
                    basis="rule", links=[("about", comp.id)])
        interfaces[it.id] = roles
    for rec in faces.values():
        rec["roles"] = sorted(set(rec["roles"]))
    return interfaces, star


def _interface_findings(j: Journal, problems: list[str], comp, feat_faces: dict, interfaces: dict, star: list[int]) -> None:
    # a face that is wetted in two interfaces
    ids = [it.id for it in comp.interfaces]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            both = sorted(set(interfaces[a]["wetted"]) & set(interfaces[b]["wetted"]))
            if not both:
                continue
            feats = sorted(fid for fid, tags in feat_faces.items() if set(tags) & set(both))
            one = len(both) == 1
            f = j.find(rule="geometry.wetted_overlap", severity="major", category="model", subject=a,
                       key=f"wetted-overlap-{b}", title=f"Wetted faces in the interfaces {a} and {b}",
                       statement=f"The face{'' if one else 's'} {_faces_text(both)} {'is a wetted face' if one else 'are wetted faces'} "
                                 f"of the interface {a} and of the interface {b}.",
                       rationale="A wetted face gets the pressure and the fluid temperature of its interface. "
                                 f"{'This face gets' if one else 'These faces get'} the loads of two connections in the same phase.",
                       suggestion=f"Give {'this face' if one else 'each of these faces'} to the wetted role of one interface only.",
                       values={"faces": both, "features": feats, "interfaces": [a, b]},
                       links=[("about", b), *[("about", fid) for fid in feats]])
            problems.append(f.statement)
    # a feature that no interface names
    named = {fid for it in comp.interfaces for feats in it.features.values() for fid in feats}
    star_users = [(it.id, role) for it in comp.interfaces for role, feats in it.features.items() if "*" in feats]
    for fid in comp.features:
        if fid in named:
            continue
        tags = feat_faces.get(fid, [])
        if tags and star_users and set(tags) <= set(star):
            iid, role = star_users[0]
            why = (f"The {'face' if len(tags) == 1 else 'faces'} of the feature {'is' if len(tags) == 1 else 'are'} in the "
                   f"role {role} of the interface {iid}, through the selection *.")
        else:
            why = "The faces of the feature have no role in an interface."
        j.find(rule="geometry.feature_unused", severity="info", category="model", subject=fid, key="no-interface",
               title=f"The feature {fid} is in no interface",
               statement=f"No interface names the feature {fid}.", rationale=why,
               suggestion="No action is necessary if the feature is only a location for results.",
               values={"faces": tags})


def _reentrant_edges(j: Journal, comp, facts: dict, problems: list[str]) -> list[dict]:
    from ace.geometry_info import REENTRANT_DEG
    reentrant = [e for e in facts["edges"] if e["angle_deg"] > REENTRANT_DEG]
    if facts["unmeasured_edges"]:
        n = len(facts["unmeasured_edges"])
        problems.append(f"The geometry stage did not measure the material angle of {_count(n, 'edge')} "
                        f"(CAD edge {', '.join(map(str, facts['unmeasured_edges'][:LIST_MAX]))}). "
                        f"A re-entrant edge can be among them.")
    tags = [str(e["tag"]) for e in reentrant]
    shown = tags[:LIST_MAX]
    if not reentrant:
        choice, extra = "The geometry stage marks no edge of the CAD model as re-entrant.", ""
    else:
        if len(tags) == 1:
            choice = f"The geometry stage marks 1 edge of the CAD model as re-entrant. The CAD edge tag is {tags[0]}."
        else:
            listed = ", ".join(shown[:-1]) + " and " + shown[-1]
            choice = (f"The geometry stage marks {len(tags)} edges of the CAD model as re-entrant. "
                      + (f"The CAD edge tags are {listed}." if len(tags) <= LIST_MAX else
                         f"The first CAD edge tags are {listed}. The file geometry.json lists all of them."))
        extra = f" The largest material angle of this part is {max(e['angle_deg'] for e in reentrant):g} deg."
    j.decide(
        subject=comp.id, key="reentrant-edges", title="Re-entrant edges of the part", choice=choice,
        rationale=f"An edge is re-entrant when the material angle between its two faces is more than "
                  f"{REENTRANT_DEG:g} deg. The stress of a linear elastic analysis has no limit at such an edge. "
                  f"The geometry stage measures the angle at {len(ANGLE_FRACTIONS)} points of each edge.{extra}",
        alternatives=[("Use a limit of 180 deg.",
                       "Two faces that meet at a fillet give an angle near 180 deg. The margin of 10 deg keeps these "
                       "smooth edges out.")],
        basis="rule")
    return sorted(reentrant, key=lambda e: e["tag"])


def _scl_findings(j: Journal, problems: list[str], scl, rec: dict) -> None:
    if not rec["ok"]:
        f = j.find(rule="geometry.scl", severity="major", category="model", subject=scl.id, key="not-through-material",
                   title=f"The line {scl.id} is not through the material",
                   statement=f"The stress classification line {scl.id} is not a line through a wall of the part.",
                   rationale=rec["message"],
                   suggestion="Put the two ends of the line on opposite surfaces of one wall.",
                   values={k: rec.get(k) for k in ("start", "end", "start_distance", "end_distance", "outside_fraction",
                                                   "clearance") if rec.get(k) is not None})
        problems.append(f.statement)
        return
    angles = [a for a in rec["angle_deg"] if a is not None]
    if angles and max(angles) > SCL_OBLIQUE_DEG:
        j.find(rule="geometry.scl_oblique", severity="minor", category="model", subject=scl.id, key="oblique",
               title=f"The line {scl.id} is not normal to the surface",
               statement=f"The stress classification line {scl.id} is not normal to the surface of the wall.",
               rationale=f"The angle between the line and the normal of the surface is {rec['angle_deg'][0]:g} deg at the "
                         f"start point and {rec['angle_deg'][1]:g} deg at the end point. The limit of this check is "
                         f"{SCL_OBLIQUE_DEG:g} deg.",
               suggestion="Make the line normal to the middle surface of the wall.",
               values={"angle_deg": rec["angle_deg"]})


def _unit_records(j: Journal, problems: list[str], comp, cad: SiCad, extent: list[float], diag: float, volume: float,
                  mass: float, material) -> None:
    """The decision about the length unit (with the overall size, so that a person can check it), and a
    finding when the size is not usual for one component."""
    names = {"m": "metres", "mm": "millimetres", "in": "inches"}
    unit = names[cad.length_unit]
    header = {"m": "metres", "mm": "millimetres", "in": "inches"}.get(cad.header_unit, cad.header_unit)
    agree = cad.header_unit == cad.length_unit
    if cad.scale == 1.0:
        choice = "The geometry stage reads the numbers in the CAD file as metres."
    else:
        choice = (f"The geometry stage reads the numbers in the CAD file as {unit}. It multiplies them by "
                  f"{cad.scale:g} to get metres.")
    if not cad.header_unit:
        head = "The CAD file has no header with a length unit."
    elif agree:
        head = f"The header of the CAD file declares {header} too."
    else:
        head = (f"The header of the CAD file declares {header}. The geometry stage uses the unit of the component "
                f"file.")
    j.decide(
        subject=comp.id, key="length-unit", title="Length unit of the CAD file", choice=choice,
        rationale=f"The component file gives {unit} as the length unit of the CAD file. {head} "
                  f"The part is then {_len(extent[0])} long in x, {_len(extent[1])} in y and {_len(extent[2])} in z. "
                  f"Its volume is {units.display(units.from_si(volume, 'm^3'))} and its mass is "
                  f"{units.display(units.from_si(mass, 'kg'))}.",
        alternatives=[("Use the length unit in the header of the CAD file.",
                       "The header does not always give the unit of the numbers. Some programs write millimetres "
                       "for a model in metres.")] if cad.header_unit else [],
        basis="input", needs_review=not agree, links=[("uses", material.id)])
    if not SIZE_LIMITS[0] <= diag <= SIZE_LIMITS[1]:
        f = j.find(rule="geometry.length_unit", severity="major", category="model", subject=comp.id, key="size",
                   title="Size of the part",
                   statement=f"The bounding box of the part is {units.fmt(units.from_si(diag, 'm'), 'm')} from corner to "
                             f"corner. This size is not usual for one component.",
                   rationale=f"The component file gives {unit} as the length unit of the CAD file.",
                   suggestion="Make sure that the length unit of the CAD file is correct.",
                   values={"diagonal": diag, "length_unit": cad.length_unit, "header_unit": cad.header_unit})
        problems.append(f.statement)


# =========================================================================== pictures


def _clip(tri: np.ndarray, origin: np.ndarray, n: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """The parts of triangles (m, 3, 3) on the side (x - origin) . n <= 0, as polygons of 3 or 4 points,
    and the index of the triangle each polygon comes from."""
    d = (tri - origin) @ n
    inside = d <= 0
    whole = inside.all(axis=1)
    polys = [t for t in tri[whole]]
    index = list(np.nonzero(whole)[0])
    for i in np.nonzero(~whole & inside.any(axis=1))[0]:
        t, dt, poly = tri[i], d[i], []
        for k in range(3):
            p, q, dp, dq = t[k], t[(k + 1) % 3], dt[k], dt[(k + 1) % 3]
            if dp <= 0:
                poly.append(p)
            if (dp <= 0) != (dq <= 0):
                poly.append(p + (q - p) * (dp / (dp - dq)))
        if len(poly) >= 3:
            polys.append(np.asarray(poly))
            index.append(i)
    return polys, np.asarray(index, dtype=int)


def _draw(ax, P, tris: dict, colour: dict, view: tuple[float, float], cut: np.ndarray | None, diag: float,
          labels: dict | None = None, lines: list | None = None, edges: list | None = None) -> None:
    """One panel: the faces in `colour` ({tag: rgb}; the others grey) seen from `view` (elevation, azimuth).
    With `cut` (a point), only the half of the part behind the plane through it that faces the camera is
    drawn, and the inside of the outer skin is dark: it reads as the cut material."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ace.solve_feedback import _unoccluded
    el, az = np.radians(view)
    cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    polys, cols, occluders, label_at = [], [], [], {}
    for f, T in tris.items():
        tri = P[T]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(nrm, axis=1)
        nrm = nrm / np.maximum(area2, 1e-300)[:, None]
        facing = nrm @ cam
        if cut is not None:
            behind = ((tri.mean(axis=1) - cut) @ cam) <= 0
        else:
            behind = np.ones(len(tri), bool)
        base = np.asarray(colour.get(f, (0.80, 0.80, 0.80)))
        for front in (True, False):
            sel = (facing > 1e-3) if front else (facing < -1e-3)
            if not front and cut is None:
                continue
            part = tri[sel]
            if not len(part):
                continue
            shade = 0.45 + 0.55 * np.abs(facing[sel])
            pieces = list(part)
            if cut is not None:
                pieces, index = _clip(part, cut, cam)
                shade = shade[index]
            for p, s in zip(pieces, shade):
                polys.append(p)
                cols.append((*(base * s), 1.0) if front else (0.30, 0.30, 0.33, 1.0))
        vis = (facing > 1e-3) & behind
        occluders.append(tri[behind])
        if labels and f in labels and vis.any():
            idx = np.nonzero(vis)[0]
            label_at[f] = (tri[idx].mean(axis=1), area2[idx])
    if not polys:
        return
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=(0.15, 0.15, 0.15, 0.12), linewidths=0.15))
    lo, hi = P.min(0), P.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-3 * diag), zoom=1.25)
    ax.set_proj_type("ortho")
    ax.view_init(*view)
    for a in "xyz":
        getattr(ax, f"set_{a}label")(a + " (m)", fontsize=8)
    ax.tick_params(labelsize=6)
    occ = np.vstack(occluders) if occluders else np.zeros((0, 3, 3))
    eps = 1e-3 * diag
    done: set[str] = set()
    for f, (cent, area2) in label_at.items():       # one label per feature, at its largest visible triangle
        text = labels[f]
        if text in done:
            continue
        order = np.argsort(-area2)[:30]
        seen = order[_unoccluded(cent[order], occ, cam, eps)] if len(occ) else order
        if not len(seen):
            continue
        done.add(text)
        ax.text(*cent[seen[0]], text, fontsize=6.5, ha="center", va="center", zorder=20,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    for pts in edges or []:                           # re-entrant edges, where they are seen
        pts = np.asarray(pts)
        mid = 0.5 * (pts[1:] + pts[:-1])
        ok = np.ones(len(mid), bool) if cut is None else ((mid - cut) @ cam) <= eps
        if len(occ):
            ok &= _unoccluded(mid, occ, cam, eps)
        for k in np.nonzero(ok)[0]:
            ax.plot(*np.stack([pts[k], pts[k + 1]]).T, color=(0.85, 0.0, 0.75), lw=1.4, zorder=15)
    for name, a, b in lines or []:                    # stress classification lines: shown through the material
        a, b = np.asarray(a), np.asarray(b)
        ax.plot(*np.stack([a, b]).T, color="black", lw=1.8, marker="o", ms=2.5, zorder=25)
        ax.text(*(0.5 * (a + b)), name, fontsize=6.5, color="black", ha="left", va="bottom", zorder=26,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))


def _figure(path: Path, P, tris: dict, diag: float, title: str, colour: dict, legend: list, cut_axis: int,
            centre: np.ndarray, labels: dict | None = None, lines: list | None = None, edges: list | None = None) -> None:
    """A picture with two panels: the part from the outside, and its far half behind a plane through the
    middle of the bounding box."""
    from matplotlib.patches import Patch

    from ace.run_report import _new_fig
    fig = _new_fig(11.0, 5.6)
    section = {0: (0.0, 180.0), 1: (0.0, -90.0), 2: (-90.0, 90.0)}[cut_axis]     # the camera looks along +axis
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _draw(ax, P, tris, colour, (28.0, -60.0), None, diag, labels, lines, edges)
    ax.set_title("from the outside", fontsize=9)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    _draw(ax, P, tris, colour, section, centre, diag, labels, lines, edges)
    ax.set_title(f"section: the half behind the plane {'xyz'[cut_axis]} = {_len(centre[cut_axis])}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    if legend:
        fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=name) for name, c in legend],
                   loc="outside lower center", ncol=min(len(legend), 6), fontsize=8, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def _pictures(project, comp, part, geo: dict) -> list[dict]:
    """The overview and one picture per interface, in <out>/geometry/. Returns the entries of geo["pictures"]."""
    from single_agent.render import _tessellate
    with _quiet():
        P, tris, diag = _tessellate(part)
    lo, hi = np.asarray(geo["cad"]["bbox"][0]), np.asarray(geo["cad"]["bbox"][1])
    centre = 0.5 * (lo + hi)
    cut_axis = int(np.argmin(hi - lo))                 # the cut that shows the largest section
    out_dir = project.out / "geometry"
    pics = []
    lines = [(sid, s["start"], s["end"]) for sid, s in geo["scls"].items()]
    edges = [e["points"] for e in geo["edges"]["reentrant"]]
    # overview: one colour per interface (the faces of a "*" selection stay grey), SCLs and re-entrant edges
    colour, legend = {}, []
    star = {(it.id, role) for it in comp.interfaces for role, feats in it.features.items() if "*" in feats}
    k = 0
    for it in comp.interfaces:
        tags = sorted({t for role, ts in geo["interfaces"][it.id].items() if (it.id, role) not in star for t in ts})
        if not tags:
            continue
        c = PALETTE[k % len(PALETTE)]
        k += 1
        legend.append((it.id, c))
        for t in tags:
            colour.setdefault(t, c)
    if edges:
        legend.append(("re-entrant edge", (0.85, 0.0, 0.75)))
    if lines:
        legend.append(("stress classification line", (0.0, 0.0, 0.0)))
    path = out_dir / "overview.png"
    _figure(path, P, tris, diag, f"{comp.id}: {comp.name}", colour, legend, cut_axis, centre, None, lines, edges)
    pics.append({"path": str(path.relative_to(project.out)),
                 "caption": f"The part {comp.id}. Each colour is one interface. The magenta lines are the re-entrant "
                            f"edges. The black lines are the stress classification lines."})
    # one picture per interface: its faces by role, with the name of the feature
    for it in comp.interfaces:
        colour, labels, legend = {}, {}, []
        for role in (*CONTRACT_ROLES, "body"):
            tags = geo["interfaces"][it.id].get(role) or []
            if not tags:
                continue
            legend.append((role, ROLE_COLOURS[role]))
            for t in tags:
                colour.setdefault(t, ROLE_COLOURS[role])
        for role, feats in it.features.items():
            for fid in feats:
                for t in geo["features"].get(fid, {}).get("faces", []) if fid != "*" else []:
                    labels.setdefault(t, fid)
        if not colour:
            continue
        path = out_dir / f"{it.id}.png"
        _figure(path, P, tris, diag, f"{it.id}: {it.name}", colour, legend, cut_axis, centre, labels)
        pics.append({"path": str(path.relative_to(project.out)),
                     "caption": f"The faces of the interface {it.id}. Each colour is one role."})
    return pics
