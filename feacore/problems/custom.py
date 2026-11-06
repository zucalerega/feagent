"""Problems made from your own CAD: STEP files uploaded on the dashboard's "Your CAD" page.

An upload is a small gold root (feacore/problems/gold.py) holding one directory per BC set, so what
works on a gold set works here: the registry and the Benchmarks page, convergence studies, the
solver, the ACE agents (``ace.fea.get_problem``; ``ace.config.TASKS`` gets each set's entry from
``ace_tasks``) and ``scripts/gold.py --root <upload dir> verify|render`` after a reference study::

  <root>/<upload>/upload.json            file name, size, sha256, units and scale, the part's faces
  <root>/<upload>/source.step            the file as uploaded
  <root>/<upload>/geometry.step          the part in metres; its OCC face tags are the f<tag> names
  <root>/<upload>/preview.json           coarse surface triangulation for the dashboard's 3D view
  <root>/<upload>/<set>/public/problem.json, geometry.step     statement for agents
  <root>/<upload>/<set>/private/reference.json, study.json     references, BC set, reference case
  <root>/<upload>/_meshes/<upload>/L<k>.msh                    reference-study meshes (nested tet10)

Roots (``cad_roots``): each directory in ``$FEACORE_CAD_DIR`` (os.pathsep-separated), then the run
store's ``cad/`` (``$FEACORE_RUNS_DIR/cad``, else ``<repo>/runs/cad``). The dashboard writes into its
run store, and its child processes (studies, tasks) get the same FEACORE_RUNS_DIR, so every process
finds the same problems. Problem ids: ``<upload>_bc<k>`` for generated BC sets
(``feacore.goldgen.bcgen``, as for gold), ``<upload>_m<k>`` for sets defined by hand; tag ``custom``,
family = the upload id.

A new set has no reference value: runs on it are judged reference-free (the ZZ error estimate and
the agreement between agents, ``python -m ace.infer``) until a reference study certifies quantities
with the gold rules (``feacore.goldgen.run.certify``): ``python -m feacore.problems.custom reference
UPLOAD`` (nested meshes on the CPU, for small parts), or a finished convergence study adopted with
``adopt_study``. The training loops (ace.run_loop, ace.single_loop) need such a reference.

    python -m feacore.problems.custom list
    python -m feacore.problems.custom add PART.step [--units si|as_is|normalize] [--generate N]
    python -m feacore.problems.custom reference UPLOAD [--sets ID,...] [--levels 3] [--max-cells 30000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

from feacore.problems.gold import GoldProblem

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "feacore.cad_upload/1"
UNITS = {
    "si": "metres, converted from the length unit the STEP file declares (mm, inch, ...)",
    "as_is": "the file's coordinates taken as metres (e.g. a gold geometry.step)",
    "normalize": "rescaled to a 100 mm bounding-box diagonal, like the gold parts",
}
UNIT_WORDS = {"si": "in metres", "as_is": "its coordinates taken as metres",
              "normalize": "rescaled to a 100 mm bounding-box diagonal"}
DEFAULT_MAX_BYTES = 30_000_000
MAX_FACES = 2000
DEFAULT_NODE_CAP = 40_000      # as ace.infer.DEFAULT_NODE_CAP: about the gold L1 meshes
STEEL = {"E": 210e9, "nu": 0.3}
H0_FRAC = 0.237                # the gold generator's base size / bbox diagonal (scripts/goldgen_h100.sh)
H0_FRAC_MAX = 1.0              # base sizes above the diagonal are kept only if the nested splits work there
INSPECT_TIMEOUT = 240.0        # s for reading, checking and previewing one upload (FEACORE_CAD_INSPECT_TIMEOUT)
UPLOAD_RE = re.compile(r"^cad_[a-z0-9_]{1,60}$")
SET_RE = re.compile(r"^(cad_[a-z0-9_]{1,60})_(bc|m)(\d{1,4})$")
SCORE_ORDER = ("strain_energy", "u_mean_", "u_max", "vm_mean", "R")   # as feacore.problems.gold
REPORT = ("Report: strain_energy (J); u_max = maximum displacement magnitude (m); vm_mean = volume-average "
          "von Mises stress (Pa); u_mean_<face> = mean displacement magnitude over each loaded face (m); "
          "with several supports, the support reactions R<x|y|z>_<face> (N).")


class CadError(ValueError):
    """An upload or a BC set that cannot be used; the message says why, in plain words."""


# --------------------------------------------------------------------------- where problems live


def max_bytes() -> int:
    return int(os.environ.get("FEACORE_CAD_MAX_BYTES") or DEFAULT_MAX_BYTES)


def store_root() -> Path:
    """The run store's cad/ directory: where the dashboard and the CLI add uploads."""
    from feacore.runstore import default_root

    return default_root() / "cad"


def cad_roots() -> list[Path]:
    """Every directory scanned for uploads: $FEACORE_CAD_DIR entries, then the run store's cad/."""
    roots = [Path(p).expanduser() for p in os.environ.get("FEACORE_CAD_DIR", "").split(os.pathsep) if p.strip()]
    roots.append(store_root())
    out, seen = [], set()
    for r in roots:
        key = str(r.resolve())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def add_root(path: str | Path) -> None:
    """Scan ``path`` too, in this process and in every process it starts (through FEACORE_CAD_DIR)."""
    path = Path(path)
    if str(path.resolve()) in {str(r.resolve()) for r in cad_roots()}:
        return
    cur = os.environ.get("FEACORE_CAD_DIR", "")
    os.environ["FEACORE_CAD_DIR"] = os.pathsep.join([str(path), *([cur] if cur else [])])


# --------------------------------------------------------------------------- the problem


def _kind(name: str) -> str:
    if name.startswith("u_mean_"):
        return "u_mean_"
    return "R" if name[:1] == "R" and name[2:3] == "_" else name


class CustomProblem(GoldProblem):
    """One BC set of an uploaded part. Its files are re-read when they change on disk (a reference
    study in another process adds references), so a long-running server stays current."""

    def __init__(self, path: str | Path):
        self._mtimes: dict[str, int | None] = {}
        super().__init__(path)
        if self.dir.name != self.id:
            raise ValueError(f"directory {self.dir.name} holds problem {self.id}")

    @property
    def upload_dir(self) -> Path:
        return self.dir.parent

    def _private(self) -> dict:
        p = self.dir / "private" / "reference.json"
        m = p.stat().st_mtime_ns
        if self._private_data is None or self._mtimes.get("reference") != m:
            self._private_data = json.loads(p.read_text())
            self._mtimes["reference"] = m
        return self._private_data

    def study(self) -> dict:
        p = self.dir / "private" / "study.json"
        m = p.stat().st_mtime_ns if p.exists() else None
        if self._study_data is None or self._mtimes.get("study") != m:
            self._study_data = json.loads(p.read_text()) if p.exists() else {"runs": [], "qois": {}}
            self._study_data.setdefault("runs", [])
            self._study_data.setdefault("qois", {})
            self._mtimes["study"] = m
        return self._study_data

    @property
    def material(self) -> dict:
        m = self._private().get("material") or STEEL
        return {"E": float(m.get("E", STEEL["E"])), "nu": float(m.get("nu", STEEL["nu"]))}

    @property
    def node_cap(self) -> int:
        return int((self._private().get("ace") or {}).get("node_cap") or DEFAULT_NODE_CAP)

    def part_obj(self):
        """The part as a feacore.goldgen.parts.Part (its geometry.step, scale 1)."""
        from feacore.goldgen.parts import Part

        return Part.from_json(json.dumps({**self.part, "scale": 1.0}))

    def scored_qoi(self) -> str | None:
        """A certified quantity when there is one (GoldProblem's rule), else the first statement
        quantity in the same order of preference (strain energy first): then there is no reference
        value and runs are judged reference-free."""
        hit = super().scored_qoi()
        if hit:
            return hit
        names = [q.name for q in self.qois()]
        for k in SCORE_ORDER:
            match = [n for n in names if _kind(n) == k]
            if match:
                return match[0]
        return names[0] if names else None

    def ace_task(self) -> dict:
        """This set's ace.config.TASKS entry: the scored quantity and the node cap."""
        return {"qoi": self.scored_qoi(), "node_cap": self.node_cap}

    def study_sizes(self, order: int = 2) -> list[float]:
        """The reference study's sizes when it ran, else three uniform sizes for a quick study."""
        hs = sorted({float(r["h"]) for r in self.study()["runs"] if r.get("h")}, reverse=True)
        return hs if len(hs) >= 3 else default_sizes(self.part)

    def mesh(self, path: str | Path, h: float, geom_order: int = 2, refine_steps: int = 0) -> Path:
        """Reference mesh at size h with the gold generator's pipeline (feacore.goldgen.mesh: curvature
        sizing, repaired linear tets, curved tet10 checked for the solver), split ``refine_steps``
        times for a nested study. A linear mesh (geom_order 1) is the plain one of Problem.mesh."""
        if geom_order < 2:
            return super().mesh(path, h, geom_order=geom_order, refine_steps=refine_steps)
        from feacore.goldgen.mesh import STRATEGIES, MeshFailed, mesh_level_safe

        part = self.part_obj()
        tried = []
        for scheme in (*STRATEGIES, "straight"):
            try:
                mesh_level_safe(self.step_path, part, int(refine_steps), Path(path), threads=1,
                                h0_frac=float(h) / part.diag, scheme=scheme, nested=True)
                return Path(path)
            except MeshFailed as e:
                tried.append(str(e)[:200])
        raise RuntimeError(f"{self.id}: no usable mesh at h = {h:.4g}: " + "; ".join(tried))


def default_sizes(part: dict, target_cells: int = 20_000) -> list[float]:
    """Three uniform sizes (ratio 1.5) whose finest mesh has roughly ``target_cells`` tets (gmsh
    fills a volume V with about 7 V / h^3 of them, more near curved faces), kept within
    diag / 60 .. diag / 6."""
    b = part["bbox"]
    diag = math.dist(b[:3], b[3:])
    h = (7.0 * float(part["volume"]) / target_cells) ** (1 / 3)
    h = min(max(h, diag / 60), diag / 6)
    return [h * 2.25, h * 1.5, h]


# --------------------------------------------------------------------------- loading and registering


def _sets_in(updir: Path) -> tuple[list[CustomProblem], dict[str, str]]:
    found, errors = [], {}
    for pj in sorted(updir.glob("*/public/problem.json")):
        d = pj.parent.parent
        if d.name.startswith(".") or not SET_RE.match(d.name):
            continue
        try:
            found.append(CustomProblem(d))
        except Exception as e:  # noqa: BLE001 - one unreadable set must not hide the others
            errors[str(d)] = f"{type(e).__name__}: {e}"
    return found, errors


def upload_dirs(roots: list[Path] | None = None) -> list[Path]:
    out = []
    for root in roots if roots is not None else cad_roots():
        if root.is_dir():
            out += [d for d in sorted(root.iterdir())
                    if UPLOAD_RE.match(d.name) and (d / "upload.json").is_file()]
    return out


def load_all(roots: list[Path] | None = None) -> tuple[list[CustomProblem], dict[str, str]]:
    """Every BC set under the roots (default ``cad_roots()``), and {directory: error} for the ones that
    could not be read. An id found twice keeps its first directory."""
    found, errors, seen = [], {}, set()
    for updir in upload_dirs(roots):
        sets, errs = _sets_in(updir)
        errors.update(errs)
        for p in sets:
            if p.id in seen:
                errors[str(p.dir)] = f"duplicate problem id {p.id} (an earlier root has it)"
                continue
            seen.add(p.id)
            found.append(p)
    return found, errors


def _registry():
    from feacore.problems import registry

    return registry


def _register(problems: list[CustomProblem]) -> list[str]:
    reg = _registry()
    added = []
    for p in problems:
        if p.id not in reg._REGISTRY:   # noqa: SLF001 - this module owns its entries
            reg.register(p)
            added.append(p.id)
    return added


def _unregister(ids) -> None:
    reg = _registry()
    for pid in ids:
        if isinstance(reg._REGISTRY.get(pid), CustomProblem):   # noqa: SLF001
            reg._REGISTRY.pop(pid)   # noqa: SLF001


def refresh() -> dict:
    """Bring the registry up to date with the roots: register new sets, drop deleted ones."""
    reg = _registry()
    found, errors = load_all()
    ids = {p.id for p in found}
    gone = [pid for pid, p in list(reg._REGISTRY.items())   # noqa: SLF001
            if isinstance(p, CustomProblem) and (pid not in ids or not p.dir.is_dir())]
    _unregister(gone)
    return {"added": _register(found), "removed": gone, "errors": errors}


def ace_tasks() -> dict[str, dict]:
    """{problem id: ace.config.TASKS entry} for every custom set (read by ace/config.py)."""
    tasks = {}
    for p in load_all()[0]:
        try:
            tasks[p.id] = p.ace_task()
        except Exception:  # noqa: BLE001 - an unreadable set has no task
            continue
    return tasks


# --------------------------------------------------------------------------- uploads


def check_step(data: bytes) -> None:
    """Refuse what is not a complete STEP file within the size limit (CadError says why)."""
    if not data:
        raise CadError("the file is empty")
    limit = max_bytes()
    if len(data) > limit:
        raise CadError(f"the file is {len(data) / 1e6:.1f} MB; the limit is {limit / 1e6:.0f} MB "
                       f"(FEACORE_CAD_MAX_BYTES)")
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    if not head.startswith(b"ISO-10303-21;"):
        raise CadError("this is not a STEP file: a STEP file starts with 'ISO-10303-21;'. Export the part "
                       "from your CAD tool as STEP (AP203, AP214 or AP242; .step or .stp)")
    if b"HEADER;" not in head:
        raise CadError("this is not a valid STEP file: it has no HEADER section")
    if b"DATA;" not in data:
        raise CadError("this is not a valid STEP file: it has no DATA section")
    if b"END-ISO-10303-21;" not in data[-4096:]:
        raise CadError("the STEP file is incomplete: it does not end with 'END-ISO-10303-21;' "
                       "(was the upload cut off?)")


def clean_name(filename: str | None) -> str:
    name = Path(str(filename or "part.step").replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip() or "part.step"
    return name[:120]


def slug(name: str) -> str:
    stem = re.sub(r"\.(step|stp)$", "", name, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return s[:24].strip("_") or "part"


def upload_id(name: str, data: bytes, units: str) -> str:
    """cad_<file name>_<6 hex of the content and units>: the same file uploaded again is the same upload."""
    digest = hashlib.sha256(data + b"\0" + units.encode()).hexdigest()[:6]
    return f"cad_{slug(name)}_{digest}"


def find_upload(uid: str) -> Path:
    if not isinstance(uid, str) or not UPLOAD_RE.match(uid):
        raise KeyError(f"invalid upload id {uid!r}")
    for root in cad_roots():
        d = root / uid
        if (d / "upload.json").is_file():
            return d
    raise KeyError(f"no upload {uid!r}")


def find_set(pid: str) -> Path:
    m = SET_RE.match(pid) if isinstance(pid, str) else None
    if not m:
        raise KeyError(f"invalid problem id {pid!r}")
    d = find_upload(m.group(1)) / pid
    if not (d / "public" / "problem.json").is_file():
        raise KeyError(f"no problem {pid!r}")
    return d


def read_upload(updir: Path) -> dict:
    return json.loads((Path(updir) / "upload.json").read_text())


def create_upload(filename: str | None, data: bytes, units: str = "si", root: str | Path | None = None,
                  timeout: float | None = None) -> dict:
    """Check a STEP file and store it as an upload (no BC set yet). The CAD kernel runs in its own
    process (``inspect`` below): a file that crashes or hangs it is refused, not fatal. Uploading
    the same file with the same units again returns the existing upload."""
    if units not in UNITS:
        raise CadError(f"units must be one of {sorted(UNITS)}")
    name = clean_name(filename)
    check_step(data)
    root = Path(root) if root is not None else store_root()
    uid = upload_id(name, data, units)
    final = root / uid
    if (final / "upload.json").is_file():
        return {**read_upload(final), "existing": True}
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{uid}.{secrets.token_hex(3)}.tmp"
    tmp.mkdir()
    try:
        (tmp / "source.step").write_bytes(data)
        info = _inspect_in_subprocess(tmp, units, uid, name, timeout)
        part = info["part"]
        b = part["bbox"]
        rec = {
            "schema": SCHEMA, "id": uid, "name": name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "units": units, "scale": info["scale"],
            "created_at": _now(), "part": part, "n_faces": len(part["faces"]),
            "diag_mm": 1000 * math.dist(b[:3], b[3:]), "size_mm": [1000 * (b[i + 3] - b[i]) for i in range(3)],
            "volume_cm3": 1e6 * float(part["volume"]), "warnings": info.get("warnings", []),
        }
        _write_json(tmp / "upload.json", rec)
        try:
            tmp.rename(final)
        except OSError:
            if (final / "upload.json").is_file():          # the same upload, finished by another request
                shutil.rmtree(tmp, ignore_errors=True)
                return {**read_upload(final), "existing": True}
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rec


def _inspect_in_subprocess(d: Path, units: str, uid: str, name: str, timeout: float | None) -> dict:
    timeout = float(timeout or os.environ.get("FEACORE_CAD_INSPECT_TIMEOUT") or INSPECT_TIMEOUT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])])}
    cmd = [sys.executable, "-m", "feacore.problems.custom", "inspect", str(d), f"--units={units}", f"--id={uid}",
           f"--name={name}"]
    try:
        p = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CadError(f"the CAD kernel needed more than {timeout:.0f} s for this file; simplify the part "
                       f"or raise FEACORE_CAD_INSPECT_TIMEOUT") from None
    res = d / "inspect.json"
    if not res.is_file():
        tail = " | ".join((p.stderr or p.stdout).strip().splitlines()[-2:])
        raise CadError(f"the CAD kernel crashed on this file (exit {p.returncode}){': ' + tail if tail else ''}")
    info = json.loads(res.read_text())
    res.unlink()
    if not info.get("ok"):
        raise CadError(info.get("error") or "the file could not be read")
    return info


def _occ_import(step: Path, scale: float = 1.0, unit: str = "MM"):
    """Import into the current gmsh model. The length unit is set every time: OpenCASCADE keeps it
    process-wide, and every other reader of geometry.step uses the default (mm-numbered) import."""
    import gmsh

    gmsh.option.setString("Geometry.OCCTargetUnit", unit)
    gmsh.option.setNumber("Geometry.OCCScaling", scale)
    ents = gmsh.model.occ.importShapes(str(step), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    return ents


def _one_solid(ents) -> int:
    vols = [t for d, t in ents if d == 3]
    if not vols:
        kinds = sorted({["points", "curves", "surfaces"][d] for d, _ in ents if d < 3})
        raise CadError("the file has no solid" + (f" (only {', '.join(kinds)})" if kinds else "")
                       + ": export the part as one closed solid body")
    if len(vols) > 1:
        raise CadError(f"the file has {len(vols)} solids: upload one solid per file (merge the bodies, "
                       f"or export one of them)")
    return vols[0]


def inspect(src: Path, out: Path, units: str, uid: str, name: str) -> dict:
    """In THIS process (gmsh): one solid, its scale for ``units``, geometry.step in metres, its faces
    and the 3D preview. Raises CadError for a file that cannot be used."""
    import gmsh
    import numpy as np

    from feacore.geometry.gmsh_tools import gmsh_session

    with gmsh_session(uid):
        try:
            ents = _occ_import(src)
        except Exception as e:  # noqa: BLE001 - OCC raises plain Exceptions
            raise CadError(f"the CAD kernel could not read the file: {e}") from None
        b = np.asarray(gmsh.model.getBoundingBox(3, _one_solid(ents)))
        diag0 = float(np.linalg.norm(b[3:] - b[:3]))
    if not diag0 > 0:
        raise CadError("the solid has an empty bounding box")
    if units == "si":
        with gmsh_session(uid):
            b = np.asarray(gmsh.model.getBoundingBox(3, _one_solid(_occ_import(src, unit="M"))))
        scale = float(np.linalg.norm(b[3:] - b[:3])) / diag0
    else:
        scale = 1.0 if units == "as_is" else 0.1 / diag0
    geo = out / "geometry.step"
    with gmsh_session(uid):
        _occ_import(src, scale=scale)
        gmsh.write(str(geo))
    part = describe(geo, uid, name)
    from feacore.ui.worker import geometry_preview

    _write_json(out / "preview.json", geometry_preview(_StepGeometry(uid, geo)))
    notes = []
    if part.diag < 1e-3:
        notes.append(f"the part is {part.diag * 1000:.3g} mm across: if its coordinates are metres already, "
                     f"upload it again with units 'as_is'")
    if part.diag > 20:
        notes.append(f"the part is {part.diag:.3g} m across: check the units")
    return {"scale": scale, "part": asdict(part), "warnings": notes}


class _StepGeometry:
    """What feacore.ui.worker.geometry_preview needs: a 3D part whose faces are groups f<tag>."""

    dim = 3

    def __init__(self, pid: str, step: Path):
        self.id = pid
        self.step_path = Path(step)

    def build_geometry(self) -> None:
        import gmsh

        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        GoldProblem.build_geometry(self)


def describe(step: Path, part_id: str, source: str):
    """The part in ``step`` (scale 1) as a feacore.goldgen.parts.Part: volume, bounding box and every
    face (tag, kind, area, centroid, outward normal of planes, axis and radius of cylinders). A coarse
    tet mesh must succeed; its surface triangles give each plane a point on the face to orient its
    normal from (the face's centroid can lie off the face, inside the solid, e.g. under a rib)."""
    import gmsh
    import numpy as np

    from feacore.geometry.gmsh_tools import gmsh_session
    from feacore.goldgen.parts import Face, Part, _cylinder_axis, _outward_normal

    with gmsh_session(part_id):
        vol = _one_solid(_occ_import(step))
        mass = gmsh.model.occ.getMass(3, vol)
        if not mass > 0:
            raise CadError("the solid has no volume: is it an open shell?")
        bbox = list(gmsh.model.getBoundingBox(3, vol))
        diag = float(np.linalg.norm(np.subtract(bbox[3:], bbox[:3])))
        tags = [t for _, t in gmsh.model.getBoundary([(3, vol)], oriented=False)]
        if len(tags) > MAX_FACES:
            raise CadError(f"the part has {len(tags)} faces; at most {MAX_FACES} are supported")
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 8)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:  # noqa: BLE001
            raise CadError(f"gmsh could not mesh the solid (is it closed and valid?): {e}") from None
        if len(gmsh.model.mesh.getElementsByType(4)[0]) == 0:
            raise CadError("gmsh produced no tetrahedra for this solid (is it closed?)")
        faces = []
        for t in tags:
            c = np.asarray(gmsh.model.occ.getCenterOfMass(2, t))
            f = Face(tag=t, kind="other", area=gmsh.model.occ.getMass(2, t), centroid=c.tolist())
            try:
                typ = gmsh.model.getType(2, t)
                if typ == "Plane" or (typ != "Cylinder" and _is_flat(t, diag)):
                    f.normal = _outward_normal(t, _point_on_face(t, c), vol, diag).round(9).tolist()
                    f.kind = "plane"
                elif typ == "Cylinder":
                    axis, radius = _cylinder_axis(t)
                    f.axis, f.radius, f.kind = axis, (radius if math.isfinite(radius) else None), "cylinder"
            except Exception:  # noqa: BLE001 - a face the kernel cannot classify stays "other"
                f.normal = f.axis = f.radius = None
            faces.append(f)
    return Part(id=part_id, source=source, scale=1.0, volume=mass, bbox=bbox, faces=faces)


def _point_on_face(tag: int, fallback):
    """The centroid of the face's largest mesh triangle: a point on the (trimmed) face."""
    import gmsh
    import numpy as np

    types, _, nodes = gmsh.model.mesh.getElements(2, tag)
    ntags, x, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
    if not len(types) or not len(ntags):
        return np.asarray(fallback)
    at = dict(zip(np.asarray(ntags, dtype=np.int64).tolist(), np.asarray(x, float).reshape(-1, 3), strict=True))
    tri = np.asarray(nodes[0], dtype=np.int64).reshape(-1, 3)
    p = np.array([[at[int(n)] for n in row] for row in tri])
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return p[int(np.argmax(area))].mean(axis=0)


def _is_flat(tag: int, diag: float, tol: float = 1e-7) -> bool:
    """Is a face planar whatever its CAD type? Some CAD tools export every face as a B-spline: a face
    whose normals agree at 3 x 3 points of its parameter range and whose points lie in one plane is
    a plane (then it gets a normal, so forces and rollers can be generated on it)."""
    import gmsh
    import numpy as np

    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    uv = np.array([[lo[0] + (hi[0] - lo[0]) * a, lo[1] + (hi[1] - lo[1]) * b]
                   for a in (0.1, 0.5, 0.9) for b in (0.1, 0.5, 0.9)]).ravel()
    n = np.asarray(gmsh.model.getNormal(tag, uv), float).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1)[:, None]
    x = np.asarray(gmsh.model.getValue(2, tag, uv), float).reshape(-1, 3)
    return bool(np.abs(n @ n[0]).min() > 1 - tol and np.abs((x - x[0]) @ n[0]).max() < tol * diag)


def delete_upload(uid: str) -> list[str]:
    """Remove an upload and every BC set made from it; returns the removed problem ids."""
    d = find_upload(uid)
    ids = [p.id for p in _sets_in(d)[0]]
    shutil.rmtree(d)
    _unregister(ids)
    return ids


def delete_set(pid: str) -> None:
    d = find_set(pid)
    shutil.rmtree(d)
    _unregister([pid])


# --------------------------------------------------------------------------- BC sets


def _part(up: dict):
    from feacore.goldgen.parts import Part

    return Part.from_json(json.dumps({**up["part"], "scale": 1.0}))


def reference_case(mesh: str | Path, bcset, E: float, nu: float):
    """The reference setup of a set: static linear elasticity, P2, one material, the set's BCs and QoIs."""
    from feacore.case import Analysis, Case, Material, Region
    from feacore.goldgen.run import qois_for

    return Case(mesh=str(mesh), analysis=Analysis(physics="elasticity", formulation="3d", order=2),
                materials={"material": Material(E=E, nu=nu)}, bcs=list(bcset.bcs),
                regions=[Region("solid", "material")], qois=[q for q, _ in qois_for(bcset)],
                meta={"problem": bcset.id, "setup": "custom_reference"})


def statement(up: dict, part, bcset, E: float, nu: float, title: str) -> dict:
    """The agent-facing statement, worded as the gold sets' (feacore.goldgen.run.statement)."""
    from feacore.goldgen.run import qois_for

    lines = [
        f"Static linear-elastic analysis of the part in geometry.step (your CAD {up['name']}, "
        f"{UNIT_WORDS.get(up.get('units'), 'in metres')}; bounding-box diagonal {part.diag * 1000:.1f} mm).",
        f"Material: isotropic linear elastic, E = {E / 1e9:g} GPa, Poisson's ratio {nu:g}. No gravity.",
        "Faces are named f<tag> by their OCC face tag in this STEP file; positions below are in mm.",
        *bcset.words,
        "All other faces are free.",
        REPORT,
    ]
    return {"id": bcset.id, "title": title, "family": up["id"], "tags": ["custom", "3d", "cad"], "dim": 3,
            "physics": "elasticity", "formulation": "3d", "units": "SI (m, N, Pa)",
            "description": "\n".join(lines), "qois": [asdict(q) for q, _ in qois_for(bcset)],
            "geometry_file": "geometry.step"}


def _write_set(updir: Path, up: dict, bcset, E: float, nu: float, node_cap: int, kind: str, title: str) -> str:
    """One set directory in the gold layout, written to a temporary name and renamed into place."""
    d = updir / bcset.id
    if d.exists():
        raise CadError(f"{bcset.id} exists already")
    tmp = updir / f".{bcset.id}.{secrets.token_hex(3)}.tmp"
    (tmp / "public").mkdir(parents=True)
    (tmp / "private").mkdir()
    try:
        _write_json(tmp / "public" / "problem.json", statement(up, _part(up), bcset, E, nu, title))
        shutil.copyfile(updir / "geometry.step", tmp / "public" / "geometry.step")
        _write_json(tmp / "private" / "reference.json", {
            "id": bcset.id, "references": [], "bcset": bcset.to_dict(),
            "reference_case": reference_case("MESH.msh", bcset, E, nu).to_dict(), "part": up["part"],
            "material": {"E": E, "nu": nu}, "ace": {"node_cap": int(node_cap)}, "kind": kind,
            "created_at": _now()})
        _write_json(tmp / "private" / "study.json", {"runs": [], "qois": {}})
        tmp.rename(d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return bcset.id


def _number(v, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise CadError(f"{what}: {v!r} is not a number") from None
    if not math.isfinite(x) or abs(x) > 1e15:
        raise CadError(f"{what}: {v!r} is out of range")
    return x


def _material(spec: dict) -> tuple[float, float]:
    m = spec.get("material") or {}
    E = _number(m.get("E", STEEL["E"]), "Young's modulus E (Pa)")
    nu = _number(m.get("nu", STEEL["nu"]), "Poisson's ratio")
    if E <= 0:
        raise CadError("Young's modulus E must be positive (in Pa, e.g. 210e9 for steel)")
    if not 0 <= nu < 0.5:
        raise CadError("Poisson's ratio must be in [0, 0.5)")
    return E, nu


def _node_cap(spec: dict) -> int:
    cap = int(_number(spec.get("node_cap") or DEFAULT_NODE_CAP, "node cap"))
    if not 1000 <= cap <= 5_000_000:
        raise CadError("the node cap must be between 1,000 and 5,000,000")
    return cap


def generate_sets(uid: str, count: int, spec: dict | None = None) -> list[str]:
    """``count`` more BC sets from the gold generator's rules (feacore.goldgen.bcgen): clamps and
    rollers on large faces, forces and pressures away from them, seeded by the upload id."""
    from feacore.goldgen.bcgen import generate

    spec = spec or {}
    if not 1 <= int(count) <= 20:
        raise CadError("generate between 1 and 20 BC sets at a time")
    E, nu = _material(spec)
    cap = _node_cap(spec)
    updir = find_upload(uid)
    up = read_upload(updir)
    have = {d.name for d in updir.iterdir() if d.is_dir() and SET_RE.match(d.name)}
    n_auto = sum(1 for n in have if SET_RE.match(n).group(2) == "bc")
    sets = generate(_part(up), n_auto + int(count))
    new = [s for s in sets if s.id not in have][: int(count)]
    if not new:
        raise CadError(f"no more distinct BC sets for this part ({len(sets)} in all): supports and loads need "
                       f"faces with at least 2 % of the surface area, loads away from the supports")
    return [_write_set(updir, up, s, E, nu, cap, "generated",
                       f"{up['name']}: generated BC set {s.id.rsplit('_bc', 1)[1]}") for s in new]


def define_set(uid: str, spec: dict) -> str:
    """A BC set defined by hand::

        {"fixed": ["f12"], "rollers": [{"group": "f7", "component": 0}],
         "loads": [{"group": "f20", "type": "pressure", "value": 2e6},
                   {"group": "f18", "type": "force", "value": [0, 0, -1000]}],
         "material": {"E": 210e9, "nu": 0.3}, "node_cap": 40000, "label": "bolt load"}

    Values in SI (Pa, N). Pressure pushes into the part when positive. At least one clamped face and
    one load; one support or load per face."""
    from feacore.case import BC
    from feacore.goldgen.bcgen import BCSet
    from feacore.goldgen.bcgen import _describe as describe_face

    updir = find_upload(uid)
    up = read_upload(updir)
    part = _part(up)
    faces = {f"f{f.tag}": f for f in part.faces}
    used: set[str] = set()

    def face(g, what):
        if not isinstance(g, str) or g not in faces:
            raise CadError(f"{what}: {g!r} is not a face of this part (faces are named f<tag>)")
        if g in used:
            raise CadError(f"face {g} has two supports or loads: one per face")
        used.add(g)
        return faces[g]

    bcs, words = [], []
    for g in spec.get("fixed") or []:
        bcs.append(BC("fixed", group=g))
        words.append(f"Clamped (all displacements zero): {describe_face(face(g, 'clamp'))}.")
    if not bcs:
        raise CadError("clamp at least one face: without a clamped face the part can move as a rigid body")
    for r in spec.get("rollers") or []:
        k = r.get("component") if isinstance(r, dict) else None
        if k not in (0, 1, 2):
            raise CadError("a roller needs component 0, 1 or 2 (the x, y or z displacement held at zero)")
        f = face(r.get("group"), "roller")
        bcs.append(BC("displacement", group=r["group"], components=[k], value=0.0))
        words.append(f"Roller (zero {'xyz'[k]}-displacement only): {describe_face(f)}.")
    loads = spec.get("loads") or []
    if not loads:
        raise CadError("add at least one load: a force or a pressure on a face")
    for ld in loads:
        if not isinstance(ld, dict):
            raise CadError("each load is {group, type, value}")
        kind = ld.get("type")
        f = face(ld.get("group"), kind or "load")
        if kind == "pressure":
            p = _number(ld.get("value"), "pressure (Pa)")
            if p == 0:
                raise CadError(f"the pressure on {ld['group']} is zero")
            bcs.append(BC("pressure", group=ld["group"], value=p))
            words.append(f"Uniform pressure {p / 1e6:g} MPa pushing into the part on {describe_face(f)}." if p > 0
                         else f"Uniform suction {-p / 1e6:g} MPa pulling outward on {describe_face(f)}.")
        elif kind == "force":
            v = ld.get("value")
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise CadError("a force is a vector [Fx, Fy, Fz] in N")
            vec = [_number(x, "force (N)") + 0.0 for x in v]
            if not any(vec):
                raise CadError(f"the force on {ld['group']} is zero")
            bcs.append(BC("force", group=ld["group"], value=vec))
            words.append(f"Total force ({', '.join(f'{x:.1f}' for x in vec)}) N spread uniformly over "
                         f"{describe_face(f)}.")
        else:
            raise CadError(f"load type must be 'force' or 'pressure', not {kind!r}")
    E, nu = _material(spec)
    cap = _node_cap(spec)
    have = {int(SET_RE.match(d.name).group(3)) for d in updir.iterdir()
            if d.is_dir() and SET_RE.match(d.name) and SET_RE.match(d.name).group(2) == "m"}
    k = next(i for i in range(len(have) + 1) if i not in have)
    label = re.sub(r"[\x00-\x1f\x7f]", "", str(spec.get("label") or "")).strip()[:80]
    bcset = BCSet(id=f"{uid}_m{k}", attempt=-1, bcs=bcs, words=words)
    return _write_set(updir, up, bcset, E, nu, cap, "manual",
                      f"{up['name']}: {label or 'your BC set'} (m{k})")


def bcset_of(p: GoldProblem):
    from feacore.case import BC
    from feacore.goldgen.bcgen import BCSet

    b = p._private()["bcset"]   # noqa: SLF001
    return BCSet(id=b["id"], attempt=b.get("attempt", 0), bcs=[BC(**x) for x in b["bcs"]],
                 words=b.get("words", []))


# --------------------------------------------------------------------------- references


def adopt_study(pid: str, runs: list[dict], source: dict) -> dict:
    """Certify a set's mesh sequence with the gold rules and store the result as its reference.

    ``runs`` (coarse to fine, at least three): ``{"level", "h", "h_rep", "n_dofs", "qois"}``. A QoI becomes a
    reference when it converges and its fine-grid GCI is within the gold tolerance (1 % energy, 2 %
    displacement and reactions, 3 % stress); diverging QoIs are recorded as singular."""
    from feacore.goldgen.run import certify

    if len(runs) < 3:
        raise CadError("a reference needs a study on at least three meshes")
    d = find_set(pid)
    p = CustomProblem(d)
    refs, qois = certify(bcset_of(p), runs)
    _write_json(d / "private" / "study.json", {"runs": runs, "qois": qois, "source": source})
    ref = json.loads((d / "private" / "reference.json").read_text())
    ref["references"] = [asdict(r) for r in refs]
    ref["reference_study"] = {**source, "certified_at": _now()}
    _write_json(d / "private" / "reference.json", ref)
    return {"problem_id": pid, "references": ref["references"],
            "singular": [k for k, v in qois.items() if v.get("kind") == "singular"],
            "kinds": {k: v.get("kind") for k, v in qois.items()}}


def study_levels(study: dict) -> list[dict]:
    """A finished feacore.ui convergence study (run-store record) as adopt_study's runs."""
    res = study.get("result") or {}
    return [{"level": i, "h": r.get("h_nominal"), "h_rep": (r.get("info") or {}).get("h_rep"),
             "n_dofs": (r.get("info") or {}).get("n_dofs"), "solve_s": (r.get("info") or {}).get("wall_time"),
             "qois": r.get("qois") or {}} for i, r in enumerate(res.get("runs") or [])]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def study_meshes(updir: Path, levels: int, max_cells: int, h0_frac: float, threads: int, log=print) -> list[dict]:
    """The upload's nested reference meshes L0..L<levels-1> (feacore.goldgen.mesh.mesh_part, the gold
    generator's plan search), reused when they exist for the same settings."""
    from feacore.goldgen.mesh import fit_h0, mesh_part

    up = read_upload(updir)
    part = _part(up)
    mdir = updir / "_meshes" / up["id"]
    plan = {"levels": levels, "max_cells": max_cells, "h0_frac": h0_frac, "h0_frac_max": H0_FRAC_MAX}
    sides = [mdir / f"L{k}.json" for k in range(levels)]
    if (mdir / "plan.json").is_file() and json.loads((mdir / "plan.json").read_text()) == plan and \
            all(s.is_file() and s.with_suffix(".msh").is_file() for s in sides):
        log(f"meshes: reusing {mdir}")
        return [{**json.loads(s.read_text()), "path": str(s.with_suffix(".msh"))} for s in sides]
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "plan.json").unlink(missing_ok=True)
    frac = fit_h0(updir / "geometry.step", part, h0_frac, levels - 1, max_cells, mdir, threads=threads) \
        if max_cells else h0_frac
    log(f"meshes: base size {frac:.4g} x diagonal ({frac * part.diag * 1000:.2f} mm), {levels} nested levels")
    t0 = time.perf_counter()
    metas = mesh_part(updir / "geometry.step", part, list(range(levels)), mdir, threads=threads, h0_frac=frac)
    if frac > H0_FRAC_MAX and not all(m.get("nested", True) for m in metas):
        # the cap asked for a base size above the part's size and splitting such a mesh failed (goldgen fell
        # back to remeshing, whose levels barely refine there): mesh again from the diagonal
        log(f"meshes: splitting failed at a base size of {frac:.3g} x the diagonal; meshing again at "
            f"{H0_FRAC_MAX:g} x (the finest level will have more cells than the cap)")
        frac = max(h0_frac, H0_FRAC_MAX)
        metas = mesh_part(updir / "geometry.step", part, list(range(levels)), mdir, threads=threads, h0_frac=frac)
    out = []
    for k, m in enumerate(metas):
        path = Path(m["path"])
        m = {**m, "mesh_level": k, "file": f"_meshes/{up['id']}/{path.name}", "bytes": path.stat().st_size,
             "sha256": _sha256(path)}
        _write_json(path.with_suffix(".json"), m)
        out.append(m)
        log(f"  L{k}: {m['cells']} cells, h {m['h'] * 1000:.2f} mm, {m['scheme']}, min scaled Jacobian "
            f"{m['min_sj']:.3f}")
    if max_cells and out[-1]["cells"] > max_cells:   # the part's faces and edges set a floor
        log(f"meshes: the finest level has {out[-1]['cells']} cells, above the cap of {max_cells}: the coarsest mesh "
            f"of this part already has {out[0]['cells']} cells (its faces and edges set that floor)")
    nested = all(m.get("nested", True) for m in out)
    if not nested:
        log("meshes: splitting failed the curved-mesh checks, so each level was meshed on its own (sizes halved): "
            "a less clean refinement than nested splits (feacore.goldgen.mesh.mesh_part)")
    _write_json(mdir / "plan.json", plan)
    up["meshes"] = {f"L{m['mesh_level']}": {k: m.get(k) for k in ("file", "h", "cells", "scheme", "min_sj", "bytes",
                                                                    "sha256", "nested")} for m in out}
    up["mesh_plan"] = {**plan, "h0_frac_used": frac, "nested": nested, "seconds": round(time.perf_counter() - t0, 1)}
    _write_json(updir / "upload.json", up)
    return out


def reference_study(uid: str, sets: list[str] | None = None, levels: int = 3, max_cells: int = 30_000,
                    h0_frac: float = H0_FRAC, threads: int = 1, record: bool = True, log=print) -> dict:
    """The gold study of an upload's BC sets on the CPU: nested curved tet10 meshes (the finest at
    most ``max_cells`` cells), every set solved on every level with the fixed solver, each QoI
    certified with the gold rules (adopt_study). With ``record`` each solve also goes into the run
    store (agent ``reference:L<k>``), where the dashboard's field viewer shows it."""
    from feacore.diagnostics import precheck
    from feacore.solver import solve

    updir = find_upload(uid)
    problems = [p for p in _sets_in(updir)[0] if not sets or p.id in sets]
    missing = sorted(set(sets or []) - {p.id for p in problems})
    if missing:
        raise CadError(f"not sets of {uid}: {', '.join(missing)}")
    if not problems:
        raise CadError(f"{uid} has no BC set yet: define or generate one first")
    metas = study_meshes(updir, levels, max_cells, h0_frac, threads, log)
    store = None
    if record:
        from feacore.runstore import RunStore

        store = RunStore()
    summary = {}
    for p in problems:
        bcset, mat = bcset_of(p), p.material
        pre = precheck(reference_case(metas[0]["path"], bcset, mat["E"], mat["nu"]))
        if pre.status == "fail":
            log(f"{p.id}: the setup fails the precheck ({', '.join(pre.ids())}); not solved")
            summary[p.id] = {"result": "invalid_setup", "flags": pre.ids()}
            continue
        runs = []
        for k, m in enumerate(metas):
            t0 = time.perf_counter()
            res = solve(reference_case(m["path"], bcset, mat["E"], mat["nu"]), backend="skfem")
            vals = {q: float(v) for q, v in res.evaluate_all().items()}
            runs.append({"level": k, "h": m["h"], "h_rep": res.info.h_rep, "n_dofs": res.info.n_dofs,
                         "solve_s": res.info.wall_time, "total_s": time.perf_counter() - t0, "qois": vals})
            msg = f"{p.id} L{k}: {res.info.n_dofs:,} DOFs, {time.perf_counter() - t0:.1f} s"
            if store is not None:
                rid = store.record_result(res, mesh_path=m["path"], problem_id=p.id, agent_id=f"reference:L{k}",
                                          level=k, h_nominal=m["h"], source="cad", copy_mesh=False,
                                          meta={"source": "cad", "part": uid, "mesh_level": k})
                runs[-1]["run_id"] = rid
                msg += f", run {rid}"
            log(msg)
        out = adopt_study(p.id, runs, {"kind": "nested reference study", "levels": levels, "max_cells": max_cells,
                                       "h0_frac": h0_frac, "meshes": [m["file"] for m in metas]})
        refs = ", ".join(f"{r['qoi']} = {r['value']:.5g} (tol {r['tolerance']:.1%})" for r in out["references"])
        log(f"{p.id}: certified {refs or 'nothing (no QoI converged within the gold tolerance)'}"
            f"{'; singular: ' + ', '.join(out['singular']) if out['singular'] else ''}")
        summary[p.id] = {"result": "ok", "references": [r["qoi"] for r in out["references"]],
                         "singular": out["singular"], "dofs": [r["n_dofs"] for r in runs]}
    return summary


# --------------------------------------------------------------------------- helpers + CLI


def _now() -> str:
    from feacore.runstore import now_iso

    return now_iso()


def _write_json(path: Path, data) -> None:
    from feacore.runstore import write_json

    write_json(path, data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m feacore.problems.custom", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="uploads and their BC sets")
    s = sub.add_parser("add", help="upload a STEP file (into the run store's cad/)")
    s.add_argument("step")
    s.add_argument("--units", choices=sorted(UNITS), default="si")
    s.add_argument("--generate", type=int, default=0, help="also generate this many BC sets")
    s = sub.add_parser("reference", help="nested reference study of an upload's BC sets (CPU)")
    s.add_argument("upload")
    s.add_argument("--sets", default="", help="comma-separated set ids (default: every set of the upload)")
    s.add_argument("--levels", type=int, default=3)
    s.add_argument("--max-cells", type=int, default=30_000, help="cap on the finest level's cells")
    s.add_argument("--h0-frac", type=float, default=H0_FRAC, help="base size / bounding-box diagonal")
    s.add_argument("--threads", type=int, default=1, help="gmsh threads")
    s.add_argument("--no-record", action="store_true", help="do not record the solves in the run store")
    s = sub.add_parser("inspect", help=argparse.SUPPRESS)
    s.add_argument("dir")
    s.add_argument("--units", required=True)
    s.add_argument("--id", required=True)
    s.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "inspect":
        d = Path(a.dir)
        try:
            out = {"ok": True, **inspect(d / "source.step", d, a.units, a.id, a.name)}
        except CadError as e:
            out = {"ok": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001 - reported to the uploader in plain words
            out = {"ok": False, "error": f"the CAD kernel failed on this file: {type(e).__name__}: {e}"}
        _write_json(d / "inspect.json", out)
        return 0 if out["ok"] else 3
    if a.cmd == "list":
        for updir in upload_dirs():
            up = read_upload(updir)
            print(f"{up['id']}  {up['name']}  {up['n_faces']} faces, diagonal {up['diag_mm']:.1f} mm  ({updir})")
            for p in _sets_in(updir)[0]:
                refs = ", ".join(r.qoi for r in p.references()) or "none (reference-free scoring)"
                print(f"    {p.id}: scored {p.scored_qoi()}, node cap {p.node_cap}, references: {refs}")
        return 0
    try:
        if a.cmd == "add":
            path = Path(a.step)
            rec = create_upload(path.name, path.read_bytes(), a.units)
            print(f"{rec['id']}: {rec['n_faces']} faces, diagonal {rec['diag_mm']:.1f} mm"
                  f"{' (already uploaded)' if rec.get('existing') else ''} -> {find_upload(rec['id'])}")
            for w in rec.get("warnings") or []:
                print(f"  warning: {w}")
            for pid in generate_sets(rec["id"], a.generate) if a.generate else []:
                print(f"  {pid}")
            return 0
        sets = [s.strip() for s in a.sets.split(",") if s.strip()]
        summary = reference_study(a.upload, sets or None, a.levels, a.max_cells, a.h0_frac, a.threads,
                                  record=not a.no_record, log=lambda m: print(m, flush=True))
        return 0 if all(v["result"] == "ok" for v in summary.values()) else 1
    except (CadError, KeyError) as e:
        print(f"error: {str(e).strip(chr(34))}", file=sys.stderr)
        return 2


if __name__ != "__main__":
    _found, _errors = load_all()
    _register(_found)
    for _dir, _err in _errors.items():
        warnings.warn(f"custom problem {_dir} skipped: {_err}", stacklevel=1)
else:   # python -m: run the importable copy, so there is one CustomProblem class and one registry entry
    from feacore.problems.custom import main as _main

    sys.exit(_main())
