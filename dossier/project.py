"""A project on disk: the inputs people write, the records the pipeline derives, the reviews people give.

    <project>/
      project.yaml                 name, the component under verification, where the other files are
      system.yaml                  dossier.model.System
      materials.yaml               {materials: [Material, ...]}
      policy.yaml                  dossier.model.Policy
      terms.yaml                   (optional) project technical names and verbs for the STE check
      components/<name>/
        component.yaml             dossier.model.Component
        requirements.yaml          {document: "...", requirements: [Requirement, ...]}
        geometry.step              the CAD
      reviews.yaml                 written by `dossier review` / the dashboard; owned by people
      out/                         derived, regenerable; never edited by hand
        context.json  geometry.json  validate.json  plan.json  assess.json
        analyses/<load case>/analysis.json  (+ meshes, pictures)
        report/index.html

Every stage file has the same header and holds lists of records under fixed keys (COLLECTIONS). A stage
is a pure function of the inputs and of the stage files before it; `Project.write_stage` does the I/O.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pydantic
import yaml

from dossier import __version__, model
from dossier.model import (AnalysisRecord, Assumption, ComplianceItem, Component, Decision, Finding,
                           HandCalcResult, LoadCase, Material, Policy, ProjectFile, Record, Requirement,
                           RequirementVerdict, Review, System)

# record collections a stage file may hold: key -> model
COLLECTIONS: dict[str, type[pydantic.BaseModel]] = {
    "findings": Finding,
    "decisions": Decision,
    "assumptions": Assumption,
    "load_cases": LoadCase,
    "hand_calcs": HandCalcResult,
    "analyses": AnalysisRecord,
    "compliance": ComplianceItem,
    "verdicts": RequirementVerdict,
}
STAGE_FILES = ("context", "geometry", "validate", "plan", "assess")


class ProjectError(ValueError):
    """The project's input files are wrong. `problems` lists every problem found, one per line."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("\n".join(self.problems))


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ProjectError(f"{path}: file not found")
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ProjectError(f"{path}: not valid YAML: {e}") from e


def _validate(cls, data: Any, path: Path, what: str = ""):
    try:
        return cls.model_validate(data)
    except pydantic.ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            lines.append(f"{path}{': ' + what if what else ''}: {loc or '(top level)'}: {err['msg']}")
        raise ProjectError(lines) from e


def digest_of(obj: Any) -> str:
    """A short content hash of a record or of plain data (reviews keep it to detect later changes)."""
    if isinstance(obj, pydantic.BaseModel):
        obj = obj.model_dump(mode="json", by_alias=True)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Project:
    def __init__(self, root: Path, file: ProjectFile, system: System, components: dict[str, Component],
                 component_dirs: dict[str, Path], requirements: dict[str, list[Requirement]],
                 requirement_docs: dict[str, str], materials: dict[str, Material], policy: Policy, terms: dict):
        self.root = Path(root)
        self.file = file
        self.system = system
        self.components = components
        self.component_dirs = component_dirs
        self.requirements = requirements          # component id -> its requirements
        self.requirement_docs = requirement_docs  # component id -> the specification's name
        self.materials = materials
        self.policy = policy
        self.terms = terms

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, root: str | Path) -> "Project":
        """Read and check every input file. Raises ProjectError listing every problem."""
        root = Path(root).resolve()
        file = _validate(ProjectFile, _read_yaml(root / "project.yaml"), root / "project.yaml")
        system = _validate(System, _read_yaml(root / file.system), root / file.system)
        mats_raw = _read_yaml(root / file.materials)
        materials = {}
        for i, m in enumerate(mats_raw.get("materials", []) if isinstance(mats_raw, dict) else []):
            mat = _validate(Material, m, root / file.materials, f"materials[{i}]")
            materials[mat.id] = mat
        policy = _validate(Policy, _read_yaml(root / file.policy), root / file.policy)
        terms = _read_yaml(root / file.terms) if file.terms else {}
        components, dirs, reqs, docs = {}, {}, {}, {}
        for rel in file.components:
            d = root / rel
            comp = _validate(Component, _read_yaml(d / "component.yaml"), d / "component.yaml")
            components[comp.id] = comp
            dirs[comp.id] = d
            rpath = d / "requirements.yaml"
            raw = _read_yaml(rpath) if rpath.exists() else {}
            docs[comp.id] = str(raw.get("document", "")) if isinstance(raw, dict) else ""
            items = raw.get("requirements", []) if isinstance(raw, dict) else raw
            reqs[comp.id] = [_validate(Requirement, r, rpath, f"requirements[{i}]") for i, r in enumerate(items or [])]
        p = cls(root, file, system, components, dirs, reqs, docs, materials, policy, terms)
        problems = p._integrity()
        if problems:
            raise ProjectError(problems)
        return p

    def _integrity(self) -> list[str]:
        """References that point nowhere and ids used twice. Softer checks are findings (validate stage)."""
        out: list[str] = []
        s = self.system
        seen: dict[str, str] = {}

        def claim(i: str, what: str):
            if i in seen:
                out.append(f"id {i!r} is used twice: {seen[i]} and {what}")
            seen[i] = what

        claim(s.id, "the system")
        elements = {e.id for e in s.elements}
        phases = {ph.id for ph in s.phases}
        for e in s.elements:
            claim(e.id, "an element")
        for ph in s.phases:
            claim(ph.id, "a phase")
        for dc in s.duty_cycles:
            claim(dc.id, "a duty cycle")
            out += [f"duty cycle {dc.id}: phase {x!r} does not exist" for x in dc.sequence if x not in phases]
        for c in s.connections:
            claim(c.id, "a connection")
            for end in (c.from_, c.to):
                if end not in elements:
                    out.append(f"connection {c.id}: element {end!r} does not exist")
            for ef in c.effects:
                claim(ef.id, "an effect")
                out += [f"effect {ef.id}: phase {x!r} does not exist" for x in ef.phases() if x not in phases]
                if ef.acts_on is not None and ef.acts_on not in (c.from_, c.to):
                    out.append(f"effect {ef.id}: `acts_on: {ef.acts_on}` is not an end of connection {c.id}")
        connections = {c.id for c in s.connections}
        if self.file.subject not in self.components:
            out.append(f"project.yaml: subject {self.file.subject!r} is not one of the components "
                       f"({', '.join(self.components) or 'none'})")
        for comp in self.components.values():
            claim(comp.id, "a component")
            if comp.element not in elements:
                out.append(f"component {comp.id}: element {comp.element!r} does not exist in the system")
            if comp.material not in self.materials:
                out.append(f"component {comp.id}: material {comp.material!r} is not in {self.file.materials}")
            if not (self.component_dirs[comp.id] / comp.cad.file).exists():
                out.append(f"component {comp.id}: CAD file {comp.cad.file!r} not found in {self.component_dirs[comp.id]}")
            for f in comp.features:
                claim(f, "a feature")
            for it in comp.interfaces:
                claim(it.id, "an interface")
                if it.connection not in connections:
                    out.append(f"interface {it.id}: connection {it.connection!r} does not exist in the system")
                for role, feats in it.features.items():
                    out += [f"interface {it.id}: feature {f!r} ({role}) is not defined in the component"
                            for f in feats if f not in comp.features and not (f == "*" and role == "external")]
            for scl in comp.scls:
                claim(scl.id, "a stress classification line")
            for hc in comp.hand_calcs:
                claim(hc.id, "a hand calculation")
            for r in self.requirements.get(comp.id, []):
                claim(r.id, "a requirement")
                out += [f"requirement {r.id}: phase {x!r} does not exist" for x in r.applies.phases if x not in phases]
                dcs = {dc.id for dc in s.duty_cycles}
                out += [f"requirement {r.id}: duty cycle {x!r} does not exist" for x in r.applies.duty_cycles
                        if x not in dcs]
        for m in self.materials.values():
            claim(m.id, "a material")
        return out

    # ------------------------------------------------------------------ lookups
    @property
    def subject(self) -> Component:
        return self.components[self.file.subject]

    @property
    def out(self) -> Path:
        return self.root / "out"

    def component_dir(self, component_id: str | None = None) -> Path:
        return self.component_dirs[component_id or self.file.subject]

    def cad_path(self, component_id: str | None = None) -> Path:
        c = self.components[component_id or self.file.subject]
        return self.component_dirs[c.id] / c.cad.file

    def material_of(self, component_id: str | None = None) -> Material:
        return self.materials[self.components[component_id or self.file.subject].material]

    def reqs(self, component_id: str | None = None) -> list[Requirement]:
        return self.requirements.get(component_id or self.file.subject, [])

    def requirement(self, req_id: str) -> Requirement:
        for rs in self.requirements.values():
            for r in rs:
                if r.id == req_id:
                    return r
        raise KeyError(req_id)

    def phase(self, phase_id: str) -> model.Phase:
        return next(p for p in self.system.phases if p.id == phase_id)

    def phases_of(self, req: Requirement) -> list[str]:
        """The phases a requirement applies to: its list, plus every phase of its category."""
        ids = list(req.applies.phases)
        cat = req.applies.category
        if cat:
            ids += [p.id for p in self.system.phases if (cat == "all" or p.category == cat) and p.id not in ids]
        return ids

    def inputs(self) -> dict[str, tuple[str, Any]]:
        """Every input record by id: (kind, object). Kinds: system, element, phase, duty_cycle, connection,
        effect, component, feature, interface, scl, hand_calc_spec, requirement, material, parameter, policy."""
        s = self.system
        out: dict[str, tuple[str, Any]] = {s.id: ("system", s), self.policy.id: ("policy", self.policy)}
        out.update({e.id: ("element", e) for e in s.elements})
        out.update({p.id: ("phase", p) for p in s.phases})
        out.update({d.id: ("duty_cycle", d) for d in s.duty_cycles})
        for c in s.connections:
            out[c.id] = ("connection", c)
            out.update({ef.id: ("effect", ef) for ef in c.effects})
        for name, prm in s.parameters.items():
            out[name] = ("parameter", prm)
        for comp in self.components.values():
            out[comp.id] = ("component", comp)
            out.update({k: ("feature", f) for k, f in comp.features.items()})
            out.update({i.id: ("interface", i) for i in comp.interfaces})
            out.update({x.id: ("scl", x) for x in comp.scls})
            out.update({h.id: ("hand_calc_spec", h) for h in comp.hand_calcs})
            for r in self.requirements.get(comp.id, []):
                out[r.id] = ("requirement", r)
                for name, prm in r.parameters.items():
                    out.setdefault(name, ("parameter", prm))
        out.update({m.id: ("material", m) for m in self.materials.values()})
        return out

    def inputs_digest(self) -> str:
        """A hash of every input file: a stage file made from other inputs is stale."""
        h = hashlib.sha256()
        files = [self.root / "project.yaml", self.root / self.file.system, self.root / self.file.materials,
                 self.root / self.file.policy]
        for d in self.component_dirs.values():
            files += sorted(p for p in d.iterdir() if p.is_file())
        for f in files:
            if f.exists():
                h.update(f.name.encode())
                h.update(f.read_bytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------ stage files
    def stage_path(self, stage: str) -> Path:
        return self.out / f"{stage}.json"

    def write_stage(self, stage: str, path: Path | None = None, **payload: Any) -> Path:
        """Write a stage file. Lists under COLLECTIONS keys hold records (models or dicts); other keys
        hold plain JSON data."""
        path = path or self.stage_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc: dict[str, Any] = {"schema": model.SCHEMA_VERSION, "stage": stage, "tool": f"dossier {__version__}",
                               "inputs_digest": self.inputs_digest(),
                               "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        for k, v in payload.items():
            if k in COLLECTIONS:
                doc[k] = [x.model_dump(mode="json", by_alias=True) if isinstance(x, pydantic.BaseModel) else x for x in v]
            elif isinstance(v, pydantic.BaseModel):
                doc[k] = v.model_dump(mode="json", by_alias=True)
            else:
                doc[k] = v
        path.write_text(json.dumps(doc, indent=1, default=float) + "\n")
        return path

    def read_stage(self, stage: str, path: Path | None = None) -> dict | None:
        path = path or self.stage_path(stage)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def stage_files(self) -> list[Path]:
        files = [self.stage_path(s) for s in STAGE_FILES if self.stage_path(s).exists()]
        files += sorted((self.out / "analyses").glob("*/analysis.json")) if (self.out / "analyses").exists() else []
        files += sorted((self.root / "agent").glob("*.json")) if (self.root / "agent").exists() else []
        return files

    def analysis_dir(self, load_case_id: str) -> Path:
        return self.out / "analyses" / load_case_id

    def collection(self, key: str) -> list:
        """Every record of one collection ("findings", "decisions", "load_cases", ...) across the stage
        files, as models, in stage order."""
        cls = COLLECTIONS[key]
        out = []
        for f in self.stage_files():
            doc = json.loads(f.read_text())
            out += [cls.model_validate(x) for x in doc.get(key, [])]
        return out

    def records(self) -> dict[str, Record]:
        """Every derived record that has an id, by id."""
        out: dict[str, Record] = {}
        for key in COLLECTIONS:
            if key == "verdicts":
                continue
            for r in self.collection(key):
                out[r.id] = r
        return out

    def stale(self) -> list[str]:
        """Stage files made from inputs that have changed since."""
        d = self.inputs_digest()
        return [str(f.relative_to(self.root)) for f in self.stage_files()
                if json.loads(f.read_text()).get("inputs_digest") not in (None, d)]

    # ------------------------------------------------------------------ reviews
    @property
    def reviews_path(self) -> Path:
        return self.root / "reviews.yaml"

    def reviews(self) -> dict[str, Review]:
        if not self.reviews_path.exists():
            return {}
        raw = yaml.safe_load(self.reviews_path.read_text()) or {}
        return {k: Review.model_validate({"record": k, **v}) for k, v in (raw.get("reviews") or {}).items()}

    def set_review(self, record_id: str, state: str, by: str, comment: str = "") -> Review:
        recs = self.records()
        if record_id not in recs:
            raise KeyError(f"{record_id} is not a derived record of this project")
        rv = Review(record=record_id, state=state, by=by, comment=comment,
                    date=datetime.date.today().isoformat(), digest=digest_of(recs[record_id]))
        allr = self.reviews()
        allr[record_id] = rv
        body = {"reviews": {k: v.model_dump(exclude={"record"}) for k, v in sorted(allr.items())}}
        self.reviews_path.write_text(
            "# Reviews of derived records. Written by `python -m dossier review` and the dashboard.\n"
            + yaml.safe_dump(body, sort_keys=False, allow_unicode=True))
        return rv

    def review_of(self, record: Record, reviews: dict[str, Review] | None = None) -> tuple[Review, bool]:
        """(the review, stale): stale when the record changed after the person reviewed it."""
        rv = (reviews if reviews is not None else self.reviews()).get(record.id) or Review(record=record.id)
        return rv, bool(rv.digest) and rv.digest != digest_of(record)


def record_ids(records: Iterable[Record]) -> list[str]:
    return [r.id for r in records]


def make_id(prefix: str, *parts: str) -> str:
    """A stable, readable record id from its key: make_id("DEC", "LC-FIRE", "support") -> DEC-LC-FIRE-SUPPORT.
    The same key gives the same id on every run, so reviews survive a re-run."""
    body = "-".join(p for p in parts if p)
    body = "".join(ch if ch.isalnum() else "-" for ch in body.upper())
    while "--" in body:
        body = body.replace("--", "-")
    body = body.strip("-")
    if body.startswith(prefix + "-"):
        body = body[len(prefix) + 1:]
    return f"{prefix}-{body}" if body else prefix
