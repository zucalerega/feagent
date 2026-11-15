"""Everything the report shows, read once from the project.

`Data.load(project)` reads the inputs, the context, every stage file, the reviews and the trace graph
(`dossier.trace.graph`, the one graph of the project). The sections only read from it.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dossier import __version__, trace
from dossier import context as context_mod
from dossier.project import COLLECTIONS, STAGE_FILES, Project

REVIEWABLE = ("decision", "assumption", "finding")
STAGE_ORDER = ("context", "geometry", "validate", "plan", "idealize", "mesh", "solve", "assess", "report")
PIPELINE = ("context", "geometry", "validate", "plan", "analyze", "assess")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class StageFile:
    path: str                 # relative to the project folder
    stage: str
    tool: str = ""
    written: str = ""
    inputs_digest: str = ""
    stale: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    options: dict | None = None


@dataclass
class Data:
    project: Project
    ctx: Any
    graph: dict
    built: str
    version: str
    digest: str
    records: dict[str, Any]
    col: dict[str, list]                      # collection name -> records, in stage order, one per id
    verdicts: dict[str, Any]                  # requirement id -> RequirementVerdict
    geo: dict | None
    files: list[StageFile]
    reviews: dict[str, dict]                  # record id -> {"state", "by", "date", "comment", "stale"}
    about: dict[str, list]                    # record id -> decisions, assumptions and findings about it
    inputs: dict[str, tuple[str, Any]]
    input_files: list[dict]
    source_of: dict[str, str]                 # record id -> the stage file it comes from

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, project: Project) -> "Data":
        ctx = context_mod.load(project)
        digest = project.inputs_digest()
        col: dict[str, dict[str, Any]] = {k: {} for k in COLLECTIONS}
        verdicts: dict[str, Any] = {}
        files: list[StageFile] = []
        source_of: dict[str, str] = {}
        for f in project.stage_files():
            doc = json.loads(f.read_text())
            rel = str(f.relative_to(project.root))
            sf = StageFile(path=rel, stage=str(doc.get("stage", "")), tool=str(doc.get("tool", "")),
                           written=str(doc.get("written", "")), inputs_digest=str(doc.get("inputs_digest", "")),
                           stale=doc.get("inputs_digest") not in (None, digest), options=doc.get("options"))
            for key, model in COLLECTIONS.items():
                items = [model.model_validate(x) for x in doc.get(key, [])]
                if items:
                    sf.counts[key] = len(items)
                for r in items:
                    if key == "verdicts":
                        verdicts[r.requirement] = r
                    else:
                        col[key][r.id] = r
                        source_of[r.id] = rel
            files.append(sf)
        records = {i: r for key in COLLECTIONS if key != "verdicts" for i, r in col[key].items()}
        raw_reviews = project.reviews()
        reviews: dict[str, dict] = {}
        for r in records.values():
            if getattr(r, "kind", "") not in REVIEWABLE:
                continue
            rv, stale = project.review_of(r, raw_reviews)
            reviews[r.id] = {"state": rv.state, "by": rv.by, "date": rv.date, "comment": rv.comment, "stale": bool(stale)}
        about: dict[str, list] = {}
        for key in ("decisions", "assumptions", "findings"):
            for r in col[key].values():
                targets = [r.subject] + [l.to for l in r.links if l.rel == "about"]
                for t in dict.fromkeys(targets):
                    if t != r.id:
                        about.setdefault(t, []).append(r)
        geo_doc = project.read_stage("geometry")
        input_files = []
        for p in cls._input_paths(project):
            if p.exists():
                input_files.append({"path": str(p.relative_to(project.root)), "sha256": _sha(p), "size": p.stat().st_size})
        return cls(project=project, ctx=ctx, graph=trace.graph(project, ctx),
                   built=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                   version=__version__, digest=digest, records=records,
                   col={k: list(v.values()) for k, v in col.items()}, verdicts=verdicts,
                   geo=(geo_doc or {}).get("geometry"), files=files, reviews=reviews, about=about,
                   inputs=project.inputs(), input_files=input_files, source_of=source_of)

    @staticmethod
    def _input_paths(project: Project) -> list[Path]:
        f = project.file
        paths = [project.root / "project.yaml", project.root / f.system, project.root / f.materials, project.root / f.policy]
        if f.terms:
            paths.append(project.root / f.terms)
        for d in project.component_dirs.values():
            paths += sorted(p for p in d.iterdir() if p.is_file())
        if project.reviews_path.exists():
            paths.append(project.reviews_path)
        return paths

    # ------------------------------------------------------------------ collections
    @property
    def findings(self) -> list:
        return self.col["findings"]

    @property
    def decisions(self) -> list:
        return self.col["decisions"]

    @property
    def assumptions(self) -> list:
        return self.col["assumptions"]

    @property
    def load_cases(self) -> list:
        return self.col["load_cases"]

    @property
    def hand_calcs(self) -> list:
        return self.col["hand_calcs"]

    @property
    def analyses(self) -> list:
        """The analyses in the order of their load cases in the plan."""
        order = {lc.id: i for i, lc in enumerate(self.load_cases)}
        return sorted(self.col["analyses"], key=lambda a: order.get(a.load_case, len(order)))

    @property
    def compliance(self) -> list:
        return self.col["compliance"]

    # ------------------------------------------------------------------ lookups
    def ran(self, stage: str) -> bool:
        if stage == "analyze":
            return any(f.stage == "analyze" for f in self.files)
        return any(f.stage == stage and f.path.startswith("out") for f in self.files)

    def stage_file(self, stage: str) -> StageFile | None:
        return next((f for f in self.files if f.stage == stage and f.path.startswith("out")), None)

    @property
    def stale_files(self) -> list[str]:
        return [f.path for f in self.files if f.stale]

    def node(self, rid: str) -> dict | None:
        return self.graph["nodes"].get(rid)

    def title(self, rid: str) -> str:
        n = self.node(rid)
        return n["title"] if n else rid

    def kind(self, rid: str) -> str:
        n = self.node(rid)
        return n["kind"] if n else ""

    def lc(self, lc_id: str):
        return next((x for x in self.load_cases if x.id == lc_id), None)

    def analysis_of(self, lc_id: str):
        return next((a for a in self.col["analyses"] if a.load_case == lc_id), None)

    def hand_calcs_of(self, lc_id: str) -> list:
        return [h for h in self.hand_calcs if h.load_case == lc_id]

    def items_of_req(self, req_id: str) -> list:
        return [i for i in self.compliance if i.requirement == req_id]

    def items_of_lc(self, lc_id: str) -> list:
        return [i for i in self.compliance if i.load_case == lc_id]

    def findings_about(self, rid: str) -> list:
        return [r for r in self.about.get(rid, []) if r.kind == "finding"]

    def choices_about(self, rid: str) -> list:
        return [r for r in self.about.get(rid, []) if r.kind in ("decision", "assumption")]

    def review_state(self, rid: str) -> str:
        """open | accepted | rejected | changes_requested | stale (the record changed after its review)."""
        rv = self.reviews.get(rid)
        if rv is None:
            return ""
        return "stale" if rv["stale"] else rv["state"]

    def anchors(self) -> set[str]:
        """Every record id that has a card in the report: the nodes of the trace graph that are records."""
        return {i for i, n in self.graph["nodes"].items() if n["kind"] not in ("document", "missing")}

    def stage_order(self, stage: str) -> int:
        return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
