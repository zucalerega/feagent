"""Command line of dossier.

    python -m dossier new DIR [--example mov_valve]     copy an example project to DIR
    python -m dossier check PROJECT                     read the inputs, evaluate every number, list problems
    python -m dossier run PROJECT [--from S] [--to S] [--quick] [--only LC ...] [--force]
    python -m dossier status PROJECT                    stages done, stale files, verdicts, open reviews
    python -m dossier show PROJECT ID                   one record (input or derived) with what it links to
    python -m dossier review PROJECT ID --accept|--reject|--changes --by NAME [--comment TEXT]
    python -m dossier params PROJECT [NAME ...]         the named numbers with their sources

Stages: context, geometry, validate, plan, analyze, assess, report (dossier/pipeline.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from dossier import context as context_mod
from dossier import pipeline
from dossier.project import Project, ProjectError

EXAMPLES = Path(__file__).with_name("examples")


def _project(path: str) -> Project:
    return Project.load(path)


def cmd_new(a) -> int:
    src = EXAMPLES / a.example
    if not src.is_dir():
        print(f"no example {a.example!r}; examples: {', '.join(p.name for p in EXAMPLES.iterdir() if p.is_dir())}")
        return 2
    dst = Path(a.dir)
    if dst.exists() and any(dst.iterdir()):
        print(f"{dst} exists and is not empty")
        return 2
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("out", "__pycache__", "reviews.yaml"))
    print(f"{dst}: a copy of the example {a.example}. Next: python -m dossier run {dst}")
    return 0


def cmd_check(a) -> int:
    p = _project(a.project)
    ctx = context_mod.build(p)
    print(f"{p.file.id}: {p.file.name}")
    print(f"  system {p.system.id}: {len(p.system.elements)} elements, {len(p.system.connections)} connections, "
          f"{len(p.system.phases)} phases, {len(p.system.duty_cycles)} duty cycles")
    print(f"  subject {p.subject.id}: {len(p.subject.features)} features, {len(p.subject.interfaces)} interfaces, "
          f"{len(p.reqs())} requirements, material {p.subject.material}")
    print(f"  {len(ctx.params)} named numbers and {len(ctx.effects)} effects evaluated; units are consistent")
    return 0


def cmd_params(a) -> int:
    p = _project(a.project)
    ctx = context_mod.build(p)
    for name, prm in ctx.params.items():
        if a.names and name not in a.names:
            continue
        src = prm.expr and f"= {prm.expr}" or prm.source
        print(f"{name:22s} {prm.display:>22s}   [{prm.origin}] {prm.title}{' ' + src if src else ''}")
    return 0


def cmd_run(a) -> int:
    opts = pipeline.Options(quick=a.quick, only=a.only or [], force=a.force, pictures=not a.no_pictures,
                            record_runs=a.record_runs)
    if a.levels:
        opts.levels = a.levels
    if a.node_cap:
        opts.node_cap = a.node_cap
    out = pipeline.run(_project(a.project), first=a.first, last=a.last, opts=opts)
    print(json.dumps(out, indent=1, default=str))
    return 0


def cmd_status(a) -> int:
    p = _project(a.project)
    files = p.stage_files()
    print(f"{p.file.id}: {len(files)} stage file(s)")
    for f in files:
        print(f"  {f.relative_to(p.root)}")
    stale = p.stale()
    if stale:
        print("stale (the inputs changed after these were written): " + ", ".join(stale))
    verdicts = p.collection("verdicts")
    for v in verdicts:
        m = f"  margin {v.margin:+.2f}" if v.margin is not None else ""
        print(f"  {v.requirement:14s} {v.verdict:13s}{m}")
    recs = p.records()
    reviews = p.reviews()
    need = [r for r in recs.values() if getattr(r, "kind", "") in ("decision", "assumption", "finding")]
    open_ = [r.id for r in need if p.review_of(r, reviews)[0].state == "open"]
    stale_reviews = [r.id for r in need if p.review_of(r, reviews)[1]]
    print(f"{len(need)} decisions, assumptions and findings; {len(open_)} without a review"
          + (f"; {len(stale_reviews)} changed after their review" if stale_reviews else ""))
    return 0


def cmd_show(a) -> int:
    p = _project(a.project)
    recs = p.records()
    inputs = p.inputs()
    if a.id in recs:
        r = recs[a.id]
        print(json.dumps(r.model_dump(mode="json", by_alias=True), indent=1))
        rv, stale = p.review_of(r)
        print(f"review: {rv.state}{' (the record changed after the review)' if stale else ''}"
              f"{' by ' + rv.by if rv.by else ''}{': ' + rv.comment if rv.comment else ''}")
        used_by = [x.id for x in recs.values() if any(l.to == a.id for l in x.refs())]
        if used_by:
            print("used by: " + ", ".join(used_by))
        return 0
    if a.id in inputs:
        kind, obj = inputs[a.id]
        print(f"{kind} (input)")
        print(json.dumps(obj.model_dump(mode="json", by_alias=True), indent=1) if hasattr(obj, "model_dump") else obj)
        used_by = [x.id for x in recs.values() if any(l.to == a.id for l in x.refs())]
        if used_by:
            print("used by: " + ", ".join(used_by))
        return 0
    print(f"no record {a.id!r}")
    return 2


def cmd_review(a) -> int:
    p = _project(a.project)
    state = "accepted" if a.accept else "rejected" if a.reject else "changes_requested"
    rv = p.set_review(a.id, state, a.by, a.comment or "")
    print(f"{rv.record}: {rv.state} by {rv.by} on {rv.date}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dossier", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new"); s.add_argument("dir"); s.add_argument("--example", default="mov_valve"); s.set_defaults(fn=cmd_new)
    s = sub.add_parser("check"); s.add_argument("project"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("params"); s.add_argument("project"); s.add_argument("names", nargs="*"); s.set_defaults(fn=cmd_params)
    s = sub.add_parser("run"); s.add_argument("project")
    s.add_argument("--from", dest="first", default="context", choices=pipeline.STAGES)
    s.add_argument("--to", dest="last", default="report", choices=pipeline.STAGES)
    s.add_argument("--quick", action="store_true", help="coarse meshes, two levels, no pictures (tests, first looks)")
    s.add_argument("--only", nargs="*", help="load case ids to analyse")
    s.add_argument("--force", action="store_true", help="analyse again even when the analysis file is current")
    s.add_argument("--levels", type=int); s.add_argument("--node-cap", type=int)
    s.add_argument("--no-pictures", action="store_true"); s.add_argument("--record-runs", action="store_true")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("status"); s.add_argument("project"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("show"); s.add_argument("project"); s.add_argument("id"); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("review"); s.add_argument("project"); s.add_argument("id")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--accept", action="store_true"); g.add_argument("--reject", action="store_true")
    g.add_argument("--changes", action="store_true")
    s.add_argument("--by", required=True); s.add_argument("--comment"); s.set_defaults(fn=cmd_review)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ProjectError as e:
        print("The project has problems:", file=sys.stderr)
        for line in e.problems:
            print("  " + line, file=sys.stderr)
        return 1
    except pipeline.StageError as e:
        print(str(e), file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"not found: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
