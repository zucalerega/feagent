"""Command line of the judgement roles.

    python -m dossier.agents roles
    python -m dossier.agents brief  PROJECT ROLE [--out FILE] [--focus ID ...]   what the role reads (Markdown)
    python -m dossier.agents schema ROLE                                          JSON schema of what it submits
    python -m dossier.agents submit PROJECT ROLE FILE.json [--by MODEL_NAME] [--check]
    python -m dossier.agents run    PROJECT ROLE [--mock] [--model sonnet|opus]   one call through the Claude API

ROLE is a role name or a start of it that only one role has (requirements-reviewer, plan-reviewer,
analysis-checker; `req`, `plan`, `analysis`). FILE.json can be `-` for the standard input.

Exit status: 0 = done (a submission was accepted); 1 = the submission was rejected, the project has
problems, or `run` has no API key; 2 = a wrong command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dossier.agents import brief as brief_mod
from dossier.agents import loop as loop_mod
from dossier.agents import roles as roles_mod
from dossier.agents import submit as submit_mod
from dossier.project import Project, ProjectError


def cmd_roles(a) -> int:
    for r in roles_mod.ROLES.values():
        print(f"{r.name:24s} {r.title} (records of stage `{r.stage}`)")
        for x in r.look_for:
            print(f"    - {x}")
    return 0


def cmd_brief(a) -> int:
    text = brief_mod.build(Project.load(a.project), a.role, focus=a.focus or (), max_chars=a.max_chars)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"{a.out}: the brief of {roles_mod.get(a.role).name}, {len(text)} characters")
    else:
        sys.stdout.write(text)
    return 0


def cmd_schema(a) -> int:
    print(json.dumps(submit_mod.schema(a.role), indent=1))
    return 0


def cmd_submit(a) -> int:
    try:
        raw = sys.stdin.read() if a.file == "-" else Path(a.file).read_text()
    except OSError as e:
        print(f"cannot read the submission: {e}", file=sys.stderr)
        return 1
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"REJECTED: {a.file} is not valid JSON: {e}. Nothing was written.", file=sys.stderr)
        return 1
    project = Project.load(a.project)
    res = (submit_mod.check if a.check else submit_mod.submit)(project, a.role, data, by=a.by or "")
    print(res.text(), file=sys.stdout if res.ok else sys.stderr)
    return 0 if res.ok else 1


def cmd_run(a) -> int:
    try:
        out = loop_mod.run(Project.load(a.project), a.role, mock=a.mock, model=a.model, pictures=not a.no_pictures,
                           log=print)
    except loop_mod.NoKey as e:
        print(str(e), file=sys.stderr)
        return 1
    print(out.text(), file=sys.stdout if out.ok else sys.stderr)
    return 0 if out.ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dossier.agents", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("roles", help="the roles and what each one looks for")
    s.set_defaults(fn=cmd_roles)
    s = sub.add_parser("brief", help="what a role reads: its instructions and the dossier, Markdown")
    s.add_argument("project")
    s.add_argument("role")
    s.add_argument("--out", help="write the brief to this file (default: the standard output)")
    s.add_argument("--focus", nargs="*", metavar="ID", help="load cases or analyses to give in full; the other ones "
                                                            "get one line each")
    s.add_argument("--max-chars", type=int, default=brief_mod.MAX_CHARS,
                   help="the size the brief aims at; a longer dossier is written in the compact form")
    s.set_defaults(fn=cmd_brief)
    s = sub.add_parser("schema", help="the JSON schema of the submission")
    s.add_argument("role")
    s.set_defaults(fn=cmd_schema)
    s = sub.add_parser("submit", help="check a submission and write its records to PROJECT/agent/ROLE.json")
    s.add_argument("project")
    s.add_argument("role")
    s.add_argument("file", help="the submission, a JSON file (`-`: the standard input)")
    s.add_argument("--by", metavar="MODEL_NAME", help="the model that wrote the submission (goes into origin.by)")
    s.add_argument("--check", action="store_true", help="check only, write nothing")
    s.set_defaults(fn=cmd_submit)
    s = sub.add_parser("run", help="brief, one structured call through the Claude API, check, records")
    s.add_argument("project")
    s.add_argument("role")
    s.add_argument("--mock", action="store_true", help="no call: a canned submission that says so in its text")
    s.add_argument("--model", default=loop_mod.DEFAULT_MODEL, help="sonnet | opus (default: %(default)s)")
    s.add_argument("--no-pictures", action="store_true", help="do not send the picture files of the analyses")
    s.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ProjectError as e:
        print("The project has problems:", file=sys.stderr)
        for line in e.problems:
            print("  " + line, file=sys.stderr)
        return 1
    except (KeyError, ValueError) as e:        # an unknown role, an unknown model
        print(e.args[0] if e.args else e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
