"""Command line interface: ``python -m feabench <command> ...`` (or ``feabench ...``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_calculix_verify(a):
    from .suites import calculix_tests as ct

    rep = ct.verify(a.test_dir, a.work, a.out, workers=a.workers, timeout=a.timeout, only=a.only or None)
    print(json.dumps(rep["counts"], indent=1))


def _cmd_calculix_fetch(a):
    from .suites import calculix_tests as ct

    print(ct.fetch(a.dest))


def _cmd_aster_manifest(a):
    from .suites import code_aster as ca

    man = ca.build_manifest(a.astest, families=a.families, mesh=not a.no_mesh)
    Path(a.out).write_text(json.dumps(man, indent=1))
    print(json.dumps(man["summary"], indent=1))


def _cmd_aster_fetch(a):
    from .suites import code_aster as ca

    print(ca.fetch(a.dest, families=a.families))


def _cmd_list(a):
    from .cases import list_cases

    for c in list_cases(a.suite):
        print(c)


def _cmd_export_task(a):
    from .cases import export_task

    print(export_task(a.case, a.dest))


def _cmd_evaluate(a):
    from .evaluate import evaluate_submission

    rep = evaluate_submission(a.case, a.submission, out=a.out, timeout=a.timeout, threads=a.threads,
                              agent_cost=json.loads(a.agent_cost) if a.agent_cost else None)
    print(json.dumps(rep["summary"], indent=1))


def _cmd_reference(a):
    from .suites.verification import run_reference

    rep = run_reference(a.case, a.level, a.out, threads=a.threads)
    print(json.dumps(rep["summary"], indent=1))


def _cmd_convergence(a):
    from .suites.verification import convergence_study

    rep = convergence_study(a.case, levels=a.levels, workdir=a.work, threads=a.threads, write=not a.dry)
    print(json.dumps(rep["summary"], indent=1))


def _cmd_build_verification(a):
    from .suites.verification import build_all

    for p in build_all(only=a.only or None):
        print(p)


def _cmd_simjeb(a):
    from .suites import simjeb

    if a.action == "fetch":
        simjeb.fetch(a.dest, include=a.include or None, dry_run=a.dry_run)
    elif a.action == "build-tasks":
        for p in simjeb.build_tasks(a.dest, a.out, ids=a.ids or None, load_cases=a.load_cases or None):
            print(p)


def main(argv=None):
    p = argparse.ArgumentParser(prog="feabench")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("calculix-fetch", help="sparse-clone the pinned upstream ccx test suite")
    s.add_argument("dest")
    s.set_defaults(fn=_cmd_calculix_fetch)

    s = sub.add_parser("calculix-verify", help="run upstream ccx tests with the installed ccx")
    s.add_argument("test_dir")
    s.add_argument("--work", default="/tmp/feabench-ccx-tests")
    s.add_argument("--out", default="calculix_verification.json")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--timeout", type=float, default=900)
    s.add_argument("--only", nargs="*")
    s.set_defaults(fn=_cmd_calculix_verify)

    s = sub.add_parser("aster-fetch", help="sparse-clone code_aster astest files for given families")
    s.add_argument("dest")
    s.add_argument("--families", nargs="*", default=["sslv"])
    s.set_defaults(fn=_cmd_aster_fetch)

    s = sub.add_parser("aster-manifest", help="extract TEST_RESU references from code_aster tests")
    s.add_argument("astest")
    s.add_argument("--families", nargs="*", default=["sslv"])
    s.add_argument("--out", default="code_aster_manifest.json")
    s.add_argument("--no-mesh", action="store_true", help="do not resolve group coordinates from meshes")
    s.set_defaults(fn=_cmd_aster_manifest)

    s = sub.add_parser("list", help="list agent-facing benchmark cases")
    s.add_argument("--suite", default=None)
    s.set_defaults(fn=_cmd_list)

    s = sub.add_parser("export-task", help="copy only the agent-visible files of a case")
    s.add_argument("case")
    s.add_argument("dest")
    s.set_defaults(fn=_cmd_export_task)

    s = sub.add_parser("evaluate", help="run the fixed solver on a submission and score it")
    s.add_argument("case")
    s.add_argument("submission", help="directory containing model.inp")
    s.add_argument("--out", default=None)
    s.add_argument("--timeout", type=float, default=900)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--agent-cost", default=None,
                   help='JSON with agent-side cost, e.g. {"input_tokens":1e5,"output_tokens":2e4,"wall_s":300}')
    s.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("reference", help="run the scripted reference pipeline for a verification case")
    s.add_argument("case")
    s.add_argument("--level", type=int, default=2)
    s.add_argument("--out", default="/tmp/feabench-ref")
    s.add_argument("--threads", type=int, default=1)
    s.set_defaults(fn=_cmd_reference)

    s = sub.add_parser("convergence", help="mesh-convergence study of the reference pipeline")
    s.add_argument("case")
    s.add_argument("--levels", type=int, nargs="*", default=None)
    s.add_argument("--work", default="/tmp/feabench-conv")
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--dry", action="store_true", help="do not write convergence.json into the case")
    s.set_defaults(fn=_cmd_convergence)

    s = sub.add_parser("build-verification", help="(re)generate verification case files")
    s.add_argument("--only", nargs="*")
    s.set_defaults(fn=_cmd_build_verification)

    s = sub.add_parser("simjeb", help="SimJEB dataset: fetch from Harvard Dataverse / build tasks")
    s.add_argument("action", choices=["fetch", "build-tasks"])
    s.add_argument("--dest", default="data/simjeb")
    s.add_argument("--out", default="data/simjeb/cases")
    s.add_argument("--include", nargs="*", help="glob patterns of dataset files to download")
    s.add_argument("--ids", nargs="*", type=int)
    s.add_argument("--load-cases", nargs="*")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=_cmd_simjeb)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
