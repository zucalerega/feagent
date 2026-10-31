"""Command-line entry point for the fixed solver.

    python -m feacore.cli case.json [--backend skfem] [--out result.json] [--vtu out.vtu]
                                    [--linear-solver auto|scipy|pardiso|cudss] [--quiet]

Loads the case JSON (mesh path relative to the JSON file), runs ``feacore.solver.solve`` with
the chosen backend, writes the result summary (``info`` + ``qois`` + ``meta``) as JSON to
``--out`` (or stdout) and optionally a VTU file for ParaView. Exit code 0 on success, 2 for an
invalid case / mesh, 1 for any other failure (the error is printed to stderr and, with
``--out``, written as ``{"error": ...}``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m feacore.cli", description="Run one FE case with the fixed solver.")
    ap.add_argument("case", help="case JSON file")
    ap.add_argument("--backend", default="skfem", help="solver backend (default: skfem)")
    ap.add_argument("--out", help="write the result summary JSON here (default: stdout)")
    ap.add_argument("--vtu", help="also write the fields to this .vtu file (if the backend supports it)")
    ap.add_argument("--linear-solver", dest="linear_solver",
                    help="override analysis.options['linear_solver'] (auto|scipy|pardiso|cudss)")
    ap.add_argument("--quiet", action="store_true", help="no progress output on stderr")
    args = ap.parse_args(argv)

    from feacore.case import Case, CaseError
    from feacore.solver import solve

    def fail(msg: str, code: int) -> int:
        print(f"error: {msg}", file=sys.stderr)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"error": msg}, indent=2))
        return code

    try:
        case = Case.load(args.case)
    except (OSError, json.JSONDecodeError, CaseError) as e:
        return fail(f"cannot load case {args.case!r}: {e}", 2)
    options = {}
    if args.linear_solver:
        options["linear_solver"] = args.linear_solver
    try:
        res = solve(case, backend=args.backend, **options)
        summary = res.summary()
    except (CaseError, ValueError) as e:
        return fail(f"{type(e).__name__}: {e}", 2)
    except Exception as e:  # noqa: BLE001 - report every failure to the caller
        return fail(f"{type(e).__name__}: {e}", 1)

    text = json.dumps(summary, indent=2, default=float)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    else:
        print(text)
    if args.vtu:
        try:
            res.write_vtu(args.vtu)
        except NotImplementedError:
            print(f"warning: backend {args.backend!r} cannot write VTU", file=sys.stderr)
    if not args.quiet and args.out:
        i = summary["info"]
        print(f"{i['backend']} {i['physics']}/{i['formulation']} P{i['order']} {i['cell_type']}: "
              f"{i['n_dofs']} dofs, {i['wall_time']:.2f}s -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
