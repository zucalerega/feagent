"""Export every registered problem to benchmarks/ (public statement + STEP, private references).

    python scripts/export_benchmarks.py [--tags part] [--out benchmarks]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = ap.parse_args(argv)

    from feacore.problems.registry import all_problems, discover
    from feacore.problems.store import export_all

    for mod, err in discover().items():
        print(f"warning: {mod} failed to import: {err}", file=sys.stderr)
    done, failed = export_all(args.out, problems=all_problems(tags=args.tags))
    print(f"exported {len(done)} problems to {args.out}")
    for pid, err in failed.items():
        print(f"  FAILED {pid}: {err}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
