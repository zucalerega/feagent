"""Gold-mesh actions on this machine (CPU), for the dashboard's Gold page and the command line.

    python -m feacore.goldgen.local regenerate PART [--levels 0,1] [--threads 1] [--install]
                                                    [--source ORIGINAL.step] [--h0-frac F] [--out DIR]
    python -m feacore.goldgen.local import ARCHIVE [--no-meshes]
    python -m feacore.goldgen.local nonlinear SET[,SET...] --out DIR [--levels 0,1,2] [--threads 4]
                                                    [--increments 10] [--meshes stored|regenerated]

regenerate  meshes a gold part's nested study levels again with the generator's settings: the base size
            of the stored L0 (h0 / bounding-box diagonal, 0.237 for the imported sets), nested splits,
            the curving scheme each stored level used (feacore.goldgen.mesh.mesh_level), gmsh's HXT
            mesher on one thread. The input is the set's public geometry.step at scale 1, or with
            --source the part's original STEP at the part's scale, which is what the generator meshed.
            Each level goes to <run store>/gold_regen/<part>/L<k>.msh with a sidecar, and regen.json
            records its sha256 against index.json. --install copies a level into the gold root's
            _meshes/ only when the sha256 matches (then it is the stored mesh, byte for byte).
import      scripts/gold.py import, with "@runs/..." paths resolved in the run store.
nonlinear   scripts/gold_nonlinear.py on the gold root's layout (<set>/, _meshes/<part>/): a view
            directory of links makes it look like the generator's out/ folder. --meshes regenerated
            uses the meshes of `regenerate` (not the certified ones: the linear check against the
            stored study then differs by the discretisation error).

Locally regenerated meshes need not match the stored sha256: gmsh and OpenCASCADE versions, and the
round trip through the rescaled geometry.step, move nodes. gmsh is not even bitwise reproducible from
run to run (two regenerations here differ in the last digit of some coordinates), so regen.json also
records the cell counts next to the sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DEFAULT_H0_FRAC = 0.237           # scripts/goldgen_h100.sh --h0-frac


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gold_root(root: str | None) -> Path:
    from feacore.problems.gold import gold_root

    return Path(root) if root else gold_root()


def _index(root: Path) -> dict:
    p = root / "index.json"
    index = json.loads(p.read_text()) if p.is_file() else {}
    index.setdefault("sets", {})
    index.setdefault("parts", {})
    return index


def resolve(value: str) -> Path:
    """A path argument as the dashboard passes it: "@runs/x" = inside the run store, else as given
    (relative to the repository root)."""
    if value.startswith("@runs/"):
        from feacore.runstore import default_root

        return default_root() / value[len("@runs/"):]
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def _levels(text: str) -> list[int]:
    try:
        ks = sorted({int(x) for x in re.split(r"[\s,]+", text.strip()) if x != ""})
    except ValueError:
        raise SystemExit(f"levels must be integers like 0,1,2, not {text!r}") from None
    if not ks or ks[0] < 0 or ks[-1] > 4:
        raise SystemExit("levels must be between 0 and 4")
    return ks


def part_sets(root: Path, part: str) -> list[Path]:
    """The gold set directories of a part (statement family = part)."""
    out = []
    for pj in sorted(root.glob("*/public/problem.json")):
        try:
            if json.loads(pj.read_text()).get("family") == part:
                out.append(pj.parent.parent)
        except (OSError, json.JSONDecodeError):
            continue
    return out


# --------------------------------------------------------------------------- regenerate


def regen_dir(part: str) -> Path:
    from feacore.runstore import default_root

    return default_root() / "gold_regen" / part


def stored_h0_frac(stored: dict, diag: float) -> float:
    """The generator's base size / diagonal, from a stored level's h (h0 / 2^k). The generator's own
    argument was a short decimal (0.237) unless --max-cells enlarged it: when 4 digits give back the
    stored size, that decimal is used, so h0 = diag x h0_frac is the same float as the generator's."""
    base = next(((int(k[1:]), v["h"]) for k, v in sorted(stored.items()) if v.get("h")), None)
    if base is None:
        return DEFAULT_H0_FRAC
    h0 = base[1] * 2 ** base[0]
    short = float(f"{h0 / diag:.4g}")
    return short if abs(short * diag - h0) <= 1e-9 * h0 else h0 / diag


def regenerate(part: str, levels: list[int], root: Path, out: Path | None = None, threads: int = 1,
               source: Path | None = None, h0_frac: float | None = None, install: bool = False,
               log=print) -> dict:
    """Mesh ``levels`` of ``part`` again (see the module doc); returns the regen.json record."""
    from feacore.goldgen.mesh import STRATEGIES, MeshFailed, mesh_level_safe
    from feacore.goldgen.parts import Part

    if not ID_RE.match(part):
        raise SystemExit(f"invalid part id {part!r}")
    sets = part_sets(root, part)
    if not sets:
        raise SystemExit(f"no gold set of part {part} under {root}")
    stored = _index(root)["parts"].get(part, {}).get("meshes", {})
    ref = json.loads((sets[0] / "private" / "reference.json").read_text())
    pd = dict(ref["part"])
    if source is None:
        step, pd["scale"], how = sets[0] / "public" / "geometry.step", 1.0, "public geometry.step (scale 1)"
    else:
        step, how = Path(source), f"original STEP {Path(source).name} (scale {pd['scale']:.6g})"
        if not step.is_file():
            raise SystemExit(f"no file {step}")
    part_obj = Part.from_json(json.dumps(pd))
    if h0_frac is None:
        h0_frac = stored_h0_frac(stored, part_obj.diag)
    out = Path(out) if out is not None else regen_dir(part)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "regen.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.update(part=part, root=str(root), step=str(step), input=how, h0_frac=h0_frac, levels=summary.get("levels", {}))
    log(f"{part}: {how}, base size {h0_frac:.6g} x diagonal = {h0_frac * part_obj.diag * 1000:.3f} mm, "
        f"levels {', '.join(f'L{k}' for k in levels)} -> {out}")
    for k in levels:
        lv = f"L{k}"
        want = stored.get(lv, {})
        schemes = [want["scheme"]] if want.get("scheme") in (*STRATEGIES, "straight") else []
        schemes += [s for s in (*STRATEGIES, "straight") if s not in schemes]
        meta, tried = None, []
        for scheme in schemes:
            try:
                meta = mesh_level_safe(step, part_obj, k, out / f"{lv}.msh", threads, h0_frac, scheme,
                                       timeout=float(os.environ.get("GOLDGEN_ATTEMPT_TIMEOUT", 1800)))
                break
            except MeshFailed as e:
                tried.append(str(e)[:200])
                log(f"  {lv}: {scheme} failed: {str(e)[:160]}")
        if meta is None:
            summary["levels"][lv] = {"error": "; ".join(tried), "created_at": _now()}
            continue
        path = out / f"{lv}.msh"
        sha = _sha256(path)
        rec = {k_: meta.get(k_) for k_ in ("h", "cells", "scheme", "min_sj", "seconds", "nested", "alg3d")}
        rec.update(file=path.name, bytes=path.stat().st_size, sha256=sha, stored_sha256=want.get("sha256"),
                   stored_cells=want.get("cells"), stored_scheme=want.get("scheme"),
                   match=bool(want.get("sha256")) and sha == want.get("sha256"),
                   cells_match=want.get("cells") is not None and meta.get("cells") == want.get("cells"),
                   input=how, threads=threads, created_at=_now(), installed=False)
        if want.get("sha256") is None:
            verdict = "no stored sha256 to compare with"
        elif rec["match"]:
            verdict = "sha256 MATCHES the stored mesh"
        else:
            verdict = (f"sha256 differs from the stored mesh (stored {want.get('cells')} cells, "
                       f"{want.get('scheme')}): not byte-identical")
        if install and rec["match"]:
            dst = root / "_meshes" / part / path.name
            if not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dst)
                (dst.with_suffix(".json")).write_text(json.dumps({**meta, "mesh_level": k, "part": part,
                                                                  "regenerated": True}))
            rec["installed"] = True
            verdict += f"; installed as {dst}"
        elif install:
            verdict += "; not installed (only a byte-identical mesh is)"
        (out / f"{lv}.json").write_text(json.dumps({**meta, **rec}, indent=1))
        summary["levels"][lv] = rec
        log(f"  {lv}: {rec['cells']} cells, {rec['scheme']}, {rec['seconds']:.1f} s, sha256 {sha[:12]}: {verdict}")
    summary["updated_at"] = _now()
    summary_path.write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------- import / nonlinear


def run_script(args: list[str]) -> int:
    """A repo script with this interpreter, its output passed through (the task's log)."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])])}
    print(f"$ python {' '.join(args)}", flush=True)
    return subprocess.call([sys.executable, *args], cwd=REPO_ROOT, env=env)


def nonlinear_view(root: Path, sets: list[str], out: Path, meshes: str, levels: list[int]) -> Path:
    """<out>/gold_view/out/<set> and .../_meshes/<part>: links into the gold root (or the regenerated
    meshes), the layout scripts/gold_nonlinear.py reads. Only links are written; the gold root is read."""
    view = out / "gold_view"
    if view.exists():
        shutil.rmtree(view)
    (view / "out" / "_meshes").mkdir(parents=True)
    missing = []
    for sid in sets:
        if not ID_RE.match(sid) or not (root / sid / "private" / "reference.json").is_file():
            raise SystemExit(f"no gold set {sid!r} under {root}")
        (view / "out" / sid).symlink_to((root / sid).resolve(), target_is_directory=True)
        part = json.loads((root / sid / "public" / "problem.json").read_text())["family"]
        src = (root / "_meshes" / part) if meshes == "stored" else regen_dir(part)
        link = view / "out" / "_meshes" / part
        if not link.exists():
            link.symlink_to(src.resolve(), target_is_directory=True)
        missing += [f"{part}/L{k}.msh" for k in levels if not (src / f"L{k}.msh").is_file()]
    if missing:
        hint = ("import the archive with its meshes (Gold page, Import)" if meshes == "stored"
                else "regenerate them first (Gold page, Regenerate)")
        raise SystemExit(f"missing {meshes} meshes: {', '.join(sorted(set(missing)))}; {hint}")
    return view


def _now() -> str:
    from feacore.runstore import now_iso

    return now_iso()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m feacore.goldgen.local", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", help="gold root (default FEACORE_GOLD_DIR or benchmarks/gold)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("regenerate", help="mesh a part's study levels again and compare the sha256")
    s.add_argument("part")
    s.add_argument("--levels", default="0")
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--source", help="the part's original STEP (default: the public geometry.step, scale 1)")
    s.add_argument("--h0-frac", type=float, default=None, help="default: the stored L0 size / diagonal")
    s.add_argument("--install", action="store_true", help="install byte-identical levels into _meshes/")
    s.add_argument("--out", help="default: <run store>/gold_regen/<part>")
    s = sub.add_parser("import", help="scripts/gold.py import with @runs/ paths")
    s.add_argument("archive")
    s.add_argument("--no-meshes", action="store_true")
    s = sub.add_parser("nonlinear", help="scripts/gold_nonlinear.py on the gold root")
    s.add_argument("sets", help="comma-separated linear gold set ids")
    s.add_argument("--out", required=True)
    s.add_argument("--levels", default="0,1,2")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--increments", type=int, default=10)
    s.add_argument("--meshes", choices=("stored", "regenerated"), default="stored")
    a = ap.parse_args(argv)
    root = _gold_root(a.root)
    if a.cmd == "regenerate":
        t0 = time.perf_counter()
        summary = regenerate(a.part, _levels(a.levels), root, resolve(a.out) if a.out else None, a.threads,
                             resolve(a.source) if a.source else None, a.h0_frac, a.install,
                             log=lambda m: print(m, flush=True))
        failed = [k for k, v in summary["levels"].items() if v.get("error")]
        print(f"done in {time.perf_counter() - t0:.0f} s" + (f"; failed: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0
    if a.cmd == "import":
        src = resolve(a.archive)
        if not src.exists():
            raise SystemExit(f"no archive {src}")
        return run_script(["scripts/gold.py", *(["--root", str(root)] if a.root else []), "import", str(src),
                           *(["--no-meshes"] if a.no_meshes else [])])
    sets = [x for x in re.split(r"[\s,]+", a.sets) if x]
    levels = _levels(a.levels)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    view = nonlinear_view(root, sets, out, a.meshes, levels)
    return run_script(["scripts/gold_nonlinear.py", str(view), *sets, "--out", str(out), "--levels",
                       *map(str, levels), "--threads", str(a.threads), "--increments", str(a.increments)])


if __name__ == "__main__":
    sys.exit(main())
