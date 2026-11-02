"""Pack finished gold benchmarks into numbered archives while a goldgen run is going, so they can be
copied off the node every few BC sets (nothing exists only on the GPU node).

  python scripts/goldgen_batcher.py ~/gold [--every 10] [--poll 30] [--tag goldgen-l2]

Every --every finished BC sets (out/<id>/private/reference.json written) become
<gold>/batches/<tag>_<nnnn>.tgz holding those benchmark folders, the meshes of their parts that
no earlier batch carried (out/_meshes/<part>/), and the current meshes / summary / solves logs.
batches/manifest.jsonl lists every archive. When no goldgen run is left, the remainder is flushed
as a last (possibly smaller) batch, batches/DONE is written, and the batcher exits.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import tarfile
import time
from pathlib import Path


def run_alive() -> bool:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("goldgen_h100.sh" in ln or "feacore.goldgen run" in ln or "feacore.goldgen screen" in ln
               for ln in out.splitlines() if "goldgen_batcher" not in ln)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--tag", default=socket.gethostname())
    a = ap.parse_args()
    gold = Path(a.gold).expanduser()
    out, bdir = gold / "out", gold / "batches"
    bdir.mkdir(exist_ok=True)
    (bdir / "DONE").unlink(missing_ok=True)
    man = bdir / "manifest.jsonl"
    sent, sent_parts = set(), set()
    if man.exists():
        for r in map(json.loads, man.read_text().splitlines()):
            sent.update(r["bcsets"])
            sent_parts.update(r["parts"])
    n = len(man.read_text().splitlines()) if man.exists() else 0
    while True:
        alive = run_alive()
        done = sorted(p.parent.parent.name for p in out.glob("*/private/reference.json"))
        new = [d for d in done if d not in sent]
        final = not alive                     # the run script is gone: flush the rest and stop
        while len(new) >= a.every or (final and new):
            take, new = new[:a.every], new[a.every:]
            parts = sorted({d.rsplit("_bc", 1)[0] for d in take} - sent_parts)
            name = bdir / f"{a.tag}_{n:04d}.tgz"
            tmp = name.with_suffix(".tmp")
            with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
                for d in take:
                    tf.add(out / d, arcname=f"out/{d}")
                for p in parts:
                    if (out / "_meshes" / p).exists():
                        tf.add(out / "_meshes" / p, arcname=f"out/_meshes/{p}")
                for log in ("meshes.jsonl", "summary.jsonl", "solves.jsonl", "gpu.jsonl"):
                    if (out / log).exists():
                        tf.add(out / log, arcname=f"logs/{a.tag}_{log}")
            tmp.rename(name)
            sent.update(take)
            sent_parts.update(parts)
            with open(man, "a") as f:
                f.write(json.dumps({"batch": name.name, "bcsets": take, "parts": parts,
                                    "bytes": name.stat().st_size, "t": time.time()}) + "\n")
            print(f"{name.name}: {len(take)} BC sets, {len(parts)} new parts' meshes, "
                  f"{name.stat().st_size / 1e6:.1f} MB", flush=True)
            n += 1
        if final:
            (bdir / "DONE").write_text(json.dumps({"batches": n, "bcsets": len(sent)}) + "\n")
            print(f"run finished: {n} batches, {len(sent)} BC sets", flush=True)
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
