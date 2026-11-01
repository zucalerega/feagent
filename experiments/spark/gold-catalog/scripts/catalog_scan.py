"""Read-only scan of the gold archives: streams every .tgz once, writes raw per-batch JSON."""
import hashlib, json, re, sys, tarfile, time
from pathlib import Path

DRIVE = Path("/media/dell/jaimesThumbdrive/gold_benchmarks")
OUT = Path.home() / "gold-catalog" / "scan_raw.json"


def scan(path: Path) -> dict:
    rec = {"file": path.name, "bytes": path.stat().st_size, "members": 0, "sets": {}, "meshes": {},
           "logs": [], "summary": {}, "errors": []}
    with open(path, "rb") as raw:
        try:
            with tarfile.open(fileobj=raw, mode="r|gz") as tf:
                for m in tf:
                    rec["members"] += 1
                    if not m.isfile():
                        continue
                    n = m.name
                    mm = re.match(r"(?:\./)?out/(abc_\d+_bc\d+)/(public|private)/(.+)$", n)
                    if mm:
                        sid, side, fn = mm.groups()
                        s = rec["sets"].setdefault(sid, {"files": {}})
                        s["files"][f"{side}/{fn}"] = m.size
                        if fn.endswith(".json"):
                            d = json.load(tf.extractfile(m))
                            b = json.dumps(d, sort_keys=True).encode()
                            s.setdefault("sha", {})[f"{side}/{fn}"] = hashlib.sha256(b).hexdigest()[:16]
                            if fn == "reference.json":
                                s["references"] = [{"qoi": r["qoi"], "value": r["value"], "tol": r["tolerance"]}
                                                   for r in d.get("references", [])]
                                s["part"] = d.get("part", {}).get("id")
                                s["n_faces"] = len(d.get("part", {}).get("faces", []))
                                s["bcs"] = [(b_["type"], b_.get("group")) for b_ in d.get("reference_case", {}).get("bcs", [])]
                            elif fn == "study.json":
                                s["levels"] = sorted({r["level"] for r in d.get("runs", [])})
                                s["dofs"] = {str(r["level"]): r.get("n_dofs") for r in d.get("runs", [])}
                                s["qoi_kinds"] = {k: v.get("kind") for k, v in d.get("qois", {}).items()}
                                s["gold_qois"] = [k for k, v in d.get("qois", {}).items() if v.get("gold")]
                            elif fn == "problem.json":
                                s["stated_qois"] = [q["name"] for q in d.get("qois", [])]
                        elif fn.endswith(".step"):
                            data = tf.extractfile(m).read()
                            un = re.sub(rb"FILE_NAME\('([^']*)','[^']*'", rb"FILE_NAME('\1',''", data, count=1)
                            s["step_sha"] = hashlib.sha256(un).hexdigest()[:16]
                        continue
                    mm = re.match(r"(?:\./)?out/_meshes/(abc_\d+)/L(\d+)\.(msh|json)$", n)
                    if mm:
                        part, lev, ext = mm.groups()
                        e = rec["meshes"].setdefault(part, {}).setdefault(lev, {})
                        if ext == "msh":
                            e["msh_bytes"] = m.size
                        else:
                            d = json.load(tf.extractfile(m))
                            e.update(h=d.get("h"), cells=d.get("cells"), scheme=d.get("scheme"), min_sj=d.get("min_sj"))
                        continue
                    if "logs/" in n:
                        rec["logs"].append([n, m.size])
                        if n.endswith("summary.jsonl"):
                            for line in tf.extractfile(m).read().decode().splitlines():
                                if line.strip():
                                    r = json.loads(line)
                                    rec["summary"][r.get("id")] = {k: r.get(k) for k in ("precheck", "flags", "result", "gold", "singular", "dofs", "error", "reason") if k in r}
                        continue
                    rec["errors"].append(f"unexpected member {n}")
        except Exception as e:  # noqa
            rec["errors"].append(f"read error: {type(e).__name__}: {e}")
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rec


def main():
    files = sorted((DRIVE / "archives").glob("*.tgz"))
    out = []
    t0 = time.time()
    for i, f in enumerate(files):
        out.append(scan(f))
        if i % 20 == 0:
            print(f"{i}/{len(files)} {f.name} {time.time() - t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(out))
    print("wrote", OUT, len(out), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
