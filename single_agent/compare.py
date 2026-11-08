"""Arms side by side: score against generation (train and holdout), the score's parts, and cost.

    python -m single_agent.compare sa_runs/sonnet sa_runs/opus [sa_runs/sonnet-norules ...] [--out DIR]

Reads each run directory's config.json and generations.jsonl (single_agent.loop) and writes, to --out
(default: the first run directory): compare.md (one table per split) and compare.png (score against
generation, solid = train, dashed = holdout, one colour per arm). An oracle baseline directory
(single_agent.oracle --out) can be passed with --oracle to draw its mean score as a reference line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PARTS = ("setup", "answers", "calibration", "study")


def load(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(ln) for ln in (run_dir / "generations.jsonl").read_text().splitlines() if ln.strip()]
    name = cfg.get("arm", "?") + ("" if cfg.get("rules") == "handbook" else f"/rules={cfg.get('rules')}") \
        + ("" if cfg.get("learning", True) else "/no-learn") + (" (mock)" if cfg.get("mock") else "")
    if cfg.get("reflect_model") and cfg.get("reflect_model") != cfg.get("model"):
        name += f"/reflector={cfg['reflect_model']}"
    return {"dir": run_dir, "name": name, "cfg": cfg, "rows": rows}


def oracle_mean(oracle_dir: Path, problems: list[str]) -> float | None:
    scores = []
    for p in problems:
        f = oracle_dir / f"{p}.json"
        if f.exists():
            scores.append(json.loads(f.read_text())["scorecard"].get("score", 0.0))
    return sum(scores) / len(scores) if scores else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--oracle", help="an oracle output directory (single_agent.oracle --out)")
    a = ap.parse_args(argv)
    arms = [load(Path(r)) for r in a.runs]
    out = Path(a.out or a.runs[0])
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Single-agent arms compared", ""]
    for arm in arms:
        c = arm["cfg"]
        lines.append(f"- **{arm['name']}**: {c['model']} (reflector {c['reflect_model']}), train {c['train']}, "
                     f"holdout {c['holdout']}, {len(arm['rows'])} generations, {arm['dir']}")
    for split in ("train", "holdout"):
        lines += ["", f"## {split}", "", "| arm | gen | n | score | " + " | ".join(PARTS) +
                  " | setups right | submitted | $ total |", "|---" * (7 + len(PARTS)) + "|"]
        for arm in arms:
            for r in arm["rows"]:
                if split in r:
                    s = r[split]
                    lines.append(f"| {arm['name']} | {r['generation']} | {s['n']} | {s['score']:.3f} | "
                                 + " | ".join(f"{s[k]:.2f}" for k in PARTS)
                                 + f" | {s['setup_equivalent']}/{s['n']} | {s['submitted']}/{s['n']} | {r['usd_total']:.2f} |")
    if a.oracle:
        for split, key in (("train", "train"), ("holdout", "holdout")):
            m = oracle_mean(Path(a.oracle), arms[0]["cfg"][key])
            if m is not None:
                lines.append(f"\nOracle (reference setup, scripted study) mean score on the {split} problems: {m:.3f}")
    (out / "compare.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for i, arm in enumerate(arms):
        col = f"C{i}"
        for split, ls in (("train", "-"), ("holdout", "--")):
            pts = [(r["generation"], r[split]["score"]) for r in arm["rows"] if split in r]
            if pts:
                ax.plot(*zip(*pts), ls, marker="o", color=col, label=f"{arm['name']} {split}")
    if a.oracle:
        m = oracle_mean(Path(a.oracle), arms[0]["cfg"]["train"])
        if m is not None:
            ax.axhline(m, color="0.4", lw=1, ls=":", label="oracle (train problems)")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean composite score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out / "compare.png", dpi=120)
    print(f"wrote {out / 'compare.md'} and {out / 'compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
