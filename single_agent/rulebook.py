"""The rulebook: the general rules every arm starts from (handbook.md) plus the rules it learns.

    handbook   hand-written, general (A1..F3); fixed during a run; --rules none leaves it out (ablation)
    learned    rules the learning step adds, edits, votes on and removes (single_agent.reflect); ids L001...

A learned rule must be general: it may not name a CAD face, a problem, a part or a load case, may not
carry a load magnitude, and is at most config.RULE_MAX_WORDS words (lint below). Near-duplicates of an
existing rule become a helpful vote on that rule instead. Rules voted harmful (harmful >= helpful +
config.PRUNE_MARGIN) are dropped, and at most config.LEARNED_MAX are kept.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from single_agent import config

SECTIONS = ("idealization", "faces", "meshing", "studies", "singularities", "reporting", "workflow")

LINT = [
    (r"(?<![\w-])f\d+(?![\w-])", "names a CAD face"),
    (r"\babc_\d+|\bbc\d\b|\bpart-[0-9a-f]{8}\b", "names a problem"),
    (r"\b(?:TC|SC)-\d+|\bLC-\d+|\bDR-\d+|\bAI-\d+", "names a part, load case or review item"),
    (r"\btrunnion\b|\bsplit clamp\b|\bclamp body\b", "names a specific component"),
    (r"\d[\d.,]*\s*(?:kN|MN|N|MPa|GPa|kPa)\b", "carries a load magnitude"),
    (r"\d(?:\.\d+)?[eE][-+]?\d+|\d[\d.,]*\s*(?:mm|m|J|Pa|µm|um)\b", "carries a specific value of a result or size"),
]


def lint(text: str) -> list[str]:
    issues = [why for pat, why in LINT if re.search(pat, text, re.I)]
    words = len(text.split())
    if words > config.RULE_MAX_WORDS:
        issues.append(f"too long ({words} words > {config.RULE_MAX_WORDS})")
    if words < 5:
        issues.append("too short to be a rule")
    return issues


def _similar(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a, b)
    except ImportError:          # pragma: no cover  (rapidfuzz is in the ace extra)
        import difflib
        return 100 * difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


class Rulebook:
    def __init__(self, handbook: str | None, learned: list[dict] | None = None):
        self.handbook = handbook
        self.learned = list(learned or [])

    # ------------------------------------------------------------------ io
    @classmethod
    def initial(cls, rules: str = "handbook") -> "Rulebook":
        return cls(config.HANDBOOK.read_text() if rules == "handbook" else None)

    @classmethod
    def load(cls, path: str | Path) -> "Rulebook":
        d = json.loads(Path(path).read_text())
        return cls(d.get("handbook"), d.get("learned"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"handbook": self.handbook, "learned": self.learned}, indent=1) + "\n")

    def render(self) -> str:
        """The rulebook as the agent reads it (part of its system prompt)."""
        parts = []
        if self.handbook:
            parts.append(self.handbook.strip())
        if self.learned:
            lines = ["# Learned rules", "", "Rules learned from earlier analyses (cite them by id like the handbook's).", ""]
            for sec in SECTIONS:
                rows = [r for r in self.learned if r["section"] == sec]
                if rows:
                    lines.append(f"## {sec}")
                    lines += [f"- **{r['id']}.** {r['text']}" for r in rows]
                    lines.append("")
            parts.append("\n".join(lines).strip())
        return "\n\n".join(parts) if parts else "(no rules: rely on your own judgement)"

    def ids(self) -> set[str]:
        hb = set(re.findall(r"\*\*([A-F]\d+)\.\*\*", self.handbook or ""))
        return hb | {r["id"] for r in self.learned}

    def _next_id(self) -> str:
        n = max((int(r["id"][1:]) for r in self.learned), default=0)
        return f"L{n + 1:03d}"

    # ------------------------------------------------------------------ updates
    def apply(self, ops: list[dict], generation: int) -> list[dict]:
        """Apply the learning step's operations; returns what happened to each (for the diff)."""
        log = []
        by_id = {r["id"]: r for r in self.learned}
        handbook_rules = re.findall(r"\*\*[A-F]\d+\.\*\*\s*(.+?)(?=\n- \*\*|\n\n|\Z)", self.handbook or "", re.S)
        for op in ops:
            kind, rid, text = op.get("op"), op.get("id"), (op.get("text") or "").strip()
            entry = {"op": kind, "id": rid, "text": text, "reason": op.get("reason", "")}
            if kind == "add":
                issues = lint(text)
                if issues:
                    log.append({**entry, "result": "rejected: " + "; ".join(issues)})
                    continue
                dup = max(((_similar(text, r["text"]), r["id"]) for r in self.learned), default=(0, None))
                if dup[0] >= 85:
                    by_id[dup[1]]["helpful"] += 1
                    log.append({**entry, "result": f"duplicate of {dup[1]}: counted as a helpful vote"})
                    continue
                if any(_similar(text, h) >= 90 for h in handbook_rules):
                    log.append({**entry, "result": "rejected: already in the handbook"})
                    continue
                sec = op.get("section") if op.get("section") in SECTIONS else "workflow"
                r = {"id": self._next_id(), "section": sec, "text": text, "helpful": 0, "harmful": 0,
                     "added": generation, "edited": None}
                self.learned.append(r)
                by_id[r["id"]] = r
                log.append({**entry, "id": r["id"], "result": "added"})
            elif kind == "edit" and rid in by_id:
                issues = lint(text)
                if issues:
                    log.append({**entry, "result": "rejected: " + "; ".join(issues)})
                    continue
                by_id[rid].update(text=text, edited=generation)
                log.append({**entry, "result": "edited"})
            elif kind == "remove" and rid in by_id:
                self.learned.remove(by_id.pop(rid))
                log.append({**entry, "result": "removed"})
            elif kind == "vote" and rid in by_id and op.get("vote") in ("helpful", "harmful"):
                by_id[rid][op["vote"]] += 1
                log.append({**entry, "result": f"voted {op['vote']}"})
            else:
                log.append({**entry, "result": "ignored (unknown op or id; handbook rules are not edited)"})
        for r in [r for r in self.learned if r["harmful"] >= r["helpful"] + config.PRUNE_MARGIN]:
            self.learned.remove(r)
            log.append({"op": "prune", "id": r["id"], "text": r["text"], "result": "pruned (voted harmful)"})
        while len(self.learned) > config.LEARNED_MAX:
            worst = min(self.learned, key=lambda r: (r["helpful"] - r["harmful"], -r["added"]))
            self.learned.remove(worst)
            log.append({"op": "prune", "id": worst["id"], "text": worst["text"], "result": "pruned (over the cap)"})
        return log


def diff_markdown(before: Rulebook, after: Rulebook, log: list[dict], generation: int) -> str:
    lines = [f"# Rulebook changes after generation {generation}", ""]
    for e in log:
        lines.append(f"- **{e['op']}** {e.get('id') or ''}: {e['result']}" + (f" | {e['text']}" if e.get("text") else "")
                     + (f" (why: {e['reason']})" if e.get("reason") else ""))
    lines += ["", "## Learned rules now", ""]
    lines += [f"- {r['id']} [{r['section']}] (+{r['helpful']}/-{r['harmful']}) {r['text']}" for r in after.learned] \
        or ["(none)"]
    return "\n".join(lines) + "\n"
