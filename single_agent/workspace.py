"""Design packages and workspaces: what an analyst agent is given for one problem, and nothing more.

A design package lives in single_agent/design/<problem_id>/:
  case.md    the agent-facing text of the load case: a title line, a `<!-- part: NAME -->` marker, then the
             system context, the design review notes and the analysis request (questions Q1, Q2, ...)
  key.json   evaluator-only: which gold reference (or singular quantity) each question asks for, and which
             phrase of the design data implies each item of the reference setup
The part's own description (the component section) is shared by all load cases of a part:
single_agent/design/_parts/NAME.md. package_text() puts them together.

    make(problem_id, root) -> <root>/<label>/ with geometry.step (the CAD) and package.md (the package).
The label is neutral (single_agent.part.label_for); the benchmark id, the reference setup and the
reference values stay outside the workspace, so a workspace can be handed to any agent.

    python -m single_agent.workspace check          # lint every package, check every key against the gold data
    python -m single_agent.workspace show abc_00000065_bc0
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from single_agent import config
from single_agent.part import label_for

PART_MARKER = re.compile(r"^<!--\s*part:\s*([\w-]+)\s*-->\s*$", re.M)

# What the agent-facing text must not contain: solver input instead of the physical situation, or a way to
# look the benchmark up. Device words (a clamp, a pin, a bearing block) are fine; the agent must still decide
# how to model them.
FORBIDDEN = [
    (r"(?<![\w-])f\d+(?![\w-])", "names a CAD face tag"),
    (r"\bboundary[- ]conditions?\b|\bBCs?\b", "names boundary conditions"),
    (r"\bfixed\b|\bencastr[eé]\b|\brollers?\b|\bpinned\b|\bclamped support\b", "uses a support-type word"),
    (r"\b(?:zero|null|prescribed)\s+(?:[\w-]+\s+){0,2}displacements?\b", "prescribes a displacement"),
    (r"\bconstrain(?:ed|t|ts)?\b|\bdegrees? of freedom\b|\bDOFs?\b", "uses solver vocabulary"),
    (r"\btractions?\b|\bsymmetry (?:plane|condition|boundary)\b", "uses solver load/support vocabulary"),
    (r"\babc[_ ]?\d+|\bgold\b|\bbenchmark\b|\bABC dataset\b", "names the benchmark"),
]


def case_path(problem_id: str) -> Path:
    return config.DESIGN_DIR / problem_id / "case.md"


def key_path(problem_id: str) -> Path:
    return config.DESIGN_DIR / problem_id / "key.json"


def packaged() -> list[str]:
    """Problems with a design package and an evaluator key."""
    return sorted(p.parent.name for p in config.DESIGN_DIR.glob("*/case.md") if (p.parent / "key.json").exists())


def load_key(problem_id: str) -> dict:
    return json.loads(key_path(problem_id).read_text())


def package_text(problem_id: str) -> str:
    """The design package as the agent reads it: title, the part's description, then the load case."""
    case = case_path(problem_id).read_text()
    m = PART_MARKER.search(case)
    if not m:
        raise ValueError(f"{case_path(problem_id)} has no <!-- part: NAME --> marker")
    part = (config.DESIGN_DIR / "_parts" / f"{m.group(1)}.md").read_text().strip()
    title, rest = case[: m.start()].strip(), case[m.end():].strip()
    return f"{title}\n\n{part}\n\n{rest}\n"


def questions(problem_id: str) -> list[str]:
    """The question ids (Q1, Q2, ...) the package asks, in order."""
    return re.findall(r"\*\*(Q\d+)\.\*\*", package_text(problem_id))


def step_path(problem_id: str) -> Path:
    from feacore.problems import registry
    p = registry.get(problem_id)
    if not hasattr(p, "step_path"):
        raise ValueError(f"{problem_id} has no STEP file (only STEP-based gold parts can be packaged)")
    return Path(p.step_path)


def make(problem_id: str, root: str | Path, fresh: bool = True) -> Path:
    """Create the workspace of `problem_id` under `root`; returns its directory."""
    ws = Path(root) / label_for(problem_id)
    if fresh and ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(step_path(problem_id), ws / "geometry.step")
    (ws / "package.md").write_text(package_text(problem_id))
    return ws


def problem_for_label(label: str) -> str:
    for pid in packaged():
        if label_for(pid) == label:
            return pid
    raise KeyError(f"no packaged problem has label {label!r}")


# --------------------------------------------------------------------------- checks


def lint(text: str) -> list[str]:
    issues = []
    for pat, why in FORBIDDEN:
        for m in re.finditer(pat, text, re.I):
            line = text[: m.start()].count("\n") + 1
            issues.append(f"line {line}: {m.group(0)!r} {why}")
    return issues


def check(problem_id: str) -> list[str]:
    """Lint the agent-facing text and check the key against the gold problem: every question is asked in the
    package and answered by the key, every keyed QoI is a reference (expect converge) or a singular QoI
    (expect diverge) of the problem, every face of the reference setup appears in the key's idealization."""
    from feacore.problems import registry
    issues = [f"lint: {i}" for i in lint(package_text(problem_id))]
    key = load_key(problem_id)
    p = registry.get(problem_id)
    refs = {r.qoi for r in p.references()}
    singular = set(p.singular_qois()) if hasattr(p, "singular_qois") else set()
    asked = questions(problem_id)
    keyed = [q["id"] for q in key["questions"]]
    if asked != keyed:
        issues.append(f"questions asked {asked} != keyed {keyed}")
    for q in key["questions"]:
        if q["expect"] == "converge" and q["qoi"] not in refs:
            issues.append(f"{q['id']}: {q['qoi']} is not a reference QoI ({sorted(refs)})")
        if q["expect"] == "diverge" and q["qoi"] not in singular:
            issues.append(f"{q['id']}: {q['qoi']} is not a singular QoI ({sorted(singular)})")
    ref_groups = {b.group for b in p.case("X.msh").bcs if b.group}
    mapped = " ".join(m["reference"] for m in key.get("idealization", []))
    for g in sorted(ref_groups):
        if not re.search(rf"\b{g}\b", mapped):
            issues.append(f"reference group {g} has no idealization entry in the key")
    return issues


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["show"] and len(argv) == 2:
        print(package_text(argv[1]))
        return 0
    if argv[:1] == ["check"]:
        bad = 0
        for pid in argv[1:] or packaged():
            issues = check(pid)
            bad += bool(issues)
            print(f"{pid}: {'ok' if not issues else ''}")
            for i in issues:
                print(f"   {i}")
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
