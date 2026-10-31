"""Merge: apply the Curator's ops to the playbook. Plain deterministic Python, no LLM.

Order: VOTEs, then ADDs (each checked for near-duplicates with rapidfuzz; a near-duplicate
becomes a helpful vote on the existing bullet instead), then prune bullets where
harmful >= helpful + PRUNE_MARGIN. playbook_gen{N}.json is the playbook generation N's agents
used; diff_gen{N}.md says what changed from generation N-1 to N, in plain words.
"""

from __future__ import annotations

from pathlib import Path

from rapidfuzz import fuzz

from ace.config import DEDUP_THRESHOLD, PRUNE_MARGIN
from ace.playbook import Bullet, Playbook


def find_duplicate(playbook: Playbook, section: str, content: str) -> tuple[Bullet | None, float]:
    """Most similar bullet in the section, if its similarity is at/above the threshold."""
    best, best_score = None, 0.0
    for b in playbook.in_section(section):
        s = fuzz.token_set_ratio(content.lower(), b.content.lower())
        if s > best_score:
            best, best_score = b, s
    return (best, best_score) if best_score >= DEDUP_THRESHOLD else (None, best_score)


def apply_ops(playbook: Playbook, ops: list, generation: int) -> dict:
    """Mutates `playbook`. Returns a change log for the diff."""
    changes = {"added": [], "voted": [], "deduped": [], "pruned": []}

    for op in ops:
        if op.kind == "VOTE" and (b := playbook.get(op.bullet_id)):
            setattr(b, op.vote, getattr(b, op.vote) + 1)
            changes["voted"].append((b.id, op.vote))

    for op in ops:
        if op.kind != "ADD":
            continue
        dup, score = find_duplicate(playbook, op.section, op.content)
        if dup:
            dup.helpful += 1
            dup.source_runs = sorted(set(dup.source_runs) | set(op.source_runs))
            changes["deduped"].append((op.content, dup.id, round(score)))
            continue
        b = Bullet(id=playbook.next_id(op.section), section=op.section, content=op.content,
                   source_runs=list(op.source_runs), created_gen=generation)
        playbook.bullets.append(b)
        changes["added"].append((b.id, b.content))

    keep = []
    for b in playbook.bullets:
        if b.harmful >= b.helpful + PRUNE_MARGIN:
            changes["pruned"].append((b.id, b.content, b.helpful, b.harmful))
        else:
            keep.append(b)
    playbook.bullets = keep
    return changes


def diff_markdown(changes: dict, generation: int, playbook: Playbook) -> str:
    lines = [f"# Playbook for generation {generation} (learned from generation {generation - 1})", ""]
    lines += ["## Added"] + [f"- `{i}` {c}" for i, c in changes["added"]]
    lines += ["", "## Votes"] + [f"- `{i}` +1 {v}" for i, v in changes["voted"]]
    lines += ["", "## Near-duplicates (counted as a helpful vote instead of a new bullet)"]
    lines += [f"- matched `{i}` (similarity {s}): {c}" for c, i, s in changes["deduped"]]
    lines += ["", "## Pruned"] + [f"- `{i}` (helpful {h}, harmful {x}) {c}" for i, c, h, x in changes["pruned"]]
    lines += ["", f"## Playbook now ({len(playbook.bullets)} bullets)", "", playbook.render() or "(empty)", ""]
    return "\n".join(lines)


def save_generation(playbook: Playbook, changes: dict, outdir: Path, generation: int) -> None:
    outdir = Path(outdir)
    playbook.save(outdir / f"playbook_gen{generation}.json")
    (outdir / f"diff_gen{generation}.md").write_text(diff_markdown(changes, generation, playbook))
