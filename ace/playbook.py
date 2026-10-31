"""The playbook: a JSON list of short, voted-on bullets that gets pasted into the agent's prompt.

Each bullet has a stable ID (e.g. "mesh-0003") so the agent can cite it and the Reflector can
vote on it. The playbook is only ever changed by ace.merge (deterministic code), never by an LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ace.config import SECTIONS, TOP_N_BULLETS


@dataclass
class Bullet:
    id: str                     # "<prefix>-<4 digits>", prefix from config.SECTIONS
    section: str
    content: str                # "When X, do Y, because Z."
    helpful: int = 0
    harmful: int = 0
    source_runs: list[str] = field(default_factory=list)
    created_gen: int = 0
    # How the lesson was established: "counterfactual" (a rerun confirmed it), "contrast"
    # (best-vs-worst comparison), or None (older single-call Reflector lessons).
    evidence_type: str | None = None
    # --deep only (ace/organizer.py): what kind of note this is ("fix" | "idea" | "keep" | "avoid" |
    # "highlight") and who wrote it ("teacher", "self", "peer:a2", ...). Older files load with None.
    kind: str | None = None
    source: str | None = None

    @property
    def net(self) -> int:
        return self.helpful - self.harmful


@dataclass
class Playbook:
    bullets: list[Bullet] = field(default_factory=list)
    # Last number handed out per ID prefix, so an ID is never reused after its bullet is pruned.
    counters: dict[str, int] = field(default_factory=dict)
    id_prefix: str = ""          # private playbooks: "a3-" gives IDs like "a3-mesh-0001"

    def __post_init__(self):
        # Reserve the numbers of bullets that arrived without next_id (older files, tests).
        for b in self.bullets:
            prefix, n = b.id.rsplit("-", 1)
            self.counters[prefix] = max(self.counters.get(prefix, 0), int(n))

    # ---- persistence
    @classmethod
    def load(cls, path: Path) -> "Playbook":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls([Bullet(**b) for b in data["bullets"]], data.get("counters", {}), data.get("id_prefix", ""))

    def save(self, path: Path) -> None:
        data = {"bullets": [asdict(b) for b in self.bullets], "counters": self.counters, "id_prefix": self.id_prefix}
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    # ---- lookup
    def get(self, bullet_id: str) -> Bullet | None:
        return next((b for b in self.bullets if b.id == bullet_id), None)

    def in_section(self, section: str) -> list[Bullet]:
        return [b for b in self.bullets if b.section == section]

    def next_id(self, section: str) -> str:
        """Hand out a fresh ID for the section (this reserves it)."""
        prefix = self.id_prefix + SECTIONS[section]
        used = [int(b.id.rsplit("-", 1)[1]) for b in self.bullets if b.id.startswith(prefix + "-")]
        n = max(self.counters.get(prefix, 0), max(used, default=0)) + 1
        self.counters[prefix] = n
        return f"{prefix}-{n:04d}"

    # ---- prompt text
    def render(self, top_n: int = TOP_N_BULLETS) -> str:
        """Top-N bullets, grouped by section, each prefixed by its ID: counterfactual-backed bullets
        first, then by (helpful - harmful); ties keep the older bullet first. Empty -> ""."""
        top = sorted(self.bullets, key=lambda b: (b.evidence_type != "counterfactual", -b.net))[:top_n]
        lines = []
        for section in SECTIONS:
            chosen = [b for b in top if b.section == section]   # keeps the order above within a section
            if chosen:
                lines.append(f"## {section}")
                lines += [f"[{b.id}] {b.content}" for b in chosen]
        return "\n".join(lines)
