"""--deep loop (ace/run_loop.py deep_learn, ace/process_brief.py, Memory.replace_notes). Mock only."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from ace.memory import Memory, notes_as_dicts
from ace.playbook import Bullet, Playbook
from ace.process_brief import PROCESS_BRIEF, deep_context

REPO = Path(__file__).resolve().parents[1]


def test_replace_notes_keeps_edits_adds_and_removes_in_the_given_order():
    pb = Playbook(id_prefix="a1-")
    for text in ("keep me", "edit me", "drop me"):
        pb.bullets.append(Bullet(id=pb.next_id("meshing"), section="meshing", content=text, helpful=2))
    new = [{"id": None, "section": "bcs", "content": "a new note", "kind": "fix", "source": "teacher"},
           {"id": "a1-mesh-0002", "section": "meshing", "content": "edited", "kind": "idea", "source": "self"},
           {"id": "a1-mesh-0001", "section": "meshing", "content": "keep me", "kind": None, "source": None}]
    ch = Memory.replace_notes(pb, new, generation=3)
    assert ch["kept"] == ["a1-mesh-0001"]
    assert ch["edited"] == [("a1-mesh-0002", "edited")]
    assert ch["removed"] == [("a1-mesh-0003", "drop me")]
    assert ch["added"] == [("a1-bc-0001", "a new note")]
    assert [b.id for b in pb.bullets] == ["a1-bc-0001", "a1-mesh-0002", "a1-mesh-0001"]   # organizer's order
    assert pb.get("a1-mesh-0002").helpful == 2 and pb.get("a1-mesh-0002").kind == "idea"  # history kept
    d = notes_as_dicts(pb)[0]
    assert d == {"id": "a1-bc-0001", "section": "bcs", "content": "a new note", "kind": "fix", "source": "teacher", "gen": 3}
    # the pruned id is never handed out again
    assert pb.next_id("meshing") == "a1-mesh-0004"


def test_old_playbook_json_without_kind_or_source_still_loads(tmp_path):
    old = {"bullets": [{"id": "mesh-0001", "section": "meshing", "content": "x", "helpful": 1, "harmful": 0,
                        "source_runs": [], "created_gen": 0, "evidence_type": None}], "counters": {}, "id_prefix": ""}
    (tmp_path / "pb.json").write_text(json.dumps(old))
    pb = Playbook.load(tmp_path / "pb.json")
    assert pb.bullets[0].kind is None and pb.bullets[0].source is None
    pb.save(tmp_path / "pb2.json")
    assert Playbook.load(tmp_path / "pb2.json").bullets[0].content == "x"


def test_deep_context_sections_in_order():
    fb = {"run_id": "g2-cooks_membrane-a1", "diagnosis": "DIAG", "mistakes": ["M"], "strengths": ["S"],
          "next_mesh_call": {"h": 2.0, "order": 2, "refinements": [{"group": "load", "size": 0.5, "radius": 3}]}}
    refl = {"run_id": "g2-cooks_membrane-a1", "plan_next": "PLAN", "next_mesh_call": {"h": 1.0, "order": 1, "refinements": []}}
    hl = [{"agent": "a2", "run_id": "g2-nafems_le1-a2", "what": "W2", "why_it_works": "Y", "how_to_copy": "H"},
          {"agent": "a1", "run_id": "g2-cooks_membrane-a1", "what": "OWNHL", "why_it_works": "", "how_to_copy": ""}]
    text = deep_context("a1", "nafems_le1", "SHARED", "PRIVATE", fb, refl, hl, "\n\n## WHAT OTHER AGENTS DID LAST")
    order = [PROCESS_BRIEF.strip()[:40], "YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN (task cooks_membrane",
             "YOUR OWN REFLECTION", "WHAT OTHER AGENTS DID WELL", "WHAT OTHER AGENTS DID LAST", "SHARED PLAYBOOK",
             "YOUR NOTES", "THIS GENERATION"]
    pos = [text.index(s) for s in order]
    assert pos == sorted(pos)
    assert "{group: load, size: 0.5, radius: 3}" in text and "W2" in text and "OWNHL" not in text   # own highlight hidden
    assert 250 <= len(PROCESS_BRIEF.split()) <= 400


# The mock organizer keeps every note and appends; this one also drops the oldest current note and
# rewrites the next, so the loop test sees notes being replaced, not only added.
RUN = textwrap.dedent("""
    import sys
    from ace import mock
    from ace.run_loop import main
    orig = mock.MockLLM._OrganizerOutput
    def organizer(self, text, schema):
        out = orig(self, text, schema)
        kept = [n for n in out.notes if n.id]
        if len(kept) >= 2:
            out.notes.remove(kept[0])
            kept[1].content = "rewritten: " + kept[1].content
        return out
    mock.MockLLM._OrganizerOutput = organizer
    sys.exit(main(sys.argv[1:]))
""")


def test_deep_loop_mock(tmp_path):
    out = tmp_path / "deep"
    args = ["--mock", "--deep", "--split", "--problems", "cooks_membrane,nafems_le1", "--holdout", "",
            "--memory", "shared+private", "--agents", "6", "--concurrency", "6", "--generations", "3", "--out", str(out)]
    # A subprocess: field scoring is read at import (ACE_SCORE), and --deep sets ACE_SKIP_CONVERGENCE.
    p = subprocess.run([sys.executable, "-c", RUN, *args], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "ACE_SCORE": "field"}, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert p.stdout.count("context: private") == 2                  # gens 0 and 1 learn; gen 2 is evaluated

    for g in range(3):
        gdir = out / f"gen{g}"
        groups = json.loads((gdir / "groups.json").read_text())
        for a, task in groups.items():
            rid = f"g{g}-{task}-{a}"
            rep = json.loads((gdir / "reports" / f"{rid}.json").read_text())
            assert "numbers" in rep and rep["numbers"]["group_size"] == 3
            assert all(Path(x).exists() for x in rep["images"].values())
            ctx = (gdir / "context" / f"{a}.md").read_text()
            assert json.loads((gdir / "runs" / f"{rid}.json").read_text())["playbook_text"] == ctx
            if g < 2:
                assert (gdir / "reflections" / f"{rid}.json").exists()
                assert (gdir / "organizer" / f"{a}.json").exists()
        if g < 2:
            assert all((gdir / f"teacher_{t}.json").exists() for t in ("cooks_membrane", "nafems_le1"))
            assert (gdir / "organizer" / "shared.json").exists()

    for a in [f"a{k}" for k in range(6)]:
        c0, c1, c2 = ((out / f"gen{g}" / "context" / f"{a}.md").read_text() for g in range(3))
        assert c0 != c1 != c2
        assert "(none yet)" in c0 and "YOUR TEACHER'S FEEDBACK ON YOUR LAST RUN (task" in c1
        org = json.loads((out / "gen1" / "organizer" / f"{a}.json").read_text())
        ch = org["changes"]
        assert ch["added"] and ch["edited"] and ch["removed"]          # replaced, not only appended
        gone = ch["removed"][0][0]
        assert f"[{gone}]" in c1 and f"[{gone}]" not in c2

    phases = [json.loads(line) for line in (out / "solver.jsonl").read_text().splitlines()]
    learned = [ph for ph in phases if ph.get("what") == "phase" and "organize" in ph]
    assert len(learned) == 2
    assert all({"agents", "scoring", "report", "teacher", "reflect", "organize"} <= set(ph) for ph in learned)
