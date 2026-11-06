"""Claude Code Stop hook: do not let the agent finish while a mesh spec of this session is not passing.

Reads the state file written by ``feagent mesh/check/waive`` (``check.state_path``). A spec blocks
when its entry is not "pass" ("fail", "unchecked") or is stale (the spec's mtime differs from the
recorded ``spec_mtime``), unless it has a valid waiver (``waiver_spec_mtime`` equals the spec's
current mtime). Specs whose file no longer exists are ignored. Blocking = reason on stderr and exit
code 2 (Claude Code feeds stderr back to the agent).

Scope: the hook reads the Stop payload (JSON on stdin: ``transcript_path``, ``stop_hook_active``).
The session starts at the first timestamped entry of the transcript; only entries recorded
(``checked_at``) since then can block, so specs from earlier sessions never do. Without a payload
or a readable transcript every entry counts. (The transcript's ctime is no use: every append
updates it.)

Escape: after ``MAX_BLOCKS`` consecutive blocks for the same blocking set -- the exact
(spec, spec mtime, status) triples -- the hook lets the agent stop (with a note) and keeps allowing
that same set; it re-arms only when the set changes (a spec is edited, re-checked with another
status, or a new spec starts failing). A payload with ``stop_hook_active: false`` is a fresh stop
attempt, so the count restarts. State: ``"hook_blocks": {"count", "blocking", "allowed"}`` in the
state file, re-read right before each (atomic) write. A malformed state file never blocks.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .check import load_state, spec_mtime, write_state

MAX_BLOCKS = 3
_KEY = "hook_blocks"
_SCAN_LINES = 200  # transcript lines searched for the first timestamp
_ADVICE = ("Fix each spec and re-run `feagent check SPEC` (MCP: check_mesh) until it passes. If "
           "finishing as is is acceptable, run `feagent waive SPEC --reason \"...\"` and tell the "
           "user which gates fail and why.")

Fingerprint = list[list]  # sorted [spec path, current spec mtime, status] of the blocking specs


def main(stdin_text: str | None = None) -> int:
    """Run the hook on the Claude Code Stop payload (read from stdin if None); return exit code."""
    payload = _payload(sys.stdin.read() if stdin_text is None else stdin_text)
    data = load_state()
    problem = malformed(data)
    if problem:
        sys.stderr.write(f"feagent: state file is malformed ({problem}); allowing stop\n")
        return 0
    specs = in_session(data["specs"], session_start(payload))
    reasons = {path: why for path, entry in sorted(specs.items())
               if (why := blocking_reason(path, entry)) is not None}
    verdict, count = _decide(_fingerprint(reasons, specs), payload.get("stop_hook_active"))
    if verdict == "block":
        sys.stderr.write(_message(reasons) + f"\n(feagent stop check {count} of {MAX_BLOCKS})\n")
        return 2
    if verdict == "escape":
        sys.stderr.write(_message(reasons) + f"\nfeagent: blocked {MAX_BLOCKS} times in a row for "
                         f"these specs; allowing stop. Tell the user the specs are not passing.\n")
    elif verdict == "escaped":
        sys.stderr.write(f"feagent: {len(reasons)} spec(s) still not passing (already reported, "
                         "unchanged); allowing stop\n")
    return 0


def _payload(text: str) -> dict:
    """The hook's JSON input (unused fields are fine; malformed input is treated as empty)."""
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def malformed(data: dict) -> str | None:
    """Why the state cannot be read as ``{"specs": {path: {...}}}``, or None if it can."""
    specs = data.get("specs")
    if not isinstance(specs, dict):
        return "'specs' is not a mapping"
    bad = sorted(str(k) for k, v in specs.items() if not isinstance(v, dict))
    return f"entries of {bad} are not mappings" if bad else None


def session_start(payload: dict) -> float | None:
    """Epoch time of the first timestamped transcript entry; None if there is no readable one."""
    path = payload.get("transcript_path")
    if not isinstance(path, str) or not path:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for _, line in zip(range(_SCAN_LINES), fh):
                ts = _timestamp(line)
                if ts is not None:
                    return ts
    except OSError:
        return None
    return None


def _timestamp(line: str) -> float | None:
    try:
        return datetime.fromisoformat(json.loads(line)["timestamp"]).timestamp()
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def in_session(specs: dict[str, dict], since: float | None) -> dict[str, dict]:
    """The entries recorded at or after ``since`` (all of them if ``since`` is None)."""
    if since is None:
        return dict(specs)
    return {p: e for p, e in specs.items()
            if isinstance(e.get("checked_at"), (int, float)) and e["checked_at"] >= since}


def blocking_reason(spec_path: str, entry: dict) -> str | None:
    """Why ``entry`` (a state entry of ``spec_path``) blocks stopping, or None if it does not."""
    mtime = spec_mtime(Path(spec_path))
    if mtime is None:
        return None
    if entry.get("waiver") and entry.get("waiver_spec_mtime") == mtime:
        return None
    status = entry.get("status", "unchecked")
    if entry.get("spec_mtime") != mtime:
        return f"{status}, but edited since the last run (stale): re-run check"
    if status == "pass":
        return None
    if status == "unchecked":
        return "unchecked (meshed but never checked): run check"
    return "fail: " + _failure_detail(spec_path, entry)


def _failure_detail(spec_path: str, entry: dict) -> str:
    if entry.get("error"):
        return "spec error: " + str(entry["error"]).splitlines()[0]
    gates = _failing_gates(spec_path, entry.get("report_created_at"))
    if gates:
        return "failing gates " + ", ".join(gates)
    return "no mesh (see the `mesh` output)" if gates is None else "see REPORT.md"


def _failing_gates(spec_path: str, created_at: str | None) -> list[str] | None:
    """Failing gate ids from the spec's report.json, if that report is the recorded one."""
    if created_at is None:
        return None
    from .errors import FeagentError
    from .spec import load_spec

    try:
        report = json.loads((load_spec(spec_path).out_dir() / "report.json").read_text())
    except (FeagentError, OSError, ValueError):
        return None
    if not isinstance(report, dict) or report.get("created_at") != created_at:
        return None
    return [g["id"] for g in report.get("gates", []) if g.get("status") == "fail"]


def _message(reasons: dict[str, str]) -> str:
    lines = [f"feagent: {len(reasons)} mesh spec(s) are not passing; do not finish yet:"]
    lines += [f"- {path}: {why}" for path, why in reasons.items()]
    return "\n".join(lines + [_ADVICE])


def _fingerprint(reasons: dict[str, str], specs: dict[str, dict]) -> Fingerprint:
    return [[p, spec_mtime(Path(p)), str(specs[p].get("status", "unchecked"))] for p in sorted(reasons)]


def _decide(fp: Fingerprint, active: Any) -> tuple[str, int]:
    """("allow" | "block" | "escape" | "escaped", block count) for the blocking set ``fp``; updates
    the counter in the state file (re-read here, just before writing)."""
    data = load_state()
    old = data.get(_KEY) if isinstance(data.get(_KEY), dict) else {}
    if not fp:
        if old.get("count") or old.get("allowed"):
            _store(data, 0, None, None)
        return "allow", 0
    if old.get("allowed") == fp:
        return "escaped", 0
    prev = old.get("count") if isinstance(old.get("count"), int) else 0
    count = prev + 1 if old.get("blocking") == fp and active is not False else 1
    if count > MAX_BLOCKS:
        _store(data, 0, None, fp)
        return "escape", MAX_BLOCKS
    _store(data, count, fp, None)
    return "block", count


def _store(data: dict, count: int, blocking: Fingerprint | None, allowed: Fingerprint | None) -> None:
    data[_KEY] = {"count": count, "blocking": blocking, "allowed": allowed}
    write_state(data)
