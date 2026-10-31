"""Claude Code Stop hook: do not let the agent finish while a tracked mesh spec is not passing.

Reads the state file written by ``feagent mesh/check/waive`` (``check.state_path``). A spec blocks
when its entry is not "pass" ("fail", "unchecked") or is stale (the spec's mtime differs from the
recorded ``spec_mtime``), unless it has a valid waiver (``waiver_spec_mtime`` equals the spec's
current mtime). Specs whose file no longer exists are ignored. Blocking = reason on stderr and exit
code 2 (Claude Code feeds stderr back to the agent). After ``MAX_BLOCKS`` consecutive blocks for
the same set of specs the hook lets the agent stop (with a note) so it cannot loop forever; the
counter lives in the state file under ``"hook_blocks"``: ``{"count": n, "specs": [paths]}``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .check import _mtime, _write_state, load_state

MAX_BLOCKS = 3
_ADVICE = ("Fix each spec and re-run `feagent check SPEC` (MCP: check_mesh) until it passes. If "
           "finishing as is is acceptable, run `feagent waive SPEC --reason \"...\"` and tell the "
           "user which gates fail and why.")


def main(stdin_text: str | None = None) -> int:
    """Run the hook on the Claude Code Stop payload (read from stdin if None); return exit code."""
    _payload(sys.stdin.read() if stdin_text is None else stdin_text)
    data = load_state()
    reasons = {path: why for path, entry in sorted(data["specs"].items())
               if (why := blocking_reason(path, entry)) is not None}
    count = _bump_counter(data, sorted(reasons))
    if not reasons:
        return 0
    if count > MAX_BLOCKS:
        sys.stderr.write(_message(reasons) + f"\nfeagent: blocked {MAX_BLOCKS} times in a row for "
                         f"these specs; allowing stop. Tell the user the specs are not passing.\n")
        _reset_counter(data)
        return 0
    sys.stderr.write(_message(reasons) + f"\n(feagent stop check {count} of {MAX_BLOCKS})\n")
    return 2


def _payload(text: str) -> dict:
    """The hook's JSON input (unused fields are fine; malformed input is treated as empty)."""
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def blocking_reason(spec_path: str, entry: dict) -> str | None:
    """Why ``entry`` (a state entry of ``spec_path``) blocks stopping, or None if it does not."""
    mtime = _mtime(Path(spec_path))
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
    return "no mesh (see `feagent mesh SPEC` output)" if gates is None else "see REPORT.md"


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


def _bump_counter(data: dict, specs: list[str]) -> int:
    """Consecutive-block count including this call (0 and reset when nothing blocks)."""
    old = data.get("hook_blocks")
    old = old if isinstance(old, dict) else {}
    if not specs:
        if old.get("count"):
            _reset_counter(data)
        return 0
    count = old.get("count", 0) + 1 if old.get("specs") == specs else 1
    data["hook_blocks"] = {"count": count, "specs": specs}
    _write_state(data)
    return count


def _reset_counter(data: dict) -> None:
    data["hook_blocks"] = {"count": 0, "specs": []}
    _write_state(data)
