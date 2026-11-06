"""Test data section tools (dashboard #/testdata): physical tests read the way the ACE loop reads them.

  testdata.summarize   python -m ace.evidence summarize SPEC [--predict]: the loop's processing of a test (load
                       cycles, fitted compliance, uncertainty budget) and, with predict, the problem's own setup
                       solved on a fine mesh and compared with the test

Uploads themselves (parse, preview, save) are quick and run in the server: feacore/ui/testdata_api.py.
"""

from __future__ import annotations

from feacore.ui.tasks import Param, Tool, register_tool


def _summarize_argv(a: dict) -> list[str]:
    argv = [a["spec"]]
    if a.get("predict"):
        argv.append("--predict")
        if a.get("h"):
            argv += ["--h", f"{a['h'] / 1000.0:g}"]
    return argv


register_tool(Tool(
    name="testdata.summarize", group="testdata", command=["-m", "ace.evidence", "summarize"],
    title="Check a test the way the loop reads it",
    description="Runs the loop's own reader on a test spec: the machine export's columns and units, the load cycles, "
                "the fitted compliance and its uncertainty budget. With predict it also solves the problem's own "
                "setup (linear, a fine P2 mesh) and compares it with the test: a minute or two.",
    params=[Param("spec", "path", required=True, help="the test's JSON spec (test.json of an upload)"),
            Param("predict", "bool", False, help="also solve the problem's own setup and compare it with the test"),
            Param("h", "float", None, label="mesh size (mm)", help="for predict (default: the problem's h0 / 2)",
                  min=0.2, max=50.0, advanced=True)],
    timeout=3600, build=_summarize_argv))
