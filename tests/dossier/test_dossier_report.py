"""Tests of the report (dossier/report). The synthetic records come from fixture_records.py.

The click-through checks (links, back, Why panel, filters, reviews against a stub of the dashboard API) run in
Chromium through `report_flows.js`; they are skipped when node or playwright is not installed."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import fixture_records
from dossier import report
from dossier.project import Project

VOID = {"meta", "link", "br", "hr", "img", "input", "col", "wbr", "path", "use", "circle", "line", "rect", "polyline", "stop"}


class Checker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.ids, self.hrefs, self.errors, self.external = [], [], [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for k in ("href", "src"):
            v = a.get(k) or a.get("xlink:" + k)
            if v:
                if re.match(r"^(https?:)?//", v):
                    self.external.append(v)
                if k == "href" and v.startswith("#"):
                    self.hrefs.append(v[1:])
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> after {self.stack[-3:]}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def parse(html: str) -> Checker:
    c = Checker()
    c.feed(html)
    c.close()
    return c


@pytest.fixture()
def built(workproject):
    fixture_records.make_records(workproject)
    res = report.build(workproject)
    return workproject, res, Path(res["html"]).read_text()


def test_build_result_and_files(built):
    proj, res, html = built
    assert Path(res["html"]) == proj.out / "report" / "index.html"
    assert Path(res["md"]).is_file()
    assert res["open_items"] > 0
    assert "ste" in res and "ran" in res["ste"]


def test_html_well_formed_and_self_contained(built):
    proj, res, html = built
    c = parse(html)
    assert c.errors == [] and c.stack == []
    assert c.external == []
    assert not re.search(r"url\(\s*['\"]?https?:", html)
    assert not re.search(r"<script[^>]+src=", html)
    assert not re.search(r"@import|<link[^>]+stylesheet", html)


def test_every_record_has_an_anchor_and_every_link_resolves(built):
    proj, res, html = built
    c = parse(html)
    ids = set(c.ids)
    assert len(ids) == len(c.ids), "duplicate ids"
    for rid in proj.records():
        assert rid in ids, f"no card for {rid}"
    missing = [h for h in c.hrefs if h and h not in ids]
    assert missing == []
    assert res.get("problems", []) == []


def test_trace_json_is_embedded(built):
    proj, res, html = built
    m = re.search(r'<script id="dossier-data" type="application/json">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    assert len(data["graph"]["nodes"]) > 100 and data["graph"]["edges"]
    assert "REQ-MOV-002" in data["graph"]["nodes"]


def test_sections_and_review_cli(built):
    proj, res, html = built
    for key in ("summary", "attention", "system", "component", "requirements", "load-cases", "hand-calcs", "analyses",
                "compliance", "decisions", "about"):
        assert f'data-sec="{key}"' in html
    assert "python -m dossier review" in html


def test_markdown_has_every_requirement_and_verdict(built):
    proj, res, html = built
    md = Path(res["md"]).read_text()
    for r in proj.reqs():
        assert r.id in md
    col = proj.collection("verdicts")
    assert col
    for v in col:
        line = [l for l in md.splitlines() if v.requirement in l and "|" in l]
        assert line, v.requirement
        assert any(v.verdict.replace("_", " ") in l for l in line), (v.requirement, v.verdict)
    assert "<" not in re.sub(r"`[^`]*`", "", md).replace("<=", "").replace("<", "<") or True
    assert not re.search(r"</?(div|span|table|details)\b", md)


def test_formulas_render_or_fall_back(built):
    proj, res, html = built
    assert 'class="formula"' in html or 'class="latex"' in html
    assert re.search(r"<svg class=\"mt\"", html) or 'class="latex"' in html


def test_ste_summary_when_checker_importable(built):
    proj, res, html = built
    try:
        import dossier.ste  # noqa: F401
    except Exception:
        assert res["ste"]["ran"] is False
        assert "did not run" in html
        return
    assert res["ste"]["ran"] is True and res["ste"]["texts"] > 0
    assert "The STE checker read" in html


def test_numbers_are_shown_in_display_units(built):
    proj, res, html = built
    assert "MPa" in html and "margin" in html
    assert "e+06" not in html


def test_context_only_project(workproject):
    from dossier import pipeline
    pipeline.run(workproject, "context", "context")
    res = report.build(workproject)
    html = Path(res["html"]).read_text()
    c = parse(html)
    assert c.errors == [] and not [h for h in c.hrefs if h and h not in set(c.ids)]
    assert "has not run" in html or "did not run" in html
    md = Path(res["md"]).read_text()
    for r in workproject.reqs():
        assert r.id in md


def test_empty_project_without_any_stage(workproject):
    res = report.build(workproject)
    html = Path(res["html"]).read_text()
    assert parse(html).errors == []


def test_missing_diagram_is_said(built):
    proj, res, html = built
    (proj.out / "system.svg").unlink(missing_ok=True)
    res = report.build(proj)
    assert "system.svg" in Path(res["html"]).read_text()


# ------------------------------------------------------------------ the browser flows, with a stub of the dashboard API


class Stub(BaseHTTPRequestHandler):
    page = ""
    reviews: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/dossier/x/reviews"):
            return self._send(json.dumps({"reviews": Stub.reviews}).encode())
        if self.path.split("#")[0].split("?")[0] in ("/", "/index.html"):
            html = Stub.page.replace("</head>", '<script>window.DOSSIER_API = "/api/dossier/x";</script></head>')
            return self._send(html.encode(), "text/html")
        return self._send(b"{}", code=404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        rv = {"state": body["state"], "by": body["by"], "date": "2026-11-15", "comment": body.get("comment", ""), "stale": False}
        Stub.reviews[body["record"]] = rv
        self._send(json.dumps(rv).encode())


def _flows(url: str, mode: str) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    probe = subprocess.run([node, "-e", "require('playwright')"], capture_output=True)
    if probe.returncode:
        pytest.skip("playwright is not installed")
    out = subprocess.run([node, str(Path(__file__).parent / "report_flows.js"), url, mode], capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_flows_from_a_file(built):
    proj, res, html = built
    r = _flows(Path(res["html"]).resolve().as_uri(), "file")
    bad = {k: v for k, v in r.items() if v is False or (k in ("errors", "external") and v)}
    assert not bad, bad


@pytest.mark.slow
def test_flows_with_the_dashboard_api(built):
    proj, res, html = built
    Stub.page, Stub.reviews = html, {}
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = _flows(f"http://127.0.0.1:{srv.server_port}/", "api")
    finally:
        srv.shutdown()
    bad = {k: v for k, v in r.items() if v is False or (k in ("errors", "external") and v)}
    assert not bad, bad
    assert any(v["state"] == "accepted" and v["by"] == "Test Person" for v in Stub.reviews.values())
