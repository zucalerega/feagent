"""Solve a part's generated BC sets on one mesh and write an Ansys-style interactive results page.

    python scripts/view_results.py PARTS.jsonl PART_ID MESH.msh [--bcs 10] [--out results.html] [--open]

Each BC set goes through feacore.diagnostics.diagnose (precheck -> fixed solver -> failure-mode
checks). The page shows, per BC set: equivalent (von Mises) stress or total deformation as banded
contours (Ansys 9-band legend), the deformed shape (scale slider), supports (blue) and loads (red
arrows), the BCs in words, the diagnostics flags and key numbers. Section plane as in view_mesh.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_mesh import load  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#10233f; --line:#6b7a90; }
  html,body { margin:0; height:100%; overflow:hidden; font:13px/1.35 "Segoe UI",Roboto,Arial,sans-serif; color:var(--ink);
    background:linear-gradient(#5c85c2 0%, #a9bfdf 55%, #f7f9ff 100%); }
  #c { position:fixed; inset:0; }
  #title { position:fixed; left:14px; top:10px; pointer-events:none; max-width:46vw; }
  #title b { font-size:17px; display:block; } #title .sub { font-size:13px; }
  #legend { position:fixed; left:14px; top:74px; background:rgba(255,255,255,.88); border:1px solid var(--line); padding:8px 10px; }
  #legend .hd { font-weight:600; margin-bottom:4px; } #legend .meta { font-size:11px; color:#3a4d6b; margin-bottom:6px; }
  .band { display:flex; align-items:center; gap:8px; height:20px; font:12px ui-monospace,Menlo,Consolas,monospace; }
  .band i { width:26px; height:20px; display:inline-block; border-bottom:1px solid rgba(0,0,0,.25); }
  .band .tag { font:600 10px Arial; color:#fff; padding:1px 4px; border-radius:2px; }
  #panel { position:fixed; right:12px; top:12px; width:330px; max-height:calc(100vh - 24px); overflow:auto;
    background:rgba(255,255,255,.94); border:1px solid var(--line); padding:10px 12px; box-shadow:0 2px 8px rgba(0,0,0,.15); }
  #panel h4 { margin:4px 0 6px; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#3a4d6b; }
  select, #panel button { font:inherit; padding:4px 6px; border:1px solid var(--line); background:#f4f6fa; cursor:pointer; }
  select { width:100%; }
  .row { display:flex; gap:4px; margin:4px 0 8px; } .row button { flex:1; } .row button.on { background:#2c5aa0; color:#fff; border-color:#2c5aa0; }
  #panel label { display:flex; align-items:center; gap:6px; margin:4px 0; } input[type=range] { width:100%; }
  #words { margin:6px 0; padding-left:16px; } #words li { margin:3px 0; font-size:12px; }
  table { border-collapse:collapse; width:100%; font-size:12px; } td { padding:2px 4px; border-bottom:1px solid #e3e8f0; }
  td:last-child { text-align:right; font-family:ui-monospace,Menlo,Consolas,monospace; }
  .flag { font-size:12px; margin:3px 0; padding:3px 6px; border-left:4px solid; background:#f6f8fb; }
  .flag.fail { border-color:#c62828; } .flag.warn { border-color:#ef8f00; } .flag.info { border-color:#5b7db1; }
  .flag b { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11px; }
  #status { font-weight:600; } .s-ok{color:#2e7d32} .s-warn{color:#b26a00} .s-fail{color:#c62828}
  #triad { position:fixed; left:8px; bottom:8px; width:120px; height:120px; pointer-events:none; }
  #tip { position:fixed; pointer-events:none; background:#10233f; color:#fff; padding:3px 7px; font-size:12px; border-radius:2px; display:none; }
  hr { border:0; border-top:1px solid #d5dbe6; margin:8px 0; }
</style></head>
<body>
<canvas id="c"></canvas>
<div id="title"><b>__TITLE__</b><span class="sub" id="sub"></span></div>
<div id="legend"></div>
<div id="panel">
  <h4>Boundary conditions</h4>
  <select id="case"></select>
  <ol id="words"></ol>
  <div>Status: <span id="status"></span></div>
  <div id="flags"></div>
  <hr><h4>Result</h4>
  <div class="row"><button data-f="vm" class="on">Equivalent stress</button><button data-f="u">Total deformation</button></div>
  <label>Deformation scale <span id="scaleLbl"></span></label>
  <input type="range" id="scale" min="0" max="100" value="60">
  <label><input type="checkbox" id="edges"> Element edges</label>
  <label><input type="checkbox" id="bcs" checked> Show supports / loads</label>
  <hr><h4>Section plane</h4>
  <label><input type="checkbox" id="secOn"> Enable section</label>
  <div class="row"><button data-ax="0">X</button><button data-ax="1" class="on">Y</button><button data-ax="2">Z</button></div>
  <input type="range" id="secPos" min="0" max="1000" value="500">
  <hr><h4>Numbers</h4>
  <table id="nums"></table>
</div>
<canvas id="triad"></canvas>
<div id="tip"></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const DATA = __DATA__, CASES = __CASES__;
const b64 = (s, T) => new T(Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer);
const P = b64(DATA.points, Float32Array), T = b64(DATA.tets, Uint32Array), nt = T.length / 4, nv = P.length / 3;
for (const c of CASES) if (c.vm) { c.VM = b64(c.vm, Float32Array); c.U = b64(c.u, Float32Array); }
const BANDS = [[0,0,255],[0,178,255],[0,255,255],[0,255,178],[0,255,0],[178,255,0],[255,255,0],[255,178,0],[255,0,0]];
const tex = new THREE.DataTexture(new Uint8Array(BANDS.flatMap(c => [...c, 255])), 9, 1);
tex.magFilter = tex.minFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
const GREY = new THREE.Color(0.60, 0.72, 0.74);

function key(a, b, c) { if (a > b) [a, b] = [b, a]; if (b > c) [b, c] = [c, b]; if (a > b) [a, b] = [b, a]; return a + "," + b + "," + c; }
const FACES = [[0,2,1],[0,1,3],[0,3,2],[1,2,3]];
const cent = new Float32Array(nt * 3);
for (let e = 0; e < nt; e++) for (let d = 0; d < 3; d++)
  cent[3*e+d] = (P[3*T[4*e]+d] + P[3*T[4*e+1]+d] + P[3*T[4*e+2]+d] + P[3*T[4*e+3]+d]) / 4;
function boundary(keep) {
  const seen = new Map();
  for (let e = 0; e < nt; e++) { if (!keep(e)) continue;
    for (const f of FACES) { const a = T[4*e+f[0]], b = T[4*e+f[1]], c = T[4*e+f[2]], k = key(a, b, c);
      if (seen.has(k)) seen.set(k, null); else seen.set(k, [a, b, c]); } }
  return [...seen.values()].filter(Boolean);
}

const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById("c"), antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene(), R = DATA.radius;
const camera = new THREE.PerspectiveCamera(30, 1, R / 100, R * 100);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.0));
const key1 = new THREE.DirectionalLight(0xffffff, 1.3); scene.add(key1);
const mat = new THREE.MeshPhongMaterial({ map: tex, shininess: 25, specular: 0x222222, side: THREE.DoubleSide,
  polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
const greyMat = new THREE.MeshPhongMaterial({ color: GREY, flatShading: true, side: THREE.DoubleSide });
const lineMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 });
const bcGroup = new THREE.Group(); scene.add(bcGroup);
let mesh = null, lines = null, faces = [];
const st = { c: 0, f: "vm", scale: 0.6, edges: false, bcs: true, sec: false, ax: 1, pos: 0.5 };

function field(c) {
  if (!c.VM) return null;
  if (st.f === "vm") return c.VM;
  const m = new Float32Array(nv); for (let i = 0; i < nv; i++) m[i] = Math.hypot(c.U[3*i], c.U[3*i+1], c.U[3*i+2]); return m;
}
function rebuild() {
  const c = CASES[st.c], F = field(c);
  const plane = DATA.lo[st.ax] + st.pos * (DATA.hi[st.ax] - DATA.lo[st.ax]);
  faces = boundary(st.sec ? (e => cent[3*e+st.ax] >= plane) : () => true);
  let lo = 0, hi = 1;
  if (F) { lo = Infinity; hi = -Infinity; for (const v of F) { if (v < lo) lo = v; if (v > hi) hi = v; } if (hi <= lo) hi = lo + 1e-12; }
  const s = c.U ? st.scale * c.autoscale : 0;
  const n = faces.length, pos = new Float32Array(n * 9), uv = new Float32Array(n * 6), seg = new Float32Array(n * 18);
  const X = v => [P[3*v] + s * (c.U ? c.U[3*v] : 0), P[3*v+1] + s * (c.U ? c.U[3*v+1] : 0), P[3*v+2] + s * (c.U ? c.U[3*v+2] : 0)];
  faces.forEach((f, i) => {
    const xs = f.map(X);
    xs.forEach((x, j) => pos.set(x, 9*i + 3*j));
    f.forEach((v, j) => uv.set([F ? Math.min(0.9999, Math.max(0, (F[v] - lo) / (hi - lo))) : 0, 0.5], 6*i + 2*j));
    [[0,1],[1,2],[2,0]].forEach(([a, b], j) => seg.set([...xs[a], ...xs[b]], 18*i + 6*j));
  });
  if (mesh) { scene.remove(mesh, lines); mesh.geometry.dispose(); lines.geometry.dispose(); }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2)); g.computeVertexNormals();
  mesh = new THREE.Mesh(g, F ? mat : greyMat); scene.add(mesh);
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(seg, 3));
  lines = new THREE.LineSegments(lg, lineMat); lines.visible = st.edges; scene.add(lines);
  legend(c, lo, hi, F);
  document.getElementById("scaleLbl").textContent = c.U ? `${(s).toFixed(0)}x (auto ${c.autoscale.toFixed(0)}x)` : "";
}
function legend(c, lo, hi, F) {
  const el = document.getElementById("legend");
  if (!F) { el.innerHTML = `<div class="hd">No result</div><div class="meta">${c.status === "fail" ? "setup rejected before solving" : ""}</div>`; return; }
  const vm = st.f === "vm", unit = vm ? "MPa" : "mm";
  let h = `<div class="hd">${vm ? "Equivalent (von-Mises) Stress" : "Total Deformation"}</div><div class="meta">Type: ${vm ? "Equivalent (von-Mises) Stress" : "Total Deformation"}<br>Unit: ${unit}<br>Case: ${c.id}</div>`;
  for (let k = 8; k >= 0; k--) {
    const v = lo + (hi - lo) * (k + 1) / 9, col = BANDS[k];
    const tag = k === 8 ? `<span class="tag" style="background:#c62828">Max</span>` : "";
    h += `<div class="band"><i style="background:rgb(${col})"></i>${v.toPrecision(4)} ${tag}</div>`;
  }
  h += `<div class="band"><i style="background:transparent;border:0"></i>${lo.toPrecision(4)} <span class="tag" style="background:#1f4fd6">Min</span></div>`;
  el.innerHTML = h;
}
function drawBCs() {
  bcGroup.clear(); const c = CASES[st.c]; bcGroup.visible = st.bcs;
  for (const b of c.markers) {
    const p = new THREE.Vector3(...b.at), L = R * 0.35;
    if (b.kind === "support") {
      const cone = new THREE.Mesh(new THREE.ConeGeometry(R * 0.05, R * 0.1, 3), new THREE.MeshBasicMaterial({ color: b.roller ? 0x00a0ff : 0x1f3fd6 }));
      const n = new THREE.Vector3(...b.dir); cone.position.copy(p.clone().addScaledVector(n, R * 0.05));
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), n.clone().negate()); bcGroup.add(cone);
    } else {
      const d = new THREE.Vector3(...b.dir).normalize();
      bcGroup.add(new THREE.ArrowHelper(d, p.clone().addScaledVector(d, -L), L, 0xd62728, L * 0.3, L * 0.18));
    }
  }
}
function showCase() {
  const c = CASES[st.c];
  document.getElementById("sub").textContent = ` ${c.id}`;
  document.getElementById("words").innerHTML = c.words.map(w => `<li>${w}</li>`).join("");
  const s = document.getElementById("status"); s.textContent = c.status.toUpperCase(); s.className = "s-" + c.status;
  document.getElementById("flags").innerHTML = c.flags.map(([sev, id, msg]) => `<div class="flag ${sev}"><b>${id}</b> ${msg}</div>`).join("") || "<div class='flag info'>no flags</div>";
  document.getElementById("nums").innerHTML = c.numbers.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  rebuild(); drawBCs();
}
const sel = document.getElementById("case");
CASES.forEach((c, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${c.id} [${c.status}]`; sel.appendChild(o); });
sel.onchange = e => { st.c = +e.target.value; showCase(); };
document.querySelectorAll("[data-f]").forEach(b => b.onclick = () => { st.f = b.dataset.f;
  document.querySelectorAll("[data-f]").forEach(x => x.classList.toggle("on", x === b)); rebuild(); });
let pend = 0; const later = fn => { cancelAnimationFrame(pend); pend = requestAnimationFrame(fn); };
document.getElementById("scale").oninput = e => { st.scale = e.target.value / 100; later(rebuild); };
document.getElementById("edges").onchange = e => { st.edges = e.target.checked; lines.visible = st.edges; };
document.getElementById("bcs").onchange = e => { st.bcs = e.target.checked; bcGroup.visible = st.bcs; };
document.getElementById("secOn").onchange = e => { st.sec = e.target.checked; rebuild(); };
document.querySelectorAll("[data-ax]").forEach(b => b.onclick = () => { st.ax = +b.dataset.ax;
  document.querySelectorAll("[data-ax]").forEach(x => x.classList.toggle("on", x === b)); if (st.sec) rebuild(); });
document.getElementById("secPos").oninput = e => { st.pos = e.target.value / 1000; if (st.sec) later(rebuild); };

// hover: value under the cursor
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = document.getElementById("tip");
renderer.domElement.addEventListener("pointermove", ev => {
  mouse.set(ev.clientX / innerWidth * 2 - 1, -ev.clientY / innerHeight * 2 + 1); ray.setFromCamera(mouse, camera);
  const hit = mesh && ray.intersectObject(mesh)[0], F = field(CASES[st.c]);
  if (!hit || !F) { tip.style.display = "none"; return; }
  const f = faces[hit.faceIndex], v = (F[f[0]] + F[f[1]] + F[f[2]]) / 3;
  tip.textContent = `${v.toPrecision(4)} ${st.f === "vm" ? "MPa" : "mm"}`;
  tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY + 10 + "px"; tip.style.display = "block";
});

// triad
const tr = new THREE.WebGLRenderer({ canvas: document.getElementById("triad"), antialias: true, alpha: true });
tr.setPixelRatio(devicePixelRatio); tr.setSize(120, 120, false);
const tScene = new THREE.Scene(), tCam = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
[[0xd62728,[1,0,0],"X"],[0x2ca02c,[0,1,0],"Y"],[0x1f4fd6,[0,0,1],"Z"]].forEach(([c, d, l]) => {
  tScene.add(new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(), 1, c, 0.3, 0.16));
  const cv = document.createElement("canvas"); cv.width = cv.height = 64; const x = cv.getContext("2d");
  x.fillStyle = "#" + c.toString(16).padStart(6, "0"); x.font = "bold 44px Arial"; x.textAlign = "center"; x.textBaseline = "middle"; x.fillText(l, 32, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv) })); s.position.set(...d.map(v => v * 1.35)); s.scale.set(0.45, 0.45, 1); tScene.add(s);
});
function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();
camera.up.set(0, 0, 1); camera.position.set(R * 2.6, -R * 3.4, R * 2.3); controls.update();
showCase();
(function loop() {
  requestAnimationFrame(loop); controls.update();
  key1.position.copy(camera.position).add(new THREE.Vector3(R, R * 0.6, R * 2)); renderer.render(scene, camera);
  tCam.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(4); tCam.up.copy(camera.up); tCam.lookAt(0, 0, 0); tr.render(tScene, tCam);
})();
</script></body></html>
"""


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def _fmt(v, unit="", scale=1.0, digits=4):
    return "-" if v is None else f"{v * scale:.{digits}g} {unit}".strip()


def solve_cases(part, sets, mesh: str, X: np.ndarray, offset: np.ndarray, diag_mm: float) -> list[dict]:
    from feacore.diagnostics import diagnose
    from feacore.goldgen.run import make_case

    cases = []
    for s in sets:
        t0 = time.perf_counter()
        rep, res = diagnose(make_case(mesh, s))
        m = rep.metrics
        c = {"id": s.id, "words": s.words, "status": rep.status,
             "flags": [[f.severity, f.id, f.message] for f in rep.flags], "markers": []}
        for b in s.bcs:
            f = part.face(int(b.group[1:]))
            at = (np.asarray(f.centroid) * 1000 - offset).tolist()
            n = np.asarray(f.normal if f.normal else (f.axis or [0, 0, 1]), float)
            if b.type in ("fixed", "displacement"):
                c["markers"].append({"kind": "support", "at": at, "dir": n.tolist(), "roller": b.type == "displacement"})
            elif b.type == "pressure":
                c["markers"].append({"kind": "load", "at": at, "dir": (-n).tolist()})
            else:
                c["markers"].append({"kind": "load", "at": at, "dir": list(b.value)})
        if res is not None:
            vm = res.probe("von_mises", X) / 1e6
            U = np.column_stack([res.probe(k, X) for k in ("ux", "uy", "uz")]) * 1000
            umax = float(np.linalg.norm(U, axis=1).max())
            c.update(vm=_b64(vm), u=_b64(U), autoscale=float(0.06 * diag_mm / umax) if umax > 0 else 1.0)
        c["numbers"] = [
            ["DOFs", _fmt(m.get("n_dofs"), digits=7)],
            ["Solve + checks", f"{time.perf_counter() - t0:.1f} s"],
            ["Max von Mises (raw peak)", _fmt(m.get("vm_max"), "MPa", 1e-6)],
            ["Max von Mises away from BC edges", _fmt(m.get("vm_trusted"), "MPa", 1e-6)],
            ["Max deformation", _fmt(m.get("max_disp"), "mm", 1e3)],
            ["Strain energy", _fmt(m.get("strain_energy"), "J")],
            ["Est. discretization error (ZZ)", _fmt(m.get("zz_eta_rel"), "", 100, 3) + " %" if m.get("zz_eta_rel") is not None else "-"],
        ]
        print(f"[{s.id}] {rep.status} flags={rep.ids()} t={time.perf_counter() - t0:.1f}s", flush=True)
        cases.append(c)
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts")
    ap.add_argument("part_id")
    ap.add_argument("mesh")
    ap.add_argument("--bcs", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    from feacore.goldgen.bcgen import generate
    from feacore.goldgen.parts import Part

    row = next(r for r in map(json.loads, open(a.parts)) if r.get("ok") and r["part"]["id"] == a.part_id)
    part = Part.from_json(json.dumps(row["part"]))
    sets = generate(part, a.bcs)
    data, X = load(a.mesh)
    cases = solve_cases(part, sets, a.mesh, X, np.asarray(data["offset"]), part.diag * 1000)
    title = a.title or f"{a.part_id}  ({Path(a.mesh).stem}, {len(cases)} BC sets)"
    out = Path(a.out or Path(a.mesh).with_suffix(".results.html"))
    out.write_text(PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data))
                   .replace("__CASES__", json.dumps(cases)))
    print(out)
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
