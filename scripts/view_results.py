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
  .row { display:flex; gap:4px; margin:4px 0 8px; } .row button { flex:1; } #panel .row button.on { background:#2c5aa0; color:#fff; border-color:#2c5aa0; }
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
const P = b64(DATA.points, Float32Array), T = b64(DATA.tets, Uint32Array), nv = P.length / 3;
for (const c of CASES) if (c.vm) { c.VM = b64(c.vm, Float32Array); c.U = b64(c.u, Float32Array); }
const SF = b64(DATA.surf, Uint32Array), groupOf = new Map(), gindex = new Map(DATA.groups.map((g, i) => [g, i]));
const BC_COL = { fixed: 0x3d5afe, roller: 0x00b8d4, pressure: 0xff1744, force: 0xff6d00 };
const BANDS = [[0,0,255],[0,178,255],[0,255,255],[0,255,178],[0,255,0],[178,255,0],[255,255,0],[255,178,0],[255,0,0]];
const tex = new THREE.DataTexture(new Uint8Array(BANDS.flatMap(c => [...c, 255])), 9, 1);
tex.magFilter = tex.minFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
const GREY = new THREE.Color(0.60, 0.72, 0.74);

function key(a, b, c) { if (a > b) [a, b] = [b, a]; if (b > c) [b, c] = [c, b]; if (a > b) [a, b] = [b, a]; return a + "," + b + "," + c; }
const NPE = DATA.npe || 4, nt = T.length / NPE;
// faces: corners (outward) + the midside nodes of edges ab, bc, ca. Node numbering is meshio's
// (VTK): 4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3) 8=(1,3) 9=(2,3); gmsh's own files swap 8 and 9.
for (let i = 0; i < SF.length; i += 4) groupOf.set(key(SF[i], SF[i+1], SF[i+2]), SF[i+3]);
const FACES = [[0,2,1,6,5,4],[0,1,3,4,8,7],[0,3,2,7,9,6],[1,2,3,5,9,8]];
const SUB = NPE === 10 ? 4 : 1;
const cent = new Float32Array(nt * 3);
for (let e = 0; e < nt; e++) for (let d = 0; d < 3; d++)
  cent[3*e+d] = (P[3*T[NPE*e]+d] + P[3*T[NPE*e+1]+d] + P[3*T[NPE*e+2]+d] + P[3*T[NPE*e+3]+d]) / 4;
// boundary faces (used by exactly one kept tet): [a, b, c, mab, mbc, mca] (mids = -1 for tet4)
function boundary(keep) {
  const seen = new Map();
  for (let e = 0; e < nt; e++) { if (!keep(e)) continue;
    for (const f of FACES) {
      const a = T[NPE*e+f[0]], b = T[NPE*e+f[1]], c = T[NPE*e+f[2]], k = key(a, b, c);
      const m = NPE === 10 ? [T[NPE*e+f[3]], T[NPE*e+f[4]], T[NPE*e+f[5]]] : [-1, -1, -1];
      if (seen.has(k)) seen.set(k, null); else seen.set(k, [a, b, c, ...m]);
    } }
  return [...seen.values()].filter(Boolean);
}
// a face as drawn: 4 sub-triangles through the midside nodes (curved), or the flat triangle
const subTris = f => SUB === 4 ? [[f[0],f[3],f[5]],[f[3],f[1],f[4]],[f[5],f[4],f[2]],[f[3],f[4],f[5]]] : [[f[0],f[1],f[2]]];
// element edges as drawn: a-mab-b, b-mbc-c, c-mca-a (6 segments) or the 3 straight edges
const subEdges = f => SUB === 4 ? [[f[0],f[3]],[f[3],f[1]],[f[1],f[4]],[f[4],f[2]],[f[2],f[5]],[f[5],f[0]]] : [[f[0],f[1]],[f[1],f[2]],[f[2],f[0]]];

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
  const n = faces.length, nt3 = SUB * 9, ns = SUB === 4 ? 6 : 3;
  const pos = new Float32Array(n * nt3), uv = new Float32Array(n * SUB * 6), seg = new Float32Array(n * ns * 6);
  const X = v => [P[3*v] + s * (c.U ? c.U[3*v] : 0), P[3*v+1] + s * (c.U ? c.U[3*v+1] : 0), P[3*v+2] + s * (c.U ? c.U[3*v+2] : 0)];
  const tv = v => F ? Math.min(0.9999, Math.max(0, (F[v] - lo) / (hi - lo))) : 0;
  faces.forEach((f, i) => {
    subTris(f).forEach((t, q) => t.forEach((v, j) => { pos.set(X(v), i*nt3 + 9*q + 3*j); uv.set([tv(v), 0.5], i*SUB*6 + 6*q + 2*j); }));
    subEdges(f).forEach(([u, v], j) => seg.set([...X(u), ...X(v)], (i*ns + j) * 6));
  });
  if (mesh) { scene.remove(mesh, lines); mesh.geometry.dispose(); lines.geometry.dispose(); }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2)); g.computeVertexNormals();
  mesh = new THREE.Mesh(g, F ? mat : greyMat); scene.add(mesh);
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(seg, 3));
  lines = new THREE.LineSegments(lg, lineMat); lines.visible = st.edges; scene.add(lines);
  legend(c, lo, hi, F);
  drawBCs(c, X);
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
function label(text, colour) {
  const cv = document.createElement("canvas"), x = cv.getContext("2d"); x.font = "bold 30px Arial";
  const w = Math.ceil(x.measureText(text).width) + 22; cv.width = w; cv.height = 44;
  x.fillStyle = colour; x.fillRect(0, 0, w, 44); x.strokeStyle = "#fff"; x.lineWidth = 3; x.strokeRect(1.5, 1.5, w - 3, 41);
  x.font = "bold 30px Arial"; x.fillStyle = "#fff"; x.textBaseline = "middle"; x.fillText(text, 11, 23);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false }));
  sp.scale.set(R * 0.065 * w / 44, R * 0.065, 1); sp.renderOrder = 10; return sp;
}
// supports / loads on the (deformed) result: shaded BC faces, arrows across load faces, lettered labels
function drawBCs(c, X) {
  bcGroup.clear(); bcGroup.visible = st.bcs; const placed = [];
  for (const b of c.markers) {
    const gi = gindex.get(b.group), col = BC_COL[b.kind], mine = faces.filter(f => groupOf.get(key(f[0], f[1], f[2])) === gi);
    if (!mine.length) continue;
    const pos = new Float32Array(mine.length * SUB * 9);
    mine.forEach((f, i) => subTris(f).forEach((t, q) => t.forEach((v, j) => pos.set(X(v), i*SUB*9 + 9*q + 3*j))));
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    bcGroup.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.72, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, depthWrite: false })));
    // x-ray copy: BC faces hidden behind the part still show, faintly
    const xr = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.22, side: THREE.DoubleSide,
      depthTest: false, depthWrite: false })); xr.renderOrder = 5; bcGroup.add(xr);
    // centroid of the (deformed) face group
    const ctr = new THREE.Vector3(); let area = 0;
    const V = v => new THREE.Vector3(...X(v));
    const tri = mine.map(f => {                        // per face: centroid + outward normal (faces are outward)
      const a = V(f[0]), bb = V(f[1]), cc = V(f[2]), nrm = bb.clone().sub(a).cross(cc.clone().sub(a));
      const ar = nrm.length() / 2, p = a.clone().add(bb).add(cc).divideScalar(3);
      ctr.addScaledVector(p, ar); area += ar; return { p, n: nrm.normalize() };
    });
    ctr.divideScalar(area || 1);
    if (b.kind === "pressure" || b.kind === "force") {
      const L = R * 0.16, step = Math.max(1, Math.floor(tri.length / 14));
      for (let i = 0; i < tri.length; i += step) {
        const d = b.dir ? new THREE.Vector3(...b.dir).normalize() : tri[i].n.clone().negate();
        bcGroup.add(new THREE.ArrowHelper(d, tri[i].p.clone().addScaledVector(d, -L), L, col, L * 0.32, L * 0.2));
      }
    }
    // label pushed out from the part's centre along the face direction, joined by a leader line
    const out = ctr.clone().lengthSq() > 1e-12 ? ctr.clone().normalize() : new THREE.Vector3(0, 0, 1);
    const at = ctr.clone().addScaledVector(out, R * (0.38 + 0.08 * placed.length));
    placed.push(at);
    const tag = label(`${b.letter}  ${b.label}`, "#" + col.toString(16).padStart(6, "0"));
    tag.position.copy(at); tag.center.set(out.x >= 0 ? 0 : 1, 0.5); bcGroup.add(tag);
    const lg = new THREE.BufferGeometry().setFromPoints([ctr, at]);
    const ln = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: col, depthTest: false })); ln.renderOrder = 9; bcGroup.add(ln);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(R * 0.018), new THREE.MeshBasicMaterial({ color: col, depthTest: false }));
    dot.position.copy(ctr); dot.renderOrder = 9; bcGroup.add(dot);
  }
}
// camera on the side where the BC faces are (mean of their centroids), else iso
function aim(c) {
  const m = new THREE.Vector3(); c.markers.forEach(b => m.add(new THREE.Vector3(...b.at)));
  const d = (m.lengthSq() > 1e-9 ? m.normalize() : new THREE.Vector3(0.6, -0.8, 0.5)).add(new THREE.Vector3(0.35, -0.45, 0.4)).normalize();
  camera.up.set(0, 0, 1); camera.position.copy(d.multiplyScalar(R * 4.4)); controls.target.set(0, 0, 0); controls.update();
}
function showCase() {
  const c = CASES[st.c];
  document.getElementById("sub").textContent = ` ${c.id}`;
  document.getElementById("words").innerHTML = c.words.map((w, i) => { const m = c.markers[i];
    const col = m ? "#" + BC_COL[m.kind].toString(16).padStart(6, "0") : "#999";
    return `<li style="list-style:none;margin-left:-16px"><b style="background:${col};color:#fff;padding:0 6px;margin-right:6px;border-radius:2px">${m ? m.letter : ""}</b>${w}</li>`; }).join("");
  const s = document.getElementById("status"); s.textContent = c.status.toUpperCase(); s.className = "s-" + c.status;
  document.getElementById("flags").innerHTML = c.flags.map(([sev, id, msg]) => `<div class="flag ${sev}"><b>${id}</b> ${msg}</div>`).join("") || "<div class='flag info'>no flags</div>";
  document.getElementById("nums").innerHTML = c.numbers.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  rebuild();
}
const sel = document.getElementById("case");
CASES.forEach((c, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${c.id} [${c.status}]`; sel.appendChild(o); });
sel.onchange = e => { st.c = +e.target.value; showCase(); aim(CASES[st.c]); };
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
  const f = faces[Math.floor(hit.faceIndex / SUB)], v = (F[f[0]] + F[f[1]] + F[f[2]]) / 3;
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
// keep the model centred in the space left of the side panel
function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight;
  const shift = Math.min(180, innerWidth * 0.12); camera.setViewOffset(innerWidth, innerHeight, shift, 0, innerWidth, innerHeight);
  camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();
showCase(); aim(CASES[0]);
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


def nodal_at(res, X: np.ndarray):
    """von Mises (MPa) and displacement (mm) at the mesh nodes X (m). The solver's nodal values sit
    on the same nodes, so one k-d tree match replaces per-point location (probe), which takes
    minutes at ~300k nodes; nodes not matched exactly (none expected) fall back to probe."""
    from scipy.spatial import cKDTree

    C, vm = res.nodal("von_mises")
    d, idx = cKDTree(C).query(X[:, : C.shape[1]])
    U = np.column_stack([res.nodal(k)[1][idx] for k in ("ux", "uy", "uz")])
    vm = vm[idx]
    miss = d > 1e-9 * max(float(np.ptp(X, axis=0).max()), 1e-300)
    if miss.any():
        vm[miss] = res.probe("von_mises", X[miss])
        U[miss] = np.column_stack([res.probe(k, X[miss]) for k in ("ux", "uy", "uz")])
    return vm / 1e6, U * 1000


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
        for i, b in enumerate(s.bcs):
            f = part.face(int(b.group[1:]))
            at = (np.asarray(f.centroid) * 1000 - offset).tolist()
            n = np.asarray(f.normal if f.normal else (f.axis or [0, 0, 1]), float)
            letter = "ABCDEFGH"[i]
            if b.type == "fixed":
                kind, label, d = "fixed", f"Fixed Support ({b.group})", n
            elif b.type == "displacement":
                comp = "".join("xyz"[k] for k in b.components)
                kind, label, d = "roller", f"Displacement {comp} = 0 ({b.group})", n
            elif b.type == "pressure":
                kind, label, d = "pressure", f"Pressure {b.value / 1e6:g} MPa ({b.group})", -n if f.normal else None
            else:
                v = np.asarray(b.value, float)
                kind, label, d = "force", f"Force {np.linalg.norm(v):g} N ({b.group})", v / np.linalg.norm(v)
            c["markers"].append({"kind": kind, "group": b.group, "letter": letter, "label": label, "at": at,
                                 "dir": None if d is None else np.asarray(d, float).tolist()})
        if res is not None:
            vm, U = nodal_at(res, X)
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
    ap.add_argument("--only", type=int, nargs="*", help="keep only these BC-set indices (e.g. --only 2)")
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    from feacore.goldgen.bcgen import generate
    from feacore.goldgen.parts import Part

    row = next(r for r in map(json.loads, open(a.parts)) if r.get("ok") and r["part"]["id"] == a.part_id)
    part = Part.from_json(json.dumps(row["part"]))
    sets = generate(part, a.bcs)
    if a.only is not None:
        sets = [s for s in sets if int(s.id.rsplit("_bc", 1)[1]) in a.only]
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
