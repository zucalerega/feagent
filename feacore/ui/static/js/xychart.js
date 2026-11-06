// Force-displacement style x-y line chart: simulation curves over a reference curve (the physical test).
// Dependency-free SVG like plot.js: one y axis, hairline grid, 2px lines, an HTML legend with line keys, and a
// crosshair whose tooltip lists every curve at the pointer's x (values first, names after).
import { fmt, showTip, hideTip, esc } from './api.js';
import { niceTicks } from './plot.js';

const NS = 'http://www.w3.org/2000/svg';
// the dashboard's categorical order (app.css --series-*): blue, orange, aqua, yellow, magenta, green, violet, red
export const SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
export const REF_COLOR = '#0b0b0b';
const C = { ink: '#0b0b0b', ink2: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7', surface: '#fcfcfb' };

function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  for (const c of children.flat()) if (c !== null && c !== undefined) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

const finite = p => Number.isFinite(p[0]) && Number.isFinite(p[1]);

/** y of a curve at x: linear between the bracketing points when x only grows along the curve, else the
 *  point nearest in x (a crush curve drops at a held platen). null outside the curve's x range. */
export function valueAt(pts, x) {
  if (!pts.length || x < pts[0][0] - 1e-12) return null;
  let monotone = true;
  for (let i = 1; i < pts.length && monotone; i++) if (pts[i][0] < pts[i - 1][0]) monotone = false;
  const last = pts[pts.length - 1][0];
  if (monotone) {
    if (x > last + 1e-12) return null;
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid][0] <= x) lo = mid; else hi = mid; }
    const [x0, y0] = pts[lo], [x1, y1] = pts[hi];
    return x1 === x0 ? y1 : y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }
  let best = null, bd = Infinity;
  for (const p of pts) { const d = Math.abs(p[0] - x); if (d < bd) { bd = d; best = p; } }
  const span = Math.max(...pts.map(p => p[0])) - pts[0][0];
  return bd <= span * 0.02 ? best[1] : null;
}

/**
 * opts: series [{id, label, color, points [[x, y], ...], width, dash, ref, markers [{x, y, label, shape}]}],
 *       xTitle, yTitle, xUnit, yUnit, xMax, yMax, width, height. Reference series (ref: true) draw first, in ink.
 * Returns an SVG element.
 */
export function xyChart(opts) {
  const W = opts.width || 900, H = opts.height || 420, m = { l: 70, r: 20, t: 14, b: 50 };
  const series = (opts.series || []).map(sr => ({ ...sr, points: (sr.points || []).filter(finite) })).filter(sr => sr.points.length);
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': opts.ariaLabel || 'force vs displacement' });
  if (!series.length) {
    svg.append(s('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'axis-title' }, opts.empty || 'no curve to show'));
    return svg;
  }
  const xs = series.flatMap(sr => sr.points.map(p => p[0])), ys = series.flatMap(sr => sr.points.map(p => p[1]));
  let x0 = Math.min(0, ...xs), x1 = opts.xMax ?? Math.max(...xs) * 1.02;
  let y0 = Math.min(0, ...ys), y1 = opts.yMax ?? Math.max(...ys) * 1.06;
  if (!(x1 > x0)) x1 = x0 + 1;
  if (!(y1 > y0)) y1 = y0 + 1;
  const xt = niceTicks(x0, x1, 8), yt = niceTicks(y0, y1, 6);
  x0 = Math.min(x0, xt[0]); x1 = Math.max(x1, xt[xt.length - 1]);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const g = s('g');
  svg.append(g);
  const clipId = `xyclip${Math.random().toString(36).slice(2, 8)}`;
  svg.append(s('defs', {}, s('clipPath', { id: clipId }, s('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }))));

  const tick = v => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? fmt(v, 3) : String(+v.toPrecision(6));
  yt.forEach(t => {
    g.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t === 0 ? C.axis : C.grid, 'stroke-width': 1 }));
    g.append(s('text', { x: m.l - 9, y: Y(t) + 4.5, 'text-anchor': 'end', class: 'tick' }, tick(t)));
  });
  xt.forEach(t => {
    g.append(s('line', { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: t === 0 ? C.axis : C.grid, 'stroke-width': 1 }));
    g.append(s('text', { x: X(t), y: H - m.b + 19, 'text-anchor': 'middle', class: 'tick' }, tick(t)));
  });
  g.append(s('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, opts.xTitle || 'x'));
  g.append(s('text', { x: 15, y: (m.t + H - m.b) / 2, transform: `rotate(-90 15 ${(m.t + H - m.b) / 2})`, 'text-anchor': 'middle', class: 'axis-title' }, opts.yTitle || 'y'));

  const plot = s('g', { 'clip-path': `url(#${clipId})` });
  g.append(plot);
  const order = [...series.filter(sr => sr.ref), ...series.filter(sr => !sr.ref)];
  for (const sr of order) {
    const d = sr.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    plot.append(s('path', { d, fill: 'none', stroke: sr.color || (sr.ref ? REF_COLOR : SERIES_COLORS[0]),
      'stroke-width': sr.width || 2, 'stroke-dasharray': sr.dash || null, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      opacity: sr.opacity ?? 1 }));
  }
  for (const sr of order) for (const mk of sr.markers || []) {
    if (!Number.isFinite(mk.x) || !Number.isFinite(mk.y)) continue;
    const col = sr.color || REF_COLOR, cx = X(mk.x), cy = Y(mk.y);
    let el;
    if (mk.shape === 'ring') el = s('circle', { cx, cy, r: 5, fill: C.surface, stroke: col, 'stroke-width': 2 });
    else if (mk.shape === 'diamond') el = s('rect', { x: cx - 5, y: cy - 5, width: 10, height: 10, transform: `rotate(45 ${cx} ${cy})`, fill: col, stroke: C.surface, 'stroke-width': 2 });
    else if (mk.shape === 'triangle') el = s('path', { d: `M${cx},${cy + 6} l-6,-10 h12 z`, fill: col, stroke: C.surface, 'stroke-width': 1.5 });
    else el = s('circle', { cx, cy, r: 5, fill: col, stroke: C.surface, 'stroke-width': 2 });
    plot.append(el);
    if (mk.label) el.addEventListener('mousemove', e => showTip(e, esc(mk.label)));
  }
  g.append(s('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: C.axis, 'stroke-width': 1 }));

  // crosshair: snaps to the pointer's x, one tooltip for every curve there
  const hair = s('line', { y1: m.t, y2: H - m.b, stroke: C.ink2, 'stroke-width': 1, visibility: 'hidden', 'pointer-events': 'none' });
  const dots = s('g', { 'pointer-events': 'none' });
  g.append(hair, dots);
  const overlay = s('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent', style: 'cursor:crosshair' });
  g.append(overlay);
  const xu = opts.xUnit ? ` ${opts.xUnit}` : '', yu = opts.yUnit ? ` ${opts.yUnit}` : '';
  overlay.addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) * W / r.width;
    const x = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
    hair.setAttribute('x1', px); hair.setAttribute('x2', px); hair.setAttribute('visibility', 'visible');
    while (dots.firstChild) dots.firstChild.remove();
    const rows = [];
    for (const sr of order) {
      const y = valueAt(sr.points, x);
      if (y === null) continue;
      const col = sr.color || REF_COLOR;
      dots.append(s('circle', { cx: px, cy: Y(y), r: 4, fill: col, stroke: C.surface, 'stroke-width': 2 }));
      rows.push(`<div style="display:flex;align-items:center;gap:8px"><span style="display:inline-block;width:14px;border-top:2px ${sr.dash ? 'dashed' : 'solid'} ${col}"></span><b>${esc(fmt(y, 4))}${esc(yu)}</b><span class="t-muted">${esc(sr.label)}</span></div>`);
    }
    showTip(e, `<div class="t-muted">${esc(fmt(x, 4))}${esc(xu)}</div>${rows.join('') || '<span class="t-muted">no curve here</span>'}`);
  });
  overlay.addEventListener('mouseleave', () => { hair.setAttribute('visibility', 'hidden'); while (dots.firstChild) dots.firstChild.remove(); hideTip(); });
  return svg;
}

/** Legend with line keys (mirrors the marks), text in ink. */
export function xyLegend(series) {
  const el = document.createElement('div');
  el.className = 'legend';
  for (const sr of series) {
    const key = document.createElement('span');
    key.className = 'key';
    const ln = document.createElement('span');
    ln.className = `ln${sr.dash ? ' dash' : ''}`;
    ln.style.borderTopColor = sr.color || REF_COLOR;
    if (sr.width) ln.style.borderTopWidth = `${Math.max(2, Math.round(sr.width))}px`;
    key.append(ln, document.createTextNode(sr.label));
    el.append(key);
  }
  return el;
}
