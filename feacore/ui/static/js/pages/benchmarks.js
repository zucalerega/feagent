// Benchmarks library: tag filter + cards; detail view with statement, 3D geometry preview and the
// evaluator view (references, expectations, reference studies).
import { api, postJSON, h, clear, fmt, fmtPct, loading, errorBox, statusChip, qoiDesc, timeAgo, decode } from '../api.js';

const BEHAVIOR = {
  smooth: ['good', 'smooth'],
  stress_concentration: ['warning', 'stress concentration'],
  singular: ['critical', 'singular'],
  thermal: ['info', 'thermal'],
};

let selected = new Set();
let search = '';

export async function render(el, args) {
  if (args.length) { await renderDetail(el, args[0]); return cleanup; }
  return renderList(el);
}

async function renderList(el) {
  el.append(loading('Loading problems…'));
  let data;
  try { data = await api('/api/problems'); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Benchmarks'),
      h('p', { class: 'lede' }, 'The CAD design problems agents are assigned to. Each has an agent-facing statement and an evaluator-only reference used to judge every submission.')),
    h('a', { class: 'btn primary', href: '#/batch' }, 'Run benchmarks')));
  if (!data.problems.length && !data.total) {
    el.append(h('div', { class: 'card empty-state' }, h('h2', {}, 'No problems registered yet'),
      h('p', {}, 'Problems are auto-discovered from feacore/problems/. They will appear here as soon as they register.'),
      importErrors(data)));
    return;
  }
  const filters = h('div', { class: 'filters' });
  const grid = h('div', { class: 'cards' });
  const count = h('span', { class: 'secondary small' });
  const input = h('input', { type: 'search', placeholder: 'Search title, id or tag…', value: search,
    oninput: e => { search = e.target.value; draw(); } });
  const tagRow = h('div', { class: 'chips' });
  const tags = Object.entries(data.tags).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  let showAll = false;
  const drawTags = () => {
    clear(tagRow);
    const shown = showAll ? tags : tags.filter(([t], i) => i < 10 || selected.has(t));
    shown.forEach(([t, n]) => tagRow.append(h('button', { class: `chip${selected.has(t) ? ' on' : ''}`, onclick: () => { selected.has(t) ? selected.delete(t) : selected.add(t); drawTags(); draw(); } },
      t.replace(/_/g, ' '), h('span', { class: 'count' }, n))));
    if (tags.length > 10) tagRow.append(h('button', { class: 'chip', onclick: () => { showAll = !showAll; drawTags(); } }, showAll ? 'fewer tags' : `+ ${tags.length - 10} more`));
    if (selected.size) tagRow.append(h('button', { class: 'chip', onclick: () => { selected.clear(); drawTags(); draw(); } }, 'clear filters ✕'));
  };
  filters.append(h('div', { class: 'search' }, input, count), tagRow);
  el.append(filters, grid, importErrors(data));
  const draw = () => {
    clear(grid);
    const q = search.trim().toLowerCase();
    const list = data.problems.filter(p => [...selected].every(t => p.tags.includes(t)) &&
      (!q || `${p.id} ${p.title} ${p.family} ${p.tags.join(' ')}`.toLowerCase().includes(q)));
    count.textContent = `${list.length} of ${data.total} problems`;
    if (!list.length) grid.append(h('div', { class: 'muted' }, 'No problem matches these filters.'));
    for (const p of list) grid.append(card(p));
  };
  drawTags();
  draw();
}

function behaviorChip(b) {
  const [st, label] = BEHAVIOR[b] || ['info', (b || '').replace(/_/g, ' ')];
  return statusChip(st, label);
}

function card(p) {
  const [st, label] = BEHAVIOR[p.expected_behavior] || ['info', (p.expected_behavior || '').replace(/_/g, ' ')];
  return h('a', { class: 'card pcard', href: `#/benchmarks/${encodeURIComponent(p.id)}` },
    h('div', { class: 'title' }, p.title || p.id),
    h('div', { class: 'id' }, p.id),
    h('div', { class: 'meta-line' }, `${p.dim}D · ${p.physics} · ${p.formulation.replace(/_/g, ' ')}`),
    h('div', { class: 'foot' },
      h('span', { class: 'nowrap' }, h('span', { class: `dot ${st}`, style: { marginRight: '6px' } }), label),
      h('span', {}, `${p.n_qois} QoI${p.n_qois === 1 ? '' : 's'}`),
      h('span', {}, p.n_references ? `${p.n_references} reference${p.n_references === 1 ? '' : 's'}` : 'no reference'),
      p.n_studies ? h('span', {}, `${p.n_studies} stud${p.n_studies === 1 ? 'y' : 'ies'}`) : null));
}

function importErrors(data) {
  const errs = Object.entries(data.import_errors || {});
  const extra = data.extra_errors || [];
  if (!errs.length && !extra.length) return null;
  return h('details', { class: 'card tight', style: { marginTop: '20px' } },
    h('summary', {}, `${errs.length + extra.length} problem module(s) failed to import`),
    h('pre', { class: 'code' }, [...errs.map(([m, e]) => `${m}: ${e}`), ...extra].join('\n')));
}

// ------------------------------------------------------------------ detail
async function renderDetail(el, id) {
  el.append(loading('Loading problem…'));
  let d;
  try { d = await api(`/api/problems/${encodeURIComponent(id)}`); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  const st = d.statement || {};
  const ev = d.evaluator || {};
  el.append(h('div', { class: 'page-head' },
    h('div', {},
      h('div', { class: 'secondary small' }, h('a', { href: '#/benchmarks' }, '← Benchmarks')),
      h('h1', {}, st.title || id),
      h('div', { class: 'meta-line', style: { display: 'flex', gap: '14px', alignItems: 'center', flexWrap: 'wrap' } },
        h('span', { class: 'mono' }, id), h('span', {}, `${st.dim}D · ${st.physics} · ${(st.formulation || '').replace(/_/g, ' ')}`),
        behaviorChip(ev.expected_behavior)),
      (st.tags || []).length ? h('div', { class: 'small muted', style: { marginTop: '4px' } }, (st.tags || []).map(t => t.replace(/_/g, ' ')).join(' · ')) : null),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', title: 'reference study now: skfem, P2, the problem\u2019s default mesh sizes', onclick: async e => {
        e.target.disabled = true;
        try { const st = await postJSON('/api/studies', { problem_id: id, backend: 'skfem', order: 2 }); location.hash = `#/study/${st.study_id}`; }
        catch (err) { e.target.disabled = false; alert(err.message); }
      } }, 'Run with defaults'),
      h('a', { class: 'btn', href: `#/study?problem=${encodeURIComponent(id)}` }, 'Custom study\u2026'),
      h('a', { class: 'btn ghost', href: `#/consensus/${encodeURIComponent(id)}` }, `Consensus (${d.n_submissions})`))));

  const left = h('div', { class: 'stack' });
  const right = h('div', { class: 'stack' });
  el.append(h('div', { class: 'grid-2' }, left, right));

  left.append(h('div', { class: 'card' }, h('div', { class: 'section-title' }, 'Problem statement (agent-facing)'),
    h('div', { class: 'desc' }, st.description || st.error || '')));
  left.append(h('div', { class: 'card' }, h('div', { class: 'section-title' }, 'Quantities of interest to report'),
    h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'name'), h('th', {}, 'definition'))),
      h('tbody', {}, (st.qois || []).map(q => h('tr', {}, h('td', { class: 'mono' }, q.name), h('td', {}, qoiDesc(q))))))));
  const params = Object.entries(st.params || {});
  if (params.length) left.append(h('div', { class: 'card' }, h('div', { class: 'section-title' }, `Parameters · units ${st.units || ''}`),
    h('table', { class: 'kv' }, h('tbody', {}, params.map(([k, v]) => h('tr', {}, h('th', { class: 'mono' }, k), h('td', {}, typeof v === 'number' ? fmt(v, 6) : JSON.stringify(v))))))));

  // geometry preview
  const vp = h('div', { class: 'viewport' });
  const legend = h('div', { class: 'chips', style: { marginTop: '10px' } });
  const note = h('div', { class: 'overlay-note' }, st.dim === 2 ? 'drag: pan · wheel: zoom' : 'drag: rotate · right-drag: pan · wheel: zoom');
  const msg = h('div', { class: 'center-msg' }, h('span', {}, h('span', { class: 'spinner' }), ' generating geometry preview (gmsh, worker process)…'));
  const wrap = h('div', { class: 'viewport-wrap' }, vp, note, msg);
  right.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('div', { class: 'section-title', style: { margin: 0 } }, 'Geometry (CAD) and named groups'),
    h('span', { class: 'small muted', id: 'geo-stats' }, '')), wrap, legend));

  // evaluator view
  const refs = ev.references || [];
  right.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Evaluator view'), h('span', { class: 'badge' }, 'private: never shown to agents')),
    refs.length ? h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'QoI'), h('th', { class: 'num' }, 'reference'), h('th', {}, 'kind'), h('th', { class: 'num' }, 'tolerance'))),
      h('tbody', {}, refs.map(r => h('tr', {}, h('td', { class: 'mono' }, r.qoi), h('td', { class: 'num' }, fmt(r.value, 6)), h('td', {}, r.kind.replace(/_/g, ' ')), h('td', { class: 'num' }, `\u00b1${fmtPct(r.tolerance, 2)}`)))))
      : h('p', { class: 'muted' }, 'No reference values: this problem is judged on self-consistency (convergence) and agreement between agents.'),
    expectations(ev),
    h('table', { class: 'kv', style: { marginTop: '12px' } }, h('tbody', {},
      h('tr', {}, h('th', {}, 'expected behaviour'), h('td', {}, (ev.expected_behavior || '').replace(/_/g, ' '))),
      h('tr', {}, h('th', {}, 'reference study sizes h'), h('td', {}, Array.isArray(ev.study_sizes) ? ev.study_sizes.map(v => fmt(v, 4)).join(', ') : '–')))),
    refs.length ? h('details', { class: 'more' }, h('summary', {}, 'Sources and notes'),
      h('div', { class: 'stack', style: { marginTop: '10px' } }, refs.map(r => h('div', { class: 'ref-src' }, h('b', { class: 'mono' }, r.qoi), ': ', r.source, r.notes ? ` \u2014 ${r.notes}` : '')))) : null,
    ev.reference_case ? h('details', { class: 'more' }, h('summary', {}, 'Reference setup (case JSON)'), h('pre', { class: 'code' }, JSON.stringify(ev.reference_case, null, 2))) : null));

  // studies
  const studies = [...(ev.validation_studies || []).map(v => ({ ...v, source: 'validation' })), ...(d.studies || [])];
  right.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, 'Convergence studies'),
    h('a', { class: 'btn small', href: `#/study?problem=${encodeURIComponent(id)}` }, 'New study')),
    studies.length ? h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'study'), h('th', {}, 'backend'), h('th', {}, 'status'), h('th', {}, 'QoI verdicts'))),
      h('tbody', {}, studies.map(s => h('tr', { class: s.source === 'validation' ? '' : 'click', onclick: s.source === 'validation' ? null : () => { location.hash = `#/study/${s.study_id}`; } },
        h('td', {}, h('div', { class: 'mono small' }, s.study_id), h('div', { class: 'small muted' }, s.source === 'validation' ? 'exported validation study' : timeAgo(s.created_at))),
        h('td', {}, `${s.backend || '?'} · P${s.order || '?'}`),
        h('td', {}, s.source === 'validation' ? statusChip(s.passed ? 'good' : s.passed === false ? 'critical' : 'info', s.passed ? 'passed' : s.passed === false ? 'failed' : 'n/a') : statusChip(s.status === 'succeeded' ? (s.passed === false ? 'serious' : 'good') : s.status === 'failed' ? 'critical' : 'running', s.status)),
        h('td', {}, h('div', { class: 'chips' }, Object.entries(s.verdicts || s.interpretation || {}).map(([q, v]) => h('span', { class: 'chip', title: v.label }, h('span', { class: `dot ${v.status}` }), q))))))))
      : h('p', { class: 'muted' }, 'No studies yet. Run one to validate the fixed solver on this problem.')));

  loadGeometry(id, vp, legend, msg, st.dim);
}

function expectations(ev) {
  const e = Object.entries(ev.expectations || {});
  if (!e.length) return null;
  return h('div', { class: 'small secondary', style: { marginTop: '10px' } }, 'Expected convergence: ',
    e.map(([q, v], i) => [i ? ', ' : '', h('span', { class: 'mono' }, q), ` → ${v}`]));
}

let viewer = null;
async function loadGeometry(id, vp, legendEl, msg, dim) {
  let g;
  try {
    g = await api(`/api/problems/${encodeURIComponent(id)}/geometry`);
  } catch (e) {
    msg.textContent = `Geometry preview unavailable: ${e.message}`;
    return;
  }
  if (!document.body.contains(vp)) return;
  const { MeshView, webglAvailable } = await import('../mesh3d.js');
  if (!webglAvailable()) { msg.textContent = 'WebGL is not available in this browser.'; return; }
  msg.remove();
  if (viewer) viewer.dispose();
  viewer = new MeshView(vp, { dim: g.dim, margin: 1.1 });
  const legend = viewer.setGeometry({
    dim: g.dim, points: decode(g.points), triangles: decode(g.triangles), triEntity: decode(g.tri_entity),
    lines: decode(g.lines), lineEntity: decode(g.line_entity), groups: g.groups, entityGroups: g.entity_groups,
  });
  viewer.fit('iso');
  const stats = document.getElementById('geo-stats');
  if (stats) stats.textContent = `${g.n_triangles.toLocaleString()} preview triangles · ${g.groups.length} physical groups`;
  const dimName = ['point', 'curve', 'surface', 'volume'];
  let active = null;
  const items = legend.map(l => h('span', { class: 'legend-item', title: 'click to highlight', onclick: () => {
    active = active === l.index ? null : l.index;
    viewer.setHighlight(active);
    items.forEach((it, k) => it.classList.toggle('off', active !== null && legend[k].index !== active));
  } }, h('span', { class: 'sw', style: { background: l.color } }), l.name, h('span', { class: 'gdim' }, dimName[l.dim])));
  clear(legendEl).append(...items);
}

export function cleanup() { if (viewer) { viewer.dispose(); viewer = null; } }
