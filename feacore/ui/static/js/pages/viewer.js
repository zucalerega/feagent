// Field viewer: boundary surface (3D) or mesh (2D) coloured by a chosen field, colour bar,
// mesh edges, deformed shape with scale. Run picker when no run id is given.
import { api, h, clear, fmt, fmtInt, fmtTime, loading, errorBox, runStatusChip, timeAgo, decode, statusChip } from '../api.js';
import { COLORMAPS, cssGradient, defaultMap } from '../colormap.js';
import { niceTicks } from '../plot.js';

const vstate = { field: null, cmap: 'auto', bands: 0, edges: true, deform: false, scale: null, scaleFor: null, range: 'auto', lo: null, hi: null };

export async function render(el, args) {
  if (!args.length) return renderIndex(el);
  return renderRun(el, args[0]);
}

// ------------------------------------------------------------------ run picker
async function renderIndex(el) {
  el.append(loading('Loading runs…'));
  let data;
  try { data = await api('/api/runs?limit=300'); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Field viewer'),
    h('p', { class: 'lede' }, 'Every stored solve: convergence-study levels, API solves from agents and imported runs. Pick one to see its fields on the mesh.'))));
  const runs = data.runs;
  if (!runs.length) {
    el.append(h('div', { class: 'card empty' }, h('h2', {}, 'No runs yet'),
      h('p', {}, 'Run a convergence study or POST a case + mesh to /api/solve; the results appear here.'),
      h('div', { class: 'row', style: { justifyContent: 'center' } }, h('a', { class: 'btn primary', href: '#/study' }, 'Run a study'), h('a', { class: 'btn', href: '#/solve' }, 'Solve a case'))));
    return;
  }
  const filter = h('input', { type: 'search', placeholder: 'Filter by problem, agent, run id…', style: { width: '360px' } });
  const tbody = h('tbody');
  const draw = () => {
    const q = filter.value.trim().toLowerCase();
    clear(tbody);
    runs.filter(r => !q || `${r.run_id} ${r.problem_id || ''} ${r.agent_id || ''} ${r.backend}`.toLowerCase().includes(q)).forEach(r => {
      const i = r.info || {};
      tbody.append(h('tr', { class: r.fields ? 'click' : 'dim', onclick: () => { if (r.fields) location.hash = `#/viewer/${r.run_id}`; } },
        h('td', {}, h('div', { class: 'mono small' }, r.run_id), h('div', { class: 'small muted' }, timeAgo(r.created_at))),
        h('td', { class: 'mono small' }, r.problem_id || '–'),
        h('td', {}, r.agent_id || (r.study_id ? h('span', { class: 'muted' }, `study, mesh ${(r.level ?? 0) + 1}`) : '–')),
        h('td', {}, `${r.backend}${i.order ? ` · P${i.order}` : ''}`),
        h('td', { class: 'num' }, fmtInt(i.n_cells)), h('td', { class: 'num' }, fmtInt(i.n_dofs)), h('td', { class: 'num' }, fmtTime(i.wall_time)),
        h('td', {}, runStatusChip(r.status)),
        h('td', { class: 'small' }, r.fields ? `${r.fields.length} fields` : h('span', { class: 'muted' }, 'no fields'))));
    });
  };
  filter.addEventListener('input', draw);
  el.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, `${runs.length} runs`), filter),
    h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'run'), h('th', {}, 'problem'), h('th', {}, 'agent / origin'), h('th', {}, 'backend'),
      h('th', { class: 'num' }, 'cells'), h('th', { class: 'num' }, 'dofs'), h('th', { class: 'num' }, 'time'), h('th', {}, 'status'), h('th', {}, 'fields'))), tbody)));
  draw();
}

// ------------------------------------------------------------------ viewer
async function renderRun(el, rid) {
  el.append(loading('Loading run…'));
  let run;
  try { run = await api(`/api/runs/${encodeURIComponent(rid)}`); } catch (e) { clear(el).append(errorBox(e)); return; }
  if (!run.fields) {
    clear(el).className = '';
    el.append(h('div', { class: 'card empty' }, h('h2', {}, 'No field data for this run'),
      h('p', {}, `Status: ${run.status}. ${run.error || 'Runs imported without a mesh, or failed solves, have no fields.'}`),
      h('a', { class: 'btn', href: '#/viewer' }, '← all runs')));
    return;
  }
  const fields = run.fields.available;
  if (!vstate.field || !fields.includes(vstate.field)) vstate.field = ['von_mises', 'T', 'u_mag'].find(f => fields.includes(f)) || fields[0];
  let data;
  try { data = await api(`/api/runs/${encodeURIComponent(rid)}/field?name=${encodeURIComponent(vstate.field)}`); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  const side = h('div', { class: 'viewer-side' });
  const main = h('div', { class: 'viewer-main' });
  el.append(h('div', { class: 'viewer-layout' }, side, main));

  const { MeshView, webglAvailable } = await import('../mesh3d.js');
  if (!webglAvailable()) { main.append(h('div', { class: 'center-msg' }, 'WebGL is not available in this browser.')); return; }
  const view = new MeshView(main, { dim: data.dim, padRight: 190 });
  const geom = { points: decode(data.points), triangles: decode(data.triangles), edges: decode(data.edges), displacement: decode(data.displacement) };
  view.setField(geom);
  let values = decode(data.values), cur = data;

  // overlays
  const cb = h('div', { class: 'colorbar' });
  const info = h('div', { class: 'view-info' });
  const buttons = h('div', { class: 'view-buttons' },
    h('button', { class: 'btn small', onclick: () => view.fit('iso') }, 'Fit'),
    data.dim === 3 ? ['x', 'y', 'z'].map(v => h('button', { class: 'btn small', onclick: () => view.fit(v) }, `${v.toUpperCase()} view`)) : null);
  main.append(cb, info, buttons);

  const range = () => {
    if (vstate.range === 'custom' && isFinite(vstate.lo) && isFinite(vstate.hi) && vstate.hi > vstate.lo) return [vstate.lo, vstate.hi];
    const r = cur.range || [0, 1];
    if (vstate.range === 'symmetric') { const m = Math.max(Math.abs(r[0]), Math.abs(r[1])); return [-m, m]; }
    return r[0] === r[1] ? [r[0] - 1, r[1] + 1] : r;
  };
  const cmapName = () => vstate.cmap === 'auto' ? defaultMap(vstate.field, cur.range) : vstate.cmap;
  const paint = () => {
    const [lo, hi] = range();
    view.setValues(values, lo, hi, cmapName(), vstate.bands);
    drawBar(cb, lo, hi, cmapName(), vstate.bands, vstate.field, cur.stats);
    drawInfo();
  };
  const drawInfo = () => {
    const i = run.info || {};
    clear(info).append(h('div', { class: 't' }, run.problem_id || run.run_id),
      h('div', { class: 'secondary' }, `${run.agent_id ? `agent ${run.agent_id} · ` : ''}${run.study_id ? `study mesh ${(run.level ?? 0) + 1} · ` : ''}${run.backend} · P${i.order} · ${fmtInt(i.n_cells)} cells · ${fmtInt(i.n_dofs)} dofs`),
      h('div', { class: 'secondary small' }, `${fmtInt(cur.n_triangles)} display triangles${vstate.deform && geom.displacement ? ` · deformed ×${fmt(vstate.scale, 3)}` : ''}`));
  };

  // ---- side panel
  const fieldSel = h('select', { onchange: async e => {
    vstate.field = e.target.value;
    vstate.range = vstate.range === 'custom' ? 'auto' : vstate.range;
    cur = await api(`/api/runs/${encodeURIComponent(rid)}/field?name=${encodeURIComponent(vstate.field)}&geometry=false`);
    values = decode(cur.values);
    drawRangeInputs();
    paint();
  } }, fields.map(f => h('option', { value: f, selected: f === vstate.field }, f)));
  const cmapSel = h('select', { onchange: e => { vstate.cmap = e.target.value; paint(); } },
    h('option', { value: 'auto', selected: vstate.cmap === 'auto' }, 'Auto (sequential / diverging)'),
    Object.entries(COLORMAPS).map(([k, v]) => h('option', { value: k, selected: vstate.cmap === k }, v.label)));
  const bandsChk = h('input', { type: 'checkbox', checked: vstate.bands > 0, onchange: e => { vstate.bands = e.target.checked ? 12 : 0; paint(); } });
  const edgesChk = h('input', { type: 'checkbox', checked: vstate.edges, onchange: e => { vstate.edges = e.target.checked; view.setEdgesVisible(vstate.edges); } });
  const rangeSel = h('select', { onchange: e => { vstate.range = e.target.value; drawRangeInputs(); paint(); } },
    [['auto', 'Auto (data min / max)'], ['symmetric', 'Symmetric about 0'], ['custom', 'Custom']].map(([v, l]) => h('option', { value: v, selected: vstate.range === v }, l)));
  const rangeInputs = h('div', { class: 'row', style: { gap: '8px' } });
  const drawRangeInputs = () => {
    clear(rangeInputs);
    if (vstate.range !== 'custom') return;
    const [lo, hi] = cur.range || [0, 1];
    if (!isFinite(vstate.lo)) vstate.lo = lo;
    if (!isFinite(vstate.hi)) vstate.hi = hi;
    const mk = key => h('input', { type: 'number', step: 'any', value: +vstate[key].toPrecision(6), style: { width: '135px' }, onchange: e => { vstate[key] = +e.target.value; paint(); } });
    rangeInputs.append(mk('lo'), mk('hi'));
  };
  if (vstate.scaleFor !== rid || !vstate.scale) { vstate.scale = +view.autoScale().toPrecision(2); vstate.scaleFor = rid; }
  const scaleNum = h('input', { type: 'number', step: 'any', min: 0, value: vstate.scale, onchange: e => { vstate.scale = Math.max(0, +e.target.value || 0); slider.value = toSlider(vstate.scale); applyDeform(); } });
  const auto = view.autoScale();
  const toSlider = s => (s > 0 ? Math.log10(s / auto) : -3);
  const fromSlider = v => +(auto * Math.pow(10, v)).toPrecision(2);
  const slider = h('input', { type: 'range', min: -2, max: 1.5, step: 0.01, value: toSlider(vstate.scale), oninput: e => { vstate.scale = fromSlider(+e.target.value); scaleNum.value = vstate.scale; applyDeform(); } });
  const deformChk = h('input', { type: 'checkbox', checked: vstate.deform, onchange: e => { vstate.deform = e.target.checked; applyDeform(); } });
  const applyDeform = () => { view.setDeformation(vstate.deform ? vstate.scale : 0); drawInfo(); };

  const levels = h('div');
  side.append(
    h('div', {}, h('a', { class: 'small', href: '#/viewer' }, '← all runs'), h('h2', { style: { marginTop: '6px', marginBottom: '2px' } }, 'Field viewer'),
      h('div', { class: 'mono small muted' }, run.run_id), h('div', { style: { marginTop: '6px' } }, runStatusChip(run.status))),
    levels,
    h('div', { class: 'side-section' }, h('div', { class: 'section-title' }, 'Field'), fieldSel),
    h('div', { class: 'side-section' }, h('div', { class: 'section-title' }, 'Colour map'), cmapSel,
      h('label', { class: 'check' }, bandsChk, 'Banded contours (12 levels)')),
    h('div', { class: 'side-section' }, h('div', { class: 'section-title' }, 'Range'), rangeSel, rangeInputs),
    h('div', { class: 'side-section' }, h('div', { class: 'section-title' }, 'Display'),
      h('label', { class: 'check' }, edgesChk, 'Show mesh edges'),
      geom.displacement ? h('label', { class: 'check' }, deformChk, 'Deformed shape') : h('span', { class: 'small muted' }, 'no displacement field (heat problem)')),
    geom.displacement ? h('div', { class: 'side-section' }, h('div', { class: 'section-title' }, 'Deformation scale'), slider,
      h('div', { class: 'scale-row' }, scaleNum, h('button', { class: 'btn small', onclick: () => { vstate.scale = +auto.toPrecision(2); scaleNum.value = vstate.scale; slider.value = toSlider(vstate.scale); applyDeform(); } }, 'auto'))) : null,
    qoiTable(run));
  drawRangeInputs();
  view.setEdgesVisible(vstate.edges);
  applyDeform();
  paint();
  view.fit('iso');
  if (run.study_id) studyLevels(levels, run);
  return () => view.dispose();
}

function qoiTable(run) {
  const q = Object.entries(run.qois || {});
  if (!q.length) return null;
  return h('div', { class: 'side-section' }, h('div', { class: 'section-title' }, 'Quantities of interest'),
    h('table', { class: 'kv' }, h('tbody', {}, q.map(([k, v]) => h('tr', {}, h('th', { class: 'mono small' }, k), h('td', { class: 'small' }, fmt(v, 6)))))));
}

async function studyLevels(el, run) {
  try {
    const st = await api(`/api/studies/${encodeURIComponent(run.study_id)}`);
    const ids = st.run_ids || [];
    if (ids.length < 2) return;
    el.append(h('div', { class: 'side-section' }, h('div', { class: 'section-title' }, 'Study meshes (refinement)'),
      h('div', { class: 'seg' }, ids.map((id, i) => h('button', { class: id === run.run_id ? 'on' : '', onclick: () => { location.hash = `#/viewer/${id}`; } }, `mesh ${i + 1}`))),
      h('a', { class: 'small', href: `#/study/${run.study_id}` }, 'convergence results →')));
  } catch { /* study gone: ignore */ }
}

function drawBar(el, lo, hi, cmap, bands, field, stats) {
  const span = hi - lo || 1;
  const ticks = niceTicks(lo, hi, 6).filter(t => t >= lo - 1e-9 * span && t <= hi + 1e-9 * span);
  const H = 340;
  const ts = h('div', { class: 'ticks' });
  for (const t of ticks) ts.append(h('span', { style: { top: `${H - (t - lo) / span * H}px` } }, fmt(t, 4)));
  clear(el).append(h('div', { class: 'cb-title' }, field),
    h('div', { class: 'cb-body' }, h('div', { class: 'bar', style: { background: cssGradient(cmap, bands) } }), ts),
    stats ? h('div', { class: 'cb-foot' }, h('div', {}, `max ${fmt(stats.max, 5)}`), h('div', {}, `min ${fmt(stats.min, 5)}`),
      stats.n_nan ? h('div', {}, `grey: ${stats.n_nan} vertices without value`) : null) : null);
}
