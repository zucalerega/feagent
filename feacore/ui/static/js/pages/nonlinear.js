// Nonlinear: CalculiX analyses past yield of the printed parts (elastoplastic ramps, the agents' frozen scoring
// ramp, crush to fracture), their force-displacement curves against the physical test, and their pictures and
// videos. #/nonlinear?tab=runs|analyses|gallery; #/nonlinear/run/<ref> shows one run.
// Server: feacore/ui/nonlinear_api.py (runs, curves, gallery), feacore/ui/tools/nonlinear.py (the tools),
// feacore/ui/testdata_api.py (the tests to compare with).
import { api, h, clear, loading, errorBox, fmt, timeAgo, statusChip, qs, toast } from '../api.js';
import { toolForm, taskTable, taskChip } from '../tasks.js';
import { xyChart, xyLegend, valueAt, SERIES_COLORS, REF_COLOR } from '../xychart.js';
import { ensureStyles, mediaGrid, fileList, fileUrl } from '../media.js';

const TABS = [['runs', 'Runs'], ['analyses', 'Run an analysis'], ['gallery', 'Gallery']];
const BUILTIN_TEST = 'cload-compression-1';
const KIND = { crush: 'crush to fracture', ramp: 'elastoplastic ramp', score: 'scoring ramp' };
const RUN_STATUS = { finished: ['good', 'finished'], running: ['running', 'running'], stopped: ['warning', 'stopped'],
                     failed: ['critical', 'failed'], queued: ['info', 'queued'], cancelled: ['warning', 'cancelled'],
                     interrupted: ['warning', 'interrupted'], unreadable: ['critical', 'unreadable'] };
const MAX_SERIES = SERIES_COLORS.length;

export const runLink = ref => `#/nonlinear/run/${encodeURIComponent(ref)}`;
// empty an element and append the parts (the DOM's append would write null as the text "null")
const put = (el, ...kids) => { clear(el).append(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false)); return el; };
const runChip = st => { const [c, l] = RUN_STATUS[st] || ['info', st || 'unknown']; return statusChip(c, l); };
const pct = x => (x === null || x === undefined || !isFinite(x)) ? '–' : `${(100 * x).toFixed(x < 0.1 ? 1 : 0)} %`;
const kN = x => (x === null || x === undefined || !isFinite(x)) ? '–' : `${(x / 1000).toFixed(2)} kN`;
const mm = x => (x === null || x === undefined || !isFinite(x)) ? '–' : `${fmt(x, 3)} mm`;

// ------------------------------------------------------------------ view settings (per browser)
const VIEW_KEY = 'feacore_nonlinear_view';
function loadView() {
  let v = {};
  try { v = JSON.parse(localStorage.getItem(VIEW_KEY) || '{}') || {}; } catch { v = {}; }
  return { mode: v.mode === 'part' ? 'part' : 'cross', test: v.test ?? BUILTIN_TEST, delta: v.delta, comp: v.comp,
           sel: Array.isArray(v.sel) ? v.sel : null };
}
function saveView(v) { try { localStorage.setItem(VIEW_KEY, JSON.stringify(v)); } catch { /* private mode */ } }

// ------------------------------------------------------------------ tests to compare with
let testsCache = null;
async function testList() {
  if (!testsCache) testsCache = api('/api/testdata').then(d => d.tests.filter(t => !t.error)).catch(e => { testsCache = null; throw e; });
  return testsCache;
}
const testCurves = new Map();
async function testCurve(id) {
  if (!id) return null;
  if (!testCurves.has(id)) {
    const p = id === BUILTIN_TEST
      ? api('/api/nonlinear/test').then(t => ({ id, label: 'Instron test (C-bracket)', curve: t.curve, checkpoints: t.checkpoints_mm,
                                                checkpointsN: t.checkpoints_N, delta: t.delta_mm, comp: t.compliance_mm_per_N, mapping: t.mapping }))
      : api(`/api/testdata/item?id=${encodeURIComponent(id)}`).then(t => ({ id, label: `test ${t.test_id || id}`, curve: t.curve }));
    testCurves.set(id, p.catch(e => { testCurves.delete(id); throw e; }));
  }
  return testCurves.get(id);
}

// ------------------------------------------------------------------ curves -> chart series
function simPoints(c, view) {
  const F = c.F || [], u = c.u || [];
  if (view.mode === 'cross') {
    if (c.u_cross && c.u_cross.length === F.length) return c.u_cross.map((x, i) => [x, F[i]]);
    return u.map((x, i) => [x > 0 ? x + view.delta + view.comp * F[i] : 0, F[i]]);
  }
  return u.map((x, i) => [x, F[i]]);
}

function testPoints(t, view) {
  if (view.mode === 'cross') return t.curve;
  return t.curve.map(([x, F]) => [x - view.delta - view.comp * F, F]).filter(p => p[0] >= 0);
}

/** Mean |F - F_test| / F_test at the test's checkpoints (crosshead travel); unreached checkpoints count 100 %. */
function checkpointError(points, t) {
  if (!t || !t.checkpoints || points.length < 2) return null;
  let run = -Infinity;
  const mono = points.map(([x, y]) => { run = Math.max(run, x); return [run, y]; });
  const last = mono[mono.length - 1][0];
  const errs = t.checkpoints.map((ck, i) => ck > last ? 1 : Math.abs(valueAt(mono, ck) - t.checkpointsN[i]) / t.checkpointsN[i]);
  return errs.reduce((a, b) => a + b, 0) / errs.length;
}

const colorOf = new Map();          // series id -> palette slot: color follows the curve, not its rank
function assignColors(ids) {
  for (const id of [...colorOf.keys()]) if (!ids.includes(id)) colorOf.delete(id);
  for (const id of ids) {
    if (colorOf.has(id)) continue;
    const used = new Set(colorOf.values());
    const free = SERIES_COLORS.findIndex((_, k) => !used.has(k));
    colorOf.set(id, free >= 0 ? free : 0);
  }
}

function buildSeries(runs, test, view) {
  const items = [];
  for (const r of runs) for (const c of r.curves || []) {
    if (!c.F || !c.F.length) continue;
    items.push({ id: `${r.ref}|${c.name}`, run: r, c });
  }
  const shown = items.slice(0, MAX_SERIES);
  assignColors(shown.map(x => x.id));
  const sim = shown.map(({ id, run, c }) => {
    const pts = simPoints(c, view);
    const running = run.status === 'running';     // checkpoints it has not reached yet would count 100 %
    const err = view.mode === 'cross' && test && test.checkpoints && !running ? checkpointError(pts, test) : null;
    const short = runs.length > 1 ? `${run.short || run.title}: ` : '';
    const markers = [];
    if (c.first_failure) markers.push({ x: view.mode === 'cross' ? c.first_failure[0] + view.delta + view.comp * c.first_failure[1] : c.first_failure[0], y: c.first_failure[1], shape: 'diamond', label: `first failed cell: ${fmt(c.first_failure[1], 4)} N` });
    if (c.first_yield) markers.push({ x: view.mode === 'cross' ? c.first_yield[0] + view.delta + view.comp * c.first_yield[1] : c.first_yield[0], y: c.first_yield[1], shape: 'triangle', label: `first yield: ${fmt(c.first_yield[1], 4)} N` });
    return { id, label: `${short}${c.label}${err !== null ? ` · ${pct(err)} off the test` : running ? ' · running' : ''}`, color: SERIES_COLORS[colorOf.get(id)], points: pts, markers, error: err };
  });
  const out = [];
  if (test) {
    const ck = view.mode === 'cross' && test.checkpoints ? test.checkpoints.map((x, i) => ({ x, y: test.checkpointsN[i], shape: 'ring', label: `checkpoint ${x} mm: ${fmt(test.checkpointsN[i], 4)} N` })) : [];
    out.push({ id: 'test', label: test.label, ref: true, color: REF_COLOR, width: 2.4, points: testPoints(test, view), markers: ck });
  }
  return { series: [...out, ...sim], dropped: items.length - shown.length };
}

/** Chart + legend + controls for a set of runs (refetched by `load`). Returns {el, update(runs)}. */
function curvePanel({ onView = null } = {}) {
  const view = loadView();
  const el = h('div', { class: 'stack' });
  const controls = h('div', { class: 'nl-controls' });
  const chartBox = h('div');
  const note = h('p', { class: 'muted small' });
  el.append(controls, chartBox, note);
  let runs = [], tests = [], defaults = null;

  const drawControls = () => {
    const testSel = h('select', { onchange: () => { view.test = testSel.value; saveView({ ...loadView(), test: view.test }); draw(); } },
      h('option', { value: '' }, '(no test)'),
      tests.map(t => h('option', { value: t.id, selected: t.id === view.test ? true : null }, `${t.test_id || t.id}${t.uploaded ? ' (uploaded)' : ''} · ${t.problem_id}`)));
    const seg = h('div', { class: 'seg' }, [['cross', 'crosshead travel'], ['part', 'part shortening']].map(([k, label]) =>
      h('button', { type: 'button', class: view.mode === k ? 'on' : '', onclick: () => { view.mode = k; saveView({ ...loadView(), mode: k }); drawControls(); draw(); } }, label)));
    const num = (key, step, title) => h('input', { type: 'number', step, value: view[key], title,
      onchange: e => { const v = parseFloat(e.target.value); if (isFinite(v)) { view[key] = v; saveView({ ...loadView(), [key]: v }); draw(); } } });
    put(controls, 
      h('label', {}, 'Compare with', testSel),
      h('label', {}, 'x axis', seg),
      h('label', { title: 'seating gap added to the simulated travel (crosshead mode)' }, 'seating gap (mm)', num('delta', '0.01', 'seating gap')),
      h('label', { title: 'load-train compliance: travel per newton added to the simulated travel' }, 'compliance (mm/N)', num('comp', '1e-6', 'compliance')),
      defaults ? h('button', { class: 'btn small', type: 'button', onclick: () => { view.delta = defaults.delta; view.comp = defaults.comp; saveView({ ...loadView(), delta: view.delta, comp: view.comp }); drawControls(); draw(); } }, 'Frozen values') : null);
  };

  const draw = async () => {
    let test = null;
    try { test = await testCurve(view.test); } catch (e) { toast(String(e)); }
    const { series, dropped } = buildSeries(runs, test, view);
    const xTitle = view.mode === 'cross' ? 'crosshead travel (mm)' : 'platen travel = part shortening (mm)';
    const xMax = test && view.mode === 'cross' && test.id === BUILTIN_TEST ? Math.max(3.2, ...series.filter(s => !s.ref).map(s => Math.max(...s.points.map(p => p[0])))) : null;
    put(chartBox, 
      xyLegend(series),
      xyChart({ series, xTitle, yTitle: 'force (N)', xUnit: 'mm', yUnit: 'N', xMax: xMax && xMax < 12 ? xMax : null, empty: 'Tick runs below to plot their curves' }));
    const bits = [];
    if (view.mode === 'cross') bits.push(`Simulated travel mapped to the crosshead: u + ${fmt(view.delta, 4)} mm + ${fmt(view.comp, 4)} mm/N × F (scoring runs carry their own mapping).`);
    else bits.push('The test is moved back by the same seating gap and compliance, so both show the part’s own shortening.');
    if (test && test.checkpoints && view.mode === 'cross') bits.push(`Error: mean force error at the test's checkpoints ${test.checkpoints.join(', ')} mm, as the agents are scored.`);
    if (dropped > 0) bits.push(`${dropped} more curve${dropped > 1 ? 's' : ''} not shown (at most ${MAX_SERIES}).`);
    note.textContent = bits.join(' ');
    if (onView) onView(view, series);
  };

  (async () => {
    try {
      tests = await testList();
      const t = await testCurve(BUILTIN_TEST);
      defaults = { delta: t.delta, comp: t.comp };
      if (!isFinite(view.delta)) view.delta = t.delta;
      if (!isFinite(view.comp)) view.comp = t.comp;
    } catch (e) {
      if (!isFinite(view.delta)) view.delta = 0;
      if (!isFinite(view.comp)) view.comp = 0;
    }
    drawControls();
    draw();
  })();
  return { el, update: r => { runs = r; if (isFinite(view.delta)) draw(); } };
}

// ------------------------------------------------------------------ page
export async function render(el, args) {
  ensureStyles();
  testsCache = null;              // a test uploaded since the last visit shows up in the selector
  if (args && args[0] === 'run' && args[1]) return runPage(el, args.slice(1).join('/'));
  const q = qs();
  const tab = TABS.some(([k]) => k === q.get('tab')) ? q.get('tab') : 'runs';
  if (q.get('test')) saveView({ ...loadView(), test: q.get('test') });
  const body = h('div');
  el.append(h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Nonlinear'),
      h('p', { class: 'lede' }, 'CalculiX past yield: the printed C-bracket ramped against its real Instron test, the agents’ frozen scoring ramp, and crush-to-fracture runs with element erosion. Curves against the test, videos and stills.')),
    h('div', { class: 'row', style: { gap: '10px' } }, h('a', { class: 'btn', href: '#/testdata' }, 'Test data'), h('a', { class: 'btn primary', href: '#/nonlinear?tab=analyses' }, 'Run an analysis'))),
  h('div', { class: 'toolbar' }, h('div', { class: 'seg' }, TABS.map(([k, label]) =>
    h('button', { class: k === tab ? 'on' : '', onclick: () => { location.hash = `#/nonlinear?tab=${k}`; } }, label)))),
  body);
  return ({ runs: runsTab, analyses: analysesTab, gallery: galleryTab })[tab](body);
}

// ------------------------------------------------------------------ runs tab
async function runsTab(el) {
  el.append(loading('Looking for nonlinear runs…'));
  const panel = curvePanel();
  const tableBox = h('div');
  const details = new Map();        // ref -> {stamp, rec}
  let alive = true, timer = null, list = [];
  const view = loadView();
  let sel = view.sel;

  async function fetchDetail(r) {
    const d = details.get(r.ref);
    if (d && d.stamp === r.modified && r.status !== 'running') return d.rec;
    const rec = await api(`/api/nonlinear/run?ref=${encodeURIComponent(r.ref)}`);
    details.set(r.ref, { stamp: r.modified, rec });
    return rec;
  }

  async function plot() {
    const chosen = list.filter(r => sel.includes(r.ref) && (r.n_curves || 0) > 0);
    const recs = [];
    for (const r of chosen) { try { recs.push(await fetchDetail(r)); } catch (e) { console.warn(e); } }
    if (alive) panel.update(recs);
  }

  function drawTable() {
    const toggle = (ref, on) => {
      sel = on ? [...new Set([...sel, ref])] : sel.filter(x => x !== ref);
      saveView({ ...loadView(), sel });
      plot();
    };
    put(tableBox, list.length ? h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['plot', 'run', 'kind', 'source', 'status', 'curves', 'peak', 'error vs test', 'updated'].map(x => h('th', { class: ['curves', 'peak', 'error vs test'].includes(x) ? 'num' : null }, x)))),
      h('tbody', {}, list.map(r => h('tr', {},
        h('td', {}, (r.n_curves || 0) > 0 ? h('input', { type: 'checkbox', checked: sel.includes(r.ref) ? true : null, onchange: e => toggle(r.ref, e.target.checked), title: 'plot its curves' }) : null),
        h('td', {}, h('a', { href: runLink(r.ref), class: 'strong' }, r.title || r.ref), h('div', { class: 'mono small muted' }, r.ref)),
        h('td', { class: 'small' }, KIND[r.kind] || r.kind),
        h('td', { class: 'small' }, r.source),
        h('td', {}, runChip(r.status), r.kind === 'crush' && r.n_failed ? h('div', { class: 'muted small' }, `${r.n_failed} failed cells${r.separated ? ', separated' : ''}`) : null),
        h('td', { class: 'num' }, r.n_curves ?? 0),
        h('td', { class: 'num' }, kN(r.peak_N)),
        h('td', { class: 'num', title: 'mean force error at the test’s checkpoints, frozen crosshead mapping (best curve of the run)' }, pct(r.status === 'running' ? null : r.best_error)),
        h('td', { class: 'small muted' }, r.modified ? timeAgo(new Date(r.modified * 1000).toISOString()) : ''))))) :
      h('div', { class: 'empty' }, h('h3', {}, 'No nonlinear runs yet'), h('p', { class: 'muted' }, 'Start one under Run an analysis.')));
  }

  async function tick() {
    try {
      const d = await api('/api/nonlinear/runs');
      if (!alive) return;
      list = d.runs;
      if (!sel) {
        const own = list.find(r => r.source !== 'committed' && (r.n_curves || 0) > 0);
        sel = [own ? own.ref : 'physical_tests/abc_00000065_compression/ccx_ramp'];
      }
      drawTable();
      await plot();
      const busy = list.some(r => r.status === 'running');
      timer = setTimeout(tick, busy ? 5000 : 30000);
    } catch (e) { put(tableBox, errorBox(e)); }
  }

  put(el, 
    h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, 'Force against travel'), h('span', { class: 'muted small' }, 'tick runs in the table to overlay them; the black curve is the test')), panel.el),
    h('div', { class: 'card flush' }, tableBox),
    h('p', { class: 'muted small' }, 'Runs are found in the run store (', h('code', {}, '<runs>/nonlinear/'), ', ', h('code', {}, '<runs>/ccx_ramp/'), ') and in the committed results. Dashboard tools write to ', h('code', {}, '<runs>/nonlinear/<tool>/'), '.'));
  await tick();
  return () => { alive = false; clearTimeout(timer); };
}

// ------------------------------------------------------------------ one run
async function runPage(el, ref) {
  el.append(loading());
  const panel = curvePanel();
  let alive = true, timer = null, rec = null;
  const head = h('div'), facts = h('div'), curvesBox = h('div'), mediaBox = h('div'), renderBox = h('div'), filesBox = h('div'), logBox = h('div');

  const drawHead = () => {
    const t = rec.task;
    put(head, h('div', { class: 'page-head' },
      h('div', {},
        h('h1', {}, rec.title || ref, ' ', runChip(rec.status)),
        h('p', { class: 'lede' }, `${KIND[rec.kind] || rec.kind} · ${rec.source || ''}`, rec.part ? ` · part ${rec.part}` : ''),
        h('p', { class: 'muted small' }, h('code', {}, ref), t ? [' · task ', h('a', { href: `#/tasks/${t.id}` }, t.id), ' ', taskChip(t.state)] : null)),
      h('div', { class: 'row', style: { gap: '8px' } },
        rec.kind === 'crush' && rec.status === 'stopped' ? h('a', { class: 'btn', href: `#/nonlinear?tab=analyses&resume=${encodeURIComponent(rec.abs)}&part=${encodeURIComponent(rec.part || '')}` }, 'Resume') : null,
        (rec.curves || []).length ? h('button', { class: 'btn', type: 'button', onclick: () => { const v = loadView(); saveView({ ...v, sel: [...new Set([...(v.sel || []), ref])] }); location.hash = '#/nonlinear?tab=runs'; } }, 'Compare with other runs') : null,
        h('a', { class: 'btn', href: '#/nonlinear?tab=runs' }, 'All runs'))));
  };

  const tile = (label, value, sub) => h('div', { class: 'tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), sub ? h('div', { class: 'sub' }, sub) : null);
  const drawFacts = () => {
    const f = rec.facts || {};
    const items = [];
    if (rec.kind === 'crush') {
      items.push(tile('peak force', kN(f.peak_N), f.travel_at_peak_mm !== undefined ? `at ${mm(f.travel_at_peak_mm)}` : null),
        tile('first failed cell', mm(f.first_failure_mm), f.first_failure_height_fraction != null ? `height fraction ${fmt(f.first_failure_height_fraction, 2)}` : null),
        tile('failed cells', f.n_failed ?? '–', f.n_detached ? `${f.n_detached} detached` : null),
        tile('travel', mm(f.travel_mm), `${f.increments ?? 0} increments`),
        tile('end', f.separated ? 'separated' : (rec.status === 'finished' ? 'stopped' : rec.status), f.stop_reason || null),
        tile('mesh', f.n_nodes ? `${f.n_nodes.toLocaleString()} nodes` : '–', f.n_cells ? `${f.n_cells.toLocaleString()} cells` : null),
        tile('wall time', f.wall_s ? `${(f.wall_s / 60).toFixed(1)} min` : '–', f.threads ? `${f.threads} threads` : null),
        tile('setup', f.platen || '–', `${f.failure || 'peeq'} erosion${f.release ? `, ${f.release} release` : ''}`));
    }
    const errs = (rec.curves || []).filter(c => c.error != null);
    if (errs.length && rec.status !== 'running') {
      const best = errs.reduce((a, b) => (a.error <= b.error ? a : b));
      items.push(tile('error vs the test', pct(best.error), errs.length > 1 ? `best of ${errs.length}: ${best.name}` : 'frozen crosshead mapping'));
    }
    put(facts, items.length ? h('div', { class: 'nl-facts' }, items) : null);
  };

  const drawCurves = () => {
    const cs = rec.curves || [];
    put(curvesBox, cs.length ? h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['curve', 'points', 'peak', 'travel at peak', 'error vs test', ''].map(x => h('th', { class: ['points', 'peak', 'travel at peak', 'error vs test'].includes(x) ? 'num' : null }, x)))),
      h('tbody', {}, cs.map(c => {
        const F = c.F || [], u = c.u || [];
        const ip = F.length ? F.indexOf(Math.max(...F)) : -1;
        return h('tr', {},
          h('td', {}, h('div', { class: 'strong' }, c.label), c.failed ? h('div', { class: 'muted small' }, c.failed) : null),
          h('td', { class: 'num' }, F.length), h('td', { class: 'num' }, ip >= 0 ? kN(F[ip]) : '–'),
          h('td', { class: 'num' }, ip >= 0 ? mm(u[ip]) : '–'), h('td', { class: 'num' }, pct(c.error)),
          h('td', { class: 'small' }, c.has_series && c.abs ? h('button', { class: 'btn small', type: 'button', onclick: () => showRender('nonlinear.render_score', c.abs) }, 'Video') : null));
      }))) : h('p', { class: 'muted' }, rec.status === 'running' ? 'No curve yet: the run writes it as it goes (crush) or at the end (ramps).' : 'No curve.'));
  };

  const drawMedia = () => {
    const own = (rec.media || []).map(m => ({ ...m, title: m.name }));
    const rendered = (rec.renders || []).flatMap(r => (r.media || []).map(m => ({ ...m, title: m.name, caption: `${r.task.title} (${r.task.id})` })));
    // renders still going, and the ones that ended without a picture (their log says why)
    const notes = (rec.renders || []).filter(r => ['queued', 'running'].includes(r.task.state) || (['failed', 'cancelled', 'interrupted'].includes(r.task.state) && !(r.media || []).length));
    put(mediaBox,
      notes.length ? h('p', { class: 'muted small' }, notes.map(r => [taskChip(r.task.state), ' ', h('a', { href: `#/tasks/${r.task.id}` }, r.task.title), r.task.error ? ` (${r.task.error})` : '', '  '])) : null,
      mediaGrid([...rendered, ...own], { empty: 'No pictures or videos yet: render them below.' }));
  };

  let renderShown = null;
  function showRender(tool, runAbs) {
    renderShown = tool;
    const after = () => { toast('Rendering: the result appears under Pictures and videos'); refresh(); };
    put(renderBox, h('div', { class: 'card' }, h('h2', {}, 'Render'),
      toolForm(tool, { defaults: { run: runAbs }, hide: ['run'], submitLabel: 'Render', onStarted: after })));
    renderBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  const drawRenderers = () => {
    if (renderShown) return;
    const tool = { crush: 'nonlinear.render_crush', ramp: 'nonlinear.render_ramp' }[rec.kind];
    if (!tool || !rec.abs || rec.source === 'committed') { clear(renderBox); return; }
    if (rec.kind === 'ramp' && !(rec.files || []).some(f => f.name.endsWith('.json') && f.name !== 'summary.json')) { clear(renderBox); return; }
    const after = () => { toast('Rendering: the result appears under Pictures and videos'); refresh(); };
    put(renderBox, h('div', { class: 'card' }, h('h2', {}, rec.kind === 'crush' ? 'Make a video of this run' : 'Make stills of this ramp'),
      toolForm(tool, { defaults: { run: rec.abs }, hide: ['run'], submitLabel: 'Render', onStarted: after })));
  };

  const drawLog = () => {
    if (!rec.log_tail) { clear(logBox); return; }
    const text = rec.log_tail.slice(rec.log_tail.indexOf('\n') + 1);     // the tail starts mid-line
    const pre = h('pre', { class: 'nl-pre', style: { maxHeight: '260px', overflow: 'auto' } }, text);
    put(logBox, h('div', { class: 'card' }, h('h2', {}, 'Progress (progress.log)'), pre));
    pre.scrollTop = pre.scrollHeight;
  };

  async function refresh() {
    try { rec = await api(`/api/nonlinear/run?ref=${encodeURIComponent(ref)}`); } catch (e) { if (!rec) put(el, errorBox(e)); return; }
    if (!alive) return;
    drawHead(); drawFacts(); drawCurves(); drawMedia(); drawRenderers(); drawLog();
    put(filesBox, fileList(rec.files));
    panel.update([rec]);
    const busy = rec.status === 'running' || (rec.renders || []).some(r => ['queued', 'running'].includes(r.task.state));
    clearTimeout(timer);
    if (busy) timer = setTimeout(refresh, 5000);
  }

  put(el, head,
    facts,
    h('div', { class: 'card' }, h('h2', {}, 'Force against travel'), panel.el),
    h('div', { class: 'card' }, h('h2', {}, 'Curves'), curvesBox),
    h('div', { class: 'card' }, h('h2', {}, 'Pictures and videos'), mediaBox),
    renderBox, logBox,
    h('div', { class: 'card' }, h('h2', {}, 'Files'), filesBox));
  await refresh();
  return () => { alive = false; clearTimeout(timer); };
}

// ------------------------------------------------------------------ analyses tab
async function analysesTab(el) {
  const q = qs();
  const caps = h('div');
  api('/api/nonlinear/capabilities').then(c => {
    const chip = (ok, label, title) => h('span', { class: 'chip', title }, h('span', { class: `dot ${ok ? 'good' : 'critical'}` }), ' ', label);
    put(caps, h('div', { class: 'chips' },
      chip(!!c.ccx, c.ccx ? `CalculiX: ${c.ccx}` : 'no CalculiX', c.notes.ccx),
      chip(!!c.ffmpeg, c.ffmpeg ? `ffmpeg: ${c.ffmpeg}` : 'no ffmpeg', 'videos are encoded with ffmpeg'),
      chip(c.pyvista && c.display !== 'none', `3D renders: ${c.display === 'xvfb' ? 'virtual display (xvfb-run)' : c.display}`, c.notes.render),
      chip(false, 'Omniverse RTX: committed media only', c.notes.rtx),
      h('span', { class: 'chip' }, `${c.cpus} CPUs`)));
  }).catch(e => put(caps, errorBox(e)));

  const goRun = t => { if (t.out) location.hash = runLink(t.out); };
  const card = (title, sub, ...body) => h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, title), sub ? h('span', { class: 'muted small' }, sub) : null), ...body);

  // crush: mesh -> (imperfection) -> crush, with the meshes in the store to pick from
  const crushForm = h('div'), impForm = h('div'), meshBox = h('div');
  let chosen = { mesh_dir: '', part: q.get('part') || 'abc_00000065', resume: q.get('resume') || '' };
  const drawCrushForms = () => {
    const defaults = { part: chosen.part };
    if (chosen.mesh_dir) defaults.mesh_dir = chosen.mesh_dir;
    if (chosen.resume) defaults.resume = chosen.resume;
    put(crushForm, 
      chosen.resume ? h('p', { class: 'small' }, 'Resuming ', h('code', {}, chosen.resume), ' into a new run directory (open More options to change it).') : null,
      toolForm('nonlinear.crush', { defaults, submitLabel: 'Start the crush', onStarted: goRun }));
    put(impForm, toolForm('nonlinear.imperfection', { defaults: { part: chosen.part, ...(chosen.mesh_dir ? { mesh_dir: chosen.mesh_dir } : {}) }, submitLabel: 'Compute the imperfection', onStarted: () => { toast('The moved mesh appears in the list when it is done'); setTimeout(loadMeshes, 4000); } }));
  };
  async function loadMeshes() {
    let d;
    try { d = await api('/api/nonlinear/meshes'); } catch (e) { put(meshBox, errorBox(e)); return; }
    put(meshBox, d.meshes.length ? h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['mesh', 'part', 'nodes', 'cells', 'size', 'quality gates', '', ''].map(x => h('th', {}, x)))),
      h('tbody', {}, d.meshes.map(m => h('tr', { class: chosen.mesh_dir === m.abs ? 'selected' : null },
        h('td', {}, h('div', { class: 'strong' }, m.name), h('div', { class: 'mono small muted' }, m.ref)),
        h('td', { class: 'small' }, m.part || '–'),
        h('td', { class: 'num' }, m.n_nodes ? m.n_nodes.toLocaleString() : '–'),
        h('td', { class: 'num' }, m.n_elements ? m.n_elements.toLocaleString() : '–'),
        h('td', { class: 'small' }, m.h_mm ? `${m.h_mm} mm${m.straight ? ', straight edges' : ''}` : '', m.imperfection ? ` · moved by buckling mode ${m.imperfection.mode} (${m.imperfection.amp_mm} mm)` : ''),
        h('td', { class: 'small', title: 'harness quality gates that did not pass (straight edges facet the bore: the volume gate can fail)' },
          Object.keys(m.gates || {}).length ? Object.entries(m.gates).map(([g, s]) => h('div', {}, h('span', { class: `dot ${s === 'fail' ? 'critical' : 'warning'}` }), ` ${g.replace(/_/g, ' ')} ${s}`)) : h('span', { class: 'muted' }, 'all pass')),
        h('td', {}, m.image ? h('a', { href: fileUrl(m.image), target: '_blank' }, 'picture') : null),
        h('td', {}, h('button', { class: chosen.mesh_dir === m.abs ? 'btn small primary' : 'btn small', type: 'button', onclick: () => { chosen = { ...chosen, mesh_dir: m.abs, part: m.part || chosen.part, resume: '' }; drawCrushForms(); loadMeshes(); toast('Mesh chosen for the crush run'); } }, chosen.mesh_dir === m.abs ? 'chosen' : 'Use'))))))
      : h('p', { class: 'muted' }, 'No crush meshes yet: make one with step 1.'));
  }

  const tasks = taskTable({ group: 'nonlinear' }, { link: t => (t.out && ['nonlinear.crush', 'nonlinear.ramp', 'nonlinear.score'].includes(t.tool) ? runLink(t.out) : null), limit: 30, empty: 'No nonlinear tasks yet.' });
  el.append(
    h('div', { class: 'stack' },
      caps,
      card('Score meshes against the real test', 'about 1 min', toolForm('nonlinear.score', { submitLabel: 'Run the scoring ramp', onStarted: goRun })),
      card('Elastoplastic ramp of the C-bracket test', 'about 30 s at the defaults', toolForm('nonlinear.ramp', { submitLabel: 'Run the ramp', onStarted: goRun })),
      card('Crush to fracture', 'long: about 6 min for a coarse mesh, hours for the study meshes',
        h('p', { class: 'muted' }, 'Three steps: mesh the part, optionally move the mesh by its first buckling mode (the neck-study recipe: tension-only erosion, compression factor 0, switch 0.5:5), then crush it. Meshes made here are listed below; pick one for the crush.'),
        h('h3', {}, '1. Mesh the part'), toolForm('nonlinear.crush_mesh', { submitLabel: 'Make the mesh', onStarted: () => { toast('The mesh appears in the list when it is done'); setTimeout(loadMeshes, 20000); } }),
        h('h3', { style: { marginTop: '18px' } }, 'Meshes in the run store'), meshBox,
        h('details', { style: { marginTop: '12px' } }, h('summary', {}, '2. Buckling imperfection (optional)'), impForm),
        h('h3', { style: { marginTop: '18px' } }, '3. Crush'), crushForm),
      card('Renderers', null, h('p', { class: 'muted' }, 'Open a finished run to render it: crush runs make a 3-panel video (pyvista, through a virtual display here), ramps make stills of plastic strain and tension (matplotlib), and each mesh of a scoring run can make a video of its ramp. The Omniverse RTX videos in the gallery were rendered on the DGX Spark; this server cannot make them.')),
      card('Nonlinear tasks', null, tasks.el)));
  drawCrushForms();
  loadMeshes();
  return () => tasks.stop();
}

// ------------------------------------------------------------------ gallery tab
async function galleryTab(el) {
  el.append(loading('Loading the gallery…'));
  let d;
  try { d = await api('/api/nonlinear/gallery'); } catch (e) { put(el, errorBox(e)); return; }
  const nav = h('div', { class: 'chips', style: { marginBottom: '16px' } }, d.sections.map(s =>
    h('a', { class: 'chip', href: `#/nonlinear?tab=gallery`, onclick: e => { e.preventDefault(); document.getElementById(`nlg-${s.id}`).scrollIntoView({ behavior: 'smooth' }); } }, s.title, h('span', { class: 'count' }, ` ${s.items.length}`))));
  put(el, 
    h('p', { class: 'muted' }, 'Committed figures and videos of the nonlinear work, with their captions. Click one to open it.'),
    nav,
    h('div', { class: 'stack' }, d.sections.map(s => h('div', { class: 'card', id: `nlg-${s.id}` },
      h('div', { class: 'card-head' }, h('h2', {}, s.title), h('code', { class: 'muted small' }, s.source)),
      s.text ? h('p', { class: 'secondary' }, s.text) : null,
      mediaGrid(s.items)))));
}
