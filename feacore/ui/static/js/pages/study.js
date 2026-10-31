// Run study: pick problem / backend / order / refinement; live progress; per-QoI convergence plots
// (QoI vs log h with Richardson extrapolation, GCI band and reference), observed vs theoretical order,
// and the verification flags in plain words.
import { api, postJSON, withToken, h, clear, fmt, fmtPct, fmtSigned, fmtTime, fmtInt, loading, errorBox, statusChip,
  runStatusChip, flagList, qs, qoiDesc, timeAgo } from '../api.js';
import { convergenceChart, orderChart } from '../plot.js';

export async function render(el, args) {
  if (args.length) return renderStudy(el, args[0]);
  return renderForm(el);
}

// ------------------------------------------------------------------ form
async function renderForm(el) {
  el.append(loading());
  let probs, health, studies;
  try {
    [probs, health, studies] = await Promise.all([api('/api/problems'), api('/api/health').catch(() => null), api('/api/studies?limit=25')]);
  } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Run a convergence study'),
    h('p', { class: 'lede' }, 'The fixed solver runs the reference setup on a sequence of refined meshes. Richardson extrapolation and the Grid Convergence Index (Celik et al. 2008) give the mesh-independent estimate, its uncertainty and the observed order of convergence.'))));
  if (!probs.problems.length) {
    el.append(h('div', { class: 'card empty' }, h('h2', {}, 'No problems registered yet'), h('p', {}, 'Studies need a problem from the benchmark library.')));
    return;
  }
  const want = qs().get('problem');
  const state = { problem: probs.problems.some(p => p.id === want) ? want : probs.problems[0].id, backend: 'skfem', order: 2, mode: 'default', levels: 3, ratio: 2, sizes: '', nested: false };
  const backends = (health && health.backends) || {};

  const problemSel = h('select', { style: { width: '100%' }, onchange: e => { state.problem = e.target.value; loadDefaults(); } },
    probs.problems.map(p => h('option', { value: p.id, selected: p.id === state.problem }, `${p.title}  [${p.id}]`)));
  const seg = (key, options, onchange) => {
    const box = h('div', { class: 'seg' });
    const draw = () => {
      clear(box);
      options.forEach(([v, label, disabled, title]) => box.append(h('button', { type: 'button', class: state[key] === v ? 'on' : '', disabled: !!disabled, title,
        onclick: () => { state[key] = v; draw(); onchange && onchange(); } }, label)));
    };
    draw();
    return box;
  };
  const backendOpts = ['skfem', 'ccx', 'nn'].map(b => {
    const info = backends[b];
    const ok = !info || info.ok;
    return [b, b + (b === 'skfem' ? ' (fixed)' : b === 'ccx' ? ' (cross-check)' : ' (surrogate)'), !ok, info && !info.ok ? info.error : info && info.doc];
  });
  const defaultsEl = h('div', { class: 'small secondary' }, '');
  const customEl = h('div', { class: 'form-grid', style: { marginTop: '10px' } });
  const sizesPreview = h('div', { class: 'small secondary', style: { marginTop: '8px' } });
  let defaults = null, pdim = 2;
  const preview = () => {
    const sizes = state.mode === 'default' ? null : currentSizes();
    sizesPreview.textContent = sizes ? `Mesh sizes h: ${sizes.map(v => fmt(v, 4)).join(' → ')}` : '';
  };
  const currentSizes = () => {
    if (state.mode === 'default') return defaults;
    if (state.mode === 'explicit') {
      const v = state.sizes.split(/[\s,;]+/).map(Number).filter(x => x > 0);
      return v.length ? v.sort((a, b) => b - a) : null;
    }
    if (!defaults) return null;
    const h0 = Math.max(...defaults);
    return Array.from({ length: state.levels }, (_, k) => h0 / Math.pow(state.ratio, k));
  };
  const drawCustom = () => {
    clear(customEl);
    if (state.mode === 'levels') {
      customEl.append(
        h('label', { class: 'field' }, 'Number of meshes', h('input', { type: 'number', min: 2, max: 8, value: state.levels, oninput: e => { state.levels = Math.max(2, Math.min(8, +e.target.value || 3)); preview(); } })),
        h('label', { class: 'field' }, h('span', {}, 'Refinement ratio r ', h('span', { class: 'hint' }, '(≥ 1.3)')), h('input', { type: 'number', min: 1.3, max: 4, step: 0.1, value: state.ratio, oninput: e => { state.ratio = Math.max(1.05, +e.target.value || 2); preview(); } })));
    } else if (state.mode === 'explicit') {
      customEl.append(h('label', { class: 'field', style: { gridColumn: '1 / -1' } }, 'Mesh sizes h (comma separated)',
        h('input', { type: 'text', value: state.sizes, placeholder: 'e.g. 0.05, 0.025, 0.0125', oninput: e => { state.sizes = e.target.value; preview(); } })));
    }
    preview();
  };
  async function loadDefaults() {
    defaults = null;
    defaultsEl.textContent = 'loading default sizes…';
    try {
      const d = await api(`/api/problems/${encodeURIComponent(state.problem)}?order=${state.order}`);
      pdim = d.statement.dim;
      const s = d.evaluator.study_sizes;
      defaults = Array.isArray(s) ? [...s].sort((a, b) => b - a) : null;
      defaultsEl.textContent = defaults ? `Problem default: ${defaults.length} meshes, h = ${defaults.map(v => fmt(v, 4)).join(', ')} (${pdim}D)` : 'This problem has no default sizes: give explicit sizes.';
      if (state.mode === 'levels') state.ratio = pdim === 2 ? 2 : 1.5;
    } catch (e) { defaultsEl.textContent = `Could not load defaults: ${e.message}`; }
    drawCustom();
  }
  const errEl = h('div');
  const submit = h('button', { class: 'btn primary', type: 'button', onclick: async () => {
    submit.disabled = true;
    clear(errEl);
    try {
      const body = { problem_id: state.problem, backend: state.backend, order: state.order, nested: state.nested };
      const sizes = currentSizes();
      if (state.mode !== 'default') {
        if (!sizes || sizes.length < 2) throw new Error('need at least two mesh sizes (three for a GCI estimate)');
        body.sizes = sizes;
      }
      const st = await postJSON('/api/studies', body);
      location.hash = `#/study/${st.study_id}`;
    } catch (e) { errEl.append(errorBox(e)); submit.disabled = false; }
  } }, 'Run study');

  el.append(h('div', { class: 'card stack' },
    h('label', { class: 'field' }, 'Problem', problemSel),
    h('div', { class: 'row', style: { gap: '34px' } },
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Solver backend'), seg('backend', backendOpts)),
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Element order'), seg('order', [[1, 'P1 linear'], [2, 'P2 quadratic']], loadDefaults)),
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Refinement'), seg('mode', [['default', 'problem default'], ['levels', 'levels × ratio'], ['explicit', 'explicit sizes']], drawCustom))),
    h('div', {}, defaultsEl, customEl, sizesPreview),
    h('label', { class: 'check', title: 'Mesh once at the coarsest size and split every element (ratio 2 per level): cleaner observed orders. Each 3D split multiplies the cell count by 8.' },
      h('input', { type: 'checkbox', onchange: e => { state.nested = e.target.checked; } }), 'Nested refinement (split the coarsest mesh instead of remeshing)'),
    h('div', { class: 'row', style: { alignItems: 'center' } }, submit,
      h('span', { class: 'small muted' }, 'Runs in a background worker process; you can leave the page.')),
    errEl));
  loadDefaults();

  el.append(h('div', { class: 'card', style: { marginTop: '20px' } }, h('h2', {}, 'Recent studies'),
    studies.studies.length ? studyTable(studies.studies) : h('p', { class: 'muted' }, 'No studies yet.')));
}

function studyTable(list) {
  return h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'study'), h('th', {}, 'problem'), h('th', {}, 'backend'), h('th', {}, 'status'), h('th', {}, 'QoI verdicts'))),
    h('tbody', {}, list.map(s => h('tr', { class: 'click', onclick: () => { location.hash = `#/study/${s.study_id}`; } },
      h('td', {}, h('div', { class: 'mono small' }, s.study_id), h('div', { class: 'small muted' }, timeAgo(s.created_at))),
      h('td', { class: 'mono small' }, s.problem_id),
      h('td', {}, `${s.backend} · P${s.order} · ${(s.sizes || []).length} meshes`),
      h('td', {}, runStatusChip(s.status), s.status === 'running' && s.progress ? h('div', { class: 'small muted' }, `${s.progress.done}/${s.progress.total}`) : null),
      h('td', {}, h('div', { class: 'chips' }, Object.entries(s.verdicts || {}).map(([q, v]) => h('span', { class: 'chip', title: v.label }, h('span', { class: `dot ${v.status}` }), q))))))));
}

// ------------------------------------------------------------------ study view
async function renderStudy(el, sid) {
  el.append(loading('Loading study…'));
  let timer = null, alive = true;
  const head = h('div'), prog = h('div'), results = h('div', { class: 'stack' });
  let first = true;
  const tick = async () => {
    let st;
    try { st = await api(`/api/studies/${encodeURIComponent(sid)}`); } catch (e) { clear(el).append(errorBox(e)); return; }
    if (!alive) return;
    if (first) { clear(el).append(head, prog, results); first = false; }
    drawHead(head, st);
    drawProgress(prog, st);
    if (st.status === 'queued' || st.status === 'running') {
      timer = setTimeout(tick, 1000);
    } else {
      drawResults(results, st);
    }
  };
  await tick();
  return () => { alive = false; clearTimeout(timer); };
}

function drawHead(el, st) {
  const passed = st.result && st.result.passed !== null && st.result.passed !== undefined ? st.result.passed : null;
  clear(el).append(h('div', { class: 'page-head' }, h('div', {},
    h('div', { class: 'secondary small' }, h('a', { href: '#/runs' }, '← Runs'), ' · ', h('a', { href: `#/benchmarks/${encodeURIComponent(st.problem_id)}` }, 'problem'),
      st.batch_id ? [' · ', h('a', { href: `#/batch/${st.batch_id}` }, 'batch')] : null),
    h('h1', {}, st.problem_title || st.problem_id),
    h('div', { class: 'meta-line', style: { display: 'flex', gap: '18px', alignItems: 'center', flexWrap: 'wrap' } },
      runStatusChip(st.status),
      passed !== null ? statusChip(passed ? 'good' : 'critical', passed ? 'matches the references' : 'misses a reference') : null,
      h('span', {}, `${st.backend} · P${st.order} · ${(st.sizes || []).length} meshes${st.nested ? ' (nested)' : ''}`),
      h('span', { class: 'mono small muted' }, st.study_id)))));
}

function drawProgress(el, st) {
  const p = st.progress || {};
  const total = p.total || (st.sizes || []).length || 1;
  const done = p.done || 0;
  const running = st.status === 'queued' || st.status === 'running';
  const levels = p.levels && p.levels.length ? p.levels : (st.sizes || []).map((h0, i) => ({ index: i, h: h0, status: 'pending' }));
  clear(el).append(h('div', { class: 'card', style: { marginBottom: '20px' } },
    h('div', { class: 'card-head' }, h('h2', {}, running ? h('span', {}, h('span', { class: 'spinner' }), ' Running') : 'Mesh sequence'),
      h('span', { class: 'secondary' }, st.status === 'queued' ? `queued${st.queue_position ? ` (position ${st.queue_position})` : ''}: ${p.message || ''}` : (p.message || ''),
        running && st.status === 'running' && p.stage_started_at && st.server_time ? h('b', {}, ` \u00b7 ${fmtTime(st.server_time - p.stage_started_at)}`) : null,
        !running && st.elapsed ? ` \u00b7 total ${fmtTime(st.elapsed)}` : null)),
    running ? h('div', { class: 'progress', style: { marginBottom: '14px' } }, h('div', { style: { width: `${Math.max(3, 100 * done / total)}%` } })) : null,
    h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'mesh'), h('th', { class: 'num' }, 'h (nominal)'), h('th', { class: 'num' }, 'h (measured)'), h('th', { class: 'num' }, 'cells'), h('th', { class: 'num' }, 'dofs'), h('th', { class: 'num' }, 'solve time'), h('th', {}, 'status'), h('th', {}, ''))),
      h('tbody', {}, levels.map(l => h('tr', {},
        h('td', {}, `${l.index + 1}`), h('td', { class: 'num' }, fmt(l.h, 4)), h('td', { class: 'num' }, fmt(l.h_rep, 4)),
        h('td', { class: 'num' }, fmtInt(l.n_cells)), h('td', { class: 'num' }, fmtInt(l.n_dofs)), h('td', { class: 'num' }, fmtTime(l.wall_time)),
        h('td', {}, l.status === 'done' ? statusChip('good', 'done') : l.status === 'pending' ? statusChip('info', 'pending') : l.status === 'failed' ? statusChip('critical', 'failed') : h('span', {}, h('span', { class: 'spinner' }), ` ${l.status}`, l.started_at && st.server_time ? h('span', { class: 'muted' }, ` ${fmtTime(st.server_time - l.started_at)}`) : null)),
        h('td', {}, l.run_id ? h('a', { href: `#/viewer/${l.run_id}` }, 'view field →') : null))))),
    st.error ? h('div', { style: { marginTop: '14px' } }, errorBox(st.error),
      st.traceback ? h('details', { class: 'more' }, h('summary', {}, 'Traceback'), h('pre', { class: 'code' }, st.traceback)) : null,
      h('div', { class: 'small', style: { marginTop: '6px' } }, h('a', { href: withToken(`/api/studies/${st.study_id}/log`), target: '_blank' }, 'worker log'))) : null));
}

function drawResults(el, st) {
  clear(el);
  const res = st.result;
  if (!res) return;
  const interp = st.interpretation || {};
  const hrep = res.h_rep || [];
  const runs = res.runs || [];
  for (const q of res.qois || []) {
    const it = interp[q.name] || {};
    el.append(qoiCard(q, it, hrep, runs, st));
  }
}

function qoiCard(q, it, hrep, runs, st) {
  const g = q.gci || null;
  const ref = q.reference;
  const pts = q.values.map((v, i) => ({
    h: hrep[i], value: v,
    tip: `<b>mesh ${i + 1}</b><br>h = ${fmt(hrep[i], 4)}<br>${q.name} = ${fmt(v, 7)}<br><span class="t-muted">${fmtInt(runs[i] && runs[i].info.n_dofs)} dofs · ${fmtInt(runs[i] && runs[i].info.n_cells)} cells</span>`,
  }));
  const fine = q.values[q.values.length - 1];
  const band = it.band_abs !== null && it.band_abs !== undefined ? [fine - it.band_abs, fine + it.band_abs] : null;
  const bandLabel = it.verdict === 'plateau' ? 'fine \u2013 medium difference' : 'GCI band of the finest mesh';
  const yTitle = q.name;
  const chart = convergenceChart({
    points: pts, phiExt: g && g.phi_ext !== null ? g.phi_ext : NaN, p: g ? g.p : NaN, band, extReliable: it.extrapolation_reliable,
    hFine: Math.min(...hrep), reference: ref ? { value: ref.value, tol: ref.tolerance } : null, yTitle,
  });
  const legend = h('div', { class: 'legend' },
    h('span', { class: 'key' }, h('span', { class: 'mk', style: { background: '#2a78d6' } }), 'FE solution per mesh'),
    g && g.phi_ext !== null ? h('span', { class: 'key' }, h('span', { class: 'ln dash', style: { borderColor: '#2a78d6' } }), 'Richardson extrapolation (h→0)') : null,
    band ? h('span', { class: 'key' }, h('span', { class: 'box', style: { background: 'rgba(42,120,214,0.22)' } }), bandLabel) : null,
    ref ? h('span', { class: 'key' }, h('span', { class: 'ln', style: { borderColor: '#0b0b0b' } }), `reference ± ${fmtPct(ref.tolerance, 2)}`) : null);
  // order plot: true error when a reference exists, else the Richardson error estimate
  let ord = null;
  const target = ref ? ref.value : (g && g.phi_ext !== null ? g.phi_ext : null);
  if (target !== null && q.values.length >= 2) {
    const ep = q.values.map((v, i) => ({ h: hrep[i], err: Math.abs(v - target) })).filter(p => p.err > Math.abs(target) * 1e-13);
    if (ep.length >= 2) ord = orderChart({ points: ep, pTheory: q.p_theory, pObs: g ? g.p : NaN, yTitle: ref ? '|error vs reference|' : '|φ − φ_ext|' });
  }
  const left = h('div', {}, legend, chart,
    ord ? h('div', { style: { marginTop: '12px' } }, h('div', { class: 'section-title' }, ref ? 'Error vs reference (log–log): slope = order of convergence' : 'Estimated error (log–log): slope = order of convergence'), ord) : null);

  const stats = h('div', { class: 'stat-row' },
    stat('finest mesh', fmt(fine, 6)),
    g && g.phi_ext !== null ? stat(it.extrapolation_reliable === false ? 'extrapolated (unreliable)' : 'extrapolated (h→0)', fmt(g.phi_ext, 6)) : null,
    g && g.gci_fine !== null ? stat('GCI (fine)', fmtPct(g.gci_fine, 2)) : null);
  const refStats = ref ? h('div', { class: 'stat-row' },
    stat('reference', fmt(ref.value, 6), 'sm'),
    stat('error, finest mesh', fmtSigned(q.err_fine, 2), 'sm'),
    q.err_ext !== null && q.err_ext !== undefined ? stat('error, extrapolated', fmtSigned(q.err_ext, 2), 'sm') : null,
    q.passed !== null && q.passed !== undefined ? h('div', { class: 'stat' }, h('div', { class: 'label' }, 'verdict vs reference'), statusChip(q.passed ? 'good' : 'critical', q.passed ? `within ±${fmtPct(ref.tolerance, 2)}` : `outside ±${fmtPct(ref.tolerance, 2)}`)) : null) : null;
  const right = h('div', {},
    stats, gauge(g ? g.p : null, q.p_theory), refStats,
    h('p', { class: 'secondary', style: { marginTop: '10px' } }, it.summary || ''),
    flagList(it.explanations),
    h('details', { class: 'more' }, h('summary', {}, 'Values per mesh and GCI details'),
      h('table', { class: 'data', style: { marginTop: '6px' } },
        h('thead', {}, h('tr', {}, h('th', {}, 'mesh'), h('th', { class: 'num' }, 'h'), h('th', { class: 'num' }, 'dofs'), h('th', { class: 'num' }, q.name))),
        h('tbody', {}, q.values.map((v, i) => h('tr', {}, h('td', {}, i + 1), h('td', { class: 'num' }, fmt(hrep[i], 4)), h('td', { class: 'num' }, fmtInt(runs[i] && runs[i].info.n_dofs)), h('td', { class: 'num' }, fmt(v, 8)))))),
      g ? h('table', { class: 'kv', style: { marginTop: '10px' } }, h('tbody', {},
        kv('convergence', g.convergence), kv('ratio R = ε₂₁/ε₃₂', fmt(g.R, 4)), kv('refinement ratios', `${fmt(g.r21, 3)}, ${fmt(g.r32, 3)}`),
        kv('relative change fine/medium', fmtPct(g.e_a21, 3)), kv('GCI fine (Fs 1.25)', fmtPct(g.gci_fine, 3)), kv('GCI conservative (Fs 3)', fmtPct(g.gci_fine_conservative, 3)),
        kv('asymptotic ratio (≈ 1)', fmt(g.asymptotic_ratio, 4)), q.fit ? kv(`power-law fit (${q.fit.n} meshes)`, `φ₀ = ${fmt(q.fit.phi0, 6)}, p = ${fmt(q.fit.p, 3)}`) : null)) : null));
  return h('div', { class: 'card' },
    h('div', { class: 'qoi-head' }, h('h3', { class: 'mono' }, q.name), statusChip(it.status, it.label, true)),
    h('div', { class: 'qoi-sub' }, qoiDesc((st.qoi_defs || {})[q.name] || q), q.expectation === 'diverge' ? ' · expected to diverge (singular quantity)' : ''),
    h('div', { class: 'qoi-card' }, left, right));
}

const stat = (label, value, cls = '') => h('div', { class: 'stat' }, h('div', { class: 'label' }, label), h('div', { class: `value ${cls}` }, value));
const kv = (k, v) => h('tr', {}, h('th', {}, k), h('td', {}, v ?? '–'));

function gauge(p, pt) {
  if (p === null || p === undefined || !isFinite(p)) {
    return h('div', { class: 'stat' }, h('div', { class: 'label' }, 'observed order p'), h('div', { class: 'value sm' }, `–  (theory ${fmt(pt, 2)})`));
  }
  const max = Math.max(4, Math.ceil(Math.max(p, pt || 0) + 0.5));
  const pos = v => `${Math.min(100, 100 * v / max)}%`;
  return h('div', { class: 'stat', style: { maxWidth: '420px' } },
    h('div', { class: 'label' }, `observed order p = ${fmt(p, 3)}  vs theory ${pt !== null && pt !== undefined ? fmt(pt, 2) : '–'}`),
    h('div', { class: 'order-gauge' }, h('div', { class: 'track' }),
      pt ? h('div', { class: 'theory', style: { left: pos(pt) }, title: `theory ${pt}` }) : null,
      h('div', { class: 'obs', style: { left: pos(Math.min(p, max)) }, title: `observed ${fmt(p, 3)}` })),
    h('div', { class: 'order-labels' }, h('span', {}, '0'), h('span', {}, `${max}${p > max ? '+' : ''}`)));
}
