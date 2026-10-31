// Run benchmarks: select many problems, queue one convergence study each (through the server's study
// worker pool), follow live progress, and read / download the results table.
import { api, postJSON, withToken, h, clear, fmt, fmtTime, fmtInt, loading, errorBox, statusChip, runStatusChip, timeAgo, toast } from '../api.js';

const sel = new Set();
const opts = { backend: 'skfem', order: 2, nested: false, levels: null };
let search = '';

export async function render(el, args) {
  if (args.length) return renderBatch(el, args[0]);
  return renderNew(el);
}

// ------------------------------------------------------------------ new batch
async function renderNew(el) {
  el.append(loading());
  let probs, health, batches;
  try {
    [probs, health, batches] = await Promise.all([api('/api/problems'), api('/api/health').catch(() => null), api('/api/batches?limit=20')]);
  } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  const all = probs.problems;
  const workers = batches.study_workers;
  el.append(h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Run benchmarks'),
    h('p', { class: 'lede' }, `Queue a convergence study for each selected benchmark. The server runs ${workers} at a time; each study meshes the reference setup at several sizes, solves, and checks the result against the reference with Richardson extrapolation and the GCI.`))));
  if (!all.length) { el.append(h('div', { class: 'card empty' }, h('h2', {}, 'No problems registered yet'))); return; }
  for (const id of [...sel]) if (!all.some(p => p.id === id)) sel.delete(id);

  // ---- selection helpers
  const tagCounts = Object.entries(probs.tags).sort((a, b) => b[1] - a[1]);
  const families = [...new Set(all.map(p => p.family).filter(Boolean))].sort();
  const quick = h('div', { class: 'chips' });
  const tableBody = h('tbody');
  const countEl = h('span', { class: 'secondary' });
  const runBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => start() });
  const errEl = h('div');
  const allChk = h('input', { type: 'checkbox', title: 'select all visible', onchange: e => { visible().forEach(p => e.target.checked ? sel.add(p.id) : sel.delete(p.id)); redraw(); } });
  const visible = () => {
    const q = search.trim().toLowerCase();
    return all.filter(p => !q || `${p.id} ${p.title} ${p.family} ${p.tags.join(' ')}`.toLowerCase().includes(q));
  };
  const toggleGroup = ids => {
    const allIn = ids.every(id => sel.has(id));
    ids.forEach(id => allIn ? sel.delete(id) : sel.add(id));
    redraw();
  };
  const drawQuick = () => {
    clear(quick);
    const withRefs = all.filter(p => p.n_references > 0).map(p => p.id);
    quick.append(...[
      h('button', { class: `chip${withRefs.length && withRefs.every(id => sel.has(id)) ? ' on' : ''}`, onclick: () => toggleGroup(withRefs) }, 'all with references', h('span', { class: 'count' }, withRefs.length)),
      h('button', { class: 'chip', onclick: () => { all.forEach(p => sel.add(p.id)); redraw(); } }, 'all', h('span', { class: 'count' }, all.length)),
      h('button', { class: 'chip', onclick: () => { sel.clear(); redraw(); } }, 'clear'),
      h('span', { class: 'secondary small', style: { alignSelf: 'center', margin: '0 4px 0 10px' } }, 'by tag:'),
      tagCounts.slice(0, 14).map(([t, n]) => {
        const ids = all.filter(p => p.tags.includes(t)).map(p => p.id);
        return h('button', { class: `chip${ids.every(id => sel.has(id)) ? ' on' : ''}`, onclick: () => toggleGroup(ids) }, t.replace(/_/g, ' '), h('span', { class: 'count' }, n));
      }),
      h('select', { style: { fontSize: '14px', padding: '3px 8px' }, onchange: e => { if (e.target.value) toggleGroup(all.filter(p => p.family === e.target.value).map(p => p.id)); e.target.value = ''; } },
        h('option', { value: '' }, 'by family…'), families.map(f => h('option', { value: f }, `${f} (${all.filter(p => p.family === f).length})`)))].flat());
  };
  const drawTable = () => {
    clear(tableBody);
    const vis = visible();
    allChk.checked = vis.length > 0 && vis.every(p => sel.has(p.id));
    for (const p of vis) {
      const chk = h('input', { type: 'checkbox', checked: sel.has(p.id), onchange: e => { e.target.checked ? sel.add(p.id) : sel.delete(p.id); redraw(); } });
      tableBody.append(h('tr', { class: 'click', onclick: e => { if (e.target !== chk && e.target.tagName !== 'A') { chk.checked = !chk.checked; chk.dispatchEvent(new Event('change')); } } },
        h('td', {}, chk),
        h('td', {}, h('div', { style: { fontWeight: 560 } }, p.title), h('a', { class: 'mono small', href: `#/benchmarks/${encodeURIComponent(p.id)}` }, p.id)),
        h('td', {}, `${p.dim}D`),
        h('td', { class: 'small' }, `${p.physics} · ${p.formulation.replace(/_/g, ' ')}`),
        h('td', { class: 'num' }, p.n_references || h('span', { class: 'muted' }, '0')),
        h('td', { class: 'small' }, (p.expected_behavior || '').replace(/_/g, ' ')),
        h('td', { class: 'num small' }, p.n_studies || '')));
    }
  };
  const redraw = () => {
    drawQuick();
    drawTable();
    countEl.textContent = `${sel.size} selected`;
    runBtn.textContent = sel.size ? `Run ${sel.size} benchmark${sel.size === 1 ? '' : 's'}` : 'Select problems';
    runBtn.disabled = !sel.size;
  };

  // ---- settings
  const seg = (key, options) => {
    const box = h('div', { class: 'seg' });
    const draw = () => { clear(box); options.forEach(([v, label, dis, title]) => box.append(h('button', { type: 'button', class: opts[key] === v ? 'on' : '', disabled: !!dis, title, onclick: () => { opts[key] = v; draw(); } }, label))); };
    draw();
    return box;
  };
  const b = (health && health.backends) || {};
  const nestedChk = h('input', { type: 'checkbox', checked: opts.nested, onchange: e => { opts.nested = e.target.checked; } });

  async function start() {
    clear(errEl);
    runBtn.disabled = true;
    try {
      const res = await postJSON('/api/batches', { problem_ids: [...sel], backend: opts.backend, order: opts.order, nested: opts.nested, levels: opts.levels });
      location.hash = `#/batch/${res.batch_id}`;
    } catch (e) { errEl.append(errorBox(e)); runBtn.disabled = false; }
  }

  el.append(h('div', { class: 'card stack' },
    h('div', { class: 'card-head' }, h('h2', {}, '1 · Select benchmarks'), countEl),
    quick,
    h('input', { type: 'search', placeholder: 'Filter the table…', value: search, style: { width: '340px' }, oninput: e => { search = e.target.value; drawTable(); } }),
    h('div', { style: { maxHeight: '440px', overflow: 'auto', border: '1px solid var(--grid)', borderRadius: '8px' } },
      h('table', { class: 'data' }, h('thead', { style: { position: 'sticky', top: 0, background: 'var(--surface)', zIndex: 1 } },
        h('tr', {}, h('th', {}, allChk), h('th', {}, 'problem'), h('th', {}, 'dim'), h('th', {}, 'analysis'), h('th', { class: 'num' }, 'refs'), h('th', {}, 'expected'), h('th', { class: 'num' }, 'studies'))), tableBody))));
  el.append(h('div', { class: 'card stack', style: { marginTop: '20px' } },
    h('h2', {}, '2 · Settings'),
    h('div', { class: 'row', style: { gap: '34px' } },
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Backend'),
        seg('backend', [['skfem', 'skfem (fixed solver)', b.skfem && !b.skfem.ok], ['ccx', 'ccx (CalculiX)', b.ccx && !b.ccx.ok]])),
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Element order'), seg('order', [[1, 'P1'], [2, 'P2']])),
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Meshes per study'),
        seg('levels', [[null, 'problem default'], [3, '3'], [4, '4'], [5, '5']])),
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Refinement'),
        h('label', { class: 'check', title: 'Mesh once at the coarsest size and split every element (ratio 2 per level). Cleaner observed orders; each 3D split multiplies the cell count by 8.' }, nestedChk, 'nested (split the coarse mesh)'))),
    h('p', { class: 'small secondary', style: { margin: 0 } }, `Independent remeshing (default) is what agents do; nested refinement gives cleaner observed orders. With "problem default" each problem uses its own reference sizes. Studies run ${workers} at a time in worker processes.`),
    h('div', { class: 'row', style: { alignItems: 'center' } }, runBtn), errEl));
  redraw();

  el.append(h('div', { class: 'card', style: { marginTop: '20px' } }, h('h2', {}, 'Recent batches'),
    batches.batches.length ? h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'batch'), h('th', {}, 'settings'), h('th', { class: 'num' }, 'problems'), h('th', {}, 'status'), h('th', {}, 'outcomes'))),
      h('tbody', {}, batches.batches.map(bt => h('tr', { class: 'click', onclick: () => { location.hash = `#/batch/${bt.batch_id}`; } },
        h('td', {}, h('div', { class: 'mono small' }, bt.batch_id), h('div', { class: 'small muted' }, timeAgo(bt.created_at))),
        h('td', { class: 'small' }, settingsText(bt.settings)),
        h('td', { class: 'num' }, `${bt.n_done} / ${bt.n_problems}`),
        h('td', {}, runStatusChip(bt.status === 'finished' ? 'succeeded' : bt.status)),
        h('td', {}, outcomeChips(bt.outcomes))))))
      : h('p', { class: 'muted' }, 'No batches yet.')));
}

function settingsText(s) {
  if (!s) return '';
  return `${s.backend} · P${s.order} · ${s.levels ? `${s.levels} meshes` : 'default meshes'}${s.nested ? ' · nested' : ''}`;
}

const OUT = { pass: ['good', 'pass'], fail: ['critical', 'fail'], divergence_detected: ['good', 'divergence detected'],
              divergence_missed: ['critical', 'divergence missed'], no_reference: ['info', 'no reference'] };

function outcomeChips(o) {
  return h('div', { class: 'chips' }, Object.entries(o || {}).map(([k, n]) => h('span', { class: 'chip' }, h('span', { class: `dot ${(OUT[k] || ['info'])[0]}` }), `${n} ${(OUT[k] || [0, k])[1]}`)));
}

// ------------------------------------------------------------------ batch view
async function renderBatch(el, bid) {
  el.append(loading('Loading batch…'));
  let timer = null, alive = true, first = true;
  const head = h('div'), tiles = h('div'), prog = h('div'), results = h('div');
  const tick = async () => {
    let d;
    try { d = await api(`/api/batches/${encodeURIComponent(bid)}`); } catch (e) { clear(el).append(errorBox(e)); return; }
    if (!alive) return;
    if (first) { clear(el).append(head, tiles, prog, results); first = false; }
    const skew = Date.now() / 1000 - d.server_time;
    drawHead(head, d);
    drawTiles(tiles, d);
    drawProgress(prog, d, skew);
    drawResults(results, d);
    if (d.status === 'queued' || d.status === 'running') timer = setTimeout(tick, 1500);
  };
  await tick();
  return () => { alive = false; clearTimeout(timer); };
}

function drawHead(el, d) {
  const running = d.status === 'queued' || d.status === 'running';
  clear(el).append(h('div', { class: 'page-head' },
    h('div', {},
      h('div', { class: 'secondary small' }, h('a', { href: '#/batch' }, '← Run benchmarks'), ' · ', h('a', { href: '#/runs?tab=batches' }, 'all batches')),
      h('h1', {}, d.name || `Benchmark batch · ${d.n_problems} problem${d.n_problems === 1 ? '' : 's'}`),
      h('div', { class: 'meta-line', style: { display: 'flex', gap: '18px', alignItems: 'center', flexWrap: 'wrap' } },
        runStatusChip(d.status === 'finished' ? 'succeeded' : d.status),
        h('span', {}, `${settingsText(d.settings)} · ${d.settings.study_workers} at a time · started ${timeAgo(d.created_at)}`),
        h('span', { class: 'mono small muted' }, d.batch_id))),
    h('div', { class: 'row', style: { gap: '10px' } },
      running ? h('button', { class: 'btn ghost', onclick: async () => { await postJSON(`/api/batches/${d.batch_id}/cancel`, {}); toast('Cancelled the queued and running studies'); } }, 'Cancel') : null,
      h('a', { class: 'btn', href: withToken(`/api/batches/${d.batch_id}/results.csv`), download: `${d.batch_id}.csv` }, 'Download CSV'),
      h('a', { class: 'btn', href: withToken(`/api/batches/${d.batch_id}/results.json`), download: `${d.batch_id}.json` }, 'JSON'))));
}

function drawTiles(el, d) {
  const o = d.outcomes || {};
  const tile = (label, value, sub, st) => h('div', { class: 'tile' }, h('div', { class: 'label' }, st ? h('span', { class: `dot ${st}`, style: { marginRight: '6px' } }) : null, label), h('div', { class: 'value' }, value), h('div', { class: 'sub' }, sub));
  clear(el).append(h('div', { class: 'tiles', style: { marginBottom: '20px' } },
    tile('Studies finished', `${d.n_done} / ${d.n_problems}`, d.counts.failed ? `${d.counts.failed} failed` : (d.counts.running ? `${d.counts.running} running` : 'all problems'), null),
    tile('QoIs within reference tolerance', o.pass || 0, 'pass', 'good'),
    tile('QoIs outside tolerance', o.fail || 0, 'fail', o.fail ? 'critical' : null),
    tile('Singularities detected', o.divergence_detected || 0, 'divergence, as expected', 'good'),
    o.divergence_missed ? tile('Expected divergence missed', o.divergence_missed, 'check the setup', 'critical') : null,
    tile('No reference value', o.no_reference || 0, 'judged on convergence only', null)));
}

function drawProgress(el, d, skew) {
  const total = d.n_problems || 1;
  const running = d.status === 'queued' || d.status === 'running';
  const now = Date.now() / 1000 - skew;  // server clock
  clear(el).append(h('div', { class: 'card', style: { marginBottom: '20px' } },
    h('div', { class: 'card-head' }, h('h2', {}, running ? h('span', {}, h('span', { class: 'spinner' }), ' Running') : 'Studies'),
      h('span', { class: 'secondary' }, `${d.n_done} of ${d.n_problems} done`)),
    running ? h('div', { class: 'progress', style: { marginBottom: '14px' } }, h('div', { style: { width: `${Math.max(2, 100 * (d.n_done + d.items.reduce((a, it) => a + (it.status === 'running' && it.progress.total ? it.progress.done / it.progress.total : 0), 0)) / total)}%` } })) : null,
    h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'problem'), h('th', {}, 'status'), h('th', {}, 'meshes'), h('th', {}, 'current step'), h('th', { class: 'num' }, 'runtime'), h('th', {}, ''))),
      h('tbody', {}, d.items.map(it => {
        const p = it.progress || {};
        const stageAge = it.status === 'running' && p.stage_started_at ? now - p.stage_started_at : null;
        return h('tr', {},
          h('td', {}, h('div', { style: { fontWeight: 560 } }, it.title), h('div', { class: 'mono small muted' }, it.problem_id)),
          h('td', {}, runStatusChip(it.status)),
          h('td', {}, meshDots(it)),
          h('td', { class: 'small' }, it.error ? h('span', { style: { color: 'var(--critical)' } }, it.error) : (p.message || ''), stageAge !== null && stageAge >= 0 ? h('span', { class: 'muted' }, ` · ${fmtTime(stageAge)}`) : null),
          h('td', { class: 'num' }, fmtTime(it.elapsed)),
          h('td', { class: 'small nowrap' }, h('a', { href: `#/study/${it.study_id}` }, 'study →')));
      })))));
}

function meshDots(it) {
  const lv = it.levels && it.levels.length ? it.levels : Array.from({ length: (it.progress && it.progress.total) || 0 }, () => ({ status: 'pending' }));
  const col = { done: 'var(--good)', meshing: 'var(--accent)', solving: 'var(--accent)', storing: 'var(--accent)', failed: 'var(--critical)', pending: 'var(--grid)' };
  return h('div', { style: { display: 'flex', gap: '4px', alignItems: 'center' } },
    lv.map(l => h('span', { title: `${l.status}${l.n_dofs ? `, ${fmtInt(l.n_dofs)} dofs` : ''}`,
      style: { width: '14px', height: '14px', borderRadius: '4px', background: col[l.status] || 'var(--grid)', opacity: ['meshing', 'solving', 'storing'].includes(l.status) ? 0.65 : 1 } })));
}

const pct = (x, signed = false) => {
  if (x === null || x === undefined || !isFinite(x)) return '\u2013';
  if (Math.abs(x) < 1e-3) return '<0.001%';
  return `${signed && x > 0 ? '+' : ''}${fmt(x, 2)}%`;
};

function drawResults(el, d) {
  const rows = d.rows || [];
  clear(el);
  if (!rows.length) {
    if (d.n_done < d.n_problems) el.append(h('div', { class: 'card muted' }, 'Results appear here as each study finishes.'));
    return;
  }
  const byProblem = new Map();
  rows.forEach(r => { if (!byProblem.has(r.problem_id)) byProblem.set(r.problem_id, []); byProblem.get(r.problem_id).push(r); });
  const body = h('tbody');
  for (const [, rs] of byProblem) {
    rs.forEach((r, i) => {
      const first = i === 0;
      body.append(h('tr', { style: first ? { borderTop: '2px solid var(--axis)' } : {} },
        first ? h('td', { rowspan: rs.length, style: { maxWidth: '270px' } }, h('div', { style: { fontWeight: 600 } }, r.title), h('div', { class: 'mono small muted' }, r.problem_id),
          h('div', { class: 'small muted' }, `${r.n_meshes} meshes · finest ${fmtInt(r.finest_dofs)} dofs`),
          h('div', { class: 'small', style: { marginTop: '4px' } }, h('a', { href: `#/study/${r.study_id}` }, 'convergence plot'), r.finest_run_id ? [' · ', h('a', { href: `#/viewer/${r.finest_run_id}` }, 'field, finest mesh')] : null)) : null,
        h('td', { class: 'mono small' }, r.qoi),
        h('td', { class: 'num' }, r.reference !== null ? fmt(r.reference, 5) : h('span', { class: 'muted' }, '–')),
        h('td', { class: 'num' }, fmt(r.finest, 6)),
        h('td', { class: 'num' }, fmt(r.extrapolated, 6), r.extrapolated !== null && r.extrapolation_reliable === false ? h('div', { class: 'small muted' }, 'unreliable') : null),
        h('td', { class: 'num' }, `${fmt(r.p_observed, 3)} / ${fmt(r.p_theory, 2)}`),
        h('td', { class: 'num' }, pct(r.gci_pct)),
        h('td', { class: 'num' }, r.error_pct !== null ? h('span', {}, pct(r.error_pct, true), h('div', { class: 'small muted' }, r.error_basis)) : '–'),
        h('td', {}, statusChip(r.outcome_status, r.outcome_label), r.outcome === 'no_reference' && r.verdict ? h('div', { class: 'small muted', style: { marginTop: '3px' } }, r.verdict) : null),
        first ? h('td', { class: 'num', rowspan: rs.length }, fmtTime(r.runtime_s)) : null));
    });
  }
  el.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Results'), h('span', { class: 'secondary small' }, 'error vs reference uses the extrapolated value when the extrapolation is reliable, else the finest mesh')),
    h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'problem'), h('th', {}, 'QoI'), h('th', { class: 'num' }, 'reference'), h('th', { class: 'num' }, 'finest mesh'),
        h('th', { class: 'num' }, 'extrapolated'), h('th', { class: 'num' }, 'p obs / theory'), h('th', { class: 'num' }, 'GCI'), h('th', { class: 'num' }, 'error'),
        h('th', {}, 'outcome'), h('th', { class: 'num' }, 'runtime'))), body))));
}
