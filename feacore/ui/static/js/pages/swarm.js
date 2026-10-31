// Swarm: live view of the ACE agent loop. Each generation's agents per problem (score, QoI vs
// reference, mesh, solves, gates) and the score distribution across generations.
import { api, h, clear, fmt, fmtInt, loading, errorBox, timeAgo, qs } from '../api.js';
import { stripChart } from '../plot.js';

let sel = { loop: null, gen: null };

export async function render(el) {
  const want = qs();
  if (want.get('loop')) sel.loop = want.get('loop');
  if (want.get('gen') !== null) sel.gen = want.get('gen');
  el.append(loading('Loading the swarm…'));
  let alive = true, timer = null, first = true;
  const body = h('div');
  const tick = async () => {
    let d;
    const p = new URLSearchParams();
    if (sel.loop) p.set('loop_id', sel.loop);
    if (sel.gen !== null && sel.gen !== undefined) p.set('generation', sel.gen);
    try { d = await api(`/api/swarm?${p}`); } catch (e) { clear(el).append(errorBox(e)); return; }
    if (!alive) return;
    if (first) { clear(el).append(body); first = false; }
    draw(body, d, () => { clearTimeout(timer); tick(); });
    timer = setTimeout(tick, d.active ? 4000 : 15000);
  };
  await tick();
  return () => { alive = false; clearTimeout(timer); };
}

function draw(el, d, refresh) {
  clear(el);
  const head = h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Agent swarm'),
      h('p', { class: 'lede' }, 'ACE agents set up and mesh the benchmark problems; each finished agent run is scored against the best uniform mesh with the same node budget.')),
    d.loops.length ? h('div', { class: 'row', style: { alignItems: 'center', gap: '14px' } },
      live(d),
      h('select', { onchange: e => { sel.loop = e.target.value; sel.gen = null; refresh(); } },
        d.loops.map(l => h('option', { value: l.loop_id, selected: l.loop_id === d.loop_id }, `loop ${l.loop_id}`)))) : null);
  el.append(head);
  if (!d.loops.length) { el.append(emptyState(d)); return; }

  const loop = d.loops.find(l => l.loop_id === d.loop_id);
  const g = d.generations.find(x => x.generation === d.generation) || {};
  // generation selector
  el.append(h('div', { class: 'toolbar' },
    h('span', { class: 'section-title' }, 'Generation'),
    h('div', { class: 'seg' }, loop.generations.map(x => h('button', { class: x === d.generation ? 'on' : '', onclick: () => { sel.loop = d.loop_id; sel.gen = x; refresh(); } }, String(x)))),
    h('span', { class: 'muted small' }, `${loop.problems.length} problem${loop.problems.length === 1 ? '' : 's'} · ${loop.n_submissions} finished agent runs · ${loop.n_runs} solves`)));

  // score distribution + generation summary
  const chart = stripChart({
    groups: d.generations.map(x => ({
      label: `gen ${x.generation}`, sub: `${x.n_submissions} agent${x.n_submissions === 1 ? '' : 's'}`, median: x.median,
      points: x.points.map(p => ({ y: p.score, muted: p.gated, tip: `<b>${p.ace_run_id}</b> · ${p.problem_id}<br>score ${fmt(p.score, 3)}${p.gated ? '<br><span class="t-muted">gated</span>' : ''}` })),
    })),
    yTitle: 'score', zeroLabel: '0 = uniform mesh, same node budget', width: 600, height: 280,
  });
  el.append(h('div', { class: 'grid-2', style: { marginBottom: '28px' } },
    h('div', { class: 'card' }, h('h2', {}, 'Score by generation'),
      h('p', { class: 'muted small', style: { marginTop: '-6px' } }, 'One dot per finished agent run; the bar is the median. +1 = ten times more accurate than a uniform mesh with the same number of nodes; a failed gate scores −2 (hollow dots).'),
      chart),
    h('div', { class: 'card' }, h('h2', {}, `Generation ${d.generation}`),
      h('div', { class: 'stat-list' },
        stat('finished agent runs', g.n_submissions ?? 0),
        stat('still running', g.n_running ?? 0),
        stat('median score', signed(g.median)),
        stat('best score', signed(g.best)),
        stat('gated', g.n_gated ?? 0)),
      d.active ? h('p', { class: 'small muted' }, 'The loop is running: this page refreshes every few seconds.') : null)));

  // leaderboards
  if (!d.problems.length) el.append(h('div', { class: 'card empty-state' }, h('h3', {}, 'No agent runs in this generation yet')));
  for (const p of d.problems) el.append(board(p));
}

function live(d) {
  return h('span', { class: `live${d.active ? ' on' : ''}`, title: d.active ? 'new records in the last few minutes' : 'no new records recently' },
    h('span', { class: 'live-dot' }), d.active ? 'live' : 'idle');
}

const stat = (label, value) => h('div', { class: 'stat-line' }, h('span', { class: 'label' }, label), h('span', { class: 'value' }, value));
const signed = x => (x === null || x === undefined || !isFinite(x)) ? '–' : `${x > 0 ? '+' : ''}${x.toFixed(2)}`;

function scoreBar(score, gated) {
  if (score === null || score === undefined) return h('span', { class: 'muted small' }, 'running…');
  const w = 90, c = w / 2, sc = Math.max(-2, Math.min(2, score)) / 2 * (c - 2);
  const bar = `<svg width="${w}" height="12" style="vertical-align:middle"><line x1="${c}" x2="${c}" y1="0" y2="12" stroke="#c3c2b7"/>` +
    `<rect x="${sc >= 0 ? c : c + sc}" y="3" width="${Math.abs(sc)}" height="6" rx="2" fill="${score >= 0 && !gated ? '#2a78d6' : '#b9b8b2'}"/></svg>`;
  return h('span', { class: 'nowrap' }, h('span', { html: bar }), ' ', h('b', { class: 'num' }, signed(score)));
}

function board(p) {
  const ref = p.reference;
  const rows = p.rows;
  return h('div', { class: 'card', style: { marginBottom: '24px' } },
    h('div', { class: 'card-head' },
      h('div', {}, h('h2', { style: { marginBottom: '2px' } }, p.title),
        h('div', { class: 'meta-line' }, p.problem_id, p.qoi ? ` · QoI ${p.qoi}` : '', ref ? ` · reference ${fmt(ref.value, 5)}` : ' · no reference value')),
      h('div', { class: 'row', style: { gap: '10px' } },
        h('a', { class: 'btn ghost', href: `#/benchmarks/${encodeURIComponent(p.problem_id)}` }, 'Problem'),
        h('a', { class: 'btn', href: `#/consensus/${encodeURIComponent(p.problem_id)}` }, 'Consensus view'))),
    h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'agent run'), h('th', {}, 'score'), h('th', { class: 'num' }, p.qoi || 'QoI'),
        h('th', { class: 'num' }, 'error vs ref'), h('th', {}, 'mesh'), h('th', { class: 'num' }, 'DOFs'), h('th', { class: 'num' }, 'solves'),
        h('th', {}, 'gates'), h('th', {}, ''))),
      h('tbody', {}, rows.map((r, i) => h('tr', { class: r.status === 'running' ? 'dim' : '' },
        h('td', { class: 'num muted' }, r.status === 'submitted' ? i + 1 : ''),
        h('td', {}, h('div', { class: 'mono' }, r.ace_run_id), h('div', { class: 'small muted' }, r.status === 'running' ? `running · ${timeAgo(r.created_at)}` : timeAgo(r.created_at),
          r.tokens ? ` · ${fmtInt(r.tokens)} tokens` : '', r.n_cited_bullets ? ` · ${r.n_cited_bullets} playbook bullets` : '')),
        h('td', {}, scoreBar(r.score, r.gates.length > 0)),
        h('td', { class: 'num' }, fmt(r.value, 6)),
        h('td', { class: 'num' }, r.error === null || r.error === undefined ? '–' : `${r.error > 0 ? '+' : ''}${fmt(100 * r.error, 2)}%`),
        h('td', { class: 'small' }, meshText(r)),
        h('td', { class: 'num' }, fmtInt(r.n_dofs)),
        h('td', { class: 'num' }, r.n_attempts > r.n_solves ? `${r.n_solves} / ${r.n_attempts}` : r.n_solves),
        h('td', { class: 'small' }, r.gates.length ? h('span', { class: 'gate', title: r.gates.join(', ') }, h('span', { class: 'dot critical' }), r.gates.join(', ')) : h('span', { class: 'muted' }, 'none'),
          r.setup_diagnostics.length ? h('div', { class: 'muted', title: r.setup_diagnostics.join(', ') }, `${r.setup_diagnostics.length} setup diagnostic${r.setup_diagnostics.length === 1 ? '' : 's'}`) : null),
        h('td', { class: 'nowrap small' }, r.submitted_run_id || r.last_run_id ? h('a', { href: `#/viewer/${r.submitted_run_id || r.last_run_id}` }, 'field →') : null)))))));
}

function meshText(r) {
  const m = r.mesh || {};
  const parts = [];
  if (m.h !== undefined && m.h !== null) parts.push(`h ${fmt(m.h, 3)}`);
  if (r.order) parts.push(`P${r.order}`);
  if (m.refinements) parts.push(`${m.refinements} refinement${m.refinements === 1 ? '' : 's'}`);
  return parts.join(' · ') || '–';
}

function emptyState(d) {
  return h('div', { class: 'card empty-state' },
    h('h2', {}, 'No ACE agent runs yet'),
    h('p', {}, 'This page fills in as Aryan’s ACE loop runs with the run-store bridge enabled. Every solve an agent makes becomes a run record, and every finished agent run a scored submission, in this server’s run store:'),
    h('pre', { class: 'code' }, d.store_root || 'runs/'),
    h('p', {}, 'Start the loop on this machine with the bridge pointed at that directory (FEACORE_RUNS_DIR), and the leaderboards and the score chart update live. The record format is in docs/ui.md (“ACE swarm records”).'),
    h('p', { class: 'muted small' }, 'Meanwhile, benchmark studies and agent submissions from other sources are under Runs and in each problem’s Consensus view.'));
}
