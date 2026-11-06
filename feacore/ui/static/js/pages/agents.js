// Agents: train ACE swarms and the single-agent baseline, watch the runs live, stop them, browse
// finished runs, compare swarm and single agent, export a trained swarm and run it on any problem.
//   #/agents?tab=train|runs|evaluate      the three sections
//   #/agents/run/<key>[?run=<run id>]      one training run: curves, agents, live tracker, report
//   #/agents/infer/<key>, #/agents/swarm/<key>, #/agents/compare/<key>
// Server: feacore/ui/tools/agents.py (the tools, started through the task lane) and
// feacore/ui/ace_runs.py (GET /api/ace/...: the folders they write). <key> is a folder's path relative
// to the repository (ace_runs/div_s8) or @runs/... inside the run store. Everything the runs wrote
// (LLM text included) is put on the page as text, never as HTML.
import { api, postJSON, h, clear, fmt, fmtInt, loading, errorBox, timeAgo, qs, statusChip, toast, withToken, esc, showTip, hideTip } from '../api.js';
import { toolForm, taskTable, taskLog, cancelButtons } from '../tasks.js';
import { niceTicks } from '../plot.js';

const TABS = [['train', 'Train'], ['runs', 'Runs'], ['evaluate', 'Evaluate']];
const enc = encodeURIComponent;
const href = (view, key, q = '') => `#/agents/${view}/${enc(key)}${q}`;
const fileUrl = (key, path, swarm = false) => withToken(`/api/ace/file?dir=${enc(key)}&path=${enc(path)}${swarm ? '&swarm=true' : ''}`);
const LIVE = new Set(['queued', 'running', 'stopping', 'active']);
const STATUS = {
  queued: ['info', 'Queued'], running: ['running', 'Running'], stopping: ['running', 'Stopping after this generation'],
  active: ['running', 'Being written'], finished: ['good', 'Finished'], stopped: ['good', 'Stopped early'],
  failed: ['critical', 'Failed'], cancelled: ['warning', 'Cancelled'], interrupted: ['warning', 'Interrupted'],
  incomplete: ['warning', 'Incomplete'], unreadable: ['critical', 'Unreadable'], done: ['good', 'Done'], ready: ['good', 'Ready'],
};
const chip = s => { const [st, label] = STATUS[s] || ['info', s || 'unknown']; return statusChip(st, label); };
const ok = x => x !== null && x !== undefined && Number.isFinite(x);
const sc = x => ok(x) ? x.toFixed(2) : '–';
const pct = x => ok(x) ? `${fmt(100 * x, 3)}%` : '–';
const usd = x => ok(x) ? `$${x.toFixed(2)}` : '–';
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
/** Append kids to el: nested arrays flattened; null, undefined and false skipped (as h() does). */
function put(el, ...kids) {
  for (const k of kids.flat(Infinity)) if (k !== null && k !== undefined && k !== false) el.append(k);
  return el;
}
const fill = (el, ...kids) => put(clear(el), ...kids);
// categorical slots in fixed order (validated palette, the one plot.js uses); a problem keeps its colour
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];

export async function render(el, args = []) {
  ensureStyle();
  const [view, key] = args;
  const pages = { run: runPage, infer: inferPage, swarm: swarmPage, compare: comparePage };
  if (pages[view] && key) return pages[view](el, key);
  const tab = TABS.some(([k]) => k === qs().get('tab')) ? qs().get('tab') : 'train';
  const body = h('div');
  el.append(h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Agents'),
      h('p', { class: 'lede' }, 'Train swarms of agents that set up and mesh FEA problems (and the single-agent baseline), watch the runs live, then compare them, export a trained swarm and run it on any problem: a benchmark, a gold part or your own CAD.')),
    h('div', { class: 'row', style: { gap: '10px' } },
      h('a', { class: 'btn', href: '#/swarm' }, 'Live swarm view'),
      h('a', { class: 'btn ghost', href: '#/tasks' }, 'All tasks'))),
  h('div', { class: 'toolbar' }, h('div', { class: 'seg' }, TABS.map(([k, label]) =>
    h('button', { class: k === tab ? 'on' : '', onclick: () => { location.hash = `#/agents?tab=${k}`; } }, label)))),
  body);
  return ({ train: trainTab, runs: runsTab, evaluate: evaluateTab })[tab](body);
}

// ====================================================================== Train

const RECIPES = [
  { name: 'Quick try', note: 'Mock LLM, 2 agents, 2 generations on Cook’s membrane: free, done in under a minute. Good for seeing the whole loop.',
    args: { mock: true, problems: 'cooks_membrane', holdout: 'none', agents: 2, concurrency: 2, generations: 2, budget_usd: 1 } },
  { name: 'Deep swarm', note: 'The Result 2 setup: 8 persistent agents (4 with personas) on the gold part abc_00000065_bc0, deep teaching, field scoring. Needs its field reference (python -m ace.gold_reference abc_00000065_bc0).',
    args: { deep: true, split: true, score: 'field', problems: 'abc_00000065_bc0', holdout: 'none', agents: 8, concurrency: 8, persona_count: 4, generations: 8, budget_usd: 45 } },
  { name: 'Post-train on the real test', note: 'The Result 3 run: 4 agents of the trained div_s8 swarm, pre-taught for the Instron compression test of the printed C-bracket and scored against it (needs CalculiX).',
    args: { deep: true, split: true, score: 'field', problems: 'abc_00000065_compression', holdout: 'none', agents: 4, concurrency: 4, init_from: 'results/trained_swarms/div_s8', init_agents: 'a6,a3,a1,a7', pre_teach: true, generations: 10, budget_usd: 45 } },
  { name: 'Basic loop', note: 'The original ACE loop: point scoring on the elasticity suite (3 train, 2 held-out problems), one shared playbook plus each agent’s notes.',
    args: {} },
];

async function trainTab(el) {
  el.append(loading());
  let cat, runs;
  try { [cat, runs] = await Promise.all([api('/api/ace/catalog'), api('/api/ace/runs').then(d => d.runs)]); }
  catch (e) { fill(el, errorBox(e)); return; }
  const tasks = taskTable({ group: 'agents' }, { link: taskHref, limit: 30, empty: 'No agent tasks yet: start a training run above.' });
  fill(el, 
    h('div', { class: 'grid-2', style: { marginBottom: '24px' } }, swarmCard(cat), singleCard(cat, runs)),
    h('div', { class: 'card flush' },
      h('div', { style: { padding: '16px 22px 0' } }, h('h2', {}, 'Agent tasks'),
        h('p', { class: 'muted small' }, 'Training runs, reports, exports, comparisons and inference started here. Stop asks a training run to end after its current generation (it writes STOP into the run folder, and the summary is still written); Kill ends it now.')),
      tasks.el));
  return tasks.stop;
}

function swarmCard(cat) {
  const card = h('div', { class: 'card' });
  let pick = 0;
  const draw = () => {
    const r = RECIPES[pick];
    const values = { ...r.args };   // problems / holdout are not in the form: the picker below edits them here
    fill(card,
      h('div', { class: 'card-head' }, h('h2', {}, 'Train a swarm'), h('code', { class: 'muted small' }, 'ace.run_loop')),
      h('p', { class: 'muted small' }, 'A population of agents (ACE, ace/README.md). Each generation every agent meshes and solves its task, the runs are scored, and a teacher, the agents’ own reflections and an organizer rewrite the agents’ notes for the next generation.'),
      h('div', { class: 'row wrap', style: { gap: '8px', alignItems: 'center', marginBottom: '8px' } },
        h('span', { class: 'section-title', style: { margin: 0 } }, 'Start from'),
        RECIPES.map((x, i) => h('button', { class: `chip${i === pick ? ' on' : ''}`, type: 'button', onclick: () => { pick = i; draw(); } }, x.name))),
      h('p', { class: 'muted small' }, r.note),
      problemPicker(cat, values, { holdout: true }),
      h('div', { class: 'ag-form' }, toolForm('ace.swarm', { defaults: values, hide: ['problems', 'holdout'], submitLabel: 'Start training', onStarted: t => openTask(t, 'run') })));
  };
  draw();
  return card;
}

function singleCard(cat, runs) {
  const values = {};
  const deep = runs.filter(r => r.kind === 'swarm' && r.deep);
  const sel = h('select', { onchange: () => { values.swarm_dir = sel.value; } },
    h('option', { value: '' }, '(none: 60 s per generation)'),
    deep.map(r => h('option', { value: r.key }, `${r.name} · ${plural(r.generations_scored || 0, 'generation')} · ${(r.train || []).join(', ')}`)));
  return h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Train a single agent (the baseline)'), h('code', { class: 'muted small' }, 'ace.single_loop')),
    h('p', { class: 'muted small' }, 'The swarm’s baseline: one agent does every task in one conversation each generation, with the whole swarm’s solve and turn budget, and learns through the same teacher, reflection and organizer. Always field-scored. Run it next to a deep swarm on the same tasks, then compare the two under Evaluate.'),
    problemPicker(cat, values),
    h('label', { class: 'field', style: { marginBottom: '14px' } }, h('span', { class: 'section-title' }, 'Swarm run to match'), sel,
      h('span', { class: 'muted small' }, 'a deep swarm run on the same tasks: its time per generation sets this agent’s, and the two are compared at the end')),
    h('div', { class: 'ag-form' }, toolForm('ace.single', { defaults: values, hide: ['problems', 'swarm_dir'], submitLabel: 'Start training', onStarted: t => openTask(t, 'run') })));
}

function groupTasks(cat) {
  const g = [['Elasticity benchmarks', []], ['Gold CAD parts', []], ['Physical tests', []], ['Flow (CFD, OpenFOAM)', []]];
  for (const [id, t] of Object.entries(cat.tasks)) {
    const i = t.physics === 'flow' ? 3 : t.score === 'nonlinear' || /_(pull|compression)$/.test(id) ? 2
      : /^abc_\d+_bc\d+$/.test(id) || (t.tags || []).includes('gold') ? 1 : 0;
    g[i][1].push(id);
  }
  return g.filter(([, ids]) => ids.length);
}

/** Chips for the loop's task catalogue. Swarm: click = train, again = held out, again = off. Writes
 *  values.problems (and values.holdout: '' = the suite's, 'none' = none). */
function problemPicker(cat, values, { holdout = false } = {}) {
  const state = {};
  for (const id of (values.problems || '').split(',').filter(Boolean)) state[id] = 'train';
  if (holdout) for (const id of (values.holdout || '').split(',').filter(x => x && x !== 'none')) state[id] = 'hold';
  const note = h('p', { class: 'muted small', style: { margin: '8px 0 0' } });
  const ids = s => Object.keys(state).filter(k => state[k] === s);
  const sync = () => {
    const train = ids('train'), hold = ids('hold');
    values.problems = train.join(',');
    if (holdout) values.holdout = train.length || hold.length ? (hold.join(',') || 'none') : '';
    if (!train.length && !hold.length) {
      const fea = cat.suites.fea;
      note.textContent = holdout ? `Nothing picked: the suite’s problems (elasticity: train ${fea.train.join(', ')}; held out ${fea.holdout.join(', ')}).`
        : `Nothing picked: ${cat.single_default.join(', ')}.`;
    } else {
      note.textContent = `Train on ${train.join(', ') || 'nothing'}` + (holdout ? `; held out: ${hold.join(', ') || 'none'}.` : '.');
    }
  };
  const chipFor = id => {
    const t = cat.tasks[id];
    const b = h('button', { type: 'button', title: `${t.title}\nscored quantity ${t.qoi} · node cap ${fmtInt(t.node_cap)}${t.score ? ` · scored: ${t.score}` : ''}` });
    const paint = () => {
      const s = state[id];
      b.className = `ag-chip${s ? ` ${s}` : ''}`;
      fill(b, id, s ? h('span', { class: 'tag' }, s === 'hold' ? 'held out' : 'train') : null);
    };
    b.onclick = () => {
      const next = !state[id] ? 'train' : holdout && state[id] === 'train' ? 'hold' : null;
      if (next) state[id] = next; else delete state[id];
      paint(); sync();
    };
    paint();
    return b;
  };
  sync();
  return h('div', { class: 'field', style: { marginBottom: '16px' } },
    h('span', { class: 'section-title' }, holdout ? 'Problems · click: train, again: held out, again: off' : 'Tasks · click to pick'),
    h('div', { class: 'ag-picker' }, groupTasks(cat).map(([title, list]) =>
      h('div', {}, h('div', { class: 'ag-group' }, title), h('div', { class: 'ag-chips' }, list.map(chipFor))))),
    note);
}

/** A task's page in this section (null for an output path the dashboard cannot name). */
function taskHref(t) {
  const key = v => (v && !v.startsWith('/') ? v.replace(/^\.\//, '').replace(/\/+$/, '') : null);
  const view = { 'ace.swarm': ['run', t.out], 'ace.single': ['run', t.out], 'ace.report': ['run', (t.args || {}).run_dir],
                 'ace.infer': ['infer', t.out], 'ace.export_swarm': ['swarm', t.out], 'ace.compare': ['compare', t.out] }[t.tool];
  return view && key(view[1]) ? href(view[0], key(view[1])) : null;
}

/** After a POST /api/tasks: open the page of what the task writes (found by its task id). */
async function openTask(task, view) {
  const list = { run: ['/api/ace/runs', 'runs'], infer: ['/api/ace/infers', 'infers'], swarm: ['/api/ace/swarms', 'swarms'], compare: ['/api/ace/compares', 'compares'] }[view];
  try {
    const d = await api(list[0]);
    const x = d[list[1]].find(r => r.task && r.task.id === task.id);
    if (x) { location.hash = href(view, x.key); return; }
  } catch { /* fall through to the task page */ }
  location.hash = `#/tasks/${task.id}`;
}

// ====================================================================== Runs

async function runsTab(el) {
  let alive = true, timer = null;
  el.append(loading());
  const tick = async () => {
    let d;
    try { d = await api('/api/ace/runs'); } catch (e) { fill(el, errorBox(e)); return; }
    if (!alive) return;
    fill(el, d.runs.length ? runsTable(d.runs) : h('div', { class: 'card empty-state' },
      h('h3', {}, 'No training runs yet'),
      h('p', {}, `Runs started from the Train tab, and any run folder in ${d.roots.join(', ')}/, show up here.`),
      h('a', { class: 'btn primary', href: '#/agents?tab=train' }, 'Train a swarm')));
    timer = setTimeout(tick, d.runs.some(r => LIVE.has(r.status)) ? 4000 : 20000);
  };
  await tick();
  return () => { alive = false; clearTimeout(timer); };
}

const kindText = r => `${r.kind === 'single' ? 'single agent' : 'swarm'}${r.deep && r.kind !== 'single' ? ' · deep' : ''}${r.mock ? ' · mock LLM' : ''}`;

function runsTable(runs) {
  const head = ['run', 'status', 'generations', 'problems', 'agents', 'last generation', 'spent', 'updated'];
  return h('div', { class: 'card flush' }, h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, head.map((x, i) => h('th', { class: [2, 4, 6].includes(i) ? 'num' : null }, x)))),
    h('tbody', {}, runs.map(r => h('tr', { class: 'click', onclick: () => { location.hash = href('run', r.key); } },
      h('td', {}, h('div', { class: 'strong' }, r.name), h('div', { class: 'mono small muted' }, r.key),
        r.kind ? h('div', { class: 'small muted' }, kindText(r)) : h('div', { class: 'small muted' }, r.error || '')),
      h('td', {}, chip(r.status), r.budget_stop ? h('div', { class: 'small muted' }, 'budget reached') : null),
      h('td', { class: 'num' }, `${r.generations_scored ?? 0}${r.planned ? ` / ${r.planned}` : ''}`),
      h('td', { class: 'small' }, (r.train || []).join(', '), (r.holdout || []).length ? h('div', { class: 'muted' }, `held out: ${r.holdout.join(', ')}`) : null),
      h('td', { class: 'num' }, (r.agents || []).length || '–'),
      h('td', { class: 'small nowrap' }, r.last ? `mean ${sc(r.last.mean)} · best ${sc(r.last.best)}` : h('span', { class: 'muted' }, 'not scored yet'),
        r.last ? h('div', { class: 'muted' }, `generation ${r.last.generation}`) : null),
      h('td', { class: 'num' }, usd(r.usd)),
      h('td', { class: 'small muted nowrap' }, timeAgo(r.updated_at))))))));
}

// ====================================================================== one run

async function runPage(el, key) {
  el.append(loading('Loading the run…'));
  let d;
  try { d = await api(`/api/ace/run?dir=${enc(key)}`); }
  catch (e) { fill(el, backLink('runs'), errorBox(e)); return; }
  const view = { measure: 'mean', run: qs().get('run') };
  const box = { head: h('div'), tiles: h('div', { style: { marginBottom: '24px' } }), curve: h('div'), matrix: h('div'), panel: h('div'), files: h('div') };
  let alive = true, timer = null, panelStop = null;
  const select = rid => {
    view.run = rid;
    history.replaceState(null, '', href('run', key, `?run=${enc(rid)}`));
    drawMatrix(box.matrix, d, view, select);
    if (panelStop) panelStop();
    panelStop = agentRunPanel(box.panel, key, rid);
    box.panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  const draw = () => {
    drawHead(box.head, d, refresh);
    drawTiles(box.tiles, d);
    drawCurve(box.curve, d, view, draw);
    drawMatrix(box.matrix, d, view, select);
    drawFiles(box.files, d);
  };
  async function refresh() {
    try { d = await api(`/api/ace/run?dir=${enc(key)}`); } catch { /* keep the last view */ }
    if (!alive) return;
    draw();
    clearTimeout(timer);
    if (LIVE.has(d.summary.status)) timer = setTimeout(refresh, 4000);
  }
  const log = d.summary.task ? taskLog(d.summary.task.id, { height: '360px' }) : null;
  fill(el, box.head, box.tiles, box.curve,
    h('div', { class: 'card', style: { marginBottom: '24px' } },
      h('div', { class: 'card-head' }, h('h2', {}, 'Agents'), h('span', { class: 'muted small' }, 'click a run for the agent’s view')),
      box.matrix, box.panel),
    trackerCard(key), box.files,
    log ? h('div', { class: 'card', style: { marginBottom: '24px' } },
      h('details', { open: LIVE.has(d.summary.status) ? true : null }, h('summary', {}, 'Console output of the task'),
        h('div', { style: { marginTop: '12px' } }, log.el))) : null);
  draw();
  if (view.run) { panelStop = agentRunPanel(box.panel, key, view.run); }
  if (LIVE.has(d.summary.status)) timer = setTimeout(refresh, 4000);
  return () => { alive = false; clearTimeout(timer); if (log) log.stop(); if (panelStop) panelStop(); };
}

const backLink = tab => h('p', {}, h('a', { href: `#/agents?tab=${tab}` }, `← ${tab === 'runs' ? 'Runs' : 'Evaluate'}`));

function describeRun(s) {
  const bits = [kindText(s), `train ${(s.train || []).join(', ') || '(not known yet)'}`];
  if ((s.holdout || []).length) bits.push(`held out ${s.holdout.join(', ')}`);
  bits.push(plural((s.agents || []).length || s.planned_agents || 0, 'agent'));
  if ((s.env || {}).ACE_SCORE === 'field') bits.push('field scoring');
  return bits.join(' · ');
}

function drawHead(el, d, refresh) {
  const s = d.summary;
  const live = LIVE.has(s.status);
  const task = s.task && (s.task.state === 'queued' || s.task.state === 'running') ? { ...s.task, out: s.key, can_stop: true } : null;
  fill(el, h('div', { class: 'page-head' },
    h('div', {}, backLink('runs'), h('h1', {}, s.name, ' ', chip(s.status)),
      h('p', { class: 'lede' }, describeRun(s)),
      h('p', { class: 'muted small mono', style: { margin: '6px 0 0' } }, s.key,
        s.started_at ? ` · started ${timeAgo(s.started_at)}` : '', s.budget_stop ? ' · stopped by the budget' : '')),
    h('div', { class: 'row wrap', style: { gap: '10px', alignItems: 'center' } },
      task ? cancelButtons(task, refresh) : null,
      h('a', { class: 'btn', href: `#/swarm?loop=${enc(s.loop_id)}`, title: 'leaderboards from the run-store records the loop writes' }, 'Swarm view'),
      reportButton(d, refresh),
      s.has.private && !live ? h('a', { class: 'btn', href: `#/agents?tab=evaluate&export=${enc(s.key)}` }, 'Export swarm') : null,
      s.kind === 'single' && !live ? h('a', { class: 'btn', href: `#/agents?tab=evaluate&single=${enc(s.key)}` }, 'Compare') : null,
      s.task ? h('a', { class: 'btn ghost', href: `#/tasks/${s.task.id}` }, 'Task') : null)));
}

function reportButton(d, refresh) {
  const s = d.summary;
  const building = (d.tasks || []).find(t => t.tool === 'ace.report' && (t.state === 'queued' || t.state === 'running'));
  if (building) return h('button', { class: 'btn', disabled: true }, h('span', { class: 'spinner' }), 'Building report…');
  const build = async () => {
    try {
      const t = await postJSON('/api/tasks', { tool: 'ace.report', args: { run_dir: s.key } });
      toast('Building the report');
      await waitTask(t.id);
      refresh();
    } catch (e) { toast(String(e.message || e)); }
  };
  if (!s.has.report) return h('button', { class: 'btn', onclick: build, title: 'python -m ace.report: the whole run as one page' }, 'Build report');
  return h('span', { class: 'row', style: { gap: '6px', alignItems: 'center' } },
    h('a', { class: 'btn', href: withToken(`/api/ace/report?dir=${enc(s.key)}`), target: '_blank', rel: 'noopener' }, 'Report'),
    h('button', { class: 'btn small ghost', onclick: build, title: 'rebuild report.html from the run as it is now' }, 'rebuild'));
}

async function waitTask(id) {
  for (let i = 0; i < 900; i++) {
    const t = await api(`/api/tasks/${id}`);
    if (!['queued', 'running'].includes(t.state)) {
      if (t.state !== 'done') toast(`${t.title}: ${t.error || t.state}`);
      return t;
    }
    await new Promise(r => setTimeout(r, 2000));
  }
  return null;
}

const tile = (label, value, sub) => h('div', { class: 'tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), sub ? h('div', { class: 'sub' }, sub) : null);

function drawTiles(el, d) {
  const s = d.summary, gens = d.generations;
  const cur = gens.length ? gens[gens.length - 1] : null;
  const first = s.first, last = s.last;
  const delta = first && last && last.generation !== first.generation && ok(last.mean) && ok(first.mean) ? last.mean - first.mean : null;
  const label = d.score_label === 'base score' ? 'base score' : 'score';
  fill(el, h('div', { class: 'tiles' },
    tile('Generations scored', `${s.generations_scored}${s.planned ? ` of ${s.planned}` : ''}`,
      cur && cur.in_progress ? `${plural(cur.in_progress, 'agent run')} in progress` : null),
    tile('Agent runs', fmtInt(s.n_runs), s.n_crashed ? `${s.n_crashed} crashed` : null),
    tile('Spent', usd(s.usd), `${fmtInt(s.calls)} model calls${s.mock ? ' (mock: not real dollars)' : ''}`),
    tile(`Mean ${label}, last generation`, last ? sc(last.mean) : '–',
      delta !== null ? `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(2)} vs generation ${first.generation}` : (last ? `generation ${last.generation}` : null)),
    tile(`Best ${label}, last generation`, last ? sc(last.best) : '–', last ? `${last.within_tol} of ${last.n_scored} within the target` : null)));
}

// ---------------------------------------------------------------- training curve

function drawCurve(el, d, view, redraw) {
  const gens = d.generations.filter(g => Object.keys(g.problems).length);
  const card = h('div', { class: 'card', style: { marginBottom: '24px' } });
  if (!gens.some(g => Object.values(g.problems).some(p => p.n_scored))) {
    fill(el, card);
    put(card, h('h2', {}, 'Training curve'), h('p', { class: 'muted' }, 'Fills in when the first generation is scored.'));
    return;
  }
  const base = d.score_label === 'base score';
  const k = view.measure === 'best' ? (base ? 'base_best' : 'best') : (base ? 'base_mean' : 'mean');
  const probs = d.problems.filter(p => gens.some(g => g.problems[p.id]));
  const series = probs.slice(0, SERIES.length).map((p, i) => ({
    name: p.id, held: p.split === 'holdout', color: SERIES[i],
    points: gens.filter(g => g.problems[p.id] && ok(g.problems[p.id][k])).map(g => ({ x: g.gen, y: g.problems[p.id][k], n: g.problems[p.id].n_scored })),
  }));
  const field = d.runs.some(r => ok(r.field_error));
  const seg = h('div', { class: 'seg' }, [['mean', 'Mean'], ['best', 'Best']].map(([m, label]) =>
    h('button', { class: m === view.measure ? 'on' : '', onclick: () => { view.measure = m; redraw(); } }, label)));
  put(card, 
    h('div', { class: 'card-head' }, h('h2', {}, 'Training curve'), seg),
    h('p', { class: 'muted small', style: { marginTop: '-4px' } },
      `The ${view.measure} ${d.score_label} of the agent runs on each problem, generation by generation${probs.some(p => p.split === 'holdout') ? '; held-out problems (dashed) are never taught' : ''}. `,
      field ? 'Field scoring: 0.25 = the gmsh automesher, 1 = twice its accuracy at its cost' + (base ? '; bonuses (be the best, be different) are left out.' : '.')
        : 'Point scoring: valid × accuracy × efficiency, relative to the cheapest run within tolerance on the problem, so earlier generations can move as the loop goes.'),
    lineChart({ series, yTitle: `${view.measure} ${d.score_label}` }),
    keepOpen(genTable(d, probs, gens), view, 'tableOpen'));
  fill(el, card);
}

/** A <details> that stays open (or closed) across the live page's redraws. */
function keepOpen(det, view, flag) {
  if (view[flag] !== undefined) det.open = view[flag];
  det.addEventListener('toggle', () => { view[flag] = det.open; });
  return det;
}

function genTable(d, probs, gens) {
  const rows = [];
  for (const g of gens) {
    probs.filter(p => g.problems[p.id]).forEach((p, i) => {
      const s = g.problems[p.id];
      rows.push(h('tr', {},
        h('td', { class: 'num' }, i === 0 ? String(g.gen) : ''),
        h('td', {}, p.id, p.split === 'holdout' ? h('span', { class: 'muted small' }, ' held out') : null),
        h('td', { class: 'num' }, `${s.n_scored}${s.n !== s.n_scored ? ` / ${s.n}` : ''}`),
        h('td', { class: 'num strong' }, sc(s.mean)), h('td', { class: 'num' }, sc(s.base_mean)),
        h('td', { class: 'num' }, sc(s.best)), h('td', { class: 'num' }, sc(s.min)),
        h('td', { class: 'num' }, `${s.within_tol} / ${s.n_scored}`),
        h('td', { class: 'num' }, s.failed || ''), h('td', { class: 'num' }, s.crashed || ''),
        h('td', { class: 'num' }, fmtInt(s.median_dofs)),
        h('td', { class: 'num' }, i === 0 ? usd(g.usd) : '')));
    });
  }
  return h('details', { class: 'more' }, h('summary', {}, 'Per generation and problem (table)'),
    h('div', { style: { overflowX: 'auto', marginTop: '10px' } }, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['gen', 'problem', 'runs', 'mean', 'base', 'best', 'min', 'within target', 'failed', 'crashed', 'median DOFs', '$ (generation)']
        .map((x, i) => h('th', { class: i === 1 ? null : 'num' }, x)))),
      h('tbody', {}, rows))));
}

// SVG line chart: one 2px line per series, 8px markers with a surface ring, a crosshair that snaps to
// the nearest generation with one tooltip for every series, a legend above (identity is never colour
// alone: held-out series are dashed and named). The table under it is its table view.
const NS = 'http://www.w3.org/2000/svg';
function S(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [a, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(a, v);
  for (const c of kids.flat()) if (c !== null && c !== undefined) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

function lineChart({ series, yTitle, height = 300 }) {
  const W = 1100, H = height, m = { l: 62, r: 24, t: 14, b: 44 };
  const xs = [...new Set(series.flatMap(s => s.points.map(p => p.x)))].sort((a, b) => a - b);
  const ys = series.flatMap(s => s.points.map(p => p.y));
  const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': yTitle });
  if (!xs.length) return svg;
  const x0 = xs.length > 1 ? xs[0] : xs[0] - 0.5, x1 = xs.length > 1 ? xs[xs.length - 1] : xs[0] + 0.5;
  const X = v => m.l + (v - x0) / ((x1 - x0) || 1) * (W - m.l - m.r);
  const top = Math.max(1, ...ys) * 1.05, bottom = Math.min(0, ...ys);
  const yt = niceTicks(bottom, top, 5);
  const y0 = Math.min(bottom, yt[0]), y1 = Math.max(top, yt[yt.length - 1]);
  const Y = v => H - m.b - (v - y0) / ((y1 - y0) || 1) * (H - m.t - m.b);
  for (const t of yt) {
    svg.append(S('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t === 0 ? '#c3c2b7' : '#e1e0d9', 'stroke-width': 1 }));
    svg.append(S('text', { x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end', class: 'tick' }, +t.toFixed(2)));
  }
  const step = Math.max(1, Math.ceil(xs.length / 12));
  xs.forEach((x, i) => { if (i % step === 0 || i === xs.length - 1) svg.append(S('text', { x: X(x), y: H - m.b + 18, 'text-anchor': 'middle', class: 'tick' }, x)); });
  svg.append(S('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-title' }, 'generation'));
  svg.append(S('text', { x: 14, y: (m.t + H - m.b) / 2, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, 'text-anchor': 'middle', class: 'axis-title' }, yTitle));
  const cross = S('line', { y1: m.t, y2: H - m.b, stroke: '#898781', 'stroke-width': 1, visibility: 'hidden' });
  svg.append(cross);
  for (const s of series) {
    if (s.points.length > 1) svg.append(S('path', { d: s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(''),
      fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.held ? '6 4' : null }));
    for (const p of s.points) svg.append(S('circle', { cx: X(p.x), cy: Y(p.y), r: 4, fill: s.color, stroke: '#fcfcfb', 'stroke-width': 2 }));
  }
  const hit = S('rect', { x: m.l - 10, y: m.t, width: W - m.l - m.r + 20, height: H - m.t - m.b, fill: 'transparent' });
  hit.addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) * W / r.width;
    const x = xs.reduce((a, b) => Math.abs(X(b) - px) < Math.abs(X(a) - px) ? b : a, xs[0]);
    cross.setAttribute('x1', X(x)); cross.setAttribute('x2', X(x)); cross.setAttribute('visibility', 'visible');
    const rows = series.map(s => [s, s.points.find(p => p.x === x)]).filter(([, p]) => p).sort((a, b) => b[1].y - a[1].y);
    showTip(e, `<b>generation ${esc(x)}</b>` + rows.map(([s, p]) =>
      `<br><svg width="16" height="8" style="vertical-align:middle"><line x1="0" x2="16" y1="4" y2="4" stroke="${s.color}" stroke-width="2"${s.held ? ' stroke-dasharray="4 3"' : ''}/></svg> <b>${p.y.toFixed(3)}</b> <span class="t-muted">${esc(s.name)} · ${plural(p.n, 'run')}</span>`).join(''));
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  svg.append(hit);
  const legend = h('div', { class: 'legend' }, series.map(s => h('span', { class: 'key' },
    S('svg', { width: 22, height: 10 }, S('line', { x1: 1, x2: 21, y1: 5, y2: 5, stroke: s.color, 'stroke-width': 2.5, 'stroke-dasharray': s.held ? '5 3' : null })),
    `${s.name}${s.held ? ' (held out)' : ''}`)));
  return h('div', {}, series.length > 1 ? legend : null, svg);
}

// ---------------------------------------------------------------- agents x generations

function cellStyle(score) {
  if (!ok(score)) return {};
  const a = 0.08 + 0.62 * Math.max(0, Math.min(1, score));
  return { background: `rgba(42, 120, 214, ${a.toFixed(3)})`, color: a > 0.45 ? '#fff' : null };
}

function drawMatrix(el, d, view, select) {
  const gens = d.generations.map(g => g.gen);
  if (!d.runs.length) { fill(el, h('p', { class: 'muted' }, 'No finished agent runs yet. Agents still meshing show in the live tracker below.')); return; }
  const many = d.problems.length > 1;
  const runCell = r => h('div', { class: `ag-run${r.run_id === view.run ? ' on' : ''}`, style: cellStyle(r.score), title: `${r.run_id}${r.rank ? ` · rank ${r.rank} of ${r.of}` : ''}`,
    onclick: e => { e.stopPropagation(); select(r.run_id); } },
  r.crashed ? 'crashed' : sc(r.score), many ? h('span', { class: 'ag-task' }, r.problem_id) : null);
  const content = [];
  if (d.agents.length) {
    content.push(h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data ag-matrix' },
      h('thead', {}, h('tr', {}, h('th', {}, 'agent'), gens.map(g => h('th', { class: 'num' }, `gen ${g}`)), h('th', { class: 'num' }, 'best'))),
      h('tbody', {}, d.agents.map(a => h('tr', {},
        h('td', {}, h('div', { class: 'mono strong' }, a.agent), a.persona ? h('div', { class: 'small muted', title: a.persona }, String(a.persona).split(/[:.]/)[0].slice(0, 40)) : null),
        gens.map(g => h('td', { class: 'num' }, (a.generations[String(g)] || []).map(x => runCell({ ...x, ...d.runs.find(r => r.run_id === x.run_id) })))),
        h('td', { class: 'num strong' }, sc(a.best))))))));
    content.push(h('p', { class: 'muted small', style: { marginTop: '8px' } },
      `Each cell is one agent run: its ${d.score_label === 'base score' ? 'score with bonuses' : 'score'} (darker = higher)${many ? ' and its task' : ''}. Click one for what the agent saw, did and was told.`));
  }
  const rows = [...d.runs].sort((a, b) => (a.generation - b.generation) || String(a.run_id).localeCompare(b.run_id)).map(r =>
    h('tr', { class: `click${r.run_id === view.run ? ' ag-sel' : ''}`, onclick: () => select(r.run_id) },
      h('td', { class: 'mono small' }, r.run_id), h('td', { class: 'num' }, r.generation), h('td', { class: 'small' }, r.problem_id),
      h('td', { class: 'num strong' }, sc(r.score)), h('td', { class: 'num' }, sc(r.base)),
      h('td', { class: 'num' }, r.rank ? `${r.rank} / ${r.of}` : ''), h('td', { class: 'num' }, fmtInt(r.dofs)),
      h('td', { class: 'num' }, pct(r.rel_error)), h('td', { class: 'num' }, `${r.n_ok} / ${r.n_solves}`),
      h('td', { class: 'small' }, r.crashed ? h('span', { class: 'gate', title: r.error || '' }, h('span', { class: 'dot critical' }), 'crashed') : (r.failed_gate || '')),
      h('td', { class: 'num' }, usd(r.usd))));
  const head = ['run', 'gen', 'problem', 'score', 'base', 'rank', 'DOFs', 'error', 'solves', 'gate / crash', '$'];
  const table = h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, head.map((x, i) => h('th', { class: [0, 2, 9].includes(i) ? null : 'num' }, x)))),
    h('tbody', {}, rows));
  content.push(keepOpen(h('details', { class: 'more', open: d.agents.length ? null : true },
    h('summary', {}, `All ${plural(d.runs.length, 'agent run')}`),
    h('div', { style: { overflowX: 'auto', marginTop: '10px' } }, table)), view, 'allOpen'));
  fill(el, ...content);
}

// ---------------------------------------------------------------- one agent run (the old Streamlit tracker)

const AGENT_VIEWS = [['context', 'Context it saw'], ['teacher', 'Teacher'], ['reflection', 'Self-reflection'], ['organizer', 'Organizer'], ['trace', 'What it did'], ['solves', 'Solves']];
let agentView = 'context';

function agentRunPanel(el, key, rid) {
  let alive = true;
  fill(el, h('div', { class: 'ag-panel' }, loading('Loading the agent run…')));
  (async () => {
    let a;
    try { a = await api(`/api/ace/agent-run?dir=${enc(key)}&run_id=${enc(rid)}`); }
    catch (e) { if (alive) fill(el, h('div', { class: 'ag-panel' }, errorBox(e))); return; }
    if (!alive) return;
    const b = a.brief, c = a.checks || {};
    const body = h('div', { style: { marginTop: '14px' } });
    const seg = h('div', { class: 'seg' });
    const show = v => {
      agentView = v;
      fill(seg, AGENT_VIEWS.map(([k, label]) => h('button', { class: k === v ? 'on' : '', onclick: () => show(k) }, label)));
      fill(body, ...AGENT_RENDER[v](a, key));
    };
    const field = ok(c.field_error);
    const facts = [
      ['score', sc(c.score)], ['base score', ok(c.base_score) ? sc(c.base_score) : null],
      ['bonus', ok(c.best_bonus) || ok(c.diversity_bonus) ? `${sc((c.best_bonus || 0) + (c.diversity_bonus || 0))} (best ${sc(c.best_bonus)}, different ${sc(c.diversity_bonus)})` : null],
      ['DOFs', fmtInt(c.dofs)], ['field error', field ? pct(c.field_error) : null], ['check-point error', ok(c.probe_error) ? pct(c.probe_error) : null],
      ['scored error', pct(c.rel_error)], ['accuracy · efficiency', ok(c.accuracy) ? `${fmt(c.accuracy, 3)} · ${fmt(c.efficiency, 3)}` : null],
      [field ? 'meets the 2x target' : 'within tolerance', c.meets_tol === true ? 'yes' : c.meets_tol === false ? 'no' : null],
      ['gate', c.failed_gate || (c.valid === false ? `invalid${c.invalid_because ? `: ${c.invalid_because}` : ''}` : null)],
      ['solves', `${b.n_ok} ok of ${b.n_solves}`], ['turns', b.turns], ['cited bullets', b.cited], ['spent', usd(b.usd)],
    ].filter(([, v]) => v !== null && v !== undefined && v !== '–');
    const imgs = a.report && a.report.images ? Object.entries(a.report.images) : [];
    fill(el, h('div', { class: 'ag-panel' },
      h('div', { class: 'card-head' },
        h('div', {}, h('h3', { style: { margin: 0 } }, `${b.agent || 'run'} · generation ${b.generation} · ${b.problem_id}`),
          h('div', { class: 'meta-line mono small' }, rid, b.crashed ? ` · crashed: ${b.error || ''}` : '')),
        b.ui_submission_id ? h('a', { class: 'btn small', href: `#/consensus/${enc(b.problem_id)}` }, 'Consensus view') : null),
      a.error ? h('pre', { class: 'code', style: { maxHeight: '220px' } }, String(a.error).slice(-4000)) : null,
      h('div', { class: 'row', style: { gap: '28px' } },
        h('table', { class: 'kv' }, h('tbody', {}, facts.map(([k, v]) => h('tr', {}, h('th', {}, k), h('td', {}, v))))),
        imgs.length ? h('div', { class: 'grow' }, h('div', { class: 'section-title' }, 'Run report: what the teacher and the agent’s self-reflection looked at'),
          h('div', { class: 'ag-imgs' }, imgs.map(([name, path]) => h('figure', {},
            h('a', { href: fileUrl(key, path), target: '_blank', rel: 'noopener' }, h('img', { src: fileUrl(key, path), alt: `${rid}: ${name}`, loading: 'lazy' })),
            h('figcaption', { class: 'small muted' }, name))))) : null),
      a.context_history && a.context_history.length > 1 ? contextHistory(a.context_history) : null,
      h('div', { style: { marginTop: '16px' } }, seg), body));
    show(AGENT_VIEWS.some(([k]) => k === agentView) ? agentView : 'context');
  })();
  return () => { alive = false; };
}

function contextHistory(rows) {
  return h('details', { class: 'more' }, h('summary', {}, 'Context over generations'),
    h('table', { class: 'data', style: { marginTop: '8px' } },
      h('thead', {}, h('tr', {}, ['gen', 'context it saw (characters)', 'notes added', 'edited', 'removed'].map(x => h('th', { class: 'num' }, x)))),
      h('tbody', {}, rows.map(r => h('tr', {}, h('td', { class: 'num' }, r.gen), h('td', { class: 'num' }, fmtInt(r.context_chars)),
        h('td', { class: 'num' }, r.added ?? ''), h('td', { class: 'num' }, r.edited ?? ''), h('td', { class: 'num' }, r.removed ?? ''))))),
    h('p', { class: 'muted small' }, 'The organizer’s changes after a generation show up in the next generation’s context.'));
}

const noteLine = n => h('li', {},
  n.section ? h('span', { class: 'badge' }, n.section) : null, n.kind ? h('span', { class: 'badge' }, n.kind) : null,
  n.new ? h('span', { class: 'badge ag-new' }, 'new') : null, n.cited ? h('span', { class: 'badge ag-cited' }, 'cited') : null,
  n.id ? h('code', { class: 'small' }, ` ${n.id} `) : ' ', n.removed ? h('span', { class: 'ag-was' }, n.content || '') : (n.content || ''));
const noteList = (notes, empty = 'none') => notes && notes.length ? h('ul', { class: 'ag-notes' }, notes.map(noteLine)) : h('p', { class: 'muted small' }, empty);
const para = (title, text) => text ? [h('div', { class: 'section-title', style: { marginTop: '14px' } }, title), md(String(text))] : [];
const codeBlock = (title, obj) => obj ? [h('div', { class: 'section-title', style: { marginTop: '14px' } }, title), h('pre', { class: 'code' }, typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1))] : [];

const AGENT_RENDER = {
  context(a) {
    const c = a.context;
    if (!c) return [h('p', { class: 'muted' }, 'The run record has no context (older runs).')];
    const pre = h('pre', { class: 'ag-ctx' });
    const toggle = h('input', { type: 'checkbox', checked: c.diff ? true : null });
    const paint = () => {
      clear(pre);
      if (c.diff && toggle.checked) {
        for (const line of c.diff) {
          const cls = line[0] === '+' ? 'add' : line[0] === '-' ? 'del' : null;
          pre.append(h('span', { class: cls }, line.slice(1) + '\n'));
        }
      } else pre.append(c.text);
    };
    toggle.onchange = paint;
    paint();
    const notes = a.notes || {};
    return [
      h('div', { class: 'row', style: { gap: '18px', alignItems: 'center', marginBottom: '8px' } },
        h('span', { class: 'muted small' }, `${fmtInt(c.text.length)} characters`),
        c.diff ? h('label', { class: 'check' }, toggle, `Highlight changes since generation ${c.prev_gen} (+${c.added} / −${c.removed} lines)`)
          : c.prev_gen !== null ? h('span', { class: 'muted small' }, `unchanged since generation ${c.prev_gen}`) : null),
      pre,
      h('details', { class: 'more' }, h('summary', {}, 'Notes as stored for this generation'),
        h('div', { class: 'section-title', style: { marginTop: '10px' } }, `Shared playbook (${(notes.shared || []).length})`), noteList(notes.shared),
        notes.private ? [h('div', { class: 'section-title', style: { marginTop: '10px' } }, `The agent’s own notes (${notes.private.length}; kept across tasks)`), noteList(notes.private)] : null),
    ];
  },
  teacher(a) {
    const out = [];
    const f = a.teacher;
    if (f) {
      out.push(...para('Diagnosis of this run', f.diagnosis));
      if (f.assignment) out.push(...codeBlock('Its role next generation', f.assignment));
      if ((f.strengths || []).length) out.push(h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'What it did well'), h('ul', {}, f.strengths.map(x => h('li', {}, String(x)))));
      if ((f.mistakes || []).length) out.push(h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'Mistakes'), h('ul', {}, f.mistakes.map(x => h('li', {}, String(x)))));
      out.push(...codeBlock('Suggested next mesh call', f.next_mesh_call));
      out.push(h('div', { class: 'section-title', style: { marginTop: '14px' } }, `Notes for the agent (${(f.notes || []).length}; they go to the organizer)`), noteList(f.notes));
    }
    const t = a.teacher_task;
    if (t) {
      out.push(h('details', { class: 'more' }, h('summary', {}, 'The whole task: summary, setup warnings, highlights, shared lessons'),
        ...para('Summary', t.task_summary),
        (t.setup_warnings || []).length ? [h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'Setup warnings'), h('ul', {}, t.setup_warnings.map(w => h('li', {}, String(w))))] : null,
        (t.highlights || []).length ? [h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'Highlights'),
          h('ul', {}, t.highlights.map(x => h('li', {}, h('b', {}, `${x.agent || ''}: ${x.what || ''}`), x.why_it_works ? ` ${x.why_it_works}` : '', x.how_to_copy ? h('div', { class: 'muted small' }, `How to copy: ${x.how_to_copy}`) : null)))] : null,
        (t.shared_lessons || []).length ? [h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'Shared lessons'), noteList(t.shared_lessons)] : null));
    }
    const g = a.diagnosis;
    if (g) {
      out.push(h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'Advice to this agent (goes into its own notes)'),
        g.advice.length ? h('ul', {}, g.advice.map(x => h('li', {}, String(x.content || '')))) : h('p', { class: 'muted small' }, 'none addressed to it'));
      if (g.this_run) out.push(...para('Diagnosis of this run', [g.this_run.stage, g.this_run.category].filter(Boolean).join(' / ') + (g.this_run.cause ? `: ${g.this_run.cause}` : '')));
      out.push(h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'Lessons for everyone on the task'), noteList(g.lessons.map(x => ({ ...x, kind: x.evidence_type }))),
        h('p', { class: 'muted small' }, `${g.tool_calls ?? 0} teacher tool calls, ${g.reruns} counterfactual reruns${g.divergence ? ` · ${g.divergence}` : ''}`));
    }
    return out.length ? out : [h('p', { class: 'muted' }, 'No teacher feedback for this run: the last generation is only evaluated, and anonymous runs get none.')];
  },
  reflection(a) {
    const r = a.reflection;
    if (!r) return [h('p', { class: 'muted' }, 'No self-reflection for this run (only deep runs write one, and not in the last generation).')];
    return [...para('What happened (the agent, looking at its own results and whole mesh)', r.what_happened),
      h('div', { class: 'section-title', style: { marginTop: '14px' } }, 'What it learned'), noteList(r.what_i_learned),
      ...para('Plan for next time', r.plan_next), ...codeBlock('Next mesh call it plans', r.next_mesh_call)];
  },
  organizer(a) {
    const o = a.organizer;
    if (!o) return [h('p', { class: 'muted' }, 'No organizer record for this agent in this generation (deep runs only, not in the last generation).')];
    const edited = (o.edited || []).map(n => h('li', {}, h('code', { class: 'small' }, `${n.id} `), h('span', { class: 'ag-was' }, n.was || ''), h('br'), h('span', { class: 'ag-now' }, n.content || '')));
    return [
      h('p', { class: 'muted small' }, `After this generation: ${o.before} notes before, ${o.incoming.length} incoming, ${o.after} after. These are the notes the agent sees in the next generation.`),
      o.error ? errorBox(o.error) : null,
      h('div', { class: 'section-title', style: { marginTop: '10px' } }, `Added (${o.added.length})`), noteList(o.added),
      h('div', { class: 'section-title', style: { marginTop: '10px' } }, `Edited (${o.edited.length})`), edited.length ? h('ul', { class: 'ag-notes' }, edited) : h('p', { class: 'muted small' }, 'none'),
      h('div', { class: 'section-title', style: { marginTop: '10px' } }, `Removed (${o.removed.length})`), noteList(o.removed.map(n => ({ ...n, removed: true }))),
      h('details', { class: 'more' }, h('summary', {}, 'Incoming notes (teacher, self-reflection, peers, shared lessons)'),
        noteList(o.incoming.map(n => ({ ...n, kind: n.source })))),
    ];
  },
  trace(a) {
    if (!a.trace.length) return [h('p', { class: 'muted' }, 'No trace recorded.')];
    return [h('div', { class: 'ag-trace' }, a.trace.map(t =>
      t.type === 'call' ? h('div', { class: 'call' }, h('b', {}, t.name), t.using && t.using.length ? h('span', { class: 'muted small' }, ` · using ${t.using.join(', ')}`) : null, h('pre', {}, t.input))
        : t.type === 'result' ? h('div', { class: `result${t.is_error ? ' err' : ''}` }, h('span', { class: 'small muted' }, `${t.name} →`), h('pre', {}, t.text))
          : h('div', { class: 'think' }, t.type === 'thinking' ? h('span', { class: 'small muted' }, 'thinking: ') : null, t.text)))];
  },
  solves(a) {
    const rows = a.solves || [];
    if (!rows.length) return [h('p', { class: 'muted' }, 'No solves.')];
    return [h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['#', 'mesh', 'DOFs', 'quantity', 'value', '', 'field'].map((x, i) => h('th', { class: [0, 2, 4].includes(i) ? 'num' : null }, x)))),
      h('tbody', {}, rows.map(s => h('tr', {},
        h('td', { class: 'num' }, s.index, s.scored ? h('div', { class: 'small muted' }, 'scored') : null),
        h('td', { class: 'small' }, `${fmtInt(s.mesh.nodes)} nodes · P${s.mesh.order ?? '?'}${ok(s.mesh.h) ? ` · h ${fmt(s.mesh.h, 3)}` : ''}`),
        h('td', { class: 'num' }, fmtInt(s.dofs)), h('td', { class: 'small' }, s.qoi || ''),
        h('td', { class: 'num' }, ok(s.value) ? fmt(s.value, 6) : ''),
        h('td', { class: 'small' }, s.ok ? '' : h('span', { class: 'gate', title: s.error || '' }, h('span', { class: 'dot critical' }), (s.error || 'failed').slice(0, 80))),
        h('td', { class: 'small' }, s.ui_run_id ? h('a', { href: `#/viewer/${enc(s.ui_run_id)}` }, 'open →') : h('span', { class: 'muted' }, '–')))))),
    rows.some(s => s.ui_run_id) ? null : h('p', { class: 'muted small' }, 'Field links appear when the loop records its solves in this server’s run store (runs started from this page always do).')];
  },
};

function trackerCard(key) {
  const src = withToken(`/api/ace/tracker?dir=${enc(key)}`);
  return h('div', { class: 'card', style: { marginBottom: '24px' } },
    h('div', { class: 'card-head' }, h('h2', {}, 'Live tracker'),
      h('a', { class: 'btn small', href: src, target: '_blank', rel: 'noopener' }, 'Open in a new tab')),
    h('p', { class: 'muted small', style: { marginTop: '-6px' } }, 'The ACE tracker (python -m ace.watch): the shared playbook and what each generation taught the next, one lane per task with every agent’s card, and where each agent placed. It refreshes itself every few seconds while the run goes.'),
    h('iframe', { class: 'ag-frame', src, title: 'ACE live tracker', loading: 'lazy' }));
}

function drawFiles(el, d) {
  const s = d.summary;
  const parts = [];
  if (d.summary_md || s.has.scores_png) parts.push(h('div', { class: 'card', style: { marginBottom: '24px' } },
    h('h2', {}, 'Summary'),
    d.summary_md ? md(d.summary_md) : null,
    s.has.scores_png ? h('figure', { style: { margin: '10px 0 0', maxWidth: '640px' } },
      h('img', { src: fileUrl(s.key, 'scores.png'), alt: 'mean score per generation, one line per problem', style: { width: '100%' } }),
      h('figcaption', { class: 'small muted' }, 'scores.png, written by the loop when it ends')) : null));
  if (d.compare_md) parts.push(h('div', { class: 'card', style: { marginBottom: '24px' } }, h('h2', {}, 'Compared with its swarm'), md(d.compare_md),
    h('img', { src: fileUrl(s.key, 'compare.png'), alt: 'swarm vs single agent', style: { maxWidth: '100%' }, onerror: e => e.target.remove() })));
  if (d.init) parts.push(h('div', { class: 'card', style: { marginBottom: '24px' } }, h('h2', {}, 'Started from a trained swarm'), h('pre', { class: 'code' }, JSON.stringify(d.init, null, 1))));
  fill(el, ...parts);
}

// ====================================================================== Evaluate

async function evaluateTab(el) {
  el.append(loading());
  let swarms, runs, problems;
  try {
    [swarms, runs, problems] = await Promise.all([api('/api/ace/swarms').then(d => d.swarms), api('/api/ace/runs').then(d => d.runs),
      api('/api/problems').then(d => d.problems)]);
  } catch (e) { fill(el, errorBox(e)); return; }
  const want = qs();
  const lists = h('div');
  fill(el, 
    inferCard(swarms, problems, want.get('swarm')),
    h('div', { class: 'grid-2', style: { marginBottom: '24px' } }, exportCard(runs, want.get('export')), compareCard(runs, want.get('single'))),
    lists);
  let alive = true, timer = null;
  const tick = async () => {
    let inf, sw, cmp;
    try { [inf, sw, cmp] = await Promise.all([api('/api/ace/infers'), api('/api/ace/swarms'), api('/api/ace/compares')]); }
    catch (e) { fill(lists, errorBox(e)); return; }
    if (!alive) return;
    fill(lists, infersTable(inf.infers), swarmsTable(sw.swarms), comparesTable(cmp.compares));
    const busy = [...inf.infers, ...sw.swarms, ...cmp.compares].some(x => x.task && ['queued', 'running'].includes(x.task.state));
    timer = setTimeout(tick, busy ? 4000 : 20000);
  };
  await tick();
  return () => { alive = false; clearTimeout(timer); };
}

function inferCard(swarms, problems, want) {
  const ready = swarms.filter(s => s.status === 'ready');
  const card = h('div', { class: 'card', style: { marginBottom: '24px' } },
    h('div', { class: 'card-head' }, h('h2', {}, 'Run a trained swarm on a problem'), h('code', { class: 'muted small' }, 'ace.infer')),
    h('p', { class: 'muted' }, 'Each agent of an exported swarm sets up, meshes and solves the problem with the context training gave it; nothing learns. With no ground truth the answer is the agreement between agents: the median and spread of every quantity, and the agent most confident by its own error estimate.'));
  if (!ready.length) { put(card, h('p', {}, 'No exported swarm yet: export one below, or add one to results/trained_swarms/.')); return card; }
  const values = { swarm: '', problem: '', agents: 'all' };
  const picked = new Set();
  const agentsBox = h('div', { class: 'ag-chips' });
  const info = h('p', { class: 'muted small', style: { margin: '6px 0 0' } });
  const swarmSel = h('select', {}, ready.map(s => h('option', { value: s.key, selected: s.key === want ? true : null },
    `${s.name} · ${s.kind === 'single' ? 'single agent' : plural(s.agents.length, 'agent')} · generation ${s.generation}`)));
  const syncAgents = () => {
    const s = ready.find(x => x.key === swarmSel.value);
    values.agents = !s || picked.size === s.agents.length ? 'all' : [...picked].join(',');
    info.textContent = s ? `${s.root}/${(s.tasks || []).length ? ` · trained on ${s.tasks.join(', ')}` : ''}${s.exported_at ? ` · exported ${s.exported_at}` : ''}` : '';
  };
  const drawAgents = () => {
    const s = ready.find(x => x.key === swarmSel.value);
    values.swarm = swarmSel.value;
    picked.clear();
    (s ? s.agents : []).forEach(a => picked.add(a));
    fill(agentsBox, (s ? s.agents : []).map(a => {
      const b = h('button', { type: 'button', class: 'ag-chip train', title: s.personas[a] || '' }, a);
      b.onclick = () => {
        if (picked.has(a) && picked.size > 1) picked.delete(a); else picked.add(a);
        b.className = `ag-chip${picked.has(a) ? ' train' : ''}`;
        syncAgents();
      };
      return b;
    }));
    syncAgents();
  };
  swarmSel.onchange = drawAgents;
  const probs = problemSelect(problems, v => { values.problem = v; });
  put(card, 
    h('div', { class: 'row wrap', style: { gap: '22px', marginBottom: '14px' } },
      h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'Exported swarm'), swarmSel, info),
      h('div', { class: 'field' }, h('span', { class: 'section-title' }, 'Problem'), probs)),
    h('div', { class: 'field', style: { marginBottom: '14px' } }, h('span', { class: 'section-title' }, 'Agents · click to leave one out'), agentsBox),
    h('div', { class: 'ag-form' }, toolForm('ace.infer', { defaults: values, hide: ['swarm', 'problem', 'agents'], submitLabel: 'Run the swarm', onStarted: t => openTask(t, 'infer') })));
  drawAgents();
  return card;
}

/** A filterable problem list: your CAD (tag custom) and gold parts first, then everything else. */
function problemSelect(problems, onChange) {
  const groups = [['Your CAD', p => (p.tags || []).includes('custom')], ['Gold parts', p => (p.tags || []).includes('gold')], ['Benchmarks and other problems', () => true]];
  const member = new Map();
  for (const p of problems) member.set(p.id, groups.findIndex(([, test]) => test(p)));
  const sel = h('select', { style: { maxWidth: '520px' } });
  const filter = h('input', { type: 'search', placeholder: 'filter by id, title or tag…', style: { maxWidth: '520px' } });
  const note = h('span', { class: 'muted small' });
  const describe = () => {
    const p = problems.find(x => x.id === sel.value);
    note.textContent = !p ? 'pick a problem' : `${p.title} · ${p.dim}D ${p.physics}${p.n_references ? ' · has reference values' : ' · no reference: the agents’ agreement is the answer'}`;
    onChange(sel.value);
  };
  const options = () => {
    const q = filter.value.trim().toLowerCase(), keep = sel.value;
    fill(sel, h('option', { value: '' }, '(choose a problem)'));
    groups.forEach(([label], gi) => {
      const items = problems.filter(p => member.get(p.id) === gi && (!q || `${p.id} ${p.title} ${(p.tags || []).join(' ')}`.toLowerCase().includes(q)));
      if (items.length) sel.append(h('optgroup', { label: `${label} (${items.length})` },
        items.map(p => h('option', { value: p.id, selected: p.id === keep ? true : null }, `${p.id} · ${p.title}`))));
    });
    describe();
  };
  sel.addEventListener('change', describe);
  filter.addEventListener('input', options);
  options();
  return h('div', { class: 'ag-probsel' }, filter, sel, note);
}

function exportCard(runs, want) {
  const card = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, 'Export a trained swarm'), h('code', { class: 'muted small' }, 'ace.export_swarm')),
    h('p', { class: 'muted small' }, 'Copies what a run learned (the shared playbook, each agent’s notes, personas and last feedback, and the context each agent gets on a new problem) into a small folder: no meshes, no traces. A run stopped mid-generation exports its most trained state.'));
  const ready = runs.filter(r => r.has && r.has.private && !LIVE.has(r.status));
  if (!ready.length) { put(card, h('p', {}, 'Nothing to export yet: export needs a finished run with persistent agents (a deep swarm, or the single agent).')); return card; }
  const formBox = h('div');
  const sel = h('select', {}, ready.map(r => h('option', { value: r.key, selected: r.key === want ? true : null },
    `${r.name} · ${r.kind === 'single' ? 'single agent' : plural((r.agents || []).length, 'agent')} · ${plural(r.generations_scored || 0, 'generation')}`)));
  const draw = () => {
    const r = ready.find(x => x.key === sel.value);
    fill(formBox, h('div', { class: 'ag-form' }, toolForm('ace.export_swarm', { defaults: { run_dir: r.key, out_dir: `swarms/${r.name}` }, hide: ['run_dir'], submitLabel: 'Export', onStarted: t => openTask(t, 'swarm') })));
  };
  sel.onchange = draw;
  put(card, h('label', { class: 'field', style: { marginBottom: '14px' } }, h('span', { class: 'section-title' }, 'Run'), sel), formBox);
  draw();
  return card;
}

function compareCard(runs, wantSingle) {
  const card = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, 'Compare swarm and single agent'), h('code', { class: 'muted small' }, 'ace.compare')),
    h('p', { class: 'muted small' }, 'Base score per task per generation of a deep swarm run and a single-agent run, against generations, cumulative $ and wall time.'));
  const swarmRuns = runs.filter(r => r.kind === 'swarm' && r.deep), singles = runs.filter(r => r.kind === 'single');
  if (!swarmRuns.length || !singles.length) {
    put(card, h('p', {}, `Needs a deep swarm run and a single-agent run (on the same tasks); there ${swarmRuns.length === 1 ? 'is' : 'are'} ${swarmRuns.length} deep swarm and ${plural(singles.length, 'single-agent run')}.`));
    return card;
  }
  const values = {};
  const pick = (list, name, want) => {
    const sel = h('select', { onchange: () => { values[name] = sel.value; } }, list.map(r => h('option', { value: r.key, selected: r.key === want ? true : null }, `${r.name} · ${(r.train || []).join(', ')}`)));
    values[name] = sel.value || (list[0] && list[0].key);
    return sel;
  };
  put(card, h('div', { class: 'row wrap', style: { gap: '18px', marginBottom: '14px' } },
    h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'Swarm run'), pick(swarmRuns, 'swarm_dir')),
    h('label', { class: 'field' }, h('span', { class: 'section-title' }, 'Single-agent run'), pick(singles, 'single_dir', wantSingle))),
  h('div', { class: 'ag-form' }, toolForm('ace.compare', { defaults: values, hide: ['swarm_dir', 'single_dir'], submitLabel: 'Compare', onStarted: t => openTask(t, 'compare') })));
  return card;
}

const listCard = (title, sub, table) => h('div', { class: 'card flush', style: { marginBottom: '24px' } },
  h('div', { style: { padding: '16px 22px 0' } }, h('h2', {}, title), sub ? h('p', { class: 'muted small' }, sub) : null), table);

function infersTable(list) {
  return listCard('Inference results', 'ace.infer outputs in ace_runs/ and from this page.', list.length ? h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['problem', 'swarm', 'status', 'agents', 'scored quantity (median, spread)', 'most confident', 'spent', 'updated'].map((x, i) => h('th', { class: [3, 6].includes(i) ? 'num' : null }, x)))),
    h('tbody', {}, list.map(r => h('tr', { class: 'click', onclick: () => { location.hash = href('infer', r.key); } },
      h('td', {}, h('div', { class: 'strong' }, r.problem || '?'), h('div', { class: 'mono small muted' }, r.key)),
      h('td', { class: 'small' }, (r.swarm || {}).run || '', ok((r.swarm || {}).generation) ? h('div', { class: 'muted' }, `generation ${r.swarm.generation}`) : null),
      h('td', {}, chip(r.status), r.mode ? h('div', { class: 'small muted' }, `${r.mode} mode`) : null),
      h('td', { class: 'num' }, r.n_agents === null ? '' : `${r.n_ok} / ${r.n_agents}`),
      h('td', { class: 'small' }, r.consensus ? `${r.scored_quantity} = ${fmt(r.consensus.median, 5)} · spread ${pct(r.consensus.spread_rel)}` : ''),
      h('td', { class: 'small' }, r.most_confident ? `${r.most_confident.agent} (ZZ ${pct(r.most_confident.eta_rel)})` : ''),
      h('td', { class: 'num' }, usd(r.usd)), h('td', { class: 'small muted nowrap' }, timeAgo(r.updated_at))))))) : h('p', { class: 'muted', style: { padding: '0 22px 14px' } }, 'None yet.'));
}

function swarmsTable(list) {
  return listCard('Exported swarms', 'results/trained_swarms/ (committed with the repo), swarms/, and exports started here.', list.length ? h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['swarm', 'from run', 'agents', 'generation', 'trained on', 'exported'].map((x, i) => h('th', { class: [2, 3].includes(i) ? 'num' : null }, x)))),
    h('tbody', {}, list.map(s => h('tr', { class: 'click', onclick: () => { location.hash = href('swarm', s.key); } },
      h('td', {}, h('div', { class: 'strong' }, s.name, s.status !== 'ready' ? [' ', chip(s.status)] : null), h('div', { class: 'mono small muted' }, s.key)),
      h('td', { class: 'small' }, s.run || '', s.kind ? h('div', { class: 'muted' }, s.kind === 'single' ? 'single agent' : 'swarm') : null),
      h('td', { class: 'num' }, (s.agents || []).length || ''), h('td', { class: 'num' }, s.generation ?? ''),
      h('td', { class: 'small' }, (s.tasks || []).join(', ')), h('td', { class: 'small muted nowrap' }, s.exported_at || '')))))) : h('p', { class: 'muted', style: { padding: '0 22px 14px' } }, 'None yet.'));
}

function comparesTable(list) {
  return listCard('Comparisons', null, list.length ? h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['comparison', 'swarm', 'single agent', 'updated'].map(x => h('th', {}, x)))),
    h('tbody', {}, list.map(c => h('tr', { class: 'click', onclick: () => { location.hash = href('compare', c.key); } },
      h('td', {}, h('div', { class: 'strong' }, c.name, c.status !== 'done' ? [' ', chip(c.status)] : null), h('div', { class: 'mono small muted' }, c.key)),
      h('td', { class: 'small mono' }, c.swarm || ''), h('td', { class: 'small mono' }, c.single || ''),
      h('td', { class: 'small muted nowrap' }, timeAgo(c.updated_at)))))) : h('p', { class: 'muted', style: { padding: '0 22px 14px' } }, 'None yet.'));
}

// ---------------------------------------------------------------- inference result

async function inferPage(el, key) {
  el.append(loading());
  let alive = true, timer = null, log = null;
  const tick = async () => {
    let d;
    try { d = await api(`/api/ace/infer?dir=${enc(key)}`); } catch (e) { fill(el, backLink('evaluate'), errorBox(e)); return; }
    if (!alive) return;
    const running = d.task && ['queued', 'running'].includes(d.task.state);
    if (running && !log) log = taskLog(d.task.id, { height: '300px' });
    drawInfer(el, d, key, running ? log : null);
    if (running) timer = setTimeout(tick, 4000);
  };
  await tick();
  return () => { alive = false; clearTimeout(timer); if (log) log.stop(); };
}

function drawInfer(el, d, key, log) {
  const r = d.results || {};
  const cons = r.consensus || {};
  const task = log ? d.task : null;
  const sq = (r.task || {}).qoi;
  const field = r.mode === 'field';
  fill(el, 
    h('div', { class: 'page-head' },
      h('div', {}, backLink('evaluate'), h('h1', {}, `Inference: ${d.problem || '?'} `, chip(d.status)),
        h('p', { class: 'lede' }, `Swarm ${(d.swarm || {}).run || '?'}${ok((d.swarm || {}).generation) ? ` (generation ${d.swarm.generation}, trained on ${(d.swarm.tasks || []).join(', ')})` : ''}`,
          r.mode ? ` · ${r.mode} mode` : '', ok(r.usd) ? ` · spent ${usd(r.usd)}` : ''),
        h('p', { class: 'muted small mono', style: { margin: '6px 0 0' } }, key)),
      h('div', { class: 'row wrap', style: { gap: '10px' } },
        task ? cancelButtons({ ...task, can_stop: false }, null) : null,
        d.problem ? h('a', { class: 'btn', href: `#/benchmarks/${enc(d.problem)}` }, 'Problem') : null,
        d.agents.some(a => a.ui_submission_id) ? h('a', { class: 'btn', href: `#/consensus/${enc(d.problem)}` }, 'Consensus view') : null,
        d.task ? h('a', { class: 'btn ghost', href: `#/tasks/${d.task.id}` }, 'Task') : null)),
    task ? h('div', { class: 'card', style: { marginBottom: '24px' } }, h('p', {}, 'The agents are working. Their predictions appear here when all of them are done.'), log.el) : null,
    Object.keys(cons.qois || {}).length ? h('div', { class: 'card', style: { marginBottom: '24px' } },
      h('h2', {}, 'Consensus'),
      h('p', { class: 'muted small', style: { marginTop: '-6px' } }, 'For each quantity of the statement: the median of the agents’ values, their range and the relative spread (max − min) / |median|.'),
      h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, ['quantity', 'median', 'min', 'max', 'spread', 'agents'].map((x, i) => h('th', { class: i ? 'num' : null }, x)))),
        h('tbody', {}, Object.entries(cons.qois).map(([q, c]) => h('tr', {},
          h('td', { class: q === sq ? 'strong' : null }, q, q === sq ? h('span', { class: 'muted small' }, ' scored') : null),
          h('td', { class: 'num strong' }, fmt(c.median, 6)), h('td', { class: 'num' }, fmt(c.min, 6)), h('td', { class: 'num' }, fmt(c.max, 6)),
          h('td', { class: 'num' }, pct(c.spread_rel)), h('td', { class: 'num' }, c.n))))),
      cons.most_confident ? h('p', { style: { marginTop: '12px' } }, 'Most confident by its own error estimate: ', h('b', {}, cons.most_confident.agent), ` (ZZ relative energy-norm error ${pct(cons.most_confident.eta_rel)}).`) : null) : null,
    d.agents.length ? h('div', { class: 'card', style: { marginBottom: '24px' } },
      h('h2', {}, 'Agents'),
      h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, ['agent', 'solve', 'nodes', 'DOFs', 'order', 'h', 'ZZ error est.', sq || 'scored quantity', ...(field ? ['field error', 'score'] : [])].map((x, i) => h('th', { class: i ? 'num' : null }, x)))),
        h('tbody', {}, d.agents.map(a => {
          const p = a.prediction || {}, m = p.mesh || {}, fs = a.field_score || {};
          if (a.error || p.error || !p.mesh) return h('tr', {}, h('td', { class: 'mono strong' }, a.agent), h('td', { colspan: field ? 9 : 7, class: 'small' }, h('span', { class: 'gate' }, h('span', { class: 'dot critical' }), 'no prediction: '), String(a.error || p.error || 'no solve').split('\n').filter(Boolean).pop()));
          return h('tr', {}, h('td', { class: 'mono strong' }, a.agent),
            h('td', { class: 'num' }, p.solve, p.submitted ? '' : h('div', { class: 'small muted' }, 'not submitted')),
            h('td', { class: 'num' }, fmtInt(m.nodes)), h('td', { class: 'num' }, fmtInt(m.dofs)), h('td', { class: 'num' }, m.order ?? ''),
            h('td', { class: 'num' }, ok(m.h) ? fmt(m.h, 3) : ''), h('td', { class: 'num' }, pct((p.error_estimate || {}).eta_rel)),
            h('td', { class: 'num' }, fmt((p.scored_quantity || {}).value, 6)),
            ...(field ? [h('td', { class: 'num' }, pct(fs.rel_error)), h('td', { class: 'num' }, sc(fs.base_score))] : []));
        }))))) : null,
    d.agents.some(a => Object.keys(a.images).length) ? h('div', { class: 'card', style: { marginBottom: '24px' } },
      h('h2', {}, 'Each agent’s submitted solve'),
      h('div', { class: 'ag-imgs' }, d.agents.flatMap(a => Object.entries(a.images).map(([kind, path]) => h('figure', {},
        h('a', { href: fileUrl(key, path), target: '_blank', rel: 'noopener' }, h('img', { src: fileUrl(key, path), alt: `${a.agent}: ${kind}`, loading: 'lazy' })),
        h('figcaption', { class: 'small muted' }, `${a.agent} · ${kind}`)))))) : null,
    d.agents.some(a => a.context || a.solves.some(s => s.ui_run_id)) ? h('div', { class: 'card', style: { marginBottom: '24px' } },
      h('h2', {}, 'What each agent was given and did'),
      d.agents.map(a => h('details', { class: 'more' }, h('summary', {}, `${a.agent}${a.stop ? ` · stopped: ${a.stop}` : ''}${a.turns ? ` · ${plural(a.turns, 'turn')}` : ''}`),
        a.solves.length ? AGENT_RENDER.solves(a) : null,
        a.context ? lazyText(key, a.context) : null))) : null,
    d.summary_md ? h('div', { class: 'card' }, h('details', { class: 'more', style: { marginTop: 0, borderTop: 0, paddingTop: 0 } },
      h('summary', {}, 'summary.md, as ace.infer wrote it'), md(d.summary_md))) : null);
}

/** A text file of a folder, fetched when its <details> opens. */
function lazyText(key, path, swarm = false) {
  const pre = h('pre', { class: 'ag-ctx' }, 'loading…');
  const det = h('details', { class: 'more' }, h('summary', {}, `${path}`), pre);
  det.addEventListener('toggle', async () => {
    if (!det.open || pre.dataset.loaded) return;
    try {
      const res = await fetch(fileUrl(key, path, swarm));
      pre.textContent = res.ok ? await res.text() : `${res.status} ${res.statusText}`;
      pre.dataset.loaded = '1';
    } catch (e) { pre.textContent = String(e); }
  });
  return det;
}

// ---------------------------------------------------------------- exported swarm

async function swarmPage(el, key) {
  el.append(loading());
  let s;
  try { s = await api(`/api/ace/swarm?dir=${enc(key)}`); } catch (e) { fill(el, backLink('evaluate'), errorBox(e)); return; }
  const models = Object.entries(s.models || {});
  fill(el, 
    h('div', { class: 'page-head' },
      h('div', {}, backLink('evaluate'), h('h1', {}, s.name, ' ', chip(s.status)),
        h('p', { class: 'lede' }, `${s.kind === 'single' ? 'Single agent' : `Swarm of ${plural(s.agents.length, 'agent')}`} from run ${s.run || '?'}, generation ${s.generation ?? '?'}${(s.tasks || []).length ? ` · trained on ${s.tasks.join(', ')}` : ''}`),
        h('p', { class: 'muted small mono', style: { margin: '6px 0 0' } }, s.key, s.exported_at ? ` · exported ${s.exported_at}` : '', s.git_commit ? ` · commit ${String(s.git_commit).slice(0, 10)}` : '')),
      h('div', { class: 'row wrap', style: { gap: '10px' } },
        s.status === 'ready' ? h('a', { class: 'btn primary', href: `#/agents?tab=evaluate&swarm=${enc(s.key)}` }, 'Run it on a problem') : null,
        s.source_key ? h('a', { class: 'btn', href: href('run', s.source_key) }, 'Training run') : null)),
    h('div', { class: 'tiles', style: { marginBottom: '24px' } },
      tile('Agents', String(s.agents.length)), tile('Notes of generation', ok(s.generation) ? String(s.generation) : '–', 'the most trained state the run saved'),
      tile('Shared notes', String((s.shared || []).length)), tile('Personas', String(Object.keys(s.personas || {}).length))),
    h('div', { class: 'card', style: { marginBottom: '24px' } }, h('h2', {}, 'Agents'),
      h('p', { class: 'muted small', style: { marginTop: '-6px' } }, 'The context is exactly what the agent gets on a new problem (ace.infer); on a problem it trained on it also gets that task’s last feedback.'),
      h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['agent', 'persona', 'own notes', 'context'].map((x, i) => h('th', { class: [2, 3].includes(i) ? 'num' : null }, x)))),
        h('tbody', {}, (s.agent_list || []).map(a => h('tr', {}, h('td', { class: 'mono strong' }, a.agent),
          h('td', { class: 'small' }, a.persona || h('span', { class: 'muted' }, 'none')), h('td', { class: 'num' }, a.notes),
          h('td', { class: 'num' }, a.context_chars ? `${fmtInt(a.context_chars)} chars` : '–'))))),
      (s.agent_list || []).filter(a => a.context).map(a => lazyText(s.key, a.context, true))),
    h('div', { class: 'card', style: { marginBottom: '24px' } }, h('h2', {}, `Shared playbook (${(s.shared || []).length})`),
      noteList(s.shared, Object.keys(s.playbooks || {}).length ? `one playbook per task: ${Object.entries(s.playbooks).map(([t, n]) => `${t} (${n})`).join(', ')}` : 'empty')),
    models.length ? h('div', { class: 'card' }, h('h2', {}, 'Models it was trained with'), h('table', { class: 'kv' }, h('tbody', {}, models.map(([k, v]) => h('tr', {}, h('th', {}, k), h('td', { class: 'mono' }, String(v))))))) : null);
}

// ---------------------------------------------------------------- comparison

async function comparePage(el, key) {
  el.append(loading());
  let c;
  try { c = await api(`/api/ace/compare?dir=${enc(key)}`); } catch (e) { fill(el, backLink('evaluate'), errorBox(e)); return; }
  fill(el, 
    h('div', { class: 'page-head' }, h('div', {}, backLink('evaluate'), h('h1', {}, 'Swarm vs single agent ', chip(c.status)),
      h('p', { class: 'lede mono' }, `${c.swarm || '?'}  vs  ${c.single || '?'}`))),
    c.png ? h('div', { class: 'card', style: { marginBottom: '24px' } }, h('img', { src: fileUrl(key, 'compare.png'), alt: 'base score against generations, cumulative $ and wall time', style: { maxWidth: '100%' } })) : null,
    c.compare_md ? h('div', { class: 'card' }, md(c.compare_md)) : h('p', { class: 'muted' }, 'No compare.md yet.'));
}

// ====================================================================== markdown (safe: text nodes only)

function md(text) {
  const root = h('div', { class: 'ag-md' });
  const lines = String(text || '').split('\n');
  const cells = l => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(x => x.trim());
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const buf = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) buf.push(lines[i]);
      i++;
      root.append(h('pre', { class: 'code' }, buf.join('\n')));
    } else if (/^#{1,4}\s/.test(line)) {
      const m = /^(#{1,4})\s+(.*)$/.exec(line);
      root.append(h(`h${Math.min(4, m[1].length + 1)}`, {}, inline(m[2])));
      i++;
    } else if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      const head = cells(line);
      const body = [];
      for (i += 2; i < lines.length && /^\s*\|/.test(lines[i]); i++) body.push(cells(lines[i]));
      root.append(h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, head.map(x => h('th', {}, inline(x))))),
        h('tbody', {}, body.map(r => h('tr', {}, r.map(x => h('td', { class: /^[-+]?[\d.,$%e]+$/i.test(x) ? 'num' : null }, inline(x)))))))));
    } else if (/^\s*[-*]\s+/.test(line)) {
      const ul = h('ul');
      for (; i < lines.length && /^\s*[-*]\s+/.test(lines[i]); i++) ul.append(h('li', {}, inline(lines[i].replace(/^\s*[-*]\s+/, ''))));
      root.append(ul);
    } else if (!line.trim()) {
      i++;
    } else {
      const buf = [line];
      for (i++; i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*\||\s*[-*]\s)/.test(lines[i]); i++) buf.push(lines[i]);
      root.append(h('p', {}, inline(buf.join(' '))));
    }
  }
  return root;
}

function inline(s) {
  const out = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index));
    out.push(m[0][0] === '`' ? h('code', {}, m[0].slice(1, -1)) : h('b', {}, m[0].slice(2, -2)));
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

// ====================================================================== styles (this page only)

function ensureStyle() {
  if (document.getElementById('agents-css')) return;
  document.head.append(h('style', { id: 'agents-css' }, `
.ag-picker { display: flex; flex-direction: column; gap: 10px; }
.ag-group { font-size: 12.5px; color: var(--muted); font-weight: 600; margin-bottom: 4px; }
.ag-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.ag-chip { font: inherit; font-size: 13.5px; padding: 3px 11px; border-radius: 999px; border: 1px solid var(--line, #e6e5df); background: var(--surface); color: var(--ink-2); cursor: pointer; }
.ag-chip:hover { border-color: var(--axis); color: var(--ink); }
.ag-chip.train { background: var(--accent-wash); border-color: rgba(42, 120, 214, 0.45); color: var(--accent-ink); font-weight: 600; }
.ag-chip.hold { background: #fdf1ea; border-color: rgba(235, 104, 52, 0.55); color: #8a3a12; font-weight: 600; }
.ag-chip .tag { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; margin-left: 6px; opacity: 0.85; }
.legend .key svg { vertical-align: middle; }
.ag-matrix td { vertical-align: top; }
.ag-run { display: block; min-width: 58px; padding: 3px 8px; margin: 2px 0; border-radius: 6px; cursor: pointer; font-variant-numeric: tabular-nums; font-weight: 600; border: 2px solid transparent; }
.ag-run:hover { border-color: var(--axis); }
.ag-run.on { border-color: var(--ink); }
.ag-run .ag-task { display: block; font-size: 11.5px; font-weight: 500; opacity: 0.85; }
tr.ag-sel td { background: var(--accent-wash); }
.ag-panel { margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--line, #e6e5df); }
.ag-imgs { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.ag-imgs figure { margin: 0; }
.ag-imgs img { width: 100%; border: 1px solid var(--grid); border-radius: 8px; background: #fff; display: block; }
.ag-ctx { white-space: pre-wrap; font-family: var(--mono); font-size: 12.5px; line-height: 1.5; max-height: 560px; overflow: auto; background: var(--surface-2); border-radius: 8px; padding: 12px 14px; margin: 0; }
.ag-ctx .add { background: #e3f3e3; display: inline; }
.ag-ctx .del { background: #fbe9e9; color: #7a1d1b; text-decoration: line-through; display: inline; }
.ag-notes { padding-left: 18px; margin: 4px 0 8px; }
.ag-notes li { margin: 3px 0; font-size: 14.5px; }
.ag-notes .badge { margin-right: 4px; font-size: 11.5px; }
.badge.ag-new { background: #e6f0fb; color: var(--accent-ink); }
.badge.ag-cited { background: #e3f3e3; color: var(--good-ink); }
.ag-was { color: #7a1d1b; text-decoration: line-through; }
.ag-now { color: var(--good-ink); }
.ag-trace { display: flex; flex-direction: column; gap: 6px; max-height: 640px; overflow: auto; }
.ag-trace pre { margin: 2px 0 0; white-space: pre-wrap; font-size: 12.5px; background: var(--surface-2); border-radius: 6px; padding: 6px 9px; }
.ag-trace .result pre { color: var(--ink-2); }
.ag-trace .result.err pre { background: #fdf0ef; color: #7a1d1b; }
.ag-trace .think { font-size: 13.5px; color: var(--ink-2); font-style: italic; }
.ag-frame { width: 100%; height: 900px; border: 1px solid var(--grid); border-radius: 10px; background: #fff; }
.ag-md table.data { margin: 8px 0 14px; font-size: 14px; }
.ag-md h2 { font-size: 19px; margin-top: 18px; } .ag-md h3, .ag-md h4 { font-size: 16px; margin-top: 14px; }
.ag-md p { max-width: 90ch; }
.ag-probsel { display: flex; flex-direction: column; gap: 6px; }
.ag-form > .stack > p.muted:first-child { display: none; }   /* the tool's description: the card says it */
.ag-form .field input[type=text] { min-width: 380px; }
`));
}
