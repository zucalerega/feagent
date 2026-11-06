// Tasks: every long-running tool started from the dashboard (agent runs, gold meshing, nonlinear crush, ...).
// #/tasks lists them; #/tasks/<id> shows one with its live log.
import { api, h, clear, loading, errorBox, timeAgo } from '../api.js';
import { taskTable, taskLog, taskChip, cancelButtons } from '../tasks.js';

export async function render(el, args) {
  if (args && args[0]) return detail(el, args[0]);
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Tasks'),
    h('p', { class: 'lede' }, 'Long-running tools started from the dashboard: agent training and evaluation runs, gold-mesh generation, nonlinear crush runs. Each runs in its own process on the server, with its output saved in the run store.'))));
  const t = taskTable({}, { limit: 200 });
  el.append(h('div', { class: 'card flush' }, t.el));
  return t.stop;
}

async function detail(el, id) {
  el.append(loading());
  let t;
  try { t = await api(`/api/tasks/${id}`); } catch (e) { clear(el).append(errorBox(e)); return; }
  const log = taskLog(id);
  let timer = null, alive = true;
  const head = h('div');
  const drawHead = rec => clear(head).append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, rec.title, ' ', taskChip(rec.state)),
    h('p', { class: 'lede' }, h('code', {}, `python ${rec.argv.join(' ')}`)),
    h('p', { class: 'muted' }, `${rec.id} · created ${timeAgo(rec.created_at)}`, rec.out ? [' · output ', h('code', {}, rec.out)] : null,
      rec.queue_position ? ` · queue position ${rec.queue_position}` : null, rec.error ? ` · ${rec.error}` : null)),
    h('div', {}, cancelButtons(rec, refresh), ' ', h('a', { class: 'btn small', href: '#/tasks' }, 'All tasks'))));
  async function refresh() {
    try { t = await api(`/api/tasks/${id}`); } catch { return; }
    if (!alive) return;
    drawHead(t);
    if (t.state === 'queued' || t.state === 'running') timer = setTimeout(refresh, 3000);
  }
  clear(el).append(head, h('div', { class: 'card' }, h('h2', {}, 'Output'), log.el));
  drawHead(t);
  if (t.state === 'queued' || t.state === 'running') timer = setTimeout(refresh, 3000);
  return () => { alive = false; clearTimeout(timer); log.stop(); };
}
