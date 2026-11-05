// Agents: placeholder page (filled in by the dashboard consolidation).
import { h } from '../api.js';

export async function render(el) {
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Agents'),
    h('p', { class: 'lede' }, 'Train and evaluate agent swarms and single agents: launch runs, watch them live, compare, export a trained swarm and run it on new problems.'))));
}
