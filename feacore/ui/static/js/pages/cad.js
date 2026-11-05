// Your CAD: placeholder page (filled in by the dashboard consolidation).
import { h } from '../api.js';

export async function render(el) {
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Your CAD'),
    h('p', { class: 'lede' }, 'Bring your own STEP file: turn it into a problem the agents and the solver can work on.'))));
}
