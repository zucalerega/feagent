// Gold benchmarks: placeholder page (filled in by the dashboard consolidation).
import { h } from '../api.js';

export async function render(el) {
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Gold benchmarks'),
    h('p', { class: 'lede' }, 'Real CAD parts with certified reference answers: the BC sets, their nested study meshes and re-solves.'))));
}
