import { describe, expect, it } from 'vitest';
import { buildGraph, lossAfterUpdate, type Params } from './graph';

const P: Params = { x1: 2, w1: -0.5, x2: 1, w2: 0.8, b: 0.4, y: 0.6 };
const lossAt = (p: Params) => {
  const a = Math.tanh(p.x1 * p.w1 + p.x2 * p.w2 + p.b);
  return (a - p.y) ** 2;
};

describe('backprop lesson graph', () => {
  it('matches direct computation and numerical gradients', () => {
    const g = buildGraph(P);
    expect(g.nodes.L.v.data).toBeCloseTo(lossAt(P), 12);
    for (const k of ['w1', 'w2', 'b', 'x1', 'x2'] as const) {
      const h = 1e-6;
      const num = (lossAt({ ...P, [k]: P[k] + h }) - lossAt({ ...P, [k]: P[k] - h })) / (2 * h);
      expect(g.nodes[k].v.grad).toBeCloseTo(num, 6);
    }
  });
  it('accumulates two contributions into d (used twice by L = d·d)', () => {
    const g = buildGraph(P);
    const intoD = g.edges.filter((e) => e.from === 'd' && e.to === 'L');
    expect(intoD).toHaveLength(2);
    expect(g.nodes.d.v.grad).toBeCloseTo(2 * g.nodes.d.v.data, 12);
  });
  it('every node in the backward order only needs gradients that were already computed', () => {
    const g = buildGraph(P);
    const done = new Set<string>();
    for (const k of g.backwardOrder) {
      for (const e of g.edges.filter((ed) => ed.from === k)) expect(done.has(e.to)).toBe(true);
      done.add(k);
    }
  });
  it('a small update step reduces the loss', () => {
    const r = lossAfterUpdate(P, 0.1);
    expect(r.after).toBeLessThan(r.before);
  });
});
