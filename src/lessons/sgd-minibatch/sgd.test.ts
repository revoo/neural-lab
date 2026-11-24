import { describe, expect, it } from 'vitest';
import { batchGradients, best, grad, loss, lossFast, makeData, run, spread, stats } from './sgd';

const d = makeData();

describe('mini-batch gradients', () => {
  it('the average of the gradients over disjoint batches is the full gradient', () => {
    const B = 50;
    let gw = 0;
    let gb = 0;
    for (let s = 0; s < 1000; s += B) {
      const [a, c] = grad(d, 3, 20, Array.from({ length: B }, (_, k) => s + k));
      gw += a / (1000 / B);
      gb += c / (1000 / B);
    }
    const [tw, tb] = grad(d, 3, 20);
    expect(gw).toBeCloseTo(tw, 9);
    expect(gb).toBeCloseTo(tb, 9);
  });
  it('the estimates scatter about 1/√B as much', () => {
    const g = grad(d, 0, 10);
    const s4 = spread(batchGradients(d, 0, 10, 4, 600, 1), g);
    const s64 = spread(batchGradients(d, 0, 10, 64, 600, 2), g);
    expect(s4 / s64).toBeGreaterThan(3.3);
    expect(s4 / s64).toBeLessThan(4.7);
    expect(spread(batchGradients(d, 0, 10, 1000, 5, 3), g)).toBeCloseTo(0, 9);
  });
  it('the fast loss matches the direct one', () => {
    const s = stats(d);
    for (const [w, b] of [[0, 10], [12, 30], [-3, 44]]) expect(lossFast(s, w, b)).toBeCloseTo(loss(d, w, b), 8);
  });
  it('the least-squares solution has zero gradient', () => {
    const b = best(d);
    const [gw, gb] = grad(d, b.w, b.b);
    expect(Math.abs(gw) + Math.abs(gb)).toBeLessThan(1e-9);
  });
});

describe('runs on a budget', () => {
  it('counts examples, not steps', () => {
    expect(run(d, 1000, 0.5, 3000, [0, 10], 1).steps).toBe(3);
    expect(run(d, 32, 0.1, 3000, [0, 10], 1).steps).toBe(93);
    expect(run(d, 1, 0.02, 3000, [0, 10], 1).steps).toBe(3000);
  });
  it('mini-batches get much closer than full batch on the same budget', () => {
    const opt = best(d).loss;
    const full = run(d, 1000, 0.5, 3000, [0, 10], 1).points.at(-1)!.loss - opt;
    const mini = run(d, 32, 0.1, 3000, [0, 10], 1).points.at(-1)!.loss - opt;
    expect(full).toBeGreaterThan(4);
    expect(mini).toBeLessThan(0.1);
  });
});
