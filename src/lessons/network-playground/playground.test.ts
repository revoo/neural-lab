import { describe, expect, it } from 'vitest';
import { MLP } from '../../lib/mlp';
import { createRng } from '../../lib/rng';
import { createOptimizer } from '../../lib/optimizers';
import { accuracy, forwardMasked, fullMask, importance, makeDataset, minimalNeurons, paramTotal, predict } from './playground';

const train = (net: MLP, name: 'circles' | 'xor' | 'moons' | 'clusters', steps = 600) => {
  const d = makeDataset(name);
  const opt = createOptimizer({ name: 'adam', lr: 0.03 });
  const Y = d.y.map((v) => [v]);
  for (let s = 0; s < steps; s++) {
    net.backprop(d.X, Y, 'bce');
    opt.step(net.params(), net.grads());
  }
  return d;
};

describe('network playground logic', () => {
  it('masked forward with everything on matches the library forward pass', () => {
    const net = new MLP([2, 4, 3, 1], ['tanh', 'relu', 'sigmoid'], createRng(2));
    for (const x of [[0.3, -0.2], [1, 1], [-1.2, 0.4]]) expect(predict(net, x, fullMask(net.sizes))).toBeCloseTo(net.predict(x)[0], 12);
  });

  it('switching off every neuron in the last hidden layer leaves only the output bias', () => {
    const net = new MLP([2, 3, 1], ['tanh', 'sigmoid'], createRng(4));
    net.b[1][0] = 0.7;
    const m = fullMask(net.sizes);
    m[1] = [false, false, false];
    const t = forwardMasked(net, [0.5, 0.5], m);
    expect(t.a[1]).toEqual([0, 0, 0]);
    expect(t.a[2][0]).toBeCloseTo(1 / (1 + Math.exp(-0.7)), 12);
  });

  it('datasets are balanced and deterministic', () => {
    for (const n of ['circles', 'xor', 'moons', 'clusters'] as const) {
      const a = makeDataset(n);
      expect(a.y.filter((v) => v === 1).length).toBe(60);
      expect(makeDataset(n).X).toEqual(a.X);
    }
  });

  it('a trained network solves circles, and the pruning search is exact', () => {
    const net = new MLP([2, 6, 1], ['tanh', 'sigmoid'], createRng(1));
    const d = train(net, 'circles', 800);
    expect(accuracy(net, d)).toBeGreaterThanOrEqual(0.95);
    const res = minimalNeurons(net, d, 0.95)!;
    expect(res).not.toBeNull();
    // the mask found really reaches the threshold…
    expect(accuracy(net, d, res.mask)).toBeGreaterThanOrEqual(0.95);
    // …and no mask with one fewer neuron does (brute-force check of the claim).
    const hidden = 6;
    for (let bits = 0; bits < 1 << hidden; bits++) {
      let on = 0;
      for (let j = 0; j < hidden; j++) if (bits & (1 << j)) on++;
      if (on >= res.count) continue;
      const m = fullMask(net.sizes);
      m[1] = Array.from({ length: hidden }, (_, j) => !!(bits & (1 << j)));
      expect(accuracy(net, d, m)).toBeLessThan(0.95);
    }
    // a circle needs at least three lines
    expect(res.count).toBeGreaterThanOrEqual(3);
  });

  it('importance is the accuracy lost by switching one neuron off', () => {
    const net = new MLP([2, 4, 1], ['tanh', 'sigmoid'], createRng(3));
    const d = train(net, 'xor');
    const m = fullMask(net.sizes);
    const imp = importance(net, d, m);
    const base = accuracy(net, d, m);
    m[1][2] = false;
    expect(imp[1][2]).toBeCloseTo(base - accuracy(net, d, m), 12);
    expect(imp[0]).toEqual([0, 0]);
  });

  it('counts parameters', () => {
    expect(paramTotal([2, 3, 1])).toBe(2 * 3 + 3 + 3 * 1 + 1);
    expect(paramTotal([2, 8, 8, 1])).toBe(24 + 72 + 9);
  });
});
