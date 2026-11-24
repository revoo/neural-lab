import { describe, expect, it } from 'vitest';
import { evaluate, foldNet, linearPieces, mlpParams, paramCount, neuronCount, wideNet, wideSawtooth } from './pieces';

/** Independent check: count slope changes by dense sampling (only valid when pieces are much wider than the grid). */
function sampledPieces(f: (x: number) => number, lo: number, hi: number, n = 200000) {
  const h = (hi - lo) / n;
  let pieces = 1;
  let prev = (f(lo + h) - f(lo)) / h;
  let pending = false;
  for (let k = 1; k < n; k++) {
    const s = (f(lo + (k + 1) * h) - f(lo + k * h)) / h;
    if (Math.abs(s - prev) > 1e-6 * Math.max(1, Math.abs(prev))) {
      // a kink inside a sample interval produces one in-between slope; count the change once
      if (!pending) pieces++;
      pending = true;
    } else pending = false;
    prev = s;
  }
  return pieces;
}

describe('fold network', () => {
  it('computes the tent map once per layer', () => {
    const t = (u: number) => (u < 0.5 ? 2 * u : 2 - 2 * u);
    for (const L of [1, 2, 3, 5])
      for (const x of [0, 0.1, 0.33, 0.5, 0.77, 1]) {
        let v = x;
        for (let k = 0; k < L; k++) v = t(v);
        expect(evaluate(foldNet(L), x)).toBeCloseTo(v, 9);
      }
  });

  it('has 2^L pieces and 6L + 1 parameters', () => {
    for (let L = 1; L <= 10; L++) {
      const net = foldNet(L);
      expect(linearPieces(net, 0, 1).pieces).toBe(2 ** L);
      expect(paramCount(net)).toBe(6 * L + 1);
      expect(neuronCount(net)).toBe(2 * L);
    }
  });

  it('exact piece count agrees with dense sampling', () => {
    for (const L of [1, 3, 5]) expect(linearPieces(foldNet(L), 0, 1).pieces).toBe(sampledPieces((x) => evaluate(foldNet(L), x), 0, 1));
  });
});

describe('wide networks', () => {
  it('the wide sawtooth matches the fold network exactly', () => {
    for (const L of [1, 2, 4, 6])
      for (const x of [0, 0.03, 0.2, 0.5, 0.61, 0.99]) expect(evaluate(wideSawtooth(L), x)).toBeCloseTo(evaluate(foldNet(L), x), 9);
  });
  it('needs 2^L neurons for 2^L pieces on [0, 1]', () => {
    for (const L of [1, 3, 5]) {
      const net = wideSawtooth(L);
      expect(neuronCount(net)).toBe(2 ** L);
      expect(linearPieces(net, 0, 1).pieces).toBe(2 ** L);
      expect(paramCount(net)).toBe(3 * 2 ** L + 1);
    }
  });
  it('n neurons give n + 1 pieces on the whole line, never more', () => {
    for (const n of [1, 2, 5, 9]) {
      const net = wideNet(n);
      expect(neuronCount(net)).toBe(n);
      // all kinks (including the one at 0) are inside [−1, 2]
      expect(linearPieces(net, -1, 2).pieces).toBe(n + 1);
    }
  });
});

describe('parameter counts', () => {
  it('counts a few real shapes', () => {
    expect(mlpParams([784, 128, 64, 10])).toBe(784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10);
    expect(mlpParams([768, 3072, 768])).toBe(4722432);
  });
});
