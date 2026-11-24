import { describe, expect, it } from 'vitest';
import { avgExpectedError, avgTestError, biasVariance, errorCurve, fitPoly, legendre, lstsq, mse, sample } from './fit';
import { CurveNet, history } from './net';

describe('least squares', () => {
  it('recovers an exact polynomial', () => {
    const xs = [-0.9, -0.5, -0.1, 0.2, 0.6, 0.95];
    const A = xs.map((x) => legendre(x, 2));
    const c = lstsq(A, xs.map((x) => 1 - 2 * x + 3 * x * x));
    // 1 − 2x + 3x² = 2·P0 − 2·P1 + 2·P2  (since x² = (2·P2 + 1)/3)
    expect(c[0]).toBeCloseTo(2, 9);
    expect(c[1]).toBeCloseTo(-2, 9);
    expect(c[2]).toBeCloseTo(2, 9);
  });
  it('training error never increases with degree, and degree n−1 interpolates', () => {
    const c = errorCurve(sample(12, 3), 11);
    for (let d = 1; d <= 11; d++) expect(c[d].train).toBeLessThanOrEqual(c[d - 1].train + 1e-9);
    expect(c[11].train).toBeLessThan(1e-12);
  });
  it('minimum-norm fits interpolate when there are more coefficients than points', () => {
    const s = sample(12, 2);
    expect(mse(fitPoly(s, 30), s)).toBeLessThan(1e-12);
  });
});

/** Every number quoted in the lesson text, pinned. */
describe('numbers quoted in the overfitting lesson', () => {
  it('capacity dial default sample (12 points, seed 1)', () => {
    const c = errorCurve(sample(12, 1), 11);
    expect(c[0].test).toBeCloseTo(1.07, 2);
    expect(c[3].test).toBeCloseTo(0.13, 2);
    expect(c[11].test).toBeGreaterThan(60);
    expect(c[11].test).toBeLessThan(80);
    // regimes: 3–5 about right, 7+ overfit
    for (const d of [3, 4, 5]) expect(c[d].test).toBeLessThan(0.14);
    for (const d of [7, 8, 9, 10, 11]) expect(c[d].test).toBeGreaterThan(1);
  });
  it('more data moves the sweet spot (averaged over 100 training sets)', () => {
    expect(avgTestError(12, 11, 100)).toBeGreaterThan(1000);
    expect(avgTestError(12, 11, 100)).toBeLessThan(2000);
    expect(avgTestError(100, 11, 100)).toBeCloseTo(0.099, 3);
    expect(avgTestError(100, 3, 100)).toBeCloseTo(0.093, 3);
  });
  it('bias–variance numbers, and the decomposition adds up', () => {
    const d1 = biasVariance(12, 1, 200);
    expect(d1.bias2).toBeCloseTo(0.095, 3);
    expect(d1.variance).toBeCloseTo(0.016, 3);
    const d3 = biasVariance(12, 3, 200);
    expect(d3.bias2).toBeCloseTo(0.001, 3);
    expect(d3.variance).toBeCloseTo(0.031, 3);
    const d9 = biasVariance(12, 9, 200);
    expect(d9.variance).toBeCloseTo(5.8, 1);
    for (const r of [d1, d3, d9]) expect(Math.abs(r.total - r.direct) / r.direct).toBeLessThan(1e-9);
  });
  it('double descent past the interpolation threshold', () => {
    const at = (d: number) => avgExpectedError(12, d, 60);
    expect(at(11)).toBeGreaterThan(2000);
    expect(at(11)).toBeLessThan(3000);
    expect(at(15)).toBeCloseTo(0.58, 2);
    expect(at(20)).toBeCloseTo(0.27, 2);
    expect(at(60)).toBeCloseTo(0.17, 2);
    expect(at(3)).toBeCloseTo(0.12, 2);
    expect(at(60)).toBeGreaterThan(at(3)); // never beats the best small model here
  });
  it('the network: 1,153 parameters, best test ≈ 0.13 around step 350, then overfits', () => {
    const s = sample(12, 1);
    expect(new CurveNet(s).net.paramCount).toBe(1153);
    const h = history(s, 4000, 10);
    const best = h.reduce((b, r) => (r.test < b.test ? r : b));
    expect(best.test).toBeCloseTo(0.13, 2);
    expect(best.step).toBeGreaterThan(250);
    expect(best.step).toBeLessThan(450);
    const at2000 = h.find((r) => r.step === 2000)!;
    expect(at2000.train).toBeLessThan(1e-3);
    expect(at2000.test).toBeGreaterThan(0.2);
    expect(h[h.length - 1].test).toBeLessThan(0.25); // "around 0.2, not 70"
  }, 30000);
});
