import { describe, expect, it } from 'vitest';
import { bestFit, bestPeak, lossAt, makeData, makeSchedule, meanFinalExcess, stats, train } from './schedules';

const d = makeData();
const s = stats(d);
const seeds = Array.from({ length: 20 }, (_, k) => k + 1);
const grid = Array.from({ length: 25 }, (_, k) => Math.round(10 ** (-3 + k * 0.125) * 1e5) / 1e5);

describe('schedules', () => {
  it('have the advertised shapes', () => {
    const T = 100;
    expect(makeSchedule('constant', 0.1)(57, T)).toBe(0.1);
    expect(makeSchedule('step', 0.1)(10, T)).toBeCloseTo(0.1, 12);
    expect(makeSchedule('step', 0.1)(60, T)).toBeCloseTo(0.01, 12);
    expect(makeSchedule('step', 0.1)(80, T)).toBeCloseTo(0.001, 12);
    expect(makeSchedule('cosine', 0.1)(0, T)).toBeCloseTo(0.1, 12);
    expect(makeSchedule('cosine', 0.1)(50, T)).toBeCloseTo(0.05, 12);
    const w = makeSchedule('warmup-cosine', 0.1);
    expect(w(0, T)).toBeCloseTo(0.01, 12);
    expect(w(9, T)).toBeCloseTo(0.1, 12);
    const c = makeSchedule('custom', 0, [0.1, 0.001]);
    expect(c(0, T)).toBeCloseTo(0.1, 12);
    expect(c(99, T)).toBeCloseTo(0.001, 12);
    expect(c(49.5, T)).toBeCloseTo(0.01, 9); // geometric midpoint
  });
  it('the fast loss agrees with the best fit', () => {
    const b = bestFit(s);
    expect(lossAt(s, b.w + 0.1, b.b)).toBeGreaterThan(b.loss);
    expect(lossAt(s, b.w, b.b - 0.1)).toBeGreaterThan(b.loss);
  });
  it('a bigger constant step size leaves more jitter in the long run', () => {
    const small = meanFinalExcess(d, s, makeSchedule('constant', 0.01), 4000, 4, seeds.slice(0, 8));
    const big = meanFinalExcess(d, s, makeSchedule('constant', 0.1), 4000, 4, seeds.slice(0, 8));
    expect(big).toBeGreaterThan(5 * small);
  });
  it('decaying schedules beat the best constant step size on a 400-step budget', () => {
    const constant = bestPeak(d, s, 'constant', 400, 4, seeds, grid).excess;
    const cosine = bestPeak(d, s, 'cosine', 400, 4, seeds, grid).excess;
    const step = bestPeak(d, s, 'step', 400, 4, seeds, grid).excess;
    expect(cosine).toBeLessThan(0.6 * constant);
    expect(step).toBeLessThan(constant);
  });
  it('is reproducible for a given seed', () => {
    const a = train(d, s, makeSchedule('cosine', 0.04), 400, 4, 3).final;
    const b = train(d, s, makeSchedule('cosine', 0.04), 400, 4, 3).final;
    expect(a).toBe(b);
  });
});

describe('designer reference', () => {
  it('the knot search beats the best cosine schedule', async () => {
    const { searchKnots } = await import('./schedules');
    const found = searchKnots(d, s, 400, 4, seeds);
    expect(found.excess).toBeLessThan(bestPeak(d, s, 'cosine', 400, 4, seeds, grid).excess);
    expect(found.excess).toBeLessThan(0.06);
  });
});
