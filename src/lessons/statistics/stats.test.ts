// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import {
  OPENING,
  SOURCES,
  WORKSHOP,
  accuracySE,
  bellGap,
  binomial,
  checkRecipe,
  convolve,
  distMean,
  distSd,
  mean,
  meanAbsDev,
  median,
  normalCdf,
  probLooksBetter,
  recipeDist,
  recipeMoments,
  sd,
  solveWorkshop,
  sumOf,
  uniformInts,
  variance,
  type Recipe,
} from './stats';

const recipe = (counts: Partial<Recipe['counts']>, plus = 0): Recipe => ({ counts: { d2: 0, d4: 0, d6: 0, d8: 0, d12: 0, d20: 0, ...counts }, plus });

describe('mean, median and spread', () => {
  it('pins the spread-explorer numbers: one outlier doubles the SD and leaves the median alone', () => {
    const pts = [2, 3, 4, 5, 6, 7, 8];
    expect(mean(pts)).toBe(5);
    expect(variance(pts)).toBe(4);
    expect(sd(pts)).toBe(2);
    expect(median(pts)).toBe(5);
    expect(meanAbsDev(pts)).toBeCloseTo(12 / 7, 12);
    const out = [2, 3, 4, 5, 6, 7, 15];
    expect(mean(out)).toBe(6);
    expect(variance(out)).toBe(16);
    expect(sd(out)).toBe(4);
    expect(median(out)).toBe(5);
    expect(meanAbsDev(out)).toBeCloseTo(20 / 7, 12);
  });
});

describe('exact distributions of sums', () => {
  it('convolution matches the textbook 2d6 table and conserves probability', () => {
    const two = sumOf(uniformInts(1, 6), 2);
    expect(two.min).toBe(2);
    expect(two.p.map((q) => Math.round(q * 36))).toEqual([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]);
    expect(two.p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('means add and variances add: SD of a sum of n grows as √n, of an average shrinks as 1/√n', () => {
    const die = uniformInts(1, 6);
    expect(distSd(die)).toBeCloseTo(Math.sqrt(35 / 12), 12);
    for (const n of [1, 4, 9, 25]) {
      const s = sumOf(die, n);
      expect(distMean(s)).toBeCloseTo(3.5 * n, 9);
      expect(distSd(s)).toBeCloseTo(Math.sqrt(n) * Math.sqrt(35 / 12), 9);
      expect(distSd(s) / n).toBeCloseTo(Math.sqrt(35 / 12) / Math.sqrt(n), 9);
    }
    const r = recipe({ d4: 1, d8: 2 }, 3);
    const d = recipeDist(r);
    expect(recipeMoments(r).mean).toBeCloseTo(distMean(d), 10);
    expect(recipeMoments(r).sd).toBeCloseTo(distSd(d), 10);
  });

  it('agrees with simulation (a sanity check on the exact maths)', () => {
    const rng = createRng(7);
    const N = 200000;
    let hits = 0;
    for (let i = 0; i < N; i++) if (1 + rng.int(6) + 1 + rng.int(6) === 7) hits++;
    expect(Math.abs(hits / N - 6 / 36)).toBeLessThan(0.004);
  });

  it('pins the bell-gap numbers quoted in the lesson', () => {
    const die = uniformInts(1, 6);
    expect(bellGap(die)).toBeCloseTo(0.1875, 4);
    expect(bellGap(uniformInts(1, 12))).toBeCloseTo(0.1974, 4);
    expect(bellGap(sumOf(die, 2))).toBeCloseTo(0.0581, 4);
    expect(bellGap(sumOf(die, 3))).toBeCloseTo(0.0283, 4);
    expect(bellGap(sumOf(die, 5))).toBeCloseTo(0.015, 3);
    expect(bellGap(sumOf(die, 30))).toBeCloseTo(0.002, 3);
    const shot = SOURCES.ticket.dist;
    expect(bellGap(sumOf(shot, 5))).toBeCloseTo(0.172, 3);
    expect(bellGap(sumOf(shot, 30))).toBeCloseTo(0.062, 3);
    expect(bellGap(sumOf(shot, 100))).toBeCloseTo(0.034, 3);
    // lopsided sources get worse before they get better
    expect(bellGap(sumOf(shot, 5))).toBeGreaterThan(bellGap(sumOf(shot, 2)));
  });

  it('the normal CDF is accurate', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 7);
    expect(normalCdf(1.96)).toBeCloseTo(0.975, 4);
    expect(normalCdf(1) - normalCdf(-1)).toBeCloseTo(0.6827, 4);
    expect(normalCdf(2) - normalCdf(-2)).toBeCloseTo(0.9545, 4);
  });
});

describe('test-set noise', () => {
  it('pins the opening question: a 0.6-point gap on 500 examples is well within noise', () => {
    expect(OPENING.accA - OPENING.accB).toBeCloseTo(0.006, 10);
    expect(accuracySE(0.91, 500)).toBeCloseTo(0.0128, 4);
    expect(accuracySE(0.91, 5000)).toBeCloseTo(0.004, 4);
    const one = probLooksBetter(0.909, 500, 0.006);
    expect(one).toBeCloseTo(0.3916, 4);
    expect(2 * one).toBeCloseTo(0.7831, 4);
    expect(probLooksBetter(0.909, 5000, 0.006)).toBeCloseTo(0.1525, 4);
    expect(probLooksBetter(0.909, 20000, 0.006)).toBeCloseTo(0.0189, 4);
  });

  it('the exact binomial has mean np and SD √(np(1 − p))', () => {
    const b = binomial(500, 0.9);
    expect(b.p.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 9);
    expect(distMean(b)).toBeCloseTo(450, 6);
    expect(distSd(b)).toBeCloseTo(Math.sqrt(45), 6);
    // and probLooksBetter agrees with a brute-force double sum on a small case
    const n = 40;
    const bb = binomial(n, 0.7);
    let brute = 0;
    for (let a = 0; a <= n; a++) for (let c = 0; c <= n; c++) if (a - c >= 3) brute += bb.p[a] * bb.p[c];
    expect(probLooksBetter(0.7, n, 3 / n)).toBeCloseTo(brute, 12);
  });
});

describe('Lab: the dice workshop (exhaustive over 0…8 of each of six dice)', () => {
  it('counts the recipes that work and finds the fewest-dice, most bell-like one', () => {
    const [b1, b2, b3] = WORKSHOP;
    const r1 = solveWorkshop(b1);
    expect(r1.passing.length).toBe(3);
    expect(r1.fewest).toBe(3);
    expect(r1.best.r.counts.d6).toBe(3);
    expect(r1.best.gap).toBeCloseTo(0.0283, 4);
    const r2 = solveWorkshop(b2);
    expect(r2.passing.length).toBe(451);
    expect(r2.fewest).toBe(4);
    expect(r2.fewestRecipes.length).toBe(4);
    expect(r2.best.r).toEqual(recipe({ d8: 4 }, 12));
    const r3 = solveWorkshop(b3);
    expect(r3.passing.length).toBe(5);
    expect(r3.fewest).toBe(4);
    expect(r3.fewestRecipes.length).toBe(1);
    expect(r3.best.r).toEqual(recipe({ d2: 1, d4: 3 }, 41));
  });

  it('checks the examples the lesson discusses', () => {
    const [b1] = WORKSHOP;
    expect(checkRecipe(b1, recipe({ d20: 1 })).pass).toBe(false); // right-ish mean, flat
    expect(checkRecipe(b1, recipe({ d20: 1 })).bellOk).toBe(false);
    expect(checkRecipe(b1, recipe({ d6: 3 })).pass).toBe(true);
    expect(checkRecipe(b1, recipe({ d4: 1, d6: 1, d8: 1 })).pass).toBe(true);
    expect(checkRecipe(b1, recipe({ d2: 1, d8: 2 })).pass).toBe(true);
    expect(bellGap(convolve(uniformInts(1, 4), uniformInts(1, 8)))).toBeCloseTo(0.0997, 4);
    expect(bellGap(uniformInts(1, 20))).toBeCloseTo(0.197, 3);
    const twoD12 = checkRecipe(WORKSHOP[1], recipe({ d12: 2 }, 17));
    expect(twoD12.meanOk && twoD12.sdOk).toBe(true);
    expect(twoD12.bellOk).toBe(false);
    expect(twoD12.gap).toBeCloseTo(0.0525, 4);
    const steady = recipeMoments(recipe({ d2: 1, d4: 3 }, 41));
    expect(steady.mean).toBe(50);
    expect(steady.sd).toBeCloseTo(2, 12);
  });
});
