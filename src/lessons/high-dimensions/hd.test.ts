// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import {
  PACK_BRIEFS,
  ballInCube,
  checkPacking,
  cosineDensity,
  distancesFromQuery,
  lnGamma,
  log10BallInCube,
  maxPacking,
  meanContrast,
  probAbsCosAtLeast,
  sampleCosines,
  shellFraction,
} from './hd';

describe('volume', () => {
  it('pins the peel fractions quoted in the lesson', () => {
    expect(shellFraction(3, 0.01)).toBeCloseTo(0.0297, 4);
    expect(shellFraction(100, 0.01)).toBeCloseTo(0.634, 3);
    expect(shellFraction(768, 0.01)).toBeCloseTo(0.9996, 4);
  });

  it('ball-in-cube agrees with the known low-dimensional values and collapses fast', () => {
    expect(lnGamma(5)).toBeCloseTo(Math.log(24), 12);
    expect(lnGamma(0.5)).toBeCloseTo(0.5 * Math.log(Math.PI), 12);
    expect(ballInCube(2)).toBeCloseTo(Math.PI / 4, 12);
    expect(ballInCube(3)).toBeCloseTo(Math.PI / 6, 12);
    expect(ballInCube(10)).toBeCloseTo(0.00249, 5);
    expect(ballInCube(20)).toBeCloseTo(2.461e-8, 10);
    expect(log10BallInCube(768)).toBeCloseTo(-867.6, 1);
  });
});

describe('random directions', () => {
  it('the exact cosine density integrates to 1 and has variance 1/d', () => {
    for (const d of [3, 10, 100, 768]) {
      let area = 0;
      let second = 0;
      const n = 40000;
      for (let i = 0; i < n; i++) {
        const t = -1 + (2 * (i + 0.5)) / n;
        const f = cosineDensity(d, t) * (2 / n);
        area += f;
        second += t * t * f;
      }
      expect(area).toBeCloseTo(1, 5);
      expect(second).toBeCloseTo(1 / d, 5);
    }
  });

  it('pins the tail probabilities quoted in the lesson (3D: exactly uniform)', () => {
    expect(probAbsCosAtLeast(3, 0.1)).toBeCloseTo(0.9, 6);
    expect(probAbsCosAtLeast(3, 0.3)).toBeCloseTo(0.7, 6);
    expect(probAbsCosAtLeast(100, 0.1)).toBeCloseTo(0.3197, 3);
    expect(probAbsCosAtLeast(768, 0.1)).toBeCloseTo(0.00551, 4);
    const p = probAbsCosAtLeast(768, 0.3);
    expect(p).toBeGreaterThan(1.7e-17);
    expect(p).toBeLessThan(2.0e-17);
  });

  it('sampled cosines have spread close to 1/√d', () => {
    for (const d of [10, 768]) {
      const cs = sampleCosines(d, 4000, 3);
      const m = cs.reduce((a, b) => a + b, 0) / cs.length;
      const sd = Math.sqrt(cs.reduce((a, b) => a + (b - m) ** 2, 0) / cs.length);
      expect(Math.abs(sd - 1 / Math.sqrt(d)) / (1 / Math.sqrt(d))).toBeLessThan(0.05);
    }
  });
});

describe('distance concentration (1,000 points, 20 seeds)', () => {
  it('pins the farthest/nearest contrasts quoted in the lesson', () => {
    const c10 = meanContrast(10);
    expect(c10.mean).toBeCloseTo(2.676, 2);
    const c100 = meanContrast(100);
    expect(c100.mean).toBeCloseTo(0.441, 2);
    const c1000 = meanContrast(1000);
    expect(c1000.mean).toBeCloseTo(0.115, 2);
    expect(c1000.min).toBeGreaterThan(0.09);
    expect(c1000.max).toBeLessThan(0.13);
    expect(meanContrast(2).mean).toBeGreaterThan(20);
    const ratios2: number[] = [];
    for (let s = 1; s <= 20; s++) ratios2.push(1 + distancesFromQuery(2, 1000, s * 101 + 2).contrast);
    ratios2.sort((a, b) => a - b);
    expect((ratios2[9] + ratios2[10]) / 2).toBeCloseTo(56, -1);
    expect(1 + meanContrast(10).mean).toBeCloseTo(3.7, 1);
    expect(1 + meanContrast(100).mean).toBeCloseTo(1.44, 2);
  });
});

describe('Lab: superposition (exhaustive maxima over all ±1 vectors)', () => {
  it('finds the true maximum for every brief, and Brief 4 is impossible', () => {
    const max = PACK_BRIEFS.map((b) => maxPacking(b.d, b.maxDot).length);
    expect(max).toEqual([4, 4, 8, 2, 16]);
    PACK_BRIEFS.forEach((b, i) => {
      if (b.impossible) expect(max[i]).toBeLessThan(b.goal);
      else expect(max[i]).toBeGreaterThanOrEqual(b.goal);
    });
  });

  it('the maximal arrangements really satisfy their briefs', () => {
    for (const b of PACK_BRIEFS.filter((x) => !x.impossible)) {
      const rows = maxPacking(b.d, b.maxDot);
      const c = checkPacking(b, rows);
      expect(c.bad).toEqual([]);
      expect(c.pass).toBe(true);
    }
    // the cube's four diagonals
    const cube = PACK_BRIEFS[0];
    expect(checkPacking(cube, [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]).pass).toBe(true);
    // a clash is reported
    expect(checkPacking(cube, [[1, 1, 1], [-1, -1, -1]]).bad).toEqual([[0, 1]]);
  });

  it('8 exactly perpendicular in 8D; in 7D a |cos| of 1/7 already fits 8', () => {
    expect(maxPacking(8, 0).length).toBe(8);
    expect(maxPacking(7, 1).length).toBe(8); // |cos| ≤ 1/7 already allows more than 7
  });
});
