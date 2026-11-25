// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAB_BRIEFS,
  applyScaler,
  dist,
  evaluateKnnLab,
  evaluateScalingTrap,
  fitScaler,
  getMoonsData,
  getNoiseDimCurve,
  knnMetrics,
  knnQuery,
  labExhaustive,
} from './knn';

const p1 = (v: number) => (100 * v).toFixed(1);

describe('kNN core primitives', () => {
  it('L2 and L1 distances and uniform vs distance-weighted voting', () => {
    expect(dist([0, 0], [3, 4], 'l2')).toBeCloseTo(5, 12);
    expect(dist([0, 0], [3, 4], 'l1')).toBeCloseTo(7, 12);
    const pts = [
      { x: [0.1, 0], y: 1 as const },
      { x: [2.0, 0], y: 0 as const },
      { x: [2.1, 0], y: 0 as const },
    ];
    // Uniform k=3: two class-0 votes beat one class-1 vote
    expect(knnQuery(pts, [0, 0], 3, 'l2', 'uniform').pred).toBe(0);
    // Distance-weighted k=3: very close class-1 point (1/0.1 = 10) beats two far class-0 points (~0.98)
    expect(knnQuery(pts, [0, 0], 3, 'l2', 'distance').pred).toBe(1);
  });

  it('standardiser centres training features to mean 0 and SD 1', () => {
    const train = [
      { x: [20, 40000], y: 0 as const },
      { x: [40, 80000], y: 1 as const },
      { x: [60, 120000], y: 1 as const },
    ];
    const sc = fitScaler(train);
    const scaled = applyScaler(train, sc);
    const m0 = scaled.reduce((s, p) => s + p.x[0], 0) / 3;
    const m1 = scaled.reduce((s, p) => s + p.x[1], 0) / 3;
    expect(m0).toBeCloseTo(0, 12);
    expect(m1).toBeCloseTo(0, 12);
    expect(scaled[0].x[0]).toBeCloseTo(scaled[0].x[1], 12);
  });
});

describe('numbers quoted in the kNN lesson', () => {
  it('moons: k=1 hits 100.0% train / 84.4% test; k=9 peaks at 93.9% test; k=45 underfits to 81.1%', () => {
    const moons = getMoonsData();
    expect(p1(knnMetrics(moons.train, moons.train, 1).accuracy)).toBe('100.0');
    expect(p1(knnMetrics(moons.train, moons.test, 1).accuracy)).toBe('84.4');
    expect(p1(knnMetrics(moons.train, moons.test, 9).accuracy)).toBe('93.9');
    expect(p1(knnMetrics(moons.train, moons.test, 45).accuracy)).toBe('81.1');
  });

  it('scaling trap (Age vs Income): k=5 jumps from 62.0% raw to 93.0% standardised', () => {
    expect(p1(evaluateScalingTrap(5, false).accuracy)).toBe('62.0');
    expect(p1(evaluateScalingTrap(5, true).accuracy)).toBe('93.0');
  });

  it('curse of dimensionality: adding noise columns collapses accuracy 96.0% -> 46.0% and far/near 30.13x -> 1.43x', () => {
    const curve = getNoiseDimCurve();
    const byNoise = Object.fromEntries(curve.map((r) => [r.dNoise, r]));
    expect(p1(byNoise[0].testAcc)).toBe('96.0');
    expect(byNoise[0].farOverNear.toFixed(2)).toBe('30.13');
    expect(p1(byNoise[5].testAcc)).toBe('67.5');
    expect(p1(byNoise[20].testAcc)).toBe('65.0');
    expect(p1(byNoise[80].testAcc)).toBe('46.0');
    expect(byNoise[80].farOverNear.toFixed(2)).toBe('1.43');
  });

  it('Lab: Rescue the kNN Classifier exhaustive optimums across all 3 briefs', () => {
    const [b1, b2, b3] = LAB_BRIEFS;
    expect(p1(evaluateKnnLab(b1, { scaled: false, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1] }).accuracy)).toBe('67.2');
    expect(p1(evaluateKnnLab(b2, { scaled: false, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1, 2, 3, 4, 5] }).accuracy)).toBe('80.0');
    expect(p1(evaluateKnnLab(b3, { scaled: false, metric: 'l2', weight: 'uniform', k: 25, cols: [0, 1] }).posRecall)).toBe('42.9');

    const [ex1, ex2, ex3] = LAB_BRIEFS.map(labExhaustive);
    expect([ex1.total, ex1.passing, p1(ex1.bestAcc)]).toEqual([48, 24, '99.4']);
    expect([ex2.total, ex2.passing, ex2.bestCols, p1(ex2.bestAcc)]).toEqual([1512, 31, 2, '95.0']);
    expect([ex3.total, ex3.passing, p1(ex3.bestAcc), p1(ex3.bestRecall)]).toEqual([24, 19, '96.0', '100.0']);
    expect(ex3.best[0].cfg).toEqual({ scaled: false, metric: 'l2', weight: 'distance', k: 15, cols: [0, 1] });
  }, 15_000);
});
