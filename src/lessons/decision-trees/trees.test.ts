// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAB_BRIEFS,
  accuracyOf,
  bestGiniSplit,
  evaluateSplit,
  fitClassTree,
  fitRandomForest,
  getGiniData,
  getMessyBench,
  getNoisy2DData,
  giniCounts,
  labExhaustive,
  predictClassTree,
  predictForest,
} from './trees';

const p1 = (v: number) => (100 * v).toFixed(1);

describe('decision tree & ensemble core properties', () => {
  it('Gini impurity is 0 for pure nodes and 0.5 for a 50/50 split', () => {
    expect(giniCounts(10, 0)).toBe(0);
    expect(giniCounts(0, 10)).toBe(0);
    expect(giniCounts(10, 10)).toBeCloseTo(0.5, 12);
    expect(giniCounts(3, 1)).toBeCloseTo(0.375, 12);
  });

  it('decision trees are strictly invariant to monotone feature scaling (1,000x column rescale)', () => {
    const { train, test } = getNoisy2DData();
    const tRaw = fitClassTree(train, { maxDepth: 4, minLeaf: 2 });
    const trainScaled = train.map((s) => ({ x: [s.x[0], s.x[1] * 1000], y: s.y }));
    const testScaled = test.map((s) => ({ x: [s.x[0], s.x[1] * 1000], y: s.y }));
    const tScaled = fitClassTree(trainScaled, { maxDepth: 4, minLeaf: 2 });
    for (let i = 0; i < test.length; i++) {
      expect(predictClassTree(tScaled, testScaled[i].x)).toBeCloseTo(predictClassTree(tRaw, test[i].x), 12);
    }
  });
});

describe('numbers quoted in the decision-trees lesson', () => {
  it('Gini split explorer: optimal split x1 <= 5.19 drops Gini 0.486 -> 0.248 (gain 0.238)', () => {
    const gd = getGiniData();
    const best = bestGiniSplit(gd, [0, 1], 1)!;
    expect(best.feature).toBe(0);
    expect(best.threshold.toFixed(2)).toBe('5.19');
    const sp = evaluateSplit(gd, best.feature, best.threshold);
    expect(sp.parentGini.toFixed(3)).toBe('0.486');
    expect(sp.weightedGini.toFixed(3)).toBe('0.248');
    expect(sp.gain.toFixed(3)).toBe('0.238');
  });

  it('Single tree vs Random Forest on noisy 2D: tree depth 10 hits 99.3% train / 84.8% test; RF 40 trees hits 98.8% test', () => {
    const n2 = getNoisy2DData();
    const t3 = fitClassTree(n2.train, { maxDepth: 3, minLeaf: 1 });
    const t10 = fitClassTree(n2.train, { maxDepth: 10, minLeaf: 1 });
    expect(p1(accuracyOf(n2.test, (x) => predictClassTree(t3, x)))).toBe('90.8');
    expect(p1(accuracyOf(n2.train, (x) => predictClassTree(t10, x)))).toBe('99.3');
    expect(p1(accuracyOf(n2.test, (x) => predictClassTree(t10, x)))).toBe('84.8');

    const rf1 = fitRandomForest(n2.train, { nTrees: 1, maxDepth: 8, minLeaf: 2, maxFeatures: 1, seed: 202 });
    const rf40 = fitRandomForest(n2.train, { nTrees: 40, maxDepth: 8, minLeaf: 2, maxFeatures: 1, seed: 202 });
    expect(p1(accuracyOf(n2.test, (x) => predictForest(rf1, x)))).toBe('79.2');
    expect(p1(accuracyOf(n2.test, (x) => predictForest(rf40, x)))).toBe('98.8');
  });

  it('Messy 8-column benchmark: kNN raw 70.0%, kNN scaled 81.2%, Tree 91.2%, RF 91.9%, GBT peaks at r5 (93.1%) then overfits to 89.2%', () => {
    const b = getMessyBench();
    expect(p1(b.knnRaw)).toBe('70.0');
    expect(p1(b.knnScaled)).toBe('81.2');
    expect(p1(b.singleTree)).toBe('91.2');
    expect(p1(b.randomForest)).toBe('91.9');
    expect(p1(b.gbt)).toBe('89.2');
    const r5 = b.gbtRoundAccs.find((r) => r.round === 5)!;
    expect(p1(r5.testAcc)).toBe('93.1');
    expect(p1(b.rfImportances[0] + b.rfImportances[1])).toBe('61.7');
  });

  it('Lab: each tree family wins one brief with fewest total leaves (Tree 3, GBT 40, RF 30)', () => {
    const [b1, b2, b3] = LAB_BRIEFS.map(labExhaustive);

    expect([b1.total, b1.passing, b1.bestLeaves, p1(b1.bestAccAtMin)]).toEqual([35, 26, 3, '99.6']);
    expect(b1.best[0].cfg.method).toBe('tree');

    expect([b2.total, b2.passing, b2.bestLeaves, p1(b2.bestAccAtMin)]).toEqual([35, 2, 40, '98.8']);
    expect(b2.best[0].cfg).toEqual({ method: 'gbt', maxDepth: 3, nTrees: 5 });

    expect([b3.total, b3.passing, b3.bestLeaves, p1(b3.bestAccAtMin)]).toEqual([35, 6, 30, '97.6']);
    expect(b3.best[0].cfg).toEqual({ method: 'rf', maxDepth: 1, nTrees: 15 });
  }, 15_000);
});
