// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  BRIEF4_DATA,
  OVERLAP_POINTS,
  RING_POINTS,
  RIVAL_LINES,
  SEPARABLE_POINTS,
  evalFeatureSubset,
  evaluateLine2D,
  generateSeparableTestCloud,
  searchCGrid,
  searchFeatureSubsets,
  solveHardMargin2D,
  solveHingeSubgradient,
  solveSoftMarginSMO,
} from './svm';

describe('SVM exact 2D hard-margin solver and independent cross-checks', () => {
  it('finds exact widest street and 3 support vectors on SEPARABLE_POINTS', () => {
    const opt = solveHardMargin2D(SEPARABLE_POINTS);
    expect(opt.separable).toBe(true);
    if (!opt.separable) return;
    expect(opt.w[0]).toBeCloseTo(0.5, 6);
    expect(opt.w[1]).toBeCloseTo(1.0, 6);
    expect(opt.b).toBeCloseTo(0.4, 6);
    expect(opt.streetWidth).toBeCloseTo(4 / Math.sqrt(5), 6);
    expect(opt.supportIndices).toEqual([0, 1, 5]);
  });

  it('agrees with both dual SMO and primal subgradient hinge-loss solvers', () => {
    const opt = solveHardMargin2D(SEPARABLE_POINTS);
    expect(opt.separable).toBe(true);
    if (!opt.separable) return;
    const X = SEPARABLE_POINTS.map((p) => [p.x, p.y]);
    const y = SEPARABLE_POINTS.map((p) => p.label);

    const smo = solveSoftMarginSMO(X, y, 1000, 120);
    expect(smo.w[0]).toBeCloseTo(opt.w[0], 4);
    expect(smo.w[1]).toBeCloseTo(opt.w[1], 4);
    expect(smo.b).toBeCloseTo(opt.b, 4);
    expect(smo.streetWidth).toBeCloseTo(opt.streetWidth, 4);
    expect(smo.supportIndices).toEqual([0, 1, 5]);

    const sub = solveHingeSubgradient(X, y, 1000, 15000, 0.2);
    expect(sub.w[0]).toBeCloseTo(opt.w[0], 2);
    expect(sub.w[1]).toBeCloseTo(opt.w[1], 2);
    expect(sub.b).toBeCloseTo(opt.b, 2);
  });

  it('moving a non-support vector inside its side leaves the street untouched, while moving a support vector changes it', () => {
    const base = solveHardMargin2D(SEPARABLE_POINTS);
    expect(base.separable).toBe(true);
    if (!base.separable) return;

    // Move non-support vector P4 (index 3) deeper into class +1
    const movedNonSV = SEPARABLE_POINTS.map((p, i) => (i === 3 ? { ...p, x: 0.8, y: 2.6 } : p));
    const afterNonSV = solveHardMargin2D(movedNonSV);
    expect(afterNonSV.separable).toBe(true);
    if (afterNonSV.separable) {
      expect(afterNonSV.streetWidth).toBeCloseTo(base.streetWidth, 9);
      expect(afterNonSV.w[0]).toBeCloseTo(base.w[0], 9);
      expect(afterNonSV.w[1]).toBeCloseTo(base.w[1], 9);
    }

    // Move support vector N1 (index 5) closer to the street
    const movedSV = SEPARABLE_POINTS.map((p, i) => (i === 5 ? { ...p, y: -0.5 } : p));
    const afterSV = solveHardMargin2D(movedSV);
    expect(afterSV.separable).toBe(true);
    if (afterSV.separable) {
      expect(afterSV.streetWidth).toBeLessThan(base.streetWidth - 0.3);
    }
  });

  it('pins the opening Predict rival lines: all separate train, widest street wins on fresh test cloud', () => {
    const cloud = generateSeparableTestCloud();
    const res = RIVAL_LINES.map((rl) => ({
      id: rl.id,
      train: evaluateLine2D(SEPARABLE_POINTS, rl.a, rl.b),
      test: evaluateLine2D(cloud, rl.a, rl.b),
    }));
    for (const r of res) expect(r.train.separates).toBe(true);
    expect(res[0].test.accuracy).toBeCloseTo(0.89, 3);
    expect(res[1].test.accuracy).toBeCloseTo(0.91, 3);
    expect(res[2].test.accuracy).toBeCloseTo(0.97, 3);
    expect(res[2].train.streetWidth).toBeCloseTo(4 / Math.sqrt(5), 3);
  });
});

describe('Soft margin C and Kernel lift lab solvers', () => {
  it('soft margin narrows the street and reduces slack as C rises on OVERLAP_POINTS', () => {
    const X = OVERLAP_POINTS.map((p) => [p.x, p.y]);
    const y = OVERLAP_POINTS.map((p) => p.label);
    const lowC = solveSoftMarginSMO(X, y, 0.05, 120);
    const midC = solveSoftMarginSMO(X, y, 0.5, 120);
    const highC = solveSoftMarginSMO(X, y, 50, 120);
    expect(lowC.streetWidth).toBeCloseTo(3.429, 2);
    expect(midC.streetWidth).toBeCloseTo(2.693, 2);
    expect(highC.streetWidth).toBeCloseTo(1.937, 2);
    expect(lowC.supportIndices.length).toBe(12);
    expect(midC.supportIndices.length).toBe(7);
    expect(highC.supportIndices.length).toBe(5);
    const lowSlack = lowC.slacks.reduce((s, v) => s + v, 0);
    const midSlack = midC.slacks.reduce((s, v) => s + v, 0);
    const highSlack = highC.slacks.reduce((s, v) => s + v, 0);
    expect(lowSlack).toBeCloseTo(4.34, 1);
    expect(midSlack).toBeCloseTo(3.09, 1);
    expect(highSlack).toBeCloseTo(2.73, 1);
    expect(lowC.streetWidth).toBeGreaterThan(highC.streetWidth);
    expect(highSlack).toBeLessThan(lowSlack);
  });

  it('Brief 3: rings are not linearly separable in raw (x1, x2), and exhaustive search finds 1-feature optima', () => {
    expect(evalFeatureSubset(RING_POINTS, ['x1', 'x2']).separates).toBe(false);
    expect(evalFeatureSubset(RING_POINTS, ['r2']).separates).toBe(true);
    expect(evalFeatureSubset(RING_POINTS, ['x1sq', 'x2sq']).separates).toBe(true);

    const search = searchFeatureSubsets(RING_POINTS);
    expect(search.total).toBe(127);
    expect(search.passing.length).toBe(104);
    expect(search.minCount).toBe(1);
    expect(search.minimalPassing).toEqual([['r2'], ['l1']]);
  });

  it('Brief 4: C grid has a clear interior sweet spot beating both tiny C and huge C', () => {
    const res = searchCGrid(BRIEF4_DATA.train, BRIEF4_DATA.val);
    const atMin = res.rows[0];
    const atMax = res.rows[res.rows.length - 1];
    expect(atMin.C).toBe(0.005);
    expect(atMin.valAcc).toBeCloseTo(0.7, 3);
    expect(atMax.C).toBe(50);
    expect(atMax.trainAcc).toBe(1);
    expect(atMax.valAcc).toBeCloseTo(55 / 60, 3);
    expect(res.bestValAcc).toBe(1);
    expect(res.bestCs).toEqual([0.1, 0.3, 1, 3]);
  });
});
