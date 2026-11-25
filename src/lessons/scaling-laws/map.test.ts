// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { BUDGET_BRIEFS, briefOptimum, budgetBoundaryLogD, inferenceFlops, predictedLoss, targetContourLogD, trainFlops } from './scaling';

describe('compute-budget map', () => {
  it('the budget boundary spends exactly the budget', () => {
    for (const b of BUDGET_BRIEFS) {
      for (const logN of [10, 10.5, 11]) {
        const ld = budgetBoundaryLogD(b, logN);
        if (ld === null) continue;
        const N = 10 ** logN;
        const D = 10 ** ld;
        const spent = trainFlops(N, D) + (b.dInf ? inferenceFlops(N, b.dInf) : 0);
        expect(spent / (b.maxTrainC ?? b.maxTotalC!)).toBeCloseTo(1, 9);
      }
    }
  });
  it('the target contour sits exactly at the target loss', () => {
    const b = BUDGET_BRIEFS[0];
    const ld = targetContourLogD(b, 10.5)!;
    expect(predictedLoss(10 ** 10.5, 10 ** ld)).toBeCloseTo(b.targetLoss, 9);
  });
  it('every brief optimum passes its own brief', () => {
    for (const b of BUDGET_BRIEFS) {
      const o = briefOptimum(b);
      // the lifecycle optimum sits on the boundary; nudge into the feasible side for the check
      expect(b.check(o.N, o.D * (b.id === 'product' ? 1.0001 : 0.9999))).toBe(true);
    }
  });
});
