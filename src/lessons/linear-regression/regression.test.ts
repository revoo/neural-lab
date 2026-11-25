// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAB_BRIEFS,
  collinearExperiment,
  confusionAt,
  evaluateLab,
  fitLogistic,
  fitRidge,
  getFraudDataset,
  labExhaustive,
  linearVsLogistic1D,
  logisticLossAndGrad,
} from './regression';

const p1 = (v: number) => (100 * v).toFixed(1);

describe('linear & logistic regression maths', () => {
  it('normal equations recover exact weights on noiseless linear data', () => {
    const X = [
      [1, 0, 1],
      [1, 1, 0],
      [1, 2, 1],
      [1, -1, 2],
      [1, 3, -1],
    ];
    const y = X.map(([, x1, x2]) => 2.5 - 1.2 * x1 + 0.8 * x2);
    const w = fitRidge(X, y, 0, false);
    expect(w[0]).toBeCloseTo(2.5, 10);
    expect(w[1]).toBeCloseTo(-1.2, 10);
    expect(w[2]).toBeCloseTo(0.8, 10);
  });

  it('hand-written logistic regression gradient matches central finite differences', () => {
    const X = [
      [1, -1.2, 0.5],
      [1, 0.4, -0.8],
      [1, 1.5, 1.1],
      [1, -0.3, -1.4],
      [1, 2.0, -0.2],
    ];
    const y = [0, 0, 1, 0, 1];
    const w = [0.3, -0.7, 1.1];
    const lambda = 0.8;
    const { grad } = logisticLossAndGrad(X, y, w, lambda);
    const h = 1e-6;
    for (let j = 0; j < w.length; j++) {
      const wp = [...w];
      const wm = [...w];
      wp[j] += h;
      wm[j] -= h;
      const num = (logisticLossAndGrad(X, y, wp, lambda).loss - logisticLossAndGrad(X, y, wm, lambda).loss) / (2 * h);
      expect(grad[j]).toBeCloseTo(num, 7);
    }
  });

  it('Newton-Raphson converges to near-zero gradient on convex logistic loss', () => {
    const X = [
      [1, -2],
      [1, -1],
      [1, 0.2],
      [1, 1.5],
      [1, 2.5],
    ];
    const y = [0, 0, 0, 1, 1];
    const w = fitLogistic(X, y, 0.1);
    const { grad } = logisticLossAndGrad(X, y, w, 0.1);
    expect(Math.hypot(...grad)).toBeLessThan(1e-8);
  });
});

describe('numbers quoted in the lesson', () => {
  it('collinearity: condition number rises 1.6 -> 3,090.9, GD slows 4 -> 500+, Ridge stabilises test MSE', () => {
    const c0 = collinearExperiment(0, 0.5);
    expect(c0.condNumber.toFixed(1)).toBe('1.6');
    expect(c0.gdStepsToTol).toBe(4);

    const c99 = collinearExperiment(0.99, 0.5);
    expect(c99.condNumber.toFixed(1)).toBe('303.3');
    expect(c99.gdStepsToTol).toBe(473);

    const c999 = collinearExperiment(0.999, 0.5);
    expect(c999.condNumber.toFixed(1)).toBe('3090.9');
    expect(c999.testMseOls).toBeCloseTo(0.347, 3);
    expect(c999.testMseRidge).toBeCloseTo(0.218, 3);
    expect(c999.wOls[0].toFixed(2)).toBe('0.93');
    expect(c999.wOls[1].toFixed(2)).toBe('2.17');
    expect(c999.wRidge[0].toFixed(2)).toBe('1.52');
    expect(c999.wRidge[1].toFixed(2)).toBe('1.57');
  });

  it('1D outlier: OLS drifts 4.14 -> 5.64 (3 errors, 75.0%), logistic stays fixed at 4.11 (0 errors)', () => {
    const r7 = linearVsLogistic1D(7);
    expect(r7.boundLin.toFixed(2)).toBe('4.14');
    expect(r7.boundLog.toFixed(2)).toBe('4.11');
    expect([r7.linErrors, r7.logErrors]).toEqual([0, 0]);

    const r15 = linearVsLogistic1D(15);
    expect(r15.boundLin.toFixed(2)).toBe('4.81');
    expect([r15.linErrors, r15.logErrors]).toEqual([1, 0]);

    const r25 = linearVsLogistic1D(25);
    expect(r25.boundLin.toFixed(2)).toBe('5.64');
    expect(r25.boundLog.toFixed(2)).toBe('4.11');
    expect([r25.linErrors, r25.logErrors]).toEqual([3, 0]);
    expect(p1(r25.linAcc)).toBe('75.0');
  });

  it('fraud ROC: AUC 0.939; t=0.50 hits 97.0% acc but 50.0% recall (€505); t=0.10 hits 80.0% recall (€275)', () => {
    const f = getFraudDataset();
    expect(f.auc.toFixed(3)).toBe('0.939');

    const c50 = confusionAt(f.probs, f.labels, 0.5);
    expect(p1(c50.accuracy)).toBe('97.0');
    expect(p1(c50.precision)).toBe('83.3');
    expect(p1(c50.recall)).toBe('50.0');
    expect(c50.costEur).toBe(505);

    const c10 = confusionAt(f.probs, f.labels, 0.1);
    expect(p1(c10.accuracy)).toBe('91.5');
    expect(p1(c10.precision)).toBe('34.8');
    expect(p1(c10.recall)).toBe('80.0');
    expect(c10.costEur).toBe(275);
  });

  it('Lab: exhaustive minimum-parameter optimums for all 3 briefs', () => {
    expect(p1(evaluateLab('ring', ['x1', 'x2'], 0.1, 0.5).testAcc)).toBe('54.8');
    expect(p1(evaluateLab('xor-risk', ['x1', 'x2'], 0.1, 0.5).testAcc)).toBe('52.8');
    expect(p1(evaluateLab('imbalanced', ['x1', 'x2'], 0.1, 0.5).testRecall)).toBe('76.9');

    const [b1, b2, b3] = LAB_BRIEFS.map(labExhaustive);

    expect([b1.total, b1.passing, b1.bestParams]).toEqual([252, 29, 3]);
    expect(b1.best[0].keys).toEqual(['x1sq', 'x2sq']);
    expect(p1(b1.bestAccAtMin)).toBe('94.8');

    expect([b2.total, b2.passing, b2.bestParams]).toEqual([252, 80, 2]);
    expect(b2.best[0].keys).toEqual(['x1x2']);
    expect(p1(b2.bestAccAtMin)).toBe('86.4');

    expect([b3.total, b3.passing, b3.bestParams]).toEqual([1008, 8, 4]);
    expect(b3.best[0].keys).toEqual(['x1', 'x2', 'abs']);
    expect(p1(b3.best[0].recall)).toBe('89.7');
    expect(p1(b3.best[0].precision)).toBe('72.9');
  });
});
