// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  contrastLabExhaustive,
  evalContrastBrief1,
  evalContrastBrief2,
  evalContrastBrief3,
  hardNegativeFocus,
  hardNegativeProb,
  infoNceLossAndGrad,
  trainCircleContrastive,
} from './contrastive';

describe('contrastive-learning: InfoNCE gradients, temperature focus, and circle uniformity', () => {
  it('matches central finite differences on InfoNCE gradients to < 1e-8', () => {
    const A = [
      [0.8, 0.6],
      [-0.5, 0.866],
      [0.2, -0.98],
      [-0.9, -0.436],
    ];
    const B = [
      [0.7, 0.714],
      [-0.6, 0.8],
      [0.3, -0.954],
      [-0.85, -0.527],
    ];
    const tau = 0.2;
    const base = infoNceLossAndGrad(A, B, tau);
    const eps = 1e-5;
    let maxErr = 0;

    for (let i = 0; i < A.length; i++) {
      for (let k = 0; k < 2; k++) {
        const oldA = A[i][k];
        A[i][k] = oldA + eps;
        const lpA = infoNceLossAndGrad(A, B, tau).loss;
        A[i][k] = oldA - eps;
        const lmA = infoNceLossAndGrad(A, B, tau).loss;
        A[i][k] = oldA;
        maxErr = Math.max(maxErr, Math.abs((lpA - lmA) / (2 * eps) - base.gA[i][k]));

        const oldB = B[i][k];
        B[i][k] = oldB + eps;
        const lpB = infoNceLossAndGrad(A, B, tau).loss;
        B[i][k] = oldB - eps;
        const lmB = infoNceLossAndGrad(A, B, tau).loss;
        B[i][k] = oldB;
        maxErr = Math.max(maxErr, Math.abs((lpB - lmB) / (2 * eps) - base.gB[i][k]));
      }
    }
    expect(maxErr).toBeLessThan(1e-8);
  });

  it('pins hard-negative temperature focus and batch-size hard-negative probability', () => {
    expect(hardNegativeFocus(1.0).hardShareOfNegs).toBeCloseTo(0.495, 3);
    expect(hardNegativeFocus(0.25).hardShareOfNegs).toBeCloseTo(0.877, 3);
    expect(hardNegativeFocus(0.1).hardShareOfNegs).toBeCloseTo(0.997, 3);
    expect(hardNegativeFocus(0.05).hardShareOfNegs).toBeCloseTo(1.0, 3);

    expect(hardNegativeProb(64, 0.01)).toBeCloseTo(0.469, 3);
    expect(hardNegativeProb(512, 0.01)).toBeCloseTo(0.994, 3);
  });

  it('shows alignment-only collapses to cosine 1.0 while InfoNCE tau=0.15 spreads into near-uniform octagon (-0.136)', () => {
    const col = trainCircleContrastive(false, 0.15, 90, 61);
    expect(col.meanPairCos).toBeCloseTo(1.0, 3);
    expect(col.uniformity).toBeCloseTo(0.0, 3);

    const good = trainCircleContrastive(true, 0.15, 90, 61);
    expect(good.top1Acc).toBe(1.0);
    expect(good.alignment).toBeLessThan(0.001);
    expect(good.meanPairCos).toBeCloseTo(-0.136, 3);
    expect(good.uniformity).toBeCloseTo(-2.336, 3);
  });
});

describe('contrastive-learning: Lab (Tune Contrastive Training) exhaustive checks', () => {
  it('pins all 3 exhaustive lab briefs', () => {
    expect(evalContrastBrief1(false, 0.15).pass).toBe(false);
    expect(evalContrastBrief1(true, 2.0).pass).toBe(false);
    expect(evalContrastBrief1(true, 0.15).pass).toBe(true);
    expect(contrastLabExhaustive('circle')).toMatchObject({ total: 8, passing: 3 });

    expect(evalContrastBrief2(0.02).pass).toBe(false); // 1/tau = 50 > 25
    expect(evalContrastBrief2(0.5).pass).toBe(false);  // hard share 65.6% < 80%
    expect(evalContrastBrief2(0.05).pass).toBe(true);
    expect(contrastLabExhaustive('temp')).toMatchObject({ total: 6, passing: 3 });

    expect(evalContrastBrief3('simclr-64').pass).toBe(false);
    expect(evalContrastBrief3('simclr-2048').pass).toBe(false);
    expect(evalContrastBrief3('moco-64-q512').pass).toBe(true);
    expect(contrastLabExhaustive('batch')).toMatchObject({ total: 6, passing: 3 });
  });
});
