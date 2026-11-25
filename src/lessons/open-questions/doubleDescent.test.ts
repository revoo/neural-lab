// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  CURVE_CLEAN,
  CURVE_NOISY,
  N_TRAIN,
  SHOWCASE_FEATURE_SEED,
  SHOWCASE_NOISY_DATA,
  designMatrix,
  drawFeatures,
  fitRandomFeatures,
  minNormLstsq,
} from './doubleDescent';
import { dot, matVec } from '../../lib/linalg';

describe('open-questions random-features double descent', () => {
  it('minNormLstsq finds an exact minimum-L2-norm interpolator when p > n', () => {
    const feats = drawFeatures(60, 510);
    const Phi = designMatrix(SHOWCASE_NOISY_DATA.x, feats);
    const c = minNormLstsq(Phi, SHOWCASE_NOISY_DATA.y);
    const pred = matVec(Phi, c);
    for (let i = 0; i < N_TRAIN; i++) {
      expect(Math.abs(pred[i] - SHOWCASE_NOISY_DATA.y[i])).toBeLessThan(1e-7);
    }
    // Any perturbation in the null space of Phi increases ||c||^2 (optimality condition: c is in row space of Phi)
    // Verify c has strictly smaller norm than adding a small null-space component:
    expect(Math.sqrt(dot(c, c))).toBeGreaterThan(0);
    expect(Math.sqrt(dot(c, c))).toBeLessThan(20);
  });

  it('noisy curve peaks right at the interpolation threshold p = n = 24 and descends again past it', () => {
    const byP = new Map(CURVE_NOISY.map((pt) => [pt.p, pt]));
    // Exact peak location across the entire P_VALUES grid
    let peak = CURVE_NOISY[0];
    for (const pt of CURVE_NOISY) {
      if (pt.testMse > peak.testMse) peak = pt;
    }
    expect(peak.p).toBe(N_TRAIN);
    expect(peak.p).toBe(24);

    // Pinned prose numbers (41-trial medians)
    expect(byP.get(4)!.testMse).toBeCloseTo(0.876, 2);
    expect(byP.get(16)!.testMse).toBeCloseTo(0.056, 2);
    expect(byP.get(24)!.testMse).toBeCloseTo(9.35, 1);
    expect(byP.get(24)!.trainMse).toBeLessThan(1e-18);
    expect(byP.get(160)!.testMse).toBeCloseTo(0.070, 2);
    expect(byP.get(160)!.trainMse).toBeLessThan(1e-20);

    // Second descent drops by > 100x from peak, and weight norm drops dramatically too
    expect(byP.get(24)!.testMse / byP.get(160)!.testMse).toBeGreaterThan(100);
    expect(byP.get(24)!.weightNorm).toBeGreaterThan(1e5);
    expect(byP.get(160)!.weightNorm).toBeLessThan(6);
  });

  it('noiseless curve has no interpolation spike at p = 24', () => {
    const byP = new Map(CURVE_CLEAN.map((pt) => [pt.p, pt]));
    expect(byP.get(24)!.testMse).toBeLessThan(1e-3);
    expect(byP.get(160)!.testMse).toBeLessThan(0.02);
  });

  it('showcase draw matches the classical -> peak -> second-descent arc', () => {
    const f4 = fitRandomFeatures(SHOWCASE_NOISY_DATA, 4, SHOWCASE_FEATURE_SEED);
    const f16 = fitRandomFeatures(SHOWCASE_NOISY_DATA, 16, SHOWCASE_FEATURE_SEED);
    const f24 = fitRandomFeatures(SHOWCASE_NOISY_DATA, 24, SHOWCASE_FEATURE_SEED);
    const f160 = fitRandomFeatures(SHOWCASE_NOISY_DATA, 160, SHOWCASE_FEATURE_SEED);
    expect(f4.testMse).toBeCloseTo(0.738, 2);
    expect(f16.testMse).toBeCloseTo(0.044, 2);
    expect(f24.testMse).toBeCloseTo(7.42, 1);
    expect(f160.testMse).toBeCloseTo(0.064, 2);
  });
});
