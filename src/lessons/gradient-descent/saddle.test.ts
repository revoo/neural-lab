// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  minProbability,
  monteCarloCriticalPoints,
  runSaddleDescent,
  SADDLE_LR,
  SADDLE_STEPS,
  saddleGrad,
  saddleLoss,
  saddleProbability,
  sampleCurvatures,
} from './saddle';

describe('high-dimensional saddle geometry (saddle.ts)', () => {
  it('pins exact critical-point trap and saddle probabilities across dimensions', () => {
    expect(minProbability(1)).toBe(0.5);
    expect(saddleProbability(1)).toBe(0);
    expect(minProbability(2)).toBe(0.25);
    expect(saddleProbability(2)).toBe(0.5);
    expect(minProbability(10)).toBeCloseTo(1 / 1024, 12); // 0.0009765625
    expect(saddleProbability(10)).toBeCloseTo(1022 / 1024, 12); // ~99.80%
    expect(minProbability(20)).toBeLessThan(1e-6); // 1 / 1,048,576
    expect(minProbability(30)).toBeCloseTo(Math.pow(0.5, 30), 15);
  });

  it('Monte Carlo critical-point sampling agrees with exact combinatorics', () => {
    const mc4 = monteCarloCriticalPoints(4, 50000, 123);
    expect(Math.abs(mc4.minFrac - minProbability(4))).toBeLessThan(0.01);
    expect(Math.abs(mc4.saddleFrac - saddleProbability(4))).toBeLessThan(0.01);

    const sample = sampleCurvatures(10, 7);
    expect(sample.curvatures.length).toBe(10);
    expect(sample.upCount + sample.downCount).toBe(10);
  });

  it('analytic gradient matches central finite differences to < 1e-8', () => {
    const eps = 1e-6;
    for (const [x, y] of [[1.2, 0.02], [-0.7, 0.9], [0.0, -1.1], [0.45, -0.35]]) {
      const [gx, gy] = saddleGrad(x, y);
      const fdX = (saddleLoss(x + eps, y) - saddleLoss(x - eps, y)) / (2 * eps);
      const fdY = (saddleLoss(x, y + eps) - saddleLoss(x, y - eps)) / (2 * eps);
      expect(Math.abs(gx - fdX)).toBeLessThan(1e-8);
      expect(Math.abs(gy - fdY)).toBeLessThan(1e-8);
    }
  });

  it('pins plateau-then-escape from near ridge (1.2, 0.02) vs strict trap dead-on (1.2, 0.0)', () => {
    const near = runSaddleDescent(1.2, 0.02, SADDLE_LR, SADDLE_STEPS);
    expect(near.length).toBe(SADDLE_STEPS + 1);
    expect(near[0].loss).toBeCloseTo(0.71988, 5);

    // Rapid descent onto the saddle by step 12
    expect(near[12].loss).toBeLessThan(0.01);
    expect(near[12].loss).toBeGreaterThan(0);

    // Deceptive flat plateau over steps 12..25: every single step moves loss by < 0.006
    for (let s = 12; s < 25; s++) {
      expect(Math.abs(near[s + 1].loss - near[s].loss)).toBeLessThan(0.006);
    }

    // Sudden escape off the downhill y cliff by step 42
    expect(near[25].loss).toBeGreaterThan(-0.025);
    expect(near[42].loss).toBeLessThan(-0.65);
    expect(near[42].y).toBeCloseTo(0.02 * Math.pow(1 + SADDLE_LR * 0.6, 42), 10);

    // Dead on the ridge: y stays strictly 0 forever, trapped at loss ~ 0
    const dead = runSaddleDescent(1.2, 0.0, SADDLE_LR, SADDLE_STEPS);
    for (const st of dead) {
      expect(st.y).toBe(0);
    }
    expect(dead[42].loss).toBeGreaterThanOrEqual(0);
    expect(dead[42].loss).toBeLessThan(1e-6);
  });
});
