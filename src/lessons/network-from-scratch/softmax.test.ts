// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { SOFTMAX_PRESETS, crossEntropyLoss, softmax, softmaxBackprop, stepSoftmaxLogits } from './softmax';

describe('multi-class softmax + cross-entropy backprop engine', () => {
  it('matches central finite differences (< 1e-8) and sums to zero on every preset and class', () => {
    for (const preset of SOFTMAX_PRESETS) {
      for (const target of [0, 1, 2] as const) {
        const res = softmaxBackprop(preset.logits, target);
        expect(res.maxDiff).toBeLessThan(1e-8);
        expect(Math.abs(res.gradSum)).toBeLessThan(1e-14);
        const probSum = res.probs.reduce((a, b) => a + b, 0);
        expect(probSum).toBeCloseTo(1, 12);
      }
    }
  });

  it('pins the Predict case (p = [0.20, 0.70, 0.10], grad = [-0.80, +0.70, +0.10])', () => {
    const preset = SOFTMAX_PRESETS.find((p) => p.id === 'predict-case')!;
    const res = softmaxBackprop(preset.logits, preset.targetClass);
    expect(res.probs[0]).toBeCloseTo(0.2, 3);
    expect(res.probs[1]).toBeCloseTo(0.7, 3);
    expect(res.probs[2]).toBeCloseTo(0.1, 3);
    expect(res.analyticGrad[0]).toBeCloseTo(-0.8, 3);
    expect(res.analyticGrad[1]).toBeCloseTo(0.7, 3);
    expect(res.analyticGrad[2]).toBeCloseTo(0.1, 3);
  });

  it('pins the Python snippet example [2.0, 1.0, -0.5] with target class 1', () => {
    const res = softmaxBackprop([2.0, 1.0, -0.5], 1);
    expect(res.probs[0]).toBeCloseTo(0.6897, 4);
    expect(res.probs[1]).toBeCloseTo(0.2537, 4);
    expect(res.probs[2]).toBeCloseTo(0.0566, 4);
    expect(res.loss).toBeCloseTo(1.3715, 4);
    expect(res.analyticGrad[0]).toBeCloseTo(0.6897, 4);
    expect(res.analyticGrad[1]).toBeCloseTo(-0.7463, 4);
    expect(res.analyticGrad[2]).toBeCloseTo(0.0566, 4);
    expect(res.maxDiff).toBeLessThan(1e-9);
  });

  it('is shift-invariant under adding a constant to all logits', () => {
    const z = [2.0, 1.0, -0.5];
    const shifted = z.map((v) => v + 1000);
    expect(softmax(shifted)).toEqual(softmax(z));
    expect(crossEntropyLoss(shifted, 1)).toBeCloseTo(crossEntropyLoss(z, 1), 12);
  });

  it('one gradient step pushes the true logit up, every wrong logit down, raises p_true and lowers loss', () => {
    const z0 = [0.0, 1.253, -0.693];
    const before = softmaxBackprop(z0, 0);
    const z1 = stepSoftmaxLogits(z0, 0, 1.0);
    const after = softmaxBackprop(z1, 0);
    expect(z1[0]).toBeGreaterThan(z0[0]);
    expect(z1[1]).toBeLessThan(z0[1]);
    expect(z1[2]).toBeLessThan(z0[2]);
    expect(after.loss).toBeLessThan(before.loss);
    expect(after.probs[0]).toBeGreaterThan(before.probs[0]);
    expect(after.probs[1] + after.probs[2]).toBeLessThan(before.probs[1] + before.probs[2]);
  });
});
