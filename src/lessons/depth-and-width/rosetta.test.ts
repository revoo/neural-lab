// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  computeRosetta,
  directConv1D,
  directRnnUnroll,
  INPUT_PRESETS,
  LORA_A,
  LORA_B,
  maxMatrixDiff,
} from './rosetta';

describe('Architecture Rosetta Stone (rosetta.ts)', () => {
  it('1D Conv Toeplitz matrix matches direct sliding 1D convolution to < 1e-12 on every preset', () => {
    for (const preset of Object.values(INPUT_PRESETS)) {
      const view = computeRosetta('conv', preset.x);
      const direct = directConv1D(preset.x);
      expect(view.freeParams).toBe(3);
      expect(view.denseParams / view.freeParams).toBe(12); // 12x fewer free weights
      for (let i = 0; i < 6; i++) {
        expect(Math.abs(view.y[i] - direct[i])).toBeLessThan(1e-12);
      }
    }
  });

  it('Unrolled RNN block-lower-triangular matrix matches step-by-step recurrence to < 1e-12', () => {
    for (const preset of Object.values(INPUT_PRESETS)) {
      const view = computeRosetta('rnn', preset.x);
      const direct = directRnnUnroll(preset.x);
      expect(view.freeParams).toBe(8);
      // Strictly causal: upper-right blocks are identically zero
      for (let br = 0; br < 3; br++) {
        for (let bc = br + 1; bc < 3; bc++) {
          for (let r = 0; r < 2; r++) {
            for (let c = 0; c < 2; c++) {
              expect(view.W[2 * br + r][2 * bc + c]).toBe(0);
            }
          }
        }
      }
      for (let i = 0; i < 6; i++) {
        expect(Math.abs(view.y[i] - direct[i])).toBeLessThan(1e-12);
      }
    }
  });

  it('Self-Attention rows sum to 1 and W(x) changes strictly when x changes, unlike the other four', () => {
    const x1 = [...INPUT_PRESETS.edge.x];
    const x2 = [...x1];
    x2[2] = -1.0; // flip x_2 from +1.0 to -1.0

    const att1 = computeRosetta('attention', x1);
    const att2 = computeRosetta('attention', x2);
    for (const row of att1.W) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-12);
    }
    const diff = maxMatrixDiff(att1.W, att2.W);
    expect(diff).toBeGreaterThan(0.25); // unmistakable live matrix shift

    for (const arch of ['dense', 'conv', 'rnn', 'lora'] as const) {
      const v1 = computeRosetta(arch, x1);
      const v2 = computeRosetta(arch, x2);
      expect(maxMatrixDiff(v1.W, v2.W)).toBe(0);
    }
  });

  it('LoRA rank-1 update equals exact outer product a * b^T with 12 trainable params (3x fewer than 36)', () => {
    const view = computeRosetta('lora', INPUT_PRESETS.edge.x);
    expect(view.freeParams).toBe(12);
    expect(view.denseParams / view.freeParams).toBe(3);
    expect(view.lora).toBeDefined();
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        const expectedDelta = LORA_A[i] * LORA_B[j];
        expect(Math.abs(view.lora!.dW[i][j] - expectedDelta)).toBeLessThan(1e-12);
        expect(Math.abs(view.W[i][j] - (view.lora!.W0[i][j] + expectedDelta))).toBeLessThan(1e-12);
      }
    }
  });
});
