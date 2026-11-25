// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  KERNEL_PRESETS,
  TEST_IMAGES,
  compareDenseVsConvParams,
  conv2d,
  convLabExhaustive,
  droppedBorderPixels,
  evalDiagonalKernel,
  evalStackShape,
  evalVerticalKernel,
  outSize,
  validStackShrinkage,
  verifyEquivariance,
} from './conv';

describe('convolution: 2D cross-correlation, padding, stride & equivariance', () => {
  it('computes exact Sobel Gx on a 6x6 vertical step edge', () => {
    const img = Array.from({ length: 6 }, () => [0, 0, 0, 1, 1, 1]);
    const sobelX = KERNEL_PRESETS.find((k) => k.id === 'sobel-x')!.kernel;
    const out = conv2d(img, sobelX, 0, 1);
    expect(out.length).toBe(4);
    expect(out[0].length).toBe(4);
    expect(out[0]).toEqual([0, 4, 4, 0]);
  });

  it('verifies output size formula, dropped border pixels, and valid-conv shrinkage', () => {
    expect(outSize(16, 3, 1, 2)).toBe(8);
    expect(droppedBorderPixels(16, 3, 1, 2)).toBe(1);
    expect(outSize(17, 3, 1, 2)).toBe(9);
    expect(droppedBorderPixels(17, 3, 1, 2)).toBe(0);

    const shrink = validStackShrinkage(12, 5, 3);
    expect(shrink.validH).toBe(2);
    expect(shrink.sameH).toBe(12);
    expect(shrink.pixelLoss).toBeCloseTo(1 - 4 / 144, 5); // 97.22%
  });

  it('verifies exact translation equivariance of 2D convolution', () => {
    for (const [cx, cy] of [
      [2, 2],
      [6, 5],
      [4, 7],
      [7, 2],
    ]) {
      const r = verifyEquivariance(cx, cy);
      expect(r.exactMatch).toBe(true);
      expect(r.peakVal).toBe(5);
      expect(r.peakR).toBe(cy);
      expect(r.peakC).toBe(cx);
    }
    expect(TEST_IMAGES.length).toBe(3);
  });

  it('pins Dense vs Conv2d parameter counts on 224x224x3 -> 32', () => {
    const p = compareDenseVsConvParams(224, 224, 3, 32, 3);
    expect(p.denseParams).toBe(241_694_179_328);
    expect(p.convParams).toBe(896);
    expect(p.ratio).toBeCloseTo(269_747_968, 0);
  });
});

describe('convolution: Lab (Design the Kernel) exhaustive 19,683-kernel checks', () => {
  it('Brief 1 (vertical edge): 66/19,683 pass; fewest non-zero weights is 4 (21 optimal)', () => {
    expect(evalVerticalKernel([0, 0, 0, -1, 0, 1, 0, 0, 0]).pass).toBe(false); // peak 1 < 2
    expect(evalVerticalKernel([-1, 0, 1, -1, 0, 1, 0, 0, 0]).pass).toBe(true);
    const ex = convLabExhaustive('vertical');
    expect(ex.total).toBe(19_683);
    expect(ex.passing).toBe(66);
    expect(ex.summary).toContain('Fewest non-zero weights: 4 (21 optimal');
  });

  it('Brief 2 (diagonal edge): 67/19,683 pass; max diagonal response +3 (4 optimal)', () => {
    const best = evalDiagonalKernel([-1, 1, 1, 0, -1, 1, 0, 0, -1]);
    expect(best.pass).toBe(true);
    expect(best.diag).toBe(3);
    const ex = convLabExhaustive('diagonal');
    expect(ex.total).toBe(19_683);
    expect(ex.passing).toBe(67);
    expect(ex.summary).toContain('Maximum diagonal response: +3 (4 optimal');
  });

  it('Brief 3 (17->5 zero-drop stack): 2/12 pass; fewest weights K=3, P=1, S=2 (18 weights)', () => {
    const k3 = evalStackShape(3, 1, 2);
    expect(k3.pass).toBe(true);
    expect(k3.h1).toBe(9);
    expect(k3.h2).toBe(5);
    expect(k3.weightsPerFilter).toBe(18);

    const k5 = evalStackShape(5, 2, 2);
    expect(k5.pass).toBe(true);
    expect(k5.weightsPerFilter).toBe(50);

    const ex = convLabExhaustive('stack-shape');
    expect(ex.total).toBe(12);
    expect(ex.passing).toBe(2);
    expect(ex.summary).toContain('K=3, P=1, S=2');
  });
});
