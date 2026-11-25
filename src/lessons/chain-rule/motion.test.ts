// authored-by: Claude Opus 5.5
// The four WP1.7 animations compute real values at every frame. These tests pin what each frame shows.
import { describe, expect, it } from 'vitest';
import { forward, nudgeTrail } from './ChainExplorer';
import { ROTATE_FRAMES, rotatedFrom } from '../dot-product/DotExplorer';
import { SHRINK_FRAMES, shrinkLogH } from '../derivatives/TangentExplorer';
import { MORPH, MORPH_FRAMES, morphAt } from '../matrix-multiplication/CompositionDemo';
import { matMul } from '../../lib/linalg';

describe('chain-rule: send a nudge', () => {
  it('the nudge arriving at L equals the measured slope dL/dw (central differences)', () => {
    for (const w of [-3, -0.5, 0.4, 1.2]) {
      const trail = nudgeTrail(w);
      expect(trail[0]).toBe(1);
      expect(trail[1]).toBe(1.5);
      const h = 1e-6;
      const measured = (forward(w + h).L - forward(w - h).L) / (2 * h);
      expect(trail[3]).toBeCloseTo(measured, 6);
    }
  });
  it('a saturated tanh (w = −3) shrinks the nudge to almost nothing', () => {
    expect(Math.abs(nudgeTrail(-3)[2])).toBeLessThan(0.01);
  });
});

describe('dot-product: rotate', () => {
  it('traces |a||b| cos θ: 0 at 90° and 270°, minus the product at 180°', () => {
    const a: [number, number] = [3, 1];
    const len = Math.hypot(1.5, 3);
    const dot = (b: [number, number]) => a[0] * b[0] + a[1] * b[1];
    const la = Math.hypot(...a);
    expect(dot(rotatedFrom(a, len, 0))).toBeCloseTo(la * len, 10);
    expect(dot(rotatedFrom(a, len, 90))).toBeCloseTo(0, 10);
    expect(dot(rotatedFrom(a, len, 180))).toBeCloseTo(-la * len, 10);
    expect(dot(rotatedFrom(a, len, 270))).toBeCloseTo(0, 10);
    expect(360 / ROTATE_FRAMES).toBe(5);
  });
});

describe('derivatives: shrink the nudge', () => {
  it('h goes from 1 to 0.001 on a log scale', () => {
    expect(shrinkLogH(0)).toBe(-0);
    expect(shrinkLogH(SHRINK_FRAMES)).toBe(-3);
    // on x² at x = 1 the secant slope is 2 + h, so it converges to 2
    const f = (x: number) => x * x;
    const secant = (h: number) => (f(1 + h) - f(1)) / h;
    expect(secant(10 ** shrinkLogH(0))).toBeCloseTo(3, 12);
    expect(secant(10 ** shrinkLogH(SHRINK_FRAMES))).toBeCloseTo(2.001, 9);
  });
});

describe('matrix-multiplication: first A, then B', () => {
  it('passes through A, pauses, and ends at BA (and the other order ends at AB)', () => {
    const rot = [[0, -1], [1, 0]];
    const shear = [[1, 1], [0, 1]];
    expect(morphAt(rot, shear, 0)).toEqual([[1, 0], [0, 1]]);
    expect(morphAt(rot, shear, MORPH.first)).toEqual(rot);
    expect(morphAt(rot, shear, MORPH.first + MORPH.pause)).toEqual(rot);
    expect(morphAt(rot, shear, MORPH_FRAMES)).toEqual(matMul(shear, rot));
    expect(morphAt(shear, rot, MORPH_FRAMES)).toEqual(matMul(rot, shear));
    expect(matMul(shear, rot)).not.toEqual(matMul(rot, shear));
  });
});
