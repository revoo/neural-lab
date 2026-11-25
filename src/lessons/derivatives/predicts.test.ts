// authored-by: Claude Opus 5.5
// Pins every number used in the Maths Toolkit's <Predict> blocks (WP1.5 of docs/CURRICULUM_PLAN.md).
// One file covers all nine lessons so the numbers sit together; each block names its lesson.
import { describe, expect, it } from 'vitest';
import { cosineSimilarity, det2, dot, matMul, matVec, norm, scale } from '../../lib/linalg';
import { crossEntropy, softmax, surprise } from '../../lib/prob';

describe('toolkit predicts', () => {
  it('vectors: [3, 6] and [6, 12] point the same way, one twice as long', () => {
    expect(cosineSimilarity([3, 6], [6, 12])).toBeCloseTo(1, 12);
    expect(norm([6, 12]) / norm([3, 6])).toBeCloseTo(2, 12);
  });

  it('dot-product: perpendicular arrows give 0 whatever their lengths; doubling x doubles w·x', () => {
    for (const k of [0.5, 1, 7]) expect(dot(scale([2, 1], k), [-1, 2])).toBe(0);
    const w = [1, 2, -3];
    const x = [4, 5, 0];
    expect(dot(w, scale(x, 2))).toBe(2 * dot(w, x));
  });

  it('matrices: î→(2,0), ĵ→(1,1) sends (1,1) to (3,1); columns (1,2),(2,4) squash (det 0)', () => {
    expect(matVec([[2, 1], [0, 1]], [1, 1])).toEqual([3, 1]);
    const squash = [[1, 2], [2, 4]];
    expect(det2(squash)).toBe(0);
    // two different points land on the same spot, so nothing can undo it
    expect(matVec(squash, [2, 0])).toEqual(matVec(squash, [0, 1]));
  });

  it('matrix-multiplication: (32, 784) @ (784, 128) → (32, 128)', () => {
    const X = Array.from({ length: 32 }, () => new Array(784).fill(0));
    const W = Array.from({ length: 784 }, () => new Array(128).fill(0));
    const Y = matMul(X, W);
    expect([Y.length, Y[0].length]).toEqual([32, 128]);
  });

  it('derivatives: slope of x² at 3 is 6; h = 1e-15 goes haywire, h = 1e-5 (central) does not', () => {
    const f = (x: number) => x * x;
    expect((f(3 + 1e-5) - f(3 - 1e-5)) / 2e-5).toBeCloseTo(6, 8);
    const tiny = (f(3 + 1e-15) - f(3)) / 1e-15;
    expect(Math.abs(tiny - 6)).toBeGreaterThan(0.5);
  });

  it('gradients: a perpendicular step changes nothing to first order; |(3, 4)| = 5', () => {
    expect(dot([3, 4], [-4, 3])).toBe(0);
    expect(norm([3, 4])).toBe(5);
  });

  it('chain-rule: 2 × 3 × 0.5 = 3; 0.25¹⁰ ≈ 9.5 × 10⁻⁷', () => {
    expect(2 * 3 * 0.5).toBe(3);
    expect(0.25 ** 10).toBeCloseTo(9.5367e-7, 10);
    expect((0.25 ** 10).toExponential(1)).toBe('9.5e-7');
  });

  it('probability: [2,2,2] and [102,102,102] both give ⅓ each; temperature → 0 sends nearly all to the top', () => {
    for (const z of [[2, 2, 2], [102, 102, 102]]) for (const p of softmax(z)) expect(p).toBeCloseTo(1 / 3, 12);
    const p = softmax([2, 1, 0.5], 0.05);
    expect(p[0]).toBeGreaterThan(0.999);
    const flat = softmax([2, 1, 0.5], 100);
    expect(Math.max(...flat) - Math.min(...flat)).toBeLessThan(0.02);
  });

  it('information: surprise of 1/8 is 3 bits; a 50/50 guess costs 1 bit', () => {
    expect(surprise(1 / 8, 2)).toBeCloseTo(3, 12);
    expect(crossEntropy([1, 0], [0.5, 0.5], 2)).toBeCloseTo(1, 12);
  });
});
