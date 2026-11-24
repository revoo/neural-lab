import { describe, expect, it } from 'vitest';
import { elementwiseShape, matmulShape, parseShape } from './shapes';

describe('matmul shapes', () => {
  it('handles plain matrices', () => {
    expect(matmulShape([2, 3], [3, 4])).toMatchObject({ shape: [2, 4] });
    expect('error' in matmulShape([2, 3], [4, 3])).toBe(true);
  });
  it('handles vectors like numpy', () => {
    expect(matmulShape([3], [3])).toMatchObject({ shape: [] });
    expect(matmulShape([2, 3], [3])).toMatchObject({ shape: [2] });
    expect(matmulShape([3], [3, 5])).toMatchObject({ shape: [5] });
  });
  it('broadcasts batch dimensions', () => {
    expect(matmulShape([32, 128, 768], [768, 64])).toMatchObject({ shape: [32, 128, 64] });
    expect(matmulShape([8, 12, 128, 64], [8, 12, 64, 128])).toMatchObject({ shape: [8, 12, 128, 128] });
    expect('error' in matmulShape([4, 2, 3], [5, 3, 2])).toBe(true);
  });
});

describe('elementwise shapes', () => {
  it('broadcasts biases and rejects mismatches', () => {
    expect(elementwiseShape([32, 64], [64])).toMatchObject({ shape: [32, 64] });
    expect(elementwiseShape([32, 1], [1, 10])).toMatchObject({ shape: [32, 10] });
    expect('error' in elementwiseShape([32, 64], [32])).toBe(true);
  });
  it('parses shape strings', () => {
    expect(parseShape('(32, 128, 768)')).toEqual([32, 128, 768]);
    expect(parseShape('2x3')).toEqual([2, 3]);
    expect(parseShape('2, -1')).toBeNull();
  });
});
