// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import {
  BASE_WEIGHTS,
  FLOAT_FORMATS,
  LAB_BIT_BUDGET,
  LAB_RMSE_GOAL,
  decodeFloatBits,
  encodeFloatBits,
  evaluateMix,
  quantizeFloat,
  quantizeIntSymmetric,
  solveQuantLab,
} from './quant';

describe('floating-point bit formats', () => {
  it('reproduces exact max finite values and overflow/underflow behaviour', () => {
    const [fp32, fp16, bf16, fp8] = FLOAT_FORMATS;
    expect(fp16.maxFinite).toBe(65504);
    expect(bf16.maxFinite).toBeCloseTo(3.38953139e38, -31);
    expect(fp8.maxFinite).toBe(448);

    // 100,000 overflows FP16 to Infinity, stays clean in BF16 (99,840), saturates FP8 to 448
    expect(quantizeFloat(100000, fp16)).toBe(Infinity);
    expect(quantizeFloat(100000, bf16)).toBe(99840);
    expect(quantizeFloat(100000, fp8)).toBe(448);

    // 1e-9 underflows to 0 in FP16 and FP8, survives in BF16 and FP32
    expect(quantizeFloat(1e-9, fp16)).toBe(0);
    expect(quantizeFloat(1e-9, fp8)).toBe(0);
    expect(quantizeFloat(1e-9, bf16)).toBeGreaterThan(0.99e-9);
    expect(quantizeFloat(1e-9, fp32)).toBeCloseTo(1e-9, 15);

    // Exact round-trip on clean binary fractions
    for (const spec of FLOAT_FORMATS) {
      const enc = encodeFloatBits(-0.375, spec);
      expect(enc.value).toBe(-0.375);
      expect(decodeFloatBits(enc.sign, enc.exp, enc.mant, spec.expBits, spec.mantBits, spec.hasInf)).toBe(-0.375);
    }
  });
});

describe('outlier quantisation catastrophe and group-wise rescue', () => {
  it('pins the exact RMSE and zero-collapse counts quoted in prose (w[0] = 6.5)', () => {
    const w = [...BASE_WEIGHTS];
    w[0] = 6.5;

    const int4Tensor = quantizeIntSymmetric(w, 4, 64);
    const int4G16 = quantizeIntSymmetric(w, 4, 16);
    const int4G8 = quantizeIntSymmetric(w, 4, 8);
    const int8Tensor = quantizeIntSymmetric(w, 8, 64);

    expect(int4Tensor.normalZeros).toBe(52);
    expect(int4Tensor.rmse).toBeCloseTo(0.2872, 4);
    expect(int4Tensor.effBits).toBe(4.25);

    expect(int4G16.normalZeros).toBe(15);
    expect(int4G16.rmse).toBeCloseTo(0.1296, 4);
    expect(int4G16.effBits).toBe(5.0);

    expect(int4G8.normalZeros).toBe(9);
    expect(int4G8.rmse).toBeCloseTo(0.0806, 4);
    expect(int4G8.effBits).toBe(6.0);

    expect(int8Tensor.normalZeros).toBe(0);
    expect(int8Tensor.rmse).toBeCloseTo(0.0153, 4);
  });
});

describe('mixed-precision QuantLab (exhaustive 625-config search)', () => {
  it('pins the exhaustive count, optimum, and uniform baselines', () => {
    const res = solveQuantLab();
    expect(res.total).toBe(625);
    expect(res.passing).toBe(5);
    expect(res.best.choices).toEqual([1, 3, 2, 3]);
    expect(res.best.avgBits).toBe(5.625);
    expect(res.best.avgRmse).toBeCloseTo(0.0334, 4);
    expect(res.best.avgBits).toBeLessThanOrEqual(LAB_BIT_BUDGET);
    expect(res.best.avgRmse).toBeLessThanOrEqual(LAB_RMSE_GOAL);

    const uniTensor = evaluateMix([2, 2, 2, 2]);
    expect(uniTensor.avgBits).toBe(4.25);
    expect(uniTensor.avgRmse).toBeCloseTo(0.1089, 4);

    const uniG16 = evaluateMix([3, 3, 3, 3]);
    expect(uniG16.avgBits).toBe(5.0);
    expect(uniG16.avgRmse).toBeCloseTo(0.0646, 4);
  });
});
