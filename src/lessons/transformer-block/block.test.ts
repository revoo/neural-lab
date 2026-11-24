import { describe, expect, it } from 'vitest';
import { GPT2, depthStudy, gptParams, layerNorm, runBlock } from './block';

describe('the transformer block lesson', () => {
  it('layer norm gives mean 0 and spread 1', () => {
    const y = layerNorm([3, -1, 8, 2]);
    expect(y.reduce((a, b) => a + b, 0) / 4).toBeCloseTo(0, 10);
    expect(Math.sqrt(y.reduce((a, b) => a + b * b, 0) / 4)).toBeCloseTo(1, 4);
  });
  it('residual stages really are additions; the causal mask zeroes later tokens', () => {
    const s = runBlock();
    s[3].vecs.forEach((h, i) => h.forEach((v, c) => expect(v).toBeCloseTo(s[0].vecs[i][c] + s[2].vecs[i][c], 12)));
    s[6].vecs.forEach((y, i) => y.forEach((v, c) => expect(v).toBeCloseTo(s[3].vecs[i][c] + s[5].vecs[i][c], 12)));
    const w = s[2].weights!;
    expect(w[0]).toEqual([1, 0, 0]);
    expect(w[1][2]).toBe(0);
    for (const row of w) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
  it('GPT-2 parameter counts match the published sizes', () => {
    expect(gptParams(GPT2.small).total).toBe(124_439_808);
    expect(gptParams(GPT2.medium).total).toBe(354_823_168);
    expect(gptParams(GPT2.large).total).toBe(774_030_080);
    expect(gptParams(GPT2.xl).total).toBe(1_557_611_200);
    const s = gptParams(GPT2.small);
    expect(s.embed).toBe(38_597_376);
    expect(s.embed / s.total).toBeCloseTo(0.31, 2);
    expect(s.ffn / s.attn).toBeCloseTo(2, 2);
  });
  it('depth experiment numbers quoted in the text', () => {
    const p1 = depthStudy(1, false, false).grad;
    const p48 = depthStudy(48, false, false).grad;
    expect(p1).toBeCloseTo(0.64, 2);
    expect(p48).toBeCloseTo(0.037, 3);
    expect(p48 / p1).toBeCloseTo(0.06, 2);
    for (const L of [1, 4, 12, 24, 48]) expect(depthStudy(L, true, false).grad).toBeGreaterThan(1);
    expect(depthStudy(48, true, false).grad).toBeCloseTo(28, 0);
    expect(depthStudy(48, true, true).grad).toBeCloseTo(6, 0);
  });
});
