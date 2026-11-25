// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { hfGenerateLine } from './realCode';

describe('generate code line', () => {
  it('turns off top-k at the full toy vocabulary', () => {
    expect(hfGenerateLine(0.7, 16, 0.9)).toBe('model.generate(**inputs, do_sample=True, temperature=0.7, top_k=0, top_p=0.90)');
    expect(hfGenerateLine(1, 5, 1)).toBe('model.generate(**inputs, do_sample=True, temperature=1.0, top_k=5, top_p=1.00)');
  });
});

describe('live draws', () => {
  it('inverse-CDF draws respect the cut and the probabilities', async () => {
    const { drawTokens, applySampling, CONTEXTS } = await import('./sampling');
    expect(drawTokens([0.5, 0, 0.5], [0.1, 0.49, 0.5, 0.99])).toEqual([0, 0, 2, 2]);
    const flat = CONTEXTS[2];
    const r = applySampling(flat.tokens, 1, 16, 0.9);
    const us = Array.from({ length: 2000 }, (_, i) => (i + 0.5) / 2000);
    const draws = drawTokens(r.probs, us);
    expect(draws.every((i) => flat.tokens[i].valid)).toBe(true);
    const junkAll = drawTokens(applySampling(flat.tokens, 1, 16, 1).probs, us).filter((i) => !flat.tokens[i].valid).length / 2000;
    expect(junkAll).toBeCloseTo(applySampling(flat.tokens, 1, 16, 1).junkMass, 2);
  });
});
