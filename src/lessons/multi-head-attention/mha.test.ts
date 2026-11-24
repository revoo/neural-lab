import { describe, expect, it } from 'vitest';
import { COMBINED, IT, PREDICATE, REFERENT, attentionFlops, attentionParams, oneHeadShares, runHead } from './mha';

describe('the multi-head attention lesson', () => {
  it('two heads: referent 98%, predicates 99%', () => {
    expect(runHead(REFERENT)[IT].weights[1]).toBeCloseTo(0.98, 2);
    const p = runHead(PREDICATE)[IT].weights;
    expect(p[3] + p[9]).toBeCloseTo(0.99, 2);
  });
  it('one combined head: the referent takes 91%, the predicates 6%', () => {
    const w = runHead(COMBINED)[IT].weights;
    expect(w[1]).toBeCloseTo(0.91, 2);
    expect(w[3] + w[9]).toBeCloseTo(0.06, 2);
  });
  it('no balance gives both more than half; the best compromise is about 46% each', () => {
    let best = 0;
    for (let b = 0; b <= 1.00001; b += 0.01) {
      const s = oneHeadShares(b);
      expect(Math.min(s.referent, s.predicates)).toBeLessThan(0.5);
      best = Math.max(best, Math.min(s.referent, s.predicates));
    }
    expect(best).toBeCloseTo(0.46, 2);
  });
  it('GPT-2 small: 2,362,368 attention parameters per layer, independent of the number of heads', () => {
    expect(attentionParams(768, 12).total).toBe(2_362_368);
    expect(attentionParams(768, 12).weights).toBe(2_359_296);
    expect(attentionParams(768, 1).total).toBe(attentionParams(768, 12).total);
    expect(attentionParams(768, 12).headSize).toBe(64);
    expect(attentionFlops(1024, 768, 12)).toBe(attentionFlops(1024, 768, 1));
  });
});
