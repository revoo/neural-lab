import { describe, expect, it } from 'vitest';
import { PUZZLES, WORDS, attend, maxWeightStudy, sentenceAttention, softmax, solve } from './attention';

describe('attention basics', () => {
  it('weights add up to 1 and the output is the weighted blend of values', () => {
    const r = attend([1, 2], [[1, 0], [0, 1], [-1, -1]], [[10, 0], [0, 10], [5, 5]]);
    expect(r.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(r.out[0]).toBeCloseTo(10 * r.weights[0] + 5 * r.weights[2], 12);
    expect(softmax([1000, 0])[0]).toBeCloseTo(1, 12); // numerically safe
  });
});

/** Numbers and behaviours quoted in the attention lesson. */
describe('the attention lesson', () => {
  it('"it" attends to "animal" (98%); verbs split between the two nouns; function words spread evenly', () => {
    const A = sentenceAttention();
    expect(A[7].weights[1]).toBeCloseTo(0.98, 2);
    expect(A[3].weights[1]).toBeCloseTo(0.4, 2);
    expect(A[3].weights[5]).toBeCloseTo(0.4, 2);
    for (const w of A[0].weights) expect(w).toBeCloseTo(0.1, 12);
  });
  it('switching "tired" to "wide" changes nothing for "it" in one layer', () => {
    const wide = sentenceAttention(WORDS.map((t, i) => (i === 9 ? { ...t, w: 'wide' } : t)));
    expect(wide[7].weights).toEqual(sentenceAttention()[7].weights);
  });
  it('√d: score size ~1.3 at d=2 and ~11 at d=128; max weight 38% → 95% raw, ~31% scaled', () => {
    const s2 = maxWeightStudy(2);
    const s128 = maxWeightStudy(128);
    const s512 = maxWeightStudy(512);
    expect(s2.spread).toBeCloseTo(1.3, 1);
    expect(s128.spread).toBeGreaterThan(10.5);
    expect(s128.spread).toBeLessThan(11);
    expect(s2.raw).toBeCloseTo(0.38, 2);
    expect(s512.raw).toBeCloseTo(0.95, 2);
    for (const d of [2, 8, 32, 128, 512]) expect(maxWeightStudy(d).scaled).toBeCloseTo(0.31, 1);
  });
  it('puzzle optimums, from trying every grid query', () => {
    const by = Object.fromEntries(PUZZLES.map((p) => [p.id, solve(p)]));
    expect(by.spotlight.solutions.length).toBe(31);
    expect(by.spotlight.total).toBe(121);
    expect(by.spotlight.smallest).toBe(2);
    expect(by.neighbours.solutions).toEqual([[5, -3]]);
    expect(by.neighbours.bestFirst).toBeCloseTo(0.807, 3);
    expect(by.split.solutions).toEqual([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]);
    expect(by.hidden.solutions.length).toBe(0);
    expect(by.hidden.bestFirst).toBeLessThanOrEqual(1 / 3 + 1e-12);
    expect(by.hidden.bestFirst).toBeGreaterThan(0.33);
  });
  it('the hidden key really is the average of two others (the impossibility argument)', () => {
    const k = Object.fromEntries(PUZZLES.find((p) => p.id === 'hidden')!.keys.map((x) => [x.name, x.k]));
    expect(k.middle).toEqual([(k.left[0] + k.right[0]) / 2, (k.left[1] + k.right[1]) / 2]);
  });
});
