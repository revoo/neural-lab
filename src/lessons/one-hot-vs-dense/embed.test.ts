// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  MINI_VOCAB,
  embedLabExhaustive,
  embeddingCosts,
  evalEmbedBrief1,
  evalEmbedBrief2,
  evalEmbedBrief3,
  similarityMatrix,
  trainRepClassifier,
  verifyLookupEquivalence,
} from './embed';

describe('one-hot-vs-dense: orthogonality vs semantic geometry', () => {
  it('proves one-hot pairs are strictly orthogonal while dense vectors cluster by meaning', () => {
    const oh = similarityMatrix('onehot');
    for (let i = 0; i < MINI_VOCAB.length; i++) {
      for (let j = 0; j < MINI_VOCAB.length; j++) {
        expect(oh[i][j]).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
    const dn = similarityMatrix('dense');
    expect(dn[0][1]).toBeCloseTo(0.996, 3); // cat vs kitten
    expect(dn[0][2]).toBeCloseTo(0.971, 3); // cat vs dog
    expect(dn[4][5]).toBeCloseTo(0.931, 3); // car vs truck
    expect(dn[0][4]).toBeCloseTo(-0.923, 3); // cat vs car
  });

  it('verifies exact e_i^T E === E[i] equivalence and GPT-2 / Llama-3 table costs', () => {
    for (let id = 0; id < MINI_VOCAB.length; id++) {
      const r = verifyLookupEquivalence(id);
      expect(r.maxErr).toBe(0);
      expect(r.zeroMults).toBe(21);
      expect(r.nonZeroMults).toBe(3);
    }
    const gpt2 = embeddingCosts(50_257, 768, 2);
    expect(gpt2.params).toBe(38_597_376);
    expect(gpt2.mib).toBeCloseTo(73.62, 2);
    const llama3 = embeddingCosts(128_256, 4096, 2);
    expect(llama3.params).toBe(525_336_576);
    expect(llama3.mib).toBeCloseTo(1002.0, 1);
  });
});

describe('one-hot-vs-dense: zero-shot synonym transfer and exhaustive lab', () => {
  it('shows one-hot memorises train (100%) and fails unseen synonyms (50%), scalar-id fails both (50%), dense hits 100%', () => {
    const oh = trainRepClassifier('onehot');
    expect(oh.trainAcc).toBeCloseTo(1.0, 4);
    expect(oh.testAcc).toBeCloseTo(0.5, 4);

    const sc = trainRepClassifier('scalar-id');
    expect(sc.trainAcc).toBeCloseTo(0.5, 4);
    expect(sc.testAcc).toBeCloseTo(0.5, 4);

    const d2 = trainRepClassifier('dense-2');
    expect(d2.trainAcc).toBeCloseTo(1.0, 4);
    expect(d2.testAcc).toBeCloseTo(1.0, 4);
    expect(d2.params).toBe(3);
  });

  it('pins all 3 exhaustive lab briefs', () => {
    expect(evalEmbedBrief1('dense-2').pass).toBe(true);
    expect(embedLabExhaustive('transfer')).toMatchObject({ total: 4, passing: 2 });

    expect(evalEmbedBrief2(128, 1).mib).toBeCloseTo(6.104, 3);
    expect(evalEmbedBrief2(128, 1).pass).toBe(true);
    expect(evalEmbedBrief2(128, 2).pass).toBe(false);
    expect(embedLabExhaustive('edge-ram')).toMatchObject({ total: 12, passing: 3 });

    expect(evalEmbedBrief3('onehot').params).toBe(12_800_128);
    expect(evalEmbedBrief3('emb-16').params).toBe(1_602_176);
    expect(evalEmbedBrief3('emb-16').pass).toBe(true);
    expect(embedLabExhaustive('sku-mlp')).toMatchObject({ total: 5, passing: 2 });
  });
});
