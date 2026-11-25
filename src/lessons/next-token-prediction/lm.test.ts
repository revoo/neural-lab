// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import {
  BASELINES,
  ID_OF,
  TEST_SAMPLES,
  TEST_SENTENCES,
  TRAIN_SAMPLES,
  TinyNeuralLM,
  V,
  VOCAB,
  bigramProb,
  buildTrainingRun,
  evalSamples,
  makeSamples,
  unigramProb,
} from './lm';

describe('TinyNeuralLM maths and gradient check', () => {
  it('matches analytical gradients against central finite differences for every parameter tensor', () => {
    const lm = new TinyNeuralLM(13);
    const batch = TRAIN_SAMPLES.slice(0, 12);
    const grads = lm.gradients(batch);
    const eps = 1e-5;

    const checkTensor = (arr: Float64Array, gArr: Float64Array, indices: number[]) => {
      for (const idx of indices) {
        const orig = arr[idx];
        arr[idx] = orig + eps;
        const lp = lm.evaluate(batch).loss;
        arr[idx] = orig - eps;
        const lmLoss = lm.evaluate(batch).loss;
        arr[idx] = orig;
        const num = (lp - lmLoss) / (2 * eps);
        expect(gArr[idx]).toBeCloseTo(num, 5);
      }
    };

    // Sample indices across E (including active context words), W1, b1, W2, b2
    const activeWords = Array.from(new Set(batch.flatMap((d) => [d.c0, d.c1]))).slice(0, 4);
    const eIdx = activeWords.flatMap((w) => [w * 6, w * 6 + 3]);
    checkTensor(lm.E, grads.E, eIdx);
    checkTensor(lm.W1, grads.W1, [0, 7, 25, 101, 190]);
    checkTensor(lm.b1, grads.b1, [0, 5, 15]);
    checkTensor(lm.W2, grads.W2, [0, 17, 120, 450, 960]);
    checkTensor(lm.b2, grads.b2, [0, 1, 12, 35, 60]);
  });

  it('produces valid probability distributions summing to 1', () => {
    const lm = new TinyNeuralLM(7);
    const { probs } = lm.forward(1, 2);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
    expect(probs.every((p) => p > 0 && p < 1)).toBe(true);
  });
});

describe('corpus counts, baselines, and training claims quoted in prose', () => {
  it('matches vocabulary and split sizes', () => {
    expect(V).toBe(61);
    expect(TRAIN_SAMPLES.length).toBe(268);
    expect(TEST_SAMPLES.length).toBe(37);
    expect(new TinyNeuralLM(7).paramCount).toBe(1611);
  });

  it('matches validation-tuned n-gram baselines on held-out test sentences', () => {
    expect(BASELINES.uniAlpha).toBe(0.01);
    expect(BASELINES.biAlpha).toBe(0.01);
    expect(BASELINES.testUniform.ppl).toBeCloseTo(61, 1);
    expect(BASELINES.testUnigram.ppl).toBeCloseTo(32.8, 1);
    expect(BASELINES.testBigram.ppl).toBeCloseTo(5.57, 2);
  });

  it('neural LM drops from ~62 to ~4.5 test perplexity and beats bigram across 6 seeds', () => {
    const run = buildTrainingRun(7, 120, 10);
    expect(run.history[0].testPpl).toBeCloseTo(61.9, 1);
    expect(run.history[run.history.length - 1].testPpl).toBeCloseTo(4.5, 1);
    expect(run.history[run.history.length - 1].trainPpl).toBeCloseTo(3.7, 1);

    // Unseen bigram "the fox" -> "watched": neural assigns ~8x higher probability than bigram
    const theId = ID_OF.get('the')!;
    const foxId = ID_OF.get('fox')!;
    const watchedId = ID_OF.get('watched')!;
    const pBi = bigramProb(foxId, watchedId, BASELINES.biAlpha);
    const pNeu = run.final.forward(theId, foxId).probs[watchedId];
    expect(pBi).toBeCloseTo(0.0028, 3);
    expect(pNeu).toBeGreaterThan(pBi * 6);

    // Multi-seed honesty check: every seed in [1,2,3,4,5,7] beats bigram (5.57), average ~4.74
    const ppls = [1, 2, 3, 4, 5, 7].map((s) => {
      const m = new TinyNeuralLM(s);
      m.step(120);
      return m.evaluate(TEST_SAMPLES).ppl;
    });
    for (const p of ppls) expect(p).toBeLessThan(BASELINES.testBigram.ppl);
    const avg = ppls.reduce((a, b) => a + b, 0) / ppls.length;
    expect(avg).toBeCloseTo(4.74, 1);
  });

  it('pins the honest per-sentence nuance (neural wins on novel pairs, bigram wins on all-seen pairs)', () => {
    const run = buildTrainingRun(7, 120, 10);
    const s0 = makeSamples(TEST_SENTENCES[0]); // "the fox watched the small mouse in the yard ."
    const s3 = makeSamples(TEST_SENTENCES[3]); // "a bright moon shone over the quiet house ."
    const bi0 = evalSamples(s0, (_a, c1, y) => bigramProb(c1, y, BASELINES.biAlpha)).ppl;
    const neu0 = run.final.evaluate(s0).ppl;
    expect(bi0).toBeCloseTo(13.3, 1);
    expect(neu0).toBeCloseTo(5.5, 1);

    const bi3 = evalSamples(s3, (_a, c1, y) => bigramProb(c1, y, BASELINES.biAlpha)).ppl;
    const neu3 = run.final.evaluate(s3).ppl;
    expect(bi3).toBeCloseTo(2.84, 1);
    expect(neu3).toBeCloseTo(4.48, 1);
    expect(bi3).toBeLessThan(neu3);

    // Unigram on sentence 0
    const uni0 = evalSamples(s0, (_a, _b, y) => unigramProb(y, BASELINES.uniAlpha)).ppl;
    expect(uni0).toBeCloseTo(26.4, 1);
  });

  it('matches the Python snippet perplexity examples', () => {
    const ppl = (ps: number[]) => Math.exp(-ps.reduce((s, p) => s + Math.log(p), 0) / ps.length);
    expect(ppl([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])).toBeCloseTo(6.0, 2);
    expect(ppl([0.8, 0.9, 0.85, 0.9, 0.95])).toBeCloseTo(1.14, 2);
    expect(ppl([0.8, 0.9, 0.85, 0.9, 0.01])).toBeCloseTo(2.83, 2);
    expect(VOCAB[0]).toBe('<start>');
  });
});
