// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import {
  CONTEXTS,
  LAB_LEVELS,
  applySampling,
  evaluateSampler,
  runBeamSearch,
  solveLabLevel,
} from './sampling';

describe('sampling pipeline maths', () => {
  it('probabilities always sum to 1 and truncated tokens get exact 0', () => {
    for (const ctx of CONTEXTS) {
      for (const T of [0.2, 0.8, 1.0, 1.8]) {
        for (const [k, p] of [
          [1, 1.0],
          [5, 1.0],
          [16, 0.9],
          [8, 0.85],
        ]) {
          const r = applySampling(ctx.tokens, T, k, p);
          const sum = r.probs.reduce((a, b) => a + b, 0);
          expect(sum).toBeCloseTo(1, 10);
          expect(r.validMass + r.junkMass).toBeCloseTo(1, 10);
          r.kept.forEach((isKept, i) => {
            if (!isKept) expect(r.probs[i]).toBe(0);
            else expect(r.probs[i]).toBeGreaterThan(0);
          });
        }
      }
    }
  });

  it('matches the sharp vs medium vs flat comparison quoted in prose (T=1)', () => {
    const sRaw = applySampling(CONTEXTS[0].tokens, 1.0, 16, 1.0);
    const sTopK = applySampling(CONTEXTS[0].tokens, 1.0, 5, 1.0);
    const sTopP = applySampling(CONTEXTS[0].tokens, 1.0, 16, 0.9);
    expect(sRaw.validMass).toBeCloseTo(0.904, 3);
    expect(sRaw.junkMass).toBeCloseTo(0.096, 3);
    expect(sTopK.poolSize).toBe(5);
    expect(sTopK.junkMass).toBeCloseTo(0.053, 3);
    expect(sTopP.poolSize).toBe(1);
    expect(sTopP.validMass).toBe(1);
    expect(sTopP.junkMass).toBe(0);

    const mTopP = applySampling(CONTEXTS[1].tokens, 1.0, 16, 0.9);
    expect(mTopP.poolSize).toBe(7);
    expect(mTopP.effectiveChoices).toBeCloseTo(5.62, 2);
    expect(mTopP.junkMass).toBe(0);

    const fTopK = applySampling(CONTEXTS[2].tokens, 1.0, 5, 1.0);
    const fTopP = applySampling(CONTEXTS[2].tokens, 1.0, 16, 0.9);
    expect(fTopK.poolSize).toBe(5);
    expect(fTopK.effectiveChoices).toBeCloseTo(4.97, 2);
    expect(fTopP.poolSize).toBe(11);
    expect(fTopP.effectiveChoices).toBeCloseTo(10.64, 2);
    expect(fTopP.junkMass).toBe(0);
  });

  it('temperature alone never zeroes out tail junk (softmax is strictly positive)', () => {
    for (const T of [0.1, 0.5, 1.0, 2.0]) {
      const m = evaluateSampler(T, 16, 1.0);
      expect(m.maxTailJunk).toBeGreaterThan(0);
    }
  });
});

describe('SamplerLab exhaustive grid results', () => {
  it('pins all three levels’ exact counts and optimums over all 6,080 slider settings', () => {
    const l1 = solveLabLevel(LAB_LEVELS[0]);
    expect(l1.gridSize).toBe(6080);
    expect(l1.passing).toBe(4849);
    expect(l1.tempOnlyPassing).toBe(0);
    expect(l1.topKOnlyPassing).toBe(1);
    expect(l1.topPOnlyPassing).toBe(17);
    expect(l1.best!.T).toBe(0.8);
    expect(l1.best!.k).toBe(12);
    expect(l1.best!.p).toBe(0.95);
    expect(l1.best!.metrics.flatEff).toBeCloseTo(11.26, 2);

    const l2 = solveLabLevel(LAB_LEVELS[1]);
    expect(l2.gridSize).toBe(16);
    expect(l2.passing).toBe(0);
    expect(l2.best).toBeNull();
    expect(evaluateSampler(1.0, 2, 1.0).sharpAcc).toBeCloseTo(0.9802, 3);
    expect(evaluateSampler(1.0, 2, 1.0).flatEff).toBeCloseTo(2.0, 2);
    expect(evaluateSampler(1.0, 10, 1.0).sharpAcc).toBeCloseTo(0.9191, 3);
    expect(evaluateSampler(1.0, 10, 1.0).flatEff).toBeCloseTo(9.74, 2);

    const l3 = solveLabLevel(LAB_LEVELS[2]);
    expect(l3.gridSize).toBe(6080);
    expect(l3.passing).toBe(88);
    expect(l3.tempOnlyPassing).toBe(0);
    expect(l3.topKOnlyPassing).toBe(0);
    expect(l3.topPOnlyPassing).toBe(2);
    expect(l3.best!.T).toBe(0.9);
    expect(l3.best!.k).toBe(12);
    expect(l3.best!.p).toBe(0.95);
    expect(l3.best!.metrics.sharpAcc).toBeCloseTo(0.987, 3);
    expect(l3.best!.metrics.flatEff).toBeCloseTo(11.4, 2);
    expect(l3.best!.metrics.maxTailJunk).toBe(0);
  });
});

describe('beam search vs greedy tree', () => {
  it('greedy picks "The cat slept" (8.8%), while beam=2 finds "A bright sun" (34.4%, ~3.9x higher)', () => {
    const greedy = runBeamSearch(1, 3);
    expect(greedy[0].words).toEqual(['The', 'cat', 'slept']);
    expect(greedy[0].prob).toBeCloseTo(0.088, 4);

    const beam2 = runBeamSearch(2, 3);
    expect(beam2[0].words).toEqual(['A', 'bright', 'sun']);
    expect(beam2[0].prob).toBeCloseTo(0.34425, 5);
    expect(beam2[0].prob / greedy[0].prob).toBeCloseTo(3.91, 2);
  });
});
