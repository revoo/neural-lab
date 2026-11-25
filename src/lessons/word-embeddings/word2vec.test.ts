// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  evalW2vBrief1,
  evalW2vBrief2,
  evalW2vBrief3,
  getVocab,
  initSgns,
  sgnsTupleLossAndGrad,
  trainSgns,
  w2vLabExhaustive,
} from './word2vec';

describe('word-embeddings: SGNS gradients, analogies, and collapse', () => {
  it('matches central finite differences on SGNS center, positive, and negative gradients to < 1e-8', () => {
    const m = initSgns(5, 19);
    const c = 0;
    const o = 3;
    const negs = [5, 8, 12];
    const base = sgnsTupleLossAndGrad(m, c, o, negs);
    const eps = 1e-5;
    let maxErr = 0;

    const check = (vec: number[], grad: number[]) => {
      for (let j = 0; j < vec.length; j++) {
        const old = vec[j];
        vec[j] = old + eps;
        const lp = sgnsTupleLossAndGrad(m, c, o, negs).loss;
        vec[j] = old - eps;
        const lm = sgnsTupleLossAndGrad(m, c, o, negs).loss;
        vec[j] = old;
        maxErr = Math.max(maxErr, Math.abs((lp - lm) / (2 * eps) - grad[j]));
      }
    };
    check(m.W[c], base.gVc);
    check(m.U[o], base.gUo);
    negs.forEach((n, idx) => check(m.U[n], base.gNegs[idx]));
    expect(maxErr).toBeLessThan(1e-8);
    expect(getVocab().words.length).toBe(41);
  });

  it('trains from scratch to nail 4/4 gender & capital analogies with high cosine', () => {
    const res = trainSgns(6, 5, 2, 140, 42);
    expect(res.analogyHits).toBe(4);
    expect(res.meanCos).toBeCloseTo(0.0, 1);
    expect(res.analogyDetails[0].top1).toBe('queen');
    expect(res.analogyDetails[0].top1Sim).toBeGreaterThan(0.95);
    expect(res.analogyDetails[1].top1).toBe('princess');
    expect(res.analogyDetails[2].top1).toBe('berlin');
    expect(res.analogyDetails[3].top1).toBe('tokyo');
  });

  it('collapses toward one cone when negative samples K = 0', () => {
    const col = trainSgns(6, 0, 2, 140, 42);
    expect(col.meanCos).toBeGreaterThan(0.8);
    expect(col.analogyHits).toBe(1);
  });
});

describe('word-embeddings: Lab (Train & Debug Word2Vec) exhaustive checks', () => {
  it('Brief 1 (cure collapse): K=0 fails, K=1/3/5 pass (3/4)', () => {
    expect(evalW2vBrief1(0).pass).toBe(false);
    expect(evalW2vBrief1(5).pass).toBe(true);
    expect(w2vLabExhaustive('collapse')).toMatchObject({ total: 4, passing: 3 });
  });

  it('Brief 2 (minimal dimension for 4/4 analogies): d=2 fails all windows; d=3 w=3 is smallest winner (9/15)', () => {
    expect(evalW2vBrief2(2, 1).pass).toBe(false);
    expect(evalW2vBrief2(2, 2).pass).toBe(false);
    expect(evalW2vBrief2(2, 3).pass).toBe(false);
    expect(evalW2vBrief2(3, 3).pass).toBe(true);
    const ex = w2vLabExhaustive('analogies');
    expect(ex.total).toBe(15);
    expect(ex.passing).toBe(9);
    expect(ex.summary).toContain('d = 3, window = 3');
  });

  it('Brief 3 (V=100k compute scale): Full Softmax fails; sgns-5 is 16,667x faster (2/6 pass)', () => {
    expect(evalW2vBrief3('full-softmax').pass).toBe(false);
    const s5 = evalW2vBrief3('sgns-5');
    expect(s5.pass).toBe(true);
    expect(s5.dotsPerPair).toBe(6);
    expect(Math.round(s5.speedup)).toBe(16_667);
    expect(w2vLabExhaustive('scale')).toMatchObject({ total: 6, passing: 2 });
  });
});
