// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import {
  ALIGNMENT_BRIEFS,
  CANDIDATE_REPLIES,
  CLEAN_PREF_PAIRS,
  dpoLossAndGrad,
  evaluateAlignmentBrief,
  evaluateDpoPoint,
  evaluatePolicy,
  exactRlhfOptimum,
  runLiveAlignment,
  solveAlignmentBrief,
  traceGoodhartCurve,
} from './dpo';

describe('RLHF & Direct Preference Optimisation (DPO) — engine & numbers quoted in prose', () => {
  it('Goodhart over-optimisation curve: rises from SFT 1.339 to peak 2.362 at β=1.1, then crashes to -2.799 at β=0.1', () => {
    const refProbSum = CANDIDATE_REPLIES.reduce((s, r) => s + r.refProb, 0);
    expect(refProbSum).toBeCloseTo(1.0, 10);

    const ref = evaluatePolicy(CANDIDATE_REPLIES.map((r) => r.refProb));
    expect(ref.expectedTrue).toBeCloseTo(1.339, 3);
    expect(ref.hackProb).toBeCloseTo(0.01, 4);

    const { peak } = traceGoodhartCurve();
    expect(peak.beta).toBe(1.1);
    expect(peak.expectedTrue).toBeCloseTo(2.362, 3);
    expect(peak.klNats).toBeCloseTo(0.48, 2);

    const hacked = exactRlhfOptimum(0.1);
    expect(hacked.expectedProxy).toBeCloseTo(5.0, 2);
    expect(hacked.expectedTrue).toBeCloseTo(-2.799, 2);
    expect(hacked.hackProb).toBeGreaterThan(0.999);
  });

  it('DPO loss & gradient weight match analytical formula and Python snippet', () => {
    const wrong = evaluateDpoPoint(-1.0, 1.0, 0.5);
    expect(wrong.implicitRw).toBeCloseTo(-0.5, 5);
    expect(wrong.implicitRl).toBeCloseTo(0.5, 5);
    expect(wrong.margin).toBeCloseTo(-1.0, 5);
    expect(wrong.loss).toBeCloseTo(1.313, 3);
    expect(wrong.gradWeight).toBeCloseTo(0.731, 3);

    const right = evaluateDpoPoint(1.5, -2.5, 0.5);
    expect(right.margin).toBeCloseTo(2.0, 5);
    expect(right.loss).toBeCloseTo(0.127, 3);
    expect(right.gradWeight).toBeCloseTo(0.119, 3);
  });

  it('exact analytical DPO gradients match central finite differences (< 1e-8)', () => {
    const refLogits = CANDIDATE_REPLIES.map((r) => Math.log(r.refProb));
    const logits = refLogits.map((v, i) => v + (i % 2 === 0 ? 0.3 : -0.2));
    const { grad } = dpoLossAndGrad(logits, refLogits, CLEAN_PREF_PAIRS, 0.5);
    const eps = 1e-6;

    for (let k = 0; k < logits.length; k++) {
      const lp = logits.slice();
      const lm = logits.slice();
      lp[k] += eps;
      lm[k] -= eps;
      const num =
        (dpoLossAndGrad(lp, refLogits, CLEAN_PREF_PAIRS, 0.5).loss -
          dpoLossAndGrad(lm, refLogits, CLEAN_PREF_PAIRS, 0.5).loss) /
        (2 * eps);
      expect(Math.abs(grad[k] - num)).toBeLessThan(1e-8);
    }
  });

  it('live DPO beats SFT on suppressing rejected hacks', () => {
    const sft = runLiveAlignment({ mode: 'sft', beta: 0.5, steps: 50 });
    const dpo = runLiveAlignment({ mode: 'dpo', beta: 0.5, steps: 50 });
    expect(dpo.final.expectedTrue).toBeGreaterThan(sft.final.expectedTrue);
    expect(dpo.final.hackProb).toBeLessThan(sft.final.hackProb);
  });

  it('Alignment Lab exhaustive solutions across all 3 briefs', () => {
    const s1 = solveAlignmentBrief(ALIGNMENT_BRIEFS[0]);
    expect(evaluateAlignmentBrief(ALIGNMENT_BRIEFS[0], ALIGNMENT_BRIEFS[0].defaultConfig).passes).toBe(false);
    expect(s1.totalConfigs).toBe(7);
    expect(s1.passingCount).toBe(2);
    expect(s1.bestConfig.beta).toBe(1.0);
    expect(s1.bestEval.metrics.expectedTrue).toBeCloseTo(2.356, 3);

    const s2 = solveAlignmentBrief(ALIGNMENT_BRIEFS[1]);
    expect(evaluateAlignmentBrief(ALIGNMENT_BRIEFS[1], ALIGNMENT_BRIEFS[1].defaultConfig).passes).toBe(false);
    expect(s2.totalConfigs).toBe(47);
    expect(s2.passingCount).toBe(3);
    expect(s2.bestConfig).toEqual({ method: 'dpo', beta: 0.25, steps: 200 });
    expect(s2.bestEval.metrics.expectedTrue).toBeCloseTo(3.003, 3);

    const s3 = solveAlignmentBrief(ALIGNMENT_BRIEFS[2]);
    expect(evaluateAlignmentBrief(ALIGNMENT_BRIEFS[2], ALIGNMENT_BRIEFS[2].defaultConfig).passes).toBe(false);
    expect(s3.totalConfigs).toBe(47);
    expect(s3.passingCount).toBe(17);
    expect(s3.bestConfig).toEqual({ method: 'dpo', beta: 0.5, steps: 200 });
    expect(s3.bestEval.metrics.expectedTrue).toBeCloseTo(3.07, 2);
  });
});
