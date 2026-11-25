// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAB_BRIEFS,
  binomialPmf,
  evaluateEvalDesign,
  judgeModel,
  labExhaustive,
  mcnemarExactP,
  pairedComparison,
  simulateEvalDesign,
  unpairedExactP,
  winnersCurse,
} from './eval';

const p1 = (v: number) => (100 * v).toFixed(1);

describe('exact binomial & paired vs unpaired comparisons', () => {
  it('binomial PMF sums to 1 and matches exact small values', () => {
    const b = binomialPmf(6, 0.5);
    expect(b.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 12);
    expect(b[6]).toBeCloseTo(1 / 64, 12);
    expect(mcnemarExactP(6, 0)).toBeCloseTo(2 / 64, 12); // 0.03125
  });

  it('pins the 84% vs 78% on n=100 paired vs unpaired numbers quoted in the lesson', () => {
    const pure = pairedComparison(100, 84, 78, 78);
    expect([pure.n10, pure.n01, pure.disagreements]).toEqual([6, 0, 6]);
    expect(pure.mcnemarP).toBeCloseTo(0.03125, 4);
    expect(pure.unpairedP).toBeCloseTo(0.3208, 3);

    const traded = pairedComparison(100, 84, 78, 66);
    expect([traded.n10, traded.n01, traded.disagreements]).toEqual([18, 12, 30]);
    expect(traded.mcnemarP).toBeCloseTo(0.3616, 3);
    expect(unpairedExactP(100, 84, 78)).toBeCloseTo(0.3208, 3);
  });
});

describe('Winner’s Curse (eval overfitting)', () => {
  it('pins exact expected max and +5pp mirage probability for n=50 and n=500', () => {
    const k1 = winnersCurse(50, 0.8, 1);
    expect(p1(k1.expectedMax)).toBe('80.0');
    expect(p1(k1.probPlus5)).toBe('19.0');

    const k5 = winnersCurse(50, 0.8, 5);
    expect(p1(k5.expectedMax)).toBe('86.4');
    expect(p1(k5.expectedMirage)).toBe('6.4');
    expect(p1(k5.probPlus5)).toBe('65.2');

    const k20 = winnersCurse(50, 0.8, 20);
    expect(p1(k20.expectedMax)).toBe('89.9');
    expect(p1(k20.expectedMirage)).toBe('9.9');
    expect(p1(k20.probPlus5)).toBe('98.5');

    const big = winnersCurse(500, 0.8, 20);
    expect(p1(big.expectedMax)).toBe('83.3');
    expect(p1(big.expectedMirage)).toBe('3.3');
    expect(p1(big.probPlus5)).toBe('4.7');
  });
});

describe('LLM-as-a-judge position bias model', () => {
  it('cancels position bias exactly under position-swapping when models are tied', () => {
    const tied = judgeModel(0.25, 0.25, 0.85, 0.2);
    expect(p1(tied.aFirst.winA)).toBe('65.0');
    expect(p1(tied.bFirst.winA)).toBe('35.0');
    expect(tied.swapped.winA).toBeCloseTo(tied.swapped.winB, 12);
    expect(p1(tied.swapped.winA)).toBe('28.6');
    expect(p1(tied.swapped.flip)).toBe('42.8');

    const adv = judgeModel(0.35, 0.25, 0.85, 0.2);
    expect(p1(adv.bFirst.winB)).toBe('60.5'); // position bias hides A's true +10pp edge
    expect(p1(adv.swapped.winA)).toBe('33.6');
    expect(p1(adv.swapped.winB)).toBe('26.6');
  });
});

describe('Lab: Size Your Eval Suite (exact binomial + Monte Carlo check)', () => {
  it('exact paired and unpaired power/FPR match 30,000-trial Monte Carlo simulation', () => {
    const dP = { mode: 'paired' as const, grader: 'judge2' as const, n: 200, alpha: 0.05 };
    const exP = evaluateEvalDesign(dP, LAB_BRIEFS[0]);
    const mcP = simulateEvalDesign(dP, LAB_BRIEFS[0], 30_000, 99);
    expect(Math.abs(exP.power - mcP.power)).toBeLessThan(0.01);

    const dU = { mode: 'unpaired' as const, grader: 'judge1' as const, n: 300, alpha: 0.05 };
    const exU = evaluateEvalDesign(dU, LAB_BRIEFS[0]);
    const mcU = simulateEvalDesign(dU, LAB_BRIEFS[0], 30_000, 99);
    expect(Math.abs(exU.fpr - mcU.fpr)).toBeLessThan(0.005);
    expect(Math.abs(exU.power - mcU.power)).toBeLessThan(0.01);
  });

  it('pins exhaustive counts and unique cheapest optimums across all 3 briefs (one winner per grader)', () => {
    const [b1, b2, b3] = LAB_BRIEFS.map(labExhaustive);

    expect([b1.total, b1.passing, b1.bestCost]).toEqual([264, 32, 1200]);
    expect(b1.best).toHaveLength(1);
    expect(b1.best[0].design).toEqual({ mode: 'paired', grader: 'judge1', n: 2000, alpha: 0.05 });
    expect(p1(b1.best[0].eval.power)).toBe('86.3');

    expect([b2.total, b2.passing, b2.bestCost]).toEqual([216, 10, 1500]);
    expect(b2.best).toHaveLength(1);
    expect(b2.best[0].design).toEqual({ mode: 'paired', grader: 'judge2', n: 1000, alpha: 0.05 });
    expect(p1(b2.best[0].eval.power)).toBe('87.7');

    expect([b3.total, b3.passing, b3.bestCost]).toEqual([264, 5, 7500]);
    expect(b3.best).toHaveLength(1);
    expect(b3.best[0].design).toEqual({ mode: 'paired', grader: 'human', n: 1000, alpha: 0.02 });
    expect(p1(b3.best[0].eval.power)).toBe('85.4');
  }, 15_000);
});
