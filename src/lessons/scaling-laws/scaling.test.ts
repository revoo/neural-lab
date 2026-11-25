// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import {
  BUDGET_BRIEFS,
  CHINCHILLA,
  FRONTIER_RUNS,
  HOFFMANN_ROUNDED,
  chinchillaOptimal,
  inferenceFlops,
  lifecycleOptimal,
  lossTerms,
  predictedLoss,
  tokensForLoss,
  trainFlops,
} from './scaling';

describe('Chinchilla scaling law maths', () => {
  it('calculus optimum satisfies 6 N* D* = C and beats neighbouring splits', () => {
    for (const C of [1e21, 1e22, 3.15e23, 5.88e23, 6.3e24]) {
      const opt = chinchillaOptimal(C);
      expect(trainFlops(opt.N, opt.D)).toBeCloseTo(C, -15);
      // Perturbing N by ±2% at fixed C strictly increases loss
      const Lleft = predictedLoss(opt.N * 0.98, C / (6 * opt.N * 0.98));
      const Lright = predictedLoss(opt.N * 1.02, C / (6 * opt.N * 1.02));
      expect(opt.total).toBeLessThan(Lleft);
      expect(opt.total).toBeLessThan(Lright);
      // Excess loss scales cleanly as a power law in C
      expect(opt.excess).toBeCloseTo(opt.paramPenalty + opt.dataPenalty, 10);
    }
  });

  it('pins the frontier model comparisons quoted in prose', () => {
    const byId = Object.fromEntries(FRONTIER_RUNS.map((r) => [r.id, r]));

    const gpt3C = trainFlops(byId.gpt3.N, byId.gpt3.D);
    expect(gpt3C).toBeCloseTo(3.15e23, -20);
    expect(predictedLoss(byId.gpt3.N, byId.gpt3.D)).toBeCloseTo(2.0085, 4);
    const gpt3Opt = chinchillaOptimal(gpt3C);
    expect(gpt3Opt.N / 1e9).toBeCloseTo(53.0, 1);
    expect(gpt3Opt.D / 1e12).toBeCloseTo(0.99, 2);
    expect(gpt3Opt.ratio).toBeCloseTo(18.7, 1);
    expect(gpt3Opt.total).toBeCloseTo(1.9923, 4);

    const gopherC = trainFlops(byId.gopher.N, byId.gopher.D);
    expect(predictedLoss(byId.gopher.N, byId.gopher.D)).toBeCloseTo(1.9996, 4);
    const gopherOpt = chinchillaOptimal(gopherC);
    expect(gopherOpt.N / 1e9).toBeCloseTo(67.5, 1);
    expect(gopherOpt.total).toBeCloseTo(1.9782, 4);

    const chinC = trainFlops(byId.chinchilla.N, byId.chinchilla.D);
    expect(chinC).toBeCloseTo(5.88e23, -20);
    expect(predictedLoss(byId.chinchilla.N, byId.chinchilla.D)).toBeCloseTo(1.9739, 4);
    const chinOpt = chinchillaOptimal(chinC);
    expect(chinOpt.N / 1e9).toBeCloseTo(73.0, 1);
    expect(chinOpt.D / 1e12).toBeCloseTo(1.34, 2);
    expect(chinOpt.total).toBeCloseTo(1.9739, 4);

    // Chinchilla 70B beats Gopher 280B at 4x fewer parameters
    expect(predictedLoss(byId.chinchilla.N, byId.chinchilla.D)).toBeLessThan(predictedLoss(byId.gopher.N, byId.gopher.D));

    // Honesty note: Hoffmann et al. (2022) rounded Approach-3 constants imply ~93 tok/param at Chinchilla compute
    const roundedOpt = chinchillaOptimal(chinC, HOFFMANN_ROUNDED);
    expect(roundedOpt.ratio).toBeCloseTo(92.8, 1);
  });

  it('pins the inference + training lifecycle compute trade-off', () => {
    const opt0 = lifecycleOptimal(2.0, 0);
    expect(opt0.N / 1e9).toBeCloseTo(46.9, 1);
    expect(opt0.D / 1e12).toBeCloseTo(0.88, 2);
    expect(Math.round(opt0.ratio)).toBe(19);

    const opt20 = lifecycleOptimal(2.0, 20e12);
    expect(opt20.N / 1e9).toBeCloseTo(17.1, 1);
    expect(opt20.D / 1e12).toBeCloseTo(4.33, 2);
    expect(Math.round(opt20.ratio)).toBe(253);
    expect(opt20.totalC).toBeCloseTo(1.13e24, -22);

    // Compare against 50B Chinchilla model at D_inf = 20T (~2.25e24 FLOPs, ~2x worse!)
    const d50 = tokensForLoss(50e9, 2.0)!;
    const total50 = trainFlops(50e9, d50) + inferenceFlops(50e9, 20e12);
    expect(total50).toBeCloseTo(2.25e24, -22);
    expect(total50 / opt20.totalC).toBeCloseTo(2.0, 1);
  });

  it('pins all three ComputeBudgetLab briefs and their exact optimums', () => {
    // Brief 1: C <= 5.0e23, L <= 1.985
    const b1 = chinchillaOptimal(5.0e23);
    expect(b1.N / 1e9).toBeCloseTo(67.2, 1);
    expect(b1.D / 1e12).toBeCloseTo(1.24, 2);
    expect(b1.total).toBeCloseTo(1.9785, 4);
    expect(BUDGET_BRIEFS[0].check(b1.N, b1.D)).toBe(true);
    expect(BUDGET_BRIEFS[0].check(280e9, 5.0e23 / (6 * 280e9))).toBe(false);

    // Brief 2: N <= 20B, C <= 6.0e23, L <= 1.998 (boundary opt at N=20B, D=5T -> L=1.9904)
    const D2 = 6.0e23 / (6 * 20e9);
    expect(D2).toBeCloseTo(5e12, -10);
    expect(predictedLoss(20e9, D2)).toBeCloseTo(1.9904, 4);
    expect(BUDGET_BRIEFS[1].check(20e9, D2)).toBe(true);

    // Brief 3: L <= 2.00, D_inf = 20T, totalC <= 1.25e24
    const b3 = lifecycleOptimal(2.0, 20e12);
    expect(BUDGET_BRIEFS[2].check(b3.N, b3.D)).toBe(true);
    expect(BUDGET_BRIEFS[2].check(50e9, tokensForLoss(50e9, 2.0)!)).toBe(false);

    // Sanity check irreducible floor
    expect(lossTerms(1e12, 1e14).irreducible).toBe(CHINCHILLA.E);
  });
});
