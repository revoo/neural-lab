// authored-by: Gemini Antigravity
import { describe, it, expect } from 'vitest';
import {
  DEFAULT_STEPPER_SEQ,
  evalRnnBrief1,
  evalRnnBrief2,
  evalRnnBrief3,
  getStepperParams,
  initRnn,
  LAG_BENCH_LAGS,
  rnnLabExhaustive,
  rnnLossAndGrad,
  runLagBenchmark,
  runPascanuCliff,
  spectralGradientFlow,
} from './rnn';

describe('Module 11 Lesson 1: Recurrent Neural Networks & BPTT', () => {
  it('verifies exact BPTT analytic gradients against central finite differences (< 1e-8)', () => {
    for (const act of ['tanh', 'linear'] as const) {
      const p = initRnn(3, 0.9, 11);
      const xs = [0.7, -0.4, 0.9, -0.2, 0.5];
      const { grads } = rnnLossAndGrad(p, xs, 1, { act });
      const eps = 1e-6;
      let maxErr = 0;

      const checkArr = (arr: number[], gradArr: number[]) => {
        for (let i = 0; i < arr.length; i++) {
          const orig = arr[i];
          arr[i] = orig + eps;
          const lp = rnnLossAndGrad(p, xs, 1, { act }).loss;
          arr[i] = orig - eps;
          const lm = rnnLossAndGrad(p, xs, 1, { act }).loss;
          arr[i] = orig;
          maxErr = Math.max(maxErr, Math.abs((lp - lm) / (2 * eps) - gradArr[i]));
        }
      };

      checkArr(p.Wx, grads.dWx);
      checkArr(p.Wh, grads.dWh);
      checkArr(p.bh, grads.dbh);
      checkArr(p.Wy, grads.dWy);
      {
        const orig = p.by;
        p.by = orig + eps;
        const lp = rnnLossAndGrad(p, xs, 1, { act }).loss;
        p.by = orig - eps;
        const lm = rnnLossAndGrad(p, xs, 1, { act }).loss;
        p.by = orig;
        maxErr = Math.max(maxErr, Math.abs((lp - lm) / (2 * eps) - grads.dby));
      }

      expect(maxErr).toBeLessThan(1e-8);
    }
  });

  it('pins unrolled stepper hidden states and backward BPTT gradient decay', () => {
    const leaky = rnnLossAndGrad(getStepperParams('leaky'), DEFAULT_STEPPER_SEQ, 1);
    const longMem = rnnLossAndGrad(getStepperParams('long'), DEFAULT_STEPPER_SEQ, 1);

    expect(leaky.trace.hs[0][0]).toBeCloseTo(0.716, 3);
    expect(leaky.trace.hs[5][0]).toBeCloseTo(-0.188, 3);
    expect(leaky.grads.dhTrace[5][0]).toBeCloseTo(-1.539, 3);
    expect(leaky.grads.dhTrace[0][0]).toBeCloseTo(-0.031, 3);

    expect(longMem.grads.dhTrace[5][0]).toBeCloseTo(-1.740, 3);
    expect(longMem.grads.dhTrace[0][0]).toBeCloseTo(-0.974, 3);
  });

  it('pins exact 20-step spectral-radius Jacobian flow for linear and tanh', () => {
    const lin05 = spectralGradientFlow(0.5, 20, 'linear');
    const lin10 = spectralGradientFlow(1.0, 20, 'linear');
    const lin16 = spectralGradientFlow(1.6, 20, 'linear');
    expect(lin05.step1Ratio).toBeCloseTo(0.5 ** 19, 9);
    expect(lin10.step1Ratio).toBeCloseTo(1.0, 9);
    expect(lin16.step1Ratio).toBeCloseTo(1.6 ** 19, 2);

    const tanh10 = spectralGradientFlow(1.0, 20, 'tanh');
    expect(tanh10.step1Ratio).toBeCloseTo(0.5496, 3);
    expect(tanh10.step1Ratio).toBeLessThan(1.0);
  });

  it('pins Pascanu (2013) exploding cliff: unclipped SGD fails while clipping converges', () => {
    const unclipped = runPascanuCliff(0.015, null, 45, 1.32);
    expect(unclipped.maxRawGrad).toBeCloseTo(93.9, 1);
    expect(unclipped.finalLoss).toBeGreaterThan(0.20);

    const explode = runPascanuCliff(0.05, null, 45, 1.32);
    expect(explode.exploded).toBe(true);

    const clipped = runPascanuCliff(0.015, 1.0, 45, 1.32);
    expect(clipped.exploded).toBe(false);
    expect(clipped.finalLoss).toBeLessThan(1e-12);
    expect(clipped.finalW).toBeCloseTo(1.1865, 3);
  });

  it('pins multi-seed lag degradation of a plain tanh RNN', () => {
    const results = LAG_BENCH_LAGS.map((lag) => runLagBenchmark(lag));
    expect(results[0].meanAcc).toBeCloseTo(1.0, 3); // lag 1
    expect(results[1].meanAcc).toBeCloseTo(1.0, 3); // lag 3
    expect(results[2].meanAcc).toBeCloseTo(0.90, 2); // lag 6
    expect(results[3].meanAcc).toBeLessThan(0.60); // lag 10
    expect(results[4].meanAcc).toBeLessThan(0.55); // lag 15
  });

  it('verifies exhaustive Lab: Tame BPTT counts and optimums', () => {
    const b1 = rnnLabExhaustive('flow');
    expect(b1.total).toBe(10);
    expect(b1.passingCount).toBe(4);
    expect(evalRnnBrief1(1.0, 'linear').pass).toBe(true);
    expect(evalRnnBrief1(0.5, 'tanh').pass).toBe(false);

    const b2 = rnnLabExhaustive('clip');
    expect(b2.total).toBe(12);
    expect(b2.passingCount).toBe(7);
    expect(evalRnnBrief2(0.015, null).pass).toBe(false);
    expect(evalRnnBrief2(0.025, null).pass).toBe(false);
    expect(evalRnnBrief2(0.05, null).pass).toBe(false);
    expect(evalRnnBrief2(0.015, 1.0).pass).toBe(true);

    const b3 = rnnLabExhaustive('tbptt');
    expect(b3.total).toBe(10);
    expect(b3.passingCount).toBe(2);
    expect(evalRnnBrief3(6, 2).pass).toBe(false);
    expect(evalRnnBrief3(7, 2).pass).toBe(true);
    expect(evalRnnBrief3(7, 2).memSlots).toBe(14);
    expect(evalRnnBrief3(7, 4).pass).toBe(false);
  });
});
