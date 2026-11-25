// authored-by: Gemini Antigravity
import { describe, it, expect } from 'vitest';
import {
  carouselGradientFlow,
  cellParamCount,
  evalLstmBrief1,
  evalLstmBrief2,
  evalLstmBrief3,
  gruLossAndGrad,
  initGru,
  initLstm,
  lstmLabExhaustive,
  lstmLossAndGrad,
  runGatedRace,
  simulateLstmCorridor,
} from './gated';

describe('Module 11 Lesson 2: LSTMs and GRUs', () => {
  it('verifies exact LSTM and GRU BPTT gradients against central finite differences (< 1e-8)', () => {
    const xs = [0.8, -0.5, 0.3, -0.9, 0.4];
    const eps = 1e-6;

    // 1. LSTM
    const lp = initLstm(3, 1.5, 17);
    const lr = lstmLossAndGrad(lp, xs, 1);
    let maxLstmErr = 0;
    const checkArr = (fn: () => number, arr: number[], grad: number[]) => {
      for (let i = 0; i < arr.length; i++) {
        const orig = arr[i];
        arr[i] = orig + eps;
        const p1 = fn();
        arr[i] = orig - eps;
        const m1 = fn();
        arr[i] = orig;
        maxLstmErr = Math.max(maxLstmErr, Math.abs((p1 - m1) / (2 * eps) - grad[i]));
      }
    };
    const lLoss = () => lstmLossAndGrad(lp, xs, 1).loss;
    checkArr(lLoss, lp.W, lr.dW);
    checkArr(lLoss, lp.b, lr.db);
    checkArr(lLoss, lp.Wy, lr.dWy);
    {
      const orig = lp.by;
      lp.by = orig + eps;
      const p1 = lLoss();
      lp.by = orig - eps;
      const m1 = lLoss();
      lp.by = orig;
      maxLstmErr = Math.max(maxLstmErr, Math.abs((p1 - m1) / (2 * eps) - lr.dby));
    }
    expect(maxLstmErr).toBeLessThan(1e-8);

    // 2. GRU
    const gp = initGru(3, -1.5, 29);
    const gr = gruLossAndGrad(gp, xs, 1);
    let maxGruErr = 0;
    const gLoss = () => gruLossAndGrad(gp, xs, 1).loss;
    const checkGru = (arr: number[], grad: number[]) => {
      for (let i = 0; i < arr.length; i++) {
        const orig = arr[i];
        arr[i] = orig + eps;
        const p1 = gLoss();
        arr[i] = orig - eps;
        const m1 = gLoss();
        arr[i] = orig;
        maxGruErr = Math.max(maxGruErr, Math.abs((p1 - m1) / (2 * eps) - grad[i]));
      }
    };
    checkGru(gp.Wg, gr.dWg);
    checkGru(gp.bg, gr.dbg);
    checkGru(gp.Whx, gr.dWhx);
    checkGru(gp.Whh, gr.dWhh);
    checkGru(gp.bh, gr.dbh);
    checkGru(gp.Wy, gr.dWy);
    {
      const orig = gp.by;
      gp.by = orig + eps;
      const p1 = gLoss();
      gp.by = orig - eps;
      const m1 = gLoss();
      gp.by = orig;
      maxGruErr = Math.max(maxGruErr, Math.abs((p1 - m1) / (2 * eps) - gr.dby));
    }
    expect(maxGruErr).toBeLessThan(1e-8);
  });

  it('pins hostile distractor corridor retention and output shielding', () => {
    const hold = simulateLstmCorridor(0.95, 0.05, 0.05);
    expect(hold.retentionRatio).toBeCloseTo(0.6225, 3);
    expect(hold.maxCorridorH).toBeLessThan(0.05);
    expect(hold.readH).toBeCloseTo(0.485, 3);

    const openInput = simulateLstmCorridor(0.95, 0.50, 0.05);
    expect(openInput.retentionRatio).toBeLessThan(-0.7);
  });

  it('pins Constant Error Carousel forget-gate bias boost across 25 steps', () => {
    const b0 = carouselGradientFlow(25, 0, 0.8);
    const b2 = carouselGradientFlow(25, 2, 0.8);
    const b3 = carouselGradientFlow(25, 3, 0.8);

    expect(b0.lstm.step1Ratio).toBeCloseTo(0.5 ** 24, 10);
    expect(b2.lstm.step1Ratio).toBeCloseTo(0.04754, 4);
    expect(b3.lstm.step1Ratio).toBeCloseTo(0.3116, 3);
    expect(b3.lstm.step1Ratio / b0.lstm.step1Ratio).toBeGreaterThan(5e6);
  });

  it('pins exact single-layer parameter counts for RNN (1x), GRU (3x), LSTM (4x)', () => {
    expect(cellParamCount('rnn', 16, 16)).toBe(528);
    expect(cellParamCount('gru', 16, 16)).toBe(1584);
    expect(cellParamCount('lstm', 16, 16)).toBe(2112);
  });

  it('pins fair 3-seed race: Vanilla RNN collapses at lag 14 while GRU & LSTM hold 100%', () => {
    const lag3 = runGatedRace(3);
    const lag8 = runGatedRace(8);
    const lag14 = runGatedRace(14);

    expect(lag3.rnnAcc).toBeCloseTo(1.0, 3);
    expect(lag3.gruAcc).toBeCloseTo(1.0, 3);
    expect(lag3.lstmAcc).toBeCloseTo(1.0, 3);

    expect(lag8.rnnAcc).toBeCloseTo(0.796, 2);
    expect(lag8.gruAcc).toBeCloseTo(1.0, 3);
    expect(lag8.lstmAcc).toBeCloseTo(1.0, 3);

    expect(lag14.rnnAcc).toBeLessThan(0.55);
    expect(lag14.gruAcc).toBeCloseTo(1.0, 3);
    expect(lag14.lstmAcc).toBeCloseTo(1.0, 3);
  });

  it('verifies exhaustive Lab: Wire the Gated Cell counts and unique optimums', () => {
    const b1 = lstmLabExhaustive('gates');
    expect(b1.total).toBe(27);
    expect(b1.passingCount).toBe(1);
    expect(evalLstmBrief1(0.95, 0.05, 0.05).pass).toBe(true);
    expect(evalLstmBrief1(0.95, 0.50, 0.05).pass).toBe(false);

    const b2 = lstmLabExhaustive('bias');
    expect(b2.total).toBe(10);
    expect(b2.passingCount).toBe(2);
    expect(evalLstmBrief2('rnn', 3).pass).toBe(false);
    expect(evalLstmBrief2('lstm', 0).pass).toBe(false);
    expect(evalLstmBrief2('lstm', 2).pass).toBe(true);
    expect(evalLstmBrief2('lstm', 3).pass).toBe(true);

    const b3 = lstmLabExhaustive('budget');
    expect(b3.total).toBe(12);
    expect(b3.passingCount).toBe(3);
    expect(evalLstmBrief3('rnn', 20).pass).toBe(false);
    expect(evalLstmBrief3('lstm', 16).pass).toBe(false);
    expect(evalLstmBrief3('gru', 16).pass).toBe(true);
    expect(evalLstmBrief3('gru', 16).params).toBe(1584);
  });
});
