// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAYER_CATALOG,
  cnnLabExhaustive,
  cnnLossAndGrad,
  evalCnnBrief1,
  evalCnnBrief2,
  evalCnnBrief3,
  getShapeDataset,
  getShiftRace,
  initTinyCnn,
  pool2x2,
  traceReceptiveField,
  vggSavings,
} from './cnn';

describe('cnns: receptive field, VGG savings, and pooling', () => {
  it('computes exact receptive field & jump for a 6-layer Conv/Pool stack', () => {
    const conv3 = LAYER_CATALOG[0];
    const pool2 = LAYER_CATALOG[3];
    const trace = traceReceptiveField(32, [conv3, conv3, pool2, conv3, pool2, conv3], 16);
    expect(trace.map((r) => [r.rf, r.jump])).toEqual([
      [3, 1],
      [5, 1],
      [6, 2],
      [10, 2],
      [12, 4],
      [20, 4],
    ]);
  });

  it('pins VGG parameter reductions for stacked 3x3 vs 5x5 and 7x7', () => {
    expect(vggSavings(2, 64).reduction).toBeCloseTo(0.28, 4);
    expect(vggSavings(3, 64).reduction).toBeCloseTo(1 - 27 / 49, 4); // 44.90%
  });

  it('verifies 2x2 max pooling absorbs 1-pixel jitter while avg pooling dilutes it', () => {
    const g1 = [
      [1, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ];
    const g2 = [
      [0, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ];
    expect(pool2x2(g1, 'max')[0][0]).toBe(1);
    expect(pool2x2(g2, 'max')[0][0]).toBe(1);
    expect(pool2x2(g1, 'avg')[0][0]).toBe(0.25);
  });
});

describe('cnns: tiny 2D CNN backprop and unseen-shift generalisation', () => {
  it('matches central finite differences on every conv & head parameter to < 1e-8', () => {
    const p = initTinyCnn(3, 7);
    const batch = getShapeDataset().train.slice(0, 8);
    const base = cnnLossAndGrad(p, batch, 1e-3);
    const eps = 1e-5;
    let maxErr = 0;

    const check = (arr: number[], grad: number[]) => {
      for (let i = 0; i < arr.length; i++) {
        const old = arr[i];
        arr[i] = old + eps;
        const lp = cnnLossAndGrad(p, batch, 1e-3).loss;
        arr[i] = old - eps;
        const lm = cnnLossAndGrad(p, batch, 1e-3).loss;
        arr[i] = old;
        maxErr = Math.max(maxErr, Math.abs((lp - lm) / (2 * eps) - grad[i]));
      }
    };
    check(p.convW, base.gConvW);
    check(p.convB, base.gConvB);
    check(p.headW, base.gHeadW);
    check(p.headB, base.gHeadB);
    expect(maxErr).toBeLessThan(1e-8);
  });

  it('proves 55-param CNN hits 100% on unseen bottom-right shifts while 547-param Dense collapses to 28.3%', () => {
    const { cnn, dense } = getShiftRace();
    expect(cnn.paramCount).toBe(55);
    expect(cnn.trainAcc).toBeCloseTo(1.0, 4);
    expect(cnn.inDistAcc).toBeCloseTo(1.0, 4);
    expect(cnn.shiftedAcc).toBeCloseTo(1.0, 4);

    expect(dense.paramCount).toBe(547);
    expect(dense.trainAcc).toBeCloseTo(1.0, 4);
    expect(dense.inDistAcc).toBeCloseTo(1.0, 4);
    expect(dense.shiftedAcc).toBeCloseTo(17 / 60, 3); // 28.3%
  });
});

describe('cnns: Lab (Architect the CNN) exhaustive checks', () => {
  it('Brief 1 (RF >= 15 under 10k params): 3/18 pass, fewest 6,960 params (3x Conv 3x3 + 2 pools)', () => {
    const opt = evalCnnBrief1({ k: 3, nConv: 3, nPools: 2 });
    expect(opt.rf).toBe(18);
    expect(opt.params).toBe(6960);
    expect(opt.pass).toBe(true);

    const ex = cnnLabExhaustive('rf-budget');
    expect(ex.total).toBe(18);
    expect(ex.passing).toBe(3);
    expect(ex.summary).toContain('6,960 params');
  });

  it('Brief 2 (unseen shift generalisation): 0/3 Dense pass, 2/7 pass, fewest 55 params (CNN 4 filters)', () => {
    expect(evalCnnBrief2('dense-8').pass).toBe(false);
    expect(evalCnnBrief2('dense-16').pass).toBe(false);
    expect(evalCnnBrief2('cnn-4').pass).toBe(true);
    expect(evalCnnBrief2('cnn-4').params).toBe(55);

    const ex = cnnLabExhaustive('shift-gen');
    expect(ex.total).toBe(7);
    expect(ex.passing).toBe(2);
    expect(ex.summary).toContain('55 params');
  });

  it('Brief 3 (ImageNet head <= 3M params, keep 512 ch): unique GAP -> 1000 (0.513M vs VGG 106.9M)', () => {
    expect(evalCnnBrief3('vgg-fc').params).toBe(106_861_544);
    expect(evalCnnBrief3('gap-linear').params).toBe(513_000);
    expect(evalCnnBrief3('gap-linear').pass).toBe(true);

    const ex = cnnLabExhaustive('gap-head');
    expect(ex.total).toBe(4);
    expect(ex.passing).toBe(1);
  });
});
