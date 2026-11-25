// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { MLP } from '../../lib/mlp';
import { createRng } from '../../lib/rng';
import {
  GAN_PRESETS,
  GanRunner,
  backpropGenerator,
  evaluateLabSetting,
  sampleNoise,
} from './gan';
import { LAB_GRID_RESULTS, passesBrief1, passesBrief2, passesBrief3 } from './results';

describe('GAN analytical gradients through D(G(z))', () => {
  it('matches central finite differences to < 1e-8 for both non-saturating and saturating objectives', () => {
    for (const objective of ['non-saturating', 'saturating'] as const) {
      const G = new MLP([2, 5, 1], ['tanh', 'linear'], createRng(101), 'xavier');
      const D = new MLP([1, 5, 1], ['tanh', 'sigmoid'], createRng(202), 'xavier');
      const Z = sampleNoise(6, 2, createRng(303));

      backpropGenerator(G, D, Z, objective);
      const analytic = G.grads().map((arr) => arr.slice());
      const params = G.params();
      const eps = 1e-6;

      let maxErr = 0;
      for (let k = 0; k < params.length; k++) {
        for (let i = 0; i < params[k].length; i++) {
          const orig = params[k][i];
          params[k][i] = orig + eps;
          const lp = backpropGenerator(G, D, Z, objective).gLoss;
          params[k][i] = orig - eps;
          const lm = backpropGenerator(G, D, Z, objective).gLoss;
          params[k][i] = orig;
          const num = (lp - lm) / (2 * eps);
          maxErr = Math.max(maxErr, Math.abs(num - analytic[k][i]));
        }
      }
      expect(maxErr).toBeLessThan(1e-8);
    }
  });
});

describe('GAN presets: determinism, vanishing gradient and mode collapse', () => {
  it('is strictly deterministic with a fixed seed', () => {
    const a = new GanRunner(GAN_PRESETS.balanced.config);
    const b = new GanRunner(GAN_PRESETS.balanced.config);
    a.step(60);
    b.step(60);
    const sa = a.snapshot();
    const sb = b.snapshot();
    expect(sa.metrics.w1).toBe(sb.metrics.w1);
    expect(sa.metrics.modesCovered).toBe(sb.metrics.modesCovered);
    expect(sa.fakeHist).toEqual(sb.fakeHist);
  });

  it('vanishing preset: overtrained D drives D(fake) near 0 and shrinks saturating gradient > 400x vs non-saturating', () => {
    const r = new GanRunner(GAN_PRESETS.vanishing.config);
    const s0 = r.snapshot().metrics;
    const Z = sampleNoise(36, GAN_PRESETS.vanishing.config.noiseDim, createRng(GAN_PRESETS.vanishing.config.seed * 4099 + 1));
    const ns = backpropGenerator(r.G, r.D, Z, 'non-saturating');
    expect(s0.meanDFake).toBeLessThan(0.01);
    expect(s0.gGradNorm).toBeCloseTo(0.0122, 3);
    expect(ns.gGradNorm).toBeCloseTo(5.483, 2);
    expect(ns.gGradNorm / s0.gGradNorm).toBeGreaterThan(400);
  });

  it('balanced covers all 3 modes while collapse covers strictly fewer (1/3)', () => {
    const bal = new GanRunner(GAN_PRESETS.balanced.config);
    bal.step(240);
    const mBal = bal.snapshot().metrics;

    const col = new GanRunner(GAN_PRESETS.collapse.config);
    col.step(240);
    const mCol = col.snapshot().metrics;

    expect(mBal.modesCovered).toBe(3);
    expect(mBal.w1).toBeCloseTo(0.307, 2);
    expect(mCol.modesCovered).toBe(1);
    expect(mCol.modesCovered).toBeLessThan(mBal.modesCovered);
    expect(mCol.w1).toBeCloseTo(1.462, 2);
  });
});

describe('Lab: Keep the Duel Balanced recorded 5-seed grid verification', () => {
  it('recorded table has exact counts for Briefs 1, 2 and 3', () => {
    expect(LAB_GRID_RESULTS.length).toBe(30);
    expect(LAB_GRID_RESULTS.filter(passesBrief1).length).toBe(19);
    expect(LAB_GRID_RESULTS.filter(passesBrief2).length).toBe(6);
    const b3 = LAB_GRID_RESULTS.filter(passesBrief3);
    expect(b3.length).toBe(1);
    expect(b3[0]).toMatchObject({ noiseDim: 2, dSteps: 1, lrRatio: 0.5, updatesToCover3: 40 });
  });

  it('spot-checks Brief 3 fastest winner live across all 5 seeds', () => {
    const fastest = evaluateLabSetting({ noiseDim: 2, dSteps: 1, lrRatio: 0.5 });
    expect(fastest.meanModes).toBe(3);
    expect(fastest.allThreeSeeds).toBe(5);
    expect(fastest.meanW1).toBeCloseTo(0.396, 2);
    expect(fastest.updatesToCover3).toBe(40);
  });

  it('spot-checks lowest-W1 balanced setting live across all 5 seeds', () => {
    const bestW1 = evaluateLabSetting({ noiseDim: 2, dSteps: 2, lrRatio: 0.25 });
    expect(bestW1.meanModes).toBe(3);
    expect(bestW1.meanW1).toBeCloseTo(0.272, 2);
    expect(bestW1.updatesToCover3).toBe(120);
  });

  it('spot-checks failing collapse setting live across all 5 seeds', () => {
    const collapsed = evaluateLabSetting({ noiseDim: 1, dSteps: 1, lrRatio: 4 });
    expect(collapsed.meanModes).toBeCloseTo(0.4, 2);
    expect(collapsed.allThreeSeeds).toBe(0);
    expect(collapsed.updatesToCover3).toBeNull();
  });
});
