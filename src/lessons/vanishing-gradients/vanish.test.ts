// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import {
  DEEP_RACE_PRESETS,
  RESCUE_LEVELS,
  evaluateRescue,
  measureLayerVariance,
  probeSignal,
  runDeepTrainingRace,
  singleBackwardInputGrad,
  singleForwardLoss,
  solveRescueLevel,
} from './vanish';

describe('Vanishing & exploding gradients — engine & numbers quoted in prose', () => {
  it('exact analytical backward matches central finite differences across all 12 activation/LN/residual modes', () => {
    const rng = createRng(7);
    const d = 6;
    const depth = 4;
    const x0 = Array.from({ length: d }, () => rng.normal(0, 1));
    const uTop = Array.from({ length: d }, () => rng.normal(0, 1));
    const weights = Array.from({ length: depth }, () =>
      Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal(0, 0.5)))
    );
    const eps = 1e-6;

    for (const act of ['tanh', 'sigmoid', 'relu'] as const) {
      for (const useLN of [false, true]) {
        for (const useRes of [false, true]) {
          const ana = singleBackwardInputGrad(x0, uTop, weights, act, useLN, useRes);
          for (let i = 0; i < d; i++) {
            const xp = x0.slice();
            const xm = x0.slice();
            xp[i] += eps;
            xm[i] -= eps;
            const num =
              (singleForwardLoss(xp, uTop, weights, act, useLN, useRes) -
                singleForwardLoss(xm, uTop, weights, act, useLN, useRes)) /
              (2 * eps);
            expect(Math.abs(ana[i] - num)).toBeLessThan(1e-6);
          }
        }
      }
    }
  });

  it('16-layer signal telephone numbers quoted in prose', () => {
    const sigXav = probeSignal({ depth: 16, width: 32, act: 'sigmoid', init: 'xavier', useLayerNorm: false, useResidual: false });
    expect(sigXav.gradRatio).toBeLessThan(1e-9);
    expect(sigXav.gradRatio).toBeGreaterThan(1e-12);

    const reluXav = probeSignal({ depth: 16, width: 32, act: 'relu', init: 'xavier', useLayerNorm: false, useResidual: false });
    expect(reluXav.actRms[16]).toBeCloseTo(0.00311, 4);
    expect(reluXav.gradRatio).toBeCloseTo(0.00321, 4);

    const reluHe = probeSignal({ depth: 16, width: 32, act: 'relu', init: 'he', useLayerNorm: false, useResidual: false });
    expect(reluHe.actRms[16]).toBeCloseTo(0.796, 2);
    expect(reluHe.gradRatio).toBeCloseTo(0.822, 2);

    const reluNaive = probeSignal({ depth: 16, width: 32, act: 'relu', init: 'naive', useLayerNorm: false, useResidual: false });
    expect(reluNaive.actRms[16]).toBeGreaterThan(1e9);
    expect(reluNaive.gradRatio).toBeGreaterThan(1e9);

    const sigResLn = probeSignal({ depth: 16, width: 32, act: 'sigmoid', init: 'xavier', useLayerNorm: true, useResidual: true });
    expect(sigResLn.gradRatio).toBeCloseTo(1.02, 2);
  });

  it('single-layer variance conservation matches theory for linear vs ReLU', () => {
    const vLin = measureLayerVariance(64, 1 / Math.sqrt(64), 'linear');
    expect(vLin.theoryVar).toBeCloseTo(1.0, 5);
    expect(vLin.empiricalVar).toBeGreaterThan(0.85);
    expect(vLin.empiricalVar).toBeLessThan(1.15);

    const vReluXav = measureLayerVariance(64, 1 / Math.sqrt(64), 'relu');
    expect(vReluXav.theoryVar).toBeCloseTo(0.5, 5);

    const vReluHe = measureLayerVariance(64, Math.sqrt(2 / 64), 'relu');
    expect(vReluHe.theoryVar).toBeCloseTo(1.0, 5);
    expect(vReluHe.empiricalVar / vReluXav.empiricalVar).toBeCloseTo(2.0, 5);
  });

  it('8-layer training race: Sigmoid+Xavier freezes, ReLU+Naive explodes, ReLU+He and +Residual solve cleanly', () => {
    const rSig = runDeepTrainingRace(DEEP_RACE_PRESETS[0], 150, 11);
    expect(rSig.finalLoss).toBeCloseTo(0.6931, 3);
    expect(rSig.layerWeightMovement[0]).toBeLessThan(0.001);
    expect(rSig.initLayerGradNorms[0] / rSig.initLayerGradNorms[7]).toBeLessThan(1e-4);

    const rNaive = runDeepTrainingRace(DEEP_RACE_PRESETS[1], 150, 11);
    expect(rNaive.finalAcc).toBeLessThanOrEqual(0.55);

    const rHe = runDeepTrainingRace(DEEP_RACE_PRESETS[2], 150, 11);
    expect(rHe.finalAcc).toBe(1.0);
    expect(rHe.finalLoss).toBeLessThan(0.01);

    const rRes = runDeepTrainingRace(DEEP_RACE_PRESETS[3], 150, 11);
    expect(rRes.finalAcc).toBe(1.0);
    expect(rRes.layerWeightMovement[0]).toBeGreaterThan(0.08);
  });

  it('Rescue Deep Stack lab exhaustive solution counts across all 3 levels', () => {
    const s1 = solveRescueLevel(RESCUE_LEVELS[0]);
    expect(evaluateRescue(RESCUE_LEVELS[0], RESCUE_LEVELS[0].defaultRecipe).passes).toBe(false);
    expect(s1.totalRecipes).toBe(15);
    expect(s1.passingCount).toBe(2);
    expect(s1.bestRecipe).toEqual({ act: 'relu', init: 'he', useLayerNorm: false, useResidual: false });
    expect(s1.bestEval.trace.gradRatio).toBeCloseTo(0.925, 2);

    const s2 = solveRescueLevel(RESCUE_LEVELS[1]);
    expect(evaluateRescue(RESCUE_LEVELS[1], RESCUE_LEVELS[1].defaultRecipe).passes).toBe(false);
    expect(s2.totalRecipes).toBe(20);
    expect(s2.plainPassingCount).toBe(0); // Provably impossible without LN or Residual!
    expect(s2.passingCount).toBe(10);

    const s3 = solveRescueLevel(RESCUE_LEVELS[2]);
    expect(evaluateRescue(RESCUE_LEVELS[2], RESCUE_LEVELS[2].defaultRecipe).passes).toBe(false);
    expect(s3.totalRecipes).toBe(60);
    expect(s3.plainPassingCount).toBe(0); // Plain stacks cannot keep 32 layers balanced
    expect(s3.passingCount).toBe(12);
  });
});
