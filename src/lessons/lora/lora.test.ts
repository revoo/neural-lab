// authored-by: Neon High
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import {
  LORA_MODELS,
  computeSvd,
  estimateFineTuneMemory,
  loraStepGrads,
  makeWorkshopMatrix,
  reconstructRankR,
  runFullFtReference,
  runLoraExperiment,
  solveLoraLab,
} from './lora';

describe('LoRA & Low-Rank Fine-Tuning — engine & numbers quoted in prose', () => {
  it('SVD workshop: structured task shift captures 95.2% energy at r=2 vs 44.8% for random noise', () => {
    const svdTask = computeSvd(makeWorkshopMatrix('task-shift'));
    const r2Task = reconstructRankR(svdTask, 2);
    const r3Task = reconstructRankR(svdTask, 3);
    expect(r2Task.energyPct).toBeCloseTo(95.2, 1);
    expect(r3Task.energyPct).toBeCloseTo(99.1, 1);
    expect(r2Task.loraParams).toBe(48);
    expect(r2Task.fullParams).toBe(144);

    const svdNoise = computeSvd(makeWorkshopMatrix('random-noise'));
    const r2Noise = reconstructRankR(svdNoise, 2);
    const r3Noise = reconstructRankR(svdNoise, 3);
    expect(r2Noise.energyPct).toBeCloseTo(44.8, 1);
    expect(r3Noise.energyPct).toBeCloseTo(62.0, 1);
  });

  it('exact Llama-3 8B & 70B Full FT vs LoRA vs QLoRA memory footprints', () => {
    const [m8, m70] = LORA_MODELS;
    const f8 = estimateFineTuneMemory(m8, 'full', 16, 'all');
    const l8 = estimateFineTuneMemory(m8, 'lora', 16, 'all');
    const q8 = estimateFineTuneMemory(m8, 'qlora', 16, 'all');
    expect(f8.totalGiB).toBeCloseTo(119.66, 1);
    expect(l8.trainableParams / 1e6).toBeCloseTo(41.94, 1);
    expect(l8.trainablePct).toBeCloseTo(0.52, 2);
    expect(l8.totalGiB).toBeCloseTo(15.58, 1);
    expect(l8.checkpointMB).toBeCloseTo(80.0, 0);
    expect(q8.totalGiB).toBeCloseTo(4.83, 1);

    const f70 = estimateFineTuneMemory(m70, 'full', 16, 'all');
    const q70 = estimateFineTuneMemory(m70, 'qlora', 16, 'all');
    expect(f70.totalGiB).toBeCloseTo(1052.0, 0);
    expect(q70.trainableParams / 1e6).toBeCloseTo(207.1, 1);
    expect(q70.totalGiB).toBeCloseTo(40.07, 1);
  });

  it('exact analytical gradients for LoRA A and B match central finite differences (< 1e-8)', () => {
    const rng = createRng(55);
    const d = 4;
    const r = 2;
    const W0 = Array.from({ length: d }, () => Array.from({ length: d }, () => rng.normal(0, 0.5)));
    const A = Array.from({ length: r }, () => Array.from({ length: d }, () => rng.normal(0, 0.5)));
    const B = Array.from({ length: d }, () => Array.from({ length: r }, () => rng.normal(0, 0.5)));
    const X = Array.from({ length: 6 }, () => Array.from({ length: d }, () => rng.normal(0, 1)));
    const Y = Array.from({ length: 6 }, () => Array.from({ length: d }, () => rng.normal(0, 1)));
    const { gA, gB } = loraStepGrads(W0, A, B, 1.5, X, Y);
    const eps = 1e-6;

    for (let k = 0; k < r; k++) {
      for (let j = 0; j < d; j++) {
        const Ap = A.map((row) => row.slice());
        const Am = A.map((row) => row.slice());
        Ap[k][j] += eps;
        Am[k][j] -= eps;
        const num = (loraStepGrads(W0, Ap, B, 1.5, X, Y).loss - loraStepGrads(W0, Am, B, 1.5, X, Y).loss) / (2 * eps);
        expect(Math.abs(gA[k][j] - num)).toBeLessThan(1e-8);
      }
    }
    for (let i = 0; i < d; i++) {
      for (let k = 0; k < r; k++) {
        const Bp = B.map((row) => row.slice());
        const Bm = B.map((row) => row.slice());
        Bp[i][k] += eps;
        Bm[i][k] -= eps;
        const num = (loraStepGrads(W0, A, Bp, 1.5, X, Y).loss - loraStepGrads(W0, A, Bm, 1.5, X, Y).loss) / (2 * eps);
        expect(Math.abs(gB[i][k] - num)).toBeLessThan(1e-8);
      }
    }
  });

  it('zero-init symmetry experiment: B=0/A~N preserves step 0 and learns; both-zero freezes forever; both-random corrupts step 0', () => {
    const ref = runFullFtReference(80);
    expect(ref.finalLoss).toBeLessThan(0.001);

    const lora = runLoraExperiment(3, 'lora-zero-b', 80);
    expect(lora.step0Loss).toBeCloseTo(lora.pretrainedLoss, 6);
    expect(lora.step0GradANorm).toBe(0);
    expect(lora.step0GradBNorm).toBeGreaterThan(0.15);
    expect(lora.finalLoss).toBeLessThan(0.025);

    const dead = runLoraExperiment(3, 'both-zero', 80);
    expect(dead.step0GradANorm).toBe(0);
    expect(dead.step0GradBNorm).toBe(0);
    expect(dead.finalLoss).toBeCloseTo(dead.pretrainedLoss, 6);

    const noisy = runLoraExperiment(3, 'both-random', 80);
    expect(noisy.step0Loss).toBeGreaterThan(noisy.pretrainedLoss + 0.25);
  });

  it('Adapter Allocation Lab exhaustive search over all 125 rank combinations', () => {
    const sol = solveLoraLab();
    expect(sol.totalConfigs).toBe(125);
    expect(sol.passingCount).toBe(10);
    expect(sol.uniformR1.passes).toBe(false);
    expect(sol.uniformR2.passes).toBe(false);
    expect(sol.best.ranks).toEqual([0, 3, 2]);
    expect(sol.best.totalParams).toBe(80);
    expect(sol.best.avgMse).toBeCloseTo(0.0009, 4);
  });
});
