// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { evaluateLabSetting, LAB_SEEDS, type LabSetting } from './gan';
import { cpuFakes, gridJobs, labResultFromFakes, packJobs } from './ganGpu';

describe('V2-WP3 GAN lab on WebGPU: the CPU side', () => {
  it('scoring G outputs reproduces evaluateLabSetting exactly', () => {
    for (const setting of [
      { lrRatio: 0.5, dSteps: 1, noiseDim: 2 },
      { lrRatio: 4, dSteps: 2, noiseDim: 1 },
    ] as LabSetting[]) {
      const fakes = LAB_SEEDS.map((seed) => cpuFakes({ setting, seed }));
      const a = labResultFromFakes(setting, fakes);
      const b = evaluateLabSetting(setting);
      expect(a.meanModes).toBe(b.meanModes);
      expect(a.meanW1).toBeCloseTo(b.meanW1, 12);
      expect(a.meanOffMode).toBeCloseTo(b.meanOffMode, 12);
      expect(a.updatesToCover3).toBe(b.updatesToCover3);
      expect(a.finalUpdates).toBe(b.finalUpdates);
      expect(a.perSeed.map((p) => p.modes)).toEqual(b.perSeed.map((p) => p.modes));
    }
  }, 30_000);

  it('packs the start weights and the RNG state of each run', () => {
    const jobs = gridJobs([{ lrRatio: 1, dSteps: 4, noiseDim: 1 }]);
    expect(jobs).toHaveLength(5);
    const p = packJobs(jobs);
    expect(p.cfg[0]).toBe(1 * 3037 + 53);
    expect([p.cfg[1], p.cfg[2]]).toEqual([1, 4]);
    // with one noise input, the padded second column of G's first layer stays zero
    for (let j = 0; j < 14; j++) expect(p.state[j * 2 + 1]).toBe(0);
  });
});
