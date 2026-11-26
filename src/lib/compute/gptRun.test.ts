// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import text from '../corpora/grimm.txt?raw';
import { prepareCorpus } from '../corpora/grimm';
import { checkpointSteps, lrAt, runTraining, type Checkpoint, type RunSpec } from './gptRun';
import { InlineCpuTrainer } from './gptTrainerCore';
import { paramHash } from './gptPinned';

const corpus = prepareCorpus(text);
const tiny: RunSpec = {
  cfg: { vocab: 68, ctx: 16, d: 16, heads: 2, layers: 1 },
  T: 16, batch: 4, steps: 40, lr: 3e-3, warmup: 5, minLrFrac: 0.1,
  opt: { weightDecay: 0.1, clip: 1 }, initSeed: 1, dataSeed: 2, evalRows: 4,
};

describe('run schedule', () => {
  it('warms up linearly, then follows a cosine to minLrFrac', () => {
    expect(lrAt(tiny, 0)).toBeCloseTo(3e-3 / 5, 9);
    expect(lrAt(tiny, 4)).toBeCloseTo(3e-3, 9);
    expect(lrAt(tiny, 5)).toBeCloseTo(3e-3, 9);
    expect(lrAt(tiny, 40)).toBeCloseTo(3e-4, 9);
    expect(lrAt(tiny, 22)).toBeLessThan(lrAt(tiny, 10));
  });
  it('checkpoints are about log-spaced and include both ends', () => {
    const c = checkpointSteps(2000);
    expect(c[0]).toBe(0);
    expect(c.at(-1)).toBe(2000);
    expect(c.slice(0, 8)).toEqual([0, 1, 2, 3, 4, 5, 6, 8]);
    for (let i = 2; i < c.length - 1; i++) if (c[i] >= 10) expect(c[i + 1] / c[i]).toBeLessThan(1.4);
  });
});

describe('runTraining', () => {
  async function run(chunk: number) {
    const cps: Checkpoint[] = [];
    await runTraining(tiny, corpus, new InlineCpuTrainer(tiny.cfg), (c) => void cps.push(c), chunk);
    return cps;
  }
  it('gives the same run whatever the chunk size, and the loss falls', async () => {
    const a = await run(25);
    const b = await run(3);
    expect(a.map((c) => c.step)).toEqual(checkpointSteps(40));
    expect(b.map((c) => paramHash(c.params))).toEqual(a.map((c) => paramHash(c.params)));
    expect(a[0].valLoss).toBeGreaterThan(4);
    expect(a.at(-1)!.valLoss).toBeLessThan(a[0].valLoss - 0.5);
  });
});
