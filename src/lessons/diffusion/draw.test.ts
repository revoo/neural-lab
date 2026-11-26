// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { CpuMlpEngine } from '../../lib/compute/cpuBackend';
import { paramCount } from '../../lib/compute/types';
import { nodeFs } from '../../lib/nodeIo';
import { decodeDraw, DRAW_RUN, drawMlp, drawMoments, nearestTrain, realRange, sampleDraw, trainDraw } from './draw';
import { DRAW_RESULTS } from './drawResults';

async function data() {
  const fs = await nodeFs();
  return decodeDraw(new Uint8Array(fs.readFileSync(new URL('./fashion8.bin', import.meta.url))));
}

describe('learning to draw', () => {
  it('1,500 training and 150 held-out 8 × 8 pictures; the real range is 2.28; the denoiser has 102,976 weights', async () => {
    const d = await data();
    expect([d.train.length, d.held.length]).toEqual([1500, 150]);
    expect(realRange(d).toFixed(2)).toBe('2.28');
    expect(DRAW_RESULTS.bar).toBe(2.28);
    expect(paramCount(drawMlp(DRAW_RUN).sizes)).toBe(102976);
  });
  it('moments: first picture in the real range at step 631, half at 3,981; 10 of 16 at the end, none a copy', () => {
    expect(DRAW_RESULTS.moments.map((m) => [m.id, m.step])).toEqual([
      ['first', 631],
      ['half', 3981],
    ]);
    expect(drawMoments(DRAW_RESULTS.checkpoints, DRAW_RESULTS.bar)).toEqual(DRAW_RESULTS.moments);
    const last = DRAW_RESULTS.checkpoints.at(-1)!;
    expect(last.step).toBe(10000);
    expect(last.nn.filter((v) => v <= DRAW_RESULTS.bar).length).toBe(10);
    expect(Math.min(...last.nn)).toBe(1.25);
    expect(DRAW_RESULTS.checkpoints[0].nn.every((v) => v > DRAW_RESULTS.bar)).toBe(true);
  });
  it('re-trains exactly: the recorded checkpoints to step 100', async () => {
    const d = await data();
    const eng = new CpuMlpEngine(drawMlp(DRAW_RUN));
    let i = 0;
    await trainDraw(eng, d, DRAW_RUN, async (step) => {
      const rec = DRAW_RESULTS.checkpoints[i++];
      expect(step).toBe(rec.step);
      const xs = await sampleDraw(eng, 16, 7);
      expect(xs.map((x) => +nearestTrain(x, d.train).toFixed(2))).toEqual(rec.nn);
      return step < 100;
    });
    expect(DRAW_RESULTS.checkpoints[i - 1].step).toBe(100);
  }, 120_000);
});
