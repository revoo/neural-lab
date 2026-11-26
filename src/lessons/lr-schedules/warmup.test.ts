// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { CpuMlpEngine } from '../../lib/compute/cpuBackend';
import { paramCount } from '../../lib/compute/types';
import { lrAt, runWarmup, spiral, summarise, WARM, warmSpec } from './warmup';
import { WARM_RECHECK, WARM_RESULTS } from './warmupResults';

const rec = (peak: number) => WARM_RESULTS.find((r) => r.peak === peak)!;
const arm = (r: { earlyMax: number[][]; finalAcc: number[][] }, a: 0 | 1) => summarise(r.earlyMax[a], r.finalAcc[a]);

describe('V2-WP3 live warmup experiment', () => {
  it('data and schedule', () => {
    const d = spiral();
    expect(d.n).toBe(400);
    expect(d.Y.reduce((s, v) => s + v, 0)).toBe(200);
    expect(lrAt(0, 0.03, 300)).toBeCloseTo(0.0001);
    expect(lrAt(299, 0.03, 300)).toBeCloseTo(0.03);
    expect(lrAt(0, 0.03, 0)).toBe(0.03);
  });

  it('seed 1 at peak 0.03 reproduces its recorded run (both arms)', async () => {
    const run = await runWarmup(new CpuMlpEngine(warmSpec(2)), [1], 0.03);
    const r = rec(0.03);
    for (const a of [0, 1]) {
      expect(run.earlyMax[0][a]).toBeCloseTo(r.earlyMax[a][0], 2);
      expect(run.history.at(-1)!.acc[0][a]).toBeCloseTo(r.finalAcc[a][0], 2);
    }
  }, 60_000);

  it('the numbers quoted in the lesson', () => {
    // 0.01: everything solves, warmup or not
    expect([arm(rec(0.01), 0).solved, arm(rec(0.01), 1).solved]).toEqual([20, 20]);
    // 0.03: warmup roughly halves the early spike, but exactly as many runs die
    const n3 = arm(rec(0.03), 0);
    const w3 = arm(rec(0.03), 1);
    expect([n3.medianEarly, w3.medianEarly]).toEqual([4.13, 2.23]);
    expect([n3.worstEarly, w3.worstEarly]).toEqual([9.51, 6.11]);
    expect([n3.died, w3.died]).toEqual([8, 8]);
    expect([n3.solved, w3.solved]).toEqual([2, 3]);
    // 0.02: seeds 1–20 suggest a win for warmup ...
    expect([arm(rec(0.02), 0).solved, arm(rec(0.02), 1).solved]).toEqual([9, 16]);
    // ... which disappears on 40 fresh seeds
    expect(WARM_RECHECK.finalAcc[0].length).toBe(40);
    expect([arm(WARM_RECHECK, 0).solved, arm(WARM_RECHECK, 1).solved]).toEqual([24, 22]);
    expect(WARM.steps).toBe(1000);
    expect(paramCount(warmSpec(1).sizes)).toBe(29377); // "a 29,000-weight network"
  });
});
