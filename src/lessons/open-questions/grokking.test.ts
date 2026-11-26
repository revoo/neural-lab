// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { CpuMlpEngine } from '../../lib/compute/cpuBackend';
import { accuracy, firstReach, GROK, grokData, grokSplit, grokSpec, runGrokking, type GrokCheckpoint } from './grokking';
import { GROK_RESULTS } from './grokResults';

describe('V2-WP3 grokking demo', () => {
  it('the data: 289 sums, split 145 / 144, every answer one-hot', () => {
    const d = grokData();
    expect(d.n).toBe(289);
    const { train, val } = grokSplit(1);
    expect([train.length, val.length]).toEqual([145, 144]);
    expect(new Set([...train, ...val]).size).toBe(289);
    // a perfect predictor scores 100%: feed the targets back in as outputs
    expect(accuracy(d.Y, val)).toBe(1);
  });

  it('seed 1 memorises by step 100 and groks at step 1,400 (as recorded)', async () => {
    const hist: GrokCheckpoint[] = [];
    await runGrokking(new CpuMlpEngine(grokSpec(GROK.weightDecay, 1)), [1], (c) => (hist.push(c), c.step < 1400), 1400);
    expect(firstReach(hist, 'trainAcc', 1, 1)).toEqual([GROK_RESULTS.wd2.memorised[0]]);
    expect(firstReach(hist, 'valAcc', 0.99, 1)).toEqual([GROK_RESULTS.wd2.grokked[0]]);
    // at the moment it has memorised every training sum, it gets almost no unseen sum right
    const at100 = hist.find((h) => h.step === 100)!;
    expect(at100.trainAcc[0]).toBe(1);
    expect(at100.valAcc[0]).toBeLessThan(0.05);
    // the climb is gradual on a linear axis: past 50% by step 600, at 99% only at step 1,400
    expect(hist.find((h) => h.step === 600)!.valAcc[0]).toBeGreaterThan(0.5);
  }, 60_000);

  it('without weight decay, seed 1 has memorised but not generalised at the same point', async () => {
    let last: GrokCheckpoint | null = null;
    await runGrokking(new CpuMlpEngine(grokSpec(0, 1)), [1], (c) => ((last = c), true), 1400);
    expect(last!.trainAcc[0]).toBe(1);
    expect(last!.valAcc[0]).toBeLessThan(0.1);
  }, 60_000);

  it('recorded summary: all four seeds grok with decay, none without', () => {
    expect(GROK_RESULTS.wd2.grokked.every((s) => s !== null && s > 1000 && s <= 3200)).toBe(true);
    expect(GROK_RESULTS.wd0.grokked.every((s) => s === null)).toBe(true);
    expect(Math.max(...GROK_RESULTS.wd0.bestVal)).toBeLessThan(1 / 17);
  });
});
