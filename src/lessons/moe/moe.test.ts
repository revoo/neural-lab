// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { batchLossGrad, DEFAULTS, initParams, paramCount, TEST, train, TRAIN, traceToken } from './moe';
import { allSettings, passesBrief1, passesBrief2, passesBrief3, scoreSetting } from './lab';
import { LAB_RESULTS } from './results';

describe('MoE layer: gradients', () => {
  it('matches central finite differences (top-1 and top-2, with aux loss, capacity and noise)', () => {
    for (const cfg of [
      { ...DEFAULTS, k: 1 as const, alpha: 0.3, capacity: 1, noise: 0.7 },
      { ...DEFAULTS, k: 2 as const, alpha: 0.1, capacity: Infinity, noise: 0 },
    ]) {
      const p = initParams(4, 3);
      const rows = Array.from({ length: 24 }, (_, i) => (i * 37) % TRAIN.n);
      const noise = rows.map((_, t) => [0, 1, 2, 3].map((e) => cfg.noise * Math.sin(t * 4 + e)));
      const g = new Float64Array(p.length);
      batchLossGrad(p, cfg, TRAIN, rows, noise, g);
      // the aux loss's f (dispatch shares) and the top-k choice are piecewise constant: keep h small
      const h = 1e-6;
      let worst = 0;
      for (let j = 0; j < p.length; j++) {
        const a = p.slice();
        const b = p.slice();
        a[j] += h;
        b[j] -= h;
        const fd = (batchLossGrad(a, cfg, TRAIN, rows, noise).loss - batchLossGrad(b, cfg, TRAIN, rows, noise).loss) / (2 * h);
        worst = Math.max(worst, Math.abs(fd - g[j]) / Math.max(1e-3, Math.abs(fd)));
      }
      expect(worst).toBeLessThan(1e-4);
    }
  });

  it('a token’s output is Σ p_i · e_i over its chosen experts', () => {
    const p = initParams(4, 1);
    const t = traceToken(p, 4, 2, 0.3, -1.2);
    expect(t.chosen).toHaveLength(2);
    expect(t.z).toBeCloseTo(t.probs[t.chosen[0]] * t.expertOut[t.chosen[0]] + t.probs[t.chosen[1]] * t.expertOut[t.chosen[1]], 12);
    expect(t.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(paramCount(4)).toBe(4 * 3 + 4 * 33);
  });
});

describe('MoE lesson numbers', () => {
  it('without balancing every seed starves an expert; with α = 0.1 none does', () => {
    const off = [1, 2, 3, 4, 5].map((s) => train({ ...DEFAULTS, alpha: 0 }, s));
    const on = [1, 2, 3, 4, 5].map((s) => train({ ...DEFAULTS, alpha: 0.1 }, s));
    expect(off.map((r) => r.load.filter((l) => l === 0).length)).toEqual([1, 2, 1, 1, 1]);
    expect(on.every((r) => Math.min(...r.load) > 0.2)).toBe(true);
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    expect(+mean(off.map((r) => r.acc)).toFixed(3)).toBe(0.953);
    expect(+mean(on.map((r) => r.acc)).toFixed(3)).toBe(0.963);
    expect(+Math.max(...off.map((r) => r.maxLoad)).toFixed(2)).toBe(0.57);
    // the idle expert is already idle at step 50
    const h = train({ ...DEFAULTS, alpha: 0 }, 1, 50).history;
    expect(h[0].load[0]).toBe(0);
    expect(TEST.n).toBe(512);
  }, 30_000);

  it('recorded grid: counts and optimum quoted in the debrief', () => {
    expect(LAB_RESULTS).toHaveLength(120);
    expect(LAB_RESULTS.map((r) => [r.k, r.alpha, r.capacity, r.noise])).toEqual(allSettings().map((s) => [s.k, s.alpha, s.capacity, s.noise]));
    const cost = (r: { k: number; capacity: number }) => r.k * r.capacity;
    expect(LAB_RESULTS.filter(passesBrief1).length).toBe(57);
    expect(LAB_RESULTS.filter(passesBrief2).length).toBe(15);
    const b3 = LAB_RESULTS.filter(passesBrief3);
    expect(b3.length).toBe(65);
    const best = Math.min(...b3.map(cost));
    expect(best).toBe(1.25);
    const winners = b3.filter((r) => cost(r) === best);
    expect(winners.length).toBe(6);
    expect(winners.every((r) => r.k === 1 && r.alpha >= 0.1)).toBe(true);
    expect(Math.min(...b3.filter((r) => r.k === 2).map(cost))).toBe(2.5);
    // α ≤ 0.03 with top-1 never balances; top-2 with the noisiest router balances even at α = 0, unless C = 1
    expect(LAB_RESULTS.filter((r) => r.k === 1 && r.alpha <= 0.03 && passesBrief1(r)).length).toBe(1);
    expect(LAB_RESULTS.filter((r) => r.k === 2 && r.alpha === 0 && r.noise === 1).map((r) => r.maxLoad <= 0.35)).toEqual([false, true, true, true]);
    // nothing reaches 3% drops at C = 1
    expect(Math.min(...LAB_RESULTS.filter((r) => r.capacity === 1).map((r) => r.drop))).toBeCloseTo(0.0643, 4);
  });

  it('re-training a sample of rows reproduces the record', () => {
    for (const i of [0, 39, 65, 112]) {
      const r = LAB_RESULTS[i];
      const s = scoreSetting(r);
      expect(s.acc).toBeCloseTo(r.acc, 4);
      expect(s.maxLoad).toBeCloseTo(r.maxLoad, 4);
      expect(s.drop).toBeCloseTo(r.drop, 4);
    }
  }, 30_000);
});

describe('MoE debrief extras', () => {
  it('accuracy range and the top-1 balance rule', () => {
    expect(Math.min(...LAB_RESULTS.map((r) => r.acc))).toBe(0.9406);
    expect(Math.max(...LAB_RESULTS.map((r) => r.acc))).toBe(0.968);
    expect(LAB_RESULTS.filter((r) => r.k === 1 && r.alpha >= 0.1).every(passesBrief1)).toBe(true);
  });
});
