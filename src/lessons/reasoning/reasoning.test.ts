// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import { bestOfN, BENCH, benchAccuracy, flops, gradLogPi, grpoStep, initPolicy, majority, passAtN, policyStats, pSingle, RM_ACC, RM_BLIND_SPOT, softmaxRow, TIERS } from './reasoning';
import { B2_BAR, B3_BAR, B3_TOKENS, briefGrid, BRIEFS, HARD, labHint, passes, refScore, score } from './lab';

/** Monte Carlo of the same stand-in, to check the closed forms. */
function simulate(p: number, trap: number, n: number, mode: 'majority' | 'noisy', trials: number, seed: number) {
  const r = createRng(seed);
  let ok = 0;
  for (let t = 0; t < trials; t++) {
    const answers: number[] = []; // 0 = correct, 1 = trap, 2+ = unique wrong
    const scores: number[] = [];
    for (let i = 0; i < n; i++) {
      const u = r.next();
      if (u < p) answers.push(0);
      else if (u < p + (1 - p) * trap) answers.push(1);
      else answers.push(2 + i);
      if (mode === 'noisy') {
        if (answers[i] === 0) scores.push(r.next() < RM_ACC ? 1 : 0);
        else scores.push(r.next() < RM_BLIND_SPOT ? 2 : r.next() < 1 - RM_ACC ? 1 : 0);
      }
    }
    if (mode === 'majority') {
      const counts = new Map<number, number>();
      answers.forEach((a) => counts.set(a, (counts.get(a) ?? 0) + 1));
      const m = Math.max(...counts.values());
      const top = [...counts].filter(([, c]) => c === m).map(([a]) => a);
      if (top[r.int(top.length)] === 0) ok++;
    } else {
      const m = Math.max(...scores);
      const idx = scores.map((s, i) => [s, i]).filter(([s]) => s === m).map(([, i]) => i);
      if (answers[idx[r.int(idx.length)]] === 0) ok++;
    }
  }
  return ok / trials;
}

describe('test-time compute: exact formulas', () => {
  it('majority vote and noisy best-of-N match a Monte Carlo of the same model', () => {
    for (const [p, trap, n] of [
      [0.3, 0.6, 9],
      [0.4, 0.2, 15],
      [0.6, 0.8, 4],
    ]) {
      expect(Math.abs(majority(p, trap, n) - simulate(p, trap, n, 'majority', 40000, 1))).toBeLessThan(0.01);
    }
    for (const [p, n] of [
      [0.3, 8],
      [0.5, 32],
    ]) {
      // the noisy draw in the simulation spends rng calls differently, so compare loosely
      expect(Math.abs(bestOfN(p, n, 'noisy') - simulate(p, 0, n, 'noisy', 40000, 2))).toBeLessThan(0.012);
    }
  });

  it('numbers quoted in the lesson', () => {
    // a 7-step problem for the medium model, with and without room to think
    expect(+pSingle('medium', 7, 400).toFixed(3)).toBe(0.478);
    expect(+pSingle('medium', 7, 0).toFixed(3)).toBe(0.052);
    // best-of-N with the noisy judge on a p = 0.3 problem rises, peaks, then falls
    const curve = [1, 2, 4, 8, 16, 32, 64, 128].map((n) => +bestOfN(0.3, n, 'noisy').toFixed(3));
    expect(curve).toEqual([0.3, 0.44, 0.574, 0.621, 0.568, 0.454, 0.289, 0.117]);
    // majority vote helps when the right answer is the most common one, hurts when a trap is
    expect([1, 5, 31].map((n) => +majority(0.4, 0.2, n).toFixed(3))).toEqual([0.4, 0.686, 0.989]);
    expect([1, 5, 31].map((n) => +majority(0.3, 0.6, n).toFixed(3))).toEqual([0.3, 0.351, 0.214]);
    // two votes with a random tie-break are no better than one
    expect(majority(0.37, 0.5, 2)).toBeCloseTo(0.37, 12);
    expect(+passAtN(0.3, 8).toFixed(3)).toBe(0.942);
    // the bench and its single-shot numbers
    expect(BENCH).toHaveLength(50);
    const single = (tier: 'small' | 'medium' | 'large', thinking: number) => +benchAccuracy({ tier, strategy: 'single', n: 1, verifier: 'none', thinking }).toFixed(3);
    expect([single('small', 0), single('small', 400), single('large', 0), single('large', 400)]).toEqual([0.069, 0.357, 0.454, 0.804]);
    expect(flops({ tier: 'large', strategy: 'single', n: 1, verifier: 'none', thinking: 100 })).toBe(16.8e12);
    // the medium model with the noisy judge at 400 thinking tokens, by sample count
    expect([2, 8, 64].map((n) => +benchAccuracy({ tier: 'medium', strategy: 'best', n, verifier: 'noisy', thinking: 400 }).toFixed(3))).toEqual([0.727, 0.819, 0.538]);
  });

  it('GRPO: the gradient of log π, and the live run (seed 1)', () => {
    const row = [0.3, -1, 2, 0.5];
    const g = gradLogPi(row, 2);
    const h = 1e-6;
    row.forEach((_, j) => {
      const a = row.slice();
      const b = row.slice();
      a[j] += h;
      b[j] -= h;
      expect(g[j]).toBeCloseTo((Math.log(softmaxRow(a)[2]) - Math.log(softmaxRow(b)[2])) / (2 * h), 6);
    });
    const r = TIERS.medium.r;
    const pol = initPolicy();
    const rng = createRng(1);
    const s0 = policyStats(pol, r);
    for (let s = 0; s < 600; s++) grpoStep(pol, r, rng.next);
    const s1 = policyStats(pol, r, 600);
    expect([+s0.acc.toFixed(3), +s0.meanLen.toFixed(2)]).toEqual([0.263, 1.52]);
    expect([+s1.acc.toFixed(3), +s1.meanLen.toFixed(2)]).toEqual([0.556, 5.18]);
    // easy problems learn to write exactly the steps they need
    expect(s1.lenByDiff.slice(0, 7).map((v) => Math.round(v))).toEqual([1, 2, 3, 4, 5, 6, 7]);
    // the ceiling: always writing exactly d steps
    expect(+Array.from({ length: 10 }, (_, i) => r ** (i + 1) / 10).reduce((a, b) => a + b).toFixed(3)).toBe(0.586);
  });
});

describe('Lab: Spend the Test-Time Budget', () => {
  it('each brief has more than one solution, and the optimums quoted in the debrief', () => {
    const sols = Object.fromEntries(BRIEFS.map((b) => [b.id, briefGrid(b.id).map((c) => ({ c, s: score(b.id, c) })).filter((x) => passes(b.id, x.s))]));
    expect(briefGrid('cheap')).toHaveLength(70);
    expect(briefGrid('noisy')).toHaveLength(30);
    expect(briefGrid('hard')).toHaveLength(30);
    expect(+refScore().acc.toFixed(3)).toBe(0.567);
    expect(sols.cheap).toHaveLength(4);
    expect(Math.min(...sols.cheap.map((x) => x.s.flops))).toBe(6.72e12);
    expect(sols.noisy.map((x) => [x.c.n, x.c.thinking])).toEqual([
      [4, 400],
      [8, 400],
    ]);
    expect(B2_BAR).toBe(0.8);
    expect(sols.hard.map((x) => [x.c.n, x.c.thinking, x.s.tokens])).toEqual([
      [4, 400, 1680],
      [8, 300, 2560],
    ]);
    expect([B3_BAR, B3_TOKENS, HARD.length]).toEqual([0.85, 3000, 20]);
    // the best noisy-judge accuracy available
    expect(+Math.max(...briefGrid('noisy').map((c) => score('noisy', c).acc)).toFixed(3)).toBe(0.819);
  });

  it('hints describe evidence, never a setting', () => {
    const LEAK = [/N\s*=\s*\d/, /\b(400|300|200|100)\b.*token/i, /\bmedium\b|\bsmall\b|\blarge model’s\b.*use/i, /try\s+\d/i, /set\s+/i];
    for (const b of BRIEFS)
      for (const c of briefGrid(b.id)) {
        const h = labHint(b.id, c, score(b.id, c));
        expect(h.length).toBeGreaterThan(20);
        for (const p of LEAK) expect(h, h).not.toMatch(p);
      }
  });
});

describe('iso-compute widget readouts', () => {
  it('64 small answers vs one large answer, both thinking 400 tokens', () => {
    expect(flops({ tier: 'small', strategy: 'majority', n: 64, verifier: 'none', thinking: 400 })).toBeLessThan(flops({ tier: 'large', strategy: 'single', n: 1, verifier: 'none', thinking: 400 }));
    expect(Math.floor(70e9 / 1e9)).toBe(70);
    const rows = Array.from({ length: 10 }, (_, i) => ({ L: pSingle('large', i + 1, 400), S: majority(pSingle('small', i + 1, 400), 0.4, 64), X: passAtN(pSingle('small', i + 1, 400), 64) }));
    expect(rows.findIndex((r) => r.L > r.S) + 1).toBe(5);
    expect(rows.every((r) => r.X > r.L)).toBe(true);
    expect(+pSingle('small', 10, 400).toFixed(3)).toBe(0.107);
  });
});
