// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createRng } from '../../lib/rng';
import {
  GUIDE_BRIEFS,
  TRIBES,
  abarAt,
  classProbs,
  ddim,
  guess,
  guideExhaustive,
  guided,
  judge,
  labTry,
  noiseStart,
  sampleData,
  stops,
  type Den,
  type Pt,
} from './guide';

const run = (cls: number | null, w: number, n: number) => judge(TRIBES, ddim(guided(TRIBES, cls, w), stops(n), noiseStart(2000, 5)).xs, cls ?? 0);
const p1 = (v: number) => (100 * v).toFixed(1);

/** Exact density of the noisy labelled mixture, optionally for one class only. */
function density(a: number, x: Pt, cls: number | null) {
  let p = 0;
  let tot = 0;
  TRIBES.m.forEach((m, k) => {
    if (cls !== null && TRIBES.cls[k] !== cls) return;
    const v = a * TRIBES.s[k] ** 2 + 1 - a;
    const d2 = (x[0] - Math.sqrt(a) * m[0]) ** 2 + (x[1] - Math.sqrt(a) * m[1]) ** 2;
    p += (TRIBES.w[k] / (2 * Math.PI * v)) * Math.exp(-d2 / (2 * v));
    tot += TRIBES.w[k];
  });
  return p / tot;
}

describe('exact conditional denoising', () => {
  it('Tweedie holds for both the unconditional and the conditional denoiser (central differences)', () => {
    const h = 1e-5;
    for (const cls of [null, 0, 1])
      for (const a of [0.1, 0.6, 0.97])
        for (const x of [
          [0.2, 0.1],
          [-1.1, -0.3],
        ] as Pt[]) {
          const gx = (Math.log(density(a, [x[0] + h, x[1]], cls)) - Math.log(density(a, [x[0] - h, x[1]], cls))) / (2 * h);
          const gy = (Math.log(density(a, [x[0], x[1] + h], cls)) - Math.log(density(a, [x[0], x[1] - h], cls))) / (2 * h);
          const g = guess(TRIBES, a, x, cls).x0;
          // E[x0 | x] = (x + (1 − a) ∇log p) / √a
          expect(g[0]).toBeCloseTo((x[0] + (1 - a) * gx) / Math.sqrt(a), 5);
          expect(g[1]).toBeCloseTo((x[1] + (1 - a) * gy) / Math.sqrt(a), 5);
        }
  });

  it('Bayes: conditional score = unconditional score + gradient of log p(blue | x)', () => {
    const h = 1e-5;
    const a = 0.5;
    const x: Pt = [0.1, 0.3];
    const d = (f: (q: Pt) => number) => [(f([x[0] + h, x[1]]) - f([x[0] - h, x[1]])) / (2 * h), (f([x[0], x[1] + h]) - f([x[0], x[1] - h])) / (2 * h)];
    const cond = d((q) => Math.log(density(a, q, 0)));
    const unc = d((q) => Math.log(density(a, q, null)));
    const cl = d((q) => Math.log(classProbs(TRIBES, a, q)[0]));
    expect(cond[0]).toBeCloseTo(unc[0] + cl[0], 5);
    expect(cond[1]).toBeCloseTo(unc[1] + cl[1], 5);
  });

  it('from pure noise, the conditional guess is the class mean', () => {
    const g = guess(TRIBES, 0, [0.3, 0.2], 0).x0;
    expect(g[0]).toBeCloseTo(-1, 10);
    expect(g[1]).toBeCloseTo(0.2 / 3, 10);
  });
});

describe('numbers quoted in the lesson', () => {
  it('conditioning: 99.4% on blue, fair split 31.9 / 34.4 / 33.7; unconditional 48.6% blue, 50.3% orange', () => {
    const b = run(0, 1, 50);
    expect(p1(b.onClass)).toBe('99.4');
    expect(b.withinClass.map(p1)).toEqual(['31.9', '34.4', '33.7']);
    const xs = ddim(guided(TRIBES, null, 1), stops(50), noiseStart(2000, 5)).xs;
    expect(p1(judge(TRIBES, xs, 0).onClass)).toBe('48.6');
    expect(p1(judge(TRIBES, xs, 1).onClass)).toBe('50.3');
  });

  it('the classifier at step 200: ~70% blue near the inner blob, ~99% near the outer; ~50-50 by step 600', () => {
    const at = (t: number, m: Pt) => classProbs(TRIBES, abarAt(t), [Math.sqrt(abarAt(t)) * m[0], Math.sqrt(abarAt(t)) * m[1]])[0];
    expect(at(200, TRIBES.m[2])).toBeCloseTo(0.70, 2);
    expect(at(200, TRIBES.m[0])).toBeCloseTo(0.99, 2);
    expect(Math.abs(at(600, TRIBES.m[2]) - 0.5)).toBeLessThan(0.03);
    expect(Math.abs(at(600, TRIBES.m[0]) - 0.5)).toBeLessThan(0.03);
  });

  it('guidance at 50 steps: w = 2 leaves the inner blob 11.8%, w = 3 1.8%, w = 5 gives 91.7% / 0.0%, w = 8 pushes 8.9% off', () => {
    expect(p1(run(0, 2, 50).withinClass[2])).toBe('11.8');
    expect(p1(run(0, 3, 50).withinClass[2])).toBe('1.8');
    const w5 = run(0, 5, 50);
    expect(p1(w5.withinClass[0])).toBe('91.7');
    expect(p1(w5.withinClass[2])).toBe('0.0');
    expect(p1(run(0, 8, 50).offShape)).toBe('8.9');
  });

  it('at 3 steps: w = 1 strands 26.5%; w = 3 lands 92.2% on blue, 96.0% of them on the outer blob', () => {
    const w1 = run(0, 1, 3);
    expect(p1(w1.offShape)).toBe('26.5');
    expect(p1(w1.onClass)).toBe('73.5');
    const w3 = run(0, 3, 3);
    expect(p1(w3.onClass)).toBe('92.2');
    expect(p1(w3.withinClass[0])).toBe('96.0');
  });

  it('the sharpened target p(x|blue)·p(blue|x)^(w−1) keeps each blue blob near a third', () => {
    const xs = sampleData(TRIBES, 40000, createRng(1), 0);
    for (const w of [3, 5]) {
      const sh = [0, 0, 0];
      let tot = 0;
      for (const x of xs) {
        const wt = classProbs(TRIBES, 1, x)[0] ** (w - 1);
        const k = [0, 1, 2].reduce((b, j) => (Math.hypot(x[0] - TRIBES.m[j][0], x[1] - TRIBES.m[j][1]) < Math.hypot(x[0] - TRIBES.m[b][0], x[1] - TRIBES.m[b][1]) ? j : b), 0);
        sh[k] += wt;
        tot += wt;
      }
      for (const v of sh) expect(Math.abs(v / tot - 1 / 3)).toBeLessThan(0.02);
    }
  });

  it('guiding only at low noise gives back the variety and the stranding together (3 steps, w = 3)', () => {
    const lowOnly: Den = (x, a) => {
      const c = guess(TRIBES, a, x, 0).x0;
      if (a < 0.8) return c;
      const u = guess(TRIBES, a, x, null).x0;
      return [u[0] + 3 * (c[0] - u[0]), u[1] + 3 * (c[1] - u[1])];
    };
    const j = judge(TRIBES, ddim(lowOnly, stops(3), noiseStart(2000, 5)).xs, 0);
    expect(j.onClass).toBeLessThan(run(0, 1, 3).onClass);
    expect(Math.min(...j.withinClass)).toBeGreaterThan(0.25);
  });
});

describe('Lab: Tune the Guidance (exhaustive)', () => {
  it('fewest steps: catalogue 5 (w 1, 1.25; 7 settings), poster 2 (w ≥ 3; 46), both 6 (w 1–1.75; 9)', () => {
    const [c, p, b] = GUIDE_BRIEFS.map((br) => guideExhaustive(br));
    expect([c.fewest, c.passing, c.total]).toEqual([5, 7, 102]);
    expect(c.atFewest).toEqual([1, 1.25]);
    expect([p.fewest, p.passing]).toEqual([2, 46]);
    expect(p.atFewest).toEqual([3, 3.25, 3.5, 3.75, 4]);
    expect([b.fewest, b.passing]).toEqual([6, 9]);
    expect(b.atFewest).toEqual([1, 1.25, 1.5, 1.75]);
    expect(p1(labTry(2, 3).onClass)).toBe('95.3');
  }, 30_000);

  it('the fewest steps are the same with two other sets of starting noise', () => {
    for (const seed of [6, 7]) expect(GUIDE_BRIEFS.map((br) => guideExhaustive(br, seed).fewest)).toEqual([5, 2, 6]);
  }, 60_000);
});
