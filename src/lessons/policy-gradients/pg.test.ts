// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import {
  CORRIDOR_P_STAR,
  LAB_BRIEFS,
  PUSH_MEANS,
  RACE_BEST,
  clippedObjective,
  corridorReinforce,
  corridorValues,
  drawBatch,
  estimatorStats,
  estimatorStatsWithBaseline,
  evaluateLab,
  expectedPPOReward,
  gradLogSoftmax,
  hallwayOptimalReturn,
  logSoftmaxAt,
  meanOf,
  ppoRobustness,
  raceSummary,
  runContextBandit,
  sampleIndex,
  reuseBatch,
  searchHallwayRace,
  seededRng,
  simulateCorridor,
  softmax,
  solveLabExhaustive,
  stdOf,
  normalCdf,
  wrongWayProbability,
} from './pg';

const fd = (f: (x: number[]) => number, x: number[], h = 1e-5) =>
  x.map((_, i) => {
    const up = x.slice();
    const dn = x.slice();
    up[i] += h;
    dn[i] -= h;
    return (f(up) - f(dn)) / (2 * h);
  });

describe('the log-derivative trick', () => {
  it('∇ log softmax(θ)[a] = e_a − π matches central finite differences', () => {
    const theta = [0.3, -1.2, 0.9, 0.1];
    for (let a = 0; a < 4; a++) {
      const num = fd((t) => logSoftmaxAt(t, a), theta);
      const ana = gradLogSoftmax(theta, a);
      for (let i = 0; i < 4; i++) expect(Math.abs(num[i] - ana[i])).toBeLessThan(1e-9);
    }
  });

  it('the expected push E[(R − b)(e_a − π)] is the true gradient of expected reward, for any baseline', () => {
    const theta = [0, 0.5, 1];
    const J = (t: number[]) => softmax(t).reduce((s, p, i) => s + p * PUSH_MEANS[i], 0);
    const num = fd(J, theta);
    for (const [offset, b] of [
      [0, 0],
      [50, 0],
      [50, 37],
      [50, 52.3],
    ]) {
      const st = estimatorStats(theta, PUSH_MEANS, 1, offset, b);
      for (let i = 0; i < 3; i++) expect(Math.abs(st.expected[i] - num[i])).toBeLessThan(1e-9);
    }
  });

  it('pins the wrong-way fractions quoted in the push explorer (checked against simulation)', () => {
    expect(wrongWayProbability([0, 0, 0], PUSH_MEANS, 1, 0, 0, 2)).toBeCloseTo(0.6066, 3);
    expect(wrongWayProbability([0, 0, 0], PUSH_MEANS, 1, 50, 0, 2)).toBeCloseTo(2 / 3, 3);
    expect(wrongWayProbability([0, 0, 0], PUSH_MEANS, 1, 50, 52, 2)).toBeCloseTo(0.2724, 3);
    const rng = seededRng(9);
    let wrong = 0;
    const N = 50000;
    for (let i = 0; i < N; i++) {
      const a = sampleIndex([1 / 3, 1 / 3, 1 / 3], rng);
      const R = rng.normal(PUSH_MEANS[a], 1);
      if (R * ((a === 2 ? 1 : 0) - 1 / 3) < 0) wrong++;
    }
    expect(Math.abs(wrong / N - 0.6066)).toBeLessThan(0.01);
    expect(normalCdf(1)).toBeCloseTo(0.841345, 6);
  });

  it('pins the push-explorer numbers: +50 inflates the variance 2,029×; a baseline removes it', () => {
    const plain = estimatorStats([0, 0, 0], PUSH_MEANS, 1, 0, 0);
    expect(plain.expected[0]).toBeCloseTo(-1 / 3, 10);
    expect(plain.expected[2]).toBeCloseTo(1 / 3, 10);
    expect(plain.variance).toBeCloseTo(3.556, 3);
    const shifted = estimatorStats([0, 0, 0], PUSH_MEANS, 1, 50, 0);
    expect(shifted.variance).toBeCloseTo(1803.556, 3);
    // Sampling Dull pays ~51, so its own logit is pushed UP by 51 × (1 − 1/3) = 34.
    expect(shifted.pushes[0][0]).toBeCloseTo(34, 10);
    const based = estimatorStatsWithBaseline([0, 0, 0], PUSH_MEANS, 1, 50);
    expect(based.baseline).toBeCloseTo(52, 10);
    expect(based.variance).toBeCloseTo(0.889, 3);
    expect(based.pushes[0][0]).toBeLessThan(0); // with the baseline, Dull is pushed down
    expect(shifted.variance / based.variance).toBeCloseTo(2029, 0);
    expect(estimatorStats([0, 0, 0], PUSH_MEANS, 1, 10, 0).variance / based.variance).toBeCloseTo(109, 0);
  });
});

describe('the aliased corridor', () => {
  it('has its best stochastic policy at p = 2 − √2 with value −11.66; deterministic policies never finish', () => {
    expect(CORRIDOR_P_STAR).toBeCloseTo(0.5858, 4);
    expect(corridorValues(CORRIDOR_P_STAR).v1).toBeCloseTo(-11.657, 3);
    const dv = (corridorValues(CORRIDOR_P_STAR + 1e-6).v1 - corridorValues(CORRIDOR_P_STAR - 1e-6).v1) / 2e-6;
    expect(Math.abs(dv)).toBeLessThan(1e-4);
    expect(corridorValues(0).v1).toBe(-Infinity);
    expect(corridorValues(1).v1).toBe(-Infinity);
    expect(corridorValues(0.5).v1).toBeCloseTo(-12, 10);
    // ε-greedy with ε = 0.1 on an all-identical observation: P(Right) is 0.95 or 0.05
    expect(corridorValues(0.95).v1).toBeCloseTo(-44.21, 2);
    expect(corridorValues(0.05).v1).toBeCloseTo(-82.11, 2);
  });

  it('the closed form agrees with simulation and with the Bellman equations', () => {
    const p = 0.3;
    const { v1, v2, v3 } = corridorValues(p);
    expect(v1).toBeCloseTo(-1 + p * v2 + (1 - p) * v1, 10);
    expect(v2).toBeCloseTo(-1 + p * v1 + (1 - p) * v3, 10);
    expect(v3).toBeCloseTo(-1 + (1 - p) * v2, 10);
    const rng = seededRng(4);
    const steps: number[] = [];
    for (let i = 0; i < 20000; i++) steps.push(simulateCorridor(p, rng, 10000).actions.length);
    expect(meanOf(steps)).toBeGreaterThan(-v1 - 0.6);
    expect(meanOf(steps)).toBeLessThan(-v1 + 0.6);
  });

  it('REINFORCE pulls P(Right) from 0.15 into the neighbourhood of p*', () => {
    const tr = corridorReinforce(0.03, 300, 1);
    expect(tr[0]).toBeCloseTo(0.15, 10);
    const late = meanOf(tr.slice(-100));
    expect(late).toBeGreaterThan(0.5);
    expect(late).toBeLessThan(0.68);
  });
});

describe('the hallway race (each method at its own searched best, 20 seeds)', () => {
  it('re-derives the recorded best settings', () => {
    const found = searchHallwayRace();
    expect(found.map((f) => ({ est: f.est, alpha: f.alpha, beta: f.beta }))).toEqual(RACE_BEST);
  });

  it('pins the quoted results: actor-critic wins, and the plain baseline is within noise of none', () => {
    const optimum = hallwayOptimalReturn();
    expect(optimum).toBeCloseTo(3.618, 3);
    const [none, state, td] = raceSummary();
    expect(none.score).toBeCloseTo(1.372, 2);
    expect(state.score).toBeCloseTo(0.972, 2);
    expect(td.score).toBeCloseTo(2.563, 2);
    expect(stdOf(none.scores)).toBeCloseTo(2.84, 1);
    expect(stdOf(state.scores)).toBeCloseTo(3.56, 1);
    expect(stdOf(td.scores)).toBeCloseTo(0.53, 1);
    expect(td.late).toBeCloseTo(3.565, 2);
    expect(none.late).toBeCloseTo(2.649, 2);
    // the advantage signal each method feeds its update
    expect(none.advSpread).toBeCloseTo(6.1, 1);
    expect(state.advSpread).toBeCloseTo(5.4, 1);
    expect(td.advSpread).toBeCloseTo(2.3, 1);
    // none vs per-state baseline: difference smaller than one standard error of either → no winner
    const se = Math.max(stdOf(none.scores), stdOf(state.scores)) / Math.sqrt(20);
    expect(Math.abs(none.score - state.score)).toBeLessThan(se);
  });
});

describe('PPO clip', () => {
  it('the clipped objective flattens outside [1 − ε, 1 + ε] in the direction the advantage pushes', () => {
    expect(clippedObjective(1.5, 2, 0.2)).toBeCloseTo(2.4, 10);
    expect(clippedObjective(0.5, 2, 0.2)).toBeCloseTo(1.0, 10); // no floor for good actions made rarer
    expect(clippedObjective(0.5, -2, 0.2)).toBeCloseTo(-1.6, 10);
    expect(clippedObjective(1.5, -2, 0.2)).toBeCloseTo(-3.0, 10); // no ceiling on the penalty
  });

  it('one epoch on a fresh batch is the plain policy gradient, so clip changes nothing at K = 1', () => {
    const b = drawBatch([0, 0, 0], seededRng(3));
    const v = reuseBatch([0, 0, 0], b, 1, 1, null);
    const c = reuseBatch([0, 0, 0], b, 1, 1, 0.2);
    expect(v[1]).toEqual(c[1]);
  });

  it('the demo batch (seed 2) misleads: plain reuse commits to Decent, clip moves modestly', () => {
    const b = drawBatch([0, 0, 0], seededRng(2));
    const v = reuseBatch([0, 0, 0], b, 10, 1, null);
    const c = reuseBatch([0, 0, 0], b, 10, 1, 0.2);
    expect(softmax(v[10])[1]).toBeCloseTo(0.96, 2);
    expect(softmax(c[10])[1]).toBeCloseTo(0.52, 2);
    expect(expectedPPOReward(v[10])).toBeCloseTo(2.004, 3);
    expect(expectedPPOReward(c[10])).toBeCloseTo(2.059, 3);
  });

  it('pins the 100-seed robustness sweep at 10 epochs per batch', () => {
    const plain = ppoRobustness(10, null);
    const clip = ppoRobustness(10, 0.2);
    const byLr = (rows: typeof plain, lr: number) => rows.find((r) => r.lr === lr)!;
    // each at its own best step size: close
    expect(byLr(plain, 0.2).mean).toBeCloseTo(2.907, 2);
    expect(byLr(plain, 0.2).failures).toBe(5);
    expect(byLr(clip, 1).mean).toBeCloseTo(2.925, 2);
    expect(byLr(clip, 1).failures).toBe(3);
    expect(Math.max(...plain.map((r) => r.mean))).toBeCloseTo(byLr(plain, 0.2).mean, 10);
    // the difference is robustness across step sizes
    expect(byLr(plain, 1).failures).toBe(31);
    expect(byLr(clip, 1).failures).toBe(3);
    // "good" (≤ 5 of 100 stuck): plain 0.05–0.2, clip 0.05–2
    expect(plain.filter((r) => r.failures <= 5).map((r) => r.lr)).toEqual([0.05, 0.1, 0.2]);
    expect(clip.filter((r) => r.failures <= 5).map((r) => r.lr)).toEqual([0.05, 0.1, 0.2, 0.5, 1, 2]);
    expect(plain.filter((r) => r.failures <= 5).length).toBe(3);
    expect(clip.filter((r) => r.failures <= 5).length).toBe(6);
  });
});

describe('Lab: stabilise the policy gradient (exhaustive over 4 estimators × 8 step sizes, 20 seeds)', () => {
  it('counts the passing settings per brief and finds the optimum', () => {
    const paid = solveLabExhaustive('paid');
    expect(paid.total).toBe(32);
    expect(paid.passing.length).toBe(9);
    expect(paid.passing.every((p) => p.cfg.est !== 'none')).toBe(true);
    expect(paid.best.cfg.alpha).toBe(0.05);
    expect(paid.best.score).toBeCloseTo(0.8975, 3);

    const pay = solveLabExhaustive('payscales');
    expect(pay.passing.length).toBe(6);
    expect(pay.passing.every((p) => p.cfg.est === 'state' || p.cfg.est === 'td')).toBe(true);
    expect(pay.best.score).toBeCloseTo(0.8377, 3);

    const hall = solveLabExhaustive('hallway');
    expect(hall.passing.length).toBe(3);
    expect(hall.passing.every((p) => p.cfg.est === 'td')).toBe(true);
    expect(hall.best.cfg).toEqual({ est: 'td', alpha: 0.1 });
    expect(hall.best.score).toBeCloseTo(3.388, 2);
  });

  it('Brief 1 without a baseline marries its first press: 60% on the 20 seeds is luck, a third over 200', () => {
    const first = [0, 0, 0];
    for (let sd = 1; sd <= 20; sd++) {
      const r = seededRng(sd);
      r.int(1);
      first[sampleIndex([1 / 3, 1 / 3, 1 / 3], r)]++;
    }
    expect(first[2]).toBe(12);
    expect(evaluateLab('paid', { est: 'none', alpha: 0.2 }).score).toBeCloseTo(0.6, 3);
    let married = 0;
    for (let sd = 1; sd <= 200; sd++) {
      const run = runContextBandit({ offsets: [50], deltas: [-1, 0, 1], sigma: 1, pulls: 300 }, 'none', 0.2, 0.1, sd);
      if (run.pBest[0] > 0.5) married++;
    }
    expect(married).toBe(67);
    // Brief 2: one running average is worse than none
    const pay = solveLabExhaustive('payscales');
    const bestOf = (e: string) => Math.max(...pay.all.filter((x) => x.cfg.est === e).map((x) => x.score));
    expect(bestOf('average')).toBeCloseTo(0.519, 3);
    expect(bestOf('none')).toBeCloseTo(0.59, 3);
    // Brief 3: Monte-Carlo estimators top out between 2.5 and 2.9
    const hall = solveLabExhaustive('hallway');
    for (const e of ['none', 'average', 'state']) {
      const b = Math.max(...hall.all.filter((x) => x.cfg.est === e).map((x) => x.score));
      expect(b).toBeGreaterThan(2.45);
      expect(b).toBeLessThan(2.95);
    }
  });

  it('in one-step briefs the per-state baseline and actor-critic are literally the same algorithm', () => {
    for (const alpha of [0.01, 0.05]) {
      expect(evaluateLab('payscales', { est: 'td', alpha }).score).toBe(evaluateLab('payscales', { est: 'state', alpha }).score);
    }
    expect(LAB_BRIEFS.map((b) => b.threshold)).toEqual([0.8, 0.75, 3.0]);
  });
});
