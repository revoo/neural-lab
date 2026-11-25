// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import {
  BRIEFS,
  GAMMA,
  QL_EPISODES,
  boatThreshold,
  boatVerdict,
  boatWorld,
  changedSquares,
  exhaustiveLab,
  fieldFinishes,
  findPlan,
  judge,
  naiveBreakingPoint,
  rewardOf,
  shapingRace,
  solve,
  solveField,
  vaseWorld,
  type Weights,
} from './hack';

describe('the Value Iteration adversary', () => {
  it('satisfies the Bellman optimality equation to 1e-9 in every world', () => {
    const cases: { w: ReturnType<typeof boatWorld>; wts: Weights }[] = [
      { w: boatWorld(), wts: { finish: 10, pad: 3, step: 0 } },
      { w: vaseWorld(), wts: { deliver: 10, break: -2, step: -1 } },
      { w: BRIEFS[2].world(), wts: { 'looks-clean': 1, tamper: -5, clean: 1 } },
    ];
    for (const { w, wts } of cases) {
      const sol = solve(w, wts);
      sol.states.forEach((s, i) => {
        let best = -Infinity;
        for (let a = 0; a < w.actions; a++) {
          const r = w.step(s, a);
          best = Math.max(best, rewardOf(r.events, wts) + (r.done ? 0 : GAMMA * sol.V[sol.ids.get(r.next)!]));
        }
        expect(Math.abs(best - sol.V[i])).toBeLessThan(1e-9);
      });
    }
  });
});

describe('the boat race', () => {
  it('flips from ignoring the pad, to one detour, to looping for ever, at exact thresholds', () => {
    const verdict = (p: number) => boatVerdict(solve(boatWorld(), { finish: 10, pad: p, step: 0 }));
    expect(verdict(0)).toBe('direct');
    expect(verdict(3)).toBe('pad-then-finish');
    expect(verdict(5)).toBe('loops');
    const t1 = boatThreshold(10, 0, 0, 10, 'direct');
    // closed form: detour pays p·γ³ + 10γ⁸ against 10γ⁴ for driving straight
    expect(t1).toBeCloseTo(10 * (GAMMA - GAMMA ** 5), 5);
    expect(t1).toBeCloseTo(1.7622, 4);
    const t2 = boatThreshold(10, 0, t1 + 1e-4, 10, 'pad-then-finish');
    expect(t2).toBeCloseTo(4.5396, 4);
    expect(solve(boatWorld(), { finish: 10, pad: 0, step: 0 }).V[0]).toBeCloseTo(10 * GAMMA ** 4, 10);
    expect(10 * GAMMA ** 4).toBeCloseTo(8.145, 3);
  });

  it('the looping plan really never finishes, and drives over the pad again and again', () => {
    const sol = solve(boatWorld(), { finish: 10, pad: 5, step: 0 });
    const plan = findPlan(sol, 40);
    expect(plan.outcome.done).toBe(false);
    expect(plan.steps.filter((s) => s.events.includes('pad')).length).toBeGreaterThanOrEqual(4);
  });
});

describe('shaping', () => {
  it('a naive closeness bonus stops the agent finishing above k = 4/7; potential-based shaping never changes a single square', () => {
    expect(naiveBreakingPoint()).toBeCloseTo(4 / 7, 5);
    expect(fieldFinishes('naive', 0.5)).toBe(true);
    expect(fieldFinishes('naive', 0.6)).toBe(false);
    expect(changedSquares('naive', 0.5)).toBe(0);
    expect(changedSquares('naive', 1)).toBe(2);
    for (const k of [0.5, 1, 4, 10, 50]) {
      expect(changedSquares('potential', k)).toBe(0);
      expect(fieldFinishes('potential', k)).toBe(true);
    }
  });

  it('potential-based shaping shifts every value by exactly −k·Φ(s) (so it can’t change any choice)', () => {
    const base = solveField('none', 0);
    const shaped = solveField('potential', 4);
    base.states.forEach((s, i) => {
      const [r, c] = s.split(',').map(Number);
      const phi = (8 - (Math.abs(r - 0) + Math.abs(c - 4))) / 8;
      expect(shaped.V[shaped.ids.get(s)!]).toBeCloseTo(base.V[i] - 4 * phi, 9);
    });
  });

  it('pins the fair Q-learning race (each at its best step size, 30 seeds)', () => {
    const none = shapingRace('none', 0);
    expect(none.alpha).toBe(0.9);
    expect(none.mean).toBeCloseTo(15.3, 1);
    const pot = shapingRace('potential', 4);
    expect(pot.alpha).toBe(0.05);
    expect(pot.mean).toBeCloseTo(1.6, 1);
    const naive = shapingRace('naive', 0.25);
    expect(naive.mean).toBeCloseTo(39.1, 1);
    expect(naive.perSeed.filter((v) => v > QL_EPISODES).length).toBe(1);
    expect(none.curve[0]).toBeCloseTo(71.5, 1);
    expect(naive.curve[0]).toBeCloseTo(185.9, 1);
  });
});

describe('Lab: the reward-hacking sandbox (exhaustive over every reward setting)', () => {
  it('counts the rewards that work and finds the simplest one per brief', () => {
    const boat = exhaustiveLab('boat');
    expect(boat.total).toBe(90);
    expect(boat.passing.length).toBe(13);
    expect(boat.simplest!.wts).toEqual({ finish: 0, pad: 4, step: -1 });
    expect(boat.simplestCount).toBe(1);

    const vase = exhaustiveLab('vase');
    expect(vase.total).toBe(75);
    expect(vase.passing.length).toBe(34);
    expect(vase.simplest!.wts).toEqual({ deliver: 0, break: -2, step: -1 });

    const cam = exhaustiveLab('camera');
    expect(cam.total).toBe(72);
    expect(cam.passing.length).toBe(12);
    expect(cam.simplest!.wts).toEqual({ clean: 0, 'looks-clean': 1, tamper: -5, spill: 0, step: 0 });
    expect(cam.simplestCount).toBe(1);
  });

  it('pins the loopholes the lesson describes', () => {
    expect(judge('camera', { clean: 1 }).failures).toEqual(['cleans, spills and re-cleans to farm the cleaning reward']);
    expect(judge('camera', { 'looks-clean': 1 }).failures).toEqual(['switches the camera off']);
    expect(judge('camera', { 'looks-clean': 1, tamper: -5 }).pass).toBe(true);
    expect(judge('camera', { 'looks-clean': 2, tamper: -5 }).failures).toEqual(['switches the camera off']); // a fine is a price
    // once the job is done every action ties, and one of the tied plans switches the camera off
    const after = judge('camera', { clean: 1, spill: -5 });
    expect(after.pass).toBe(false);
    expect(after.failures).toEqual(['switches the camera off']);
    expect(judge('vase', { deliver: 10, break: 0, step: 0 }).failures).toEqual(['delivers, but smashes the vase on the way']);
    expect(judge('vase', { deliver: 10, break: -2, step: 0 }).pass).toBe(true);
    expect(judge('vase', { deliver: 10, break: -2, step: -1 }).pass).toBe(false); // time pressure outbids the vase
    expect(judge('vase', { deliver: 0, break: -2, step: -1 }).pass).toBe(true);
    // scaling every reward changes nothing
    expect(judge('camera', { 'looks-clean': 10, tamper: -50 }).pass).toBe(true);
    expect(judge('vase', { deliver: 0, break: 0, step: 0 }).pass).toBe(false);
  });

  it('the vase is safe once its penalty beats the 2 extra steps: |v| above 10(γ³ − γ⁵)/γ ≈ 0.880', () => {
    let a = 0;
    let b = 5;
    while (b - a > 1e-7) {
      const m = (a + b) / 2;
      if (judge('vase', { deliver: 10, break: -m, step: 0 }).pass) b = m;
      else a = m;
    }
    expect(b).toBeCloseTo((10 * (GAMMA ** 3 - GAMMA ** 5)) / GAMMA, 5);
    expect(b).toBeCloseTo(0.88, 2);
  });
});
