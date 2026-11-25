// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import {
  AND,
  AND_START,
  CLOUD,
  ETAS,
  OR,
  OR_START,
  THIN,
  THIN_TARGET,
  XOR,
  XOR_CAP,
  allCorrect,
  andSearch,
  bestMargin,
  novikoffBound,
  out,
  permutations,
  run,
  thinRun,
  thinSearch,
  update,
  type Pt,
} from './perceptron';

const zero: [number, number, number] = [0, 0, 0];
const ids = (pts: Pt[]) => pts.map((_, i) => i);

describe('the perceptron rule', () => {
  it('does nothing when the output is right, and adds or subtracts the input when it is wrong', () => {
    const p = OR[3]; // (1, 1) → 1
    expect(update([1, 1, 0], p, 1)).toEqual({ w: [1, 1, 0], mistake: false, said: 1 });
    // fired when it shouldn't: subtract
    expect(update([0, 0, 0], OR[0], 1).w).toEqual([0, 0, -1]);
    // didn't fire when it should: add (step 0.5)
    expect(update([-1, -1, -1], p, 0.5).w).toEqual([-0.5, -0.5, -0.5]);
  });

  it('fires at a score of exactly zero (same step convention as the neuron lesson)', () => {
    expect(out(zero, OR[0])).toBe(1);
  });

  it('the stepper cloud is separable by x1 + 2·x2 = 1.4', () => {
    expect(allCorrect([1, 2, -1.4], CLOUD)).toBe(true);
  });

  it('learns OR, AND and the cloud from zero for every order, and never settles on XOR', () => {
    for (const o of permutations(4)) {
      expect(run(OR, o, zero, 1).converged).toBe(true);
      expect(run(AND, o, zero, 1).converged).toBe(true);
      const x = run(XOR, o, zero, 1, XOR_CAP);
      expect(x.converged).toBe(false);
      expect(x.mistakes).toBe(XOR_CAP);
    }
    expect(run(CLOUD, ids(CLOUD), zero, 1).mistakes).toBe(18);
  });

  it('on XOR the weights come back to where they were: a cycle, not slow progress', () => {
    const r = run(XOR, [0, 1, 2, 3], zero, 1, 40, true);
    const seen = new Set<string>();
    let repeat = false;
    for (const e of r.events) {
      const k = `${e.i}|${e.before.join(',')}`;
      if (seen.has(k)) repeat = true;
      seen.add(k);
    }
    expect(repeat).toBe(true);
  });
});

describe("Novikoff's bound: at most (R/γ)² mistakes from zero weights", () => {
  it('holds for every dataset in the lesson, every order of the corner sets, and every step size', () => {
    for (const d of [OR, AND]) {
      const bound = novikoffBound(d);
      for (const o of permutations(4)) for (const eta of ETAS) expect(run(d, o, zero, eta).mistakes).toBeLessThanOrEqual(bound);
    }
    for (const d of [CLOUD, THIN]) expect(run(d, ids(d), zero, 1, 1000).mistakes).toBeLessThanOrEqual(novikoffBound(d));
  });

  it('pins the bounds quoted in the lab debrief', () => {
    expect(novikoffBound(OR)).toBeCloseTo(27, 3);
    expect(bestMargin(OR).gamma).toBeCloseTo(1 / 3, 6);
    expect(Math.round(novikoffBound(THIN))).toBe(416);
    expect(Math.round(novikoffBound(CLOUD))).toBe(161);
    const without = (name: string) => THIN.filter((p) => p.name !== name);
    expect(Math.round(novikoffBound(without('H')))).toBe(155);
    expect(Math.round(novikoffBound(without('F')))).toBe(416);
  });

  it('finds no positive margin for XOR', () => {
    expect(bestMargin(XOR).gamma).toBeLessThanOrEqual(0);
    expect(novikoffBound(XOR)).toBe(Infinity);
  });
});

describe('Lab: Train the Mark I', () => {
  it('brief 1: OR is learned from the AND setting for every order and step size, in 1 to 4 mistakes', () => {
    expect(allCorrect(OR_START, AND)).toBe(true);
    const m = permutations(4).flatMap((o) => ETAS.map((e) => run(OR, o, OR_START, e)));
    expect(m.every((r) => r.converged)).toBe(true);
    expect(Math.min(...m.map((r) => r.mistakes))).toBe(1);
    expect(Math.max(...m.map((r) => r.mistakes))).toBe(4);
  });

  it('brief 2: from the factory setting, all 72 settings learn AND; the fewest mistakes is 5, reached by 2 settings, both step 1 and the same cycle', () => {
    const s = andSearch();
    expect(s.total).toBe(72);
    expect(s.converged).toBe(72);
    expect(s.min).toBe(5);
    expect(s.worst).toBe(25);
    const range = (eta: number) => {
      const m = permutations(4).map((o) => run(AND, o, AND_START, eta).mistakes);
      return [Math.min(...m), Math.max(...m)];
    };
    expect([range(1), range(0.5), range(0.1)]).toEqual([[5, 12], [7, 12], [25, 25]]);
    expect(s.best.map((b) => [b.eta, b.order.map((i) => AND[i].name).join(' ')])).toEqual([
      [1, '(1, 0) (1, 1) (0, 0) (0, 1)'],
      [1, '(1, 1) (0, 0) (0, 1) (1, 0)'],
    ]);
    expect(AND_START).toEqual([-1, 1, -0.5]);
  });

  it('brief 3: the thin set takes 66 mistakes; removing one point gets to 30 or fewer only for F, H or K, and K alone reaches 18', () => {
    const s = thinSearch();
    expect(s.full).toBe(66);
    expect(s.each.map((e) => `${e.name}:${e.mistakes}`).join(' ')).toBe('A:60 B:66 C:66 D:33 E:77 F:28 G:51 H:27 I:66 J:66 K:18');
    expect(s.passing.map((e) => e.name)).toEqual(['F', 'H', 'K']);
    expect(s.best.map((e) => e.name)).toEqual(['K']);
    expect(s.min).toBe(18);
    expect(THIN_TARGET).toBe(30);
    expect(thinRun(null).r.converged).toBe(true);
  });

  it('brief 4: XOR never converges within the cap, whatever the start, order or step', () => {
    const starts: [number, number, number][] = [zero, [1, 1, -1.5], [-1, 1, -0.5], [1, -1, 0.5]];
    for (const w0 of starts) for (const o of permutations(4)) for (const e of ETAS) expect(run(XOR, o, w0, e, XOR_CAP).converged).toBe(false);
  });
});
