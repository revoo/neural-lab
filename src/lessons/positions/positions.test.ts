import { describe, expect, it } from 'vitest';
import { LEVELS, previousWordWeights, ropeScore, selfAttention, sinusoid, solveLevel } from './positions';

describe('the positions lesson', () => {
  it('attention is order-blind: the same word gets the same output in both orders', () => {
    const a = selfAttention(['dog', 'bites', 'man']);
    const b = selfAttention(['man', 'bites', 'dog']);
    for (const w of ['dog', 'bites', 'man']) {
      const x = a.find((t) => t.word === w)!.out;
      const y = b.find((t) => t.word === w)!.out;
      // identical in exact arithmetic; in floating point only rounding from the different order of addition remains
      x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(1e-14));
    }
  });
  it('with position vectors the outputs differ', () => {
    const a = selfAttention(['dog', 'bites', 'man'], true);
    const b = selfAttention(['man', 'bites', 'dog'], true);
    expect(a[0].out).not.toEqual(b[2].out);
  });
  it('sinusoidal agreement depends only on the offset; each hand is ~3.2× slower', () => {
    const sim = (p: number, q: number) => sinusoid(p, 16).reduce((s, v, i) => s + v * sinusoid(q, 16)[i], 0);
    expect(sim(3, 8)).toBeCloseTo(sim(30, 35), 10);
    expect(10000 ** (2 / 16)).toBeCloseTo(3.16, 2);
  });
  it('RoPE scores depend only on the offset', () => {
    expect(ropeScore([1, 0.5], [0.3, 1], 3, 5, 0.4)).toBeCloseTo(ropeScore([1, 0.5], [0.3, 1], 13, 15, 0.4), 12);
  });
  it('the lab: exhaustive results quoted in the debrief', () => {
    const [short, long, two] = LEVELS.map(solveLevel);
    expect(short.tried).toBe(180);
    expect(short.solutions.length).toBe(91);
    expect(short.best.speeds).toEqual([156]);
    expect(short.best.weight).toBeCloseTo(0.92, 2);
    expect(long.solutions.length).toBe(0);
    expect(long.best.weight).toBeCloseTo(0.15, 2);
    expect(two.tried).toBe(16290);
    expect(two.solutions.length).toBe(2689);
    expect(two.best.speeds).toEqual([5, 80]);
    expect(two.best.weight).toBeCloseTo(0.79, 2);
    expect(previousWordWeights([4, 36], 64)[1]).toBeGreaterThanOrEqual(0.6);
  });
});
