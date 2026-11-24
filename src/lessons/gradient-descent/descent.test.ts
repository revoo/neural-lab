import { describe, expect, it } from 'vitest';
import { CURVES, bowlRegime, par, runValley } from './descent';
import { derivative } from '../../lib/calculus';

describe('gradient descent lesson core', () => {
  it('curve derivatives are correct', () => {
    for (const c of Object.values(CURVES)) for (const w of [-1, 0.5, 2.7, 6]) expect(c.df(w)).toBeCloseTo(derivative(c.f, w), 5);
  });
  it('classifies the learning-rate regimes of the bowl', () => {
    expect(bowlRegime(0.01)).toBe('slow');
    expect(bowlRegime(0.3)).toBe('smooth');
    expect(bowlRegime(0.9)).toBe('oscillating');
    expect(bowlRegime(1.0)).toBe('bouncing');
    expect(bowlRegime(1.1)).toBe('diverging');
  });
  it('the regime matches actual simulation', () => {
    const sim = (lr: number) => {
      let w = -1;
      for (let i = 0; i < 50; i++) w -= lr * CURVES.bowl.df(w);
      return Math.abs(w - 3);
    };
    expect(sim(0.3)).toBeLessThan(1e-6);
    expect(sim(0.9)).toBeLessThan(0.01);
    expect(sim(1.0)).toBeCloseTo(4);
    expect(sim(1.1)).toBeGreaterThan(100);
  });
  it('valley runs reach the target, diverge when the step is too large, and momentum lowers par', () => {
    expect(runValley([-9, 6], 1.0).reached).toBe(true);
    expect(runValley([-9, 6], 2.1).diverged).toBe(true);
    const plain = par([-9, 6], false);
    const mom = par([-9, 6], true);
    expect(plain.steps).toBeLessThan(400);
    expect(mom.steps).toBeLessThan(plain.steps);
  });
});
