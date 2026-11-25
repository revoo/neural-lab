// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import {
  BIG_BEST, LR_GRID, STEP_GRID, bigCoordinate, bigGradient, bigRace, bigRandom, bowl, grad, race, randomAlignment, rivals,
} from './rivals';

const argmin = (grid: number[], f: (v: number) => number) => grid.reduce((b, v) => (f(v) < f(b) ? v : b), grid[0]);

describe('gradients: why the gradient, and not something else', () => {
  it('the hand-written gradient of the tilted bowl matches central differences', () => {
    const h = 1e-6;
    for (const [x, y] of [[0.7, -1.3], [-3, 2], [2, 0.5]]) {
      const [gx, gy] = grad(x, y);
      expect(gx).toBeCloseTo((bowl(x + h, y) - bowl(x - h, y)) / (2 * h), 6);
      expect(gy).toBeCloseTo((bowl(x, y + h) - bowl(x, y - h)) / (2 * h), 6);
    }
  });

  it('two knobs: with each method at its best step size, one knob at a time wins', () => {
    const r = race();
    expect(r.start.toFixed(1)).toBe('21.4');
    expect(r.coordinate.setting).toBe(0.77);
    expect(r.coordinate.loss.toExponential(1)).toBe('4.3e-40');
    expect(r.gradient.setting).toBe(0.43);
    expect(r.gradient.loss.toExponential(1)).toBe('3.1e-4');
    expect(r.random.setting).toBe(0.4);
    expect(r.random.loss.toFixed(3)).toBe('0.014');
    expect(r.coordinate.loss).toBeLessThan(r.gradient.loss);
  });

  it('100 knobs: the recorded best settings are the best on their grids, and the gradient wins by far', () => {
    expect(argmin(LR_GRID, bigGradient)).toBe(BIG_BEST.gradient);
    expect(argmin(LR_GRID, bigCoordinate)).toBe(BIG_BEST.coordinate);
    expect(argmin(STEP_GRID, bigRandom)).toBe(BIG_BEST.random);
    const b = bigRace();
    expect(Math.round(b.coordinate * 100)).toBe(65);
    expect(Math.round(b.random * 100)).toBe(79);
    expect((b.gradient * 100).toPrecision(1)).toBe('0.0003');
  }, 30000);

  it('random directions line up worse and worse with downhill as knobs multiply', () => {
    expect(randomAlignment(2)).toBeCloseTo(2 / Math.PI, 6);
    expect(randomAlignment(3)).toBeCloseTo(0.5, 6);
    expect(randomAlignment(100).toFixed(2)).toBe('0.08');
    expect(randomAlignment(1e6).toFixed(4)).toBe('0.0008');
  });

  it('only the gradient passes all three tests, and the cards quote the pinned numbers', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['The gradient']);
    expect(rs[0].result).toContain('4.3 × 10⁻⁴⁰');
    expect(rs[0].result).toContain('65% of the loss');
    expect(rs[2].result).toContain('0.0003% of the loss');
    expect(rs.every((r) => !/NaN|undefined/.test(r.result))).toBe(true);
  });
});
