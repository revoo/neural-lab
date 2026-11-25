// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { RULES, Z, rivals, shiftChange, slope, total } from './rivals';

const r2 = (p: number[]) => p.map((v) => +v.toFixed(2));

describe('probability: why softmax, and not something else', () => {
  it('divide by the sum gives a negative probability', () => {
    expect(r2(RULES.divide(Z))).toEqual([0.95, 0.48, 0.05, -0.48]);
  });

  it('clamping and argmax have dead slopes', () => {
    expect(r2(RULES.clamp(Z))).toEqual([0.65, 0.32, 0.03, 0]);
    expect(slope('clamp', 3, 3)).toBe(0);
    expect(RULES.argmax(Z)).toEqual([1, 0, 0, 0]);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) expect(slope('argmax', i, j)).toBe(0);
  });

  it('sigmoids do not add up to 1', () => {
    expect(r2(RULES.sigmoid(Z))).toEqual([0.88, 0.73, 0.52, 0.27]);
    expect(total(RULES.sigmoid(Z)).toFixed(2)).toBe('2.41');
  });

  it('softmax passes: positive, sums to 1, live slopes, shift-invariant', () => {
    const p = RULES.softmax(Z);
    expect(r2(p)).toEqual([0.64, 0.23, 0.1, 0.03]);
    expect(total(p)).toBeCloseTo(1, 12);
    expect(slope('softmax', 0, 0).toFixed(2)).toBe('0.23');
    expect(slope('softmax', 0, 0)).toBeCloseTo(p[0] * (1 - p[0]), 8);
    expect(shiftChange('softmax', 100)).toBeLessThan(1e-12);
    expect(shiftChange('divide', 100)).toBeGreaterThan(0.1);
  });

  it('only softmax passes all three tests', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['Softmax']);
    expect(rs.every((r) => !/NaN|undefined/.test(r.result))).toBe(true);
  });
});
