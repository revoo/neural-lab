// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { EMAIL, JUDGES, layerMultiplyAdds, loopOfDots, matrixTimes, rivals, tableRows, tableRowsLog10 } from './rivals';

describe('matrices: why a matrix, and not something else', () => {
  it('the loop of dot products gives the same scores as the matrix, as separate jobs', () => {
    const loop = loopOfDots(JUDGES, EMAIL);
    expect(loop.out).toEqual([14, 5, -4]);
    expect(matrixTimes(JUDGES, EMAIL)).toEqual(loop.out);
    expect(loop).toMatchObject({ operations: 3, multiplyAdds: 9 });
    expect(layerMultiplyAdds()).toBe(100352);
  });

  it('a lookup table explodes', () => {
    expect(tableRows(3, 10)).toBe(1000);
    expect(Math.round(tableRowsLog10(784, 256))).toBe(1888);
  });

  it('the matrix handles an email it has never seen', () => {
    expect(matrixTimes(JUDGES, [12, 0, 1])).toEqual([9, 4, -10]);
  });

  it('only the matrix passes all three tests', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['A matrix']);
    expect(rs.every((r) => !/NaN|undefined/.test(r.result))).toBe(true);
    expect(rs[0].result).toContain('100,352');
    expect(rs[1].result).toContain('3,000 numbers');
  });
});
