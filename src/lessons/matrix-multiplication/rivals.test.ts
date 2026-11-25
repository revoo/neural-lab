// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { batched, collapseCheck, collapsedSize, oneAtATime, perExample, rivals } from './rivals';

describe('matrix-multiplication: why batched multiplication, and not something else', () => {
  it('one at a time does the same arithmetic as 64 jobs instead of 2', () => {
    expect(perExample()).toBe(101632);
    expect(oneAtATime()).toEqual({ operations: 64, multiplyAdds: 3252224, weightReads: 3252224 });
    expect(batched()).toEqual({ operations: 2, multiplyAdds: 3252224, weightReads: 101632 });
  });

  it('a stack of plain matrices collapses into one; a ReLU between them breaks the collapse', () => {
    expect(collapsedSize()).toEqual({ stack: 101632, collapsed: 7840 });
    for (const seed of [1, 2, 3]) {
      const c = collapseCheck(seed);
      expect(c.linear).toBeLessThan(1e-12);
      expect(c.withRelu).toBeGreaterThan(0.5);
    }
    expect(collapseCheck(1).withRelu.toFixed(1)).toBe('3.2');
  });

  it('only batched multiplication passes all three tests', () => {
    const rs = rivals();
    expect(rs.filter((r) => r.verdicts.every((v) => v === 'pass')).map((r) => r.name)).toEqual(['Batched matrix multiplication']);
    expect(rs.every((r) => !/NaN|undefined/.test(r.result))).toBe(true);
    expect(rs[1].result).toContain('within 10⁻¹²');
  });
});
