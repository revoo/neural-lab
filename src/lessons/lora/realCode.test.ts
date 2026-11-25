// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { peftLoraLines } from './realCode';

describe('peft code line', () => {
  it('writes rank_pattern for uneven ranks and drops rank-0 layers', () => {
    expect(peftLoraLines([0, 3, 2])).toBe(
      'LoraConfig(\n    target_modules=["mid", "head"],\n    r=3,\n    rank_pattern={"head": 2},\n)\n# early: rank 0, left frozen',
    );
    expect(peftLoraLines([1, 1, 1])).toBe('LoraConfig(\n    target_modules=["early", "mid", "head"],\n    r=1,\n)');
    expect(peftLoraLines([0, 0, 0])).toMatch(/nothing is adapted/);
  });
});
