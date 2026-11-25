// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { trlConfigLine } from './realCode';

describe('TRL code line', () => {
  it('maps each method to its TRL config', () => {
    expect(trlConfigLine({ method: 'dpo', beta: 0.25, steps: 100 }, true)).toBe('DPOConfig(beta=0.25, max_steps=100)');
    expect(trlConfigLine({ method: 'sft', beta: 0.5, steps: 10 }, true)).toMatch(/^SFTConfig\(max_steps=10\)/);
    // Brief 1 locks the method to RLHF whatever the stored config says
    expect(trlConfigLine({ method: 'dpo', beta: 1, steps: 100 }, false)).toMatch(/^RLOOConfig\(beta=1\)/);
  });
});
