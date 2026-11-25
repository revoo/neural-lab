// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { hfGenerateLine } from './realCode';

describe('generate code line', () => {
  it('turns off top-k at the full toy vocabulary', () => {
    expect(hfGenerateLine(0.7, 16, 0.9)).toBe('model.generate(**inputs, do_sample=True, temperature=0.7, top_k=0, top_p=0.90)');
    expect(hfGenerateLine(1, 5, 1)).toBe('model.generate(**inputs, do_sample=True, temperature=1.0, top_k=5, top_p=1.00)');
  });
});
