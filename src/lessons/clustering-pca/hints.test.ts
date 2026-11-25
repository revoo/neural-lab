// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { unsupLabHint } from './unsupervised';

describe('clustering-pca lab hints', () => {
  it('points at evidence and never directly names k-Means++ or exact optimal settings', () => {
    const s = { b1Scaled: false, b1M: 1, b2Scaled: false, b2K: 4, b2Init: 'random' as const, b3M: 8, b3K: 4 };
    expect(unsupLabHint('pca-scale', s)).not.toMatch(/m = 2|Standardised/i);
    expect(unsupLabHint('segment', { ...s, b2Scaled: true })).not.toMatch(/k-Means\+\+/i);
    expect(unsupLabHint('denoise', s)).not.toMatch(/m = 2|k = 4/i);
  });
});
