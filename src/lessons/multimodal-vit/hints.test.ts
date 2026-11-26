// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { vitHint } from './Lab';

const LEAK = [/τ\s*=/, /\b(0\.05|0\.1|0\.3)\b/, /batch (of|size)/i, /augment|shift/i, /position embed/i, /patch size|2\s*×\s*2|4\s*×\s*4/i, /set\s+/i, /width \d/i];

describe('vision lab hints are evidence only', () => {
  it('never names a setting', () => {
    const cases = [
      ['patch', { brief: 'patch', key: 'k', accs: [0.6, 0.7, 0.65], cost: 256 }],
      ['patch', { brief: 'patch', key: 'k', accs: [], cost: 16384, over: true }],
      ['patch', { brief: 'patch', key: 'k', accs: [0.85, 0.86, 0.8], cost: 2048 }],
      ['rescue', { brief: 'rescue', key: 'k', accs: [0.55, 0.66, 0.57] }],
      ['rescue', { brief: 'rescue', key: 'k', accs: [0.85, 0.86, 0.84] }],
      ['clip', { brief: 'clip', key: 'k', accs: [0.3, 0.4, 0.35] }],
      ['clip', { brief: 'clip', key: 'k', accs: [0.93, 0.94, 0.92] }],
    ] as const;
    for (const [b, r] of cases) {
      const h = vitHint(b, r as never);
      expect(h.length).toBeGreaterThan(20);
      for (const p of LEAK) expect(h, h).not.toMatch(p);
    }
    expect(vitHint('clip', null)).toMatch(/Train/);
  });
});
