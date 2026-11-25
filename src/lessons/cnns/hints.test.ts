// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { HEAD_CHOICES, SHIFT_CHOICES, cnnLabHint } from './cnn';

describe('cnns: Lab hints stay evidence-only', () => {
  it('never names winning buttons or exact optimal settings before a pass', () => {
    const base = {
      k: 5 as const,
      nConv: 4 as const,
      nPools: 0 as const,
      shiftChoice: 'dense-8' as const,
      headChoice: 'vgg-fc' as const,
    };

    for (const k of [3, 5] as const) {
      for (const nConv of [2, 3, 4] as const) {
        for (const nPools of [0, 1, 2] as const) {
          const h = cnnLabHint('rf-budget', { ...base, k, nConv, nPools });
          expect(h.length).toBeGreaterThan(20);
          expect(h).not.toMatch(/pick 3×3|select 2 pools|set conv/i);
        }
      }
    }

    for (const sc of SHIFT_CHOICES) {
      const h = cnnLabHint('shift-gen', { ...base, shiftChoice: sc.value });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/pick cnn|select 4 filters|switch to cnn/i);
    }

    for (const hc of HEAD_CHOICES) {
      const h = cnnLabHint('gap-head', { ...base, headChoice: hc.value });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/pick global avg|click gap|choose resnet/i);
    }
  });
});
