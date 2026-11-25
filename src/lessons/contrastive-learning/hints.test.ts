// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { BATCH_OPTIONS, LAB_TAUS_B1, LAB_TAUS_B2, contrastLabHint } from './contrastive';

describe('contrastive-learning: Lab hints stay evidence-only', () => {
  it('never names winning buttons or exact optimal settings before passing', () => {
    const base = { b1Negs: false, b1Tau: 0.15, b2Tau: 1.0, b3Opt: 'simclr-64' as const };

    for (const b1Negs of [false, true]) {
      for (const b1Tau of LAB_TAUS_B1) {
        const h = contrastLabHint('circle', { ...base, b1Negs, b1Tau });
        expect(h.length).toBeGreaterThan(20);
        expect(h).not.toMatch(/turn on infonce|pick 0\.15|select τ = 0\.15/i);
      }
    }

    for (const b2Tau of LAB_TAUS_B2) {
      const h = contrastLabHint('temp', { ...base, b2Tau });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/pick 0\.05|select 0\.05|click 0\.05/i);
    }

    for (const opt of BATCH_OPTIONS) {
      const h = contrastLabHint('batch', { ...base, b3Opt: opt.value });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/pick moco|click moco|select simclr b=512/i);
    }
  });
});
