// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAB_CLIP_LRS,
  LAB_CLIP_NORMS,
  LAB_FLOW_ACTS,
  LAB_FLOW_RHOS,
  LAB_TBPTT_HS,
  LAB_TBPTT_KS,
  rnnLabHint,
} from './rnn';

const LEAK_PATTERNS = [
  /set\s+rho/i,
  /pick\s+linear/i,
  /choose\s+tanh/i,
  /set\s+clip/i,
  /set\s+K\s*=\s*7/i,
  /H\s*=\s*2/i,
  /clip\s*=\s*1/i,
  /ρ\s*=\s*1(\.0)?\b/,
];

describe('rnns lab evidence-only hints', () => {
  it('never names winning settings across all configs of all 3 briefs', () => {
    for (const act of LAB_FLOW_ACTS) {
      for (const rho of LAB_FLOW_RHOS) {
        const h = rnnLabHint('flow', { b1Act: act, b1Rho: rho, b2Lr: 0.025, b2Clip: null, b3K: 2, b3H: 4 });
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
    for (const lr of LAB_CLIP_LRS) {
      for (const clip of LAB_CLIP_NORMS) {
        const h = rnnLabHint('clip', { b1Act: 'tanh', b1Rho: 0.5, b2Lr: lr, b2Clip: clip, b3K: 2, b3H: 4 });
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
    for (const K of LAB_TBPTT_KS) {
      for (const H of LAB_TBPTT_HS) {
        const h = rnnLabHint('tbptt', { b1Act: 'tanh', b1Rho: 0.5, b2Lr: 0.025, b2Clip: null, b3K: K, b3H: H });
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
  });
});
