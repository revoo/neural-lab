// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAB_ALIGN_KINDS,
  LAB_ALIGN_TAUS,
  LAB_BN_ARCHS,
  LAB_BN_DS,
  LAB_LAT_ARCHS,
  seqLabHint,
} from './seq2seq';

const LEAK_PATTERNS = [
  /pick\s+cross-attn/i,
  /choose\s+luong/i,
  /set\s+τ\s*=\s*0\.15/i,
  /tau\s*=\s*0\.15/i,
  /pick\s+transformer/i,
  /d\s*=\s*8\b/,
];

describe('seq2seq-attention lab evidence-only hints', () => {
  it('never names winning settings across all configs of all 3 briefs', () => {
    for (const arch of LAB_BN_ARCHS) {
      for (const d of LAB_BN_DS) {
        const h = seqLabHint('bottleneck', { b1Arch: arch, b1D: d, b2Kind: 'bahdanau', b2Tau: 1.0, b3Arch: 'lstm-attn' });
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
    for (const kind of LAB_ALIGN_KINDS) {
      for (const tau of LAB_ALIGN_TAUS) {
        const h = seqLabHint('align', { b1Arch: 'fixed', b1D: 16, b2Kind: kind, b2Tau: tau, b3Arch: 'lstm-attn' });
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
    for (const arch of LAB_LAT_ARCHS) {
      const h = seqLabHint('latency', { b1Arch: 'fixed', b1D: 16, b2Kind: 'bahdanau', b2Tau: 1.0, b3Arch: arch });
      expect(h.length).toBeGreaterThan(20);
      for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
    }
  });
});
