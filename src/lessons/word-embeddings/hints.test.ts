// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_DIMS_B2, LAB_KS_B1, LAB_SCALE_MODES, LAB_WINS_B2, w2vLabHint } from './word2vec';

describe('word-embeddings: Lab hints stay evidence-only', () => {
  it('never names winning buttons or exact optimal settings before passing', () => {
    const base = { b1K: 0, b2Dim: 2, b2Win: 2, b3Mode: 'full-softmax' as const };

    for (const k of LAB_KS_B1) {
      const h = w2vLabHint('collapse', { ...base, b1K: k });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/click k =|pick k=|select k =/i);
    }

    for (const dim of LAB_DIMS_B2) {
      for (const win of LAB_WINS_B2) {
        const h = w2vLabHint('analogies', { ...base, b2Dim: dim, b2Win: win });
        expect(h.length).toBeGreaterThan(20);
        expect(h).not.toMatch(/pick 3d|select window 3|choose d=3/i);
      }
    }

    for (const mode of LAB_SCALE_MODES) {
      const h = w2vLabHint('scale', { ...base, b3Mode: mode });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/click sgns k=5|select sgns-5/i);
    }
  });
});
