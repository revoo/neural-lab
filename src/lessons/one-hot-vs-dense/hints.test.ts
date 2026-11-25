// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_DIMS_B2, LAB_DTYPES_B2, LAB_SKU_MODES, embedLabHint, type RepKind } from './embed';

describe('one-hot-vs-dense: Lab hints stay evidence-only', () => {
  it('never names winning buttons or exact settings before passing', () => {
    const base = { rep: 'onehot' as RepKind, b2Dim: 128, b2Bytes: 4 as const, sku: 'onehot' as const };

    for (const rep of ['onehot', 'scalar-id', 'dense-2', 'dense-4'] as RepKind[]) {
      const h = embedLabHint('transfer', { ...base, rep });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/pick dense 2d|click dense|choose dense-2/i);
    }

    for (const d of LAB_DIMS_B2) {
      for (const dt of LAB_DTYPES_B2) {
        const h = embedLabHint('edge-ram', { ...base, b2Dim: d, b2Bytes: dt.value });
        expect(h.length).toBeGreaterThan(20);
        expect(h).not.toMatch(/pick int8|select 128d|choose int8/i);
      }
    }

    for (const sku of LAB_SKU_MODES) {
      const h = embedLabHint('sku-mlp', { ...base, sku });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/pick emb 16|click emb-16|select 16d/i);
    }
  });
});
