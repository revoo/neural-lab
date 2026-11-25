// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { COMP_SPECS, LAB_LSH_BITS, LAB_LSH_TABLES, LAB_NLISTS, LAB_NPROBES, annLabHint } from './ann';

describe('similarity-search: Lab hints stay evidence-only', () => {
  it('never names winning buttons or exact optimal settings before passing', () => {
    const base = { nlist: 8, nprobe: 1, bits: 8, tables: 1, fmt: 'fp32' as const };

    for (const nlist of LAB_NLISTS) {
      for (const nprobe of LAB_NPROBES) {
        const h = annLabHint('ivf', { ...base, nlist, nprobe });
        expect(h.length).toBeGreaterThan(20);
        expect(h).not.toMatch(/set nlist=32|pick 32|select nprobe=6/i);
      }
    }

    for (const bits of LAB_LSH_BITS) {
      for (const tables of LAB_LSH_TABLES) {
        const h = annLabHint('lsh', { ...base, bits, tables });
        expect(h.length).toBeGreaterThan(20);
        expect(h).not.toMatch(/pick 6 bits|select 16 tables|choose b=6/i);
      }
    }

    for (const s of COMP_SPECS) {
      const h = annLabHint('ram', { ...base, fmt: s.id });
      expect(h.length).toBeGreaterThan(20);
      expect(h).not.toMatch(/pick scalar int8|click int8|select int8/i);
    }
  });
});
