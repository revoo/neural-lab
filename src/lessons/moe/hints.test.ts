// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { BRIEFS, labHint } from './lab';
import { LAB_RESULTS } from './results';

// a hint may describe what the measurements show, never a value to set
const LEAKS = [/α\s*=/, /alpha/i, /C\s*=\s*\d/, /top-\d/i, /noise/i, /σ/, /set\s+the/i, /increase|decrease|raise|lower/i];

describe('MoE lab hints are evidence only', () => {
  it('never names a setting, for every recorded row and brief', () => {
    for (const b of BRIEFS)
      for (const r of LAB_RESULTS) {
        const h = labHint(b.id, { ...r, cost: r.k * r.capacity });
        expect(h.length).toBeGreaterThan(20);
        for (const p of LEAKS) expect(h, `${b.id}: ${h}`).not.toMatch(p);
      }
  });
});
