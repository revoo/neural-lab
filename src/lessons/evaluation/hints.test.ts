// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { GRADERS, LAB_ALPHAS, LAB_BRIEFS, LAB_MODES, LAB_NS, evalLabHint, evaluateEvalDesign } from './eval';

const LEAK_PATTERNS = [
  /\bpaired\b/i,
  /\bunpaired\b/i,
  /\bmcnemar\b/i,
  /\bfast llm\b/i,
  /\bdual[- ]?pass\b/i,
  /\bdual llm\b/i,
  /\bexact human\b/i,
  /\bhuman\b/i,
  /\b1,200\b/,
  /\b1,500\b/,
  /\b7,500\b/,
];

describe('evaluation lab hints are evidence-only', () => {
  it('never names winning protocols, graders, or costs across all briefs and designs', () => {
    for (const brief of LAB_BRIEFS) {
      const ns = brief.maxN !== undefined ? LAB_NS.filter((n) => n <= brief.maxN!) : [...LAB_NS];
      for (const mode of LAB_MODES) {
        for (const grader of GRADERS.map((g) => g.id)) {
          for (const n of ns) {
            for (const alpha of LAB_ALPHAS) {
              const d = { mode, grader, n, alpha };
              const ev = evaluateEvalDesign(d, brief);
              const h = evalLabHint(brief, d, ev);
              expect(h.length).toBeGreaterThan(20);
              const isOptimal =
                ev.power >= brief.minPower - 1e-9 &&
                ev.fpr <= brief.maxFpr + 1e-9 &&
                (ev.cost === 1200 || ev.cost === 1500 || ev.cost === 7500);
              for (const pat of LEAK_PATTERNS) {
                if (isOptimal && /1,200|1,500|7,500/.test(pat.source)) continue;
                expect(h, `${brief.id} ${JSON.stringify(d)} -> ${h}`).not.toMatch(pat);
              }
            }
          }
        }
      }
    }
  });
});
