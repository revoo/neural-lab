// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS, pipelineLabHint, type PipelineConfig } from './reliability';

const LEAK_PATTERNS = [
  /7\.69/,
  /7\.72/,
  /9\.90/,
  /Strong \+ guard/i,
  /Strong on Policy/i,
  /set\s+retries/i,
];

const SAMPLE_CFGS: PipelineConfig[] = [
  [
    { model: 'fast', check: false, retries: 0 },
    { model: 'fast', check: false, retries: 0 },
    { model: 'fast', check: false, retries: 0 },
    { model: 'fast', check: false, retries: 0 },
  ],
  [
    { model: 'fast', check: true, retries: 1 },
    { model: 'fast', check: true, retries: 0 },
    { model: 'fast', check: true, retries: 1 },
    { model: 'fast', check: true, retries: 1 },
  ],
  [
    { model: 'strong', check: true, retries: 2 },
    { model: 'strong', check: true, retries: 2 },
    { model: 'strong', check: true, retries: 2 },
    { model: 'strong', check: true, retries: 2 },
  ],
];

describe('tool-calling lab evidence-only hints', () => {
  it('never names winning configs or optimum credit totals across briefs', () => {
    for (const b of LAB_BRIEFS) {
      for (const cfg of SAMPLE_CFGS) {
        const h = pipelineLabHint(b, cfg);
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
  });
});
