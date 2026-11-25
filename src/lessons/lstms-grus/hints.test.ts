// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  LAB_BIAS_CELLS,
  LAB_BIAS_VALS,
  LAB_BUDGET_HS,
  LAB_BUDGET_KINDS,
  LAB_GATE_LEVELS,
  lstmLabHint,
} from './gated';

const LEAK_PATTERNS = [
  /set\s+f\s*=\s*0\.95/i,
  /set\s+i\s*=\s*0\.05/i,
  /set\s+o\s*=\s*0\.05/i,
  /b_f\s*=\s*\+?[23]\b/i,
  /pick\s+gru/i,
  /choose\s+gru/i,
  /H\s*=\s*16\b/,
];

describe('lstms-grus lab evidence-only hints', () => {
  it('never names winning settings across all configs of all 3 briefs', () => {
    for (const f of LAB_GATE_LEVELS) {
      for (const i of LAB_GATE_LEVELS) {
        for (const o of LAB_GATE_LEVELS) {
          const h = lstmLabHint('gates', { b1F: f, b1I: i, b1O: o, b2Cell: 'rnn', b2Bias: 0, b3Kind: 'lstm', b3H: 16 });
          expect(h.length).toBeGreaterThan(20);
          for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
        }
      }
    }
    for (const cell of LAB_BIAS_CELLS) {
      for (const bias of LAB_BIAS_VALS) {
        const h = lstmLabHint('bias', { b1F: 0.5, b1I: 0.5, b1O: 0.5, b2Cell: cell, b2Bias: bias, b3Kind: 'lstm', b3H: 16 });
        expect(h.length).toBeGreaterThan(20);
        for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
      }
    }
    for (const kind of LAB_BUDGET_KINDS) {
      for (const H of LAB_BUDGET_HS) {
        const h = lstmLabHint('budget', { b1F: 0.5, b1I: 0.5, b1O: 0.5, b2Cell: 'rnn', b2Bias: 0, b3Kind: kind, b3H: H });
        expect(h.length).toBeGreaterThan(20);
        // Only check non-passing hints for H=16 / GRU leaks
        if (!(kind === 'gru' && H === 16)) {
          for (const pat of LEAK_PATTERNS) expect(h).not.toMatch(pat);
        }
      }
    }
  });
});
