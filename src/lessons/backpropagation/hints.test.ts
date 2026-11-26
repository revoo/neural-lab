// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  brief1Hint,
  brief2Hint,
  brief3Hint,
  checkBrief1,
  evaluateBrief2,
  evaluateBrief3,
  type Brief2Config,
  type Brief3Config,
} from './creditLab';

const BANNED_PATTERNS: readonly RegExp[] = [
  /\btry\b/i,
  /\bswitch to\b/i,
  /\bturn on\b/i,
  /\bturn off\b/i,
  /\bunfreeze\b/i,
  /\bfreeze\b/i,
  /\bclick\b/i,
  /\bselect\b/i,
  /\bchoose\b/i,
  /\bset\b/i,
  /\basym-small\b/i,
  /\basym-large\b/i,
  /\bxavier\b/i,
  /\btanh\b/i,
  /\bsigmoid\b/i,
  /\bl\s*r\s*=/i,
  /\blearning rate [0-9]/i,
  /\bincrease\b/i,
  /\bdecrease\b/i,
];

function assertEvidenceOnly(label: string, hint: string) {
  expect(hint.length).toBeGreaterThan(25);
  for (const pat of BANNED_PATTERNS) {
    expect(hint, `${label} leaked ${pat}: ${hint}`).not.toMatch(pat);
  }
}

describe('Credit-Assignment Workbench hints: strictly evidence-only', () => {
  it('Brief 1 hints point only at forward/backward evidence and never name an answer or button', () => {
    const states = [
      null,
      checkBrief1({ v1: '-', w11: '+', w21: '-', w12: '+' }, 'v1'),
      checkBrief1({ v1: '+', w11: '+', w21: '+', w12: '0' }, 'v1'),
      checkBrief1({ v1: '-', w11: '-', w21: '+', w12: '0' }, 'v1'),
      checkBrief1({ v1: '+', w11: '-', w21: '+', w12: '0' }, 'v1'),
      checkBrief1({ v1: '+', w11: '-', w21: '+', w12: '0' }, 'w11'),
    ];
    for (const st of states) {
      assertEvidenceOnly('Brief 1 hint', brief1Hint(st));
    }
  });

  it('Brief 2 hints point only at measured telemetry across failing and passing configs', () => {
    const samples: Brief2Config[] = [
      { initMode: 'broken', act: 'tanh', lr: 1.0 },
      { initMode: 'zeros', act: 'relu', lr: 1.0 },
      { initMode: 'asym-large', act: 'tanh', lr: 1.0 },
      { initMode: 'asym-small', act: 'sigmoid', lr: 0.2 },
      { initMode: 'asym-small', act: 'tanh', lr: 1.0 },
      { initMode: 'asym-large', act: 'relu', lr: 1.0 },
    ];
    for (const cfg of samples) {
      assertEvidenceOnly('Brief 2 hint', brief2Hint(evaluateBrief2(cfg)));
    }
  });

  it('Brief 3 hints point only at hidden-space geometry and metrics across failing and passing configs', () => {
    const samples: Brief3Config[] = [
      { trainH1: false, trainH2: false, trainOut: true, lr: 1.5 },
      { trainH1: true, trainH2: false, trainOut: false, lr: 1.5 },
      { trainH1: true, trainH2: false, trainOut: true, lr: 0.4 },
      { trainH1: false, trainH2: true, trainOut: true, lr: 2.5 },
      { trainH1: true, trainH2: true, trainOut: true, lr: 1.5 },
    ];
    for (const cfg of samples) {
      assertEvidenceOnly('Brief 3 hint', brief3Hint(evaluateBrief3(cfg)));
    }
  });
});
