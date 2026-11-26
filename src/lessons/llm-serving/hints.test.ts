// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { DESK_BRIEFS, deskHint, evaluateDesk } from './lab';
import type { DeskConfig } from './realCode';

const BANNED_DESK_PHRASES = [
  'continuous',
  'static',
  'paged',
  'prefix caching',
  'prefix sharing',
  'chunked',
  'token budget',
  'sjf',
  'fcfs',
  'try ',
  'switch to',
  'turn on',
  'set ',
  'select ',
  'choose ',
];

describe('llm-serving hints: strictly evidence-only (never name settings or buttons)', () => {
  it('The Serving Desk hints point only at observed evidence across failing and passing runs', () => {
    const sampleConfigs: DeskConfig[] = [
      { batching: 'static', maxRunning: 8, tokenBudget: null, memoryMode: 'reserve-max', admission: 'fcfs' },
      { batching: 'continuous', maxRunning: 16, tokenBudget: null, memoryMode: 'paged', admission: 'fcfs' },
      { batching: 'continuous', maxRunning: 16, tokenBudget: 512, memoryMode: 'reserve-max', admission: 'fcfs' },
      { batching: 'continuous', maxRunning: 64, tokenBudget: 512, memoryMode: 'paged', admission: 'fcfs' },
      { batching: 'continuous', maxRunning: 4, tokenBudget: 256, memoryMode: 'paged-prefix', admission: 'fcfs' },
      { batching: 'continuous', maxRunning: 64, tokenBudget: 256, memoryMode: 'paged-prefix', admission: 'fcfs' },
    ];

    for (const brief of DESK_BRIEFS) {
      for (const cfg of sampleConfigs) {
        const { sim } = evaluateDesk(brief, cfg);
        const h = deskHint(brief, sim).toLowerCase();
        expect(h.length).toBeGreaterThan(20);
        for (const banned of BANNED_DESK_PHRASES) {
          expect(h, `${brief.id} hint leaked "${banned}": ${h}`).not.toContain(banned);
        }
      }
    }
  });
});
