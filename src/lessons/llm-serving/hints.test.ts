// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { DESK_BRIEFS, SPEC_BRIEFS, deskHint, evaluateDesk, specHint } from './lab';
import type { DeskConfig } from './realCode';
import { SPEC_DRAFTERS, SPEC_KS } from './spec.run';

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

const BANNED_SPEC_PHRASES = [
  'bigram',
  'small gpt',
  'small-gpt',
  'k =',
  'k=',
  'try ',
  'switch to',
  'pick ',
  'select ',
  'choose ',
];

describe('llm-serving hints: strictly evidence-only (never name settings or buttons)', () => {
  it('Lab 1 (The Serving Desk) hints point only at observed evidence across failing and passing runs', () => {
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

  it('Lab 2 (Tune the draft) hints never name a drafter or k value across all 48 cells', () => {
    for (const brief of SPEC_BRIEFS) {
      for (const d of SPEC_DRAFTERS) {
        for (const k of SPEC_KS) {
          const h = specHint(brief, d, k).toLowerCase();
          expect(h.length).toBeGreaterThan(20);
          for (const banned of BANNED_SPEC_PHRASES) {
            expect(h, `${brief.id} (${d}, ${k}) hint leaked "${banned}": ${h}`).not.toContain(banned);
          }
        }
      }
    }
  });
});
