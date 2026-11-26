// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { vllmSpecCommand } from './realCode';

describe('speculative-decoding realCode: verified vLLM --speculative-config', () => {
  it('maps draft model and bigram/ngram settings onto current JSON flags', () => {
    const gptCmd = vllmSpecCommand('small-gpt', 5, 0);
    expect(gptCmd).toContain('--speculative-config');
    expect(gptCmd).toContain('"num_speculative_tokens":5');
    expect(gptCmd).toContain('temperature=0');
    expect(gptCmd).not.toContain('--speculative-model');

    const bgCmd = vllmSpecCommand('bigram', 4, 1);
    expect(bgCmd).toContain('"method":"ngram"');
    expect(bgCmd).toContain('"num_speculative_tokens":4');
    expect(bgCmd).toContain('prompt_lookup_max');
  });
});
