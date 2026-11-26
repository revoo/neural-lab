// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { vllmDeskCommand, vllmSpecCommand } from './realCode';

describe('llm-serving realCode: verified vLLM CLI commands', () => {
  it('maps Lab 1 continuous + chunked + paged + prefix caching to clean vllm serve flags', () => {
    const cmd = vllmDeskCommand(
      {
        batching: 'continuous',
        maxRunning: 32,
        tokenBudget: 512,
        memoryMode: 'paged-prefix',
        admission: 'fcfs',
      },
      8192,
    );
    expect(cmd).toContain('vllm serve meta-llama/Meta-Llama-3-8B-Instruct');
    expect(cmd).toContain('--max-model-len 8192');
    expect(cmd).toContain('--max-num-seqs 32');
    expect(cmd).toContain('--enable-chunked-prefill');
    expect(cmd).toContain('--max-num-batched-tokens 512');
    expect(cmd).toContain('--block-size 16');
    expect(cmd).toContain('--enable-prefix-caching');
    expect(cmd).not.toContain('# Note:');
  });

  it('notes un-representable static/contiguous/sjf settings honestly instead of inventing flags', () => {
    const cmd = vllmDeskCommand(
      {
        batching: 'static',
        maxRunning: 8,
        tokenBudget: null,
        memoryMode: 'reserve-max',
        admission: 'sjf',
      },
      4096,
    );
    expect(cmd).toContain('--no-enable-chunked-prefill');
    expect(cmd).toContain('--no-enable-prefix-caching');
    expect(cmd).toContain('always uses continuous batching');
    expect(cmd).toContain('always pages KV memory');
    expect(cmd).toContain('FCFS by default');
  });

  it('maps Lab 2 speculative decoding onto current --speculative-config JSON', () => {
    const gptCmd = vllmSpecCommand('small-gpt', 5, 0);
    expect(gptCmd).toContain('--speculative-config');
    expect(gptCmd).toContain('"num_speculative_tokens":5');
    expect(gptCmd).toContain('temperature=0');
    expect(gptCmd).not.toContain('--speculative-model');

    const bgCmd = vllmSpecCommand('bigram', 3, 1);
    expect(bgCmd).toContain('"method":"ngram"');
    expect(bgCmd).toContain('"num_speculative_tokens":3');
    expect(bgCmd).toContain('prompt_lookup_max');
  });
});
