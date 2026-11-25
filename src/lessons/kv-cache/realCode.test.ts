// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { vllmServeLines } from './realCode';

describe('vLLM code line', () => {
  it('maps settings to flags', () => {
    expect(vllmServeLines('llama3-8b', { weightFmt: 'fp16', kvFmt: 'fp8', batchSize: 16, ctxLen: 32768 })).toBe(
      'vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\\n  --dtype bfloat16 \\\n  --max-model-len 32768 \\\n  --max-num-seqs 16 \\\n  --kv-cache-dtype fp8',
    );
    const int4 = vllmServeLines('llama3-70b', { weightFmt: 'int4', kvFmt: 'int4', batchSize: 8, ctxLen: 16384 });
    expect(int4).toContain('<a 4-bit AWQ or GPTQ checkpoint of meta-llama/Meta-Llama-3-70B-Instruct>');
    expect(int4).toContain('no INT4 KV cache');
    expect(int4).not.toContain('--kv-cache-dtype');
  });
});
