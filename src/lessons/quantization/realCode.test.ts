// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { LLAMA_CPP_BLOCKS, llamaCppLines, nearestBlock } from './realCode';

describe('llama.cpp code line', () => {
  it('block sizes match llama.cpp quantize.cpp', () => {
    expect(Object.fromEntries(LLAMA_CPP_BLOCKS.map((b) => [b.name, b.bpw]))).toEqual({
      Q2_K: 2.625, Q3_K: 3.4375, Q4_K: 4.5, Q5_K: 5.5, Q6_K: 6.5625, Q8_0: 8.5, F16: 16,
    });
  });
  it('names the nearest block and the 8B file size', () => {
    expect(nearestBlock(4.25).name).toBe('Q4_K');
    expect(nearestBlock(5.75).name).toBe('Q5_K');
    expect(llamaCppLines(5.75)).toBe(
      'llama-quantize model-f16.gguf model-Q5_K_M.gguf Q5_K_M\n# your mix: 5.750 bits/weight → about 5.8 GB for Llama-3-8B\n# closest llama.cpp block format: Q5_K at 5.5 bits/weight',
    );
  });
});
