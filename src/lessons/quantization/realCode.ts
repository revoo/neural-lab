// authored-by: Claude Opus 5.5

/**
 * llama.cpp block formats and their exact bits per weight, from the block
 * layouts (e.g. Q4_K: 256 4-bit weights + 8 six-bit scales and mins + two FP16
 * super-scales = 1,152 bits / 256 = 4.5). These match the table in llama.cpp's
 * tools/quantize/quantize.cpp (checked November 2026).
 */
export const LLAMA_CPP_BLOCKS: { name: string; preset: string; bpw: number }[] = [
  { name: 'Q2_K', preset: 'Q2_K', bpw: (256 * 2 + 16 * 4 * 2 + 2 * 16) / 256 },
  { name: 'Q3_K', preset: 'Q3_K_M', bpw: (256 * 3 + 16 * 6 + 16) / 256 },
  { name: 'Q4_K', preset: 'Q4_K_M', bpw: (256 * 4 + 8 * 6 * 2 + 2 * 16) / 256 },
  { name: 'Q5_K', preset: 'Q5_K_M', bpw: (256 * 5 + 8 * 6 * 2 + 2 * 16) / 256 },
  { name: 'Q6_K', preset: 'Q6_K', bpw: (256 * 6 + 16 * 8 + 16) / 256 },
  { name: 'Q8_0', preset: 'Q8_0', bpw: (32 * 8 + 16) / 32 },
  { name: 'F16', preset: 'F16', bpw: 16 },
];

/** Parameters in Llama-3-8B (Meta's model card: 8.03B). */
export const LLAMA3_8B_PARAMS = 8.03e9;

export function nearestBlock(avgBits: number) {
  return LLAMA_CPP_BLOCKS.reduce((best, b) => (Math.abs(b.bpw - avgBits) < Math.abs(best.bpw - avgBits) ? b : best));
}

export function llamaCppLines(avgBits: number): string {
  const b = nearestBlock(avgBits);
  const gb = (avgBits * LLAMA3_8B_PARAMS) / 8 / 1e9;
  return [
    `llama-quantize model-f16.gguf model-${b.preset}.gguf ${b.preset}`,
    `# your mix: ${avgBits.toFixed(3)} bits/weight → about ${gb.toFixed(1)} GB for Llama-3-8B`,
    `# closest llama.cpp block format: ${b.name} at ${b.bpw.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')} bits/weight`,
  ].join('\n');
}
