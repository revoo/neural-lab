// authored-by: Claude Opus 5.5
import type { TetrisConfig } from './kv';

const HF_IDS: Record<string, string> = {
  'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
  'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
};

/**
 * The vLLM launch command for a VRAM-Tetris setting. vLLM reads INT8/INT4
 * weight quantisation from the checkpoint itself (e.g. an AWQ or GPTQ model),
 * so those appear as a different model; its KV cache can be FP8 but has no
 * INT4 option (flags checked against vllm/engine/arg_utils.py, November 2026).
 */
export function vllmServeLines(modelId: string, cfg: TetrisConfig): string {
  const hf = HF_IDS[modelId] ?? modelId;
  const model =
    cfg.weightFmt === 'fp16' ? hf : cfg.weightFmt === 'int8' ? `<an INT8 (W8A16) checkpoint of ${hf}>` : `<a 4-bit AWQ or GPTQ checkpoint of ${hf}>`;
  const parts = [`vllm serve ${model}`];
  if (cfg.weightFmt === 'fp16') parts.push('  --dtype bfloat16');
  parts.push(`  --max-model-len ${cfg.ctxLen}`);
  parts.push(`  --max-num-seqs ${cfg.batchSize}`);
  if (cfg.kvFmt === 'fp8') parts.push('  --kv-cache-dtype fp8');
  const cmd = parts.join(' \\\n');
  return cfg.kvFmt === 'int4'
    ? `${cmd}\n# vLLM has no INT4 KV cache; llama.cpp's -ctk q4_0 -ctv q4_0 is the nearest equivalent`
    : cmd;
}
