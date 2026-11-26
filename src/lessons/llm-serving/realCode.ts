// authored-by: Gemini Antigravity
/**
 * Maps learner settings in The Serving Desk lab to real `vllm serve` CLI commands.
 * Every flag was checked against vLLM's stable docs (`docs.vllm.ai/en/stable/`, Nov 2026):
 *   --dtype, --max-model-len, --max-num-seqs, --enable-chunked-prefill / --no-enable-chunked-prefill,
 *   --max-num-batched-tokens, --block-size, --enable-prefix-caching / --no-enable-prefix-caching.
 */
import type { AdmissionPolicy, BatchingMode } from './serve';

export type LabMemoryMode = 'reserve-max' | 'paged' | 'paged-prefix';

export interface DeskConfig {
  batching: BatchingMode;
  maxRunning: number;
  tokenBudget: number | null;
  memoryMode: LabMemoryMode;
  admission: AdmissionPolicy;
}

export function vllmDeskCommand(cfg: DeskConfig, maxModelLen: number): string {
  const parts: string[] = [
    'vllm serve meta-llama/Meta-Llama-3-8B-Instruct',
    '  --dtype bfloat16',
    `  --max-model-len ${maxModelLen}`,
    `  --max-num-seqs ${cfg.maxRunning}`,
  ];

  if (cfg.tokenBudget !== null) {
    parts.push('  --enable-chunked-prefill');
    parts.push(`  --max-num-batched-tokens ${cfg.tokenBudget}`);
  } else {
    parts.push('  --no-enable-chunked-prefill');
  }

  parts.push('  --block-size 16');
  parts.push(cfg.memoryMode === 'paged-prefix' ? '  --enable-prefix-caching' : '  --no-enable-prefix-caching');

  const notes: string[] = [];
  if (cfg.batching === 'static') {
    notes.push('# Note: vLLM always uses continuous batching; static batching is what pre-Orca servers ran.');
  }
  if (cfg.memoryMode === 'reserve-max') {
    notes.push('# Note: vLLM always pages KV memory (PagedAttention); contiguous max-length slabs cannot be enabled in vLLM.');
  }
  if (cfg.admission === 'sjf') {
    notes.push('# Note: vLLM schedules waiting requests FCFS by default; shortest-prompt-first needs a custom priority policy.');
  }

  return notes.length ? `${parts.join(' \\\n')}\n${notes.join('\n')}` : parts.join(' \\\n');
}
