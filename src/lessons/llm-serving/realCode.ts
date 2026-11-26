// authored-by: Gemini Antigravity
/**
 * Maps learner settings in Lab 1 (The Serving Desk) and Lab 2 (Tune the draft) to real
 * `vllm serve` CLI commands. Every flag was checked against vLLM's stable docs
 * (`docs.vllm.ai/en/stable/`, November 2026):
 *   --dtype, --max-model-len, --max-num-seqs, --enable-chunked-prefill / --no-enable-chunked-prefill,
 *   --max-num-batched-tokens, --block-size, --enable-prefix-caching / --no-enable-prefix-caching,
 *   --speculative-config (JSON with `model`, `method`, `num_speculative_tokens`, `prompt_lookup_max`).
 */
import type { AdmissionPolicy, BatchingMode } from './serve';
import type { DrafterId } from './spec';

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

export function vllmSpecCommand(drafter: DrafterId, k: number, temperature: number): string {
  const specJson =
    drafter === 'small-gpt'
      ? JSON.stringify({ model: 'meta-llama/Llama-3.2-1B-Instruct', num_speculative_tokens: k })
      : JSON.stringify({ method: 'ngram', num_speculative_tokens: k, prompt_lookup_max: 4 });

  const lines = [
    'vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\',
    '  --dtype bfloat16 \\',
    `  --speculative-config '${specJson}'`,
    `# Send temperature=${temperature} per request in the /v1/completions JSON body (sampling is not a server flag).`,
  ];
  if (drafter === 'bigram') {
    lines.push('# Closest model-free vLLM method is n-gram prompt lookup (our lab uses a corpus bigram table).');
  }
  return lines.join('\n');
}
