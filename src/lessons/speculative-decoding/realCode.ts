// authored-by: Gemini Antigravity
/**
 * Maps Tune-the-draft settings onto verified vLLM `--speculative-config` JSON
 * (`docs.vllm.ai/en/stable/api/vllm/config/speculative/`, November 2026).
 */
import type { DrafterId } from './spec';

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
