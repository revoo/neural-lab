// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'kv.why-exact',
    kind: 'choice',
    concepts: ['kv-cache'],
    prompt: 'Why does reusing cached K and V vectors during autoregressive generation produce the exact same output as recomputing the whole prompt from scratch?',
    options: [
      {
        text: 'Because of the causal mask, each earlier token’s hidden state — and therefore its key and value vectors — depends only on tokens at or before its own position, never on newly generated tokens.',
        correct: true,
        why: 'Earlier keys and values are completely frozen once computed. Caching them is exact memoisation (max difference 0.00 in our demo), not an approximation.',
      },
      {
        text: 'Because attention weights on earlier tokens are small enough to ignore.',
        why: 'Earlier tokens are not ignored at all — the new token’s query still scores against every single cached key and blends every cached value.',
      },
      {
        text: 'Because model weights are updated only once per conversation.',
        why: 'Weights stay fixed during inference either way. What makes past K and V invariant is the causal mask preventing future tokens from altering past activations.',
      },
    ],
  },
  {
    id: 'kv.memory-bound',
    kind: 'choice',
    concepts: ['memory-bound'],
    prompt: 'Why does quantising a model’s weights from FP16 to INT4 make single-user (batch = 1) token generation roughly 3–4× faster on an H100, even though GPUs compute at FP16 speed?',
    options: [
      {
        text: 'Single-user decoding is memory-bandwidth bound (~1 FLOP/byte vs H100’s 295 FLOP/byte ridge): shrinking weights ~3.6× cuts the bytes streamed from HBM per token by ~3.6×, directly speeding up every step.',
        correct: true,
        why: 'At B = 1, 99.6% of an H100’s Tensor Cores sit idle waiting on HBM3. Fewer weight bytes to move across the memory bus means proportionally faster tokens per second.',
      },
      {
        text: 'Because INT4 multiplication uses 4× fewer FLOPs than FP16 multiplication.',
        why: 'The number of multiply-adds (2N FLOPs per token) is unchanged; in fact weights are often dequantised right before multiplying. The win comes from moving fewer bytes from VRAM.',
      },
      {
        text: 'Because INT4 automatically enables KV caching.',
        why: 'KV caching and weight quantisation are independent techniques that complement each other.',
      },
    ],
  },
  {
    id: 'kv.gqa-calc',
    kind: 'numeric',
    concepts: ['kv-cache', 'gqa'],
    tolerance: 0.05,
    make: (r) => {
      const layers = [32, 40, 64, 80][r.int(4)];
      const kvHeads = [4, 8, 16][r.int(3)];
      const ctxK = [16, 32, 64, 128][r.int(4)];
      const bytesPerElem = 2; // FP16
      const dHead = 128;
      // 2 * L * kvHeads * 128 * (ctxK * 1024) * 2 / 1024^3
      const gib = (2 * layers * kvHeads * dHead * (ctxK * 1024) * bytesPerElem) / (1024 ** 3);
      return {
        prompt: `A Transformer has ${layers} layers, ${kvHeads} KV heads (GQA), and head dimension 128. How many GiB does its FP16 (2 bytes/element) KV cache occupy for 1 user at ${ctxK}k (${ctxK} × 1,024) tokens?`,
        answer: Number(gib.toFixed(2)),
        explain: `2 × ${layers} × ${kvHeads} × 128 × (${ctxK} × 1,024) × 2 bytes ÷ 1024³ = ${gib.toFixed(2)} GiB.`,
      };
    },
  },
  {
    id: 'kv.long-context-batch',
    kind: 'choice',
    concepts: ['kv-cache', 'memory-bound'],
    prompt: 'When serving an 8B model to 32 concurrent users at 32k context on an 80 GiB H100, why does quantising the KV cache (e.g. FP16 → FP8 or INT4) boost total throughput more than quantising the weights further?',
    options: [
      {
        text: 'Because KV cache scales with batch × context (128 GiB in FP16 for 32 users at 32k — OOM!), so shrinking KV cache both unlocks larger batches and cuts the dominant share of HBM bytes read every decode step.',
        correct: true,
        why: 'In Brief 3 of VRAM Tetris, INT8 weights (7.7 GiB) + INT4 KV cache (36 GiB for B=32 at 32k) reaches 2,284 tok/s because KV reads dominate weight reads nearly 5-to-1.',
      },
      {
        text: 'Because the GPU never reads weights during batched decoding.',
        why: 'Weights are still read once per step — it is just that reading them once is amortised across all 32 users, whereas each user has their own separate KV cache.',
      },
      {
        text: 'Because quantising the KV cache turns decoding into a single prefill pass.',
        why: 'Decoding still produces one token per user per step; KV quantisation shrinks both memory footprint and per-step HBM traffic.',
      },
    ],
  },
];
