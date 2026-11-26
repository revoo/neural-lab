// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'serving.ttft-vs-tpot',
    kind: 'choice',
    concepts: ['ttft', 'tpot', 'goodput'],
    prompt: 'On launch day, 200 people send a message in the same second and your server handles one request at a time. Which metric blows up to minutes, and which metric stays fast?',
    options: [
      {
        text: 'TTFT blows up to minutes because of queueing, while TPOT stays near single-user speed once a request finally reaches the GPU.',
        correct: true,
        why: 'On our H100 burst trace, the 200th person waits 169.5 seconds for their first token (almost all queueing), yet once their turn starts their tokens still stream at ~205 tok/s.',
      },
      {
        text: 'TPOT blows up to minutes because the GPU is busy, while TTFT stays under 50 ms.',
        why: 'Queueing happens before a request starts generating, so it adds directly to TTFT. Serving one request at a time keeps TPOT untouched.',
      },
      {
        text: 'Both TTFT and TPOT blow up equally because throughput is capped.',
        why: 'Throughput and TTFT suffer badly under serial execution, whereas TPOT only degrades when a single GPU iteration gets slower (for example from a huge batch or an unchunked prefill).',
      },
    ],
  },
  {
    id: 'serving.continuous-gain',
    kind: 'choice',
    concepts: ['continuous-batching'],
    prompt: 'Continuous (iteration-level) batching performs the exact same total model FLOPs as static batching for the same 24 requests. Where do its throughput and latency gains come from?',
    options: [
      {
        text: 'Freed batch slots are refilled on the very next iteration instead of sitting empty until the longest reply in a static batch finishes.',
        correct: true,
        why: 'With heavy-tailed reply lengths, static batching left 64.7% of slot-steps idle on our 24-request wave; continuous batching cut that to 0% and slashed TTFT p95 from 3,969 ms to 52 ms.',
      },
      {
        text: 'Continuous batching skips the attention calculation for shorter sequences.',
        why: 'Every token still attends to its full causal context — nothing in the model maths is skipped.',
      },
      {
        text: 'Continuous batching compresses the KV cache so weights do not need to be read from VRAM.',
        why: 'Weights are still read once per step; batching helps by sharing that weight read across more live sequences instead of empty slots.',
      },
    ],
  },
  {
    id: 'serving.chunked-knee',
    kind: 'choice',
    concepts: ['chunked-prefill'],
    prompt: 'In our RTX 4090 experiment with a 16,384-token document, setting the chunk budget to 256 tokens dropped worst TPOT from 2,019 ms to 37.9 ms while adding only 37 ms to the document’s TTFT — yet shrinking the budget further to 64 tokens doubled the document’s TTFT to 4,552 ms. Why?',
    options: [
      {
        text: 'At 256 tokens/step the RTX 4090 is still above its ~164 FLOP/byte ridge point (compute-bound), whereas at 64 tokens/step each chunk falls back into the memory-bound regime and pays to re-read all 16 GB of weights hundreds of times.',
        correct: true,
        why: 'Above the ridge point, weight reads are hidden behind Tensor Core arithmetic, so splitting prefill is almost free for doc TTFT and lets decodes piggyback. Below the ridge point, memory bandwidth bottlenecks every chunk.',
      },
      {
        text: 'Because 64 tokens is smaller than the KV cache block size, causing page faults.',
        why: 'Our block size is 16 tokens, so 64 tokens is four full blocks. The cliff is purely the roofline ridge point between compute-bound and memory-bound steps.',
      },
      {
        text: 'Because chunked prefill recomputes the entire earlier prompt on every chunk.',
        why: 'Each chunk writes its keys and values to the KV cache and only attends to earlier chunks — nothing is recomputed.',
      },
    ],
  },
  {
    id: 'serving.paged-mechanism',
    kind: 'choice',
    concepts: ['paged-attention', 'prefix-caching'],
    prompt: 'Why does PagedAttention raise serving throughput even though following a block table makes the attention kernel slightly more complex?',
    options: [
      {
        text: 'By eliminating reserved-but-unused slabs and external fragmentation, far more sequences fit in VRAM at once — lifting the memory-bound decode batch size.',
        correct: true,
        why: 'On our RTX 4090 workload, paging lifted concurrent conversations from 7 to 42 (and 64 with prefix sharing), raising decode throughput from 405 to 1,756–2,267 tok/s because one weight read now serves far more users.',
      },
      {
        text: 'Because fixed 16-token blocks make the QKᵀ matrix multiply asymptotically faster than O(n).',
        why: 'Attention still visits every cached key and value token; paging changes memory layout and concurrency, not the asymptotic complexity of attention.',
      },
      {
        text: 'Because paging moves the KV cache onto the CPU so GPU VRAM only holds weights.',
        why: 'Paged KV blocks live in GPU VRAM so every decode step can read them at full GPU memory bandwidth.',
      },
    ],
  },
  {
    id: 'serving.latency-calc',
    kind: 'numeric',
    concepts: ['ttft', 'tpot'],
    tolerance: 5,
    make: (r) => {
      const ttft = [120, 250, 400, 650][r.int(4)];
      const outToks = [51, 101, 151, 201][r.int(4)];
      const tpot = [15, 20, 25, 30][r.int(4)];
      const total = ttft + (outToks - 1) * tpot;
      return {
        prompt: `A request has TTFT = ${ttft} ms, generates ${outToks} output tokens, and averages TPOT = ${tpot} ms/token. What is its total request latency in milliseconds?`,
        answer: total,
        explain: `Total latency = TTFT + (O − 1) × TPOT = ${ttft} + (${outToks} − 1) × ${tpot} = ${total} ms.`,
      };
    },
  },
];
