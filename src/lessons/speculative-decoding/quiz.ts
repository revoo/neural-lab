// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'spec.lossless',
    kind: 'choice',
    concepts: ['speculative-decoding', 'acceptance-rate'],
    prompt: 'When a speculative draft token x is rejected (because u ≥ p(x)/q(x)), why must the target model draw the replacement from norm(max(0, p − q)) instead of simply resampling from p?',
    options: [
      {
        text: 'Tokens where q(x) > p(x) were already over-proposed and accepted at full probability p(x); subtracting q from p removes that already-served mass so the combined probability of every token equals p(y) exactly.',
        correct: true,
        why: 'Analytically, min(p, q) + max(0, p − q) = p identically. Resampling directly from p after rejection would double-count tokens the draft already over-represented.',
      },
      {
        text: 'Because p(x) is unavailable after a rejection.',
        why: 'The target forward pass just computed the full vector p for that position — subtracting q is deliberate so the final distribution stays unbiased.',
      },
      {
        text: 'It is only a heuristic to make the text slightly more diverse.',
        why: 'It is an exact mathematical identity: speculative sampling is lossless and preserves p to machine precision.',
      },
    ],
  },
  {
    id: 'spec.expected',
    kind: 'numeric',
    concepts: ['speculative-decoding', 'acceptance-rate'],
    tolerance: 0.05,
    make: (r) => {
      const alpha = [0.5, 0.6, 0.7, 0.8][r.int(4)];
      const k = [2, 3, 4, 5][r.int(4)];
      const expected = (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
      return {
        prompt: `Under the i.i.d. formula E = (1 − α^(k+1)) / (1 − α), if a draft model proposes k = ${k} tokens per round and each is accepted with probability α = ${alpha.toFixed(1)}, how many tokens are emitted per target forward pass on average?`,
        answer: Number(expected.toFixed(2)),
        explain: `(1 − ${alpha.toFixed(1)}^${k + 1}) / (1 − ${alpha.toFixed(1)}) = ${(1 - Math.pow(alpha, k + 1)).toFixed(4)} / ${(1 - alpha).toFixed(1)} = ${expected.toFixed(2)} tokens per target pass.`,
      };
    },
  },
  {
    id: 'spec.cost-vs-alpha',
    kind: 'choice',
    concepts: ['speculative-decoding', 'acceptance-rate'],
    prompt: 'In our Grimm tiny-GPT lab, the 5,424-parameter neural draft achieved ~15–20% higher acceptance than the bigram table, yet the bigram table won the modelled RTX 4090 speed-up at every temperature (2.30× vs 2.03× greedy, 2.28× vs 2.02× at T = 1). Why?',
    options: [
      {
        text: 'Speed-up is E / (k·c + 1): running a 1B-scale neural draft k times adds a substantial k·c denominator tax (c = 0.127), whereas a table lookup has c ≈ 0.01.',
        correct: true,
        why: 'At k = 7, the neural draft pays 7 × 0.127 = 89% of a target pass in overhead; a cheaper drafter can win net wall time even with fewer tokens accepted per round.',
      },
      {
        text: 'Because bigram tables have a larger context window than transformers.',
        why: 'A bigram table only sees the single preceding character (context 1); its sole advantage is near-zero draft cost c.',
      },
      {
        text: 'Because greedy decoding always rejects neural draft tokens.',
        why: 'The neural draft actually got 80.5% acceptance at greedy (E = 3.82 chars/pass) — it lost only after paying its own draft step cost.',
      },
    ],
  },
  {
    id: 'spec.when-fails',
    kind: 'choice',
    concepts: ['speculative-decoding', 'memory-bound'],
    prompt: 'Every single one of our 48 speculative settings ran slower than plain decoding in real wall-clock time on the CPU (0.34× to 0.83×), even when modelled GPU speed-up exceeded 2.2×. Why?',
    options: [
      {
        text: 'The 29.7k-parameter target fits in CPU cache and is compute-bound, so verifying k tokens takes about k× the time of one token — speculative decoding only pays when verification is memory-bandwidth bound.',
        correct: true,
        why: 'Speculative decoding exploits the idle 99% of Tensor Cores in memory-bound GPU decoding. Once the chip is compute-bound (tiny CPU models, or very large GPU batches past the ridge point), extra verification FLOPs cost real wall time.',
      },
      {
        text: 'Because rolling back the KV cache on rejection requires recomputing the entire prompt from scratch.',
        why: 'Rollback on Decoder is literally a single integer assignment (decoder.len = keepLen) and costs almost nothing.',
      },
      {
        text: 'Because acceptance rate on Grimm’s Fairy Tales was below 10%.',
        why: 'Acceptance was 55–80% — plenty high for >2× modelled GPU gains. The bottleneck was hardware compute-boundedness, not acceptance.',
      },
    ],
  },
];
