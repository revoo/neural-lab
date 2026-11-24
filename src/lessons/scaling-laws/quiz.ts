// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'scaling.6nd',
    kind: 'choice',
    concepts: ['scaling-law'],
    prompt: 'Why is the training compute of a dense transformer approximately C ≈ 6 N D FLOPs (six times parameters times training tokens)?',
    options: [
      {
        text: 'Each parameter takes ~2 FLOPs (multiply + add) in the forward pass and ~4 FLOPs (input gradient + weight gradient) in the backward pass, for every token processed.',
        correct: true,
        why: 'Inference (forward pass only) takes ~2 N FLOPs per token; training adds the backward pass (~4 N), totalling ~6 N FLOPs per token.',
      },
      {
        text: 'Because transformers always have 6 layers.',
        why: 'The factor of 6 comes from forward + backward matrix multiplication cost per parameter, regardless of how many layers the model has.',
      },
      {
        text: 'Because Adam stores 6 bytes per parameter.',
        why: 'Optimizer states affect memory (VRAM), not floating-point operations per token.',
      },
    ],
  },
  {
    id: 'scaling.chinchilla',
    kind: 'choice',
    concepts: ['chinchilla-optimal', 'scaling-law'],
    prompt: 'Chinchilla (70B params, 1.4T tokens) beat Gopher (280B params, 300B tokens) using roughly the same training compute. What was the core lesson?',
    options: [
      {
        text: 'Pre-Chinchilla models like GPT-3 and Gopher were severely under-trained: for a fixed training budget, scaling model size and training tokens roughly equally (~20 tokens per parameter) reaches a lower loss with a much smaller model.',
        correct: true,
        why: 'Gopher trained 280B parameters on only 1.1 tokens/param, leaving its finite-data penalty huge. Balancing capacity and data beat it at 4× smaller model size.',
      },
      {
        text: 'Smaller models always beat larger models regardless of how much data you have.',
        why: 'A 70B model beats a 280B model only when the 280B model is starved of training tokens. Give both 20 tokens/param (requiring 16× more compute for 280B) and the larger model wins.',
      },
      {
        text: 'DeepMind used a completely different architecture than the transformer.',
        why: 'Chinchilla was a standard dense transformer — almost identical in architecture to Gopher. The entire gain came from reallocating compute from parameters to tokens.',
      },
    ],
  },
  {
    id: 'scaling.llama3-overtrain',
    kind: 'choice',
    concepts: ['chinchilla-optimal'],
    prompt: 'Meta trained Llama 3 8B on 15 trillion tokens — almost 1,900 tokens per parameter, nearly 100× past Chinchilla’s ~20-to-1 ratio. Why wasn’t that a mistake?',
    options: [
      {
        text: 'Chinchilla minimises training FLOPs alone; once you deploy a model to serve trillions of inference tokens (2 N FLOPs each), paying extra training compute to make the deployed model smaller slashes total lifecycle cost.',
        correct: true,
        why: 'In our widget at L = 2.00 and 20T inference tokens, shrinking from a Chinchilla 50B model down to ~17B over-trained on 4.3T tokens cuts total lifecycle compute roughly in half.',
      },
      {
        text: 'Because Chinchilla scaling laws stop working below 10B parameters.',
        why: 'Chinchilla was fitted across models from 44M up to 16B+ parameters; the shift is about minimising inference cost, not a failure of the loss law.',
      },
      {
        text: 'Because training on more tokens costs zero extra FLOPs.',
        why: 'Training 8B on 15T tokens still costs 6 × 8B × 15T = 7.2 × 10²³ FLOPs — more training compute than Chinchilla 70B! You pay up front to save far more at serving time.',
      },
    ],
  },
  {
    id: 'scaling.flops-calc',
    kind: 'numeric',
    concepts: ['scaling-law'],
    tolerance: 0.5,
    make: (r) => {
      const N = [10, 20, 50, 70][r.int(4)]; // billions
      const D = [200, 400, 1000, 1400][r.int(4)]; // billions
      const ans = (6 * N * D) / 1e4; // in units of 10^22 FLOPs
      return {
        prompt: `Using C ≈ 6 N D, how many 10²² FLOPs does it take to train a ${N}B-parameter dense transformer on ${D}B tokens? (Give the number multiplying 10²², e.g. 12 for 12 × 10²².)`,
        answer: ans,
        explain: `6 × (${N} × 10⁹) × (${D} × 10⁹) = ${6 * N * D} × 10¹⁸ = ${ans} × 10²² FLOPs.`,
      };
    },
  },
  {
    id: 'scaling.irreducible',
    kind: 'choice',
    concepts: ['scaling-law', 'entropy'],
    prompt: 'In L(N, D) = E + A/N^α + B/D^β, what does the constant floor E represent?',
    options: [
      {
        text: 'The irreducible entropy of natural language — genuine unpredictability that even an infinitely large model trained on infinite text cannot eliminate.',
        correct: true,
        why: 'No matter how smart a model gets, if a writer might plausibly continue “I opened the box and found a ___” with “key”, “letter” or “ring”, the true distribution still has non-zero entropy.',
      },
      {
        text: 'The minimum loss caused by floating-point rounding errors.',
        why: 'Floating-point noise is tiny compared to the inherent entropy of human language (~1.7–1.8 nats per token).',
      },
      {
        text: 'The learning rate used during training.',
        why: 'E is an asymptotic loss floor measured in nats per token, not a hyperparameter.',
      },
    ],
  },
];
