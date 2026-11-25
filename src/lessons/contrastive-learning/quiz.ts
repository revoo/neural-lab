// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'contrast.collapse',
    kind: 'choice',
    concepts: ['contrastive-learning', 'alignment-uniformity'],
    prompt: 'If you train a Siamese encoder purely to minimise the distance between positive pairs (‖a_i − b_i‖² → 0) with NO negative pairs, you get 0.0000 alignment loss — yet the embeddings are completely useless. Why?',
    options: [
      {
        text: 'The network cheats by mapping EVERY input (cats, dogs, cars, espressos) to the exact same constant vector, achieving zero alignment loss with zero information (total representation collapse).',
        correct: true,
        why: 'InfoNCE prevents collapse by pairing alignment (pull positives together) with uniformity (push all distinct in-batch negatives apart around the hypersphere).',
      },
      {
        text: 'Because ‖a_i − b_i‖² is non-differentiable on unit vectors.',
        why: 'Squared L2 distance is smooth and differentiable everywhere; the failure is a trivial global minimum (constant output).',
      },
      {
        text: 'Because temperature τ must equal 1.0.',
        why: 'Even without τ at all, positive-only MSE collapses straight to a constant.',
      },
    ],
  },
  {
    id: 'contrast.temperature',
    kind: 'choice',
    concepts: ['infonce'],
    prompt: 'Inside CLIP’s InfoNCE loss, what happens as you lower the temperature τ from 1.00 down to 0.05?',
    options: [
      {
        text: 'Dividing cosines by a smaller τ sharpens the softmax so almost 100% of the repulsive gradient concentrates on the hardest (most similar) wrong negatives rather than wasting updates on easy unrelated negatives.',
        correct: true,
        why: 'On our 4-way example, lowering τ from 1.00 to 0.05 lifted the hard negative’s share of push from 49.5% to 100.0% — though setting τ too small (e.g. 0.01) risks gradient spikes and FP16 overflow.',
      },
      {
        text: 'Lowering τ makes every negative pair receive equal gradient regardless of similarity.',
        why: 'That is what *raising* τ toward ∞ does (uniform softmax).',
      },
      {
        text: 'Lowering τ disables L2 normalisation.',
        why: 'CLIP always L2-normalises image and text embeddings before multiplying by 1/τ.',
      },
    ],
  },
  {
    id: 'contrast.batch-prob',
    kind: 'numeric',
    concepts: ['infonce'],
    tolerance: 0.5,
    make: (rng) => {
      const b = [64, 128, 256][rng.int(3)];
      const pct = Number((100 * (1 - 0.99 ** (b - 1))).toFixed(1));
      return {
        prompt: `Suppose 1% of your dataset consists of hard negatives for a given image. With in-batch negatives at batch size B = ${b} (giving ${b - 1} negatives per item), what is the percentage probability (0–100, to 1 decimal place) that a batch contains at least one hard negative?`,
        answer: pct,
        explain: `1 − (1 − 0.01)^(${b} − 1) = 1 − 0.99^${b - 1} = ${pct.toFixed(1)}%.`,
      };
    },
  },
  {
    id: 'contrast.moco',
    kind: 'choice',
    concepts: ['contrastive-learning'],
    prompt: 'Why did MoCo (Momentum Contrast) achieve 99.4%+ hard-negative coverage using only 0.63 GiB of activation VRAM instead of SimCLR’s 4–16 GiB?',
    options: [
      {
        text: 'MoCo decouples negative count from active batch size by storing thousands of precomputed embeddings in a FIFO queue updated by a slow-moving momentum encoder without backpropagating through the queue.',
        correct: true,
        why: 'You only store activations for a tiny active batch (B = 64), while still contrasting every query against Q = 512–65,536 queued negatives!',
      },
      {
        text: 'MoCo drops all negative samples from the loss.',
        why: 'MoCo uses *thousands* of negative samples from its momentum queue.',
      },
      {
        text: 'MoCo trains exclusively in 1D embeddings.',
        why: 'MoCo uses standard 128D–2048D embeddings.',
      },
    ],
  },
];
