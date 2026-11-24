// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'sampling.temp-tail',
    kind: 'choice',
    concepts: ['temperature', 'softmax'],
    prompt: 'Why can temperature alone never reduce the probability of an off-topic tail token to exact zero?',
    options: [
      {
        text: 'Because softmax computes exp(z / T), and e^x is strictly positive for every finite score — lowering T shrinks the tail, but never cuts it to zero.',
        correct: true,
        why: 'Even a tiny per-token tail probability adds up across tens of thousands of junk tokens and hundreds of generated steps. That is why practical samplers pair temperature with truncation (top-p or top-k).',
      },
      {
        text: 'Because temperature is only applied to the top 10 tokens.',
        why: 'Temperature divides every logit across the entire vocabulary before softmax.',
      },
      {
        text: 'Because setting T < 1 makes all probabilities equal.',
        why: 'The opposite: T < 1 sharpens the distribution toward the top token; T > 1 flattens it toward uniform.',
      },
    ],
  },
  {
    id: 'sampling.topp-vs-topk',
    kind: 'choice',
    concepts: ['top-p', 'top-k'],
    prompt: 'What problem with fixed top-k does nucleus sampling (top-p) solve?',
    options: [
      {
        text: 'Fixed top-k keeps the exact same number of tokens on every position, letting junk into sharp factual prompts or chopping valid words off flat open-ended ones; top-p adapts the pool size to the model’s confidence.',
        correct: true,
        why: 'In our widget, top-p = 0.90 kept just 1 token (“Paris”) on the factual prompt and automatically widened to 11 tokens on the open-ended box prompt — zero tail junk in both.',
      },
      {
        text: 'Top-p runs faster because it avoids sorting the vocabulary.',
        why: 'Top-p actually requires sorting the probabilities (from highest to lowest) to compute the cumulative sum.',
      },
      {
        text: 'Top-p always chooses the single highest-probability token.',
        why: 'That is greedy decoding (top-k = 1 or T → 0). Top-p only collapses to 1 token when that single token already exceeds probability p.',
      },
    ],
  },
  {
    id: 'sampling.beam-vs-greedy',
    kind: 'choice',
    concepts: ['beam-search'],
    prompt: 'Why can beam search (width 2) find a much higher-probability sentence than greedy decoding (width 1)?',
    options: [
      {
        text: 'Greedy commits to the single highest-probability first token and permanently discards runner-ups that could lead to far higher-probability continuations.',
        correct: true,
        why: 'In our 3-step tree, “The” beat “A” at step 1 (0.55 vs 0.45), trapping greedy in “The cat slept” (8.8%). Keeping width 2 preserved “A” and found “A bright sun” (34.4% — 3.9× more likely).',
      },
      {
        text: 'Beam search raises the temperature so rare words become more likely.',
        why: 'Beam search is deterministic and uses the model’s raw log-probabilities without temperature flattening.',
      },
      {
        text: 'Beam search changes the model’s weights during generation.',
        why: 'Decoding algorithms never touch weights; they only decide how to search or sample from the model’s fixed next-token distributions.',
      },
    ],
  },
  {
    id: 'sampling.nucleus-count',
    kind: 'numeric',
    concepts: ['top-p'],
    tolerance: 0.05,
    make: (r) => {
      const cases = [
        { probs: [0.5, 0.25, 0.15, 0.06, 0.04], p: 0.9, ans: 3, desc: '0.50 + 0.25 + 0.15 = 0.90' },
        { probs: [0.6, 0.2, 0.1, 0.05, 0.05], p: 0.85, ans: 3, desc: '0.60 + 0.20 = 0.80 (< 0.85), adding 0.10 reaches 0.90' },
        { probs: [0.92, 0.04, 0.02, 0.01, 0.01], p: 0.9, ans: 1, desc: '0.92 already clears 0.90 on its own' },
        { probs: [0.3, 0.25, 0.2, 0.15, 0.1], p: 0.75, ans: 3, desc: '0.30 + 0.25 + 0.20 = 0.75' },
      ];
      const c = cases[r.int(cases.length)];
      return {
        prompt: `Sorted next-token probabilities are [${c.probs.map((x) => x.toFixed(2)).join(', ')}]. How many tokens survive nucleus sampling at top-p = ${c.p.toFixed(2)}?`,
        answer: c.ans,
        explain: `Add probabilities from largest to smallest until the cumulative total reaches at least ${c.p.toFixed(2)}: ${c.desc} → ${c.ans} ${c.ans === 1 ? 'token' : 'tokens'}.`,
      };
    },
  },
  {
    id: 'sampling.creativity-myth',
    kind: 'choice',
    concepts: ['temperature'],
    prompt: 'What happens if you crank temperature up to T = 2.0 on a real language model?',
    options: [
      {
        text: 'Logit differences are halved before softmax, lifting thousands of low-probability tail tokens and making the output chaotic and incoherent rather than genuinely creative.',
        correct: true,
        why: 'Temperature does not add new ideas or knowledge — it only flattens the model’s confidence across its existing vocabulary.',
      },
      {
        text: 'The model thinks deeper and activates extra transformer layers.',
        why: 'Sampling runs only on the final output logits after the entire transformer forward pass has already finished.',
      },
      {
        text: 'The model becomes strictly factual because every answer gets equal consideration.',
        why: 'Flattening a sharp factual distribution steals probability from the right answer (“Paris”) and hands it to wrong ones.',
      },
    ],
  },
];
