// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'moe.memory',
    kind: 'choice',
    concepts: ['moe'],
    prompt: 'A dense model and a mixture-of-experts model use the same number of weights per token. What is the MoE model’s main hardware cost?',
    options: [
      {
        text: 'Memory: every expert must be loaded, even though each token uses only a few of them.',
        correct: true,
        why: 'Mixtral holds 46.7 billion weights but each token uses 12.9 billion. The arithmetic is that of a 13B model; the memory is that of a 47B one.',
      },
      { text: 'Arithmetic: every expert runs on every token, and the router just weights their outputs.', why: 'That would be a dense mixture. The point of sparse routing is that unchosen experts don’t run at all.' },
      { text: 'Nothing: sparse models are strictly cheaper than dense ones of the same quality.', why: 'They trade memory (and communication between GPUs holding different experts) for arithmetic. Whether that’s a good deal depends on the hardware.' },
    ],
  },
  {
    id: 'moe.collapse',
    kind: 'choice',
    concepts: ['router-collapse'],
    prompt: 'Without a balancing loss, why can an expert that gets no tokens early in training stay idle for good?',
    options: [
      {
        text: 'It receives no gradient: only experts that run on a token learn from it, so it never gets better and never wins tokens back.',
        correct: true,
        why: 'In our 4-expert layer the idle expert was already idle at step 50 on seed 1, and still idle at step 600. The busy experts keep improving, which makes the router prefer them even more.',
      },
      { text: 'Its weights are deleted to save memory.', why: 'Its weights stay in memory, unused. That’s exactly what makes collapse expensive.' },
      { text: 'The softmax makes its probability exactly zero.', why: 'Softmax probabilities are never exactly zero. The expert just keeps losing the top-k contest.' },
    ],
  },
  {
    id: 'moe.aux',
    kind: 'choice',
    concepts: ['load-balancing-loss'],
    prompt: 'The Switch Transformer’s balance term is E · Σ fᵢ Pᵢ, where fᵢ is the share of tokens sent to expert i and Pᵢ its average router probability. With 4 experts, what is it when every token goes to one expert with probability 1, and when everything is perfectly even?',
    options: [
      { text: '4 when collapsed, 1 when even.', correct: true, why: 'Collapsed: 4 × (1 × 1) = 4. Even: 4 × 4 × (¼ × ¼) = 1, the smallest value it can take. Gradient descent pushes it towards 1.' },
      { text: '1 when collapsed, 0 when even.', why: 'The term never reaches 0: with any routing at all, some fᵢ and Pᵢ are positive. Its minimum is 1, at perfect balance.' },
      { text: 'The same in both cases, because shares always add up to 1.', why: 'The shares add up to 1, but the sum of products fᵢ · Pᵢ is largest when both concentrate on the same expert.' },
    ],
  },
  {
    id: 'moe.capacity',
    kind: 'choice',
    concepts: ['moe', 'load-balancing-loss'],
    prompt: 'In our lab, even with perfectly balanced routing on average, a capacity factor of 1.0 still dropped several percent of token slots. Why?',
    options: [
      {
        text: 'Each mini-batch is a random draw, so some batches send more tokens than average to one expert, and its fixed buffer overflows.',
        correct: true,
        why: 'Balance is an average over many batches. Capacity is enforced batch by batch, so a little headroom (C = 1.25 here) is needed to absorb the fluctuations.',
      },
      { text: 'The router always drops the hardest tokens on purpose.', why: 'Tokens are dropped by position in the batch once a buffer is full, not by difficulty.' },
      { text: 'A capacity factor of 1.0 means each expert can take only one token.', why: 'It means each expert can take its fair share of the batch, k × tokens ÷ experts, and no more.' },
    ],
  },
];
