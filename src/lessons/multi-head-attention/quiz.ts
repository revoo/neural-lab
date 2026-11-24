import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'multi-head-attention.budget',
    kind: 'choice',
    concepts: ['multi-head-attention', 'softmax'],
    prompt: 'Why can’t a single attention head give two different words 80% of its attention each?',
    options: [
      { text: 'Its weights come from one softmax, which always adds up to 100%.', correct: true, why: 'A softmax shares out a fixed budget. Two heads have two budgets, which is the whole point of multi-head attention.' },
      { text: 'Because the query isn’t long enough.', why: 'A longer query makes the attention sharper, but the weights still add up to 100%.' },
      { text: 'It can, if the two words have the same key.', why: 'Then they’d get equal shares — at most 50% each.' },
    ],
  },
  {
    id: 'multi-head-attention.params',
    kind: 'numeric',
    concepts: ['multi-head-attention'],
    tolerance: 0.5,
    make: (r) => {
      const d = [256, 512, 1024][r.int(3)];
      const h = [4, 8, 16][r.int(3)];
      return {
        prompt: `An attention layer has model width d = ${d} and ${h} heads. How many weights do W_Q, W_K, W_V and W_O hold together (ignore biases)?`,
        answer: 4 * d * d,
        explain: `Each of the four matrices is ${d} × ${d} — all ${h} heads’ slices together — so 4 × ${d}² = ${(4 * d * d).toLocaleString('en-GB')}. The number of heads doesn’t appear: each head just gets ${d / h} of the ${d} columns.`,
      };
    },
  },
  {
    id: 'multi-head-attention.free',
    kind: 'choice',
    concepts: ['multi-head-attention'],
    prompt: 'A model keeps its width at 768 but goes from 12 heads to 24. What changes?',
    options: [
      { text: 'Each head works with vectors of length 32 instead of 64; parameters and arithmetic stay the same.', correct: true, why: 'Heads split the same d columns into more, shorter groups. More separate softmaxes, each comparing shorter vectors, at the same cost.' },
      { text: 'The attention layer gets twice as many parameters.', why: 'W_Q, W_K, W_V and W_O are still 768 × 768 each; they’re just sliced into more pieces.' },
      { text: 'Each token can now look at twice as many other tokens.', why: 'Every head already looks at every token. More heads means more separate attention patterns, not wider ones.' },
    ],
  },
  {
    id: 'multi-head-attention.concat',
    kind: 'choice',
    concepts: ['multi-head-attention'],
    prompt: 'Why are the heads’ outputs concatenated rather than averaged?',
    options: [
      { text: 'Averaging would blend the different heads’ findings together again; concatenating keeps each in its own block for W_O to combine as it learns best.', correct: true, why: 'The point of separate heads is to keep separate pieces of information separate. W_O can still mix them, but in a learned way.' },
      { text: 'Concatenating is faster.', why: 'The cost is about the same; the reason is information, not speed.' },
      { text: 'Averaging would change the vector length.', why: 'Averaging keeps the length of one head; concatenating gives back the full width d. But the main reason is not to mix the findings.' },
    ],
  },
];
