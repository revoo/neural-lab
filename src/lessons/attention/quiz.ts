import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'attention.roles',
    kind: 'choice',
    concepts: ['attention'],
    prompt: 'In attention, what is the job of the **keys**?',
    options: [
      { text: 'They are what each word can be found by: each query is scored against every key.', correct: true, why: 'Keys are compared with queries (a dot product) to decide how much attention each word gets. The values are what gets handed over.' },
      { text: 'They are what each word hands over when it is attended to.', why: 'That’s the values. Separating “what I can be found by” (key) from “what I pass on” (value) is part of what makes attention flexible.' },
      { text: 'They are what each word is looking for.', why: 'That’s the query.' },
    ],
  },
  {
    id: 'attention.weighted-sum',
    kind: 'choice',
    concepts: ['attention'],
    prompt: 'How does attention differ from an ordinary dense layer, which also computes weighted sums?',
    options: [
      { text: 'In a dense layer the weights are fixed learned numbers; in attention they are recomputed from the input for every sentence.', correct: true, why: 'Attention’s mixing weights come from softmax(q·k/√d), which depends on the words themselves. That’s what lets “it” mix in “animal” in one sentence and something else in another.' },
      { text: 'Attention doesn’t use weighted sums.', why: 'Its output is exactly a weighted sum of value vectors.' },
      { text: 'Attention has no learned parameters.', why: 'W_Q, W_K and W_V are learned. What isn’t fixed is the attention pattern they produce.' },
    ],
  },
  {
    id: 'attention.sqrt-d',
    kind: 'choice',
    concepts: ['attention', 'softmax'],
    prompt: 'Why are the scores divided by √d?',
    options: [
      { text: 'Dot products of longer vectors are bigger (their typical size grows like √d), which would make softmax saturate; dividing by √d keeps them the same size.', correct: true, why: 'Without it, one key would take nearly all the attention in high dimensions and the gradients through softmax would vanish. In the lesson’s measurement, the largest weight rose from 38% to 95% between d = 2 and d = 512 without scaling, and stayed near 31% with it.' },
      { text: 'To make the attention weights add up to 1.', why: 'Softmax already does that, whatever the scale of the scores.' },
      { text: 'To make attention faster to compute.', why: 'It’s one extra division; the point is to keep softmax in a useful range.' },
    ],
  },
  {
    id: 'attention.softmax-calc',
    kind: 'numeric',
    concepts: ['attention', 'softmax'],
    tolerance: 0.01,
    make: (r) => {
      const a = [1, 2, 3][r.int(3)];
      const b = [0, 1][r.int(2)];
      const w = Math.exp(a) / (Math.exp(a) + Math.exp(b) + 1);
      return {
        prompt: `A query has (already scaled) scores ${a}, ${b} and 0 against three keys. What share of the attention does the first key get? Give a number between 0 and 1.`,
        answer: Number(w.toFixed(4)),
        explain: `softmax: e^${a} / (e^${a} + e^${b} + e^0) = ${Math.exp(a).toFixed(3)} / ${(Math.exp(a) + Math.exp(b) + 1).toFixed(3)} = ${w.toFixed(3)}.`,
      };
    },
  },
  {
    id: 'attention.inside',
    kind: 'choice',
    concepts: ['attention', 'dot-product'],
    prompt: 'A word’s key is exactly the average of two other words’ keys. Can any query make it get more attention than both of them?',
    options: [
      { text: 'No: its score is always the average of their two scores, so it can at best tie with them.', correct: true, why: 'The dot product is linear: q·(a + b)/2 = (q·a + q·b)/2, which can’t exceed the larger of q·a and q·b. That was the impossible puzzle.' },
      { text: 'Yes, with a long enough query.', why: 'Lengthening the query scales all three scores together; the average still can’t overtake the larger of the two.' },
      { text: 'Yes, if the query points exactly at it.', why: 'Even then its score is the average of the other two, so one of them is at least as large.' },
    ],
  },
  {
    id: 'attention.layers',
    kind: 'choice',
    concepts: ['attention'],
    prompt: 'In a single attention layer, why can’t “it” use the word “wide” to decide whether to attend to “street”?',
    options: [
      { text: 'Its query is computed from its own vector only, before it has looked at any other word.', correct: true, why: 'Context arrives through attention itself. After one layer, each word’s vector carries information from the others, so the next layer’s queries can use it. That’s why transformers stack many layers.' },
      { text: 'Because attention only looks at earlier words.', why: '“wide” comes after “it”, but in this (unmasked) attention every word can see every other. The problem is how the query is formed.' },
      { text: 'Because the √d scaling hides it.', why: 'Scaling changes how sharp the attention is, not what information a query can use.' },
    ],
  },
];
