import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'positions.blind',
    kind: 'choice',
    concepts: ['attention'],
    prompt: 'Without any position information, what happens to an attention layer’s outputs if you shuffle the order of the input words?',
    options: [
      { text: 'Each word’s output stays exactly the same; the outputs are just shuffled the same way.', correct: true, why: 'Every word still meets the same set of other words and computes the same scores. Nothing in attention looks at position.' },
      { text: 'All the outputs change, because each word now has different neighbours.', why: 'Attention has no notion of neighbours: every word is compared with every other word, wherever it is.' },
      { text: 'The layer refuses to run.', why: 'It runs happily — which is exactly the problem.' },
    ],
  },
  {
    id: 'positions.why-not-numbers',
    kind: 'choice',
    concepts: ['attention'],
    prompt: 'Why not simply add the position number (0, 1, 2, …) to every component of each word’s vector?',
    options: [
      { text: 'Far into a long text, the position would be thousands of times bigger than the word’s own numbers and drown out its meaning.', correct: true, why: 'Word vectors have components around 1. Sinusoidal positions stay between −1 and 1 however long the text, while still giving every position a different pattern.' },
      { text: 'Because neural networks can’t represent integers.', why: 'They can multiply and add any numbers; the problem is the scale.' },
      { text: 'It would work perfectly; it’s just not traditional.', why: 'The scale problem is real: by position 2,000, the word’s own information would be swamped.' },
    ],
  },
  {
    id: 'positions.rope-offset',
    kind: 'choice',
    concepts: ['rope'],
    prompt: 'With RoPE, a query at position 10 and a key at position 13 give score S. What is the score for the same two words at positions 110 and 113?',
    options: [
      { text: 'Exactly S: rotating both vectors changes the angle between them only by θ × (13 − 10), the offset.', correct: true, why: 'The score depends on contents and offset only. That’s RoPE’s defining property.' },
      { text: 'Smaller, because positions further along are rotated more.', why: 'Both are rotated more, by the same extra amount, so the angle between them is unchanged.' },
      { text: 'It depends on the length of the text.', why: 'Nothing in the rotation depends on the length of the text, only on each word’s own position.' },
    ],
  },
  {
    id: 'positions.many-hands',
    kind: 'choice',
    concepts: ['rope'],
    prompt: 'Why do position schemes use many rotation speeds rather than one?',
    options: [
      { text: 'A fast hand tells neighbours apart but repeats every 360/θ positions; a slow one never repeats but can’t tell neighbours apart. Together they do both.', correct: true, why: 'In the lab, no single speed could make a previous-word head work on 64 words; two speeds could.' },
      { text: 'To make the position vectors longer, which always helps.', why: 'Length alone doesn’t help; what matters is covering both short and long distances without ambiguity.' },
      { text: 'Because each layer needs its own speed.', why: 'All layers typically use the same set of speeds; the reason for many speeds is resolution versus ambiguity.' },
    ],
  },
  {
    id: 'positions.angle',
    kind: 'numeric',
    concepts: ['rope'],
    tolerance: 0.5,
    make: (r) => {
      const t = [10, 15, 20, 30][r.int(4)];
      const m = 2 + r.int(5);
      const n = m + 1 + r.int(5);
      return {
        prompt: `In a single RoPE pair turning ${t}° per position, a query sits at position ${m} and a key at position ${n}. By how many degrees does rotation change the angle between them?`,
        answer: t * (n - m),
        explain: `The query turns ${t * m}° and the key ${t * n}°, so the angle between them changes by ${t} × (${n} − ${m}) = ${t * (n - m)}°: it depends only on the offset.`,
      };
    },
  },
];
