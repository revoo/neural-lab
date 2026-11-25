// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'stats.variance-numeric',
    kind: 'numeric',
    concepts: ['variance', 'standard-deviation'],
    tolerance: 0.01,
    make: (r) => {
      const sets = [
        [1, 3, 5, 7],
        [2, 4, 4, 4, 5, 5, 7, 9],
        [10, 12, 14],
        [0, 0, 6, 6],
      ];
      const xs = sets[r.int(sets.length)];
      const m = xs.reduce((a, b) => a + b, 0) / xs.length;
      const v = xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length;
      return {
        prompt: `What is the standard deviation of ${xs.join(', ')}? (Average the squared distances from the mean, then take the square root.)`,
        answer: Number(Math.sqrt(v).toFixed(3)),
        explain: `Mean = ${m}. Squared distances: ${xs.map((x) => (x - m) ** 2).join(', ')}. Their average (the variance) is ${v.toFixed(3)}, and its square root is ${Math.sqrt(v).toFixed(3)}.`,
      };
    },
  },
  {
    id: 'stats.outlier',
    kind: 'choice',
    concepts: ['variance'],
    prompt: 'You measure seven page-load times. One request hit a cold cache and took ten times longer than the rest. Which summary is barely affected?',
    options: [
      { text: 'The median', correct: true, why: 'The median only cares which value is in the middle, so one extreme value can’t drag it. (In the spread explorer, an outlier left the median at 5 while the mean moved from 5 to 6.)' },
      { text: 'The mean', why: 'The mean is a balance point: one far-away value pulls it a long way.' },
      { text: 'The standard deviation', why: 'Squaring makes the outlier’s distance dominate. In the explorer one outlier doubled the SD from 2 to 4.' },
    ],
  },
  {
    id: 'stats.variances-add',
    kind: 'choice',
    concepts: ['variance', 'standard-deviation'],
    prompt: 'One die has an SD of about 1.71. What is the SD of the total of two dice?',
    options: [
      { text: 'About 2.42: variances add (2.92 + 2.92 = 5.83), and √5.83 ≈ 2.42', correct: true, why: 'For independent things, variances add. SDs grow only with the square root, because highs on one die partly cancel lows on the other.' },
      { text: 'About 3.42: SDs add', why: 'That would only be true if the two dice always rolled high together and low together. Independent dice partly cancel.' },
      { text: 'About 1.71: rolling more dice doesn’t change the spread', why: 'The total does spread out (2 to 12 instead of 1 to 6), just less than twice as much.' },
    ],
  },
  {
    id: 'stats.clt',
    kind: 'choice',
    concepts: ['normal-distribution'],
    prompt: 'Why does the total of many dice look like a bell curve, when a single die is flat?',
    options: [
      { text: 'Extreme totals need every die to go the same way at once, which is rare; middling totals can happen in many different ways. Adding independent pieces piles probability into the middle.', correct: true, why: 'That’s the central limit theorem in one sentence. A 2 with two dice can happen one way; a 7 can happen six ways.' },
      { text: 'Because each die secretly favours its middle values.', why: 'A fair die is perfectly flat: every face 1/6. The bell comes from adding, not from the dice.' },
      { text: 'It only happens with dice; other random things stay the same shape when added.', why: 'Coins, dice and even lopsided long shots all head towards a bell when added, though lopsided sources take much longer.' },
    ],
  },
  {
    id: 'stats.se-numeric',
    kind: 'numeric',
    concepts: ['standard-error'],
    tolerance: 0.02,
    make: (r) => {
      const p = [0.8, 0.9, 0.95][r.int(3)];
      const n = [400, 900, 2500][r.int(3)];
      const se = Math.sqrt((p * (1 - p)) / n) * 100;
      return {
        prompt: `A classifier is truly right ${p * 100}% of the time. You score it on ${n} test examples. How many percentage points does the measured accuracy typically wobble (one standard error, √(p(1 − p)/n), in points)?`,
        answer: Number(se.toFixed(2)),
        explain: `√(${p} × ${(1 - p).toFixed(2)} / ${n}) = ${(se / 100).toFixed(4)}, i.e. ${se.toFixed(2)} percentage points. About 95% of test sets would land within ±${(2 * se).toFixed(2)} points.`,
      };
    },
  },
  {
    id: 'stats.sqrt-n',
    kind: 'choice',
    concepts: ['standard-error'],
    prompt: 'Your benchmark’s error bars are ±2 points and you need ±0.5. How much bigger must the test set be?',
    options: [
      { text: '16 times bigger', correct: true, why: 'The wobble shrinks like 1/√n. To cut it by 4×, n must grow 4² = 16×.' },
      { text: '4 times bigger', why: 'That would only halve the error bars: √4 = 2.' },
      { text: '2 times bigger', why: 'Doubling the data shrinks the error bars by only √2 ≈ 1.41×.' },
    ],
  },
  {
    id: 'stats.zscore',
    kind: 'choice',
    concepts: ['standard-deviation', 'layer-norm'],
    prompt: 'Why do networks usually standardise their inputs (subtract the mean, divide by the SD) before training?',
    options: [
      { text: 'So every feature arrives on a comparable scale around 0; otherwise a feature measured in thousands dominates the weighted sums and the gradients, and one step size can’t suit every weight.', correct: true, why: 'Standardising puts “typical” at 0 and “unusual” at ±1 or ±2 for every feature. Layer normalisation does the same thing inside the network.' },
      { text: 'Because neural networks only accept numbers between −1 and 1.', why: 'They accept any numbers; it’s the training that suffers when scales differ wildly (see the model clinic’s “unscaled inputs” case).' },
      { text: 'To make the data bell-shaped.', why: 'Shifting and rescaling changes position and width, never shape. A lopsided feature stays lopsided after standardising.' },
    ],
  },
];
