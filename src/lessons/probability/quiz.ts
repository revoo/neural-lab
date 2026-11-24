import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'probability.valid',
    kind: 'choice',
    concepts: ['probability-distribution'],
    prompt: 'Which of these is a valid probability distribution over three outcomes?',
    options: [
      { text: '$[0.7, 0.3, 0.0]$', correct: true, why: 'All entries are non-negative and they sum to exactly 1. A probability of 0 is allowed — it just means “impossible”.' },
      { text: '$[0.5, 0.6, -0.1]$', why: 'It sums to 1, but probabilities can’t be negative. That’s exactly the failure of “divide by the sum” when scores go negative.' },
      { text: '$[0.2, 0.2, 0.2]$', why: 'Non-negative, but it only sums to 0.6. Some probability has gone missing.' },
      { text: '$[1, 1, 1]$', why: 'Each outcome can’t be certain at once. These sum to 3, not 1.' },
    ],
  },
  {
    id: 'probability.softmax-compute',
    kind: 'numeric',
    concepts: ['softmax'],
    tolerance: 0.005,
    make: (r) => {
      const d = [1, 2, Math.log(3), 0.5][r.int(4)];
      const base = r.int(5) - 2;
      const p = 1 / (1 + Math.exp(-d));
      return {
        prompt: `A two-class model outputs logits $[${(base + d).toFixed(2)}, ${base}]$. What probability does softmax assign to the **first** class? (3 decimal places)`,
        answer: Math.round(p * 1000) / 1000,
        explain: `Only the difference matters: it is $${d.toFixed(2)}$. So $p_1 = \\frac{e^{${d.toFixed(2)}}}{e^{${d.toFixed(2)}} + e^{0}} = \\frac{${Math.exp(d).toFixed(3)}}{${(Math.exp(d) + 1).toFixed(3)}} \\approx ${p.toFixed(3)}$. (With two classes, softmax reduces to the sigmoid function of the difference — you’ll meet it again in the neuron lesson.)`,
      };
    },
  },
  {
    id: 'probability.shift',
    kind: 'choice',
    concepts: ['softmax', 'logits'],
    prompt: 'You add 100 to every logit before applying softmax. What happens to the probabilities?',
    options: [
      { text: 'Nothing — they are identical.', correct: true, why: 'Adding $c$ to every logit multiplies every $e^{z_i}$ by the same factor $e^{c}$, which cancels between the top and bottom of the fraction. Softmax only sees differences between logits. (This is also why implementations can safely subtract the maximum logit for numerical stability.)' },
      { text: 'They all increase.', why: 'They can’t all increase — they must still sum to 1. And in fact none change at all.' },
      { text: 'The largest one approaches 1.', why: 'That’s what lowering the *temperature* does, by scaling differences up. Shifting leaves differences untouched.' },
      { text: 'The computation overflows and returns infinity.', why: 'A naive implementation might overflow ($e^{100}$ is large but fine; $e^{1000}$ overflows). Good implementations subtract the max first — which is allowed precisely because shifting doesn’t change the result.' },
    ],
  },
  {
    id: 'probability.temperature',
    kind: 'choice',
    concepts: ['temperature'],
    prompt: 'A language model is generating text with temperature 0.2. You raise it to 1.5. What changes?',
    options: [
      { text: 'Less likely tokens get a bigger share of probability, so output becomes more varied and less predictable.', correct: true, why: 'Higher temperature divides the logits by a larger number, shrinking the gaps between them. The distribution flattens, so sampling picks unusual tokens more often.' },
      { text: 'The model becomes more accurate.', why: 'Temperature doesn’t change what the model knows — only how its scores are turned into a distribution for sampling.' },
      { text: 'The most likely token becomes even more likely.', why: 'That’s the effect of *lowering* temperature. Dividing by a larger $T$ compresses the score differences.' },
      { text: 'Nothing — temperature only affects training.', why: 'Temperature is applied at generation time, after the model has computed its logits. It’s a knob on the sampling, not on the weights.' },
    ],
  },
];
