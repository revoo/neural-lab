import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'information.log',
    kind: 'numeric',
    concepts: ['logarithm'],
    make: (r) => {
      const k = r.int(8) + 2;
      return {
        prompt: `A binary search over a sorted list of $${2 ** k}$ items needs how many yes/no comparisons, in the worst case, to pin down one item? (That is: what is $\\log_2 ${2 ** k}$?)`,
        answer: k,
        explain: `Each comparison halves the remaining candidates, so you need as many halvings as it takes to get from ${2 ** k} down to 1: $\\log_2 ${2 ** k} = ${k}$, because $2^{${k}} = ${2 ** k}$.`,
      };
    },
  },
  {
    id: 'information.surprise',
    kind: 'numeric',
    concepts: ['surprise'],
    make: (r) => {
      const k = r.int(6) + 1;
      return {
        prompt: `An event has probability $1/${2 ** k}$. How surprising is it, in bits?`,
        answer: k,
        explain: `Surprise is $-\\log_2 p = \\log_2 \\frac{1}{p} = \\log_2 ${2 ** k} = ${k}$ bits — the same as being told the outcome of ${k} fair coin flip${k > 1 ? 's' : ''}.`,
      };
    },
  },
  {
    id: 'information.ce',
    kind: 'numeric',
    concepts: ['cross-entropy'],
    tolerance: 0.01,
    make: (r) => {
      const p = [0.5, 0.25, 0.9, 0.1, 0.8][r.int(5)];
      const ans = -Math.log(p);
      return {
        prompt: `A classifier gives the correct class a probability of $${p}$. What is its cross-entropy loss on this example, using the natural log? (2 decimal places)`,
        answer: Math.round(ans * 100) / 100,
        explain: `Cross-entropy is the model’s surprise at the truth: $-\\ln(${p}) \\approx ${ans.toFixed(3)}$. Only the probability on the correct class matters for a single-label example.`,
      };
    },
  },
  {
    id: 'information.why-ce',
    kind: 'choice',
    concepts: ['cross-entropy', 'loss'],
    prompt: 'Why do classifiers usually train with cross-entropy rather than squared error on the probabilities?',
    options: [
      { text: 'Cross-entropy punishes confident wrong answers without limit, giving strong gradients exactly when the model is badly wrong.', correct: true, why: 'As $p \\to 0$, $-\\ln p \\to \\infty$, whereas $(1-p)^2$ never exceeds 1. Combined with softmax, cross-entropy’s gradient with respect to the logits is simply $p - y$: large when the model is confidently wrong, small when it’s right. Squared error through a softmax or sigmoid can produce tiny gradients when the model is confidently wrong — the worst time to learn slowly.' },
      { text: 'Squared error can’t be computed for probabilities.', why: 'It certainly can be computed — the widget does exactly that. The problem is how it *behaves*, not whether it exists.' },
      { text: 'Cross-entropy is always smaller, so training is faster.', why: 'Its value isn’t the point; its *shape* is. And for confident wrong answers it’s far larger, not smaller.' },
      { text: 'It doesn’t matter; they lead to identical training.', why: 'They lead to quite different gradients. Confidently wrong examples are where the difference is most dramatic.' },
    ],
  },
  {
    id: 'information.entropy-max',
    kind: 'choice',
    concepts: ['entropy'],
    prompt: 'Which distribution over 4 outcomes has the **highest** entropy?',
    options: [
      { text: '$[0.25, 0.25, 0.25, 0.25]$', correct: true, why: 'Entropy is highest when you can predict nothing — all outcomes equally likely. For 4 outcomes that’s $\\log_2 4 = 2$ bits.' },
      { text: '$[1, 0, 0, 0]$', why: 'This has *zero* entropy: the outcome is certain, so learning it tells you nothing.' },
      { text: '$[0.7, 0.1, 0.1, 0.1]$', why: 'This is fairly predictable (you’d bet on the first outcome), so its entropy is lower — about 1.36 bits.' },
      { text: '$[0.5, 0.5, 0, 0]$', why: 'This is exactly as uncertain as one fair coin flip: 1 bit. The uniform distribution over all four does better.' },
    ],
  },
];
