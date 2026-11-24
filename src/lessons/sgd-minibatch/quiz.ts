import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'sgd.why-sample',
    kind: 'choice',
    concepts: ['sgd', 'mini-batch'],
    prompt: 'Why is it legitimate to estimate the gradient from a small random batch instead of the whole dataset?',
    options: [
      { text: 'The full gradient is an average of per-example gradients, and an average can be estimated from a random sample.', correct: true, why: 'Exactly like an opinion poll. The batch average is right on average, and its error shrinks as the batch grows.' },
      { text: 'Most examples have zero gradient, so they can be skipped.', why: 'Almost every example has a nonzero gradient. The point is that their *average* can be estimated without visiting all of them.' },
      { text: 'Because the batch gradient is exactly equal to the full gradient.', why: 'It isn’t — it’s a noisy estimate. The noise is the price of the speed, and it can be controlled with the batch size and step size.' },
    ],
  },
  {
    id: 'sgd.sqrt',
    kind: 'numeric',
    concepts: ['mini-batch'],
    tolerance: 0,
    make: (r) => {
      const B = [8, 16, 32, 50][r.int(4)];
      const f = [2, 3, 4][r.int(3)];
      return {
        prompt: `With batches of ${B}, the gradient estimate has a certain typical error. Roughly what batch size would make that error **${f} times smaller**?`,
        answer: B * f * f,
        explain: `The error shrinks like $1/\\sqrt{B}$, so to divide it by ${f} you multiply the batch by $${f}^2 = ${f * f}$: $${B} \\times ${f * f} = ${B * f * f}$.`,
      };
    },
  },
  {
    id: 'sgd.epoch-steps',
    kind: 'numeric',
    concepts: ['epoch', 'mini-batch'],
    tolerance: 0,
    make: (r) => {
      const N = [50000, 60000, 12800][r.int(3)];
      const B = [64, 128, 256][r.int(3)];
      const steps = Math.floor(N / B);
      return {
        prompt: `A dataset has ${N.toLocaleString('en-GB')} examples and the batch size is ${B}. How many update steps are in one epoch (drop any incomplete final batch)?`,
        answer: steps,
        explain: `One epoch is one pass through the data: $\\lfloor ${N} / ${B} \\rfloor = ${steps}$ steps. Full-batch gradient descent would take just 1.`,
      };
    },
  },
  {
    id: 'sgd.budget',
    kind: 'choice',
    concepts: ['sgd'],
    prompt: 'Given the same total number of examples processed, why does mini-batch SGD usually get further than full-batch gradient descent?',
    options: [
      { text: 'It turns the same work into many more steps; each is a bit noisy, but early on the noise is small compared with the true gradient.', correct: true, why: 'In the race, one epoch bought full batch 1 step and batches of 32 a full 31. Thirty-one roughly-right steps beat one perfect one.' },
      { text: 'Its gradients are more accurate.', why: 'They’re less accurate — that’s the trade. The advantage is the number of steps.' },
      { text: 'It uses a different loss function.', why: 'The loss is the same; only the estimate of its gradient changes.' },
    ],
  },
  {
    id: 'sgd.jitter',
    kind: 'choice',
    concepts: ['sgd', 'learning-rate'],
    prompt: 'Near the minimum, SGD with a fixed step size keeps jittering instead of settling. Why?',
    options: [
      { text: 'The true gradient shrinks to zero there, but examples still disagree, so each estimate is mostly noise.', correct: true, why: 'Noise comes from disagreement between examples, which doesn’t vanish at the minimum. Smaller steps (a learning-rate schedule) or bigger batches calm it down.' },
      { text: 'The learning rate is too small.', why: 'A smaller learning rate *reduces* the jitter. The problem is the noise in each step.' },
      { text: 'The minimum moves every epoch.', why: 'The minimum of the training loss is fixed. Only the gradient *estimates* change from batch to batch.' },
    ],
  },
  {
    id: 'sgd.shuffle',
    kind: 'choice',
    concepts: ['epoch'],
    prompt: 'A dataset is stored sorted by class: all cats first, then all dogs. What goes wrong if you train in batches without shuffling?',
    options: [
      { text: 'Each batch contains only one class, so every gradient is biased and training lurches between “everything is a cat” and “everything is a dog”.', correct: true, why: 'The poll only works if the sample is random. Shuffling every epoch keeps each batch representative.' },
      { text: 'Nothing: the full epoch still sees every example.', why: 'It sees every example, but each *step* is taken on a wildly unrepresentative batch, and the order of steps matters.' },
      { text: 'Training becomes slower but more accurate.', why: 'It becomes erratic, not more accurate.' },
    ],
  },
];
