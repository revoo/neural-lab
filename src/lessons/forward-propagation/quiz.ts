import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'forward.why-matmul',
    kind: 'choice',
    concepts: ['layer', 'matrix-multiplication'],
    prompt: 'Why can a whole layer of neurons be computed with a single matrix multiplication?',
    options: [
      { text: 'Every neuron in the layer reads the same inputs and works independently, so the layer is a stack of dot products — one row of the matrix per neuron.', correct: true, why: 'That’s the panel of judges: each judge (row) dots their own weights with the same input vector. Independence is what makes them batchable — and parallel on a GPU.' },
      { text: 'Because the neurons pass their results to each other in order.', why: 'They don’t: neurons within a layer never see each other’s outputs. If they did, they’d have to be computed one after another, and the layer couldn’t be one multiplication.' },
      { text: 'Because the activation function is linear.', why: 'The matrix multiplication only covers the weighing step. The activation is applied afterwards, to each number separately, and is usually *not* linear — that’s the point of it.' },
    ],
  },
  {
    id: 'forward.shape',
    kind: 'choice',
    concepts: ['shape', 'layer'],
    prompt: 'A layer has 128 neurons and reads a 784-number input (a flattened 28 × 28 image). What shape is its weight matrix, in the one-row-per-neuron convention used here (and by PyTorch)?',
    options: [
      { text: '128 × 784', correct: true, why: 'One row per neuron (128), one column per input (784). Each row is one neuron’s 784 preferences.' },
      { text: '784 × 128', why: 'That’s the transposed version, which appears when inputs are stacked as rows of a batch (`X @ W.T`). The stored matrix has one row per neuron.' },
      { text: '128 × 128', why: 'The matrix connects inputs to neurons, so one dimension must match the 784 inputs.' },
      { text: '784 × 784', why: 'That would be a layer with 784 neurons. The number of rows is the number of neurons.' },
    ],
  },
  {
    id: 'forward.params',
    kind: 'numeric',
    concepts: ['parameter', 'layer'],
    tolerance: 0,
    make: (r) => {
      const nin = [3, 4, 5, 10, 16][r.int(5)];
      const nout = [2, 6, 8, 20][r.int(4)];
      return {
        prompt: `A layer of ${nout} neurons reads ${nin} inputs. How many adjustable numbers (weights and biases) does it have?`,
        answer: nin * nout + nout,
        explain: `Each of the ${nout} neurons has one weight per input (${nin}) plus one bias: $${nout} \\times ${nin} + ${nout} = ${nin * nout + nout}$.`,
      };
    },
  },
  {
    id: 'forward.compute',
    kind: 'numeric',
    concepts: ['forward-pass', 'relu'],
    tolerance: 0.001,
    make: (r) => {
      const w = [[1, -1], [2, 0.5], [-1, 3]][r.int(3)];
      const x = [[3, 1], [2, -2], [1, 4]][r.int(3)];
      const b = [-1, 0.5, -6][r.int(3)];
      const z = w[0] * x[0] + w[1] * x[1] + b;
      const a = Math.max(0, z);
      return {
        prompt: `A ReLU neuron in some layer has weights $[${w.join(', ')}]$ and bias $${b}$. The previous layer’s outputs are $[${x.join(', ')}]$. What does this neuron output?`,
        answer: a,
        explain: `Weigh and add the bias: $${w[0]} \\times ${x[0]} + ${w[1]} \\times ${x[1] < 0 ? `(${x[1]})` : x[1]} + ${b < 0 ? `(${b})` : b} = ${z}$. Then respond: ReLU keeps positives and zeroes negatives, so the output is $${a}$.${z < 0 ? ' The tally was negative, so this neuron is silent for this input.' : ''}`,
      };
    },
  },
  {
    id: 'forward.batch',
    kind: 'choice',
    concepts: ['batch', 'shape'],
    prompt: 'A batch of 32 inputs, each with 784 numbers, goes through a layer of 128 neurons. What shape comes out?',
    options: [
      { text: '(32, 128) — one row of 128 neuron outputs per input', correct: true, why: '`(32, 784) @ (784, 128)` gives `(32, 128)`. The batch dimension rides along untouched; each row is one input’s trip through the layer.' },
      { text: '(128,) — the batch is averaged into one answer', why: 'The forward pass keeps every input separate. Averaging over the batch happens later, when the loss is computed.' },
      { text: '(32, 784) — the shape doesn’t change', why: 'The layer turns 784 numbers per input into 128 (one per neuron).' },
      { text: '(128, 784)', why: 'That’s the weight matrix’s shape, not the output’s.' },
    ],
  },
  {
    id: 'forward.dead-coast',
    kind: 'choice',
    concepts: ['layer', 'forward-pass'],
    prompt: 'In the island network, one coastline detector is stuck permanently at 0. Why does its whole island disappear from the final map, rather than just losing one edge?',
    options: [
      { text: 'The island detector downstream needs all three of its coastlines to say “yes” to clear its bias, and one of them now never does.', correct: true, why: 'The island neuron is effectively an AND of its three coastlines. Remove one vote permanently and the AND can never fire — so a fault in layer 1 deletes a feature in layer 2, and that shows up in layer 3.' },
      { text: 'Because the neuron’s weights were deleted.', why: 'Its weights are still there; its output is just forced to 0. What matters is how downstream neurons use that output.' },
      { text: 'Because every neuron in a network depends on every other one.', why: 'The other island survives untouched — no neuron on its path listens to the broken detector. Dependencies follow the connections.' },
    ],
  },
  {
    id: 'forward.order',
    kind: 'order',
    concepts: ['forward-pass'],
    prompt: 'Put the steps of computing one layer in the forward pass in order.',
    items: [
      'Take the previous layer’s outputs (or the input, for the first layer)',
      'Multiply by the layer’s weight matrix',
      'Add the layer’s biases',
      'Apply the activation function to each number',
      'Hand the result to the next layer as its input',
    ],
    explain: 'Mix, then bend: the matrix multiplication and bias mix every input into each neuron’s tally; the activation then acts on each tally separately. The result is the next layer’s input.',
  },
];
