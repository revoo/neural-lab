import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'depth.params',
    kind: 'numeric',
    concepts: ['parameter'],
    tolerance: 0,
    make: (r) => {
      const sizes = [
        [10, 20, 5],
        [784, 32, 10],
        [4, 16, 16, 3],
        [100, 50, 50, 1],
      ][r.int(4)];
      const parts = sizes.slice(1).map((n, l) => sizes[l] * n + n);
      const total = parts.reduce((a, b) => a + b, 0);
      return {
        prompt: `A fully connected network has layer sizes ${sizes.join(' → ')} (input first). How many parameters does it have?`,
        answer: total,
        explain: `For each layer, inputs × neurons + neurons: ${sizes
          .slice(1)
          .map((n, l) => `$${sizes[l]} \\times ${n} + ${n} = ${sizes[l] * n + n}$`)
          .join(', ')}. Total: $${total}$.`,
      };
    },
  },
  {
    id: 'depth.double-width',
    kind: 'choice',
    concepts: ['parameter', 'layer'],
    prompt: 'A network has many hidden layers, all 1,000 neurons wide. Roughly what happens to its parameter count if you make every hidden layer 2,000 wide?',
    options: [
      { text: 'It roughly quadruples.', correct: true, why: 'A layer between two hidden layers has inputs × neurons weights, and both numbers doubled: 2 × 2 = 4 times as many. Only the first and last layers (tied to the fixed input and output sizes) merely double.' },
      { text: 'It roughly doubles.', why: 'That’s what doubling the *depth* does. Doubling the width doubles both sides of every hidden-to-hidden weight matrix.' },
      { text: 'It stays about the same.', why: 'Every weight connects a pair of neurons, so more neurons per layer means many more pairs.' },
    ],
  },
  {
    id: 'depth.fold-count',
    kind: 'numeric',
    concepts: ['layer'],
    tolerance: 0,
    make: (r) => {
      const L = [3, 5, 6, 7][r.int(4)];
      return {
        prompt: `Each folding layer (2 ReLU neurons) folds the interval in half. How many straight pieces does a stack of ${L} folding layers produce?`,
        answer: 2 ** L,
        explain: `Every layer folds every existing piece, so the count doubles each time: $2^{${L}} = ${2 ** L}$. A single wide layer would need at least ${2 ** L - 1} neurons for that many pieces.`,
      };
    },
  },
  {
    id: 'depth.why-doubles',
    kind: 'choice',
    concepts: ['layer'],
    prompt: 'Why does adding a folding layer *double* the number of pieces, while adding a neuron to a wide layer adds only one?',
    options: [
      { text: 'The new layer’s fold is applied to everything the earlier layers produced, so it acts on every existing piece at once.', correct: true, why: 'Like one cut through a stack of folded paper: the later layer’s work is reused across every region the earlier layers created. A wide layer’s neurons all act on the raw input, so each contributes only its own single kink.' },
      { text: 'Deeper layers have bigger weights.', why: 'Every folding layer has the same weights. The doubling comes from what the layer is applied to, not from its size.' },
      { text: 'Each folding layer has two neurons, and 2 neurons make 2 pieces.', why: 'Two neurons in a *wide* layer would add only two kinks. In a stack, the same two neurons double the existing count, whatever it is.' },
    ],
  },
  {
    id: 'depth.random',
    kind: 'choice',
    concepts: ['layer'],
    prompt: 'A deep ReLU network *could* represent a function with exponentially many pieces. What did the random-network experiment show about typical networks?',
    options: [
      { text: 'With ordinary random weights, deep networks have roughly one bend per neuron or fewer — nowhere near the exponential maximum.', correct: true, why: 'The exponential construction needs carefully arranged weights. Depth gives the *capacity*; whether a trained network uses it depends on the data and the training.' },
      { text: 'Random deep networks always reach the maximum number of pieces.', why: 'Try the widget: the averages are tiny compared with the ceiling.' },
      { text: 'Deep networks have fewer pieces than they have parameters, so depth is useless.', why: 'Depth’s value is the *possibility* of reuse and composition, which training exploits when the data has that structure. Typical behaviour at random initialisation doesn’t rule that out.' },
    ],
  },
  {
    id: 'depth.bottleneck',
    kind: 'choice',
    concepts: ['layer'],
    prompt: 'A classifier for 2D points has a hidden layer of just **one** neuron somewhere in the middle. What does that do to every layer after it?',
    options: [
      { text: 'They only ever see one number per input, so any information that didn’t fit into that number is gone for good.', correct: true, why: 'A layer can only pass on what its neurons output. Squeeze through a width-1 layer and two inputs that give the same value there look identical to every later layer, however wide or deep it is.' },
      { text: 'Nothing much — later layers can be made wider to compensate.', why: 'Later layers can only process what reaches them. Width after a bottleneck can’t recover information lost at it.' },
      { text: 'It makes the network faster without changing what it can compute.', why: 'It is cheaper, but it also limits what the network can represent.' },
    ],
  },
  {
    id: 'depth.rosetta',
    kind: 'choice',
    concepts: ['layer', 'parameter'],
    prompt: 'When you write a 1D convolution, an unrolled linear RNN, LoRA, and self-attention as effective `N × N` mixing matrices `W`, what sets self-attention apart from the other three?',
    options: [
      { text: 'Its effective mixing matrix W(x) is recomputed dynamically from the input x on every forward pass, rather than staying fixed after training.', correct: true, why: 'Convolution shares weights across space, RNNs share weights across time, and LoRA constrains the weight update to low rank — all three still use fixed weights at inference. Self-attention builds its mixing weights live from query–key dot products of the input.' },
      { text: 'It uses fewer than 36 entries by zeroing out every off-diagonal cell.', why: 'Every token can attend to every other token (or every earlier token with a causal mask), so the attention matrix is generally dense, with rows summing to 1.' },
      { text: 'It does not multiply the inputs by weights at all.', why: 'Attention still takes a weighted sum of values — it is still a matrix multiplication W(x)·V, just with input-dependent weights.' },
    ],
  },
];
