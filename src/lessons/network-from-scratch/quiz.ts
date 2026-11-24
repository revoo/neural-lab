import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'network-from-scratch.dW-shape',
    kind: 'choice',
    concepts: ['backpropagation', 'shape'],
    prompt: 'A layer computes `Z = A @ W` with `A` of shape (32, 100) — a batch of 32 examples — and `W` of shape (100, 10). What is the shape of `dW`, and how is it computed?',
    options: [
      { text: '(100, 10), computed as `A.T @ dZ`.', correct: true, why: 'A gradient always has the shape of the thing it’s the gradient of. `A.T` is (100, 32) and `dZ` is (32, 10), so `A.T @ dZ` is (100, 10), summing input × blame over the 32 examples.' },
      { text: '(32, 10), the same as `Z`.', why: 'That’s the shape of `dZ`, the blame on the outputs. The weights’ gradient must match the weights, (100, 10).' },
      { text: '(32, 100, 10): one gradient per example.', why: 'Per-example gradients exist, but the matrix multiplication `A.T @ dZ` adds them up over the batch, which is what the update needs.' },
      { text: '(100, 10), computed as `dZ @ A`.', why: 'Right shape idea, but `dZ @ A` is (32, 10) @ (32, 100), which doesn’t even fit. The inputs must be transposed: `A.T @ dZ`.' },
    ],
  },
  {
    id: 'network-from-scratch.dA',
    kind: 'choice',
    concepts: ['backpropagation'],
    prompt: 'For the same layer, `Z = A @ W`, how does the blame flow back to the layer’s input `A`?',
    options: [
      { text: '`dA = dZ @ W.T` — each input receives the blame on the outputs, passed back along the same weights.', correct: true, why: 'Shapes: (32, 10) @ (10, 100) = (32, 100), the shape of `A`. Each input contributed to each output through one weight, so it gets that output’s blame times that weight.' },
      { text: '`dA = W @ dZ`', why: '(100, 10) @ (32, 10) doesn’t fit. And the blame must come out with one row per example, like `A`.' },
      { text: '`dA = A.T @ dZ`', why: 'That’s the weights’ gradient `dW`, shape (100, 10). The input’s gradient must be (32, 100).' },
    ],
  },
  {
    id: 'network-from-scratch.trained-anyway',
    kind: 'choice',
    concepts: ['backpropagation'],
    prompt: 'You hand-write a backward pass, train, and the loss goes down nicely. What does that prove about your gradients?',
    options: [
      { text: 'Not much: a wrong gradient often still points roughly downhill. Check it numerically against nudged-weight measurements.', correct: true, why: 'In the lab, a backward pass that forgot to divide by n, and one using the wrong loss’s slope, both still solved XOR. Only the gradient check tells right from almost-right.' },
      { text: 'That they are correct: wrong gradients make training diverge.', why: 'Some do, many don’t. A gradient that’s scaled wrongly, or slightly off in direction, can still reduce the loss — just less well than it should.' },
      { text: 'That they are correct up to a constant factor.', why: 'They could be wrong in direction as well as size, and still reduce the loss for a while.' },
    ],
  },
  {
    id: 'network-from-scratch.memory',
    kind: 'choice',
    concepts: ['backpropagation'],
    prompt: 'Why does training a network need much more memory than using it to make predictions?',
    options: [
      { text: 'The backward pass needs every layer’s forward outputs (like `h`), so they must all be kept until it runs.', correct: true, why: 'Lines like `dW2 = h.T @ dz2` and `dz1 = dh * (1 - h**2)` reuse the stored activations. For prediction you can throw each layer’s output away as soon as the next layer has used it.' },
      { text: 'Training uses more parameters than prediction.', why: 'The parameters are the same. The extra memory is for activations (and gradients and optimizer state).' },
      { text: 'The loss function is very large.', why: 'The loss is a single number.' },
    ],
  },
  {
    id: 'network-from-scratch.bias',
    kind: 'numeric',
    concepts: ['backpropagation'],
    tolerance: 0.001,
    make: (r) => {
      const a = [0.1, -0.2, 0.3, 0.05].map(() => Number((r.int(9) / 10 - 0.4).toFixed(1)));
      const s = a.reduce((t, v) => t + v, 0);
      return {
        prompt: `A bias \`b\` is added to every row of a batch: \`z = x @ W + b\`. The blame on one output column of \`z\`, for the 4 examples, is [${a.join(', ')}]. What is that bias’s gradient?`,
        answer: Number(s.toFixed(4)),
        explain: `The bias is used once per example, so its gradient is the blame summed over the batch: ${a.join(' + ')} = ${s.toFixed(2)}. That’s \`dz.sum(axis=0)\`.`,
      };
    },
  },
];
