import type { Question } from '../../components/quizTypes';

const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);

export const questions: Question[] = [
  {
    id: 'backprop.rule',
    kind: 'numeric',
    concepts: ['backpropagation', 'chain-rule'],
    make: (r) => {
      const gOut = [2, -1.5, 0.5, 3][r.int(4)];
      const other = [4, -2, 0.5, 3][r.int(4)];
      return {
        prompt: `In a graph, a node $m = x \\times w$ has gradient $\\partial L / \\partial m = ${gOut}$. The input $x$ is $${other}$. What is $\\partial L / \\partial w$?`,
        answer: gOut * other,
        explain: `Take the gradient arriving from downstream and multiply by the local ratio. For a multiplication, the local ratio with respect to one input is **the other input**: $${p(gOut)} \\times ${p(other)} = ${gOut * other}$.`,
      };
    },
  },
  {
    id: 'backprop.add',
    kind: 'choice',
    concepts: ['backpropagation'],
    prompt: 'A node computes $s = a + b$ and receives gradient $\\partial L/\\partial s = 0.7$. What gradients does it pass back to $a$ and $b$?',
    options: [
      { text: '0.7 to each of them.', correct: true, why: 'Nudging $a$ by some amount nudges $s$ by the same amount, so the local ratio is 1. An addition node simply copies its gradient to every input — it “routes” the signal unchanged.' },
      { text: '0.35 to each — the gradient is split between them.', why: 'It isn’t shared out. Each input, nudged on its own, moves $s$ one-for-one, so each receives the full 0.7.' },
      { text: '0.7 to $a$ and 0 to $b$.', why: 'Addition treats both inputs symmetrically; there’s no reason one would get nothing.' },
    ],
  },
  {
    id: 'backprop.order',
    kind: 'order',
    concepts: ['backpropagation', 'forward-pass', 'backward-pass'],
    prompt: 'Put one training step in the right order.',
    items: [
      'Run the forward pass, computing and storing every intermediate value',
      'Compute the loss by comparing the output with the target',
      'Seed the backward pass: the loss’s gradient with respect to itself is 1',
      'Walk the graph backwards, multiplying by local derivatives and adding across paths',
      'Update every parameter: subtract learning rate × its gradient',
    ],
    explain: 'Forward to get values (and remember them), backward to get gradients (which need those remembered values), then the optimizer steps. Then repeat with the next batch.',
  },
  {
    id: 'backprop.memory',
    kind: 'choice',
    concepts: ['backpropagation'],
    prompt: 'Why does training a network use much more memory than simply running it?',
    options: [
      { text: 'The backward pass needs the intermediate values from the forward pass, so they must all be kept until it runs.', correct: true, why: 'Local derivatives use forward values — tanh’s slope needs its output, a multiplication’s needs the other input. Running a model can throw each layer’s activations away once the next layer is computed; training can’t.' },
      { text: 'Gradients are much larger numbers than weights.', why: 'Gradients have the same shape as the weights; their size in memory doesn’t depend on how big the numbers are.' },
      { text: 'Backpropagation makes a copy of the model for every weight.', why: 'That’s closer to what *nudging* each weight would require. Backpropagation needs one set of stored activations and one gradient per parameter.' },
    ],
  },
  {
    id: 'backprop.not-optimizer',
    kind: 'choice',
    concepts: ['backpropagation', 'gradient-descent'],
    prompt: 'Which statement is accurate?',
    options: [
      { text: 'Backpropagation computes the gradients; a separate optimizer (like SGD or Adam) decides how to use them to change the weights.', correct: true, why: 'They’re often lumped together, but they’re different jobs. Backprop answers “how does the loss respond to each weight?”. The optimizer answers “so how far should each weight move?”.' },
      { text: 'Backpropagation is the rule that updates the weights.', why: 'Backprop stops once every gradient is known. The update rule — plain gradient descent, momentum, Adam — is a separate choice.' },
      { text: 'Backpropagation is an approximation of the gradient.', why: 'It computes the exact gradient (up to floating-point rounding). Numerical nudging is the approximate method.' },
    ],
  },
];
