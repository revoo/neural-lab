import type { Question } from '../../components/quizTypes';

const conceptQuestions: Question[] = [
  {
    id: 'activations.collapse',
    kind: 'choice',
    concepts: ['activation-function', 'linear-transformation'],
    prompt: 'A network has ten layers but no activation functions between them. Compared with a single layer, what can it do?',
    options: [
      { text: 'Exactly the same things — the ten layers collapse into one equivalent layer.', correct: true, why: 'Each layer is a linear transformation (plus a shift), and doing several of those in a row is still just one linear transformation (plus a shift). The extra layers add computation but no new ability.' },
      { text: 'Ten times as much.', why: 'It’s tempting to think more machinery means more capability. But stretching-and-rotating, done ten times, is still just one stretch-and-rotate — like the factory of multiplying machines that could be replaced by one machine.' },
      { text: 'It can learn curves, but only gentle ones.', why: 'Without activations, every layer keeps straight lines straight, so the whole network does too. Its decision boundary is always a single straight line (or flat plane).' },
    ],
  },
  {
    id: 'activations.xor-fix',
    kind: 'choice',
    concepts: ['linearly-separable', 'activation-function'],
    prompt: 'A single neuron can’t solve XOR. Which change lets a network solve it?',
    options: [
      { text: 'Add a hidden layer of a few neurons with a non-linear activation such as ReLU.', correct: true, why: 'Each hidden neuron draws its own line; the non-linearity lets the next layer *combine* the regions those lines create, building a boundary that isn’t one straight line.' },
      { text: 'Add more hidden layers, keeping them all linear.', why: 'Linear layers collapse into one, so the boundary stays a single line — you can verify this in the XOR trainer.' },
      { text: 'Train the single neuron for much longer.', why: 'The problem isn’t training time. No setting of a single neuron’s weights separates XOR, so no amount of training can find one.' },
      { text: 'Use a bigger learning rate.', why: 'Learning rate affects how fast you search, not what the model is capable of representing.' },
    ],
  },
  {
    id: 'activations.hinges',
    kind: 'choice',
    concepts: ['relu'],
    prompt: 'In the hinge builder, why does adding more ReLU neurons let the network follow the target curve more closely?',
    options: [
      { text: 'Each ReLU neuron adds one bend; with enough bends, a chain of straight pieces can follow any smooth curve as closely as you like.', correct: true, why: 'ReLU is “flat, then a straight slope”. Adding up many of them, each with its kink in a different place, gives a line that can change direction at each kink.' },
      { text: 'Each neuron memorises one point on the curve.', why: 'The neurons don’t store points. Each contributes a whole hinge shape across the entire input range; the curve emerges from adding them up.' },
      { text: 'ReLU is curved, so more of them makes a curvier result.', why: 'ReLU is made of two perfectly straight pieces. The curviness comes from adding many hinges together, not from any single one.' },
    ],
  },
  {
    id: 'activations.saturation',
    kind: 'choice',
    concepts: ['sigmoid', 'vanishing-gradients'],
    prompt: 'Why did ReLU largely replace sigmoid between the hidden layers of deep networks?',
    options: [
      { text: 'Sigmoid goes almost flat for large inputs, so very little learning signal passes back through it; ReLU keeps a steady slope for all positive inputs.', correct: true, why: 'Remember the chain rule lesson: learning signal is multiplied by each layer’s slope on the way back. Sigmoid’s slope is at most 0.25 and nearly zero at the ends, so in a deep stack the signal fades away. ReLU passes it through unchanged wherever the neuron is active.' },
      { text: 'ReLU is more accurate.', why: 'Neither is “more accurate” in itself. The difference is in how well networks using them can be *trained*, especially when deep.' },
      { text: 'Sigmoid can’t produce negative numbers, and networks need negative numbers.', why: 'Tanh produces negatives and has the same problem as sigmoid. And ReLU itself never outputs negatives. The real issue is flatness.' },
    ],
  },
];

const calcQuestions: Question[] = [
  {
    id: 'activations.math.collapse',
    kind: 'choice',
    concepts: ['matrix-multiplication'],
    prompt: 'Two linear layers compute $W_2(W_1\\mathbf{x})$. Which single matrix gives the same result?',
    options: [
      { text: '$W_2 W_1$', correct: true, why: 'Matrix products compose transformations, read right to left: $W_1$ acts first, then $W_2$. So $W_2(W_1\\mathbf{x}) = (W_2W_1)\\mathbf{x}$.' },
      { text: '$W_1 W_2$', why: 'Order matters for matrices. $W_1W_2$ would mean “do $W_2$ first”.' },
      { text: '$W_1 + W_2$', why: 'Chaining transformations multiplies their matrices; it doesn’t add them.' },
    ],
  },
  {
    id: 'activations.math.relu',
    kind: 'numeric',
    concepts: ['relu'],
    make: (r) => {
      const a = [2, -1, 0.5, 3][r.int(4)];
      const k = [-1, 0, 1][r.int(3)];
      const x = [-2, 0.5, 2, 3][r.int(4)];
      const v = a * Math.max(0, x - k);
      return {
        prompt: `One hinge in a ReLU network computes $${a}\\cdot\\mathrm{ReLU}(x - (${k}))$. What does it output at $x = ${x}$?`,
        answer: v,
        explain: `First $x - (${k}) = ${x - k}$. ReLU keeps it if positive, otherwise gives 0: $\\mathrm{ReLU}(${x - k}) = ${Math.max(0, x - k)}$. Times $${a}$: $${v}$.`,
      };
    },
  },
];

export const questions: Question[] = [...conceptQuestions, ...calcQuestions];
