import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'lab.features',
    kind: 'choice',
    concepts: ['neuron', 'linearly-separable'],
    prompt: 'A single neuron fails on the circle with inputs x and y, but succeeds with inputs x² and y². Why?',
    options: [
      { text: 'It still draws a straight line, but in the space of x² and y², where “inside the circle” (x² + y² small) is a straight-line cut.', correct: true, why: 'The neuron didn’t get smarter; its inputs did. Choosing inputs that make the problem linearly separable is feature engineering — and hidden layers learn such features automatically.' },
      { text: 'Squared inputs make the neuron non-linear, so it can draw curves.', why: 'The neuron is exactly as linear as before in its inputs. The curve you see on the map is a straight line in (x², y²) space.' },
      { text: 'Squaring removes the noise from the data.', why: 'The noise is still there; the inputs just describe each point in a more useful way.' },
    ],
  },
  {
    id: 'lab.test-set',
    kind: 'choice',
    concepts: ['loss'],
    prompt: 'Why does the lab keep some points aside as a test set that the network never trains on?',
    options: [
      { text: 'Because what matters is performance on new data, and the training points can’t tell you that: a big network can fit them, noise and all.', correct: true, why: 'Training accuracy measures fit; test accuracy measures whether the model learned the pattern. Experiment 4 showed them diverging.' },
      { text: 'To make training faster.', why: 'Holding data back makes training slightly *less* informed, not faster. Its purpose is honest evaluation.' },
      { text: 'Because the test points are harder.', why: 'They come from exactly the same source as the training points, chosen at random.' },
    ],
  },
  {
    id: 'lab.overfit',
    kind: 'choice',
    concepts: ['loss'],
    prompt: 'During training, the training loss keeps falling while the test loss starts rising. What is happening?',
    options: [
      { text: 'The network is memorising the training points (including their noise) at the expense of the underlying pattern: overfitting.', correct: true, why: 'Each step makes the training points fit better, but the decision boundary grows islands around individual noisy points, which hurts on new data.' },
      { text: 'The step size is too small.', why: 'A small step size slows both curves down; it doesn’t make them move in opposite directions.' },
      { text: 'The test set is labelled wrongly.', why: 'The test points come from the same source as the training points. The divergence comes from the model, not the labels.' },
    ],
  },
  {
    id: 'lab.saturate',
    kind: 'choice',
    concepts: ['learning-rate', 'activation-function'],
    prompt: 'With momentum at a step size of 1, the moons network stalls near 50% and its hidden neurons’ maps turn hard black-and-white. What went wrong?',
    options: [
      { text: 'Huge early steps made the weights large, so the tanh neurons saturate at ±1 where their slope is nearly zero, and almost no learning signal flows back.', correct: true, why: 'Backpropagation multiplies by each activation’s slope. On the flat parts of tanh, that slope is almost zero, so the network freezes — a quiet failure with the same cause as divergence.' },
      { text: 'The loss exploded to infinity.', why: 'It didn’t — the weights stayed finite. That’s what makes this failure easy to miss.' },
      { text: 'Momentum can’t train tanh networks.', why: 'At a step size of 0.03 the same network with momentum reaches 100%. The problem is the step size.' },
    ],
  },
  {
    id: 'lab.representable',
    kind: 'choice',
    concepts: ['neuron'],
    prompt: 'Two hidden neurons can represent XOR exactly, yet the search found that networks with three trained to 95% far more reliably. What’s the lesson?',
    options: [
      { text: 'Being able to represent a solution isn’t the same as training finding it; spare capacity makes the search easier.', correct: true, why: 'With no slack, gradient descent often gets stuck with neurons in the wrong roles. This is a big reason real networks are made larger than strictly necessary.' },
      { text: 'The logic lab was wrong: XOR needs three neurons.', why: 'Two neurons really do suffice — you can set them by hand. The difficulty is in *learning* them.' },
      { text: 'Bigger networks always generalise better.', why: 'Not always — experiment 4 showed a big network memorising. The point here is about ease of training.' },
    ],
  },
  {
    id: 'lab.params',
    kind: 'numeric',
    concepts: ['parameter'],
    tolerance: 0,
    make: (r) => {
      const f = [2, 3, 4][r.int(3)];
      const h = [[3], [2], [4, 2], [5]][r.int(4)];
      const sizes = [f, ...h, 1];
      const total = sizes.slice(1).reduce((t, n, l) => t + sizes[l] * n + n, 0);
      return {
        prompt: `In the lab, a network has ${f} input features, hidden layers of ${h.join(' and ')} neuron${h.length > 1 || h[0] > 1 ? 's' : ''}, and 1 output. How many parameters does the lab report?`,
        answer: total,
        explain: `Layer by layer, inputs × neurons + neurons: ${sizes
          .slice(1)
          .map((n, l) => `$${sizes[l]} \\times ${n} + ${n} = ${sizes[l] * n + n}$`)
          .join(', ')}; total $${total}$.`,
      };
    },
  },
];
