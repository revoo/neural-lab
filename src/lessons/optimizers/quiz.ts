import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'optim.momentum-why',
    kind: 'choice',
    concepts: ['momentum'],
    prompt: 'In a narrow valley, why does momentum make faster progress than plain gradient descent?',
    options: [
      { text: 'Slopes across the valley alternate in sign and cancel in the velocity, while slopes along the floor all agree and add up.', correct: true, why: 'The velocity is a running sum of recent slopes. Consistent pushes accumulate (up to 1/(1−β) times); back-and-forth pushes cancel. That is exactly the cure for zig-zagging.' },
      { text: 'Momentum computes a more accurate gradient.', why: 'It uses the same gradients — it just remembers and combines them differently.' },
      { text: 'Momentum automatically picks a smaller step size in steep directions.', why: 'That’s closer to what RMSProp does. Momentum has one step size; its advantage comes from accumulating velocity.' },
    ],
  },
  {
    id: 'optim.momentum-factor',
    kind: 'numeric',
    concepts: ['momentum'],
    tolerance: 0.01,
    make: (r) => {
      const beta = [0.5, 0.8, 0.9, 0.95, 0.99][r.int(5)];
      const ans = Math.round((1 / (1 - beta)) * 100) / 100;
      return {
        prompt: `Momentum with $\\beta = ${beta}$ keeps meeting the same slope, step after step. Its velocity settles at how many times that slope?`,
        answer: ans,
        explain: `The velocity stops growing when $v = \\beta v + g$, i.e. $v = \\frac{g}{1 - \\beta} = \\frac{g}{${(1 - beta).toFixed(2)}} = ${ans}\\,g$. Along a consistent direction, momentum acts like a step size ${ans}× bigger.`,
      };
    },
  },
  {
    id: 'optim.rmsprop',
    kind: 'choice',
    concepts: ['adam'],
    prompt: 'One weight’s slopes are typically around 0.001; another’s are around 10. With RMSProp, roughly how big are their steps?',
    options: [
      { text: 'About the same size: each is divided by the root of its own recent squared slopes.', correct: true, why: 'Dividing a slope by its typical size gives a number around ±1, so every weight moves by roughly the step size. That’s why RMSProp and Adam cross plateaus quickly.' },
      { text: 'The second moves 10,000 times further.', why: 'That’s what plain gradient descent would do — steps proportional to the slope.' },
      { text: 'The first moves 10,000 times further.', why: 'RMSProp equalises step sizes; it doesn’t reverse them.' },
    ],
  },
  {
    id: 'optim.tilt',
    kind: 'choice',
    concepts: ['adam'],
    prompt: 'Why did RMSProp-style scaling help so little on the tilted narrow valley?',
    options: [
      { text: 'It rescales each weight separately, but the valley’s steep and gentle directions are diagonal mixtures of the two weights.', correct: true, why: 'Per-weight scaling can stretch the x and y axes, not rotate them. When the awkward directions line up with individual weights, it helps a lot; when they’re mixtures, much less.' },
      { text: 'RMSProp doesn’t work on valleys.', why: 'It works fine on valleys aligned with the axes. The issue is the tilt.' },
      { text: 'The step size was too small.', why: 'Each optimizer used its best step size from a search. The limitation is structural.' },
    ],
  },
  {
    id: 'optim.bias',
    kind: 'choice',
    concepts: ['adam'],
    prompt: 'What problem does Adam’s “bias correction” (dividing by 1 − βᵗ) solve?',
    options: [
      { text: 'Its running averages start at zero, so for the first steps they badly underestimate the true slope sizes.', correct: true, why: 'After one step, m = 0.1·g with β₁ = 0.9. Dividing by 1 − 0.9¹ = 0.1 restores g. The correction fades as t grows.' },
      { text: 'Gradients are biased towards positive values.', why: 'Gradients can be either sign; the bias is in the averages’ warm-up, not in the gradients.' },
      { text: 'It removes the bias term from the network.', why: 'Unrelated: “bias” here is statistical bias in an estimate, not the neuron’s bias.' },
    ],
  },
  {
    id: 'optim.memory',
    kind: 'numeric',
    concepts: ['adam', 'parameter'],
    tolerance: 0.01,
    make: (r) => {
      const b = [1, 3, 7, 13][r.int(4)];
      return {
        prompt: `A model has ${b} billion parameters. Adam keeps two extra numbers per parameter, each a 4-byte float. How many GB do those optimizer states take? (1 GB = 10⁹ bytes)`,
        answer: b * 8,
        explain: `$${b} \\times 10^9$ parameters × 2 numbers × 4 bytes $= ${b * 8}$ GB — on top of the weights and gradients themselves.`,
      };
    },
  },
  {
    id: 'optim.no-winner',
    kind: 'choice',
    concepts: ['adam', 'momentum'],
    prompt: 'Which conclusion does the optimizer race support?',
    options: [
      { text: 'Each trick fixes a particular kind of trouble; Adam is a robust default rather than a universal winner.', correct: true, why: 'Momentum shone in the valley, adaptive scaling on the plateau, and the combination on the banana. Real landscapes mix all of these, which is why a method that is rarely bad is so valuable.' },
      { text: 'Adam always wins.', why: 'It tied the valley only with a huge step size, and RMSProp matched it on the plateau.' },
      { text: 'Plain gradient descent is best once tuned.', why: 'Even at its best step size, it was the slowest on every landscape.' },
    ],
  },
];
