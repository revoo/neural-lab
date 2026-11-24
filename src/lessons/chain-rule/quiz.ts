import type { Question } from '../../components/quizTypes';

const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);

export const questions: Question[] = [
  {
    id: 'chain.product',
    kind: 'numeric',
    concepts: ['chain-rule'],
    make: (r) => {
      const d1 = [2, 3, -1, 0.5, 4][r.int(5)];
      const d2 = [3, -2, 0.5, 2, 1.5][r.int(5)];
      const d3 = [0.5, 2, -3, 4, 1][r.int(5)];
      const ans = d1 * d2 * d3;
      return {
        prompt: `A pipeline computes $u$ from $x$, then $v$ from $u$, then $y$ from $v$. At the current values, $\\frac{du}{dx} = ${d1}$, $\\frac{dv}{du} = ${d2}$ and $\\frac{dy}{dv} = ${d3}$. What is $\\frac{dy}{dx}$?`,
        answer: ans,
        explain: `Sensitivities along a chain multiply, like gear ratios: $${p(d1)} \\times ${p(d2)} \\times ${p(d3)} = ${ans}$.`,
      };
    },
  },
  {
    id: 'chain.compose',
    kind: 'choice',
    concepts: ['chain-rule'],
    prompt: 'What is the derivative of $y = (3x + 1)^2$ with respect to $x$?',
    options: [
      { text: '$6(3x + 1)$', correct: true, why: 'Let $u = 3x + 1$, so $y = u^2$. Then $\\frac{dy}{du} = 2u$ and $\\frac{du}{dx} = 3$. Multiply: $2u \\cdot 3 = 6(3x+1)$.' },
      { text: '$2(3x + 1)$', why: 'That’s only the outer derivative $\\frac{dy}{du}$. It forgets that $u$ itself changes 3× as fast as $x$ — the inner link of the chain.' },
      { text: '$6x$', why: 'This treats the expression as if it were $3x^2$. Expanding gives $9x^2 + 6x + 1$, whose derivative is $18x + 6 = 6(3x+1)$.' },
      { text: '$3$', why: 'That’s only the inner derivative $\\frac{du}{dx}$. The squaring stage amplifies changes too.' },
    ],
  },
  {
    id: 'chain.paths',
    kind: 'choice',
    concepts: ['chain-rule', 'gradient-accumulation'],
    prompt: 'In a computational graph, a variable $x$ feeds into two different branches that later recombine into the output $y$. How do you get $\\frac{dy}{dx}$?',
    options: [
      { text: 'Multiply the local derivatives along each path, then add the paths together.', correct: true, why: 'Each path is a separate route by which a nudge to $x$ reaches $y$. Their effects add up. In backpropagation this is called **gradient accumulation**, and it’s why gradients are added (`+=`) rather than assigned.' },
      { text: 'Multiply all the local derivatives in the whole graph together.', why: 'Multiplication is for links *in series* along one path. Parallel paths are separate contributions, so they add.' },
      { text: 'Use only the larger of the two paths.', why: 'Both routes carry influence. Dropping one gives the wrong answer — check with the two-path demo.' },
    ],
  },
  {
    id: 'chain.vanishing',
    kind: 'choice',
    concepts: ['vanishing-gradients'],
    prompt: 'A network has 20 layers in a chain, and the local derivative through each layer is about 0.25. Roughly how does a nudge to a first-layer weight affect the loss, compared with a last-layer weight?',
    options: [
      { text: 'Its effect is about $0.25^{19} \\approx 4 \\times 10^{-12}$ times as large — practically zero.', correct: true, why: 'The chain rule multiplies the local factors, and multiplying many numbers below 1 shrinks exponentially. The early layers get almost no gradient and barely learn: the **vanishing gradient** problem. (0.25 is the *maximum* slope of the sigmoid, so this happened readily in early deep networks.)' },
      { text: 'About $20 \\times 0.25 = 5$ times as large.', why: 'Factors along a chain multiply, they don’t add. Adding is only for parallel paths.' },
      { text: 'The same — every weight matters equally.', why: 'Each layer scales the signal passing back through it. After 19 layers of ×0.25, very little remains.' },
    ],
  },
];
