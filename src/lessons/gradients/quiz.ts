import type { Question } from '../../components/quizTypes';

const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);

export const questions: Question[] = [
  {
    id: 'gradients.partial',
    kind: 'numeric',
    concepts: ['partial-derivative'],
    make: (r) => {
      const a = r.int(4) + 1;
      const x = r.int(5) - 2;
      const y = r.int(5) - 2;
      return {
        prompt: `Let $f(x, y) = ${a === 1 ? '' : a}x^2 y + 5y$. What is $\\frac{\\partial f}{\\partial x}$ at the point $(${x}, ${y})$?`,
        answer: 2 * a * x * y,
        explain: `Treat $y$ as a constant and differentiate with respect to $x$ only. The term $5y$ doesn’t involve $x$, so it contributes 0. The term $${a === 1 ? '' : a}x^2y$ behaves like a constant $(${a === 1 ? '' : a}y)$ times $x^2$, giving $${2 * a}xy$. At $(${x}, ${y})$: $${2 * a} \\times ${p(x)} \\times ${p(y)} = ${2 * a * x * y}$.`,
      };
    },
  },
  {
    id: 'gradients.vector',
    kind: 'choice',
    concepts: ['gradient'],
    prompt: 'For $f(x, y) = x^2 + 3y$, what is the gradient $\\nabla f$ at the point $(2, 5)$?',
    options: [
      { text: '$[4, 3]$', correct: true, why: '$\\partial f/\\partial x = 2x = 4$ and $\\partial f/\\partial y = 3$. Stack them in order: $\\nabla f = [4, 3]$.' },
      { text: '$[2, 5]$', why: 'That’s the *point*, not the gradient. The gradient is made of the partial derivatives evaluated at the point.' },
      { text: '$19$', why: 'That’s the function’s *value* there ($4 + 15$). The gradient is a vector of slopes, one per input.' },
      { text: '$[4, 15]$', why: 'The derivative of $3y$ with respect to $y$ is $3$, not $3y$ — the slope of a straight line doesn’t depend on where you are.' },
    ],
  },
  {
    id: 'gradients.direction',
    kind: 'choice',
    concepts: ['gradient', 'gradient-descent'],
    prompt: 'You are minimising a loss $L(w_1, w_2)$ and compute $\\nabla L = [3, -1]$. Which update moves *downhill* most steeply (for a small step size $\\eta$)?',
    options: [
      { text: '$[w_1, w_2] \\leftarrow [w_1, w_2] - \\eta\\,[3, -1]$', correct: true, why: 'The gradient points uphill; its negative points downhill. So $w_1$ decreases and $w_2$ increases — exactly opposite to the signs of the partial derivatives.' },
      { text: '$[w_1, w_2] \\leftarrow [w_1, w_2] + \\eta\\,[3, -1]$', why: 'Adding the gradient walks *uphill* and increases the loss. That’s gradient *ascent*, which is what you’d want if maximising.' },
      { text: 'Decrease both weights.', why: 'The partial derivative for $w_2$ is negative, meaning the loss falls as $w_2$ *increases*. Each weight moves against the sign of its own partial derivative.' },
    ],
  },
  {
    id: 'gradients.contours',
    kind: 'choice',
    concepts: ['gradient'],
    prompt: 'On a contour map of a loss function, how does the gradient arrow relate to the contour line through that point?',
    options: [
      { text: 'It’s perpendicular to the contour, pointing towards higher values.', correct: true, why: 'Moving *along* a contour doesn’t change the value at all — that’s what a contour is. The direction of fastest change must therefore be at right angles to it.' },
      { text: 'It runs along the contour.', why: 'Along a contour the value is constant, so that’s the direction of *zero* change — the opposite of steepest.' },
      { text: 'It always points towards the origin.', why: 'Only for special functions like the round bowl. In the “two hills” landscape, gradients point towards whichever peak is uphill from you.' },
    ],
  },
  {
    id: 'gradients.saddle',
    kind: 'choice',
    concepts: ['gradient'],
    prompt: 'An optimizer reports that the gradient of the loss is exactly zero. Is it guaranteed to be at a minimum?',
    options: [
      { text: 'No — it could be a maximum or a saddle point, where the surface curves up in some directions and down in others.', correct: true, why: 'A zero gradient only means no direction has any *first-order* slope. In high-dimensional networks, saddle points vastly outnumber true minima, which is part of why optimisers use momentum and noise.' },
      { text: 'Yes — zero gradient means the loss can’t be reduced further.', why: 'Try the saddle $x^2 - y^2$ at the origin: the gradient is zero, but moving along $y$ reduces $f$.' },
    ],
  },
];
