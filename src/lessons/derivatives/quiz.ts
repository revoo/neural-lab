import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'derivatives.power',
    kind: 'numeric',
    concepts: ['derivative'],
    make: (r) => {
      const c = r.int(4) + 1;
      const x = r.int(7) - 3;
      return {
        prompt: `Let $f(x) = ${c === 1 ? '' : c}x^2$. What is $f'(${x})$, the slope of the curve at $x = ${x}$?`,
        answer: 2 * c * x,
        explain: `The derivative of $x^2$ is $2x$, and constant multiples carry through, so $f'(x) = ${2 * c}x$. At $x = ${x}$: $${2 * c} \\times ${x < 0 ? `(${x})` : x} = ${2 * c * x}$.`,
      };
    },
  },
  {
    id: 'derivatives.linear-approx',
    kind: 'numeric',
    concepts: ['derivative', 'linear-approximation'],
    tolerance: 0.001,
    make: (r) => {
      const fx = r.int(9) + 1;
      const d = r.int(9) - 4;
      const dx = [0.1, 0.01, 0.2][r.int(3)];
      const ans = Math.round((fx + d * dx) * 1000) / 1000;
      return {
        prompt: `You know that $f(2) = ${fx}$ and $f'(2) = ${d}$. Estimate $f(${2 + dx})$ without knowing anything else about $f$.`,
        answer: ans,
        explain: `For a small nudge, output change ≈ slope × input change: $f(2 + ${dx}) \\approx f(2) + f'(2) \\times ${dx} = ${fx} + ${d < 0 ? `(${d})` : d} \\times ${dx} = ${ans}$. This “local straight-line” estimate is exactly what gradient descent relies on.`,
      };
    },
  },
  {
    id: 'derivatives.zero',
    kind: 'choice',
    concepts: ['derivative'],
    prompt: 'At some point, $f\'(x) = 0$. What can you conclude?',
    options: [
      { text: 'The curve is momentarily flat there — it could be a minimum, a maximum, or a flat spot on the way up or down.', correct: true, why: 'A zero derivative only says that tiny nudges don’t change the output (to first order). $x^2$ has a minimum at 0, $-x^2$ has a maximum, and $x^3$ has neither — all three have zero slope at $x = 0$.' },
      { text: 'It must be a minimum.', why: 'Minima do have zero slope, but so do maxima and plateaus. Try the cubic $x^3 - 3x$ in the explorer: its slope is 0 at both its peak and its valley.' },
      { text: 'The function equals zero there.', why: 'The derivative is about the *slope*, not the *value*. $x^2 + 5$ has zero slope at $x = 0$ while its value is 5.' },
      { text: 'The function is constant everywhere.', why: 'Only a derivative that is zero *everywhere* implies a constant function. Here we only know about one point.' },
    ],
  },
  {
    id: 'derivatives.training-sign',
    kind: 'choice',
    concepts: ['derivative', 'gradient-descent'],
    prompt: 'While training a model, you compute that the derivative of the loss with respect to one weight is $-4$. To reduce the loss, what should you do to that weight?',
    options: [
      { text: 'Increase it slightly.', correct: true, why: 'A negative slope means the loss goes *down* as the weight goes *up*. So nudge the weight upward. Moving against the sign of the derivative is the whole idea of gradient descent.' },
      { text: 'Decrease it slightly.', why: 'A slope of −4 means that increasing the weight by a small amount ε changes the loss by about −4ε — a decrease. Decreasing the weight would *raise* the loss.' },
      { text: 'Set it to −4.', why: 'The derivative tells you a direction and a sensitivity, not a destination. The best value for the weight could be anywhere.' },
      { text: 'Leave it alone; negative derivatives mean the weight is already optimal.', why: 'At an optimum the derivative would be 0. A slope of −4 is quite steep: there’s loss to be gained by moving.' },
    ],
  },
  {
    id: 'derivatives.float',
    kind: 'choice',
    concepts: ['numerical-derivative'],
    prompt: 'You estimate a derivative in code as `(f(x + h) - f(x)) / h`. Why does making `h = 1e-15` give a *worse* answer than `h = 1e-6`?',
    options: [
      { text: 'Floating-point numbers only keep about 16 significant digits, so subtracting two nearly equal values wipes out most of the precision.', correct: true, why: 'This is called *catastrophic cancellation*. The rounding error in each $f$ value is roughly $10^{-16}$, and dividing it by a tiny $h$ magnifies it enormously. Mathematically smaller is better; numerically there’s a sweet spot.' },
      { text: 'Because the true derivative changes when h gets that small.', why: 'The true derivative is a fixed number at each point; it doesn’t depend on h. The *estimate* is what goes wrong.' },
      { text: 'It doesn’t — smaller h is always more accurate.', why: 'In exact arithmetic, yes. In 64-bit floating point, try $h = 10^{-15}$ in the explorer and look at the error.' },
    ],
  },
];
