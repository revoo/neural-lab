import type { Question } from '../../components/quizTypes';

const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);

export const questions: Question[] = [
  {
    id: 'neuron.z',
    kind: 'numeric',
    concepts: ['neuron', 'weighted-sum'],
    make: (r) => {
      const w = [r.int(7) - 3, r.int(7) - 3];
      const x = [r.int(5) - 2, r.int(5)];
      const b = r.int(7) - 3;
      const z = w[0] * x[0] + w[1] * x[1] + b;
      return {
        prompt: `A neuron has weights $\\mathbf{w} = [${w.join(', ')}]$ and bias $b = ${b}$, and uses ReLU. What does it output for the input $\\mathbf{x} = [${x.join(', ')}]$?`,
        answer: Math.max(0, z),
        explain: `First the weighted sum: $z = ${p(w[0])}\\times${p(x[0])} + ${p(w[1])}\\times${p(x[1])} + ${p(b)} = ${z}$. Then ReLU: $\\max(0, ${z}) = ${Math.max(0, z)}$.${z < 0 ? ' The evidence came out negative, so ReLU outputs zero: this neuron stays silent for this input.' : ''}`,
      };
    },
  },
  {
    id: 'neuron.bias',
    kind: 'choice',
    concepts: ['bias', 'decision-boundary'],
    prompt: 'You increase a neuron’s bias $b$ while keeping its weights fixed. What happens to its decision boundary $\\mathbf{w}\\cdot\\mathbf{x} + b = 0$?',
    options: [
      { text: 'It slides parallel to itself, without rotating.', correct: true, why: 'The weights set the boundary’s *direction* (it’s always perpendicular to $\\mathbf{w}$). The bias only changes the threshold, i.e. *how far* along $\\mathbf{w}$ the line sits. Raising $b$ makes it easier to reach $z > 0$, so the line moves back against $\\mathbf{w}$ and the class-1 region grows.' },
      { text: 'It rotates around the origin.', why: 'Rotation comes from changing the *direction* of $\\mathbf{w}$. The bias never changes the boundary’s orientation.' },
      { text: 'It gets sharper (more confident).', why: 'Sharpness comes from scaling the weights (and bias) up together. Changing $b$ alone just moves the line.' },
      { text: 'Nothing — the bias only affects the output value, not the boundary.', why: 'The boundary is defined as where $z = 0$, and $b$ is part of $z$. Try the bias slider in the boundary lab.' },
    ],
  },
  {
    id: 'neuron.scale',
    kind: 'choice',
    concepts: ['decision-boundary', 'sigmoid'],
    prompt: 'A sigmoid neuron has $\\mathbf{w} = [1, 2]$ and $b = -3$. You multiply all three numbers by 10. What changes?',
    options: [
      { text: 'The boundary stays exactly where it is, but the output switches from 0 to 1 much more abruptly across it.', correct: true, why: 'The boundary is where $\\mathbf{w}\\cdot\\mathbf{x} + b = 0$, and multiplying an equation that equals zero by 10 still gives zero. But away from the line, $z$ is now 10× larger, so the sigmoid saturates almost immediately: the neuron becomes far more confident. This is also why training keeps growing the weights on separable data.' },
      { text: 'The boundary moves 10× further from the origin.', why: 'Check a point on the old boundary: $1\\cdot1 + 2\\cdot1 - 3 = 0$. With everything ×10: $10 + 20 - 30 = 0$. Still on the boundary.' },
      { text: 'Nothing at all changes.', why: 'The *line* doesn’t change, but the outputs do: $\\sigma(10z)$ is much steeper than $\\sigma(z)$.' },
    ],
  },
  {
    id: 'neuron.perp',
    kind: 'choice',
    concepts: ['decision-boundary', 'dot-product'],
    prompt: 'Why is a neuron’s decision boundary always perpendicular to its weight vector $\\mathbf{w}$?',
    options: [
      { text: 'Moving along the boundary must leave $\\mathbf{w}\\cdot\\mathbf{x}$ unchanged, and the directions that don’t change a dot product with $\\mathbf{w}$ are exactly the ones perpendicular to it.', correct: true, why: 'Along the boundary, $z$ stays at 0. A step $\\Delta\\mathbf{x}$ changes $z$ by $\\mathbf{w}\\cdot\\Delta\\mathbf{x}$, which is zero exactly when the step is perpendicular to $\\mathbf{w}$. It’s the same reason gradients are perpendicular to contour lines — in fact $\\mathbf{w}$ *is* the gradient of $z$.' },
      { text: 'Because the bias forces it to be.', why: 'The bias only slides the line; it can’t affect its angle.' },
      { text: 'It’s a convention chosen when drawing diagrams.', why: 'It’s a mathematical consequence, not a convention — check it in the lab by rotating the arrow.' },
    ],
  },
  {
    id: 'neuron.limit',
    kind: 'choice',
    concepts: ['linearly-separable'],
    prompt: 'Which of these tasks can a single neuron (with a step or sigmoid activation) **not** solve perfectly?',
    options: [
      { text: 'Output 1 when exactly one of two binary inputs is 1 (XOR).', correct: true, why: 'The four XOR points can’t be split by any straight line: the two “1” cases sit in opposite corners. A single neuron can only draw one line, so at least one point is always misclassified.' },
      { text: 'Output 1 when both binary inputs are 1 (AND).', why: 'AND is linearly separable: e.g. $\\mathbf{w} = [1, 1]$, $b = -1.5$ puts only $(1, 1)$ on the positive side.' },
      { text: 'Output 1 when at least one input is 1 (OR).', why: 'OR is linearly separable: $\\mathbf{w} = [1, 1]$, $b = -0.5$ works.' },
      { text: 'Output 1 when the first input is larger than the second.', why: 'That’s the line $x_1 - x_2 = 0$: weights $[1, -1]$, bias 0.' },
    ],
  },
];
