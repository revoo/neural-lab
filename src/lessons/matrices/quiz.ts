import type { Question } from '../../components/quizTypes';

const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);

export const questions: Question[] = [
  {
    id: 'matrices.where-i-lands',
    kind: 'choice',
    concepts: ['basis-vectors', 'matrix'],
    prompt: 'For the matrix $M = \\begin{bmatrix} 3 & -1 \\\\ 2 & 4 \\end{bmatrix}$, where does the basis vector $\\hat{\\imath} = [1, 0]$ land?',
    options: [
      { text: '$[3, 2]$', correct: true, why: 'The first column of a matrix is exactly where $\\hat{\\imath}$ lands. Check: $M[1,0] = [3\\times1 + (-1)\\times0,\\; 2\\times1 + 4\\times0] = [3, 2]$.' },
      { text: '$[3, -1]$', why: 'That’s the first **row**. A common slip! Rows are the “judges” (each output is a row dotted with the input); **columns** record where the basis vectors land.' },
      { text: '$[-1, 4]$', why: 'That’s the second column — where $\\hat{\\jmath} = [0, 1]$ lands.' },
      { text: '$[1, 0]$ — basis vectors never move', why: 'Basis vectors move just like every other vector. In fact, the whole point is that tracking where *they* move tells you where *everything* moves.' },
    ],
  },
  {
    id: 'matrices.matvec',
    kind: 'numeric',
    concepts: ['matrix-vector'],
    make: (r) => {
      const m = [
        [r.int(7) - 3, r.int(7) - 3],
        [r.int(7) - 3, r.int(7) - 3],
      ];
      const v = [r.int(5) - 2, r.int(5) - 1];
      const out0 = m[0][0] * v[0] + m[0][1] * v[1];
      return {
        prompt: `Compute the **first** entry of $M\\mathbf{v}$ for $M = \\begin{bmatrix} ${m[0][0]} & ${m[0][1]} \\\\ ${m[1][0]} & ${m[1][1]} \\end{bmatrix}$ and $\\mathbf{v} = [${v[0]}, ${v[1]}]$.`,
        answer: out0,
        explain: `The first output entry is the first row dotted with $\\mathbf{v}$: $${p(m[0][0])}\\times${p(v[0])} + ${p(m[0][1])}\\times${p(v[1])} = ${out0}$. (Equivalently: $${p(v[0])}$ times the first column plus $${p(v[1])}$ times the second column, and read off the top entry.)`,
      };
    },
  },
  {
    id: 'matrices.not-linear',
    kind: 'choice',
    concepts: ['linear-transformation'],
    prompt: 'Which of these transformations of the plane can **not** be done by any 2×2 matrix?',
    options: [
      { text: 'Shift every point 1 unit to the right.', correct: true, why: 'A matrix always sends $[0, 0]$ to $[0, 0]$ (any matrix times the zero vector is zero), but a shift moves the origin. That’s why neurons add a separate **bias** term after multiplying by weights: the bias is what allows shifting.' },
      { text: 'Rotate everything by 45°.', why: 'Rotation is linear: grid lines stay straight and evenly spaced, and the origin stays put. Its matrix is $\\begin{bmatrix}\\cos 45° & -\\sin 45° \\\\ \\sin 45° & \\cos 45°\\end{bmatrix}$.' },
      { text: 'Mirror everything across the vertical axis.', why: 'Reflection is linear — its matrix is $\\begin{bmatrix}-1 & 0\\\\ 0 & 1\\end{bmatrix}$. You can build it in the transformer above.' },
      { text: 'Squash everything onto a single line.', why: 'Squashing is linear (it’s a matrix with determinant 0), just not reversible.' },
    ],
  },
  {
    id: 'matrices.det',
    kind: 'numeric',
    concepts: ['determinant'],
    make: (r) => {
      const a = r.int(4) + 1, b = r.int(5) - 2, c = r.int(5) - 2, d = r.int(4) + 1;
      const det = a * d - b * c;
      return {
        prompt: `A shape has area 2. After applying $\\begin{bmatrix} ${a} & ${b} \\\\ ${c} & ${d} \\end{bmatrix}$, what is the (signed) area of the transformed shape? (Use the determinant; a negative answer means the shape was flipped.)`,
        answer: 2 * det,
        explain: `The determinant is $${a}\\times${d} - ${p(b)}\\times${p(c)} = ${det}$, so every area is multiplied by ${det}: $2 \\times ${p(det)} = ${2 * det}$.`,
      };
    },
  },
  {
    id: 'matrices.det-zero',
    kind: 'choice',
    concepts: ['determinant'],
    prompt: 'A layer in a model multiplies its 2D input by a matrix with determinant 0. What follows?',
    options: [
      { text: 'Different inputs can produce identical outputs, so the input can’t always be recovered from the output.', correct: true, why: 'Determinant 0 means the plane gets squashed onto a line (or a point). Many different inputs land on the same output, so information is destroyed — there is no inverse matrix.' },
      { text: 'Every output is the zero vector.', why: 'Only the all-zeros matrix does that. $\\begin{bmatrix}1&0\\\\0&0\\end{bmatrix}$ has determinant 0 but keeps the x-coordinate perfectly intact.' },
      { text: 'The outputs are rotated but otherwise unchanged.', why: 'Pure rotations have determinant exactly 1 — they preserve area. Zero determinant means area collapses to nothing.' },
    ],
  },
];
