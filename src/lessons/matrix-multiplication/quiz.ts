import type { Question } from '../../components/quizTypes';

const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);

export const questions: Question[] = [
  {
    id: 'matmul.shape',
    kind: 'choice',
    concepts: ['shape', 'matrix-multiplication'],
    prompt: 'A batch of 64 inputs has shape (64, 300). It passes through a weight matrix of shape (300, 50). What is the output shape?',
    options: [
      { text: '(64, 50)', correct: true, why: 'The shared inner dimension (300) is consumed by the dot products; the outer dimensions survive. Each of the 64 rows becomes a 50-dimensional output.' },
      { text: '(300, 300)', why: 'The *inner* dimensions are the ones that disappear. What remains is (rows of the left) × (columns of the right).' },
      { text: '(50, 64)', why: 'Close, but flipped. The left operand’s rows come first: (64 × 300) @ (300 × 50) → (64 × 50).' },
      { text: 'It’s an error: 64 ≠ 50', why: 'Only the inner pair (300 and 300) has to match. The outer dimensions can be anything.' },
    ],
  },
  {
    id: 'matmul.entry',
    kind: 'numeric',
    concepts: ['matrix-multiplication'],
    make: (r) => {
      const A = [[r.int(7) - 3, r.int(7) - 3], [r.int(7) - 3, r.int(7) - 3]];
      const B = [[r.int(7) - 3, r.int(7) - 3], [r.int(7) - 3, r.int(7) - 3]];
      const ans = A[1][0] * B[0][1] + A[1][1] * B[1][1];
      const t = (m: number[][]) => `\\begin{bmatrix}${m.map((row) => row.join('&')).join('\\\\')}\\end{bmatrix}`;
      return {
        prompt: `Let $A = ${t(A)}$ and $B = ${t(B)}$. What is the entry in **row 2, column 2** of $AB$?`,
        answer: ans,
        explain: `Row 2 of A is $[${A[1].join(', ')}]$; column 2 of B is $[${B[0][1]}, ${B[1][1]}]$. Their dot product: $${p(A[1][0])}\\times${p(B[0][1])} + ${p(A[1][1])}\\times${p(B[1][1])} = ${ans}$.`,
      };
    },
  },
  {
    id: 'matmul.order',
    kind: 'choice',
    concepts: ['composition'],
    prompt: 'In code, `y = A @ (B @ x)`. Which transformation is applied to `x` first, and what single matrix does the same job?',
    options: [
      { text: 'B first; the single matrix is `A @ B`.', correct: true, why: 'Parentheses make B act first, then A. Because matrix multiplication is associative, $A(B\\mathbf{x}) = (AB)\\mathbf{x}$: the product $AB$ is one matrix that performs “B then A”. Read matrix products right to left.' },
      { text: 'A first; the single matrix is `A @ B`.', why: 'The innermost operation runs first: `B @ x` is computed before A ever sees anything.' },
      { text: 'B first; the single matrix is `B @ A`.', why: '`B @ A` would mean “A first, then B”. Products read right to left, so “B then A” is `A @ B`.' },
      { text: 'Order doesn’t matter for matrices.', why: 'Order usually matters: $AB \\ne BA$ in general. Try rotate-then-shear versus shear-then-rotate in the demo.' },
    ],
  },
  {
    id: 'matmul.broadcast',
    kind: 'choice',
    concepts: ['broadcasting'],
    prompt: 'Which of these additions **fails** under NumPy/PyTorch broadcasting rules?',
    options: [
      { text: '(32, 128) + (32,)', correct: true, why: 'Shapes are aligned from the **right**: 128 is compared with 32. They differ, and neither is 1, so it fails. To add one value per row you’d need shape (32, 1).' },
      { text: '(32, 128) + (128,)', why: 'This works: the 128s align on the right, and the vector is repeated across all 32 rows — exactly how a bias is added.' },
      { text: '(32, 1) + (1, 10)', why: 'This works and produces (32, 10): each size-1 dimension stretches to match the other.' },
      { text: '(8, 32, 128) + (128,)', why: 'This works: the vector is repeated across both leading dimensions.' },
    ],
  },
  {
    id: 'matmul.flops',
    kind: 'numeric',
    concepts: ['complexity'],
    make: (r) => {
      const n = [2, 4, 10][r.int(3)];
      const k = [3, 5, 100][r.int(3)];
      const m = [2, 6, 20][r.int(3)];
      return {
        prompt: `How many multiply-add operations does it take to multiply an (${n}×${k}) matrix by a (${k}×${m}) matrix with the straightforward triple loop?`,
        answer: n * k * m,
        explain: `The output has $${n}\\times${m} = ${n * m}$ entries, and each is a dot product of length ${k}. So $${n} \\times ${m} \\times ${k} = ${n * k * m}$ multiply-adds. For two square $n\\times n$ matrices that’s $n^3$ — which is why doubling a model’s width roughly quadruples the work per layer.`,
      };
    },
  },
];
