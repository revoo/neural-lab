import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'dot.compute',
    kind: 'numeric',
    concepts: ['dot-product'],
    make: (r) => {
      const a = [r.int(7) - 3, r.int(7) - 3, r.int(5)];
      const b = [r.int(7) - 3, r.int(5), r.int(7) - 3];
      const terms = a.map((v, i) => v * b[i]);
      const ans = terms.reduce((s, v) => s + v, 0);
      const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);
      return {
        prompt: `Compute $[${a.join(', ')}] \\cdot [${b.join(', ')}]$.`,
        answer: ans,
        explain: `Multiply matching entries, then add: $${a.map((v, i) => `${p(v)} \\times ${p(b[i])}`).join(' + ')} = ${terms.map(p).join(' + ')} = ${ans}$.`,
      };
    },
  },
  {
    id: 'dot.zero-meaning',
    kind: 'choice',
    concepts: ['orthogonal'],
    prompt: 'Two non-zero vectors have a dot product of exactly 0. What does that tell you?',
    options: [
      { text: 'They are perpendicular: neither has any component along the other’s direction.', correct: true, why: 'With $\\mathbf{a}\\cdot\\mathbf{b} = \\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$ and both lengths non-zero, the only way to get 0 is $\\cos\\theta = 0$, which means $\\theta = 90°$.' },
      { text: 'They are opposites of each other.', why: 'Opposite vectors have the most *negative* dot product possible for their lengths ($\\cos 180° = -1$), not zero.' },
      { text: 'They must have the same length.', why: 'Length doesn’t decide whether the dot product is zero. $[1, 0]$ and $[0, 100]$ have very different lengths and a dot product of 0.' },
      { text: 'At least one of their entries is zero.', why: '$[1, 1]$ and $[1, -1]$ contain no zeros, yet $1\\times1 + 1\\times(-1) = 0$. Zero dot product comes from the products *cancelling out*, not from any single entry.' },
    ],
  },
  {
    id: 'dot.scale-b',
    kind: 'choice',
    concepts: ['dot-product', 'scaling'],
    prompt: 'You keep $\\mathbf{a}$ fixed and replace $\\mathbf{b}$ with $3\\mathbf{b}$ (same direction, three times as long). What happens to $\\mathbf{a} \\cdot \\mathbf{b}$?',
    options: [
      { text: 'It triples.', correct: true, why: 'Every product $a_i b_i$ becomes $a_i (3 b_i) = 3 a_i b_i$, so the sum triples too. Geometrically: the angle is unchanged, but $\\|\\mathbf{b}\\|$ is three times bigger.' },
      { text: 'It stays the same, because the direction didn’t change.', why: 'That’s true of **cosine similarity**, which divides out the lengths. The raw dot product grows with length.' },
      { text: 'It increases by 3.', why: 'Scaling multiplies; every term in the sum is multiplied by 3, so the total is too.' },
      { text: 'It becomes 9 times larger.', why: 'Only one of the two vectors was scaled. If you tripled both, *then* it would grow 9×.' },
    ],
  },
  {
    id: 'dot.cosine',
    kind: 'numeric',
    concepts: ['cosine-similarity'],
    tolerance: 0.005,
    make: (r) => {
      const pairs = [
        [[3, 4], [4, 3]],
        [[3, 4], [0, 5]],
        [[6, 8], [-3, 4]],
        [[5, 12], [12, 5]],
      ];
      const [a, b] = pairs[r.int(pairs.length)];
      const d = a[0] * b[0] + a[1] * b[1];
      const la = Math.hypot(a[0], a[1]);
      const lb = Math.hypot(b[0], b[1]);
      const ans = d / (la * lb);
      return {
        prompt: `What is the cosine similarity between $[${a.join(', ')}]$ and $[${b.join(', ')}]$? (Give 3 decimal places.)`,
        answer: Math.round(ans * 1000) / 1000,
        explain: `Dot product: $${a[0]}\\times${b[0] < 0 ? `(${b[0]})` : b[0]} + ${a[1]}\\times${b[1]} = ${d}$. Lengths: $${la}$ and $${lb}$. Cosine similarity $= \\frac{${d}}{${la} \\times ${lb}} = ${ans.toFixed(3)}$.`,
      };
    },
  },
  {
    id: 'dot.search-trap',
    kind: 'choice',
    concepts: ['cosine-similarity'],
    prompt:
      'A search system represents a query as $\\mathbf{q} = [1, 1]$. Document A is $[10, 1]$, document B is $[2, 2]$. Which document ranks higher by raw dot product, and which by cosine similarity?',
    options: [
      { text: 'A wins on dot product; B wins on cosine similarity.', correct: true, why: '$\\mathbf{q}\\cdot A = 11$ and $\\mathbf{q}\\cdot B = 4$, so A wins on raw dot product purely because it’s long. But B points in *exactly* the same direction as the query (cosine 1.0), while A is mostly about the first feature (cosine ≈ 0.77). If length means “this document is just longer”, you want cosine similarity.' },
      { text: 'B wins on both.', why: 'Compute the raw dot products: $1\\times10 + 1\\times1 = 11$ for A versus $1\\times2 + 1\\times2 = 4$ for B. Length inflates A’s score.' },
      { text: 'A wins on both.', why: 'B is perfectly aligned with q, so its cosine similarity is exactly 1, the maximum possible.' },
      { text: 'They tie on both.', why: 'They differ on both: dot products are 11 vs 4, cosine similarities ≈ 0.77 vs 1.0.' },
    ],
  },
];
