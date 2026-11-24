import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'vectors.add',
    kind: 'numeric',
    concepts: ['vector-addition'],
    make: (r) => {
      const a = [r.int(9) - 4, r.int(9) - 4];
      const b = [r.int(9) - 4, r.int(9) - 4];
      return {
        prompt: `Let $\\mathbf{a} = [${a[0]}, ${a[1]}]$ and $\\mathbf{b} = [${b[0]}, ${b[1]}]$. What is the **second** number in $\\mathbf{a} + \\mathbf{b}$?`,
        answer: a[1] + b[1],
        explain: `Vectors add entry by entry: the second entry is $${a[1]} + (${b[1]}) = ${a[1] + b[1]}$. The full result is $[${a[0] + b[0]}, ${a[1] + b[1]}]$.`,
      };
    },
  },
  {
    id: 'vectors.length',
    kind: 'numeric',
    concepts: ['magnitude'],
    make: (r) => {
      const triples = [
        [3, 4, 5],
        [6, 8, 10],
        [5, 12, 13],
        [8, 15, 17],
      ];
      const [p, q, h] = triples[r.int(triples.length)];
      const sp = r.next() < 0.5 ? -1 : 1;
      return {
        prompt: `What is the length of the vector $[${sp * p}, ${q}]$?`,
        answer: h,
        explain: `Square each entry, add, take the square root: $\\sqrt{(${sp * p})^2 + ${q}^2} = \\sqrt{${p * p} + ${q * q}} = \\sqrt{${h * h}} = ${h}$. The sign of an entry never matters for length, because squaring removes it.`,
      };
    },
  },
  {
    id: 'vectors.scale-direction',
    kind: 'choice',
    concepts: ['scaling'],
    prompt: 'You multiply a vector by $-0.5$. What happens to its arrow?',
    options: [
      { text: 'It points the opposite way and becomes half as long.', correct: true, why: 'Every entry is multiplied by −0.5. The negative sign flips the direction; the 0.5 halves the length.' },
      { text: 'It points the same way and becomes half as long.', why: 'That would be multiplying by +0.5. The minus sign flips every entry’s sign, which reverses the arrow.' },
      { text: 'It rotates by 90 degrees.', why: 'Scaling never moves a vector off its line through the origin. To rotate, you need a matrix — that’s two lessons from now.' },
      { text: 'It becomes shorter by 0.5 units.', why: 'Scaling multiplies the length; it doesn’t subtract from it. A vector of length 10 becomes length 5, not 9.5.' },
    ],
  },
  {
    id: 'vectors.distance',
    kind: 'numeric',
    concepts: ['distance'],
    make: (r) => {
      const triples = [
        [3, 4, 5],
        [6, 8, 10],
        [5, 12, 13],
      ];
      const [p, q, h] = triples[r.int(triples.length)];
      const a = [r.int(7) - 3, r.int(7) - 3];
      const b = [a[0] + p, a[1] - q];
      return {
        prompt: `How far apart are the points $\\mathbf{a} = [${a[0]}, ${a[1]}]$ and $\\mathbf{b} = [${b[0]}, ${b[1]}]$?`,
        answer: h,
        explain: `Subtract to get the arrow between them, then take its length. $\\mathbf{b} - \\mathbf{a} = [${p}, ${-q}]$, and $\\sqrt{${p}^2 + (${-q})^2} = \\sqrt{${p * p + q * q}} = ${h}$.`,
      };
    },
  },
  {
    id: 'vectors.sigma',
    kind: 'choice',
    concepts: ['summation'],
    prompt: 'Which loop computes $\\sum_{i=1}^{n} v_i^2$ for a Python list `v` of length n?',
    options: [
      { text: '`s = 0; for x in v: s += x * x`', correct: true, why: 'Σ means “add up a term for every value of the index”. Here the term is $v_i^2$, so the loop squares each entry and accumulates.' },
      { text: '`s = 0; for x in v: s += x; s = s * s`', why: 'That computes $(\\sum v_i)^2$ — the square of the sum, not the sum of the squares. The square sits *inside* the Σ, so it applies to each term before adding.' },
      { text: '`s = 1; for x in v: s *= x * x`', why: 'That’s a product of squares. The capital-pi symbol Π means product; capital-sigma Σ means sum.' },
    ],
  },
];
