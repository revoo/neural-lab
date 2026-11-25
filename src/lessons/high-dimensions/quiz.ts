// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'hd.shell-numeric',
    kind: 'numeric',
    concepts: ['high-dimensional-geometry'],
    tolerance: 0.005,
    make: (r) => {
      const d = [10, 50, 200][r.int(3)];
      const eps = [0.01, 0.05][r.int(2)];
      const f = 1 - (1 - eps) ** d;
      return {
        prompt: `In ${d} dimensions, what fraction of a ball’s volume lies within ${eps * 100}% of its surface? (Answer as a decimal, e.g. 0.25.)`,
        answer: Number(f.toFixed(3)),
        explain: `The inner ball has radius (1 − ${eps}) times the full one, so volume (1 − ${eps})^${d} = ${((1 - eps) ** d).toFixed(3)} of it. The peel holds the rest: ${f.toFixed(3)}.`,
      };
    },
  },
  {
    id: 'hd.why-perpendicular',
    kind: 'choice',
    concepts: ['high-dimensional-geometry', 'cosine-similarity'],
    prompt: 'Why are two random 768-dimensional vectors almost always nearly perpendicular?',
    options: [
      {
        text: 'Their dot product is a sum of 768 independent ± terms that mostly cancel, growing only like √768, while their lengths grow like √768 each; so the cosine shrinks like 1/√768 ≈ 0.036.',
        correct: true,
        why: 'It’s the statistics lesson’s √n law: a sum of independent pieces has spread √n, while each length is about √d. The ratio goes to zero.',
      },
      {
        text: 'Because high-dimensional vectors are always normalised to length 1.',
        why: 'Normalising changes lengths, not angles. Unnormalised random vectors are just as perpendicular.',
      },
      {
        text: 'Because most of their entries are zero.',
        why: 'Every entry here is a non-zero random number. The near-perpendicularity comes from cancellation, not emptiness.',
      },
    ],
  },
  {
    id: 'hd.cosine-signal',
    kind: 'choice',
    concepts: ['cosine-similarity'],
    prompt: 'Two sentence embeddings (768 dimensions) have cosine similarity 0.3. A colleague says “0.3 is low, they’re basically unrelated.” What’s the better reading?',
    options: [
      {
        text: '0.3 is about 8 standard deviations above what random directions give (spread 0.036), so it’s a strong sign the two are related. What counts as “high” depends on the dimension.',
        correct: true,
        why: 'In 768 dimensions a random pair reaches |cos| ≥ 0.3 with probability around 2 × 10⁻¹⁷.',
      },
      {
        text: 'The colleague is right: anything below 0.5 means unrelated.',
        why: 'That intuition comes from 2D and 3D, where random directions often have cosines of 0.5 or more. In 768 dimensions they almost never do.',
      },
      {
        text: 'Cosine similarity is meaningless in high dimensions.',
        why: 'It stays very meaningful; it just has a different scale. What breaks down is the gap between nearest and farthest for *uniformly random* data.',
      },
    ],
  },
  {
    id: 'hd.knn',
    kind: 'choice',
    concepts: ['high-dimensional-geometry'],
    prompt: 'For 1,000 uniformly random points in a 1,000-dimensional cube, the farthest point from a query is only about 12% farther than the nearest. Why does nearest-neighbour search on real embeddings still work?',
    options: [
      {
        text: 'Real data isn’t uniformly random: it clusters and lies near much lower-dimensional structure, so genuinely similar items are much closer than the crowd.',
        correct: true,
        why: 'Distance concentration is a statement about structureless data. Learned embeddings are trained precisely to put related things close together.',
      },
      {
        text: 'Because vector databases use cosine similarity instead of distance, which doesn’t concentrate.',
        why: 'For random directions cosine concentrates too (around 0). The rescue comes from structure in the data, not from the choice of measure.',
      },
      {
        text: 'It doesn’t; nearest-neighbour search on embeddings is essentially random.',
        why: 'Retrieval with embeddings works very well in practice, because the data has structure.',
      },
    ],
  },
  {
    id: 'hd.superposition',
    kind: 'choice',
    concepts: ['high-dimensional-geometry'],
    prompt: 'In 6 dimensions you can have at most 6 exactly perpendicular directions — yet the lab packed 16 vectors with every |cos| ≤ 1/3. What does this suggest about how a network can store features?',
    options: [
      {
        text: 'If small overlaps are tolerable, a d-dimensional space can hold far more than d nearly independent feature directions. The gap grows enormously with d.',
        correct: true,
        why: 'This is the idea behind “superposition” in interpretability research: models appear to represent more features than they have neurons, accepting a little interference.',
      },
      {
        text: 'Nothing: 16 vectors in 6 dimensions must be linearly dependent, so they carry no more information than 6.',
        why: 'They are linearly dependent, but if features are rarely active at the same time, nearly perpendicular directions can still be told apart with little interference.',
      },
      {
        text: 'That ±1 vectors are special; real vectors can’t do this.',
        why: 'Real-valued vectors can do even better. ±1 vectors are just a version where the search can be exhaustive.',
      },
    ],
  },
  {
    id: 'hd.attention-scale',
    kind: 'choice',
    concepts: ['attention', 'high-dimensional-geometry'],
    prompt: 'Attention divides its scores q·k by √d. Which fact from this lesson explains why?',
    options: [
      {
        text: 'A dot product of two vectors with independent unit-variance entries is a sum of d products, so its spread grows like √d; dividing by √d keeps the scores at a similar scale whatever the head size.',
        correct: true,
        why: 'Without it, larger heads would produce larger scores and push the softmax into near one-hot saturation (the attention lesson measured this).',
      },
      {
        text: 'Because most of the volume lies near the surface.',
        why: 'True, but unrelated: the √d scaling is about the spread of a sum of d independent terms.',
      },
      {
        text: 'To make the vectors unit length.',
        why: 'Unit length would mean dividing each vector by its own norm. Dividing by √d rescales the score’s typical size without normalising anything.',
      },
    ],
  },
];
