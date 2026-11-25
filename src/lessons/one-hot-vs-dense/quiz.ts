// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'onehot.orthogonality',
    kind: 'choice',
    concepts: ['one-hot-encoding', 'embedding'],
    prompt: 'Why does a one-hot classifier trained on “cat, dog, bird, horse” fail completely (50% random guessing) when asked to classify an unseen synonym like “kitten”?',
    options: [
      {
        text: 'Every distinct one-hot vector is strictly perpendicular (dot product 0) to every other, so the weight for “kitten” never received a single gradient update during training.',
        correct: true,
        why: 'One-hot vectors encode identity without geometry — sharing zero coordinates with near-synonyms.',
      },
      {
        text: 'Because one-hot vectors cannot fit the training set.',
        why: 'Our one-hot classifier hit 100.0% on the training set — it memorised each seen ID independently, then had zero transfer to unseen IDs.',
      },
      {
        text: 'Because cosine similarity is undefined for one-hot vectors.',
        why: 'Every one-hot vector has length 1, so cosine similarity is well-defined and equals exactly 0 for every distinct pair.',
      },
    ],
  },
  {
    id: 'onehot.scalar-id',
    kind: 'choice',
    concepts: ['one-hot-encoding'],
    prompt: 'Instead of a 50,000-dim one-hot vector, someone feeds raw integer token IDs (cat=0, car=1, dog=2, bus=3, …) as a single scalar input into a linear layer. What goes wrong?',
    options: [
      {
        text: 'It forces a fake 1D ordering onto arbitrary IDs — pretending “dog” (2) is twice “car” (1) and halfway between “car” and “bus” — so a linear classifier cannot even separate the training set.',
        correct: true,
        why: 'In our experiment, scalar IDs scored 50.0% on BOTH train and test!',
      },
      {
        text: 'It uses 50,000× more RAM than one-hot encoding.',
        why: 'Scalar IDs use almost zero memory; the problem is geometric nonsense, not memory.',
      },
      {
        text: 'It makes all words orthogonal to each other.',
        why: 'Scalar IDs live on a single 1D number line — the exact opposite of orthogonality.',
      },
    ],
  },
  {
    id: 'onehot.lookup-equiv',
    kind: 'choice',
    concepts: ['embedding-table'],
    prompt: 'Mathematically, passing a 1-hot vector e_i into a linear layer with V×d weight matrix E computes e_iᵀ E. How does PyTorch’s nn.Embedding actually execute this?',
    options: [
      {
        text: 'It directly indexes row E[i] in O(1) memory slice time, giving the exact same vector while skipping all (V − 1)×d multiplications by zero.',
        correct: true,
        why: 'For Llama-3 (V = 128,256, d = 4,096), direct lookup skips 99.9992% of a billion wasted FLOPs per token!',
      },
      {
        text: 'It hashes the string with SHA-256 on every forward pass.',
        why: 'nn.Embedding is literally a learnable V×d floating-point table indexed by integer token ID.',
      },
      {
        text: 'It performs a full dense matrix multiply on GPU because lookup is non-differentiable.',
        why: 'Table lookup IS differentiable: the backward pass simply routes the gradient straight back to row E[i] (sparse gradient update)!',
      },
    ],
  },
  {
    id: 'onehot.table-ram',
    kind: 'numeric',
    concepts: ['embedding-table'],
    tolerance: 0.05,
    make: (rng) => {
      const vocab = [32_000, 50_000, 64_000][rng.int(3)];
      const dim = [128, 256, 512][rng.int(3)];
      const bytes = [1, 2][rng.int(2)];
      const dtype = bytes === 1 ? 'INT8 (1 byte)' : 'FP16 (2 bytes)';
      const mib = Number(((vocab * dim * bytes) / (1024 * 1024)).toFixed(2));
      return {
        prompt: `How many MiB (divide by 1,048,576) does an embedding table with V = ${vocab.toLocaleString()} tokens and dimension d = ${dim} occupy in ${dtype}?`,
        answer: mib,
        explain: `${vocab.toLocaleString()} × ${dim} × ${bytes} bytes ÷ 1,048,576 = ${mib.toFixed(2)} MiB.`,
      };
    },
  },
];
